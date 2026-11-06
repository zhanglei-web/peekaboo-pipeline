"""Small REST adapter with auditable, context-bound caching and bounded retries."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import mimetypes
import os
from pathlib import Path
import time
from urllib import error, request
from urllib.parse import urlparse

from .io import canonical, digest, load, require, run_lock, save, sha


class ModelServiceError(RuntimeError):
    pass


class ContractExhausted(RuntimeError):
    pass


class TransportError(RuntimeError):
    def __init__(self, message, retryable=False):
        super().__init__(message)
        self.retryable = retryable


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def post_json(url, body, api_key, timeout, idempotency_key):
    headers = {"Content-Type": "application/json", "Idempotency-Key": idempotency_key}
    if api_key:
        headers["Authorization"] = "Bearer "+api_key
    req = request.Request(url, canonical(body), headers=headers, method="POST")
    try:
        handlers = [NoRedirect()]
        if urlparse(url).hostname in {"127.0.0.1", "::1", "localhost"}:
            handlers.append(request.ProxyHandler({}))
        with request.build_opener(*handlers).open(req, timeout=timeout) as response:
            return response.read()
    except error.HTTPError as exc:
        # Do not echo a service error page, URL query, request headers or credentials.
        raise TransportError(f"model service HTTP {exc.code}",
                             exc.code in {408, 409, 429} or exc.code >= 500) from None
    except (error.URLError, TimeoutError, ConnectionError, OSError):
        raise TransportError("model service network/timeout failure", True) from None


def parse_json(text):
    return json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def response_text(raw, model):
    require(isinstance(raw, dict), "service response must be a JSON object")
    if not raw.get("model"):
        raise ModelServiceError("service response omitted model identity; cannot verify configured "+model["name"])
    if raw.get("model") != model["name"]:
        raise ModelServiceError("returned model does not exactly match configured "+model["name"])
    if model["transport"] == "responses":
        if raw.get("status") != "completed":
            raise ModelServiceError("model response is not completed; inspect saved raw response")
        parts = []
        for item in raw.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "refusal":
                        raise ModelServiceError("model refused; inspect saved raw response")
                    if content.get("type") == "output_text":
                        parts.append(content["text"])
        require(len(parts) == 1, "expected exactly one model JSON text output")
        return parts[0]
    choices = raw.get("choices", [])
    require(len(choices) == 1, "expected exactly one chat choice")
    message = choices[0]["message"]
    if choices[0].get("finish_reason") != "stop" or message.get("refusal"):
        raise ModelServiceError("chat output is incomplete or refused")
    require(isinstance(message.get("content"), str), "expected text chat output")
    return message["content"]


def make_body(model, prompt, schema, context, images, repair=None):
    # Also expose the contract in text: some compatible services fail to forward
    # native structured-output metadata. Local validation remains mandatory.
    content = [{"type": "input_text", "text": canonical({"evidence_data": context,
        "output_contract": schema, "output_instruction": "完整遵守此输出契约的所有必填字段和枚举，只输出一个JSON对象。"}).decode()}]
    for evidence in images:
        path = Path(evidence["path"])
        require(sha(path) == evidence["sha256"], "image bytes changed before request")
        mime = mimetypes.guess_type(path)[0]
        require(mime in {"image/jpeg", "image/png"}, "unsupported model image type")
        if "local_llama" in model:
            from .local_llama import image_url
            url = image_url(path, model["local_llama"]["image_max_edge"])
        else:
            url = f"data:{mime};base64,"+base64.b64encode(path.read_bytes()).decode()
        content.extend([{"type": "input_text", "text": "image_ref="+evidence["image_ref"]},
                        {"type": "input_image", "image_url": url, "detail": "high"}])
    if repair is not None:
        content.append({"type": "input_text", "text": canonical({"format_repair_data": repair,
            "instruction": "在同一证据下重新输出完整JSON；只纠正契约错误，不增加未来证据。"}).decode()})
    fmt = {"type": "json_schema", "name": "visual_observation", "strict": True, "schema": schema}
    if model["transport"] == "responses":
        body = {"model": model["name"], "instructions": prompt, "store": False,
            "input": [{"role": "user", "content": content}],
            "text": {"format": fmt}, "max_output_tokens": model["max_output_tokens"]}
        if model["reasoning_effort"]:
            body["reasoning"] = {"effort": model["reasoning_effort"]}
        return body
    chat_content = []
    for part in content:
        if part["type"] == "input_text":
            chat_content.append({"type": "text", "text": part["text"]})
        else:
            chat_content.append({"type": "image_url", "image_url": {
                "url": part["image_url"], "detail": "high"}})
    body = {"model": model["name"], "messages": [{"role": "system", "content": prompt},
        {"role": "user", "content": chat_content}],
        "response_format": {"type": "json_schema", "json_schema": {k: v for k, v in fmt.items() if k != "type"}},
        "max_completion_tokens": model["max_output_tokens"]}
    if model["reasoning_effort"]:
        body["reasoning_effort"] = model["reasoning_effort"]
    if "local_llama" in model:
        body["messages"][0]["content"] += "\n输出紧凑JSON，不换行、不缩进；描述简洁具体，保留全部必填字段和可见证据，避免重复同义描述。"
        body["max_tokens"] = body.pop("max_completion_tokens")
        body.update(temperature=0, seed=42, stream=False,
                    chat_template_kwargs={"enable_thinking": False}, reasoning_format="deepseek")
    return body


@dataclass
class CallResult:
    document: dict
    cache_key: str
    cache_hit: bool
    actual_model: str
    usage: dict
    new_transport_attempts: int


class Provider:
    def __init__(self, model, cache_root, implementation_hash, transport=post_json):
        self.model = model
        self.root = Path(cache_root)
        self.implementation_hash = implementation_hash
        self.transport = transport
        self.new_transport_attempts = 0

    def signature(self, prompt, schema, context, images):
        return {"model": self.model, "prompt": prompt, "schema": schema, "context": context,
            "images": images, "transform": ({"method": "exif_rgb_jpeg_quality90_v1", **self.model["local_llama"]}
                                              if "local_llama" in self.model else "original_bytes_detail_high"),
            "implementation_hash": self.implementation_hash}

    def observe(self, prompt, schema, context, images, validate):
        key = digest(self.signature(prompt, schema, context, images))
        # Category requests can be shared by concurrently processed episodes.
        # Wait for the identical in-flight request, then validate its cache.
        with run_lock(self.root/key, blocking=True):
            return self._observe(prompt, schema, context, images, validate)

    def _observe(self, prompt, schema, context, images, validate):
        for item in images:
            require(Path(item["path"]).is_file() and sha(item["path"]) == item["sha256"],
                    "model input image missing or changed")
        signature = self.signature(prompt, schema, context, images)
        key = digest(signature)
        folder = self.root/key
        folder.mkdir(parents=True, exist_ok=True)
        ledger_path = folder/"ledger.json"
        if ledger_path.exists():
            ledger = load(ledger_path)
            require(ledger["signature_hash"] == key, "cache signature mismatch")
            for name, expected in ledger["artifacts"].items():
                require(Path(name).name == name, "unsafe cache artifact path")
                require((folder/name).is_file() and sha(folder/name) == expected, "cache artifact changed or missing")
            if ledger["status"] == "validated":
                saved = load(folder/"validated.json")
                validate(saved)
                require(ledger["actual_model"] == self.model["name"], "cached model mismatch")
                return CallResult(saved, key, True, ledger["actual_model"], ledger["usage"], 0)
            if ledger["status"] == "exhausted":
                raise ContractExhausted("cached request exhausted its budget; inspect ledger, use a new reviewed config")
            if ledger["status"] == "service_failed":
                raise ModelServiceError("cached service request failed; inspect ledger, use a new reviewed config")
        else:
            ledger = {"schema": "astra_memory.request_ledger.v1", "signature_hash": key,
                "status": "running", "artifacts": {}, "attempts": [], "repair": None,
                "format_attempt": 0, "transport_attempt": 0}
            save(folder/"request_descriptor.json", signature)
            ledger["artifacts"]["request_descriptor.json"] = sha(folder/"request_descriptor.json")
            save(ledger_path, ledger)
        api_key = os.environ.get(self.model["api_key_env"])
        require(api_key or "local_llama" in self.model, "required model API-key environment variable is absent")
        count = 0
        while ledger["format_attempt"] <= self.model["contract_retries"]:
            n = ledger["format_attempt"]
            if ledger["transport_attempt"] > self.model["transport_retries"]:
                ledger["status"] = "service_failed"
                save(ledger_path, ledger)
                raise ModelServiceError("transport retry budget exhausted")
            body = make_body(self.model, prompt, schema, context, images, ledger["repair"])
            if "local_llama" in self.model:
                from .local_llama import preflight
                budget = preflight(self.model, body, images)
                budget_name = f"format_{n:02d}_local_preflight.json"
                save(folder/budget_name, budget)
                ledger["artifacts"][budget_name] = sha(folder/budget_name)
            suffix = "responses" if self.model["transport"] == "responses" else "chat/completions"
            attempt_id = f'format_{n:02d}_transport_{ledger["transport_attempt"]:02d}'
            ledger["transport_attempt"] += 1
            ledger["attempts"].append({"id": attempt_id, "status": "started"})
            save(ledger_path, ledger)  # Crash consumes this attempt; budget survives a restart.
            started = time.monotonic()
            count += 1
            self.new_transport_attempts += 1
            try:
                raw_bytes = self.transport(self.model["base_url"].rstrip("/")+"/"+suffix,
                    body, api_key, self.model["timeout_s"], key+f":{n}")
            except TransportError as exc:
                ledger["attempts"][-1].update(status="transport_error", error=str(exc), seconds=time.monotonic()-started)
                if not exc.retryable:
                    ledger["status"] = "service_failed"
                save(ledger_path, ledger)
                if not exc.retryable:
                    raise ModelServiceError(str(exc)) from None
                continue
            # Keep exact payload bytes unless the service has echoed the supplied credential.
            if api_key:
                raw_bytes = raw_bytes.replace(api_key.encode(), b"[REDACTED]")
            name = attempt_id+".response.json"
            from .io import atomic_bytes
            atomic_bytes(folder/name, raw_bytes)
            ledger["artifacts"][name] = sha(folder/name)
            ledger["attempts"][-1].update(status="received", seconds=time.monotonic()-started,
                response_path=name)
            save(ledger_path, ledger)
            try:
                raw = parse_json(raw_bytes)
                output = response_text(raw, self.model)
            except ModelServiceError as exc:
                ledger["status"] = "service_failed"
                ledger["attempts"][-1].update(status="service_rejected", error=str(exc))
                save(ledger_path, ledger)
                raise
            except (ValueError, KeyError, TypeError) as exc:
                ledger["status"] = "service_failed"
                ledger["attempts"][-1].update(status="invalid_service_envelope", error=type(exc).__name__)
                save(ledger_path, ledger)
                raise ModelServiceError("invalid service envelope; inspect saved raw response") from None
            try:
                document = parse_json(output)
                validate(document)
            except (ValueError, TypeError, KeyError) as exc:
                # jsonschema.ValidationError is handled below as a separate schema failure.
                failure = str(exc)
            except Exception as exc:
                from jsonschema import ValidationError
                if not isinstance(exc, ValidationError):
                    raise
                failure = str(exc)
            else:
                save(folder/"validated.json", document)
                ledger["artifacts"]["validated.json"] = sha(folder/"validated.json")
                ledger.update(status="validated", actual_model=raw["model"], usage=raw.get("usage", {}))
                ledger["attempts"][-1]["status"] = "validated"
                save(ledger_path, ledger)
                return CallResult(document, key, False, raw["model"], ledger["usage"], count)
            ledger["attempts"][-1].update(status="contract_rejected", error=failure[:4000])
            ledger["repair"] = {"rejected_output": output, "validation_error": failure[:4000]}
            ledger["format_attempt"] += 1
            ledger["transport_attempt"] = 0
            save(ledger_path, ledger)
        ledger["status"] = "exhausted"
        save(ledger_path, ledger)
        raise ContractExhausted("model output failed contract after bounded format corrections")
