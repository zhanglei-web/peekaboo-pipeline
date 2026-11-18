"""Local-service JSON requests with immutable request identity and cached receipts."""
import base64
import hashlib
import json
from pathlib import Path
import time

import jsonschema
import requests


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


class JsonModel:
    def __init__(self, model_root, base_url=None):
        root = Path(model_root)
        config = json.loads((root / "config.json").read_text())
        self.model = config["model_alias"]
        host = "127.0.0.1" if config["host"] == "0.0.0.0" else config["host"]
        self.base_url = (base_url or f"http://{host}:{config['port']}/v1").rstrip("/")
        keys = [line.strip() for line in (root / config["api_key_file"]).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
        if not keys:
            raise ValueError("Configured model credential file is empty")
        self.secret = keys[0]
        self.session = requests.Session()
        self.session.trust_env = False
        self.cache_observations = []

    def infer(self, folder, prompt, images, schema, max_tokens):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        descriptors = [{"path": str(Path(path).resolve()),
                        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                       for path in images]
        descriptor = {"model": self.model, "base_url": self.base_url, "prompt": prompt,
                      "images": descriptors, "schema": schema, "max_tokens": max_tokens,
                      "temperature": 0, "seed": 42, "thinking": False}
        key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        inputs = folder / "inputs.json"
        if inputs.exists():
            if json.loads(inputs.read_text()) != descriptor:
                raise ValueError("Existing model request differs; use a new run directory")
        else:
            save_json(inputs, descriptor)
        cached = folder / "result.json"
        self.cache_observations.append({"folder": str(folder.resolve()), "hit": cached.exists()})
        if cached.exists():
            receipt = json.loads((folder / "receipt.json").read_text())
            if receipt["input_sha256"] != key:
                raise ValueError("Cached response request identity mismatch")
            if receipt["result_sha256"] != hashlib.sha256(cached.read_bytes()).hexdigest():
                raise ValueError("Cached response content changed")
            value = json.loads(cached.read_text())
            jsonschema.validate(value, schema)
            return value, receipt
        content = [{"type": "text", "text": prompt}]
        for path in images:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()}})
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": "按给定规则分析图像，只输出符合模式的JSON；未知信息明确保留未知。"},
                         {"role": "user", "content": content}],
            "temperature": 0, "seed": 42, "max_tokens": max_tokens,
            "presence_penalty": 0, "repeat_penalty": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "scene_annotation", "strict": True, "schema": schema}},
        }
        started = time.perf_counter()
        attempt = folder / f"http_{time.time_ns()}"
        attempt.mkdir()
        save_json(attempt / "request.json", {"input_sha256": key, "status": "submitted"})
        try:
            response = self.session.post(
                self.base_url + "/chat/completions", json=payload,
                headers={"Authorization": f"Bearer {self.secret}", "Idempotency-Key": key},
                timeout=(10, 900), allow_redirects=False)
        except requests.RequestException as exc:
            save_json(attempt / "failure.json", {"kind": type(exc).__name__,
                      "server_completion_unknown": True, "seconds": time.perf_counter() - started})
            raise RuntimeError("Model transport failed; inspect attempt before retry") from None
        text = response.text.replace(self.secret, "[REDACTED]")
        (attempt / "response.json").write_text(text)
        elapsed = time.perf_counter() - started
        save_json(attempt / "timing.json", {"seconds": elapsed, "http_status": response.status_code})
        if response.status_code != 200:
            raise RuntimeError(f"Model HTTP {response.status_code}; inspect {attempt}")
        raw = json.loads(text)
        choice = raw["choices"][0]
        if raw["model"] != self.model or choice["finish_reason"] != "stop":
            raise ValueError("Model response is incomplete or came from an unexpected model")
        value = json.loads(choice["message"]["content"])
        try:
            jsonschema.validate(value, schema)
        except jsonschema.ValidationError as exc:
            save_json(attempt / "validation_failure.json",
                      {"error": exc.message, "path": list(exc.absolute_path)})
            raise
        save_json(cached, value)
        receipt = {"input_sha256": key,
                   "result_sha256": hashlib.sha256(cached.read_bytes()).hexdigest(),
                   "model": self.model, "seconds": elapsed, "usage": raw.get("usage"),
                   "attempt": str(attempt.resolve())}
        save_json(folder / "receipt.json", receipt)
        return value, receipt
