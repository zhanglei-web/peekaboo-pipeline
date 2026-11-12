import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import psutil
import requests
from jsonschema import ValidationError
from .config import ROOT, load_config
from .media import prepare_image
from .validation import SCHEMA, parse_response

def save_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def new_output():
    path = ROOT / "outputs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8])
    path.mkdir(parents=True)
    return path

class ResourceSampler:
    def __init__(self, directory, interval):
        self.directory, self.interval = directory, interval
        self.done = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self):
        row = {"time_utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": time.monotonic()}
        mem = psutil.virtual_memory()
        row.update(host_used_bytes=mem.used, host_available_bytes=mem.available, client_rss_bytes=psutil.Process().memory_info().rss)
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=True)
            row["gpus"] = [dict(zip(["index", "used_mib", "free_mib", "utilization_percent"], map(int, line.split(",")))) for line in result.stdout.strip().splitlines()]
            pid_path = ROOT / "logs/server.pid.json"
            if pid_path.exists():
                row["server_rss_bytes"] = psutil.Process(json.loads(pid_path.read_text())["pid"]).memory_info().rss
        except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as exc:
            row["sampling_error"] = str(exc)
        self.samples.append(row)

    def run(self):
        while not self.done.wait(self.interval):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=8)
        self.sample()
        save_json(self.directory / "resources.json", self.samples)
        gpu_values = [g["used_mib"] for r in self.samples for g in r.get("gpus", [])]
        return {"method": "nvidia-smi periodic sampling, whole device incl. desktop; psutil host memory/RSS",
                "interval_s": self.interval, "sample_count": len(self.samples),
                "sampled_peak_gpu_used_mib": max(gpu_values) if gpu_values else None,
                "baseline": self.samples[0], "peak_host_used_bytes": max(s["host_used_bytes"] for s in self.samples)}

class Client:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.base = self.config["client"]["base_url"].removesuffix("/v1")
        self.session = requests.Session()
        # Local calls must not accidentally go through a user proxy.
        self.session.trust_env = False
        key = os.environ.get("LOCAL_VLM_API_KEY")
        if key:
            self.session.headers["Authorization"] = f"Bearer {key}"

    def post(self, endpoint, payload, raw_path=None):
        try:
            response = self.session.post(self.base + endpoint, json=payload, timeout=self.config["client"]["timeout"])
        except requests.RequestException as exc:
            if raw_path:
                save_json(raw_path, {"transport_error": type(exc).__name__, "detail": str(exc)})
            raise
        try:
            data = response.json()
        except ValueError:
            data = {"http_status": response.status_code, "raw_body": response.text}
        if raw_path:
            save_json(raw_path, data)
        if not response.ok:
            raise RuntimeError(f"HTTP {response.status_code}: {json.dumps(data, ensure_ascii=False)[:1500]}")
        return data

    def budget(self, payload, frame_count):
        # Count the actual template text with the backend tokenizer. Media token usage
        # is conservatively bounded by the explicit per-image server cap plus margin.
        text_payload = copy.deepcopy(payload)
        for message in text_payload["messages"]:
            if isinstance(message["content"], list):
                message["content"] = [{"type": "text", "text": "[image placeholder]"} if x["type"] == "image_url" else x for x in message["content"]]
        template = self.post("/apply-template", text_payload)["prompt"]
        tokens = self.post("/tokenize", {"content": template, "add_special": True})["tokens"]
        image_bound = frame_count * (self.config["server"]["image_max_tokens"] + 32)
        total = len(tokens) + image_bound + payload["max_tokens"] + 256
        limit = self.config["server"]["ctx_size"] // self.config["server"]["parallel"]
        if total > limit:
            raise ValueError(f"Conservative context budget exceeded: {total} > {limit}; explicitly reduce frames/resolution/output or increase context. No input was truncated.")
        return {"text_tokens": len(tokens), "image_token_upper_bound": image_bound,
                "output_tokens": payload["max_tokens"], "margin": 256, "total_bound": total, "context_limit": limit}

    def run(self, manifest, task="describe", prompt=None, output_dir=None):
        out = output_dir or new_output()
        cfg = self.config
        started = time.monotonic()
        metadata = {"status": "running", "created_utc": datetime.now(timezone.utc).isoformat(), "task": task,
                    "config": cfg, "inputs": manifest, "user_prompt": prompt,
                    "degradations": cfg["client"].get("degradations", []), "attempts": []}
        for key, file in [("model", ROOT / "configs/model.lock.json"), ("weight_verification", ROOT / "models/qwen3.8-27b/manifest.json"),
                          ("server_process", ROOT / "logs/server.pid.json")]:
            if file.exists():
                metadata[key] = json.loads(file.read_text())
        metadata["llama_cpp_commit"] = (ROOT / "configs/backend.commit").read_text().strip()
        sampler = ResourceSampler(out, cfg["client"]["resource_interval_s"])
        sampler.start()
        try:
            if not 1 <= len(manifest["frames"]) <= cfg["client"]["max_frames"]:
                raise ValueError("Frame count exceeds configured limit; no frames dropped")
            active = metadata.get("server_process", {}).get("config", {}).get("server")
            if active and active != cfg["server"]:
                raise ValueError("Client/server configuration differs; restart the server with this config")
            system = (ROOT / "prompts" / ("event_annotation_zh.txt" if task == "event_annotation" else "describe_zh.txt")).read_text()
            if task == "event_annotation":
                system += "\nJSON Schema:\n" + json.dumps(SCHEMA, ensure_ascii=False, separators=(",", ":"))
            metadata["system_prompt"] = system
            content = [{"type": "text", "text": f"clip_id: {manifest['clip_id']}\n" + (prompt or "请完成指定任务。")}]
            processed = []
            for index, frame in enumerate(manifest["frames"]):
                dest = out / f"input-{index:02d}.jpg"
                url, info = prepare_image(frame["path"], dest, cfg["client"]["image_max_edge"])
                processed.append({**frame, **info})
                content.extend([{"type": "text", "text": f"frame_id: {frame['frame_id']}; timestamp_s: {frame['timestamp_s']}"},
                                {"type": "image_url", "image_url": {"url": url}}])
            metadata["processed_frames"] = processed
            save_json(out / "input_manifest.json", manifest)
            payload = {"model": cfg["server"]["alias"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                       "max_tokens": cfg["client"]["max_tokens"], "temperature": cfg["client"]["temperature"], "seed": cfg["client"]["seed"],
                       "stream": False, "chat_template_kwargs": {"enable_thinking": False}, "reasoning_format": "deepseek"}
            if task == "event_annotation":
                payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "event_annotation", "strict": True, "schema": SCHEMA}}
            repairs = min(1, max(0, cfg["client"]["repair_attempts"])) if task == "event_annotation" else 0
            for attempt in range(repairs + 1):
                attempt_info = {"attempt": attempt + 1, "repair": attempt > 0}
                metadata["attempts"].append(attempt_info)
                attempt_info["budget"] = self.budget(payload, len(processed))
                request_start = time.monotonic()
                raw_path = out / f"response-attempt-{attempt + 1}.json"
                try:
                    response = self.post("/v1/chat/completions", payload, raw_path)
                finally:
                    if raw_path.exists():
                        (out / "response.json").write_bytes(raw_path.read_bytes())
                    attempt_info["elapsed_s"] = time.monotonic() - request_start
                choice = response.get("choices", [{}])[0]
                attempt_info.update(usage=response.get("usage"), finish_reason=choice.get("finish_reason"), timings=response.get("timings"),
                                    reasoning_content=choice.get("message", {}).get("reasoning_content"))
                try:
                    result = parse_response(response, manifest, task)
                except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
                    attempt_info["validation_error"] = str(exc)
                    if attempt >= repairs:
                        raise
                else:
                    save_json(out / "result.json", result)
                    metadata["status"] = "success"
                    metadata.update(usage=response.get("usage"), finish_reason=choice.get("finish_reason"))
                    return out, result
                previous = choice.get("message", {}).get("content") or ""
                payload["messages"].extend([{"role": "assistant", "content": previous},
                                            {"role": "user", "content": "仅修复刚才的 JSON，并依据原始图片与 schema 保持简短。校验错误：" + attempt_info["validation_error"]}])
        except Exception as exc:
            metadata.update(status="error", error_type=type(exc).__name__, error=str(exc))
            save_json(out / "error.json", {"status": "error", "type": type(exc).__name__, "detail": str(exc)})
            raise
        finally:
            metadata["elapsed_s"] = time.monotonic() - started
            metadata["resources"] = sampler.stop()
            save_json(out / "metadata.json", metadata)
