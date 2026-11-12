"""Local llama.cpp transport checks and explicit image/context budgets."""
from __future__ import annotations

import base64
from copy import deepcopy
import io
import json
from pathlib import Path
from urllib import request
from urllib.parse import urlparse

from PIL import Image, ImageOps

from .io import load, require, sha


def image_url(path, max_edge):
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def query(base, endpoint, body=None):
    # Local requests must not use HTTP_PROXY/HTTPS_PROXY or follow redirects.
    from .provider import NoRedirect
    payload = None if body is None else json.dumps(body).encode()
    req = request.Request(base + endpoint, payload, headers={"Content-Type": "application/json"})
    with request.build_opener(request.ProxyHandler({}), NoRedirect()).open(req, timeout=30) as response:
        return json.load(response)


def preflight(model, body, images):
    options = model["local_llama"]
    base = model["base_url"].removesuffix("/v1")
    require(urlparse(base).hostname in {"127.0.0.1", "::1"}, "local requests must remain on loopback")
    require(len(images) <= options["max_images"], "image count exceeds explicit limit; no images dropped")
    record_path = Path(options["server_record"])
    record = load(record_path)
    server = record["config"]["server"]
    # The PID record binds the API alias to the loaded weights, not merely a name.
    import psutil
    process = psutil.Process(record["pid"])
    require(abs(process.create_time()-record["create_time"]) < .01 and
            Path(process.exe()).resolve() == Path(record["binary"]).resolve() and
            process.cmdline() == record["command"], "local server process identity changed")
    require(server["alias"] == model["name"] and server["parallel"] == options.get("parallel", 1) and
            server["ctx_size"] == options["context_size"]*server["parallel"] and
            server["image_max_tokens"] == options["image_max_tokens"] and
            server["reasoning"] == "off" and not server["reasoning_preserve"], "local server settings differ")
    require(all(server.get(key, "f16") == options.get(key, "f16") for key in ["cache_type_k", "cache_type_v"]),
            "local KV cache precision differs from run configuration")
    lock = record_path.parent.parent/"configs/model.lock.json"
    require(sha(lock) == options["model_lock_sha256"], "local weight lock changed")
    locked_files = {f["name"] for f in load(lock)["files"]}
    require({Path(server["model"]).name, Path(server["mmproj"]).name} == locked_files,
            "server model/projector do not match pinned weight lock")
    props = query(base, "/props")
    actual_context = props["default_generation_settings"]["n_ctx"]
    require(actual_context == options["context_size"], "running API context differs from configuration")
    require(props["total_slots"] == server["parallel"], "running API slot count differs from configuration")
    # Use the actual template and tokenizer; preserve full evidence and schema.
    text_body = deepcopy(body)
    for message in text_body["messages"]:
        if isinstance(message["content"], list):
            message["content"] = [({"type": "text", "text": "[image placeholder]"}
                if part["type"] == "image_url" else part) for part in message["content"]]
    template = query(base, "/apply-template", text_body)["prompt"]
    text_count = len(query(base, "/tokenize", {"content": template, "add_special": True})["tokens"])
    image_bound = len(images)*(options["image_max_tokens"]+32)
    total = text_count+image_bound+body["max_tokens"]+256
    require(total <= actual_context,
            f"local context budget exceeded: {total} > {actual_context}; no evidence was truncated")
    return {"text_tokens": text_count, "image_token_upper_bound": image_bound,
        "output_tokens": body["max_tokens"], "margin": 256, "total_bound": total,
        "context_limit": actual_context, "server_pid": record["pid"],
        "server_record_sha256": sha(record_path), "model_lock_sha256": sha(lock),
        "image_transform": "exif_rgb_jpeg_quality90_v1", "image_max_edge": options["image_max_edge"],
        "image_count": len(images), "source_images": images}
