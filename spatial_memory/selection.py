"""Qwen3-VL-8B-Instruct chooses all three event frames and proposes unit associations."""
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import time

from PIL import Image
import requests

from .io import sha256, write_json


MODEL = "Qwen3-VL-8B-Instruct"


def validate_event(event):
    if not event.get("event_id") or not event.get("episode_id"):
        raise ValueError("Event and episode IDs are required")
    candidates = event.get("candidates", [])
    if len(candidates) < 3:
        raise ValueError("At least three displayed candidate frames are required")
    ids = [int(c["source_frame_id"]) for c in candidates]
    times = [float(c["timestamp_s"]) for c in candidates]
    if len(set(ids)) != len(ids) or ids != sorted(ids) or times != sorted(times):
        raise ValueError("Candidate source frames and timestamps must be unique and ordered")
    for c in candidates:
        for key in ("image", "action", "object_state", "tags"):
            if key not in c:
                raise ValueError(f"Candidate {c['source_frame_id']} lacks {key}")
        if not Path(c["image"]).is_file():
            raise FileNotFoundError(c["image"])
    required = event.get("required_reference_frame_ids", [])
    if not set(required).issubset(set(ids)):
        raise ValueError("An event/contact boundary reference was omitted from displayed candidates")


def validate_selection(event, catalog, result):
    if result.get("status") not in ("ok", "uncertain"):
        raise ValueError("Invalid selection status")
    available = {int(c["source_frame_id"]): c for c in event["candidates"]}
    if not all(role in result for role in ("pre", "pnr", "post")):
        raise ValueError("Selection must contain all three role fields, using null if uncertain")
    roles = [result.get(role) for role in ("pre", "pnr", "post")]
    if result["status"] == "ok":
        if any(not isinstance(x, int) or isinstance(x, bool) or x not in available for x in roles):
            raise ValueError("All selected frames must be displayed candidates")
        if not roles[0] < roles[1] < roles[2]:
            raise ValueError("Selected Pre, PNR, Post must be strictly time ordered")
    elif any(x is not None and x not in available for x in roles):
        raise ValueError("Uncertain selection references an unseen frame")
    allowed_units = {r["candidate_id"] for r in catalog["regions"]}
    associations = result.get("associations", [])
    ids = [a["candidate_id"] for a in associations]
    if len(ids) != len(set(ids)) or not set(ids).issubset(allowed_units):
        raise ValueError("Association must reference unique stage-one candidates")
    if any(not a.get("reason") or a.get("confidence") not in ("high", "medium", "low") for a in associations):
        raise ValueError("Associations require reason and confidence")
    if not isinstance(result.get("reason"), str):
        raise ValueError("Selection reason required")
    return result


def _jpeg_data(path, image_max_edge):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((image_max_edge, image_max_edge))
        output = BytesIO()
        im.save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def select(event, catalog, out_dir, base_url, *, model=MODEL, api_key=None, request=None,
           image_max_edge=768):
    """Call an OpenAI-compatible local VLM; `request` permits offline integration tests."""
    validate_event(event)
    if isinstance(image_max_edge, bool) or not isinstance(image_max_edge, int) or image_max_edge <= 0:
        raise ValueError("image_max_edge must be a positive integer")
    if event["episode_id"] != catalog["episode_id"]:
        raise ValueError("Event and catalog episode IDs differ")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    regions = [{"candidate_id": r["candidate_id"], "name": r["name"],
                "region_type": r["region_type"], "boundary": r["boundary_description"]}
               for r in catalog["regions"]]
    metadata = [{k: v for k, v in c.items() if k != "image"} for c in event["candidates"]]
    prompt = (
        "你是视频事件证据标注器。按时间和动作解释每张实际显示的候选图像。"
        "同时选择三帧：Pre 显示操作前状态；PNR 显示动作展开、暴露或关键状态变化最充分且清晰的时刻；"
        "Post 显示操作结果或稳定状态。事件和接触簇起止帧是重要参考，不预设任何角色。"
        "只能引用显示的 source_frame_id，且 Pre < PNR < Post。缺少可靠前态、过程或结果时 status=uncertain，"
        "相应角色写 null，不编造。若疑似多个交互，在 reason 说明。"
        "根据实际检查的空间而非仅被触碰的物体，列出关联的候选区域；不能确定时 associations 留空。"
        "只返回 JSON，字段 status(ok/uncertain), pre, pnr, post, reason, associations；"
        "associations 每项含 candidate_id, reason, confidence(high/medium/low)。\n"
        + json.dumps({"event_id": event["event_id"], "candidates": metadata,
                      "candidate_regions": regions}, ensure_ascii=False)
    )
    descriptor = {"model": model, "image_max_edge": image_max_edge, "event_id": event["event_id"],
                  "episode_id": event["episode_id"], "catalog_version": catalog["catalog_version"],
                  "prompt": prompt, "images": [{"source_frame_id": c["source_frame_id"],
                                                  "sha256": sha256(c["image"])} for c in event["candidates"]]}
    input_path = out_dir / "selection_input.json"
    if input_path.exists() and json.loads(input_path.read_text()) != descriptor:
        raise ValueError("Run directory already contains a different model request")
    write_json(input_path, descriptor)
    content = [{"type": "text", "text": prompt}]
    for c in event["candidates"]:
        content.append({"type": "text", "text": f"source_frame_id={c['source_frame_id']}"})
        content.append({"type": "image_url", "image_url": {"url": _jpeg_data(c["image"], image_max_edge)}})
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "temperature": 0, "seed": 42, "max_tokens": 1800,
               "chat_template_kwargs": {"enable_thinking": False},
               "response_format": {"type": "json_object"}}
    key = api_key if api_key is not None else os.environ.get("QWEN_API_KEY", "EMPTY")
    start = time.perf_counter()
    if request is None:
        session = requests.Session()
        session.trust_env = False
        response = session.post(base_url.rstrip("/") + "/chat/completions", json=payload,
                                headers={"Authorization": f"Bearer {key}"}, timeout=(10, 900),
                                allow_redirects=False)
        if response.status_code != 200:
            write_json(out_dir / "selection_http_failure.json", {
                "http_status": response.status_code, "response": response.text.replace(key, "[REDACTED]"),
                "seconds": time.perf_counter() - start})
        response.raise_for_status()
        raw = response.json()
    else:
        raw = request(payload)
    write_json(out_dir / "selection_raw_response.json", raw)
    if raw.get("model") != model or raw["choices"][0].get("finish_reason") != "stop":
        raise ValueError("Unexpected model or incomplete response")
    result = json.loads(raw["choices"][0]["message"]["content"])
    validate_selection(event, catalog, result)
    result.update({"event_id": event["event_id"], "episode_id": event["episode_id"],
                   "model": model, "catalog_version": catalog["catalog_version"],
                   "selected_frames": {role: next((c for c in event["candidates"]
                                                    if c["source_frame_id"] == result[role]), None)
                                       for role in ("pre", "pnr", "post")},
                   "model_seconds": time.perf_counter() - start,
                   "usage": raw.get("usage")})
    write_json(out_dir / "selection.json", result)
    return result
