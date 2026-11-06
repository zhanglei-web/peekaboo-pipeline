import copy
import json
from pathlib import Path
import pytest
from PIL import Image
import requests
from local_vlm.client import Client
from local_vlm.config import load_config
from local_vlm.media import load_manifest, prepare_image, extract_video
from local_vlm.validation import parse_response, validate_annotation

@pytest.fixture
def manifest(tmp_path):
    Image.new("RGB", (32, 32), "red").save(tmp_path / "a.png")
    return {"clip_id": "clip", "frames": [{"frame_id": "f0", "path": "a.png", "timestamp_s": 0}, {"frame_id": "f1", "path": "a.png", "timestamp_s": 1}]}

def write_manifest(tmp_path, data):
    path = tmp_path / "frames.json"
    path.write_text(json.dumps(data))
    return path

@pytest.mark.parametrize("mutation", ["order", "duplicate", "missing", "nan", "too_many"])
def test_manifest_rejects_bad_inputs(tmp_path, manifest, mutation):
    if mutation == "order": manifest["frames"][1]["timestamp_s"] = 0
    if mutation == "duplicate": manifest["frames"][1]["frame_id"] = "f0"
    if mutation == "missing": manifest["frames"][1]["path"] = "missing.jpg"
    if mutation == "nan": manifest["frames"][1]["timestamp_s"] = float("nan")
    with pytest.raises((ValueError, FileNotFoundError)):
        load_manifest(write_manifest(tmp_path, manifest), 1 if mutation == "too_many" else 4)

def test_manifest_resolves_relative_to_file(tmp_path, manifest):
    loaded = load_manifest(write_manifest(tmp_path, manifest), 4)
    assert loaded["frames"][0]["path"] == str(tmp_path / "a.png")

def test_bad_image(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_text("not an image")
    with pytest.raises(OSError):
        prepare_image(path, tmp_path / "out.jpg", 768)

def test_image_orientation_alpha_and_no_upscale(tmp_path):
    path = tmp_path / "in.png"
    Image.new("RGBA", (20, 30), (255, 0, 0, 0)).save(path)
    _, info = prepare_image(path, tmp_path / "out.jpg", 768)
    assert info["sent_size"] == [20, 30]
    assert Image.open(tmp_path / "out.jpg").getpixel((0, 0)) == (255, 255, 255)
    image = Image.new("RGB", (40, 20))
    exif = Image.Exif()
    exif[274] = 6
    image.save(tmp_path / "exif.jpg", exif=exif)
    _, info = prepare_image(tmp_path / "exif.jpg", tmp_path / "oriented.jpg", 30)
    assert info["oriented_size"] == [20, 40]
    assert info["sent_size"] == [15, 30]

@pytest.fixture
def annotation():
    state = {"frame_id": "f0", "visible_objects": ["o1"], "position_description": None, "relations": []}
    return {"clip_id": "clip", "objects": [{"id": "o1", "name": "红方块", "appearance": "红色", "evidence_frames": ["f0"]}],
            "before_state": state, "after_state": {**state, "frame_id": "f1"}, "events": [], "uncertainties": []}

@pytest.mark.parametrize("mutation", ["evidence", "object", "duplicate", "endpoint", "clip"])
def test_semantic_reference_failures(annotation, manifest, mutation):
    if mutation == "evidence": annotation["objects"][0]["evidence_frames"] = ["f99"]
    if mutation == "object": annotation["before_state"]["visible_objects"] = ["o99"]
    if mutation == "duplicate": annotation["objects"].append(annotation["objects"][0])
    if mutation == "endpoint": annotation["after_state"]["frame_id"] = "f0"
    if mutation == "clip": annotation["clip_id"] = "wrong"
    with pytest.raises(ValueError): validate_annotation(annotation, manifest)

@pytest.mark.parametrize("content,finish", [("not JSON", "stop"), ("{}", "length"), ("<think>x</think>{}", "stop")])
def test_invalid_or_truncated_response(manifest, content, finish):
    with pytest.raises(ValueError):
        parse_response({"choices": [{"message": {"content": content}, "finish_reason": finish}]}, manifest, "event_annotation")

def test_transport_timeout_and_http_error_preserved(tmp_path, monkeypatch):
    client = Client()
    def timeout(*args, **kwargs): raise requests.Timeout("simulated timeout")
    monkeypatch.setattr(client.session, "post", timeout)
    with pytest.raises(requests.Timeout): client.post("/v1/chat/completions", {}, tmp_path / "response.json")
    assert json.loads((tmp_path / "response.json").read_text())["transport_error"] == "Timeout"
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"simulated server failure"}'
    monkeypatch.setattr(client.session, "post", lambda *a, **k: response)
    with pytest.raises(RuntimeError): client.post("/v1/chat/completions", {}, tmp_path / "response.json")
    assert json.loads((tmp_path / "response.json").read_text())["error"] == "simulated server failure"

def test_video_duplicate_indices_and_timestamps(tmp_path):
    import cv2
    import numpy as np
    video = tmp_path / "two.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    for i in range(2): writer.write(np.full((32, 32, 3), i * 100, dtype=np.uint8))
    writer.release()
    path = extract_video(video, tmp_path, 0, 0.2, 4)
    result = load_manifest(path, 4)
    assert result["actual_num_frames"] == 2
    assert result["duplicate_indices_removed"] == 2
    assert [f["frame_index"] for f in result["frames"]] == [0, 1]
    assert [f["timestamp_s"] for f in result["frames"]] == pytest.approx([0, 0.1])

def test_budget_rejects_without_sending_completion(monkeypatch):
    cfg = load_config()
    cfg["server"]["ctx_size"] = 100
    client = Client(cfg)
    monkeypatch.setattr(client, "post", lambda path, payload: {"prompt": "text"} if path == "/apply-template" else {"tokens": [1] * 80})
    with pytest.raises(ValueError, match="budget exceeded"):
        client.budget({"messages": [{"role": "user", "content": "text"}], "max_tokens": 30}, 1)

def test_single_repair_then_nonzero_error_state(tmp_path, manifest, monkeypatch):
    # Unit requests must not inherit the live server's deployment profile/PID.
    import local_vlm.client as client_module
    original_root = client_module.ROOT
    for relative in ["prompts/event_annotation_zh.txt", "configs/backend.commit"]:
        path = tmp_path/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original_root/relative).read_bytes())
    monkeypatch.setattr(client_module, "ROOT", tmp_path)
    manifest["frames"][0]["path"] = str(tmp_path / "a.png")
    manifest["frames"][1]["path"] = str(tmp_path / "a.png")
    out = tmp_path / "out"
    out.mkdir()
    client = Client()
    monkeypatch.setattr(client, "budget", lambda *args: {})
    calls = []
    def invalid(endpoint, payload, raw_path):
        calls.append(endpoint)
        response = {"choices": [{"message": {"content": "invalid"}, "finish_reason": "stop"}]}
        raw_path.write_text(json.dumps(response))
        return response
    monkeypatch.setattr(client, "post", invalid)
    with pytest.raises(ValueError): client.run(manifest, "event_annotation", output_dir=out)
    assert len(calls) == 2
    assert not (out / "result.json").exists()
    assert json.loads((out / "metadata.json").read_text())["status"] == "error"
    assert (out / "response-attempt-1.json").exists()
