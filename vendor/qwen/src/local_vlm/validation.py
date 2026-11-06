import json
from jsonschema import Draft202012Validator
from .config import ROOT

SCHEMA = json.loads((ROOT / "schemas/event_annotation.schema.json").read_text())

def validate_annotation(data, manifest):
    Draft202012Validator(SCHEMA).validate(data)
    frame_ids = {f["frame_id"] for f in manifest["frames"]}
    ids = [o["id"] for o in data["objects"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object IDs")
    object_ids = set(ids)
    if data["clip_id"] != manifest["clip_id"]:
        raise ValueError("clip_id mismatch")
    def evidence(values):
        if not set(values) <= frame_ids:
            raise ValueError(f"Unknown evidence_frames: {set(values) - frame_ids}")
    def objects(values):
        if not set(values) <= object_ids:
            raise ValueError(f"Unknown object references: {set(values) - object_ids}")
    for obj in data["objects"]:
        evidence(obj["evidence_frames"])
    for key, index in [("before_state", 0), ("after_state", -1)]:
        state = data[key]
        if state["frame_id"] != manifest["frames"][index]["frame_id"]:
            raise ValueError(f"{key} must refer to the corresponding endpoint frame")
        objects(state["visible_objects"])
        for relation in state["relations"]:
            objects([relation["subject"], relation["object"]])
            if not {relation["subject"], relation["object"]} <= set(state["visible_objects"]):
                raise ValueError("State relation references an invisible object")
    for event in data["events"]:
        evidence(event["evidence_frames"])
        objects(event["object_ids"])
    return data

def parse_response(response, manifest, task):
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Incomplete response: finish_reason={choice.get('finish_reason')}")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip() or "<think>" in content or "</think>" in content:
        raise ValueError("Missing final content or reasoning leaked into final content")
    if task == "event_annotation":
        return validate_annotation(json.loads(content), manifest)
    return {"clip_id": manifest["clip_id"], "answer": content}
