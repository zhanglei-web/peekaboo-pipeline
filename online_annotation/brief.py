"""Compact transport for two complete visual passes, with full catalog expansion.

Only transport changes: the active v3 region and review rules remain inputs.
The model still supplies boundaries, existence evidence and uncertainty.
"""
from pathlib import Path
import json

import jsonschema

from .compact import KINDS, VISIBILITY, CONFIDENCE
from .contracts import object_schema, validate_proposals
from .model import save_json

ROOT = Path(__file__).resolve().parent


def tuple_schema(items):
    # The installed validator defaults to draft 7; the local runtime also
    # supports its positional items form.
    return {"type": "array", "items": items, "additionalItems": False,
            "minItems": len(items), "maxItems": len(items)}


def schema_for(frames):
    text = {"type": "string", "minLength": 1}
    coordinate = {"type": "integer", "minimum": 0, "maximum": 1000}
    evidence = {"type": "array", "minItems": 1, "maxItems": 3,
                "items": tuple_schema([
                    {"type": "integer", "enum": list(frames)},
                    coordinate, coordinate, coordinate, coordinate])}
    parent_evidence = {**evidence, "maxItems": 2}
    return object_schema({
        "s": {"type": "string"},
        "p": {"type": "array", "maxItems": 12,
              "items": tuple_schema([text, parent_evidence])},
        "r": {"type": "array", "maxItems": 16,
              "items": tuple_schema([
                  {"type": "integer", "minimum": 0, "maximum": 11},
                  text, {"type": "string", "enum": list(KINDS)},
                  text, text,
                  {"type": "string", "enum": list(VISIBILITY)},
                  {"type": "string", "enum": list(CONFIDENCE)}, evidence])},
        "u": {"type": "array", "maxItems": 10, "items": text},
    })


def expand(data, frames):
    jsonschema.validate(data, schema_for(frames))

    def evidence(rows):
        return [{"frame_index": row[0], "bbox_1000": row[1:]} for row in rows]

    parents = [{"ref": f"p{i}", "name": row[0], "description": row[0],
                "evidence": evidence(row[1])} for i, row in enumerate(data["p"])]
    regions = []
    for i, (parent, name, kind, boundary, basis, visibility, confidence, ev) in enumerate(data["r"]):
        if parent >= len(parents):
            raise ValueError("Brief region references an unknown parent")
        container = kind in ("i", "m")
        if container == (visibility == "a"):
            raise ValueError("Brief region visibility conflicts with region type")
        region_type, anchor = KINDS[kind]
        parent_name = parents[parent]["name"]
        if not name.startswith(parent_name):
            name = parent_name + "·" + name
        regions.append({
            "ref": f"r{i}", "name": name, "parent_ref": f"p{parent}",
            "region_type": region_type, "identity_anchor": anchor,
            "boundary_description": boundary, "existence_basis": basis,
            "interior_visibility": VISIBILITY[visibility],
            "confidence": CONFIDENCE[confidence], "evidence": evidence(ev),
        })
    result = {"scene_description": data["s"], "parents": parents,
              "regions": regions, "uncertainties": data["u"]}
    validate_proposals(result, frames)
    return result


def propose(model, output, images, selected, base_prompt, review_template):
    output = Path(output)
    frames = [row["frame_index"] for row in selected]
    transport = (ROOT / "prompts/brief_transport.txt").read_text()
    schema = schema_for(frames)
    prompt = base_prompt + "\n\n" + transport
    initial, initial_receipt = model.infer(output / "model/regions", prompt, images, schema, 6500)
    draft = expand(initial, frames)
    save_json(output / "model/regions/expanded_proposals.json", draft)
    print(json.dumps({"stage": "brief_full_review", "initial_regions": len(draft["regions"])},
                     ensure_ascii=False), flush=True)
    reviewed, review_receipt = model.infer(
        output / "model/region_review",
        prompt + "\n\n" + (ROOT / "prompts/brief_review.txt").read_text()
        + "\n" + json.dumps(initial, ensure_ascii=False, separators=(",", ":")),
        images, schema, 6500)
    final = expand(reviewed, frames)
    save_json(output / "model/region_review/expanded_proposals.json", final)
    return draft, final, initial_receipt, review_receipt
