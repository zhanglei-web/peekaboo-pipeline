"""Compact transport for two complete visual passes, with full catalog expansion.

Only transport changes: the active v3 region and review rules remain inputs.
The model still supplies boundaries, existence evidence and uncertainty.
"""
from pathlib import Path
from copy import deepcopy
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
    repair_schema = deepcopy(schema)
    named_evidence = object_schema({
        "frame_index": {"type": "integer", "enum": frames},
        **{key: {"type": "integer", "minimum": 0, "maximum": 1000}
           for key in ("x_min", "y_min", "x_max", "y_max")}})
    repair_schema["properties"]["p"]["items"]["items"][1]["items"] = named_evidence
    repair_schema["properties"]["r"]["items"]["items"][7]["items"] = named_evidence
    prompt = base_prompt + "\n\n" + transport
    initial, initial_receipt = model.infer(output / "model/regions", prompt, images, schema, 6500)
    def expand_or_reask(value, receipt, stage):
        for attempt in range(4):
            try:
                return value, receipt, expand(value, frames)
            except (ValueError, jsonschema.ValidationError) as exc:
                if attempt == 3:
                    raise
                # Ask the same visual model to resolve the contradiction. Do not
                # silently rewrite its type, visibility, or boundary judgment.
                repair_prompt = (prompt + "\n\n校验失败，请重新查看全部图像并输出修订后的完整JSON。"
                                 "类型 s/o 的内部可见性必须为 a；类型 i/m 的内部可见性必须为 v/p/n。"
                                 "每个证据严格按 [帧号,x_min,y_min,x_max,y_max]，必须 x_min<x_max 且 y_min<y_max；"
                                 "框应包围可见锚点且有正面积。相同 y_min/y_max 是无效的零高度框。"
                                 "应使区域的类型、边界和图像证据一致。"
                                 "本轮修订的所有图像证据改用具名对象："
                                 "{frame_index,x_min,y_min,x_max,y_max}，与返回模式一致。"
                                 "x_min/x_max是水平方向左/右坐标；y_min/y_max是竖直方向上/下坐标。"
                                 f"\n错误：{exc}\n原始输出：" + json.dumps(value, ensure_ascii=False))
                fixed, fixed_receipt = model.infer(output / f"model/{stage}_repair_{attempt + 1}",
                                                   repair_prompt, images, repair_schema, 6500)
                # Convert named fields to the unchanged public transport ordering.
                # Values are never reordered, clamped, or inferred by the program.
                fixed = deepcopy(fixed)
                for row in fixed["p"] + fixed["r"]:
                    row[-1] = [[e[key] for key in ("frame_index", "x_min", "y_min", "x_max", "y_max")]
                               if isinstance(e, dict) else e for e in row[-1]]
                receipt = {**fixed_receipt,
                           "seconds": receipt["seconds"] + fixed_receipt["seconds"],
                           "previous_attempt": receipt, "validation_repair_executed": True}
                value = fixed
    initial, initial_receipt, draft = expand_or_reask(initial, initial_receipt, "regions")
    save_json(output / "model/regions/expanded_proposals.json", draft)
    print(json.dumps({"stage": "brief_full_review", "initial_regions": len(draft["regions"])},
                     ensure_ascii=False), flush=True)
    reviewed, review_receipt = model.infer(
        output / "model/region_review",
        prompt + "\n\n" + (ROOT / "prompts/brief_review.txt").read_text()
        + "\n" + json.dumps(initial, ensure_ascii=False, separators=(",", ":")),
        images, schema, 6500)
    reviewed, review_receipt, final = expand_or_reask(reviewed, review_receipt, "region_review")
    save_json(output / "model/region_review/expanded_proposals.json", final)
    return draft, final, initial_receipt, review_receipt
