"""Compact model transport; the public candidate catalog keeps its full fields."""
import copy
import re

import jsonschema

from .contracts import PROPOSALS, object_schema, validate_proposals


BOX = {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
       "minItems": 4, "maxItems": 4}
PARENT = object_schema({
    "n": {"type": "string", "minLength": 1},
    "k": {"type": "string", "enum": ["s", "c", "f"]},
    "b": BOX,
})
REGION = object_schema({
    "p": {"type": "integer", "minimum": 0, "maximum": 11},
    "n": {"type": "string", "minLength": 1},
    "t": {"type": "string", "enum": ["s", "o", "m", "i"]},
    "d": {"type": "string", "minLength": 1},
    "e": {"type": "string", "minLength": 1},
    "b": BOX,
    "v": {"type": "string", "enum": ["a", "v", "p", "n"]},
    "c": {"type": "string", "enum": ["h", "m", "l"]},
})
FLOOR = {"type": "string", "enum": ["y", "n", "u"]}
UNCERTAINTIES = {"type": "array", "maxItems": 10, "items": {"type": "string"}}
COMPACT = object_schema({
    "d": {"type": "string"},
    "f": FLOOR,
    "p": {"type": "array", "maxItems": 12, "items": PARENT},
    "r": {"type": "array", "maxItems": 16, "items": REGION},
    "u": UNCERTAINTIES,
})
FINAL_COMPACT = copy.deepcopy(COMPACT)
FINAL_COMPACT["properties"]["r"]["maxItems"] = 24
EXPANDED = copy.deepcopy(PROPOSALS)
EXPANDED["properties"]["regions"]["maxItems"] = 24
PATCH = object_schema({
    "remove": {"type": "array", "uniqueItems": True, "maxItems": 24,
               "items": {"type": "integer", "minimum": 0, "maximum": 23}},
    "replace": {"type": "array", "maxItems": 24, "items": object_schema({
        "index": {"type": "integer", "minimum": 0, "maximum": 23}, "region": REGION,
    })},
    "add": {"type": "array", "maxItems": 24, "items": REGION},
    "parent_add": {"type": "array", "maxItems": 12, "items": PARENT},
    "f": FLOOR,
    "u": UNCERTAINTIES,
})
KINDS = {
    "s": ("open_support_surface", "fixed_support"),
    "o": ("occluder_footprint", "original_footprint"),
    "m": ("container_interior", "movable_container"),
    "i": ("container_interior", "fixed_support"),
}
VISIBILITY = {"a": "not_applicable", "v": "visible", "p": "partially_visible", "n": "not_visible"}
CONFIDENCE = {"h": "high", "m": "medium", "l": "low"}


def expand(data, frame_index, final=False):
    """Decode only model facts and deterministic field encodings; invent no regions."""
    jsonschema.validate(data, FINAL_COMPACT if final else COMPACT)
    parents = []
    for i, parent in enumerate(data["p"]):
        parents.append({"ref": f"p{i}", "name": parent["n"],
                        "description": parent["n"],
                        "evidence": [{"frame_index": frame_index, "bbox_1000": parent["b"]}]})
    regions = []
    for i, row in enumerate(data["r"]):
        if row["p"] >= len(parents):
            raise ValueError("Compact region references an unknown parent")
        container = row["t"] in ("m", "i")
        if container == (row["v"] == "a"):
            raise ValueError("Compact visibility is incompatible with region type")
        kind, anchor = KINDS[row["t"]]
        name = row["n"]
        if not name.startswith(parents[row["p"]]["name"]):
            name = parents[row["p"]]["name"] + "·" + name
        regions.append({
            "ref": f"r{i}", "name": name, "region_type": kind, "parent_ref": f'p{row["p"]}',
            "boundary_description": row["d"], "existence_basis": row["e"],
            "identity_anchor": anchor, "interior_visibility": VISIBILITY[row["v"]],
            "confidence": CONFIDENCE[row["c"]],
            "evidence": [{"frame_index": frame_index, "bbox_1000": row["b"]}],
        })
    result = {"scene_description": data["d"], "parents": parents, "regions": regions,
              "uncertainties": data["u"]}
    validate_proposals(result, [frame_index], EXPANDED if final else None)
    return result


def risks(data):
    """Limited structural checks; these do not prove semantic completeness."""
    flags = []
    floor = [i for i, p in enumerate(data["p"]) if p["k"] == "f"]
    floor_regions = [i for i, r in enumerate(data["r"]) if r["p"] in floor and r["t"] == "s"]
    if data["f"] == "y" and not floor_regions:
        flags.append("floor_visible_but_missing: self-report says visible floor, but no floor surface")
    if len(floor_regions) > 1:
        description = " ".join(data["r"][i]["d"] for i in floor_regions)
        if not re.search("墙|台阶|门槛|隔断|不同房间", description):
            flags.append("floor_fragmentation: check whether adjacent floor pieces are one continuous surface")
    for i, row in enumerate(data["r"]):
        text = row["n"] + " " + row["d"]
        if row["t"] in ("m", "i") and re.search(
            r"多个|整堆|若干|一组|整组|各个|所有|堆叠(?:的)?(?:纸箱|箱子|盒子)", text
        ):
            flags.append(f"merged_containers: region[{i}] may combine independently checkable containers")
        if re.search("其余|其他区域|杂物区", row["n"]):
            flags.append(f"vague_boundary: region[{i}] has a residual category name")
    return flags


def apply_patch(data, patch, frame_index):
    jsonschema.validate(patch, PATCH)
    replacements = {item["index"]: item["region"] for item in patch["replace"]}
    if len(replacements) != len(patch["replace"]):
        raise ValueError("Patch replaces a region more than once")
    remove = set(patch["remove"])
    if remove & set(replacements):
        raise ValueError("Patch removes and replaces the same region")
    if any(i >= len(data["r"]) for i in remove | set(replacements)):
        raise ValueError("Patch references an unknown original region index")
    revised = copy.deepcopy(data)
    revised["p"] += patch["parent_add"]
    revised["r"] = [replacements.get(i, row) for i, row in enumerate(data["r"]) if i not in remove]
    revised["r"] += patch["add"]
    revised["f"], revised["u"] = patch["f"], patch["u"]
    expand(revised, frame_index, final=True)
    return revised
