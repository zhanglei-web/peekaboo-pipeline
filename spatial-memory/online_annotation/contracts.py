"""Structured model outputs plus reference and image-boundary checks."""
import jsonschema


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


EVIDENCE = object_schema({
    "frame_index": {"type": "integer", "minimum": 0},
    "bbox_1000": {"type": "array", "items": {"type": "integer", "minimum": 0, "maximum": 1000},
             "minItems": 4, "maxItems": 4},
})

KEYFRAMES = object_schema({
    "selected_frame_indices": {"type": "array", "items": {"type": "integer"},
                               "minItems": 1, "maxItems": 4, "uniqueItems": True},
    "reason": {"type": "string"},
})

PROPOSALS = object_schema({
    "scene_description": {"type": "string"},
    "parents": {"type": "array", "maxItems": 12, "items": object_schema({
        "ref": {"type": "string"}, "name": {"type": "string"},
        "description": {"type": "string"},
        "evidence": {"type": "array", "minItems": 1, "maxItems": 2, "items": EVIDENCE},
    })},
    "regions": {"type": "array", "maxItems": 16, "items": object_schema({
        "ref": {"type": "string"}, "name": {"type": "string"},
        "region_type": {"type": "string", "enum": [
            "open_support_surface", "occluder_footprint", "container_interior"]},
        "parent_ref": {"type": "string"},
        "boundary_description": {"type": "string"},
        "existence_basis": {"type": "string"},
        "identity_anchor": {"type": "string", "enum": [
            "fixed_support", "original_footprint", "movable_container"]},
        "interior_visibility": {"type": "string", "enum": [
            "visible", "partially_visible", "not_visible", "not_applicable"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "evidence": {"type": "array", "minItems": 1, "maxItems": 3, "items": EVIDENCE},
    })},
    "uncertainties": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
})


def validate_keyframes(data, allowed):
    jsonschema.validate(data, KEYFRAMES)
    if not set(data["selected_frame_indices"]).issubset(set(allowed)):
        raise ValueError("Model selected a frame outside the provided observation shortlist")


def validate_proposals(data, allowed, schema=None):
    jsonschema.validate(data, PROPOSALS if schema is None else schema)
    allowed = set(allowed)
    parents = [row["ref"] for row in data["parents"]]
    regions = [row["ref"] for row in data["regions"]]
    if len(set(parents + regions)) != len(parents + regions):
        raise ValueError("Model references must be unique across structures and regions")
    for row in data["regions"]:
        if row["parent_ref"] not in parents:
            raise ValueError("Region references an unknown parent")
        if row["region_type"] == "occluder_footprint" and row["identity_anchor"] != "original_footprint":
            raise ValueError("Occluder footprint must stay anchored to original support position")
    for row in data["parents"] + data["regions"]:
        for evidence in row["evidence"]:
            if evidence["frame_index"] not in allowed:
                raise ValueError("Model evidence references an unseen frame")
            x1, y1, x2, y2 = evidence["bbox_1000"]
            if x2 <= x1 or y2 <= y1:
                raise ValueError("Model evidence box has non-positive area")


def build_catalog(proposals, episode_id, selected, audit, model, proposal_schema=None):
    validate_proposals(proposals, [row["frame_index"] for row in selected], proposal_schema)
    parent_ids = {row["ref"]: f"{episode_id}:P{i:03d}"
                  for i, row in enumerate(proposals["parents"], 1)}
    def normalized(row):
        return {**row, "evidence": [{**evidence, "bbox": [x / 1000 for x in evidence["bbox_1000"]]}
                                    for evidence in row["evidence"]]}
    parents = [{**normalized(row), "parent_id": parent_ids[row["ref"]], "selectable": False}
               for row in proposals["parents"]]
    regions = []
    for i, row in enumerate(proposals["regions"], 1):
        regions.append({
            **normalized(row), "candidate_id": f"{episode_id}:R{i:03d}",
            "parent_id": parent_ids[row["parent_ref"]],
            "proposal_status": "pending" if row["confidence"] == "low" else "model_supported",
            "identity_status": "provisional_within_recording",
            "source_model": model, "geometry_status": "unknown",
            "geometry_support": None, "boundary_geometry": None,
            "bbox_semantics": "visible_anchor_evidence; not_hidden_search_space_geometry",
            "searched": False, "visit_count": 0,
            "single_event_sufficiency": None, "cumulative_sufficiency": None,
            "history": [],
        })
    return {
        "schema_version": "online.candidate_catalog.v0.1",
        "episode_id": episode_id, "catalog_version": 0,
        "mode": "initial_semantic_proposals",
        "coordinate_conversion": "model bbox_1000 integer -> catalog bbox normalized by /1000; both retained",
        "scene_description": proposals["scene_description"],
        "observation_end_exclusive_ns": audit["observation_end_exclusive_ns"],
        "source_frames": selected, "parents": parents, "regions": regions,
        "uncertainties": proposals["uncertainties"],
        "limitations": [
            "Model-supported semantic candidates; no independent 3D identity confirmation.",
            "Pixel boxes identify visible evidence anchors, not hidden-space coverage boundaries.",
            "No event scoring or full hierarchy aggregation performed.",
        ],
    }
