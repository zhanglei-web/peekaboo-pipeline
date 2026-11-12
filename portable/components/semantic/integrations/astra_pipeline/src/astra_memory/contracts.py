"""Intermediate contracts: the model proposes facts, never global IDs or results."""
from __future__ import annotations

from jsonschema import Draft202012Validator

from .config import closed
from .io import require

TEXT = {"type": "string", "minLength": 1}
STRINGS = {"type": "array", "items": TEXT, "uniqueItems": True}
SUPPORT = {**STRINGS, "minItems": 1}
NULL_TEXT = {"type": ["string", "null"]}
STATUS = {"enum": ["observed", "inferred", "unknown"]}


def rows(schema):
    return {"type": "array", "items": schema}


OBJECT = closed({
    "local_id": TEXT, "name": TEXT, "category": TEXT,
    "appearance": closed({"color": NULL_TEXT, "shape": NULL_TEXT, "material_appearance": NULL_TEXT,
        "size_description": NULL_TEXT, "distinguishing_marks": STRINGS}),
    "visibility": {"enum": ["visible", "partially_visible"]},
    "bbox": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
             "minItems": 4, "maxItems": 4},
    "identity": closed({"decision": {"enum": ["new_candidate", "match_candidate", "uncertain"]},
        "candidate_ids": STRINGS, "basis": TEXT}),
    "state_description": TEXT, "epistemic_status": STATUS, "support_image_refs": SUPPORT,
})
RELATION = closed({"subject_local_id": TEXT,
    "predicate": {"enum": ["on", "inside", "mounted_on", "part_of", "left_of_image", "right_of_image",
        "above_image", "below_image", "in_front_of", "behind", "occluding", "adjacent_in_image", "overlaps_in_image"]},
    "object_local_id": TEXT, "description": TEXT, "epistemic_status": STATUS, "support_image_refs": SUPPORT})
INTERACTION = closed({"actor_local_id": TEXT, "object_local_id": TEXT,
    "action": {"enum": ["near", "touching", "holding", "uncertain"]},
    "description": TEXT, "epistemic_status": STATUS, "support_image_refs": SUPPORT})
CHANGE = closed({"object_local_id": TEXT,
    "kind": {"enum": ["moved", "lifted", "placed", "opened", "closed", "revealed", "hidden"]},
    "before_image_ref": TEXT, "after_image_ref": TEXT, "description": TEXT, "epistemic_status": STATUS})
OBSERVATION_SCHEMA = closed({"schema": {"const": "astra_memory.visual_observation.v1"},
    "image_ref": TEXT, "scene_description": TEXT, "objects": rows(OBJECT), "relations": rows(RELATION),
    "interactions": rows(INTERACTION), "changes": rows(CHANGE), "uncertainties": STRINGS})

# Independently versioned semantic role interfaces.
REVIEW_SCHEMA = closed({"schema": {"const": "astra_memory.review.v1"}, "question_id": TEXT,
    "decision": {"enum": ["same_instance", "different_instance", "insufficient_evidence"]},
    "candidate_refs": STRINGS, "support_image_refs": SUPPORT, "visual_basis": TEXT,
    "contradictions": STRINGS, "uncertainties": STRINGS})
KNOWLEDGE_SCHEMA = closed({"schema": {"const": "astra_memory.category_knowledge.v1"}, "category": TEXT,
    "usual_locations": STRINGS, "functions": STRINGS, "functional_partners": STRINGS,
    "conditional_affordances": closed({"containment": NULL_TEXT, "support": NULL_TEXT, "occlusion": NULL_TEXT}),
    "limitations": STRINGS})
UNIT_SCHEMA = closed({"schema": {"const": "astra_memory.unit_proposals.v1"},
    "proposals": rows(closed({"local_ref": TEXT, "existing_unit_candidates": STRINGS,
        "label": TEXT, "type": {"enum": ["inspection_area", "container_interior", "covered_surface", "exposed_surface"]},
        "is_container": {"type": ["boolean", "null"]}, "source_entity_candidates": STRINGS,
        "boundary": TEXT, "anchor_rule": TEXT, "support_image_refs": SUPPORT, "uncertainties": STRINGS})),
    "uncertainties": STRINGS})
# v1 was reserved before implementation. v2 makes anchoring and revisions explicit.
UNIT_SCHEMA_V2 = closed({"schema": {"const": "astra_memory.unit_proposals.v2"},
    "proposals": rows(closed({"local_ref": TEXT, "existing_unit_candidates": STRINGS,
        "label": TEXT, "type": {"enum": ["inspection_area", "container_interior", "covered_surface", "exposed_surface"]},
        "is_container": {"type": ["boolean", "null"]}, "container_visual_basis": NULL_TEXT,
        "source_entity_candidates": {**STRINGS, "minItems": 1}, "anchor_entity_id": TEXT,
        "anchor_kind": {"enum": ["object_interior", "object_surface", "object_region", "surface_patch"]},
        "boundary_key": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
        "boundary": TEXT, "anchor_rule": TEXT, "support_image_refs": SUPPORT, "uncertainties": STRINGS})),
    "uncertainties": STRINGS})
EVENT_SCHEMA = closed({"schema": {"const": "astra_memory.event_comparison.v1"}, "event_ref": TEXT,
    "search_region_proposal": TEXT, "event_summary": TEXT, "observed_changes": STRINGS,
    "unit_link_proposals": STRINGS,
    "target_evidence": closed({"category_match": {"enum": ["supported", "not_supported", "unknown"]},
        "instance_match": {"enum": ["supported", "not_supported", "unknown"]},
        "interaction_state": TEXT, "support_image_refs": STRINGS, "missing_evidence": STRINGS}),
    "final_location_candidate": {"oneOf": [{"type": "null"}, closed({
        "unit_candidates": STRINGS, "description": TEXT, "support_image_refs": SUPPORT})]},
    "uncertainties": STRINGS})

SCHEMAS = {"A": OBSERVATION_SCHEMA, "B": REVIEW_SCHEMA, "C": KNOWLEDGE_SCHEMA,
           "D": UNIT_SCHEMA_V2, "E": EVENT_SCHEMA}


def validate_review(doc, question_id, candidate_refs, image_refs, co_visible=False):
    Draft202012Validator(REVIEW_SCHEMA).validate(doc)
    require(doc["question_id"] == question_id, "review question mismatch")
    require(set(doc["candidate_refs"]) == set(candidate_refs), "review candidate mismatch")
    require(set(doc["support_image_refs"]) <= set(image_refs), "review cites unsupplied evidence")
    require(not co_visible or doc["decision"] != "same_instance", "co-visible instances cannot be the same instance")
    return doc


def validate_event(doc, event_ref, unit_ids, image_refs):
    Draft202012Validator(EVENT_SCHEMA).validate(doc)
    require(doc["event_ref"] == event_ref, "event comparison reference mismatch")
    require(set(doc["unit_link_proposals"]) <= set(unit_ids), "event cites unknown/future unit")
    evidence = doc["target_evidence"]
    require(set(evidence["support_image_refs"]) <= set(image_refs), "target cites unsupplied image")
    if evidence["category_match"] == "supported" or evidence["instance_match"] == "supported":
        require(evidence["support_image_refs"], "supported target claim needs visual evidence")
    require(evidence["instance_match"] != "supported", "no reference identity supplied; instance match must remain unconfirmed")
    final = doc["final_location_candidate"]
    if final:
        require(set(final["unit_candidates"]) <= set(unit_ids), "location cites unknown/future unit")
        require(set(final["support_image_refs"]) <= set(image_refs), "location cites unsupplied image")
    return doc


def validate_knowledge(doc, category):
    Draft202012Validator(KNOWLEDGE_SCHEMA).validate(doc)
    require(doc["category"] == category, "knowledge category differs from request")
    for value in doc["conditional_affordances"].values():
        require(value is None or value.strip(), "empty conditional affordance")
    return doc


def validate_units(doc, current_ref, entity_evidence, existing_units, spatial_facts=()):
    Draft202012Validator(UNIT_SCHEMA_V2).validate(doc)
    local_ids = [p["local_ref"] for p in doc["proposals"]]
    require(len(set(local_ids)) == len(local_ids), "duplicate local unit proposal")
    for p in doc["proposals"]:
        require(set(p["source_entity_candidates"]) <= entity_evidence.keys(), "unit cites unknown/future entity")
        require(p["anchor_entity_id"] in p["source_entity_candidates"], "unit anchor must be a cited entity")
        require(set(p["existing_unit_candidates"]) <= existing_units.keys(), "unit cites unknown/future unit")
        require(current_ref in p["support_image_refs"], "unit must cite current evidence")
        for entity in p["source_entity_candidates"]:
            require(set(p["support_image_refs"]) & set(entity_evidence[entity]), "unit entity lacks supporting supplied image")
        allowed_refs = set().union(*(set(v) for v in entity_evidence.values())) if entity_evidence else set()
        require(set(p["support_image_refs"]) <= allowed_refs, "unit cites unknown/future image")
        expected_anchor = {"container_interior": "object_interior", "exposed_surface": "object_surface",
                           "covered_surface": "surface_patch", "inspection_area": "object_region"}
        require(p["anchor_kind"] == expected_anchor[p["type"]], "unit type/anchor_kind mismatch")
        if p["type"] == "container_interior":
            require(p["is_container"] is True and p["container_visual_basis"] is not None and
                    p["container_visual_basis"].strip(), "container interior requires visual container evidence")
        if p["type"] in {"covered_surface", "exposed_surface"}:
            require(p["is_container"] is False, "surface is not a container interior")
        if p["type"] == "covered_surface":
            require(len(p["source_entity_candidates"]) >= 2, "covered surface must cite surface and covering object")
            require(any(subject in p["source_entity_candidates"] and subject != p["anchor_entity_id"] and
                        target == p["anchor_entity_id"] and predicate in {"on", "occluding"}
                        for subject, predicate, target in spatial_facts),
                    "covered surface needs an observed covering-on-surface/occluding-surface relation")
        # Different anchors/types may be plausible alternatives, never automatic merges.
    return doc


def validate_observation(doc, image_ref, support_refs, past_refs, candidate_ids):
    Draft202012Validator(OBSERVATION_SCHEMA).validate(doc)
    require(doc["image_ref"] == image_ref, "observation image_ref differs from current image")
    local = {o["local_id"] for o in doc["objects"]}
    require(len(local) == len(doc["objects"]), "duplicate local object IDs")
    for o in doc["objects"]:
        x1, y1, x2, y2 = o["bbox"]
        require(x1 < x2 and y1 < y2, "bbox must be x1,y1,x2,y2 with positive area")
        require(set(o["support_image_refs"]) <= set(support_refs) and image_ref in o["support_image_refs"],
                "current object must cite the current image, never future evidence")
        ids = o["identity"]["candidate_ids"]
        require(set(ids) <= set(candidate_ids), "identity proposal references unknown or future instance")
        if o["identity"]["decision"] == "match_candidate":
            require(len(ids) == 1, "match_candidate requires exactly one past candidate")
        if o["identity"]["decision"] == "new_candidate":
            require(not ids, "new_candidate must not claim a historical match")
    for r in doc["relations"]:
        require(r["subject_local_id"] in local and r["object_local_id"] in local,
                "relation endpoint is not a current observation")
        require(r["subject_local_id"] != r["object_local_id"], "self spatial relation")
        require(set(r["support_image_refs"]) <= set(support_refs) and image_ref in r["support_image_refs"],
                "relation has invalid or noncurrent evidence")
    for r in doc["interactions"]:
        require(r["actor_local_id"] in local and r["object_local_id"] in local,
                "interaction endpoint is not a current observation")
        require(r["actor_local_id"] != r["object_local_id"], "self interaction")
        require(set(r["support_image_refs"]) <= set(support_refs) and image_ref in r["support_image_refs"],
                "interaction evidence is invalid")
    for c in doc["changes"]:
        require(c["object_local_id"] in local, "change references unknown observation")
        require(c["before_image_ref"] in past_refs and c["after_image_ref"] == image_ref,
                "change must compare supplied past evidence with current evidence")
    return doc
