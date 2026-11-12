"""Read-only adaptation of existing scene IDs and event-local entity references."""
from copy import deepcopy
from pathlib import Path
import re
from PIL import Image
from .common import sha


def english_prompt(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each candidate needs an explicit nonempty detection prompt")
    value = value.strip().lower().rstrip(".").strip()
    # A prompt is one noun phrase, not executable instructions or multiple captions.
    if not re.fullmatch(r"[a-z0-9][a-z0-9 '\-/]*", value) or len(value) > 120:
        raise ValueError("Use a short English noun phrase in the audited prompt map: " + value)
    return value


def plan(doc, prompt_map, scope="scene_and_focus", verify_images=True):
    if "frame_anchor_enrichment" in doc:
        raise ValueError("Input already has enrichment metadata; never silently replace it")
    if scope not in ("scene_and_focus", "focus", "event_description"):
        raise ValueError("Unknown candidate scope")
    if scope == 'event_description':
        from .description_scope import check_mapping
        check_mapping(doc,prompt_map)
    scene = {}
    for obj in doc["scene_memory"]["objects"]:
        if obj["id"] in scene:
            raise ValueError("Duplicate existing scene object ID")
        scene[obj["id"]] = obj
    catalog = doc["search_event_states"]["entity_reference_catalog"]
    if not isinstance(catalog, dict):
        raise ValueError("Expected event-local entity reference catalog")
    frames, seen, event_ids = [], set(), set()
    used_prompts = set()
    for event in doc["search_event_states"]["events"]:
        if event["event_id"] in event_ids:
            raise ValueError("Duplicate event ID")
        event_ids.add(event["event_id"])
        if [s["phase"] for s in event["states"]] != ["start", "middle", "end"]:
            raise ValueError("Expected exactly ordered start/middle/end states")
        for state in event["states"]:
            sid = state["state_id"]
            if sid in seen or "object_anchors" in state:
                raise ValueError("Duplicate state ID or existing anchor field")
            seen.add(sid)
            candidates = []
            if scope == "scene_and_focus":
                for oid, obj in scene.items():
                    key = "scene:" + oid
                    candidates.append({"candidate_id": key, "reference_kind": "scene_object",
                        "source_ref": oid, "source_label": obj["identity"]["name"],
                        "existing_identity_status": obj["identity"].get("identity_status"),
                        "scene_entity_candidates": [oid], "is_frame_focus": False,
                        "prompt": english_prompt(prompt_map[key])})
                    used_prompts.add(key)
            focus_seen = set()
            for focus in state.get("focus_object_states", []):
                ref = focus["entity_ref"]
                if ref in focus_seen:
                    raise ValueError("Duplicate frame focus reference")
                focus_seen.add(ref)
                if ref not in catalog:
                    raise ValueError("Dangling existing entity reference: " + ref)
                item = catalog[ref]
                if not set(item.get("memory_entity_candidates", [])) <= set(scene):
                    raise ValueError("Dangling existing scene candidate reference")
                if scope == 'event_description':
                    # This mode uses the explicitly translated, evidence-bound mapping below.
                    # Original labels may be Chinese and must remain unchanged.
                    continue
                key = "entity:" + ref
                prompt = prompt_map.get(key, item["label"])
                candidates.append({"candidate_id": key, "reference_kind": "event_entity",
                    "source_ref": ref, "source_label": item["label"],
                    "existing_identity_status": item.get("link_status"),
                    "scene_entity_candidates": item.get("memory_entity_candidates", []),
                    "is_frame_focus": True, "prompt": english_prompt(prompt)})
                if key in prompt_map:
                    used_prompts.add(key)
            if scope == 'event_description':
                from .description_scope import candidates_for_event
                candidates=candidates_for_event(event,state,prompt_map,scene,catalog)
            image = Path(state["image_path"])
            if not image.is_absolute():
                raise ValueError("Image path must be absolute")
            if verify_images and sha(image) != state["image_sha256"]:
                raise ValueError("Original frame hash differs from source JSON: " + str(image))
            with Image.open(image) as im:
                # Preserve native raster axes; never silently rotate by EXIF.
                if im.getexif().get(274, 1) != 1:
                    raise ValueError("Nontrivial EXIF orientation requires explicit coordinate migration")
                width, height = im.size
            frames.append({"event_id": event["event_id"], "state_id": sid,
                "phase": state["phase"], "frame_index": state["original_video_frame_index"],
                "image_path": str(image), "image_sha256": state["image_sha256"],
                "width": width, "height": height, "candidates": candidates})
            if scope == 'event_description':
                frames[-1]['candidate_scope']='event_description'
                frames[-1]['excluded_description_mentions']=prompt_map['events'][event['event_id']].get('excluded_mentions',[])
    if not frames:
        raise ValueError("Empty episode")
    extra = set(prompt_map) - used_prompts if scope != 'event_description' else set()
    if extra:
        raise ValueError("Unused/unknown prompt-map keys: " + repr(sorted(extra)))
    return frames


def augment(original, results, metadata):
    doc = deepcopy(original)
    states = {s["state_id"]: s for e in doc["search_event_states"]["events"] for s in e["states"]}
    if set(results) != set(states):
        raise ValueError("Must enrich every original state exactly once")
    for sid, result in results.items():
        states[sid]["object_anchors"] = deepcopy(result)
    doc["frame_anchor_enrichment"] = deepcopy(metadata)
    return doc


def assert_preserved(original, enriched):
    restored = deepcopy(enriched)
    del restored["frame_anchor_enrichment"]
    for e in restored["search_event_states"]["events"]:
        for state in e["states"]:
            del state["object_anchors"]
    if restored != original:
        raise ValueError("An original JSON value was modified")
