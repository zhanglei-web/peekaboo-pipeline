"""Explicit human submissions, immutable revisions and evidence-bound decisions."""
from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from .config import closed
from .contracts import TEXT, STRINGS, NULL_TEXT
from .io import digest, load, require, safe_file, save, sha
from .s2 import now
from .s3 import read_s3

LOCATION = closed({"unit_id": TEXT, "evidence_state_id": TEXT, "description": TEXT})
SUBMISSION_SCHEMA = closed({
    "schema": {"const": "astra_memory.human_submission.v1"}, "episode_id": TEXT, "s3_signature": TEXT,
    "full_search_coverage_confirmed": {"type": ["boolean", "null"]}, "search_scope": NULL_TEXT,
    "events": {"type": "array", "items": closed({"event_id": TEXT,
        "result": {"enum": ["unknown", "not_found", "found"]},
        "unit_ids": {"oneOf": [STRINGS, {"type": "null"}]},
        "confirmed_target_instance_id": NULL_TEXT,
        "final_location": {"oneOf": [LOCATION, {"type": "null"}]}, "note": NULL_TEXT})},
})


def submission_template(config):
    manifest, build = read_s3(config)
    events = load(build/"search_event_states.json")["events"]
    return {"schema": "astra_memory.human_submission.v1", "episode_id": config["episode_id"],
        "s3_signature": manifest["signature"], "full_search_coverage_confirmed": None, "search_scope": None,
        "events": [{"event_id": e["event_id"], "result": "unknown", "unit_ids": None,
            "confirmed_target_instance_id": None, "final_location": None, "note": None} for e in events]}


def validate_submission(doc, config, manifest, build):
    Draft202012Validator(SUBMISSION_SCHEMA).validate(doc)
    require(doc["episode_id"] == config["episode_id"] and doc["s3_signature"] == manifest["signature"],
            "human submission is bound to a different episode/S3 build")
    events = {e["event_id"]: e for e in load(build/"search_event_states.json")["events"]}
    objects = {o["id"] for o in load(build/"scene_memory.json")["objects"]}
    units = {u["unit_id"] for u in load(build/"search_units.json")["list"]}
    snapshots = load(build/"causal_snapshots.json")
    ids = [row["event_id"] for row in doc["events"]]
    require(ids == list(events), "human submission must specify every event exactly once in order")
    if doc["search_scope"] is not None:
        require(doc["search_scope"].strip(), "empty human search scope")
    confirmed_ids = set()
    for row in doc["events"]:
        eid, result = row["event_id"], row["result"]
        available = {u["unit_id"] for u in snapshots[events[eid]["states"][-1]["state_id"]]["search_units"]["list"]}
        if row["unit_ids"] is not None:
            require(row["unit_ids"] and set(row["unit_ids"]) <= units & available, "human unit link is empty, unknown or unavailable at event end")
        if result != "unknown" or row["unit_ids"] is not None:
            require(row["note"] and row["note"].strip(), "human decision needs a note")
        oid = row["confirmed_target_instance_id"]
        require(oid is None or (result == "found" and oid in objects), "target identity requires found and a known instance")
        if oid:
            # Confirmation can resolve identity only for instances actually observed by this event.
            observed = {focus["entity_ref"] for state in events[eid]["states"] for focus in state["focus_object_states"]}
            require(oid in observed, "confirmed target was not observed in this event")
            confirmed_ids.add(oid)
        location = row["final_location"]
        if location:
            require(result == "found", "final location requires confirmed found result")
            require(location["unit_id"] in available, "final location unit is unavailable in event")
            require(location["evidence_state_id"] in {s["state_id"] for s in events[eid]["states"]}, "final location state belongs to another event")
            if row["unit_ids"] is not None:
                require(location["unit_id"] in row["unit_ids"], "final location unit is outside the confirmed event units")
    require(len(confirmed_ids) <= 1, "conflicting confirmed target instances across events; resolve identity explicitly")
    return doc


def read_confirmation(config, manifest, build):
    root = Path(config["output"])/"confirmations"
    if not (root/"manifest.json").exists():
        return None
    pointer = load(root/"manifest.json")
    previous, current = None, None
    for entry in pointer["revisions"]:
        path = safe_file(root, entry["path"])
        require(sha(path) == entry["sha256"], "human confirmation revision changed")
        current = load(path)
        require(current["previous_revision_sha256"] == previous, "broken human revision chain")
        require(current["actor"].strip() and current["source"] == "explicit_cli_human_submission", "missing explicit human provenance")
        require(digest(current["submission"]) == current["submission_sha256"], "human submission content changed")
        original = safe_file(root, current["source_copy"])
        require(sha(original) == current["source_file_sha256"] and load(original) == current["submission"], "human source file changed")
        previous = entry["sha256"]
    require(current is not None and pointer["latest_revision_sha256"] == previous, "empty or inconsistent human revision pointer")
    # Older revisions may attest earlier S3 builds, but the current submission must bind this build.
    validate_submission(current["submission"], config, manifest, build)
    return {"record": current, "revision_sha256": previous}


def import_confirmation(config, path, actor):
    require(isinstance(actor, str) and actor.strip(), "actor is required for an explicit human submission")
    manifest, build = read_s3(config)
    source = Path(path).resolve()
    payload = source.read_bytes()
    doc = load(source)
    validate_submission(doc, config, manifest, build)
    root = Path(config["output"])/"confirmations"
    pointer_path = root/"manifest.json"
    pointer = load(pointer_path) if pointer_path.exists() else {"schema": "astra_memory.confirmation_ledger.v1", "revisions": []}
    # Validate the whole historical chain before extending it. A new build can receive a new submission.
    if pointer["revisions"]:
        latest_path = safe_file(root, pointer["revisions"][-1]["path"])
        latest = load(latest_path)
        if latest["submission"]["s3_signature"] == manifest["signature"]:
            old = read_confirmation(config, manifest, build)
            if old["record"]["submission"] == doc and old["record"]["actor"] == actor.strip():
                return {"status": "unchanged", "revision_sha256": old["revision_sha256"], "new_model_requests": 0}
        else:
            previous = None
            for entry in pointer["revisions"]:
                revision = safe_file(root, entry["path"])
                require(sha(revision) == entry["sha256"] and load(revision)["previous_revision_sha256"] == previous,
                        "broken previous human revision chain")
                previous = entry["sha256"]
            require(previous == pointer["latest_revision_sha256"], "human revision pointer mismatch")
    revision = len(pointer["revisions"])+1
    source_copy = f"sources/{revision:06d}.json"
    from .io import atomic_bytes
    atomic_bytes(root/source_copy, payload)
    record = {"schema": "astra_memory.human_revision.v1", "revision": revision, "actor": actor.strip(),
        "source": "explicit_cli_human_submission", "imported_at": now(), "source_copy": source_copy,
        "source_file_sha256": sha(root/source_copy), "submission_sha256": digest(doc), "submission": doc,
        "previous_revision_sha256": pointer.get("latest_revision_sha256")}
    name = f"revisions/{revision:06d}.json"
    save(root/name, record)
    revision_hash = sha(root/name)
    pointer["revisions"].append({"path": name, "sha256": revision_hash})
    pointer.update(latest_revision_sha256=revision_hash, episode_id=config["episode_id"])
    save(pointer_path, pointer)
    return {"status": "imported", "revision": revision, "revision_sha256": revision_hash,
        "record_path": str(root/name), "new_model_requests": 0, "next_step": "export"}
