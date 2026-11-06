"""S2: verified S1 -> temporal graph, category knowledge and stable search places."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .contracts import KNOWLEDGE_SCHEMA, UNIT_SCHEMA_V2, validate_knowledge, validate_observation, validate_units
from .io import atomic_bytes, digest, load, read_rows, require, safe_file, save, save_rows, sha
from .pipeline import frozen_contract, implementation_hash
from .prompts import KNOWLEDGE_VERSION, ROLE_C, ROLE_D, UNIT_PROMPT_VERSION
from .provider import Provider
from .registry import Registry
from .s2_export import project_scene, validate_projection
from .s2_state import TemporalMemory
from .s2_units import UnitRegistry


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_artifacts(root, artifacts):
    for relative, expected in artifacts.items():
        path = safe_file(root, relative)
        require(path.is_file() and sha(path) == expected, f"artifact missing or changed: {relative}")


def read_s1(config, source, allow_partial=False):
    source = Path(source).resolve()
    require((source/"run_manifest.json").is_file(), "S1 has no run manifest; complete S1 before S2")
    manifest = load(source/"run_manifest.json")
    status = manifest["stages"]["S1"]
    require(status == "complete" or (status == "partial" and allow_partial),
            "S1 is not complete; a validated prefix requires explicit --allow-partial")
    require(manifest["episode_id"] == config["episode_id"] and manifest["model"] == config["model"]["name"], "S1 episode/model mismatch")
    require(manifest["final_contract"] == frozen_contract(), "S1 final contract mismatch")
    verify_artifacts(source, manifest["artifact_sha256"])
    prepared = load(source/"input/manifest.json")
    verify_artifacts(source/"input", prepared["artifact_sha256"])
    frames = read_rows(source/"input/frames.jsonl")
    count = manifest["processed_frame_count"]
    require(isinstance(count, int) and 0 < count <= len(frames), "S1 has no valid observation prefix")
    require(count == manifest["scope"]["requested_prefix_count"], "S1 prefix scope/count mismatch")
    require(len(frames) == manifest["scope"]["source_bag_frame_count"] == prepared["bag_frame_count"], "S1 source frame counts differ")
    require((status == "complete") == (count == len(frames)), "S1 completeness differs from actual coverage")
    require(manifest["scope"]["is_partial_prefix"] == (count < len(frames)), "S1 prefix flag differs from actual coverage")
    evidence = read_rows(source/"s1/evidence_index.jsonl")
    require(len(evidence) == count, "S1 evidence index count mismatch")
    records = []
    registry = Registry(config["episode_id"], config["association"]["minimum_bbox_iou"])
    bindings = {"run_manifest_sha256": sha(source/"run_manifest.json"),
        "input_manifest_sha256": sha(source/"input/manifest.json"), "source_root": str(source)}
    for order, (frame, ev) in enumerate(zip(frames[:count], evidence)):
        name = ev["observation_record"]
        require(name in manifest["artifact_sha256"], "S1 observation is not hash-bound")
        record = load(safe_file(source, name))
        ref, index = frame["frame_id"], frame["original_video_frame_index"]
        require(record["image_ref"] == ev["image_ref"] == ref, "S1 evidence/record/frame mismatch")
        require(record["original_video_frame_index"] == ev["original_video_frame_index"] == index, "S1 original frame mismatch")
        require(record["original_video_seconds"] == ev["original_video_seconds"] == frame["original_video_seconds"] == index/config["fps"], "S1 original time/FPS mismatch")
        require(ev["image_path"] == "input/"+frame["rgb_path"] and ev["image_sha256"] == frame["rgb_sha256"], "S1 image binding mismatch")
        require(sha(safe_file(source, ev["image_path"])) == ev["image_sha256"], "S1 RGB evidence changed")
        require(record["actual_model"] == config["model"]["name"], "S1 observation model mismatch")
        # Verify model provenance rather than accepting a hand-authored intermediate as a live observation.
        cache = source/"cache"/record["cache_key"]
        require(cache.resolve().is_relative_to((source/"cache").resolve()), "unsafe S1 cache path")
        ledger = load(cache/"ledger.json")
        require(ledger["status"] == "validated" and ledger["actual_model"] == config["model"]["name"], "S1 request not validated")
        require(ledger["signature_hash"] == record["cache_key"] and
                digest(load(cache/"request_descriptor.json")) == record["cache_key"], "S1 request signature differs")
        verify_artifacts(cache, ledger["artifacts"])
        require(load(cache/"validated.json") == record["observation"], "S1 observation differs from validated model output")
        past_refs = [f["frame_id"] for f in frames[max(0, order-config["model"]["past_images"]):order]]
        candidates = registry.candidates(config["model"]["max_candidate_objects"])
        validate_observation(record["observation"], ref, [*past_refs, ref], past_refs, {c["id"] for c in candidates})
        expected = registry.ingest(record["observation"], frame, order, record["geometry_by_local_id"])
        require(expected == record["association"], "S1 identity association cannot be replayed with configured policy")
        require(ev["model_request_ref"] == "cache/"+record["cache_key"]+"/ledger.json", "S1 request index mismatch")
        records.append(record)
    stored, replayed = load(source/"s1/registry.json"), registry.snapshot()
    # The old descriptive note predates downstream B; it is not identity state.
    require({k: v for k, v in stored.items() if k != "note"} ==
            {k: v for k, v in replayed.items() if k != "note"}, "S1 registry differs from observation replay")
    bindings["evidence_index_sha256"] = sha(source/"s1/evidence_index.jsonl")
    return records, evidence, frames, manifest, bindings


def unit_context(memory, record, evidence_by_ref, registry):
    # Current frame only; no future exemplars or template IDs are supplied to D.
    current = record["image_ref"]
    mapping = record["association"]["local_to_instance"]
    objects = [{"id": mapping[o["local_id"]], "name": o["name"], "category": o["category"],
                "appearance": o["appearance"], "state_description": o["state_description"],
                "epistemic_status": o["epistemic_status"]} for o in record["observation"]["objects"]]
    entity_evidence = {o["id"]: [current] for o in objects if o["epistemic_status"] == "observed"}
    existing = registry.candidates(entity_evidence)
    context = {"role": "D", "prompt_version": UNIT_PROMPT_VERSION,
        "current_image_ref": current, "scene_description": record["observation"]["scene_description"],
        "entities": objects, "current_relations": [row for row in memory.evidence
            if row["image_ref"] == current and row["accepted"] and row["epistemic_status"] == "observed"],
        "local_to_instance": mapping, "existing_unit_registry": list(existing.values()),
        "state_sha256": digest(memory.object_summary()), "task": "propose_stable_inspection_places_no_target_judgement"}
    spatial_facts = [(row["subject_id"], row["predicate"], row["target_id"])
                     for row in context["current_relations"]]
    return context, entity_evidence, existing, spatial_facts


def run_s2(config, source=None, allow_partial=False, through_frame=None, provider=None, progress=print):
    output = Path(config["output"])
    root = output/"s2"
    root.mkdir(parents=True, exist_ok=True)
    pointer = root/"manifest.json"
    # Failed/incomplete S1 is a dependency failure, not a successful empty memory.
    try:
        records, evidence, frames, s1_manifest, bindings = read_s1(config, source or output, allow_partial)
    except Exception as exc:
        save(root/"last_attempt.json", {"schema": "astra_memory.s2_attempt.v1", "status": "needs_input",
            "error_type": type(exc).__name__, "detail": str(exc), "at": now(), "final_episode_generated": False})
        raise
    if through_frame is not None:
        require(isinstance(through_frame, int) and through_frame >= 0, "invalid original through-frame")
        records = [r for r in records if r["original_video_frame_index"] <= through_frame]
    require(records, "no S1 observations in requested S2 scope")
    used_refs = {r["image_ref"] for r in records}
    evidence = [e for e in evidence if e["image_ref"] in used_refs]
    code_hash = implementation_hash()
    budgets = config.get("s2", {"knowledge_max_output_tokens": 2048, "unit_max_output_tokens": 4096})
    role_models = {role: {**config["model"], "max_output_tokens": min(config["model"]["max_output_tokens"], budgets[key])}
                   for role, key in [("C", "knowledge_max_output_tokens"), ("D", "unit_max_output_tokens")]}
    signature = digest({"stage": "S2.v1", "source": bindings, "records": records,
        "role_models": role_models, "implementation": code_hash, "episode": config["episode_id"],
        "knowledge_version": KNOWLEDGE_VERSION, "unit_prompt_version": UNIT_PROMPT_VERSION,
        "unit_trigger_policy": "first_or_new_entity_or_explicit_reveal_open_close_v1",
        "final_contract": frozen_contract()})
    relative = "builds/"+signature
    build = root/relative
    build.mkdir(parents=True, exist_ok=True)
    manifest_path = build/"manifest.json"
    if manifest_path.exists():
        old = load(manifest_path)
        if old["status"] in {"complete", "partial"}:
            verify_artifacts(build, old["artifact_sha256"])
            save(pointer, old)
            return {**old, "reused_completed_build": True}
    knowledge_provider = provider or Provider(role_models["C"], output.parent/"_category_knowledge_cache", code_hash)
    unit_provider = provider or Provider(role_models["D"], output/"cache_s2", code_hash)
    providers = {id(p): p for p in [knowledge_provider, unit_provider]}
    initial_attempts = sum(p.new_transport_attempts for p in providers.values())
    partial = len(records) < len(frames)
    manifest = {"schema": "astra_memory.s2_manifest.v1", "stage": "S2", "signature": signature,
        "build_path": relative, "status": "running", "episode_id": config["episode_id"],
        "source_bindings": bindings, "source_S1_status": s1_manifest["stages"]["S1"],
        "implementation_sha256": code_hash, "final_contract": frozen_contract(), "started_at": now(),
        "role_models": role_models, "knowledge_cache_root": str(knowledge_provider.root.resolve()),
        "unit_cache_root": str(unit_provider.root.resolve()),
        "scope": {"source_bag_frame_count": len(frames), "processed_frame_count": len(records),
            "is_partial": partial, "through_original_video_frame_index": records[-1]["original_video_frame_index"],
            "through_time_s": records[-1]["original_video_seconds"]},
        "new_transport_attempts": 0, "cache_hits": 0, "category_requests": 0, "unit_requests": 0,
        "S3": "pending", "final_export": "pending", "final_episode_generated": False}
    save(manifest_path, manifest)
    save(root/"last_attempt.json", manifest)
    memory, units, knowledge = TemporalMemory(config["episode_id"]), UnitRegistry(config["episode_id"]), {}
    evidence_by_ref = {e["image_ref"]: e for e in evidence}
    d_requests = []
    try:
        for order, record in enumerate(records):
            new_ids = memory.ingest(record)
            for category in sorted({o["category"] for o in record["observation"]["objects"]}):
                if category in knowledge:
                    continue
                progress(f"C category={category}")
                result = knowledge_provider.observe(ROLE_C, KNOWLEDGE_SCHEMA,
                    {"role": "C", "category": category, "knowledge_version": KNOWLEDGE_VERSION}, [],
                    lambda doc, category=category: validate_knowledge(doc, category))
                knowledge[category] = {"category": category, "document": result.document,
                    "knowledge_version": KNOWLEDGE_VERSION, "epistemic_scope": "general_category_knowledge_not_instance_fact",
                    "available_at": memory.frames[-1], "request_ref": str((knowledge_provider.root/result.cache_key/"ledger.json").resolve()),
                    "actual_model": result.actual_model, "usage": result.usage}
                manifest["category_requests"] += 1
                manifest["cache_hits"] += int(result.cache_hit)
                save(build/"knowledge.partial.json", knowledge)
            changes = {c["kind"] for c in record["observation"]["changes"] if c["epistemic_status"] != "unknown"}
            triggered = order == 0 or bool(new_ids) or bool(changes & {"opened", "closed", "revealed", "hidden"})
            if triggered and record["observation"]["objects"]:
                progress(f"D {order+1}/{len(records)} original_frame={record['original_video_frame_index']}")
                context, entity_evidence, existing, spatial_facts = unit_context(memory, record, evidence_by_ref, units)
                ev = evidence_by_ref[record["image_ref"]]
                images = [{"image_ref": record["image_ref"], "path": str(safe_file(bindings["source_root"], ev["image_path"])),
                           "sha256": ev["image_sha256"]}]
                result = unit_provider.observe(ROLE_D, UNIT_SCHEMA_V2, context, images,
                    lambda doc: validate_units(doc, record["image_ref"], entity_evidence, existing, spatial_facts))
                request_ref = str((unit_provider.root/result.cache_key/"ledger.json").resolve())
                units.ingest(result.document, memory.frames[-1], request_ref)
                d_requests.append({"image_ref": record["image_ref"], "document": result.document,
                    "request_ref": request_ref, "actual_model": result.actual_model, "usage": result.usage,
                    "trigger": "initial" if order == 0 else "new_entities_or_explicit_boundary_visibility_change"})
                manifest["unit_requests"] += 1
                manifest["cache_hits"] += int(result.cache_hit)
            manifest["new_transport_attempts"] = sum(p.new_transport_attempts for p in providers.values())-initial_attempts
            manifest["completed_observation_count"] = order+1
            save(manifest_path, manifest)
        scene, catalog = project_scene(memory, knowledge), units.export()
        validation = validate_projection(scene, catalog, config["episode_id"])
        # Raw source-relative evidence stays in S1; S2 keeps one compressed relation table plus an index.
        documents = {"scene_memory.json": scene, "search_units.json": catalog,
            "objects.json": memory.object_summary(),
            "relations.json": {"schema": "astra_memory.relation_ledger.v1", "relations": list(memory.relations.values()),
                "semantics": "one_canonical_fact_with_discrete_support; inferred_not_used_as_location"},
            "knowledge.json": {"schema": "astra_memory.knowledge_catalog.v1", "categories": knowledge},
            "unit_registry.json": units.snapshot(), "unit_requests.json": d_requests,
            "validation.json": validation}
        for filename, doc in documents.items():
            save(build/filename, doc)
        save_rows(build/"relation_evidence.jsonl", memory.evidence)
        save_rows(build/"evidence_index.jsonl", evidence)
        save_rows(build/"conflicts.jsonl", [*memory.conflicts, *units.conflicts])
        artifact_names = [*documents, "relation_evidence.jsonl", "evidence_index.jsonl", "conflicts.jsonl"]
        manifest.update(status="partial" if partial else "complete", finished_at=now(),
            artifact_sha256={name: sha(build/name) for name in artifact_names},
            category_count=len(knowledge), unit_count=len(catalog["list"]), relation_count=len(memory.relations),
            object_count=len(memory.objects), unresolved_count=len(memory.conflicts)+len(units.conflicts))
        require(sha(Path(bindings["source_root"])/"run_manifest.json") == bindings["run_manifest_sha256"],
                "S1 changed during S2 processing; rebuild with a stable source")
        save(manifest_path, manifest)
        save(pointer, manifest)
        save(root/"last_attempt.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", finished_at=now(), new_transport_attempts=sum(p.new_transport_attempts for p in providers.values())-initial_attempts,
            failure={"type": type(exc).__name__, "detail": str(exc)[:1500]})
        save(manifest_path, manifest)
        save(root/"last_attempt.json", manifest)
        raise


def query_s2(config, at_frame):
    """Read-only historical projection; no model calls and no future unit revisions."""
    root = Path(config["output"])/"s2"
    manifest = load(root/"manifest.json")
    require(manifest["status"] in {"complete", "partial"}, "S2 has no committed build")
    require(manifest["episode_id"] == config["episode_id"], "S2 query episode mismatch")
    require(isinstance(at_frame, int) and not isinstance(at_frame, bool) and at_frame >= 0, "invalid query frame")
    require(at_frame <= manifest["scope"]["through_original_video_frame_index"], "query exceeds processed S2 coverage")
    build = safe_file(root, manifest["build_path"])
    verify_artifacts(build, manifest["artifact_sha256"])
    records, _, _, _, bindings = read_s1(config, manifest["source_bindings"]["source_root"], True)
    require(bindings == manifest["source_bindings"], "S1 source changed since S2 build; rebuild S2")
    memory = TemporalMemory(config["episode_id"])
    for record in records:
        if record["original_video_frame_index"] <= at_frame:
            memory.ingest(record)
    require(memory.frames, "query precedes first observed S1 frame")
    knowledge = load(build/"knowledge.json")["categories"]
    units = UnitRegistry(config["episode_id"])
    for row in load(build/"unit_requests.json"):
        frame = next((f for f in memory.frames if f["image_ref"] == row["image_ref"]), None)
        if frame is not None:
            units.ingest(row["document"], frame, row["request_ref"])
    scene, catalog = project_scene(memory, knowledge), units.export()
    validation = validate_projection(scene, catalog, config["episode_id"])
    return {"schema": "astra_memory.s2_query.v1", "requested_original_video_frame_index": at_frame,
        "last_supported_frame": memory.frames[-1], "scene_memory": scene, "search_units": catalog,
        "validation": validation, "new_model_requests": 0, "future_evidence_used": False}
