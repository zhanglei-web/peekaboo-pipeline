"""S0/input/S1 orchestration. No placeholder final episode is emitted."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .contracts import OBSERVATION_SCHEMA, validate_observation
from .extract import bbox_geometry, prepare
from .inputs import audit_bag, confirmation_todo, triplets
from .io import digest, load, require, save, save_rows, sha
from .prompts import PROMPT_VERSION, ROLE_A
from .provider import Provider
from .registry import Registry


def implementation_hash():
    root = Path(__file__).parent
    return digest({str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
                   if p.is_file() and p.suffix in {".py", ".json"}})


def frozen_contract():
    root = Path(__file__).parent/"resources"
    lock = load(root/"contract.lock.json")
    require(sha(root/"search_episode.schema.json") == lock["schema_sha256"], "frozen final Schema changed")
    return {"contract_id": lock["contract_id"], "schema_sha256": lock["schema_sha256"],
            "reference_sha256": lock["reference_sha256"]}


def check(config):
    contract = frozen_contract()
    audit = audit_bag(config["source_bag"])
    events = triplets(config["event_images"], config["episode_id"], config["fps"])
    return audit, events, {"schema": "astra_memory.preflight.v1", "episode_id": config["episode_id"],
        "contract": contract, "source_bag": audit["source_bag"], "topic_counts": audit["topic_counts"],
        "search_event_count": events["event_count"], "event_image_count": events["image_count"],
        "success_policy": config["success_policy"], "source_files_modified": False,
        "geometry_status": audit["geometry_status"], "model": config["model"]["name"],
        "unimplemented_stages": [], "pipeline_boundary": "existing_ROS2_bag_to_frozen_search_episode_JSON"}


def prepare_inputs(config):
    audit, events, report = check(config)
    frames = prepare(config, audit, events, implementation_hash())
    save(Path(config["output"])/"preflight.json", report)
    todo = Path(config["output"])/"confirmation_todo.json"
    if not todo.exists():
        save(todo, confirmation_todo(config, events))
    return frames


def run_s1(config, frames, maximum=None, provider=None, progress=print):
    require(maximum is None or maximum > 0, "max-frames must be positive")
    output = Path(config["output"])
    input_root = output/"input"
    chosen = frames if maximum is None else frames[:maximum]
    require(chosen, "no selected frames")
    code_hash = implementation_hash()
    run_signature = digest({"config": config, "implementation": code_hash,
        "input_manifest_sha256": sha(input_root/"manifest.json"), "contract": frozen_contract()})
    manifest_path = output/"run_manifest.json"
    if manifest_path.exists():
        previous = load(manifest_path)
        require(previous["run_signature"] == run_signature, "run changed; use a new output directory")
    manifest = {"schema": "astra_memory.run_manifest.v1", "run_signature": run_signature,
        "episode_id": config["episode_id"], "model": config["model"]["name"],
        "code_sha256": code_hash, "config_sha256": digest(config), "final_contract": frozen_contract(),
        "scope": {"mode": config["selection"]["mode"], "source_bag_frame_count": len(frames),
            "requested_prefix_count": len(chosen), "is_partial_prefix": len(chosen) < len(frames),
            "event_images_observed_in_S1": False},
        "stages": {"input": "complete", "S1": "running", "B_ambiguity_review": "pending",
                   "S2": "pending", "S3": "pending", "final_export": "pending"},
        "started_at": datetime.now(timezone.utc).isoformat(), "processed_frame_count": 0,
        "new_transport_attempts": 0, "cache_hits": 0,
        "final_episode_generated": False, "success_policy": "manual_confirmation_only"}
    save(manifest_path, manifest)
    provider = provider or Provider(config["model"], output/"cache", code_hash)
    attempts_at_start = provider.new_transport_attempts
    registry = Registry(config["episode_id"], config["association"]["minimum_bbox_iou"])
    evidence, conflicts, outputs = [], [], {}
    records_root = output/"s1"/"frames"
    records_root.mkdir(parents=True, exist_ok=True)
    try:
        for order, frame in enumerate(chosen):
            past = chosen[max(0, order-config["model"]["past_images"]):order]
            past_refs = [f["frame_id"] for f in past]
            current_ref = frame["frame_id"]
            candidates = registry.candidates(config["model"]["max_candidate_objects"])
            context = {"role": "A", "prompt_version": PROMPT_VERSION,
                "current_image": {"image_ref": current_ref},
                "past_images": [{"image_ref": f["frame_id"]} for f in past],
                "past_instance_candidates": candidates,
                "registry_state_sha256": digest(registry.snapshot()),
                "scope": "causal_current_image_only; no_task_target_or_event_outcome"}
            images = [{"image_ref": f["frame_id"], "path": str(input_root/f["rgb_path"]),
                "sha256": f["rgb_sha256"]} for f in [*past, frame]]
            candidate_ids = {c["id"] for c in candidates}

            def validate(doc):
                return validate_observation(doc, current_ref, [*past_refs, current_ref], past_refs, candidate_ids)

            progress(f"A {order+1}/{len(chosen)} original_frame={frame['original_video_frame_index']}")
            result = provider.observe(ROLE_A, OBSERVATION_SCHEMA, context, images, validate)
            geometry = {o["local_id"]: bbox_geometry(frame, o["bbox"], input_root)
                        for o in result.document["objects"]}
            association = registry.ingest(result.document, frame, order, geometry)
            name = f"s1/frames/{order:06d}.json"
            save(output/name, {"schema": "astra_memory.observation_record.v1",
                "image_ref": current_ref, "original_video_frame_index": frame["original_video_frame_index"],
                "original_video_seconds": frame["original_video_seconds"],
                "observation": result.document, "association": association,
                "geometry_by_local_id": geometry, "cache_key": result.cache_key,
                "actual_model": result.actual_model, "usage": result.usage})
            outputs[name] = sha(output/name)
            evidence.append({"image_ref": current_ref, "bag_keyframe_index": frame["bag_keyframe_index"],
                "original_video_frame_index": frame["original_video_frame_index"],
                "original_video_seconds": frame["original_video_seconds"], "sensor_time_ns": frame["sensor_time_ns"],
                "image_path": "input/"+frame["rgb_path"], "image_sha256": frame["rgb_sha256"],
                "observation_record": name, "model_request_ref": "cache/"+result.cache_key+"/ledger.json"})
            for decision in association["decisions"]:
                if decision["proposed_candidates"] and decision["reason"] != "model_proposal_and_adjacent_frame_continuity_gate":
                    conflicts.append({"image_ref": current_ref, **decision, "review_status": "pending_B_or_manual_review"})
            save_rows(output/"s1"/"evidence_index.jsonl", evidence)
            save_rows(output/"s1"/"identity_conflicts.jsonl", conflicts)
            save(output/"s1"/"registry.json", registry.snapshot())
            manifest["processed_frame_count"] += 1
            manifest["new_transport_attempts"] = provider.new_transport_attempts-attempts_at_start
            manifest["cache_hits"] += int(result.cache_hit)
            save(manifest_path, manifest)
        for name in ["s1/evidence_index.jsonl", "s1/identity_conflicts.jsonl", "s1/registry.json"]:
            outputs[name] = sha(output/name)
        manifest["stages"]["S1"] = "partial" if len(chosen) < len(frames) else "complete"
        manifest.update(artifact_sha256=outputs, provisional_instance_count=len(registry.instances),
            pending_identity_review_count=len(conflicts), finished_at=datetime.now(timezone.utc).isoformat())
        save(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["stages"]["S1"] = "failed"
        manifest["new_transport_attempts"] = provider.new_transport_attempts-attempts_at_start
        manifest["failure"] = {"type": type(exc).__name__, "detail": str(exc)[:1000]}
        save(manifest_path, manifest)
        raise
