"""Triplet observations, bounded identity review and retrospective event evidence.

S1 IDs are immutable. Event-only instances have a separate namespace. Model B
reviews remain proposals, never aliases or human confirmations.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .contracts import (OBSERVATION_SCHEMA, KNOWLEDGE_SCHEMA, UNIT_SCHEMA_V2, REVIEW_SCHEMA,
    EVENT_SCHEMA, validate_observation, validate_knowledge, validate_units, validate_review, validate_event)
from .inputs import triplets
from .io import digest, load, require, safe_file, save, save_rows, sha
from .pipeline import frozen_contract, implementation_hash
from .prompts import (ROLE_A, PROMPT_VERSION, ROLE_B, REVIEW_VERSION, ROLE_C, KNOWLEDGE_VERSION,
    ROLE_D, ROLE_E, EVENT_VERSION)
from .provider import Provider
from .registry import Registry
from .s2 import now, read_s1, verify_artifacts, unit_context
from .s2_export import project_scene, validate_projection
from .s2_state import TemporalMemory
from .s2_units import UnitRegistry


def runtime_configuration(config):
    return {"model": config["model"], "association": config["association"],
        "s2": {"knowledge_max_output_tokens": 2048, "unit_max_output_tokens": 4096, **config.get("s2", {})},
        "s3": {"max_identity_reviews": 32, "review_max_output_tokens": 2048, "event_max_output_tokens": 4096, **config.get("s3", {})}}


def read_s2(config, allow_partial=False):
    root = Path(config["output"])/"s2"
    require((root/"manifest.json").is_file(), "S2 has no committed build")
    manifest = load(root/"manifest.json")
    require(manifest["episode_id"] == config["episode_id"], "S2 episode mismatch")
    require(manifest["final_contract"] == frozen_contract(), "S2 contract mismatch")
    require(manifest["status"] == "complete" or (allow_partial and manifest["status"] == "partial"),
            "S3 requires complete S2; partial source is preview only")
    for model in manifest["role_models"].values():
        for field in ["name", "base_url", "transport", "reasoning_effort"]:
            require(model[field] == config["model"][field], "S2 model service differs from current configuration")
    build = safe_file(root, manifest["build_path"])
    verify_artifacts(build, manifest["artifact_sha256"])
    records, evidence, frames, _, bindings = read_s1(config, manifest["source_bindings"]["source_root"], allow_partial)
    require(bindings == manifest["source_bindings"], "S1 changed since S2; rebuild S2")
    records = records[:manifest["scope"]["processed_frame_count"]]
    require(records and records[-1]["original_video_frame_index"] == manifest["scope"]["through_original_video_frame_index"],
            "S2 scope mismatch")
    return manifest, build, records, evidence, bindings


def stage_bindings(config, manifest):
    return {"s2_manifest_sha256": sha(Path(config["output"])/"s2/manifest.json"),
            "s2_signature": manifest["signature"], "source_bindings": manifest["source_bindings"]}


def read_s3(config, allow_partial=False):
    root = Path(config["output"])/"s3"
    require((root/"manifest.json").is_file(), "S3 has no committed build")
    manifest = load(root/"manifest.json")
    require(manifest["episode_id"] == config["episode_id"] and manifest["final_contract"] == frozen_contract(), "S3 episode/contract mismatch")
    require(manifest["status"] == "complete" or (allow_partial and manifest["status"] == "partial"), "production export requires complete S3")
    require(manifest["target"] == config["target"] and manifest["fps"] == config["fps"], "S3 target/FPS changed; rebuild S3")
    require(manifest["runtime_configuration_sha256"] == digest(runtime_configuration(config)), "S3 runtime configuration changed; rebuild S3")
    s2, _, _, _, _ = read_s2(config, allow_partial)
    require(stage_bindings(config, s2) == manifest["source_bindings"], "S2 changed since S3; rebuild S3")
    require(digest(triplets(config["event_images"], config["episode_id"], config["fps"])) == manifest["event_index_sha256"],
            "event images/index changed since S3")
    build = safe_file(root, manifest["build_path"])
    verify_artifacts(build, manifest["artifact_sha256"])
    for request in load(build/"model_requests.json"):
        ledger_path = Path(request["ledger_path"])
        require(sha(ledger_path) == request["ledger_sha256"], "S3 request ledger changed")
        ledger = load(ledger_path)
        require(ledger["status"] == "validated" and ledger["actual_model"] == config["model"]["name"], "S3 request not validated by configured model")
        verify_artifacts(ledger_path.parent, ledger["artifacts"])
    return manifest, build


def event_candidates(memory, maximum):
    objects = sorted(memory.objects.values(), key=lambda o: (o["last_seen"]["time_s"], o["id"]), reverse=True)
    return [{"id": o["id"], "name": o["name"], "category": o["category"],
        "appearance": o["appearance_latest"], "last_image_ref": o["last_seen"]["image_ref"],
        "last_bbox": o["last_bbox"], "last_state_description": o["latest_state_description"],
        "identity_status": o["identity_status"]} for o in objects[:maximum]]


def associate_event(config, doc, frame, candidates, previous_ref):
    # Reuse exactly the S1 continuity gates, while assigning disjoint deterministic IDs.
    registry = Registry(config["episode_id"]+":event:"+digest(doc["image_ref"])[:12],
                        config["association"]["minimum_bbox_iou"])
    for order, candidate in enumerate(candidates):
        appearance = {"color": None, "shape": None, "material_appearance": None,
                      "size_description": None, "distinguishing_marks": [], **candidate["appearance"]}
        registry.instances[candidate["id"]] = {**deepcopy(candidate), "appearance": appearance,
            "last_seen_order": order, "observation_count": 0}
    registry.previous_image_ref = previous_ref
    return registry.ingest(doc, frame, len(candidates), {o["local_id"]: None for o in doc["objects"]})


def state_projection(state, record):
    doc = record["observation"]
    interactions = [r["description"] for r in doc["interactions"] if r["epistemic_status"] == "observed"]
    partial = [o["name"]+"仅部分可见" for o in doc["objects"] if o["visibility"] == "partially_visible"]
    visibility = partial+doc["uncertainties"]
    return {**state, "state_id": record["image_ref"], "frame_label": "f"+str(state["original_video_frame_index"]),
        "phase_label": {"start": "开始", "middle": "进行中", "end": "结束"}[state["phase"]],
        "state_description": doc["scene_description"]+" "+" ".join(o["state_description"] for o in doc["objects"]),
        "hand_action": "；".join(interactions) or "当前采样图未能确认手部与物体的交互。",
        "focus_object_states": [{"entity_ref": record["association"]["local_to_instance"][o["local_id"]],
            "observed_state": o["state_description"]} for o in doc["objects"]],
        "visibility_and_occlusion": "；".join(visibility) or "已记录当前图中可辨识对象；画面外与被遮挡区域的内容未知。",
        "evidence_status": "validated_Astra_A_current_image_observation; identities_provisional; not_human_confirmed"}


def run_s3(config, allow_partial=False, provider=None, progress=print):
    output, code = Path(config["output"]), implementation_hash()
    root = output/"s3"
    root.mkdir(parents=True, exist_ok=True)
    try:
        s2, s2_build, bag_records, bag_evidence, s1_bindings = read_s2(config, allow_partial)
        event_index = triplets(config["event_images"], config["episode_id"], config["fps"])
        require(float(config["fps"]).is_integer(), "frozen event contract requires an integer video FPS")
        # Ordinals must be chronological; cross-event overlap is allowed and processed by real frame time.
        require([e["states"][0]["original_video_frame_index"] for e in event_index["events"]] ==
                sorted(e["states"][0]["original_video_frame_index"] for e in event_index["events"]), "event ordinals are not chronological")
    except Exception as exc:
        save(root/"last_attempt.json", {"status": "needs_input", "at": now(), "detail": str(exc), "final_episode_generated": False})
        raise
    options = {"max_identity_reviews": 32, "review_max_output_tokens": 2048, "event_max_output_tokens": 4096,
               **config.get("s3", {})}
    bindings = stage_bindings(config, s2)
    signature = digest({"stage": "S3.v1", "bindings": bindings, "events": event_index, "model": config["model"],
        "target": config["target"], "association": config["association"], "options": options, "code": code})
    relative = "builds/"+signature
    build = root/relative
    build.mkdir(parents=True, exist_ok=True)
    if (build/"manifest.json").is_file():
        old = load(build/"manifest.json")
        if old["status"] in {"complete", "partial"}:
            verify_artifacts(build, old["artifact_sha256"])
            save(root/"manifest.json", old)
            read_s3(config, allow_partial)
            return {**old, "reused_completed_build": True}
    budgets = config.get("s2", {"knowledge_max_output_tokens": 2048, "unit_max_output_tokens": 4096})
    models = {role: {**config["model"], "max_output_tokens": min(config["model"]["max_output_tokens"], tokens)}
        for role, tokens in [("A", config["model"]["max_output_tokens"]), ("B", options["review_max_output_tokens"]),
            ("C", budgets["knowledge_max_output_tokens"]), ("D", budgets["unit_max_output_tokens"]), ("E", options["event_max_output_tokens"])]}
    providers = {role: provider or Provider(model, output.parent/"_category_knowledge_cache" if role == "C" else output/"cache_s3", code)
                 for role, model in models.items()}
    unique_providers = {id(p): p for p in providers.values()}.values()
    start_attempts = sum(p.new_transport_attempts for p in unique_providers)
    manifest = {"schema": "astra_memory.s3_manifest.v1", "stage": "S3", "signature": signature, "build_path": relative,
        "episode_id": config["episode_id"], "target": config["target"], "fps": config["fps"], "status": "running",
        "final_contract": frozen_contract(), "source_bindings": bindings, "event_index_sha256": digest(event_index),
        "runtime_configuration_sha256": digest(runtime_configuration(config)),
        "scope": deepcopy(s2["scope"]), "role_models": models, "started_at": now(), "cache_hits": 0,
        "new_transport_attempts": 0, "role_requests": {r: 0 for r in providers}, "final_episode_generated": False}
    save(build/"manifest.json", manifest)
    save(root/"last_attempt.json", manifest)
    requests = []

    def call(role, prompt, schema, context, images, validator):
        result = providers[role].observe(prompt, schema, context, images, validator)
        ledger = (providers[role].root/result.cache_key/"ledger.json").resolve()
        requests.append({"role": role, "ledger_path": str(ledger), "ledger_sha256": sha(ledger)})
        manifest["role_requests"][role] += 1
        manifest["cache_hits"] += int(result.cache_hit)
        manifest["new_transport_attempts"] = sum(p.new_transport_attempts for p in unique_providers)-start_attempts
        save(build/"manifest.json", manifest)
        return result.document, str(ledger)

    try:
        memory, units = TemporalMemory(config["episode_id"]), UnitRegistry(config["episode_id"])
        knowledge = load(s2_build/"knowledge.json")["categories"]
        unit_requests = {row["image_ref"]: row for row in load(s2_build/"unit_requests.json")}
        image_index = {e["image_ref"]: {"image_ref": e["image_ref"],
            "path": str(safe_file(s1_bindings["source_root"], e["image_path"])), "sha256": e["image_sha256"]} for e in bag_evidence}
        states_by_ref, event_meta, state_records = {}, {}, {}
        for event in event_index["events"]:
            for state in event["states"]:
                ref = event["event_id"]+":"+state["phase"]
                states_by_ref[ref] = state
                event_meta[ref] = event
                image_index[ref] = {"image_ref": ref, "path": state["image_path"], "sha256": state["image_sha256"]}
        timeline = [(r["original_video_frame_index"], 0, r["image_ref"], r) for r in bag_records]
        timeline += [(s["original_video_frame_index"], 1, ref, None) for ref, s in states_by_ref.items()]
        timeline.sort(key=lambda row: row[:3])
        records, snapshots, reviews, pending = [], {}, [], []
        reviewed_pairs = set()
        d_records = []

        def review_conflicts(record):
            for decision in record["association"]["decisions"]:
                for other in decision["proposed_candidates"]:
                    oid = decision["instance_id"]
                    pair = tuple(sorted([oid, other]))
                    if oid == other or pair in reviewed_pairs or other not in memory.objects:
                        continue
                    reviewed_pairs.add(pair)
                    if len(reviews) >= options["max_identity_reviews"]:
                        pending.append({"candidate_refs": list(pair), "available_at": memory.frames[-1], "reason": "review_budget_exhausted; no_merge"})
                        continue
                    # Representative evidence is drawn only from records available at this point.
                    refs = []
                    for candidate in pair:
                        ref = next(r["image_ref"] for r in reversed(records)
                                   if candidate in r["association"]["local_to_instance"].values())
                        if ref not in refs:
                            refs.append(ref)
                    witness = next((r for r in reversed(records) if set(pair) <= set(r["association"]["local_to_instance"].values())), None)
                    co_visible = witness is not None
                    if witness and witness["image_ref"] not in refs:
                        refs.append(witness["image_ref"])
                    question = "identity:"+digest(pair)[:20]
                    context = {"role": "B", "prompt_version": REVIEW_VERSION, "question_id": question,
                        "candidate_refs": list(pair), "co_visible_in_supplied_history": co_visible,
                        "candidates": [{k: memory.objects[oid][k] for k in ["id", "name", "category", "appearance_latest",
                            "first_seen", "last_seen", "last_bbox", "latest_state_description"]} for oid in pair],
                        "co_visible_witness_image_ref": witness["image_ref"] if witness else None,
                        "available_at": memory.frames[-1], "decision_scope": "proposal_only_no_alias_or_human_confirmation"}
                    progress(f"B {question}")
                    doc, request = call("B", ROLE_B, REVIEW_SCHEMA, context, [image_index[r] for r in refs],
                        lambda doc: validate_review(doc, question, pair, refs, co_visible))
                    reviews.append({"available_at": deepcopy(memory.frames[-1]), "document": doc, "request_ref": request,
                                    "applied": False, "policy": "retain_identity_candidates_without_merging"})

        for number, (_, _, ref, original_record) in enumerate(timeline):
            if original_record is not None:
                record = deepcopy(original_record)
            else:
                state = states_by_ref[ref]
                frame = {"original_video_frame_index": state["original_video_frame_index"],
                         "original_video_seconds": state["original_video_seconds"]}
                past = records[-config["model"]["past_images"]:] if config["model"]["past_images"] else []
                past_refs = [r["image_ref"] for r in past]
                candidates = event_candidates(memory, config["model"]["max_candidate_objects"])
                context = {"role": "A", "prompt_version": PROMPT_VERSION, "current_image": {"image_ref": ref},
                    "past_images": [{"image_ref": r} for r in past_refs], "past_instance_candidates": candidates,
                    "registry_state_sha256": digest(memory.object_summary()), "scope": "causal_event_image_no_target_or_future"}
                progress(f"A event image={ref} original_frame={frame['original_video_frame_index']}")
                doc, request = call("A", ROLE_A, OBSERVATION_SCHEMA, context,
                    [image_index[r] for r in [*past_refs, ref]],
                    lambda doc: validate_observation(doc, ref, [*past_refs, ref], past_refs, {c["id"] for c in candidates}))
                association = associate_event(config, doc, frame, candidates, records[-1]["image_ref"] if records else None)
                record = {"schema": "astra_memory.observation_record.v1", "image_ref": ref, **frame,
                    "observation": doc, "association": association, "geometry_by_local_id": {o["local_id"]: None for o in doc["objects"]},
                    "request_ref": request, "actual_model": config["model"]["name"]}
                state_records[ref] = record
            new_ids = memory.ingest(record)
            records.append(record)
            for category in sorted({o["category"] for o in record["observation"]["objects"]}):
                if category not in knowledge:
                    progress(f"C event category={category}")
                    doc, request = call("C", ROLE_C, KNOWLEDGE_SCHEMA,
                        {"role": "C", "category": category, "knowledge_version": KNOWLEDGE_VERSION}, [],
                        lambda doc, category=category: validate_knowledge(doc, category))
                    knowledge[category] = {"category": category, "document": doc, "request_ref": request,
                        "knowledge_version": KNOWLEDGE_VERSION, "epistemic_scope": "general_category_knowledge_not_instance_fact"}
            if ref in unit_requests:
                row = unit_requests[ref]
                units.ingest(row["document"], memory.frames[-1], row["request_ref"])
            elif ref in states_by_ref and record["observation"]["objects"]:
                context, entity_evidence, existing, spatial = unit_context(memory, record, image_index, units)
                progress(f"D event image={ref}")
                doc, request = call("D", ROLE_D, UNIT_SCHEMA_V2, context, [image_index[ref]],
                    lambda doc: validate_units(doc, ref, entity_evidence, existing, spatial))
                units.ingest(doc, memory.frames[-1], request)
                d_records.append({"image_ref": ref, "document": doc, "request_ref": request})
            review_conflicts(record)
            if ref in states_by_ref:
                snapshots[ref] = {"available_at": deepcopy(memory.frames[-1]),
                    "state": state_projection(states_by_ref[ref], record),
                    "scene_memory": project_scene(memory, knowledge), "search_units": units.export(),
                    "evidence_refs_available": [r["image_ref"] for r in records], "future_evidence_used": False}
            manifest["completed_observation_count"] = number+1
            save(build/"manifest.json", manifest)

        events, comparisons, catalog = [], [], {}
        for event in event_index["events"]:
            eid = event["event_id"]
            refs = [eid+":"+phase for phase in ["start", "middle", "end"]]
            current_units = snapshots[refs[-1]]["search_units"]["list"]
            unit_ids = {u["unit_id"] for u in current_units}
            context = {"role": "E", "prompt_version": EVENT_VERSION, "event_ref": eid,
                "target_category_definition": config["target"], "instance_reference_supplied": False,
                "states": [snapshots[r]["state"] for r in refs], "available_search_units": current_units,
                "scope": "retrospective_triplet_only; cannot_rewrite_causal_observations_or_confirm_success"}
            progress(f"E event={eid}")
            comparison, request = call("E", ROLE_E, EVENT_SCHEMA, context, [image_index[r] for r in refs],
                lambda doc: validate_event(doc, eid, unit_ids, refs))
            comparisons.append({"event_id": eid, "document": comparison, "request_ref": request,
                "available_at_original_video_frame_index": event["states"][-1]["original_video_frame_index"]})
            for ref in refs:
                record = state_records[ref]
                for obj in record["observation"]["objects"]:
                    oid = record["association"]["local_to_instance"][obj["local_id"]]
                    catalog[oid] = {"label": obj["name"], "memory_entity_candidates": [oid],
                                    "link_status": "direct_observation_registry_reference; physical_identity_provisional"}
            events.append({"event_id": eid, "ordinal": event["ordinal"], "source_human_interval_id": None,
                "previous_json_interval_seconds": [], "supplied_image_interval_seconds": event["sample_window_s"],
                "interval_alignment": {"status": "previous_interval_unavailable", "basis_for_this_record": "supplied_triplet_filenames",
                    "note": "未导入旧JSON的事件边界；空数组表示未知，不表示旧边界与三帧相等。"},
                "phase_frame_indices": {s["phase"]: s["original_video_frame_index"] for s in event["states"]},
                "search_region": comparison["search_region_proposal"], "search_region_status": "Astra_E_visual_proposal_not_human_confirmed",
                "event_summary": comparison["event_summary"], "observed_changes": comparison["observed_changes"],
                "outcome": {"target_found": None, "target_identity_confirmed": False,
                    "reason": "三帧视觉候选不裁决成功；人工确认结果独立记录在inspections与episode_outcome。"},
                "states": [snapshots[r]["state"] for r in refs]})
        section = {"schema": "ego.search_event_triplet_states.v1", "image_directory": event_index["directory"],
            "event_count": len(events), "image_count": event_index["image_count"], "entity_reference_catalog": catalog,
            "frame_numbering": {"original_video_fps": int(config["fps"]), "original_video_frame_index_base": 0,
                "source": "supplied_image_filename", "fps_source": "run_config_verified_against_S1_frame_times",
                "time_formula": "original_video_frame_index / fps",
                "namespace_note": "文件名f数字是原视频零基帧号；bag关键帧ID与事件状态ID是独立索引。"},
            "provenance": {"description_source": "Astra_A_causal_observation_and_E_retrospective_comparison",
                "knowledge_scope": "每帧只使用当时及过去证据；事件总结仅比较本次三帧；视觉不等于人工确认。",
                "coverage": "仅描述已提供的离散三帧，不补造中间动作。"}, "events": events}
        scene, places = project_scene(memory, knowledge), units.export()
        validation = validate_projection(scene, places, config["episode_id"])
        docs = {"scene_memory.json": scene, "search_units.json": places, "search_event_states.json": section,
            "event_comparisons.json": comparisons, "causal_snapshots.json": snapshots, "identity_reviews.json": reviews,
            "pending_identity_reviews.json": pending, "knowledge.json": knowledge, "unit_registry.json": units.snapshot(),
            "unit_requests.json": d_records, "objects.json": memory.object_summary(), "validation.json": validation,
            "model_requests.json": requests, "event_image_index.json": event_index,
            "unit_link_todo.json": [{"event_id": row["event_id"], "unit_ids": [], "reason": "no_visual_unit_link; explicit_human_link_required"}
                for row in comparisons if not row["document"]["unit_link_proposals"]]}
        for name, doc in docs.items():
            save(build/name, doc)
        save_rows(build/"observations.jsonl", records)
        save_rows(build/"relation_evidence.jsonl", memory.evidence)
        save_rows(build/"conflicts.jsonl", [*memory.conflicts, *units.conflicts])
        names = [*docs, "observations.jsonl", "relation_evidence.jsonl", "conflicts.jsonl"]
        require(stage_bindings(config, load(output/"s2/manifest.json")) == bindings, "S2 changed during S3")
        require(digest(triplets(config["event_images"], config["episode_id"], config["fps"])) == digest(event_index), "event images changed during S3")
        manifest.update(status="partial" if s2["scope"]["is_partial"] else "complete", finished_at=now(),
            artifact_sha256={name: sha(build/name) for name in names}, event_count=len(events), image_count=event_index["image_count"],
            object_count=len(memory.objects), unit_count=len(places["list"]), identity_review_count=len(reviews),
            pending_identity_review_count=len(pending), unresolved_unit_link_count=len(docs["unit_link_todo.json"]))
        save(build/"manifest.json", manifest)
        save(root/"manifest.json", manifest)
        save(root/"last_attempt.json", manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", finished_at=now(), new_transport_attempts=sum(p.new_transport_attempts for p in unique_providers)-start_attempts,
            failure={"type": type(exc).__name__, "detail": str(exc)[:1500]})
        save(build/"manifest.json", manifest)
        save(root/"last_attempt.json", manifest)
        raise
