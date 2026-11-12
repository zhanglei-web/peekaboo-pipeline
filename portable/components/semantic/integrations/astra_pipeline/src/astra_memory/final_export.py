"""Deterministic adjudication and publication of the nine-field frozen document."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .confirmations import read_confirmation
from .final_validation import validate_contract
from .io import digest, load, require, safe_file, save, sha
from .pipeline import frozen_contract, implementation_hash
from .s2 import now, verify_artifacts
from .s3 import read_s3


class NeedsInputError(ValueError):
    pass


def assemble(config, manifest, build, confirmation):
    scene = load(build/"scene_memory.json")
    units = load(build/"search_units.json")
    event_states = load(build/"search_event_states.json")
    comparisons = {e["event_id"]: e["document"] for e in load(build/"event_comparisons.json")}
    human = confirmation["record"]["submission"] if confirmation else None
    human_rows = {e["event_id"]: e for e in human["events"]} if human else {}
    inspections, missing, claims = [], [], []
    for event in event_states["events"]:
        eid = event["event_id"]
        row = human_rows.get(eid)
        links = row["unit_ids"] if row and row["unit_ids"] is not None else comparisons[eid]["unit_link_proposals"]
        if not links:
            missing.append(eid)
        result = row["result"] if row else "unknown"
        source = "awaiting_explicit_human_confirmation"
        link_basis = "Astra_E_visual_unit_proposal; not_human_confirmed"
        if row and result != "unknown":
            source = f'user_confirmation:revision={confirmation["record"]["revision"]}; actor={confirmation["record"]["actor"]}; note={row["note"]}'
            claims.append(eid)
        if row and row["unit_ids"] is not None:
            link_basis = "user_confirmation:explicit_unit_link; "+row["note"]
            claims.append(eid)
        if row and row["final_location"] and row["final_location"]["unit_id"] not in links:
            raise NeedsInputError(f"confirmed final location is outside event units: {eid}; submit explicit unit_ids")
        inspections.append({"event_id": eid, "order": event["ordinal"], "unit_ids": links,
            "granularity": "single_unit_event" if len(links) == 1 else "multi_unit_event",
            "source_region_text": event["search_region"], "unit_link_basis": link_basis,
            "action_summary": event["event_summary"], "result": result, "result_source": source,
            "sample_window_s": event["supplied_image_interval_seconds"]})
    if missing:
        raise NeedsInputError("no supported inspection unit link for: "+", ".join(missing)+"; use confirm-template and confirm-import with explicit unit_ids")
    full_coverage = bool(human and human["full_search_coverage_confirmed"] is True)
    found = [row for row in inspections if row["result"] == "found"]
    winner = found[0] if found else None
    confirmed_ids = {r["confirmed_target_instance_id"] for r in human_rows.values() if r["confirmed_target_instance_id"]}
    confirmed_target = next(iter(confirmed_ids)) if confirmed_ids else None
    result = "found" if winner else "not_found" if full_coverage and inspections and all(i["result"] == "not_found" for i in inspections) else "unknown"
    outcome = {"result": result, "success_event_id": winner["event_id"] if winner else None,
        "success_event_order": winner["order"] if winner else None,
        "target_instance_id": confirmed_target if winner else None, "final_location": None}
    if winner:
        loc = human_rows[winner["event_id"]]["final_location"]
        if loc:
            state = next(s for e in event_states["events"] for s in e["states"] if s["state_id"] == loc["evidence_state_id"])
            outcome["final_location"] = {**loc, "sample_time_s": state["original_video_seconds"],
                "location_granularity": "search_unit", "source": "user_confirmation:explicit_final_location",
                "note": "人工确认的发现位置；采样时间来自对应文件名，不推断连续发现时刻。"}
    only_final = bool(full_coverage and len(found) == 1 and found[0]["order"] == len(inspections)
                      and all(i["result"] == "not_found" for i in inspections[:-1]))
    sampling = (f'从{manifest["scope"]["source_bag_frame_count"]}个bag关键帧中自适应选取'
                f'{manifest["scope"]["processed_frame_count"]}帧' if config["selection"]["mode"] == "adaptive_keyframes_v1"
                else f'全段{manifest["scope"]["processed_frame_count"]}个bag采样')
    metadata = {"episode_id": config["episode_id"], "scene_id": config["scene_id"],
        "zone_id": config["episode_id"]+":zone:tool_workspace", "time_axis": "original_video_seconds", "fps": config["fps"],
        "search_scope": human["search_scope"] if human and human["search_scope"] else units["scope"],
        "search_scope_source": "user_confirmation:explicit_search_scope" if human and human["search_scope"] else "observed_local_workspace_not_full_room",
        "full_search_coverage_confirmed": full_coverage, "only_final_event_found_target_confirmed": only_final,
        "target": {**config["target"], "confirmed_instance_id": confirmed_target,
            "identity_source": "user_confirmation:explicit_instance_reference" if confirmed_target else "unconfirmed_task_text_only"},
        "provenance": f'S1{sampling}与S3全部事件图观测；单一{config["model"]["name"]}；build={manifest["signature"]}；manual_confirmation_only。'}
    if full_coverage or only_final or confirmed_target or (human and human["search_scope"]):
        claims.append("metadata")
    doc = {"schema": "spatial_memory.search_episode.p0_p1_2.v1",
        "README": {"purpose": "以固定结构保存空间拓扑、对象记忆、三帧查找状态及可审计的结果裁决。",
            "scope": [f'{sampling}与{event_states["event_count"]}次事件的{event_states["image_count"]}张图片。',
                "输入采样覆盖不等于完整搜索覆盖，房间几何、隐藏内容与物体物理唯一性不由结构验证认证。"],
            "rules": ["空间拓扑使用节点和关系，文字描述具体状态，帧号和哈希用于证据定位。",
                "类别常识与实例观测分开；未看到不等于不在那里；明确移动使此前位置失效。",
                "单帧状态只使用当时及过去证据；三帧比较为回顾摘要，不改写因果快照。",
                "人工确认必须显式导入；未知结果保持unknown，不默认前面失败或最后成功。",
                "重复事实汇总，离散证据、位置区间、身份复核及来源账本独立保存。"]},
        "metadata": metadata, "scene_memory": scene, "search_units": units, "search_event_states": event_states,
        "inspections": inspections, "episode_outcome": outcome,
        "derived_statistics": {"counting_unit": "original_search_event", "search_event_count": len(inspections),
            "found_event_count": len(found), "search_unit_catalog_count": len(units["list"]),
            "events_before_success": winner["order"]-1 if winner else None,
            "events_through_success": winner["order"] if winner else None,
            "note": "按原始事件计数；一次检查多个地点仍为一次；成功顺序采用最早已人工确认成功的事件，不推断未采样过程。"}}
    attestation = {"schema": "astra_memory.output_attestation.v1", "episode_id": config["episode_id"],
        "output_content_sha256": digest(doc), "confirmed_claim_refs": sorted(set(claims)),
        "confirmation_revision_sha256": confirmation["revision_sha256"] if confirmation else None,
        "s3_signature": manifest["signature"], "source": "program_bound_to_explicit_human_revision" if confirmation else "no_human_claims"}
    return doc, attestation


def validate_final(doc, attestation, verify_images=True):
    report = validate_contract(doc, verify_images=verify_images, confirmation_ledger=attestation)
    expected_result = "found" if any(i["result"] == "found" for i in doc["inspections"]) else "not_found" if (
        doc["metadata"]["full_search_coverage_confirmed"] and doc["inspections"] and
        all(i["result"] == "not_found" for i in doc["inspections"])) else "unknown"
    require(doc["episode_outcome"]["result"] == expected_result, "outcome differs from manual confirmation policy")
    found = [i for i in doc["inspections"] if i["result"] == "found"]
    if found:
        require(doc["episode_outcome"]["success_event_id"] == found[0]["event_id"], "success must be earliest confirmed found event")
    for inspection in doc["inspections"]:
        require(inspection["result"] == "unknown" or inspection["result_source"].startswith("user_confirmation:"),
                "nonunknown result lacks human provenance")
    if attestation["confirmed_claim_refs"]:
        require(attestation["confirmation_revision_sha256"], "confirmation claims need an explicit imported revision")
    report.update(end_to_end_pipeline_executed=True, success_policy="manual_confirmation_only",
        semantic_accuracy="not_certified_by_contract_validation")
    return report


def export_episode(config, progress=print):
    root = Path(config["output"])/"final"
    root.mkdir(parents=True, exist_ok=True)
    attempt = {"stage": "final_export", "status": "running", "started_at": now(), "final_episode_generated": False}
    try:
        manifest, source = read_s3(config)
        confirmation = read_confirmation(config, manifest, source)
        signature = digest({"stage": "final_export.v1", "s3_manifest_sha256": sha(Path(config["output"])/"s3/manifest.json"),
            "confirmation": confirmation["revision_sha256"] if confirmation else None,
            "scene_id": config["scene_id"], "implementation": implementation_hash(), "contract": frozen_contract()})
        doc, attestation = assemble(config, manifest, source, confirmation)
        validation = validate_final(doc, attestation)
        build = root/"builds"/signature
        build.mkdir(parents=True, exist_ok=True)
        name = "search_episode_"+config["episode_id"]+".json"
        docs = {name: doc, "confirmation_attestation.json": attestation, "validation.json": validation}
        for filename, value in docs.items():
            save(build/filename, value)
        attempt.update(schema="astra_memory.final_manifest.v1", signature=signature, status="complete", finished_at=now(),
            build_path="builds/"+signature, episode_id=config["episode_id"], s3_signature=manifest["signature"],
            confirmation_revision_sha256=confirmation["revision_sha256"] if confirmation else None,
            source_s3_manifest_sha256=sha(Path(config["output"])/"s3/manifest.json"),
            artifact_sha256={n: sha(build/n) for n in docs}, final_episode_generated=True,
            output_file=str(Path(config["output"])/name), new_model_requests=0)
        save(build/"manifest.json", attempt)
        save(root/"manifest.json", attempt)
        save(Path(config["output"])/name, doc)
        save(root/"last_attempt.json", attempt)
        progress(f"export {attempt['output_file']}")
        return attempt
    except Exception as exc:
        attempt.update(status="needs_input" if isinstance(exc, NeedsInputError) else "failed", finished_at=now(),
            failure={"type": type(exc).__name__, "detail": str(exc)[:1500]})
        save(root/"last_attempt.json", attempt)
        raise


def validate_export(config):
    root = Path(config["output"])/"final"
    pointer = load(root/"manifest.json")
    require(pointer["status"] == "complete" and pointer["episode_id"] == config["episode_id"], "no complete export for episode")
    build = safe_file(root, pointer["build_path"])
    verify_artifacts(build, pointer["artifact_sha256"])
    s3, source = read_s3(config)
    require(pointer["source_s3_manifest_sha256"] == sha(Path(config["output"])/"s3/manifest.json"), "export is stale after S3 change")
    confirmation = read_confirmation(config, s3, source)
    require(pointer["confirmation_revision_sha256"] == (confirmation["revision_sha256"] if confirmation else None), "export is stale after human revision; run export")
    expected, attestation = assemble(config, s3, source, confirmation)
    filename = "search_episode_"+config["episode_id"]+".json"
    actual = load(build/filename)
    require(actual == expected and load(build/"confirmation_attestation.json") == attestation, "export differs from source evidence/decisions")
    require(sha(Path(config["output"])/filename) == sha(build/filename), "published JSON differs from validated build")
    return validate_final(actual, attestation)
