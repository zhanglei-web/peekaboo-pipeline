"""Whole-run orchestration and honest status without inference on inspection."""
from pathlib import Path

from .io import load, read_rows, require, save
from .pipeline import prepare_inputs, run_s1
from .s2 import now, read_s1, run_s2
from .s3 import read_s2, read_s3, run_s3
from .final_export import export_episode, validate_export


def run_pipeline(config, s1_source=None, progress=print):
    output = Path(config["output"])
    manifest = {"schema": "astra_memory.pipeline_manifest.v1", "episode_id": config["episode_id"],
        "started_at": now(), "status": "running", "stages": {}, "final_episode_generated": False}
    path = output/"pipeline_manifest.json"
    save(path, manifest)
    stage = "S1"
    try:
        source = Path(s1_source).resolve() if s1_source else output
        if (source/"run_manifest.json").is_file() and load(source/"run_manifest.json")["stages"]["S1"] == "complete":
            read_s1(config, source)
            manifest["stages"][stage] = "verified_existing_complete_run"
        else:
            require(s1_source is None, "explicit S1 source must already be complete")
            frames = prepare_inputs(config)
            run_s1(config, frames, progress=progress)
            manifest["stages"][stage] = "complete"
        save(path, manifest)
        stage = "S2"
        run_s2(config, source=source, progress=progress)
        manifest["stages"][stage] = "complete"
        save(path, manifest)
        stage = "S3"
        run_s3(config, progress=progress)
        manifest["stages"][stage] = "complete"
        save(path, manifest)
        stage = "final_export"
        result = export_episode(config, progress=progress)
        manifest["stages"][stage] = "complete"
        manifest.update(status="complete", final_episode_generated=True, output_file=result["output_file"], finished_at=now())
        save(path, manifest)
        return manifest
    except Exception as exc:
        from .final_export import NeedsInputError
        manifest["stages"][stage] = "needs_input" if isinstance(exc, NeedsInputError) else "failed"
        manifest.update(status=manifest["stages"][stage], finished_at=now(), failure={"stage": stage, "type": type(exc).__name__, "detail": str(exc)[:1500]})
        save(path, manifest)
        raise


def status(config):
    output = Path(config["output"])
    s2_pointer = output/"s2/manifest.json"
    if not s2_pointer.is_file() and (output/"s2/last_attempt.json").is_file():
        attempt = load(output/"s2/last_attempt.json")
        if "source_bindings" in attempt:
            s2_pointer = output/"s2/last_attempt.json"
    s1_source = load(s2_pointer)["source_bindings"]["source_root"] if s2_pointer.is_file() else output
    stages = {}
    for stage, inspect in [("S1", lambda: read_s1(config, s1_source, True)),
                           ("S2", lambda: read_s2(config, True)), ("S3", lambda: read_s3(config, True)),
                           ("final_export", lambda: validate_export(config))]:
        try:
            result = inspect()
            if stage == "S1":
                value = result[3]["stages"]["S1"]
            elif stage in {"S2", "S3"}:
                value = result[0]["status"]
            else:
                value = "complete"
            stages[stage] = {"status": value, "source_and_artifact_validation": "passed"}
        except Exception as exc:
            stages[stage] = {"status": "unavailable_or_stale", "detail": str(exc)[:1000]}
        name = "final" if stage == "final_export" else stage.lower()
        attempt = output/name/"last_attempt.json"
        if attempt.is_file():
            stages[stage]["last_attempt"] = load(attempt)
    pipeline = load(output/"pipeline_manifest.json") if (output/"pipeline_manifest.json").exists() else None
    return {"schema": "astra_memory.pipeline_status.v1", "episode_id": config["episode_id"], "stages": stages,
        "pipeline_last_attempt": pipeline, "final_episode_valid_for_current_inputs": stages["final_export"]["status"] == "complete",
        "new_model_requests": 0}


def query_s3(config, at_frame):
    from .s2_export import project_scene, validate_projection
    from .s2_state import TemporalMemory
    from .s2_units import UnitRegistry
    manifest, build = read_s3(config, True)
    require(isinstance(at_frame, int) and not isinstance(at_frame, bool) and at_frame >= 0, "invalid original frame")
    records = read_rows(build/"observations.jsonl")
    require(at_frame <= max(r["original_video_frame_index"] for r in records), "query exceeds processed coverage")
    memory = TemporalMemory(config["episode_id"])
    for record in records:
        if record["original_video_frame_index"] <= at_frame:
            memory.ingest(record)
    require(memory.frames, "query precedes first observed image")
    units = UnitRegistry(config["episode_id"])
    registry = load(build/"unit_registry.json")
    units.units = {u["unit_id"]: u for u in registry["units"]}
    units.conflicts = registry["unresolved_proposals"]
    scene = project_scene(memory, load(build/"knowledge.json"))
    catalog = units.export(at_frame/config["fps"])
    validation = validate_projection(scene, catalog, config["episode_id"])
    return {"schema": "astra_memory.s3_query.v1", "requested_original_video_frame_index": at_frame,
        "scene_memory": scene, "search_units": catalog, "validation": validation, "new_model_requests": 0,
        "future_evidence_used": False, "last_supported_frame": memory.frames[-1]}
