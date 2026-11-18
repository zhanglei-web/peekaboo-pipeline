"""Process every freshly predicted event and maintain one chronological recording memory."""
import argparse
from collections import Counter
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_memory.candidates import prepare_event
from spatial_memory.catalog import initialize
from spatial_memory.geometry_compiler import GeometryCompiler
from spatial_memory.io import read_json, write_json
from spatial_memory.memory import replay
from spatial_memory.selection import MODEL, select


def jsonl(path):
    import json
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--recording-root", required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    out = Path(args.recording_root).resolve()
    session = Path(args.session).resolve()
    catalog = read_json(out / "stage1/catalog.v0.json")
    state = initialize(catalog)
    state["available_after_s"] = 2.0 + read_json(out / "stage1/performance.json")["catalog_ready_seconds"]
    write_json(out / "initial_memory.json", state)
    geometry_config = {**config["geometry"], "score_policy": "best_effort"}
    compiler = GeometryCompiler(geometry_config, config["model_root"], config["base_url"])
    model_config = read_json(Path(config["model_root"]) / "config.json")
    if model_config["model_alias"] != MODEL:
        raise ValueError("Full test requires the actual requested 8B model")
    key = next(s.strip() for s in (Path(config["model_root"]) / model_config["api_key_file"]).read_text().splitlines()
               if s.strip() and not s.lstrip().startswith("#"))
    predictions = read_json(out / "events/predictions.json")["predictions"]
    records, registry, stages = [], [], []
    start = time.perf_counter()

    def stage(name, command, folder):
        log = folder / f"{name}.log"
        tick = time.perf_counter()
        with log.open("w") as handle:
            result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        row = {"stage": name, "event_id": folder.name, "exit_code": result.returncode,
               "seconds": time.perf_counter() - tick, "log": str(log)}
        stages.append(row)
        write_json(out / "event_stage_trace.json", stages)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")

    for index, prediction in enumerate(predictions):
        eid = prediction["event_id"]
        folder = out / "event_runs" / eid
        folder.mkdir(parents=True, exist_ok=True)
        tick = time.perf_counter()
        frontend = [config["frontend_python"], "-m", "event_frontend"]
        stage("scope", frontend + ["clip-event", "--run", str(out / "detection"), "--output", str(folder / "clip"),
              "--start-frame", str(prediction["start_source_frame_id"]),
              "--end-frame", str(prediction["end_source_frame_id"]), "--clip-id", eid], folder)
        stage("objects", frontend + ["recognize-objects", "--clip", str(folder / "clip"),
              "--weights", config["yolo_weights"], "--output", str(folder / "objects")], folder)
        stage("clusters", frontend + ["prepare-contact-cluster", "--clip", str(folder / "clip"),
              "--objects", str(folder / "objects"), "--output", str(folder / "clusters")], folder)
        stage("rank", frontend + ["rank-candidates", "--clusters", str(folder / "clusters"),
              "--clip", str(folder / "clip"), "--clip-checkpoint", config["clip_checkpoint"],
              "--output", str(folder / "rank")], folder)
        spec = read_json(out / "event_specs" / f"{eid}.json")
        packets = jsonl(folder / "rank/packets.jsonl")
        if packets:
            target = packets[0]["target"]
            spec["contact_cluster_start_source_frame_id"] = target["start_source_frame_id"]
            spec["contact_cluster_end_source_frame_id"] = target["end_source_frame_id"]
            spec["important_frames"] = [{"source_frame_id": f["source_frame_id"], "tag": f["reason"]}
                                        for f in packets[0]["frames"]]
        write_json(folder / "event_spec.json", spec)
        feedback = None
        for attempt in range(3):
            if attempt:
                spec["context_seconds"] = 1.0 + attempt * .5
            event = prepare_event(spec, folder / f"candidates_{attempt + 1}")
            try:
                selection = select(event, catalog, folder / f"selection_{attempt + 1}", config["base_url"],
                                   api_key=key, model=MODEL, feedback=feedback)
                break
            except (ValueError, RuntimeError) as exc:
                feedback = str(exc)
        else:
            raise RuntimeError(f"8B could not select three actual frames for {eid}: {feedback}")
        write_json(folder / "final_candidates.json", event)
        write_json(folder / "final_selection.json", selection)
        state, registry, record = compiler.compile(session, event, selection, catalog, state,
            folder / "scoring", registry, out / "detection/detections.jsonl")
        records.append(record)
        write_json(out / "memory.json", state)
        write_json(out / "reference_registry.json", registry)
        write_json(out / "record_progress.json", {"status": "running", "completed_events": index + 1,
            "predicted_events": len(predictions), "last_event": eid, "q_event": record["q_event_estimate"],
            "score_types": [a["score_type"] for a in record["assessments"]],
            "event_seconds": time.perf_counter() - tick, "elapsed_seconds": time.perf_counter() - start})
        print({"recording": session.name, "event": eid, "q_event": record["q_event_estimate"],
               "completed": index + 1, "total": len(predictions)}, flush=True)
    replayed = replay(catalog, records)
    if [r["event_id"] for r in records] != [p["event_id"] for p in predictions]:
        raise ValueError("Some predicted events were not committed")
    if any(replayed["units"][uid]["Q"] != u["Q"] for uid, u in state["units"].items()):
        raise ValueError("Independent event replay differs from incremental memory")
    if any(u["Q"] is None for u in state["units"].values()):
        raise ValueError("Operational Q must be numeric in best-effort mode")
    counts = Counter(a["estimation_method"] for r in records for a in r["assessments"])
    summary = {"recording": session.name, "model": MODEL, "score_policy": "best_effort",
        "predicted_events": len(predictions), "committed_events": len(records),
        "events_with_numeric_score": sum(isinstance(r.get("q_event_estimate"), (int, float)) for r in records),
        "score_method_counts": dict(counts), "final_unit_count": len(state["units"]),
        "memory_replay_matches": True, "initial_catalog_regions": len(catalog["regions"]),
        "event_scores": [{"event_id": r["event_id"], "q_event": r["q_event_estimate"],
            "assessments": [{"q": a["q"], "type": a["score_type"], "method": a["estimation_method"]}
                            for a in r["assessments"]]} for r in records],
        "seconds": time.perf_counter() - start}
    write_json(out / "recording_summary.json", summary)
    write_json(out / "record_progress.json", {"status": "complete", **summary})


if __name__ == "__main__":
    main()
