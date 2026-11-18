"""Real-video integration probe. Reuses explicit upstream caches, never fake model outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spatial_memory.candidates import prepare_event
from spatial_memory.catalog import initialize
from spatial_memory.io import read_json, sha256, write_json
from spatial_memory.memory import replay
from spatial_memory.selection import select


def jsonl(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def probe(recording, args, model, key):
    source = Path(args.pnr_root) / recording
    output = Path(args.output).resolve() / recording
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(source / "detection/manifest.json")
    session = manifest["source"]["session"]
    result = {"recording": recording, "session": session, "actual_model": model,
              "upstream_detection_actions_events": "reused_cached_predictions",
              "full_numeric_pipeline_completed": False}
    start = time.perf_counter()
    try:
        stage1 = (Path(args.stage1_root).resolve() / recording / "stage1"
                  if args.stage1_root else output / "stage1")
        result["stage1_source"] = str(stage1)
        result["stage1_inference_this_run"] = not bool(args.stage1_root)
        if not args.stage1_root:
            with (output / "stage1.log").open("w") as log:
                subprocess.run([sys.executable, "-m", "online_annotation", "initialize",
                            "--session", session, "--output", str(stage1),
                            "--candidate-mode", "brief", "--model-root", args.model_root,
                            "--base-url", args.base_url], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        catalog = read_json(stage1 / "catalog.v0.json")
        write_json(output / "initial_memory.json", initialize(catalog))
        result["stage1_regions"] = len(catalog["regions"])
        predictions = read_json(source / "events/predictions.json")["predictions"]
        # Selection is determined before running this probe, without inspecting its scores.
        candidates = [p for p in predictions if p["end_seconds"] - p["start_seconds"] >= 1.0]
        prediction = candidates[0] if candidates else predictions[0]
        eid = prediction["event_id"]
        index = jsonl(source / "detection/frame_index.jsonl")
        actions = jsonl(source / "actions/actions.jsonl")
        spec = {"episode_id": recording, "event_id": eid,
                "left_video": manifest["source"]["video"],
                "start_source_frame_id": prediction["start_source_frame_id"],
                "end_source_frame_id": prediction["end_source_frame_id"],
                "frame_index": [{"source_frame_id": r["source_frame_id"],
                                 "timestamp_s": r["relative_seconds"]} for r in index],
                "action_rows": [{"source_frame_id": r["source_frame_id"],
                                 "action_label": r["action_label"],
                                 "confidence": (r.get("top_verbs") or [{}])[0].get("probability")}
                                for r in actions],
                "event_detection_provenance": {"prediction": prediction,
                    "predictions_sha256": sha256(source / "events/predictions.json")}}
        packet_file = source / "triplets" / eid / "candidates_roi/packets.jsonl"
        if packet_file.exists() and jsonl(packet_file):
            packet = jsonl(packet_file)[0]
            target = packet["target"]
            spec["contact_cluster_start_source_frame_id"] = target["start_source_frame_id"]
            spec["contact_cluster_end_source_frame_id"] = target["end_source_frame_id"]
            spec["important_frames"] = [{"source_frame_id": r["source_frame_id"],
                                         "tag": r["reason"]} for r in packet["frames"]]
            result["cached_target_quality"] = {k: target.get(k) for k in
                ("object_category", "target_extent_review_required", "valid_fraction")}
        write_json(output / "event_spec.json", spec)
        event = prepare_event(spec, output / "candidates")
        selection = select(event, catalog, output / "selection", args.base_url,
                           model=model, api_key=key, image_max_edge=args.image_max_edge)
        result.update({"event_id": eid, "candidate_count": len(event["candidates"]),
                       "selection_status": selection["status"],
                       "frames": {role: selection[role] for role in ("pre", "pnr", "post")},
                       "association_count": len(selection["associations"])})
        if selection["status"] == "ok":
            # The selected frame IDs have no depth/mask adapter in this repository yet.
            # Explicit unknown evidence exercises replay; it is not numeric completion.
            assessments = [{"episode_id": recording, "event_id": eid,
                            "search_unit_id": a["candidate_id"],
                            "score_type": "geometric_coverage", "scope": "unestablished",
                            "status": "uncomputable", "q": None,
                            "reason": "selected_frame_depth_masks_and_reference_not_provided"}
                           for a in selection["associations"]]
            committed = {"episode_id": recording, "event_id": eid,
                         "catalog_version": catalog["catalog_version"],
                         "event_end_time_s": event["event_end_time_s"],
                         "available_after_s": max(c["timestamp_s"] for c in event["candidates"]),
                         "availability_note": "offline replay cutoff; excludes inference wall time",
                         "selection": selection, "assessments": assessments}
            write_json(output / "committed_event.json", committed)
            state = replay(catalog, [committed])
            write_json(output / "memory.json", state)
            result["memory_event_count"] = len(state["applied_events"])
            result["geometry_status"] = "missing_selected_frame_geometry"
        else:
            result["geometry_status"] = "blocked_by_uncertain_selection"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["seconds"] = round(time.perf_counter() - start, 3)
    write_json(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnr-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage1-root", help="Reuse a completed stage-one run, keeping its original receipts")
    parser.add_argument("--image-max-edge", type=int, default=768)
    args = parser.parse_args()
    config = read_json(Path(args.model_root) / "config.json")
    key_file = Path(args.model_root) / config["api_key_file"]
    key = next(s.strip() for s in key_file.read_text().splitlines()
               if s.strip() and not s.lstrip().startswith("#"))
    results = [probe(recording, args, config["model_alias"], key) for recording in args.recordings]
    write_json(Path(args.output) / "summary.json", results)


if __name__ == "__main__":
    main()
