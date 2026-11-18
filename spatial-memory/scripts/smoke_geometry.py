"""Exercise geometry packaging/scoring/replay with explicitly imported cached triplets.

This is a component test; it does not claim fresh video event detection or Qwen selection.
"""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_memory.geometry import score_job
from spatial_memory.geometry_inputs import prepare_geometry
from spatial_memory.io import read_json, write_json
from spatial_memory.memory import replay


def probe(folder, output):
    source = read_json(folder / "scoring_input.json")
    previous = read_json(folder / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(folder / "geometry.npz", allow_pickle=False) as data:
        support = data["support_mask"]
    cv2.imwrite(str(output / "support_mask.png"), support.astype("uint8") * 255)
    episode = source["episode_id"]
    uid = episode + ":local_test_footprint"
    eid = source["event_id"]
    selection = {"status": "ok", "event_id": eid, "episode_id": episode,
                 "catalog_version": 0, "source": "imported_cached_triplet_component_test",
                 "associations": [{"candidate_id": uid, "reason": "imported component scope",
                                   "confidence": "high"}]}
    frames = {}
    for phase, role in zip(("start", "middle", "end"), ("pre", "pnr", "post")):
        fid = next(f["frame_index"] for f in source["frames"] if f["phase"] == phase)
        selection[role] = fid
        frames[role] = {"source_frame_id": fid,
                        "left_image": str(folder / f"{phase}_left.png"),
                        "depth_npz": str(folder / f"{phase}_depth.npz"),
                        "foreground_mask": str(folder / f"{phase}_dynamic_mask.png"),
                        "dynamic_mask": str(folder / f"{phase}_dynamic_mask.png")}
    reference = {**frames["pre"], "object_mask": str(folder / "start_object_mask.png"),
                 "support_mask": str(output / "support_mask.png")}
    quality = previous["quality"]
    config = {"K": source["K"], "search_unit_id": uid,
              "scope": "original_support_footprint", "reference": reference,
              "frames": frames,
              "quality_audit": {"accepted": quality["accepted_for_local_diagnostic"],
                                 "reasons": quality["reasons"],
                                 "source": str(folder / "result.json")}}
    write_json(output / "sources.json", config)
    write_json(output / "imported_selection.json", selection)
    job = prepare_geometry(config, selection, output / "inputs")
    result = score_job(job, selection, output / "score")
    catalog = {"schema_version": "online.candidate_catalog.v0.1", "episode_id": episode,
               "catalog_version": 0, "mode": "imported_geometry_component_scope",
               "parents": [{"parent_id": "component_parent"}],
               "regions": [{"candidate_id": uid, "parent_id": "component_parent",
                            "name": "local original footprint component scope",
                            "region_type": "occluder_footprint",
                            "identity_anchor": "original_footprint",
                            "boundary_description": "Fixed projected object support footprint only"}]}
    committed = {"episode_id": episode, "event_id": eid, "catalog_version": 0,
                 "event_end_time_s": max(f["time_s"] for f in source["frames"]),
                 "available_after_s": max(f["time_s"] for f in source["frames"]),
                 "selection": selection, "assessments": [result]}
    memory = replay(catalog, [committed])
    write_json(output / "memory.json", memory)
    summary = {"source_event": folder.name, "episode_id": episode,
               "test_kind": "cached_geometry_component_only",
               "geometry_status": result["status"], "q": result["q"],
               "raw_q": result.get("raw_q"), "reason": result.get("reason"),
               "upstream_quality_reasons": quality["reasons"],
               "memory_Q": memory["units"][uid]["Q"],
               "memory_events": len(memory["applied_events"])}
    write_json(output / "result.json", summary)
    print(summary, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--events", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    results = []
    for name in args.events:
        results.append(probe(Path(args.source_root).resolve() / name,
                             Path(args.output).resolve() / name))
    write_json(Path(args.output) / "summary.json", results)


if __name__ == "__main__":
    main()
