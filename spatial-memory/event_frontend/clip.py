"""Explicit interaction-clip scope over cached source detections."""
from pathlib import Path
import json
import re

from .detector import sha256_file
from .video import read_jsonl, save_json, write_jsonl


def scope_detection(run, output, start, end, clip_id):
    run, output = Path(run).resolve(), Path(output).resolve()
    if start < 0 or end < start or not re.fullmatch(r"[\w-]+", clip_id):
        raise ValueError("Provide clip ID and inclusive ordered source frame boundaries")
    rows = read_jsonl(run / "detections.jsonl")
    index = read_jsonl(run / "frame_index.jsonl")
    keep = [r for r in rows if start <= r["source_frame_id"] <= end]
    if [r["source_frame_id"] for r in keep] != list(range(start, end + 1)):
        raise ValueError("Clip boundaries outside cached detection range")
    parent = json.loads((run / "manifest.json").read_text())
    identity = {**parent, "input_scope": "single_interaction_clip", "clip_id": clip_id,
                "clip_source_range": [start, end], "parent_detection_run": str(run),
                "parent_manifest_sha256": sha256_file(run / "manifest.json"),
                "parent_detections_sha256": sha256_file(run / "detections.jsonl"),
                "scope_implementation_sha256": sha256_file(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Clip scope changed; use another output directory")
    save_json(output / "manifest.json", identity)
    write_jsonl(output / "detections.jsonl", keep)
    write_jsonl(output / "frame_index.jsonl", [r for r in index if start <= r["source_frame_id"] <= end])
    write_jsonl(output / "raw_candidates.jsonl", [r for r in keep if r["interaction_candidate"] is True])
    report = {"input_scope": "single_interaction_clip", "clip_id": clip_id, "clip_source_range": [start, end],
              "processed_frames": len(keep), "detector_rerun": False, "source_frame_ids_preserved": True}
    save_json(output / "detection_report.json", report)
    return report
