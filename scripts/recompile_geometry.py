"""Recompute geometry for unchanged verified 8B selections without rerunning selection."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_memory.io import read_json, write_json
from spatial_memory.selection import MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "output", "pnr-root", "config", "geometry-python", "model-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    args = parser.parse_args()
    summaries = []
    for recording in args.recordings:
        source = Path(args.source_run).resolve() / recording
        result = read_json(source / "full_result.json")
        selection = read_json(source / "selection/selection.json")
        if selection["model"] != MODEL:
            raise ValueError("Refuse to substitute an earlier non-8B selection")
        out = Path(args.output).resolve() / recording
        out.mkdir(parents=True, exist_ok=True)
        with (out / "geometry.log").open("w") as log:
            subprocess.run([args.geometry_python, "-m", "spatial_memory", "compile-geometry",
                "--session", result["session"], "--config", args.config,
                "--event", str(source / "candidates/event_candidates.json"),
                "--selection", str(source / "selection/selection.json"),
                "--catalog", str(Path(result["stage1_source"]) / "catalog.v0.json"),
                "--memory", str(source / "initial_memory.json"),
                "--detections", str(Path(args.pnr_root) / recording / "detection/detections.jsonl"),
                "--model-root", args.model_root, "--base-url", args.base_url,
                "--output", str(out / "geometry")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        record = read_json(out / "geometry/committed_event.json")
        result["selection_source"] = str(source / "selection/selection.json")
        result["geometry_assessments"] = record["assessments"]
        result["geometry_status"] = "automatic_compiler_completed"
        result["numeric_assessment_count"] = sum(a["q"] is not None for a in record["assessments"])
        result["local_numeric_pipeline_completed"] = bool(record["assessments"]) and all(
            a["q"] is not None for a in record["assessments"])
        result["full_numeric_pipeline_completed"] = result["local_numeric_pipeline_completed"] and all(
            a.get("region_association_status") != "ambiguous_parent_region_links" for a in record["assessments"])
        write_json(out / "full_result.json", result)
        summaries.append(result)
        write_json(Path(args.output) / "full_summary.json", summaries)
        print({"recording": recording, "model": MODEL,
               "assessments": [(a["q"], a.get("reason")) for a in record["assessments"]]}, flush=True)


if __name__ == "__main__":
    main()
