"""Run the two-stage integration test using only the verified official 8B service."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from smoke_real import probe
from spatial_memory.io import read_json, write_json
from spatial_memory.selection import MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pnr-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--recordings", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--geometry-python", required=True)
    parser.add_argument("--geometry-config", required=True)
    parser.add_argument("--image-max-edge", type=int, default=768)
    parser.add_argument("--stage1-root")
    args = parser.parse_args()
    root = Path(args.model_root)
    config = read_json(root / "config.json")
    manifest = read_json(root / "manifest.json")
    if config["model_alias"] != MODEL or manifest["upstream"] != "Qwen/Qwen3-VL-8B-Instruct":
        raise ValueError("Expected the requested actual 8B model and its original weight manifest")
    key = next(s.strip() for s in (root / config["api_key_file"]).read_text().splitlines()
               if s.strip() and not s.lstrip().startswith("#"))
    results = []
    for recording in args.recordings:
        result = probe(recording, args, MODEL, key)
        out = Path(args.output).resolve() / recording
        result["weight_revision"] = manifest["revision"]
        result["weight_quantization"] = manifest.get("quantization")
        if result.get("selection_status") == "ok":
            try:
                with (out / "geometry.log").open("w") as log:
                    subprocess.run([args.geometry_python, "-m", "spatial_memory", "compile-geometry",
                        "--session", result["session"], "--config", args.geometry_config,
                        "--event", str(out / "candidates/event_candidates.json"),
                        "--selection", str(out / "selection/selection.json"),
                        "--catalog", str(Path(result["stage1_source"]) / "catalog.v0.json"),
                        "--memory", str(out / "initial_memory.json"),
                        "--detections", str(Path(args.pnr_root) / recording / "detection/detections.jsonl"),
                        "--model-root", args.model_root, "--base-url", args.base_url,
                        "--output", str(out / "geometry")], cwd=ROOT,
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                record = read_json(out / "geometry/committed_event.json")
                result["geometry_assessments"] = record["assessments"]
                result["geometry_status"] = "automatic_compiler_completed"
                result["numeric_assessment_count"] = sum(a["q"] is not None for a in record["assessments"])
                result["local_numeric_pipeline_completed"] = bool(record["assessments"]) and all(
                    a["q"] is not None for a in record["assessments"])
                result["full_numeric_pipeline_completed"] = result["local_numeric_pipeline_completed"] and all(
                    a.get("region_association_status") != "ambiguous_parent_region_links" for a in record["assessments"])
            except (subprocess.CalledProcessError, OSError, ValueError) as exc:
                result["geometry_status"] = "automatic_compiler_failed"
                result["geometry_error"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "full_result.json", result)
        results.append(result)
        write_json(Path(args.output) / "full_summary.json", results)
        print(json.dumps({"case": recording, "model": MODEL,
                          "geometry_status": result.get("geometry_status"),
                          "numeric_assessments": result.get("numeric_assessment_count", 0),
                          "error": result.get("error") or result.get("geometry_error")}), flush=True)


if __name__ == "__main__":
    main()
