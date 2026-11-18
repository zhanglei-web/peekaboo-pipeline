"""Run the unchanged legacy PNR selector on this run's shared per-event features."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--legacy-python", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    run, legacy, output = (Path(p).resolve() for p in (args.run_root, args.legacy_root, args.output))
    if output.exists():
        raise ValueError("Use a fresh output directory for an actual legacy-selector rerun")
    output.mkdir(parents=True)
    implementation = {str(p): sha(p) for p in (legacy / "pnr").glob("*.py")}
    jobs = []
    for session in read(run / "run_manifest.json")["sessions"]:
        name = Path(session).name
        for event in read(run / name / "events/predictions.json")["predictions"]:
            jobs.append((name, event["event_id"]))
    manifest = {"run_root": str(run), "legacy_root": str(legacy), "legacy_python": args.legacy_python,
                "legacy_source_hashes": implementation,
                "control": "same predicted events, detections, contact clusters and dense feature scores as current VLM run",
                "model_calls_in_rule_selector": 0, "expected_events": len(jobs)}
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

    def execute(job):
        name, eid = job
        source = run / name / "event_runs" / eid
        folder = output / name / eid
        folder.mkdir(parents=True)
        command = [args.legacy_python, "-m", "pnr", "select", "--detection-run", str(source / "clip"),
                   "--candidates", str(source / "rank"), "--output", str(folder / "rule_selection")]
        tick = time.perf_counter()
        with (folder / "select.log").open("w") as log:
            proc = subprocess.run(command, cwd=legacy, stdout=log, stderr=subprocess.STDOUT)
        receipt = {"recording": name, "event_id": eid, "command": command, "exit_code": proc.returncode,
                   "wall_seconds": time.perf_counter() - tick,
                   "input_hashes": {str(p): sha(p) for p in [source / "clip/manifest.json",
                       source / "clip/frame_index.jsonl", source / "rank/manifest.json",
                       source / "rank/packets.jsonl", source / "rank/feature_scores.jsonl"]}}
        if not proc.returncode:
            receipt["result"] = read(folder / "rule_selection/result.json")
            assert receipt["result"]["model_calls"] == 0 and receipt["result"]["vlm_used"] is False
        (folder / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
        print({"recording": name, "event": eid, "exit_code": proc.returncode,
               "triplet": [receipt.get("result", {}).get(k) for k in ("p_frame_id", "n_frame_id", "r_frame_id")]}, flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(execute, jobs))
    for p, value in implementation.items():
        assert sha(p) == value, "Legacy implementation changed during execution"
    summary = {**manifest, "executed_events": len(results), "successful_commands": sum(r["exit_code"] == 0 for r in results),
               "complete_triplets": sum(r.get("result", {}).get("status") == "complete" for r in results),
               "results": results}
    (output / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if summary["successful_commands"] != len(jobs):
        raise RuntimeError("Some legacy selections failed; inspect preserved select.log files")


if __name__ == "__main__":
    main()
