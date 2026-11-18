"""Fresh raw-recording -> every predicted event -> always-numeric spatial memory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_memory.io import read_json, sha256, write_json
from spatial_memory.selection import MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frontend-workers", type=int, default=2)
    args = parser.parse_args()
    config = read_json(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists():
        raise ValueError("Full fresh test requires a new output directory")
    model = read_json(Path(config["model_root"]) / "manifest.json")
    if model["upstream"] != "Qwen/Qwen3-VL-8B-Instruct":
        raise ValueError("The requested 8B weight manifest is required")
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for directory in ("event_frontend", "online_annotation", "spatial_memory", "scripts")
                     for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in (".py", ".json", ".txt")}
    write_json(output / "run_manifest.json", {"model": MODEL, "weight_revision": model["revision"],
        "weight_quantization": model["quantization"], "score_policy": "best_effort",
        "sessions": config["sessions"], "source_hashes": source_hashes,
        "historical_prediction_caches_used": False, "raw_recordings_processed_in_full": True})
    start = time.perf_counter()

    def frontend(session_path):
        session = Path(session_path).resolve()
        out = output / session.name
        if out.exists():
            raise ValueError(f"Recording output already exists: {out}")
        out.mkdir()
        trace = []
        def run(name, command):
            tick = time.perf_counter()
            print({"recording": session.name, "stage": name, "status": "started"}, flush=True)
            with (out / f"{name}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            trace.append({"stage": name, "exit_code": result.returncode, "seconds": time.perf_counter() - tick,
                          "command": command})
            write_json(out / "stage_trace.json", trace)
            if result.returncode:
                raise RuntimeError(f"{session.name}: {name} failed; inspect {out / (name + '.log')}")
            print({"recording": session.name, "stage": name, "status": "complete"}, flush=True)
        run("stage1", [sys.executable, "-m", "online_annotation", "initialize", "--session", str(session),
            "--output", str(out / "stage1"), "--candidate-mode", "brief",
            "--model-root", config["model_root"], "--base-url", config["base_url"]])
        command = [config["frontend_python"], "-m", "event_frontend"]
        run("detection", command + ["detect", "--session", str(session), "--output", str(out / "detection"),
            "--repository", config["detection_repository"], "--checkpoint", config["detection_checkpoint"]])
        report = read_json(out / "detection/detection_report.json")
        if report["reused_frames"] or report["computed_frames"] != report["expected_frames"]:
            raise ValueError("Detector must run all source frames without old cached predictions")
        run("actions", command + ["recognize-actions", "--run", str(out / "detection"),
            "--checkpoint", config["tsm_checkpoint"], "--labels-dir", config["tsm_labels"], "--output", str(out / "actions")])
        run("signals", command + ["extract-signals", "--run", str(out / "detection"),
            "--clip-checkpoint", config["clip_checkpoint"], "--output", str(out / "signals")])
        run("events", command + ["segment-events", "--run", str(out / "detection"),
            "--actions", str(out / "actions"), "--signals", str(out / "signals"), "--output", str(out / "events")])
        run("export_specs", command + ["export-specs", "--run", str(out / "detection"),
            "--actions", str(out / "actions"), "--events", str(out / "events"), "--output", str(out / "event_specs")])
        return session, out

    with ThreadPoolExecutor(max_workers=args.frontend_workers) as pool:
        completed = list(pool.map(frontend, config["sessions"]))
    summaries = []
    for session, out in completed:
        print({"recording": session.name, "stage": "all_events_to_memory", "status": "started"}, flush=True)
        tick = time.perf_counter()
        with (out / "all_events.log").open("w") as log:
            subprocess.run([config["geometry_python"], str(ROOT / "scripts/process_recording.py"),
                "--config", str(Path(args.config).resolve()), "--recording-root", str(out),
                "--session", str(session)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = read_json(out / "recording_summary.json")
        summary["detector_report"] = read_json(out / "detection/detection_report.json")
        summary["all_events_wall_seconds"] = time.perf_counter() - tick
        summaries.append(summary)
        write_json(output / "full_summary.json", {"recordings": summaries,
            "elapsed_seconds": time.perf_counter() - start, "model": MODEL,
            "historical_prediction_caches_used": False})
        print({"recording": session.name, "status": "complete", "events": summary["committed_events"],
               "methods": summary["score_method_counts"]}, flush=True)


if __name__ == "__main__":
    main()
