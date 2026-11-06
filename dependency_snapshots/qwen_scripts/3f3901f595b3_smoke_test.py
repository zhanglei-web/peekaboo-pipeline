#!/usr/bin/env python3
"""Real server tests, sequential single -> 2 -> 4 -> video, and warm repeats."""
import json
import subprocess
import sys
import time
from local_vlm.client import Client, new_output, save_json
from local_vlm.config import ROOT, load_config
from local_vlm.media import load_manifest, extract_video

def main():
    subprocess.run([sys.executable, str(ROOT / "scripts/create_test_media.py")], check=True)
    cfg = load_config()
    client = Client(cfg)
    report = {"synthetic_only": True, "real_workbench_quality": "NOT VALIDATED", "tests": []}
    destination = ROOT / "outputs/smoke-report.json"
    for endpoint in ["/health", "/v1/models"]:
        r = client.session.get(client.base + endpoint, timeout=10)
        r.raise_for_status()
        report[endpoint] = r.json()
    cases = ["single_first", "single_warm", "2frames", "4frames", "video", "4frames_warm"]
    for case in cases:
        print(f"Testing {case}", flush=True)
        out = new_output()
        row = {"case": case, "output": str(out)}
        report["tests"].append(row)
        save_json(destination, report)
        try:
            if case.startswith("single"):
                manifest = {"clip_id": "synthetic_single", "synthetic": True, "frames": [{"frame_id": "f0", "path": str(ROOT / "data/synthetic/frame-0.png"), "timestamp_s": 0.0}]}
                task = "describe"
            elif case == "video":
                path = extract_video(ROOT / "data/synthetic/synthetic-demo.mp4", out, 0.0, 2.0, 4)
                manifest = load_manifest(path, 4)
                task = "event_annotation"
            else:
                count = 2 if case == "2frames" else 4
                manifest = load_manifest(ROOT / f"data/synthetic/frames-{count}.json", 4)
                task = "event_annotation"
            _, result = client.run(manifest, task, output_dir=out)
            if task == "describe":
                if not all(color in result["answer"] for color in ["红", "蓝", "绿"]):
                    raise AssertionError("Visual color content check failed")
            elif not result["objects"]:
                raise AssertionError("Synthetic image contains objects but result has none")
            meta = json.loads((out / "metadata.json").read_text())
            row.update(status="passed", elapsed_s=meta["elapsed_s"], resources=meta["resources"], usage=meta.get("usage"), attempts=meta["attempts"])
            print(f"Passed {case}: {meta['elapsed_s']:.2f}s; {out}", flush=True)
        except Exception as exc:
            row.update(status="failed", error=str(exc))
            save_json(destination, report)
            raise
        save_json(destination, report)
    print(destination)

if __name__ == "__main__":
    main()
