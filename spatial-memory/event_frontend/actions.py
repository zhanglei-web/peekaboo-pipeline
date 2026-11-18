"""Offline sliding-window TSM evidence; original frame IDs, no reference inputs.

Scores are classification probabilities, not calibrated event boundaries.
All 97 verb and 300 noun probabilities are retained, including unrelated labels.
Unknown/low-confidence windows are never renamed as background or hand release.
"""
from pathlib import Path
import json
import math
import time

import cv2
import numpy as np
from PIL import Image

from .detector import sha256_file
from .video import read_jsonl, save_json, write_jsonl


def action_config(path=None):
    config = json.loads(Path(__file__).with_name("action_config.json").read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        if set(supplied) - set(config):
            raise ValueError("Unknown action configuration keys")
        config.update(supplied)
    for key, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid action setting: {key}")
    if type(config["batch_size"]) is not int:
        raise ValueError("batch_size must be an integer")
    if any(config[k] > 1 for k in ("minimum_action_probability", "minimum_action_margin")):
        raise ValueError("Action probabilities and margins must be at most one")
    if config["stride_seconds"] > config["window_seconds"]:
        raise ValueError("Action windows must overlap or touch")
    return config


def plan_windows(index, config):
    if not index:
        raise ValueError("Empty frame index")
    ids = [r["source_frame_id"] for r in index]
    times = np.asarray([r["relative_seconds"] for r in index], dtype=float)
    if any(b != a + 1 for a, b in zip(ids, ids[1:])) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("TSM needs sequential source IDs and increasing finite timestamps")
    half = config["window_seconds"] / 2
    centers = np.arange(times[0], times[-1] + 1e-8, config["stride_seconds"])
    if times[-1] - centers[-1] > 1e-8:
        centers = np.r_[centers, times[-1]]
    result, seen = [], set()
    for center in centers:
        center_i = int(np.argmin(abs(times - center)))
        if center_i in seen:
            continue
        seen.add(center_i)
        center = times[center_i]
        lo = int(np.searchsorted(times, max(times[0], center - half), side="left"))
        hi = int(np.searchsorted(times, min(times[-1], center + half), side="right"))
        # Official TSN test sampling: the middle frame of each equal bin.
        sampled = lo + np.minimum(((np.arange(8) + .5) * ((hi - lo) / 8)).astype(int), hi - lo - 1)
        result.append({"window_id": f"w{len(result):05d}", "source_frame_id": ids[center_i],
                       "relative_seconds": float(times[center_i]),
                       "start_source_frame_id": ids[lo], "end_source_frame_id": ids[hi - 1],
                       "start_seconds": float(times[lo]), "end_seconds": float(times[hi - 1]),
                       "sampled_source_frame_ids": [ids[int(i)] for i in sampled],
                       "edge_truncated": bool(center - half < times[0] or center + half > times[-1])})
    return result


def top_predictions(probabilities, names, count=5):
    return [{"class_id": int(i), "label": names[i], "probability": float(probabilities[i])}
            for i in np.argsort(-np.asarray(probabilities), kind="stable")[:count]]


def classify_window(row, verb, noun, labels, config):
    row = dict(row)
    row.update(verb_probabilities=verb.tolist(), noun_probabilities=noun.tolist(),
               top_verbs=top_predictions(verb, labels["verb"]),
               top_nouns=top_predictions(noun, labels["noun"]))
    best = row["top_verbs"]
    margin = best[0]["probability"] - best[1]["probability"]
    confident = (best[0]["probability"] >= config["minimum_action_probability"]
                 and margin >= config["minimum_action_margin"])
    row.update(action_label=best[0]["label"] if confident else None,
               verb_margin=margin, action_status="candidate" if confident else "uncertain")
    return row


def extract_actions(run, output, checkpoint, labels_dir, config, device="cuda", view="center"):
    from .tsm_model import TSMRecognizer, CHECKPOINT_SHA256, LABEL_SHA256
    run, output, checkpoint, labels_dir = [Path(p).resolve() for p in (run, output, checkpoint, labels_dir)]
    if output == run:
        raise ValueError("Use a separate action output directory")
    manifest = json.loads((run / "manifest.json").read_text())
    source = manifest["source"]
    index = read_jsonl(run / "frame_index.jsonl")
    if view not in ("center", "contact_context"):
        raise ValueError("Unknown TSM action view")
    if sha256_file(source["video"]) != source["video_sha256"]:
        raise ValueError("Source video changed")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Unexpected TSM checkpoint")
    for task, expected in LABEL_SHA256.items():
        if sha256_file(labels_dir / f"EPIC_100_{task}_classes.csv") != expected:
            raise ValueError("Unexpected TSM class file")
    windows = plan_windows(index, config)
    from .action_views import CONTEXT_SETTINGS, contact_context_view, crop_action_view
    width, height = index[0]["width"], index[0]["height"]
    if view == "contact_context":
        detections = read_jsonl(run / "detections.jsonl")
        if [r["source_frame_id"] for r in detections] != [r["source_frame_id"] for r in index]:
            raise ValueError("Action detections and source index differ")
        lookup = {r["source_frame_id"]: r for r in detections}
        for row in windows:
            row.update(contact_context_view(row, lookup, width, height))
    identity = {"stage": "tsm_sliding_window_actions", "model": "EPIC100_TSM_RGB_ResNet50_8segments",
                "checkpoint": str(checkpoint), "checkpoint_sha256": CHECKPOINT_SHA256,
                "label_sha256": LABEL_SHA256, "reference_annotations_used": False, "vlm_used": False,
                "completion_signals_used": False, "offline_centered_windows": True,
                "view": view, "view_settings": CONTEXT_SETTINGS if view == "contact_context" else None,
                "detection_manifest_sha256": sha256_file(run / "manifest.json"),
                "detections_sha256": sha256_file(run / "detections.jsonl"),
                "frame_index_sha256": sha256_file(run / "frame_index.jsonl"), "configuration": config,
                "implementation_sha256": {p: sha256_file(Path(__file__).with_name(p))
                    for p in ("actions.py", "tsm_model.py", "action_views.py")}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text()) != identity:
            raise ValueError("Action inputs changed; use a new output directory")
        if (output / "report.json").exists():
            report = json.loads((output / "report.json").read_text())
            if sha256_file(output / "actions.jsonl") != report["actions_sha256"]:
                raise ValueError("Cached action scores changed")
            if sha256_file(output / "labels.json") != report["labels_sha256"]:
                raise ValueError("Cached action labels changed")
            return report
    save_json(output / "manifest.json", identity)
    extractor = TSMRecognizer(checkpoint, labels_dir, device)
    if device.startswith("cuda"):
        import torch
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    wanted = set(f for w in windows for f in w["sampled_source_frame_ids"])
    capture = cv2.VideoCapture(source["video"])
    images, pending, scores = {}, [], []
    last_use = {f: max(i for i, w in enumerate(windows) if f in w["sampled_source_frame_ids"]) for f in wanted}
    def flush():
        if not pending:
            return
        verb, noun = extractor.encode([[crop_action_view(images[f], w, width, height)
                                       for f in w["sampled_source_frame_ids"]] for _, w in pending])
        for (_, row), v, n in zip(pending, verb, noun):
            scores.append(classify_window(row, v, n, extractor.labels, config))
        done = pending[-1][0]
        for f in list(images):
            if last_use[f] <= done:
                del images[f]
        pending.clear()
    next_window = 0
    try:
        for frame_id in range(index[-1]["source_frame_id"] + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode action frame {frame_id}")
            if frame_id in wanted:
                # Pre-resize to the official short side before retaining frames.
                frame_height, frame_width = frame.shape[:2]
                rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                short = CONTEXT_SETTINGS["retained_image_short_side"] if view == "contact_context" else 256
                resized = ((short, int(short * frame_height / frame_width)) if frame_width <= frame_height
                           else (int(short * frame_width / frame_height), short))
                images[frame_id] = rgb.resize(resized, Image.Resampling.BILINEAR)
            while next_window < len(windows) and max(windows[next_window]["sampled_source_frame_ids"]) <= frame_id:
                pending.append((next_window, windows[next_window]))
                next_window += 1
                if len(pending) >= config["batch_size"]:
                    flush()
                    if len(scores) % 40 == 0:
                        print(json.dumps({"stage": "tsm_actions", "windows": len(scores), "total": len(windows)}), flush=True)
        flush()
    finally:
        capture.release()
    if len(scores) != len(windows):
        raise ValueError("Incomplete TSM window coverage")
    write_jsonl(output / "actions.jsonl", scores)
    save_json(output / "labels.json", extractor.labels)
    elapsed = time.perf_counter() - started
    report = {"window_count": len(scores), "source_frame_count": len(index),
              "confident_window_count": sum(s["action_label"] is not None for s in scores),
              "inference_and_decode_seconds": elapsed, "windows_per_second": len(scores) / elapsed,
              "parameter_count": sum(p.numel() for p in extractor.model.parameters()),
              "peak_cuda_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.startswith("cuda") else None,
              "reference_annotations_used": False, "vlm_used": False,
              "view": view,
              "actions_sha256": sha256_file(output / "actions.jsonl"),
              "labels_sha256": sha256_file(output / "labels.json"),
              "preprocessing": extractor.preprocessing,
              "limitation": "Closed-set EPIC classification, not calibrated action or event-boundary accuracy"}
    save_json(output / "report.json", report)
    return report
