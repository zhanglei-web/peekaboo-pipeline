"""Timestamp-indexed sequential video detection with resumable per-frame records."""
from collections import Counter
from decimal import Decimal
from pathlib import Path
import hashlib
import json
import math
import subprocess
import time

import cv2

from online_annotation.frames import timestamps
from .detector import sha256_file


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def frame_pts(video):
    command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
               "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)]
    value = json.loads(subprocess.check_output(command, text=True))
    return [str(frame["best_effort_timestamp_time"])
            if "best_effort_timestamp_time" in frame else None for frame in value["frames"]]


def source_index(session, camera, max_frames=None):
    session = Path(session).resolve()
    if session.is_file():
        return video_clip_index(session, camera, max_frames)
    video = session / "videos" / f"{camera}.mkv"
    times_path = session / "timestamps" / f"{camera}_timestamps.txt"
    times = timestamps(times_path)
    pts = frame_pts(video)
    if len(pts) != len(times):
        raise ValueError(f"Decoded PTS count {len(pts)} != timestamp rows {len(times)}")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Cannot open {video}")
    metadata = {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)), "frame_count": len(times),
                "opencv_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT))}
    capture.release()
    if not math.isfinite(metadata["fps"]) or metadata["fps"] <= 0:
        raise ValueError("Invalid source fps")
    indices = times if max_frames is None else times[:max_frames]
    rows = [{"recording_id": session.name, "camera": camera, "source_frame_id": index,
             "frame_id": f"{camera}:f{index:06d}", "timestamp_ns": str(timestamp),
             "relative_seconds": float(Decimal(timestamp - times[0][1]) / Decimal(10**9)),
             "video_pts_seconds": pts[index],
             "timestamp_source": "capture_timestamp_file",
             "source_video": str(video), "width": metadata["width"], "height": metadata["height"]}
            for index, timestamp in indices]
    identity = {"session": str(session), "video": str(video),
                "video_sha256": sha256_file(video), "timestamps_sha256": sha256_file(times_path),
                "camera": camera, "max_frames": max_frames, "video_metadata": metadata,
                "frame_numbering": "zero_based_sequential_decode",
                "pts_mapping": "same decoded frame order; absolute clock relation not assumed"}
    return rows, identity


def video_clip_index(video, camera="left", max_frames=None):
    """A standalone input video is already one interaction clip; PTS is not epoch time."""
    video = Path(video).resolve()
    pts = frame_pts(video)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened() or not pts or any(t is None for t in pts):
        raise ValueError("Clip must decode and have per-frame PTS")
    metadata = {"width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)), "frame_count": len(pts)}
    capture.release()
    if not math.isfinite(metadata["fps"]) or metadata["fps"] <= 0:
        raise ValueError("Invalid clip fps")
    if any(Decimal(a) >= Decimal(b) for a, b in zip(pts, pts[1:])):
        raise ValueError("Clip PTS must strictly increase")
    origin = Decimal(pts[0])
    rows = [{"recording_id": video.stem, "camera": camera, "source_frame_id": i,
             "frame_id": f"{camera}:f{i:06d}", "timestamp_ns": str(int((Decimal(t) - origin) * 10**9)),
             "relative_seconds": float(Decimal(t) - origin), "video_pts_seconds": t,
             "timestamp_source": "clip_relative_video_pts", "source_video": str(video),
             "width": metadata["width"], "height": metadata["height"]}
            for i, t in enumerate(pts[:max_frames] if max_frames is not None else pts)]
    identity = {"session": None, "video": str(video), "video_sha256": sha256_file(video),
                "camera": camera, "max_frames": max_frames, "video_metadata": metadata,
                "frame_numbering": "zero_based_clip_decode", "timestamp_clock": "relative_video_pts",
                "input_scope": "single_interaction_clip"}
    return rows, identity


def overlay(frame, row, max_width=960):
    canvas = frame.copy()
    objects = {o["object_id"]: o for o in row.get("objects", [])}
    for obj in objects.values():
        x1, y1, x2, y2 = map(round, obj["bbox_xyxy"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 255), 3)
        cv2.putText(canvas, f'{obj["object_id"]} {obj["score"]:.2f}', (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 220, 255), 2)
    for hand in row.get("hands", []):
        x1, y1, x2, y2 = map(round, hand["bbox_xyxy"])
        color = (255, 120, 50) if hand["side"] == "left" else (60, 60, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        label = f'{hand["hand_id"]} {hand["side"]} {hand["score"]:.2f} {hand["contact_state"]}'
        cv2.putText(canvas, label, (max(0, x1), max(30, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
        obj = objects.get(hand.get("matched_object_id"))
        if obj:
            b = obj["bbox_xyxy"]
            cv2.line(canvas, ((x1 + x2) // 2, (y1 + y2) // 2),
                     (round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)), color, 3)
    text = (f'F{row["source_frame_id"]:06d}  {row["relative_seconds"]:.3f}s  '
            f'candidate={row.get("interaction_candidate")}  {row["status"]}')
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 48), (15, 15, 15), -1)
    cv2.putText(canvas, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2)
    if canvas.shape[1] > max_width:
        height = round(canvas.shape[0] * max_width / canvas.shape[1])
        canvas = cv2.resize(canvas, (max_width, height))
    return canvas


def validate_cached(row, source_row, run_id):
    if row.get("run_id") != run_id:
        raise ValueError("Cached detection identity differs")
    for key, value in source_row.items():
        if row.get(key) != value:
            raise ValueError(f"Cached detection source differs: {key}")
    if row.get("status") != "ok":
        raise ValueError("Failed frames must be retried")


def detect_video(session, output, detector, camera="left", max_frames=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    source_rows, source = source_index(session, camera, max_frames)
    # Only this stage's dependencies invalidate its cache. Changes to rules or
    # VLM prompts must not force another full detector run.
    module_root = Path(__file__).parent
    adapter = {name: sha256_file(module_root / name)
               for name in ("video.py", "detector.py", "compat.py")}
    adapter["online_annotation/frames.py"] = sha256_file(
        module_root.parent / "online_annotation/frames.py")
    identity = {"schema_version": 1, "source": source, "detector": detector.metadata,
                "adapter_sha256": adapter}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Run inputs or implementation changed; use a new output directory")
    save_json(manifest, identity)
    write_jsonl(output / "frame_index.jsonl", source_rows)
    records = output / "frames"
    records.mkdir(exist_ok=True)
    preview = output / "preview"
    preview.mkdir(exist_ok=True)
    writer = None
    capture = cv2.VideoCapture(source["video"])
    started = time.perf_counter()
    rows, computed, reused = [], 0, 0
    decode_failed = False
    save_json(output / "progress.json", {"status": "running", "expected_frames": len(source_rows)})
    try:
        for source_row in source_rows:
            index = source_row["source_frame_id"]
            record_path = records / f"f{index:06d}.json"
            # After a failed decode, the decoder's next ordinal is ambiguous.
            # Mark the remaining records unavailable instead of shifting IDs.
            success, frame = capture.read() if not decode_failed else (False, None)
            if not success:
                decode_failed = True
                row = {**source_row, "run_id": run_id, "status": "decode_error",
                       "interaction_candidate": None, "hands": [], "objects": [],
                       "error": "Sequential decode failed; later source identities not assumed"}
                save_json(record_path, row)
                rows.append(row)
                continue
            if record_path.exists():
                cached = json.loads(record_path.read_text())
                if cached.get("status") == "ok":
                    validate_cached(cached, source_row, run_id)
                    row = cached
                    reused += 1
                else:
                    row = None
            else:
                row = None
            if row is None:
                try:
                    prediction = detector.predict(frame)
                    row = {**source_row, "run_id": run_id, "status": "ok", **prediction}
                except Exception as exc:
                    row = {**source_row, "run_id": run_id, "status": "inference_error",
                           "interaction_candidate": None, "hands": [], "objects": [],
                           "error_type": type(exc).__name__, "error": str(exc)}
                save_json(record_path, row)
                computed += 1
                if row["status"] != "ok" and not any(r["status"] == "ok" for r in rows):
                    raise RuntimeError(f"First inference failed: {row['error']}")
            rows.append(row)
            rendered = overlay(frame, row)
            if writer is None:
                writer = cv2.VideoWriter(str(output / "detection_overlay.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         source["video_metadata"]["fps"],
                                         (rendered.shape[1], rendered.shape[0]))
                if not writer.isOpened():
                    raise OSError("Cannot create detection video")
            writer.write(rendered)
            if index % 30 == 0:
                if not cv2.imwrite(str(preview / f"f{index:06d}.jpg"), rendered):
                    raise OSError("Cannot write detection preview")
            if (index + 1) % 50 == 0 or index + 1 == len(source_rows):
                progress = {"status": "running", "processed_frames": len(rows),
                            "expected_frames": len(source_rows), "computed_frames": computed,
                            "reused_frames": reused, "seconds": time.perf_counter() - started}
                save_json(output / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    except BaseException as exc:
        save_json(output / "progress.json", {"status": "interrupted_or_failed",
                  "processed_frames": len(rows), "expected_frames": len(source_rows),
                  "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    write_jsonl(output / "detections.jsonl", rows)
    candidates = [r for r in rows if r["interaction_candidate"] is True]
    write_jsonl(output / "raw_candidates.jsonl", candidates)
    statuses = Counter(r["status"] for r in rows)
    report = {"status": "complete" if statuses.get("ok", 0) == len(source_rows) else "completed_with_errors",
              "processed_frames": len(rows), "expected_frames": len(source_rows),
              "status_counts": dict(statuses), "raw_candidate_frames": len(candidates),
              "computed_frames": computed, "reused_frames": reused,
              "elapsed_seconds": time.perf_counter() - started,
              "total_inference_seconds": sum(r.get("inference_seconds", 0) for r in rows),
              "contact_state_counts": dict(Counter(h["contact_state"] for r in rows for h in r["hands"])),
              "object_missing_candidate_hands": sum(h.get("matching_status") == "object_missing"
                                                     for r in rows for h in r["hands"]),
              "manual_recall_validated": False, "pnr_selected": False}
    save_json(output / "detection_report.json", report)
    save_json(output / "progress.json", report)
    return report
