"""Conservative, configurable candidate reduction; every decision is recorded."""
from pathlib import Path
import json

import cv2
import numpy as np

from .detector import sha256_file
from .video import read_jsonl, save_json, write_jsonl


DEFAULTS = {
    "hand_score_min": 0.3,
    "contact_probability_min": 0.0,
    "context_before_seconds": 1.5,
    "context_after_seconds": 1.5,
    "representative_hz": 4.0,
    "context_hz": 2.0,
    "duplicate_gray_mae": 0.012,
    "duplicate_max_gap_seconds": 0.5,
    "boundary_protection_seconds": 0.15,
    "max_frames_per_packet": 20,
    "packet_overlap_frames": 4,
    "packet_max_gap_seconds": 3.0,
    "image_max_edge": 1280,
}


def config_from(path=None):
    config = dict(DEFAULTS)
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown = set(supplied) - set(config)
        if unknown:
            raise ValueError(f"Unknown candidate parameters: {sorted(unknown)}")
        config.update(supplied)
    if not 0 <= config["hand_score_min"] <= 1 or not 0 <= config["contact_probability_min"] <= 1:
        raise ValueError("Confidence thresholds must be in [0,1]")
    for key in ("context_before_seconds", "context_after_seconds", "duplicate_gray_mae",
                "duplicate_max_gap_seconds", "boundary_protection_seconds"):
        if config[key] < 0:
            raise ValueError(f"{key} must be nonnegative")
    if any(config[k] <= 0 for k in ("representative_hz", "context_hz", "packet_max_gap_seconds", "image_max_edge")):
        raise ValueError("Rates, image dimensions and maximum gap must be positive")
    size, overlap = config["max_frames_per_packet"], config["packet_overlap_frames"]
    if not isinstance(size, int) or not isinstance(overlap, int) or not 0 <= overlap < size or size < 3:
        raise ValueError("Packet overlap must be an integer in [0, packet_size), packet_size >=3")
    return config


def filtered_contacts(row, config):
    if row["status"] != "ok":
        return []
    return [h for h in row["hands"] if h["contact_state_id"] in (3, 4)
            and h["score"] >= config["hand_score_min"]
            and h["contact_probabilities"][h["contact_state_id"]] >= config["contact_probability_min"]]


def temporal_selection(rows, config):
    """Suggest sample IDs without asserting event boundaries or object tracks."""
    times = np.array([r["relative_seconds"] for r in rows])
    contacts = [filtered_contacts(r, config) for r in rows]
    active = np.array([bool(hands) for hands in contacts])
    intervals = []
    for i in np.flatnonzero(active):
        start = times[i] - config["context_before_seconds"]
        end = times[i] + config["context_after_seconds"]
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(end, intervals[-1][1])
        else:
            intervals.append([start, end])
    boundary_times = []
    # Side, hand-count and portable/stationary labels can flicker while contact
    # continues. They are evidence, not reliable dense-sampling triggers.
    for i, is_active in enumerate(active):
        if (i == 0 and is_active) or (i and is_active != active[i - 1]):
            boundary_times.append(times[i])
    protected = np.zeros(len(rows), dtype=bool)
    for when in boundary_times:
        protected |= np.abs(times - when) <= config["boundary_protection_seconds"]
    decisions, last_time, interval_index = [], -float("inf"), 0
    for i, row in enumerate(rows):
        t = times[i]
        while interval_index < len(intervals) and intervals[interval_index][1] < t:
            interval_index += 1
        inside = interval_index < len(intervals) and intervals[interval_index][0] <= t <= intervals[interval_index][1]
        reason = "outside_candidate_context"
        selected = False
        if row["status"] != "ok":
            reason = "source_processing_error"
        elif not inside:
            pass
        elif protected[i]:
            selected, reason = True, "contact_boundary_context"
        else:
            hz = config["representative_hz"] if active[i] else config["context_hz"]
            if t - last_time >= 1 / hz - 1e-6:
                selected = True
                reason = "interaction_representative" if active[i] else "non_contact_context"
            else:
                reason = "temporal_density_reduction"
        if selected:
            last_time = t
        decisions.append({"frame_id": row["frame_id"], "source_frame_id": row["source_frame_id"],
                          "relative_seconds": float(t), "raw_candidate": row["interaction_candidate"],
                          "passes_contact_rule": bool(active[i]), "inside_candidate_context": bool(inside),
                          "protected_boundary": bool(protected[i] and inside),
                          "selected": selected, "reason": reason})
    return decisions, intervals


def packet_groups(rows, config):
    groups, current = [], []
    for row in rows:
        if current and row["relative_seconds"] - current[-1]["relative_seconds"] > config["packet_max_gap_seconds"]:
            groups.append(current)
            current = []
        current.append(row)
    if current:
        groups.append(current)
    result = []
    size, overlap = config["max_frames_per_packet"], config["packet_overlap_frames"]
    for group in groups:
        start = 0
        while start < len(group):
            chunk = group[start:start + size]
            # Rebalance the final tiny window so it can contain a triplet.
            if len(chunk) < 3 and len(group) >= 3:
                chunk = group[-min(size, len(group)):]
            if not result or [r["frame_id"] for r in result[-1]] != [r["frame_id"] for r in chunk]:
                result.append(chunk)
            if start + size >= len(group):
                break
            start += size - overlap
    return result


def labeled_image(frame, row, max_edge):
    from PIL import Image, ImageDraw
    scale = min(1, max_edge / max(frame.shape[:2]))
    image = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    canvas = Image.new("RGB", (image.width, image.height + 32), "#111827")
    canvas.paste(image, (0, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f'{row["frame_id"]} | {row["relative_seconds"]:.3f}s', fill="white")
    return canvas


def prepare_candidates(run, output, config):
    run, output = Path(run).resolve(), Path(output).resolve()
    rows = read_jsonl(run / "detections.jsonl")
    if not rows:
        raise ValueError("No source detection records")
    if any(r["source_frame_id"] != i for i, r in enumerate(rows)):
        raise ValueError("Detection records must cover sequential source frames")
    identity = {"detection_manifest_sha256": sha256_file(run / "manifest.json"),
                "detections_sha256": sha256_file(run / "detections.jsonl"),
                "config": config, "implementation_sha256": sha256_file(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Candidate inputs changed; use another output directory")
    save_json(output / "manifest.json", identity)
    decisions, intervals = temporal_selection(rows, config)
    image_dir = output / "images"
    image_dir.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(rows[0]["source_video"])
    if not capture.isOpened():
        raise ValueError("Cannot open source video for candidates")
    selected, last_gray, last_time = [], None, -float("inf")
    try:
        for row, decision in zip(rows, decisions):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {row['source_frame_id']}")
            if not decision["selected"]:
                continue
            gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 40)).astype(np.float32) / 255
            distance = float(np.abs(gray - last_gray).mean()) if last_gray is not None else None
            decision["appearance_gray_mae"] = distance
            if (distance is not None and distance < config["duplicate_gray_mae"]
                    and row["relative_seconds"] - last_time < config["duplicate_max_gap_seconds"]
                    and not decision["protected_boundary"]):
                decision.update(selected=False, reason="appearance_duplicate")
                continue
            path = image_dir / f'{row["frame_id"].replace(":", "_")}.jpg'
            labeled_image(frame, row, config["image_max_edge"]).save(path, quality=92)
            selected.append({"frame_id": row["frame_id"], "source_frame_id": row["source_frame_id"],
                             "timestamp_ns": row["timestamp_ns"], "relative_seconds": row["relative_seconds"],
                             "camera": row["camera"], "source_video": row["source_video"],
                             "image": str(path), "image_sha256": sha256_file(path),
                             "reason": decision["reason"], "interaction_candidate": row["interaction_candidate"]})
            last_gray, last_time = gray, row["relative_seconds"]
    finally:
        capture.release()
    packets = []
    for i, group in enumerate(packet_groups(selected, config)):
        packets.append({"packet_id": f"p{i:04d}", "recording_id": rows[0]["recording_id"],
                        "frames": group, "event_boundaries_known": False,
                        "source_range": [group[0]["source_frame_id"], group[-1]["source_frame_id"]]})
    write_jsonl(output / "filter_decisions.jsonl", decisions)
    write_jsonl(output / "candidates.jsonl", selected)
    write_jsonl(output / "packets.jsonl", packets)
    from collections import Counter
    report = {"source_frames": len(rows), "raw_candidate_frames": sum(r["interaction_candidate"] is True for r in rows),
              "frames_passing_contact_rule": sum(d["passes_contact_rule"] for d in decisions),
              "selected_frames": len(selected), "packet_count": len(packets),
              "context_intervals_seconds": intervals, "decision_counts": dict(Counter(d["reason"] for d in decisions)),
              "context_truncated_at_recording_start": any(start < 0 for start, _ in intervals),
              "context_truncated_at_recording_end": any(end > rows[-1]["relative_seconds"] for _, end in intervals),
              "manual_false_removal_validated": False, "event_boundaries_determined": False}
    save_json(output / "candidate_report.json", report)
    return report
