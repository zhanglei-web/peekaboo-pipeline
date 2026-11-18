"""Label-free evidence for full-recording event segmentation.

Contact-object crops describe targets. Camera-compensated hand positions
describe motion. Neither reference annotations nor event counts are inputs.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from .detector import sha256_file
from .features import CLIP_SHA256, ClipImageFeatures, fixed_crop
from .objects import area, appearance, center
from .video import read_jsonl, save_json, write_jsonl


def choose_contact(row, config, width, height, primary_side):
    objects = {o["object_id"]: o for o in row["objects"]}
    choices = []
    for hand in row["hands"]:
        if hand["score"] < config["hand_score_min"] or hand["contact_state_id"] not in (3, 4):
            continue
        obj = objects.get(hand.get("matched_object_id"))
        if obj is None:
            continue
        fraction = area(obj["bbox_xyxy"]) / (width * height)
        ratio = area(obj["bbox_xyxy"]) / max(1, area(hand["bbox_xyxy"]))
        reliable = (fraction <= config["maximum_object_image_fraction"] and
                    ratio <= config["maximum_object_hand_area_ratio"])
        # Foreground geometry takes precedence over left/right classification.
        score = (reliable, hand["side"] == primary_side, hand["score"] * obj["score"])
        choices.append((score, hand, obj, fraction, ratio))
    if not choices:
        return None
    _, hand, obj, fraction, ratio = max(choices, key=lambda v: v[0])
    return {"hand": hand, "object": obj, "reliable_target": bool(reliable_geometry(fraction, ratio, config)),
            "object_image_fraction": fraction, "object_hand_area_ratio": ratio}


def reliable_geometry(fraction, ratio, config):
    return fraction <= config["maximum_object_image_fraction"] and ratio <= config["maximum_object_hand_area_ratio"]


def background_transform(previous, current, old_hands, new_hands, width, height):
    """Estimate old->new camera motion from background LK correspondences."""
    mask = np.full(previous.shape, 255, np.uint8)
    sx, sy = previous.shape[1] / width, previous.shape[0] / height
    for hand in old_hands:
        x1, y1, x2, y2 = hand["bbox_xyxy"]
        cv2.rectangle(mask, (max(0, int(x1 * sx) - 8), max(0, int(y1 * sy) - 8)),
                      (int(x2 * sx) + 8, int(y2 * sy) + 8), 0, -1)
    points = cv2.goodFeaturesToTrack(previous, maxCorners=220, qualityLevel=.015, minDistance=10, mask=mask)
    identity = np.eye(3)
    if points is None or len(points) < 12:
        return identity, 0.
    moved, status, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None,
                                             winSize=(21, 21), maxLevel=3)
    if moved is None or status is None:
        return identity, 0.
    a, b = points[status.flatten() == 1].reshape(-1, 2), moved[status.flatten() == 1].reshape(-1, 2)
    keep = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    # Reject tracks ending on a hand as well as tracks starting on one.
    for hand in new_hands:
        x1, y1, x2, y2 = hand["bbox_xyxy"]
        keep &= ~((b[:, 0] >= x1 * sx - 8) & (b[:, 0] <= x2 * sx + 8) &
                  (b[:, 1] >= y1 * sy - 8) & (b[:, 1] <= y2 * sy + 8))
    a, b = a[keep], b[keep]
    if len(a) < 12:
        return identity, 0.
    affine, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                                ransacReprojThreshold=2.5)
    quality = float(inliers.mean()) if inliers is not None else 0.
    if affine is None or quality < .5:
        return identity, quality
    transform = np.eye(3)
    transform[:2] = affine
    scale = np.sqrt(np.linalg.det(transform[:2, :2]))
    if not np.isfinite(scale) or not .85 < scale < 1.18:
        return identity, 0.
    # Return full-resolution pixel coordinates.
    scaling = np.diag([sx, sy, 1.])
    return np.linalg.inv(scaling) @ transform @ scaling, quality


def extract_event_signals(run, output, checkpoint, config, device="cuda"):
    run, output, checkpoint = map(lambda p: Path(p).resolve(), (run, output, checkpoint))
    manifest = json.loads((run / "manifest.json").read_text())
    detections = read_jsonl(run / "detections.jsonl")
    index = read_jsonl(run / "frame_index.jsonl")
    if not detections or [r["frame_id"] for r in detections] != [r["frame_id"] for r in index]:
        raise ValueError("Detection/index frames differ or are empty")
    if any(b["source_frame_id"] != a["source_frame_id"] + 1 for a, b in zip(index, index[1:])):
        raise ValueError("Event segmentation needs sequential source frames")
    if sha256_file(manifest["source"]["video"]) != manifest["source"]["video_sha256"]:
        raise ValueError("Source video changed")
    if sha256_file(checkpoint) != CLIP_SHA256:
        raise ValueError("Unexpected CLIP checkpoint")
    keys = ("sample_stride", "hand_score_min", "maximum_object_image_fraction", "maximum_object_hand_area_ratio")
    identity = {"stage": "full_recording_event_evidence", "reference_annotations_used": False,
                "vlm_used": False, "detection_manifest_sha256": sha256_file(run / "manifest.json"),
                "detections_sha256": sha256_file(run / "detections.jsonl"),
                "frame_index_sha256": sha256_file(run / "frame_index.jsonl"),
                "checkpoint_sha256": CLIP_SHA256, "configuration": {k: config[k] for k in keys},
                "implementation_sha256": {p: sha256_file(Path(__file__).with_name(p))
                    for p in ("event_signals.py", "features.py", "objects.py")}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text()) != identity:
            raise ValueError("Evidence inputs changed; use a new output directory")
        if (output / "report.json").exists():
            if sha256_file(output / "signals.jsonl") != json.loads((output / "report.json").read_text())["signals_sha256"]:
                raise ValueError("Cached event signals changed")
            return json.loads((output / "report.json").read_text())
    save_json(output / "manifest.json", identity)
    counts = {side: sum(h["score"] >= config["hand_score_min"] and h["contact_state_id"] in (3, 4)
                       and h["side"] == side for row in detections for h in row["hands"])
              for side in ("left", "right")}
    primary = max(counts, key=counts.get)
    width, height = index[0]["width"], index[0]["height"]
    extractor = ClipImageFeatures(checkpoint, device)
    samples, pending_images, pending_rows = [], [], []
    capture = cv2.VideoCapture(manifest["source"]["video"])
    previous, old_hands, cumulative = None, [], np.eye(3)
    by_id = {r["source_frame_id"]: r for r in detections}
    def flush():
        if pending_images:
            vectors = extractor.encode(pending_images)
            for row, vector in zip(pending_rows, vectors):
                row["target_feature"] = (vector / max(float(np.linalg.norm(vector)), 1e-9)).tolist()
            pending_images.clear()
            pending_rows.clear()
    try:
        for source_id in range(index[-1]["source_frame_id"] + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {source_id}")
            if source_id not in by_id or (source_id - index[0]["source_frame_id"]) % config["sample_stride"]:
                continue
            row = by_id[source_id]
            gray = cv2.cvtColor(cv2.resize(frame, (640, round(640 * height / width))), cv2.COLOR_BGR2GRAY)
            hands = [h for h in row["hands"] if h["score"] >= config["hand_score_min"]] if row["status"] == "ok" else []
            transform, quality = (background_transform(previous, gray, old_hands, hands, width, height)
                                  if previous is not None else (np.eye(3), 1.))
            cumulative = transform @ cumulative
            inverse = np.linalg.inv(cumulative)
            positions = {}
            for side in ("left", "right"):
                choices = [h for h in hands if h["side"] == side]
                if choices:
                    h = max(choices, key=lambda h: h["score"])
                    point = inverse @ np.r_[center(h["bbox_xyxy"]), 1.]
                    positions[side] = {"xy": (point[:2] / [width, height]).tolist(),
                                       "bbox_xyxy": h["bbox_xyxy"], "score": h["score"]}
            contact = choose_contact(row, config, width, height, primary) if row["status"] == "ok" else None
            sample = {"source_frame_id": source_id, "frame_id": row["frame_id"],
                      "relative_seconds": row["relative_seconds"], "timestamp_ns": row["timestamp_ns"],
                      "status": row["status"], "primary_side": primary, "hands": positions,
                      "camera_transform_from_first": cumulative.tolist(), "camera_fit_inlier_fraction": quality,
                      "contact": contact is not None, "reliable_target": bool(contact and contact["reliable_target"]),
                      "target_feature": None, "target_histogram": None, "target_bbox_xyxy": None}
            if contact:
                obj, hand = contact["object"], contact["hand"]
                sample.update(target_bbox_xyxy=obj["bbox_xyxy"], detection_object_id=obj["object_id"],
                              object_image_fraction=contact["object_image_fraction"],
                              object_hand_area_ratio=contact["object_hand_area_ratio"],
                              contact_hand_side=hand["side"])
                if contact["reliable_target"]:
                    pending_images.append(fixed_crop(frame, obj["bbox_xyxy"]))
                    pending_rows.append(sample)
                    sample["target_histogram"] = appearance(frame, obj["bbox_xyxy"])
                    point = inverse @ np.r_[center(obj["bbox_xyxy"]), 1.]
                    sample["target_scene_xy"] = (point[:2] / [width, height]).tolist()
            samples.append(sample)
            if len(pending_images) >= 16:
                flush()
            previous, old_hands = gray, hands
            if len(samples) % 90 == 0:
                print(json.dumps({"stage": "event_evidence", "processed_source_frame": source_id}), flush=True)
        flush()
    finally:
        capture.release()
    write_jsonl(output / "signals.jsonl", samples)
    report = {"sample_count": len(samples), "reliable_target_samples": sum(s["reliable_target"] for s in samples),
              "primary_hand_side": primary, "vlm_used": False, "reference_annotations_used": False,
              "signals_sha256": sha256_file(output / "signals.jsonl")}
    save_json(output / "report.json", report)
    return report
