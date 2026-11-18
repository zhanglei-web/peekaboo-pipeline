"""Contact-crop YOLO semantics and category-independent object association."""
from collections import Counter
from pathlib import Path
import json
import math
import os

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .detector import sha256_file
from .video import read_jsonl, save_json, write_jsonl


def object_config(path=None):
    config = json.loads(Path(__file__).with_name("object_config.json").read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        if set(supplied) - set(config):
            raise ValueError("Unknown object configuration keys")
        config.update(supplied)
    if not config["categories"] or any(not isinstance(c, str) or not c.strip() for c in config["categories"]):
        raise ValueError("Provide nonempty category strings")
    if len(set(config["categories"])) != len(config["categories"]):
        raise ValueError("Category names must be unique")
    for key in ("confidence", "category_match_iou_min", "track_iou_min", "track_appearance_distance_max"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    if config["crop_padding"] < 0 or config["track_center_distance_max"] <= 0 or config["track_area_ratio_max"] < 1:
        raise ValueError("Invalid crop or tracking geometry")
    for key in ("image_size", "track_max_missing_frames"):
        if not isinstance(config[key], int) or config[key] < (1 if key == "image_size" else 0):
            raise ValueError(f"Invalid {key}")
    return config


def area(box):
    return max(0., box[2] - box[0] + 1) * max(0., box[3] - box[1] + 1)


def center(box):
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def diagonal(box):
    return max(1., math.hypot(box[2] - box[0] + 1, box[3] - box[1] + 1))


def iou(a, b):
    intersection = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    return intersection / max(area(a) + area(b) - intersection, 1e-9)


def crop(frame, box, padding=0):
    height, width = frame.shape[:2]
    dx, dy = (box[2] - box[0] + 1) * padding, (box[3] - box[1] + 1) * padding
    x1, y1 = max(0, math.floor(box[0] - dx)), max(0, math.floor(box[1] - dy))
    x2, y2 = min(width, math.ceil(box[2] + dx + 1)), min(height, math.ceil(box[3] + dy + 1))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty object crop")
    return frame[y1:y2, x1:x2], (x1, y1)


def appearance(frame, box):
    image, _ = crop(frame, box)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
    return (hist / max(float(hist.sum()), 1)).tolist()


def appearance_distance(a, b):
    # Hellinger distance, bounded in [0,1].
    return float(np.linalg.norm(np.sqrt(a) - np.sqrt(b)) / math.sqrt(2))


class ObjectTracker:
    """One-to-one IoU/position/HSV association; class labels never gate IDs.

    IDs are local to this run. Long occlusions/crossings can still split or swap
    identities; this deliberately is not a ReID model.
    """
    def __init__(self, config):
        self.config, self.tracks, self.next_id = config, {}, 1

    def update(self, frame_id, observations):
        config = self.config
        active = [t for t in self.tracks.values()
                  if frame_id - t["last_frame"] - 1 <= config["track_max_missing_frames"]]
        costs = np.full((len(active), len(observations)), 1e6)
        metrics = {}
        for i, track in enumerate(active):
            for j, obj in enumerate(observations):
                overlap = iou(track["bbox_xyxy"], obj["bbox_xyxy"])
                distance = float(np.linalg.norm(center(track["bbox_xyxy"]) - center(obj["bbox_xyxy"]))
                                 / diagonal(track["bbox_xyxy"]))
                visual = appearance_distance(track["appearance"], obj["appearance"])
                ratio = max(area(track["bbox_xyxy"]), area(obj["bbox_xyxy"])) / max(1, min(area(track["bbox_xyxy"]), area(obj["bbox_xyxy"])))
                metrics[i, j] = {"iou": overlap, "center_distance": distance, "appearance_distance": visual}
                if ratio <= config["track_area_ratio_max"] and visual <= config["track_appearance_distance_max"] and (
                        overlap >= config["track_iou_min"] or distance <= config["track_center_distance_max"]):
                    costs[i, j] = .55 * (1 - overlap) + .25 * visual + .20 * min(1, distance)
        assignments = {}
        if costs.size:
            ii, jj = linear_sum_assignment(costs)
            assignments = {int(j): (active[i], metrics[i, j]) for i, j in zip(ii, jj) if costs[i, j] < 1e5}
        for j, obj in enumerate(observations):
            if j in assignments:
                track, match = assignments[j]
                obj["track_match"] = {"method": "iou_position_hsv", **match}
            else:
                identity = f"obj_{self.next_id:05d}"
                self.next_id += 1
                track = {"object_id": identity, "first_frame": frame_id, "category_votes": Counter(), "observations": 0}
                self.tracks[identity] = track
                obj["track_match"] = {"method": "new_instance", "iou": None,
                                      "center_distance": None, "appearance_distance": None}
            if obj["category"] != "unknown":
                track["category_votes"][obj["category"]] += obj["category_score"]
            track.update(last_frame=frame_id, bbox_xyxy=obj["bbox_xyxy"], appearance=obj["appearance"])
            track["observations"] += 1
            obj["object_id"] = track["object_id"]
        return observations


class YoloContactRecognizer:
    def __init__(self, weights, config, device="cuda"):
        import torch
        # Keep Ultralytics' mutable settings separate from other projects.
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".cache/pnr-ultralytics"))
        import ultralytics
        from ultralytics import YOLO, YOLOWorld

        self.weights = Path(weights).resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(self.weights)
        torch.set_num_threads(4)
        self.config, self.device = config, device
        self.model = YOLOWorld(str(self.weights)) if "world" in self.weights.name.lower() else YOLO(str(self.weights))
        self.model.to(device)
        if isinstance(self.model, YOLOWorld):
            self.model.set_classes(config["categories"])
        self.metadata = {"weights": str(self.weights), "weights_sha256": sha256_file(self.weights),
                         "ultralytics": ultralytics.__version__, "device": device,
                         "class_names": self.model.names,
                         "open_vocabulary": isinstance(self.model, YOLOWorld),
                         "unrecognized_category": "unknown", "configuration": config}

    def predict(self, frame, objects):
        if not objects:
            return []
        crops = [crop(frame, o["bbox_xyxy"], self.config["crop_padding"]) for o in objects]
        results = self.model.predict([image for image, _ in crops], device=self.device,
                                     conf=self.config["confidence"], imgsz=self.config["image_size"],
                                     verbose=False, save=False)
        recognized = []
        for obj, (_, origin), result in zip(objects, crops, results):
            choices = []
            for box, score, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(),
                                             result.boxes.cls.cpu().tolist()):
                box = [box[0] + origin[0], box[1] + origin[1], box[2] + origin[0], box[3] + origin[1]]
                overlap = iou(box, obj["bbox_xyxy"])
                if overlap >= self.config["category_match_iou_min"]:
                    choices.append((score * overlap, score, result.names[int(class_id)], box, overlap))
            if choices:
                _, score, category, box, overlap = max(choices, key=lambda c: c[0])
                recognized.append({"category": category, "category_score": score, "category_bbox_xyxy": box,
                                   "category_match_iou": overlap, "category_status": "predicted"})
            else:
                recognized.append({"category": "unknown", "category_score": 0., "category_bbox_xyxy": None,
                                   "category_match_iou": None, "category_status": "no_matching_yolo_detection"})
        return recognized


def recognize_objects(run, output, recognizer, config):
    run, output = Path(run).resolve(), Path(output).resolve()
    rows = read_jsonl(run / "detections.jsonl")
    if not rows:
        raise ValueError("No detections")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("input_scope", manifest["source"].get("input_scope")) != "single_interaction_clip":
        raise ValueError("Recognition requires one interaction clip")
    if any(b["source_frame_id"] != a["source_frame_id"] + 1 for a, b in zip(rows, rows[1:])):
        raise ValueError("Clip detection rows must be sequential")
    source = manifest["source"]
    if sha256_file(source["video"]) != source["video_sha256"]:
        raise ValueError("Source video changed")
    identity = {"detection_manifest_sha256": sha256_file(run / "manifest.json"),
                "detections_sha256": sha256_file(run / "detections.jsonl"),
                "recognizer": recognizer.metadata, "configuration": config,
                "implementation_sha256": sha256_file(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Object-stage inputs changed; use a new output directory")
    save_json(output / "manifest.json", identity)
    cache = output / "frames"
    cache.mkdir(exist_ok=True)
    tracker, tracked = ObjectTracker(config), []
    capture = cv2.VideoCapture(rows[0]["source_video"])
    try:
        # Decode from the beginning to preserve original ordinal identities for scoped clips.
        for source_id in range(rows[0]["source_frame_id"]):
            if not capture.read()[0]:
                raise ValueError(f"Cannot reach clip start; decode failed at {source_id}")
        for row in rows:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Sequential decode failed at {row['source_frame_id']}")
            objects = row["objects"] if row["status"] == "ok" else []
            contact_ids = {h["matched_object_id"] for h in row["hands"] if h["contact_state_id"] in (3, 4)}
            contacts = [o for o in objects if o["object_id"] in contact_ids]
            path = cache / f'f{row["source_frame_id"]:06d}.json'
            if path.exists():
                entry = json.loads(path.read_text())
                if entry["frame_id"] != row["frame_id"] or entry["source_frame_id"] != row["source_frame_id"]:
                    raise ValueError("Cached recognition frame mismatch")
                observations = entry["observations"]
            else:
                labels = dict(zip([o["object_id"] for o in contacts], recognizer.predict(frame, contacts)))
                observations = [{"detection_object_id": o["object_id"], "bbox_xyxy": o["bbox_xyxy"],
                                 "score": o["score"], "appearance": appearance(frame, o["bbox_xyxy"]),
                                 **labels.get(o["object_id"], {"category": "unknown", "category_score": 0.,
                                                             "category_status": "not_in_contact"})} for o in objects]
                save_json(path, {"frame_id": row["frame_id"], "source_frame_id": row["source_frame_id"],
                                 "observations": observations})
            observations = tracker.update(row["source_frame_id"], observations)
            mapping = {o["detection_object_id"]: o["object_id"] for o in observations}
            hands = [{**h, "object_id": mapping.get(h["matched_object_id"])} for h in row["hands"]
                     if h["contact_state_id"] in (3, 4)]
            tracked.append({"frame_id": row["frame_id"], "source_frame_id": row["source_frame_id"],
                            "timestamp_ns": row["timestamp_ns"], "relative_seconds": row["relative_seconds"],
                            "objects": observations, "contacts": hands, "status": row["status"]})
            if (row["source_frame_id"] + 1) % 100 == 0:
                print(json.dumps({"stage": "recognize_track", "processed_frames": len(tracked), "total": len(rows)}), flush=True)
    finally:
        capture.release()
    summaries = []
    for t in tracker.tracks.values():
        votes = dict(t["category_votes"])
        label = max(votes, key=votes.get) if votes else "unknown"
        summaries.append({k: t[k] for k in ("object_id", "first_frame", "last_frame", "observations")} |
                         {"object_category": label, "category_votes": votes})
    write_jsonl(output / "tracked_objects.jsonl", tracked)
    save_json(output / "tracks.json", summaries)
    report = {"source_frames": len(rows), "object_tracks": len(summaries),
              "recognized_contact_observations": sum(o["category"] != "unknown" for r in tracked for o in r["objects"]),
              "contact_hands_without_object": sum(h["object_id"] is None for r in tracked for h in r["contacts"]),
              "category_counts": dict(Counter(o["category"] for r in tracked for o in r["objects"] if o["category_status"] != "not_in_contact")),
              "manually_verified": False}
    save_json(output / "object_report.json", report)
    return report
