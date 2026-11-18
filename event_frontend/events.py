"""Event layer above hand/object contact clusters, with no annotation inputs."""
from pathlib import Path
import json
import math

import numpy as np
from scipy.signal import find_peaks

from .detector import sha256_file
from .objects import appearance_distance, area, center
from .video import read_jsonl, save_json, write_jsonl


def event_config(path=None):
    config = json.loads(Path(__file__).with_name("event_config.json").read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        if set(supplied) - set(config):
            raise ValueError("Unknown event configuration keys")
        config.update(supplied)
    for key, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid event setting: {key}")
    for key in ("sample_stride", "minimum_contact_frames"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in ("hand_score_min", "maximum_object_image_fraction", "target_change_histogram_distance", "support_containment_min"):
        if config[key] > 1:
            raise ValueError(f"{key} must be at most 1")
    if config["target_change_distance"] > 2:
        raise ValueError("Cosine distance cannot exceed 2")
    return config


def primary_hand_trajectory(samples, detections, width, height, minimum_score):
    """Associate the primary hand geometrically; classifier flips do not reset it.

    Missing samples stay missing. They must never count as a stationary hand.
    """
    lookup = {r["source_frame_id"]: r for r in detections}
    positions, confident = [], []
    previous, previous_time = None, None
    for sample in samples:
        row = lookup[sample["source_frame_id"]]
        inverse = np.linalg.inv(np.asarray(sample["camera_transform_from_first"]))
        candidates = []
        for hand in row["hands"] if row["status"] == "ok" else []:
            if hand["score"] < minimum_score:
                continue
            point = (inverse @ np.r_[center(hand["bbox_xyxy"]), 1.])[:2] / [width, height]
            probability = hand.get("right_probability", float(hand["side"] == "right"))
            if sample["primary_side"] == "left":
                probability = 1 - probability
            active = previous is not None and sample["relative_seconds"] - previous_time <= .5
            distance = float(np.linalg.norm(point - previous)) if active else None
            cost = distance - .02 * probability if active else -probability
            candidates.append((cost, point, distance))
        if candidates:
            _, point, distance = min(candidates, key=lambda v: v[0])
            good = distance is None or distance <= .25
            positions.append(point)
            confident.append(good)
            previous, previous_time = point, sample["relative_seconds"]
        else:
            positions.append([np.nan, np.nan])
            confident.append(False)
    return np.asarray(positions), np.asarray(confident)


def contact_islands(detections, config):
    """Bridge bounded dropouts, keeping original valid contact frame counts."""
    groups = []
    for row in detections:
        contact = row["status"] == "ok" and any(
            h["score"] >= config["hand_score_min"] and h["contact_state_id"] in (3, 4)
            and h.get("matched_object_id") is not None for h in row["hands"])
        if not contact:
            continue
        if not groups or row["relative_seconds"] - groups[-1][-1]["relative_seconds"] > config["maximum_contact_gap_seconds"]:
            groups.append([])
        groups[-1].append(row)
    return [g for g in groups if len(g) >= config["minimum_contact_frames"]]


def _prototype(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return np.median(np.asarray(values), axis=0)


def _scene_box(sample):
    if not sample["reliable_target"]:
        return None
    x1, y1, x2, y2 = sample["target_bbox_xyxy"]
    corners = np.asarray([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
    mapped = corners @ np.linalg.inv(sample["camera_transform_from_first"]).T
    return [float(mapped[:, 0].min()), float(mapped[:, 1].min()),
            float(mapped[:, 0].max()), float(mapped[:, 1].max())]


def _returned_to_support(before, after, config):
    a = [b for s in before if (b := _scene_box(s)) is not None]
    b = [b for s in after if (b := _scene_box(s)) is not None]
    if not a or not b:
        return False, None
    a, b = np.median(a, axis=0), np.median(b, axis=0)
    ratio = area(b) / max(area(a), 1)
    intersection = area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])
    # A returned object is often replaced by a containing cabinet/drawer box.
    # This is an event's closing phase, not the next target's onset.
    support = ratio >= config["support_expansion_area_ratio"] and intersection / max(area(a), 1) >= config["support_containment_min"]
    return bool(support), float(ratio)


def boundary_evidence(samples, positions, confident, config):
    times = np.asarray([s["relative_seconds"] for s in samples])
    speed = np.full(len(samples), np.nan)
    for i in range(1, len(samples)):
        if confident[i] and confident[i - 1]:
            speed[i] = np.linalg.norm(positions[i] - positions[i - 1]) / (times[i] - times[i - 1])
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.
    def longest_settled(indices):
        longest, current = 0., 0.
        for j in indices:
            current = current + step if np.isfinite(speed[j]) and speed[j] <= config["reset_speed_max"] else 0.
            longest = max(longest, current)
        return longest
    settled_anchors = np.asarray([longest_settled(range(max(0, i - 1), min(len(samples), i + 2))) >= config["minimum_settled_seconds"]
                                  for i in range(len(samples))])
    # Median positions suppress isolated detector jumps; validity is kept separately.
    smoothed = positions.copy()
    for i in range(len(samples)):
        local = positions[max(0, i - 1):i + 2]
        local = local[np.isfinite(local).all(axis=1)]
        if len(local):
            smoothed[i] = np.median(local, axis=0)
    rows, scores = [], np.zeros(len(samples))
    window = config["comparison_window_seconds"]
    for i, sample in enumerate(samples):
        before_ids = np.where((times >= times[i] - window - .1) & (times < times[i] - .1))[0]
        after_ids = np.where((times > times[i] + .1) & (times <= times[i] + window + .1))[0]
        before, after = [samples[j] for j in before_ids], [samples[j] for j in after_ids]
        a, b = _prototype(before, "target_feature"), _prototype(after, "target_feature")
        enough = sum(s["reliable_target"] for s in before) >= max(2, len(before) * .5) and (
                 sum(s["reliable_target"] for s in after) >= max(2, len(after) * .5))
        change = 0.
        histogram = 0.
        distance = 0.
        if a is not None and b is not None and enough:
            change = float(np.clip(1 - a @ b / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)), 0, 2))
            ah, bh = _prototype(before, "target_histogram"), _prototype(after, "target_histogram")
            histogram = appearance_distance(ah / max(ah.sum(), 1e-9), bh / max(bh.sum(), 1e-9))
            ap, bp = _prototype(before, "target_scene_xy"), _prototype(after, "target_scene_xy")
            distance = float(np.linalg.norm(ap - bp))
        reset_ids = np.where(abs(times - times[i]) <= config["reset_window_seconds"])[0]
        valid_speed = [speed[j] for j in reset_ids if np.isfinite(speed[j])]
        reset = longest_settled(reset_ids) >= config["minimum_settled_seconds"]
        # A return requires a real preceding excursion, not just being still.
        history = np.where((times >= times[i] - 4) & (times <= times[i] - .7) & confident & settled_anchors)[0]
        returned = False
        excursion = 0.
        if confident[i] and len(history):
            old = min(history, key=lambda j: np.linalg.norm(smoothed[j] - smoothed[i]))
            mids = smoothed[old:i + 1]
            mids = mids[np.isfinite(mids).all(axis=1)]
            excursion = float(np.max(np.linalg.norm(mids - smoothed[i], axis=1))) if len(mids) else 0.
            returned = (np.linalg.norm(smoothed[old] - smoothed[i]) <= config["return_distance_max"] and
                        excursion >= config["excursion_distance_min"])
        appearance_switch = change >= config["target_change_distance"]
        geometry_switch = distance >= config["target_change_position_distance"]
        histogram_switch = histogram >= config["target_change_histogram_distance"]
        support, expansion = _returned_to_support(before, after, config)
        supported = enough and appearance_switch and (reset or returned) and (geometry_switch or histogram_switch) and not support
        scores[i] = change if supported else 0.
        rows.append({"source_frame_id": sample["source_frame_id"], "relative_seconds": sample["relative_seconds"],
                     "target_cosine_distance": change, "target_histogram_distance": histogram,
                     "target_scene_position_distance": distance, "reliable_windows": bool(enough),
                     "hand_speed_median": float(np.median(valid_speed)) if valid_speed else None,
                     "reset": bool(reset), "returned_after_excursion": bool(returned),
                     "excursion_distance": excursion, "return_to_containing_support": support,
                     "object_area_expansion_ratio": expansion,
                     "supported_target_switch": bool(supported), "release_reapproach": False, "selected": False})
    # Suppress the settling tail of a return-to-support transition as well.
    for i in range(len(samples)):
        if any(row["return_to_containing_support"] and 0 <= times[i] - row["relative_seconds"] <= window
               for row in rows):
            scores[i] = 0.
            rows[i]["supported_target_switch"] = False
    # Nonmaximum suppression in time; the event count is never specified.
    peaks, _ = find_peaks(scores, height=config["target_change_distance"])
    selected = []
    for i in sorted(peaks, key=lambda j: (-scores[j], samples[j]["source_frame_id"])):
        if all(abs(times[i] - times[j]) >= config["minimum_boundary_gap_seconds"] for j in selected):
            selected.append(int(i))
    return rows, selected


def release_reapproach_boundaries(samples, detections, decisions, config):
    """Separate a completed manipulation from a new target after withdrawing.

    A target dropout alone (occlusion/ID loss) is insufficient: require an actual
    contact release and changed foreground appearance on reapproach.
    """
    groups = []
    for i, sample in enumerate(samples):
        if not sample["reliable_target"]:
            if not groups or i != groups[-1][-1] + 1:
                groups.append([])
            groups[-1].append(i)
    sample_dt = float(np.median(np.diff([s["relative_seconds"] for s in samples]))) if len(samples) > 1 else 0.
    source_dt = float(np.median(np.diff([s["relative_seconds"] for s in detections]))) if len(detections) > 1 else 0.
    found = []
    for group in groups:
        first, last = group[0], group[-1]
        if first == 0 or last + 1 == len(samples):
            continue
        a, b = samples[first]["relative_seconds"], samples[last]["relative_seconds"]
        if b - a + sample_dt < config["target_absence_reset_seconds"]:
            continue
        before = [s for s in samples if a - config["comparison_window_seconds"] <= s["relative_seconds"] < a]
        after = [s for s in samples if b < s["relative_seconds"] <= b + config["comparison_window_seconds"] + sample_dt]
        av, bv = _prototype(before, "target_feature"), _prototype(after, "target_feature")
        if av is None or bv is None:
            continue
        change = float(np.clip(1 - av @ bv / max(1e-12, np.linalg.norm(av) * np.linalg.norm(bv)), 0, 2))
        longest, current = 0., 0.
        for row in detections:
            if not a <= row["relative_seconds"] <= b:
                continue
            contact = row["status"] == "ok" and any(h["score"] >= config["hand_score_min"] and h["contact_state_id"] in (3, 4)
                       and h.get("matched_object_id") is not None for h in row["hands"])
            # Processing errors never count as observed hand release.
            current = current + source_dt if row["status"] == "ok" and not contact else 0.
            longest = max(longest, current)
        i = last + 1
        decisions[i].update(target_absence_seconds=b - a + sample_dt,
                            observed_release_seconds=longest, gap_target_cosine_distance=change)
        if longest >= config["minimum_release_seconds"] and change >= config["target_change_distance"]:
            decisions[i]["release_reapproach"] = True
            found.append(i)
    return found


def segment_from_signals(samples, detections, width, height, config):
    if not samples or not detections:
        return [], []
    if any(b["relative_seconds"] <= a["relative_seconds"] for a, b in zip(samples, samples[1:])):
        raise ValueError("Evidence samples must be in strictly increasing time order")
    positions, confident = primary_hand_trajectory(samples, detections, width, height, config["hand_score_min"])
    decisions, possible = boundary_evidence(samples, positions, confident, config)
    possible = sorted(set(possible) | set(release_reapproach_boundaries(samples, detections, decisions, config)))
    events = []
    for island in contact_islands(detections, config):
        start, end = island[0]["source_frame_id"], island[-1]["source_frame_id"]
        boundaries = [i for i in possible if start < samples[i]["source_frame_id"] <= end]
        # Keep supported boundaries only if both adjacent episodes have enough evidence.
        chosen = []
        for i in sorted(boundaries):
            boundary = samples[i]["source_frame_id"]
            previous = samples[chosen[-1]]["source_frame_id"] if chosen else start
            prefix = [r for r in island if previous <= r["source_frame_id"] < boundary]
            suffix = [r for r in island if boundary <= r["source_frame_id"] <= end]
            if len(prefix) >= config["minimum_contact_frames"] and len(suffix) >= config["minimum_contact_frames"] and (
                    prefix[-1]["relative_seconds"] - prefix[0]["relative_seconds"] >= config["minimum_event_seconds"] and
                    suffix[-1]["relative_seconds"] - suffix[0]["relative_seconds"] >= config["minimum_event_seconds"]):
                chosen.append(i)
                decisions[i]["selected"] = True
        edges = [start] + [samples[i]["source_frame_id"] for i in chosen] + [end + 1]
        for first, following in zip(edges, edges[1:]):
            valid = [r for r in island if first <= r["source_frame_id"] < following]
            events.append({"event_id": f"event_{len(events) + 1:03d}",
                           "start_source_frame_id": valid[0]["source_frame_id"],
                           "end_source_frame_id": valid[-1]["source_frame_id"],
                           "start_seconds": valid[0]["relative_seconds"], "end_seconds": valid[-1]["relative_seconds"],
                           "contact_frame_count": len(valid),
                           "boundary_reason": "recording_or_contact_island_start" if first == start else (
                               "release_then_changed_target_reapproach" if next(d for d in decisions if d["source_frame_id"] == first)["release_reapproach"]
                               else "persistent_target_change_with_hand_reset"),
                           "end_truncated_by_recording": valid[-1]["source_frame_id"] == detections[-1]["source_frame_id"],
                           "status": "automatic_hypothesis", "manually_verified": False})
    return events, decisions


def segment_events(run, evidence, output, config):
    run, evidence, output = map(lambda p: Path(p).resolve(), (run, evidence, output))
    evidence_manifest = json.loads((evidence / "manifest.json").read_text())
    evidence_report = json.loads((evidence / "report.json").read_text())
    for field, name in (("detection_manifest_sha256", "manifest.json"), ("detections_sha256", "detections.jsonl"),
                         ("frame_index_sha256", "frame_index.jsonl")):
        if evidence_manifest[field] != sha256_file(run / name):
            raise ValueError("Event evidence does not match this recording")
    if evidence_report["signals_sha256"] != sha256_file(evidence / "signals.jsonl"):
        raise ValueError("Event signals changed after extraction")
    if any(evidence_manifest["configuration"][k] != config[k] for k in evidence_manifest["configuration"]):
        raise ValueError("Evidence configuration differs; re-extract signals")
    identity = {"stage": "full_recording_event_segmentation", "reference_annotations_used": False,
                "vlm_used": False, "evidence_manifest_sha256": sha256_file(evidence / "manifest.json"),
                "signals_sha256": evidence_report["signals_sha256"], "configuration": config,
                "implementation_sha256": sha256_file(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Event inputs changed; use a new output directory")
    save_json(output / "manifest.json", identity)
    detections = read_jsonl(run / "detections.jsonl")
    index = read_jsonl(run / "frame_index.jsonl")
    samples = read_jsonl(evidence / "signals.jsonl")
    events, decisions = segment_from_signals(samples, detections, index[0]["width"], index[0]["height"], config)
    save_json(output / "predictions.json", {"reference_annotations_used": False, "vlm_used": False, "predictions": events})
    write_jsonl(output / "boundary_decisions.jsonl", decisions)
    report = {"event_count": len(events), "source_frame_count": len(detections), "reference_annotations_used": False,
              "detection_manifest_sha256": sha256_file(run / "manifest.json"),
              "detections_sha256": sha256_file(run / "detections.jsonl"),
              "vlm_used": False, "predictions_sha256": sha256_file(output / "predictions.json"), "manually_verified": False}
    save_json(output / "report.json", report)
    return report
