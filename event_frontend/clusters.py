"""One interaction clip: temporal validation, largest valid object cluster, sampling."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import math

import cv2
import numpy as np

from .candidates import labeled_image
from .detector import sha256_file
from .objects import center, diagonal, iou
from .targets import contact_target_quality, refine_contact_targets, supported_contact_gap
from .video import read_jsonl, save_json, write_jsonl


def pipeline_config(path=None):
    config = json.loads(Path(__file__).with_name("pipeline_config.json").read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        if set(supplied) - set(config):
            raise ValueError("Unknown pipeline configuration keys")
        config.update(supplied)
    for key in ("hand_score_min", "contact_probability_min", "association_iou_min", "minimum_valid_fraction",
                "category_min_observation_fraction", "category_vote_share_min", "target_quality_maximum_penalty",
                "nested_target_containment_min", "supported_contact_gap_iou_min", "supported_contact_gap_appearance_max",
                "supported_contact_gap_hand_iou_min"):
        if not 0 <= config[key] <= 1:
            raise ValueError(f"{key} must be in [0,1]")
    for key in ("minimum_valid_contact_frames", "motion_inconsistency_run", "image_max_edge", "max_frames_per_packet", "category_min_observations",
                "nested_target_minimum_support_frames"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["max_frames_per_packet"] < 4:
        raise ValueError("Need at least four samples for context and cluster boundaries")
    if not isinstance(config["maximum_missing_contact_frames"], int) or config["maximum_missing_contact_frames"] < 0:
        raise ValueError("Missing contact allowance must be a nonnegative integer")
    for key in ("context_before_seconds", "context_after_seconds", "relative_speed_threshold", "hand_object_gap_max"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"Invalid {key}")
    if not isinstance(config["allow_short_cluster_fallback"], bool):
        raise ValueError("allow_short_cluster_fallback must be boolean")
    if config["target_selection_strategy"] not in ("largest_valid", "contact_quality"):
        raise ValueError("Unknown target selection strategy")
    for key in ("target_quality_area_ratio_reference", "nested_target_area_ratio_min",
                "nested_target_hand_area_ratio_max", "supported_contact_gap_seconds", "supported_contact_gap_containing_area_ratio_min"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}")
    return config


def bbox_gap(hand, obj):
    dx = max(hand[0] - obj[2], obj[0] - hand[2], 0)
    dy = max(hand[1] - obj[3], obj[1] - hand[3], 0)
    return math.hypot(dx, dy) / diagonal(obj)


def contact_clusters(rows, config):
    """Count source frames, not hand detections or interpolated gap frames."""
    by_object, decisions = defaultdict(list), []
    rows_by_id = {r["source_frame_id"]: r for r in rows}
    quality_mode = config["target_selection_strategy"] == "contact_quality"
    for row in rows:
        objects = {o["object_id"]: o for o in row["objects"]}
        grouped = defaultdict(list)
        for hand in row["contacts"]:
            reason = None
            if row["status"] != "ok":
                reason = "processing_error"
            elif hand["score"] < config["hand_score_min"]:
                reason = "low_hand_score"
            elif hand["contact_probabilities"][hand["contact_state_id"]] < config["contact_probability_min"]:
                reason = "low_contact_probability"
            elif hand.get("object_id") not in objects:
                reason = "object_unmatched"
            if reason:
                decisions.append({"source_frame_id": row["source_frame_id"], "hand_id": hand["hand_id"],
                                  "object_id": hand.get("object_id"), "valid_contact": False, "reason": reason})
            else:
                grouped[hand["object_id"]].append(hand)
        for object_id, hands in grouped.items():
            obj = objects[object_id]
            by_object[object_id].append({"source_frame_id": row["source_frame_id"], "frame_id": row["frame_id"],
                                        "timestamp_ns": row["timestamp_ns"], "relative_seconds": row["relative_seconds"],
                                        "object": obj, "hands": hands})
    clusters = []
    for object_id, contacts in by_object.items():
        groups, bridges = [], []
        for contact in contacts:
            bridge = None
            separated = groups and contact["source_frame_id"] - groups[-1][-1]["source_frame_id"] - 1 > config["maximum_missing_contact_frames"]
            if separated and quality_mode:
                bridge = supported_contact_gap(groups[-1][-1], contact, rows_by_id, config)
            if not groups or separated and bridge is None:
                groups.append([])
                bridges.append([])
            if bridge is not None:
                bridges[-1].append(bridge)
            groups[-1].append(contact)
        for group, gap_evidence in zip(groups, bridges):
            previous = None
            for contact in group:
                obj, hands = contact["object"], contact["hands"]
                metrics = []
                for hand in hands:
                    gap = bbox_gap(hand["bbox_xyxy"], obj["bbox_xyxy"])
                    speed = None
                    hand_continuous = False
                    if previous:
                        # Match each hand geometrically; a left/right classifier flip does not reset motion.
                        old_hand = max(previous["hands"], key=lambda h: iou(h["bbox_xyxy"], hand["bbox_xyxy"]))
                        hand_overlap = iou(old_hand["bbox_xyxy"], hand["bbox_xyxy"])
                        hand_continuous = hand_overlap >= config["association_iou_min"]
                        dt = (int(contact["timestamp_ns"]) - int(previous["timestamp_ns"])) / 1e9
                        if dt <= 0:
                            raise ValueError("Contact timestamps must strictly increase")
                        relative = ((center(hand["bbox_xyxy"]) - center(old_hand["bbox_xyxy"])) -
                                    (center(obj["bbox_xyxy"]) - center(previous["object"]["bbox_xyxy"])))
                        scale = (diagonal(obj["bbox_xyxy"]) + diagonal(previous["object"]["bbox_xyxy"])) / 2
                        speed = float(np.linalg.norm(relative) / scale / dt)
                    match = obj["track_match"]
                    association_unstable = previous is not None and (match.get("iou") is None or
                        match["iou"] < config["association_iou_min"]) and not hand_continuous
                    # High relative speed alone is never an invalidation criterion.
                    suspicious = gap > config["hand_object_gap_max"] or (
                        speed is not None and speed > config["relative_speed_threshold"] and association_unstable)
                    metrics.append({"hand_id": hand["hand_id"], "relative_speed_normalized_per_second": speed,
                                    "hand_object_gap_normalized": gap, "association_unstable": association_unstable,
                                    "suspicious": suspicious})
                quality = [contact_target_quality(h, obj, config) for h in hands]
                contact.update(hand_motion=metrics, suspicious=all(m["suspicious"] for m in metrics), valid_contact=True,
                               target_quality=max(quality, key=lambda q: q["weight"]))
                previous = contact
            # Reject only sustained inconsistent runs; one bad transition remains reviewable.
            suspicious_groups = []
            for i, contact in enumerate(group):
                if contact["suspicious"]:
                    if not suspicious_groups or i != suspicious_groups[-1][-1] + 1:
                        suspicious_groups.append([])
                    suspicious_groups[-1].append(i)
            for indices in suspicious_groups:
                if len(indices) >= config["motion_inconsistency_run"]:
                    for i in indices:
                        group[i]["valid_contact"] = False
            valid = [c for c in group if c["valid_contact"]]
            fraction = len(valid) / len(group)
            short = len(valid) < config["minimum_valid_contact_frames"]
            status = "invalid" if not valid or fraction < config["minimum_valid_fraction"] else (
                "low_confidence_short" if short else "valid")
            votes = Counter()
            counts = Counter()
            for contact in valid:
                obj = contact["object"]
                if obj["category"] != "unknown":
                    votes[obj["category"]] += obj["category_score"]
                    counts[obj["category"]] += 1
            dominant = max(votes, key=votes.get) if votes else "unknown"
            supported = bool(votes) and counts[dominant] >= config["category_min_observations"] and (
                counts[dominant] / max(1, len(valid)) >= config["category_min_observation_fraction"] and
                votes[dominant] / sum(votes.values()) >= config["category_vote_share_min"])
            category = dominant if supported else "unknown"
            weighted_count = sum(c["target_quality"]["weight"] for c in valid)
            known_quality = [c["target_quality"] for c in valid if c["target_quality"]["evidence_available"]]
            ambiguous_fraction = sum(q["extent_ambiguous"] for q in known_quality) / max(1, len(known_quality))
            cluster_id = f"c{len(clusters):04d}"
            clusters.append({"cluster_id": cluster_id, "object_id": object_id, "object_category": category,
                             "category_votes": dict(votes), "start_source_frame_id": group[0]["source_frame_id"],
                             "category_observation_counts": dict(counts), "category_supported": supported,
                             "end_source_frame_id": group[-1]["source_frame_id"],
                             "start_seconds": group[0]["relative_seconds"], "end_seconds": group[-1]["relative_seconds"],
                             "contact_frame_count": len(group), "valid_contact_frame_count": len(valid),
                             "valid_fraction": fraction, "missing_contact_frames": group[-1]["source_frame_id"] - group[0]["source_frame_id"] + 1 - len(group),
                             "valid_source_frame_ids": [c["source_frame_id"] for c in valid],
                             "quality_weighted_contact_frames": weighted_count,
                             "target_selection_strategy": config["target_selection_strategy"],
                             "target_quality_available_frames": len(known_quality),
                             "target_extent_ambiguous_fraction": ambiguous_fraction,
                             "target_extent_review_required": bool(known_quality) and ambiguous_fraction >= .5,
                             "supported_gap_bridges": gap_evidence,
                             "status": status, "selected": False})
            for contact in group:
                decisions.append({"source_frame_id": contact["source_frame_id"], "frame_id": contact["frame_id"],
                                  "object_id": object_id, "cluster_id": cluster_id,
                                  "valid_contact": contact["valid_contact"], "suspicious": contact["suspicious"],
                                  "motion": contact["hand_motion"],
                                  "target_quality": contact["target_quality"],
                                  "reason": "sustained_inconsistent_association" if not contact["valid_contact"] else (
                                      "transient_inconsistency_retained" if contact["suspicious"] else "valid_contact")})
    eligible = [c for c in clusters if c["status"] == "valid"]
    if not eligible and config["allow_short_cluster_fallback"]:
        eligible = [c for c in clusters if c["status"] == "low_confidence_short"]
    selected = max(eligible, key=lambda c: (c["quality_weighted_contact_frames"] if quality_mode else c["valid_contact_frame_count"],
                                          c["valid_contact_frame_count"], c["valid_fraction"], -c["start_source_frame_id"])) if eligible else None
    if selected:
        selected["selected"] = True
    return clusters, selected, sorted(decisions, key=lambda d: d["source_frame_id"])


def sample_cluster(index, selected, config):
    if selected is None:
        return [], None
    start = selected["start_seconds"] - config["context_before_seconds"]
    end = selected["end_seconds"] + config["context_after_seconds"]
    available = [r for r in index if start <= r["relative_seconds"] <= end and r.get("status", "ok") == "ok"]
    if not available:
        return [], None
    # Preserve clip/context edges and both contact boundaries, then fill uniformly in time.
    lookup = {r["source_frame_id"]: r for r in available}
    anchors = {available[0]["source_frame_id"], available[-1]["source_frame_id"],
               selected["start_source_frame_id"], selected["end_source_frame_id"]} & set(lookup)
    selected_ids = set(anchors)
    while len(selected_ids) < min(config["max_frames_per_packet"], len(available)):
        row = max((r for r in available if r["source_frame_id"] not in selected_ids),
                  key=lambda r: min(abs(r["relative_seconds"] - lookup[i]["relative_seconds"]) for i in selected_ids))
        selected_ids.add(row["source_frame_id"])
    context = {"requested_seconds": [start, end], "available_source_range": [available[0]["source_frame_id"], available[-1]["source_frame_id"]],
               "truncated_at_clip_start": start < index[0]["relative_seconds"],
               "truncated_at_clip_end": end > index[-1]["relative_seconds"]}
    return [lookup[i] for i in sorted(selected_ids)], context


def prepare_cluster_candidates(run, objects_run, output, config):
    run, objects_run, output = map(lambda p: Path(p).resolve(), (run, objects_run, output))
    rows = read_jsonl(run / "detections.jsonl")
    detection_manifest = json.loads((run / "manifest.json").read_text())
    if detection_manifest.get("input_scope", detection_manifest["source"].get("input_scope")) != "single_interaction_clip":
        raise ValueError("Preparation requires one interaction clip")
    tracked = read_jsonl(objects_run / "tracked_objects.jsonl")
    object_manifest = json.loads((objects_run / "manifest.json").read_text())
    if object_manifest["detections_sha256"] != sha256_file(run / "detections.jsonl") or object_manifest["detection_manifest_sha256"] != sha256_file(run / "manifest.json"):
        raise ValueError("Recognition and detection inputs differ")
    if [r["frame_id"] for r in rows] != [r["frame_id"] for r in tracked]:
        raise ValueError("Tracked frames do not match detection clip")
    identity = {"pipeline": "single_interaction_clip_contact_quality_cluster" if config["target_selection_strategy"] == "contact_quality" else "single_interaction_clip_largest_valid_object_cluster", "detection_manifest_sha256": sha256_file(run / "manifest.json"),
                "detections_sha256": sha256_file(run / "detections.jsonl"),
                "object_manifest_sha256": sha256_file(objects_run / "manifest.json"),
                "tracked_objects_sha256": sha256_file(objects_run / "tracked_objects.jsonl"), "config": config,
                "implementation_sha256": {p: sha256_file(Path(__file__).with_name(p)) for p in ("clusters.py", "targets.py", "candidates.py", "objects.py")}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Cluster inputs changed; use a new output directory")
    save_json(output / "manifest.json", identity)
    refinement = []
    if config["target_selection_strategy"] == "contact_quality":
        tracked, refinement = refine_contact_targets(tracked, config)
    clusters, selected, decisions = contact_clusters(tracked, config)
    wanted, context = sample_cluster(rows, selected, config)
    images = output / "images"
    images.mkdir(exist_ok=True)
    samples = []
    if wanted:
        tracked_by_id = {r["source_frame_id"]: r for r in tracked}
        wanted_ids = {r["source_frame_id"]: r for r in wanted}
        capture = cv2.VideoCapture(rows[0]["source_video"])
        try:
            for source_id in range(wanted[-1]["source_frame_id"] + 1):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode source frame {source_id}")
                if source_id not in wanted_ids:
                    continue
                row = wanted_ids[source_id]
                path = images / f"f{source_id:06d}.jpg"
                obj = next((o for o in tracked_by_id[source_id]["objects"] if o["object_id"] == selected["object_id"]), None)
                target_bbox = obj["bbox_xyxy"] if obj else None
                if target_bbox:
                    x1, y1, x2, y2 = map(round, target_bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 50), 3)
                    cv2.putText(frame, selected["object_id"], (x1, max(20, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 230, 50), 2)
                labeled_image(frame, row, config["image_max_edge"]).save(path, quality=94)
                samples.append({k: row[k] for k in ("frame_id", "source_frame_id", "timestamp_ns", "relative_seconds", "camera", "source_video", "width", "height")} |
                               {"image": str(path), "image_sha256": sha256_file(path),
                                "target_bbox_xyxy": target_bbox,
                                "reason": "cluster_contact" if source_id in selected["valid_source_frame_ids"] else "context_or_gap"})
        finally:
            capture.release()
    packets = [{"packet_id": "p0000", "recording_id": rows[0]["recording_id"],
                "clip_id": detection_manifest.get("clip_id", Path(rows[0]["source_video"]).stem), "frames": samples,
                "source_range": context["available_source_range"], "clip_source_range": [rows[0]["source_frame_id"], rows[-1]["source_frame_id"]],
                "target": selected, "context": context, "maximum_events": 1,
                "target_boxes": {r["frame_id"]: next((o["bbox_xyxy"] for o in r["objects"] if o["object_id"] == selected["object_id"]), None) for r in tracked},
                "max_frames_per_packet": config["max_frames_per_packet"]}] if len(samples) >= 3 else []
    save_json(output / "clusters.json", clusters)
    save_json(output / "selected_cluster.json", selected)
    write_jsonl(output / "filter_decisions.jsonl", decisions)
    write_jsonl(output / "target_decisions.jsonl", refinement)
    write_jsonl(output / "candidates.jsonl", samples)
    write_jsonl(output / "packets.jsonl", packets)
    report = {"input_scope": "single_interaction_clip", "clip_source_range": [rows[0]["source_frame_id"], rows[-1]["source_frame_id"]],
              "cluster_count": len(clusters), "cluster_status_counts": dict(Counter(c["status"] for c in clusters)),
              "selected_cluster": selected, "context": context, "selected_frames": len(samples), "packet_count": len(packets),
              "target_selection_strategy": config["target_selection_strategy"],
              "reassociated_contacts": sum(d["changed"] for d in refinement),
              "supported_gap_bridge_count": sum(len(c["supported_gap_bridges"]) for c in clusters),
              "status": "ready" if packets else "no_valid_cluster_or_insufficient_frames", "manually_verified": False}
    save_json(output / "candidate_report.json", report)
    write_cluster_review(output, tracked, clusters, selected, report)
    return report


def write_cluster_review(output, rows, clusters, selected, report):
    import html
    from PIL import Image, ImageDraw
    width, lane = 1200, 32
    image = Image.new("RGB", (width, max(100, 55 + lane * len(clusters))), "white")
    draw = ImageDraw.Draw(image)
    start, end = rows[0]["source_frame_id"], rows[-1]["source_frame_id"]
    def x(frame):
        return 230 + (frame - start) / max(1, end - start) * 950
    draw.text((12, 10), f"SINGLE INTERACTION CLIP | original frames {start}-{end}", fill="black")
    for i, c in enumerate(clusters):
        y = 42 + lane * i
        color = "#059669" if c["selected"] else ("#dc2626" if c["status"] == "invalid" else "#d97706")
        draw.text((8, y), f'{c["cluster_id"]} {c["object_id"]} {c["object_category"]} ({c["valid_contact_frame_count"]})', fill=color)
        draw.line((x(c["start_source_frame_id"]), y + 8, x(c["end_source_frame_id"]), y + 8), fill=color, width=8)
    image.save(output / "cluster_timeline.png")
    samples = read_jsonl(output / "candidates.jsonl")
    tiles = []
    for r in samples:
        with Image.open(r["image"]) as original:
            tile = original.copy()
            tile.thumbnail((360, 250))
            tiles.append(tile)
    if tiles:
        montage = Image.new("RGB", (1440, math.ceil(len(tiles) / 4) * 250), "#e5e7eb")
        for i, tile in enumerate(tiles):
            montage.paste(tile, ((i % 4) * 360, (i // 4) * 250))
        montage.save(output / "candidate_montage.jpg", quality=94)
    (output / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>单次交互片段候选复核</title>'
        '<h1>单次交互片段：接触簇选择</h1><p>绿色为选中簇，红色为无效簇，橙色为其他候选。短簇单独标为低可信。'
        '默认按接触质量加权帧数选择；大框只软降权，不按类别删除。有证据连接的间隙不计入有效接触数。</p>'
        '<img style="max-width:100%" src="cluster_timeline.png"><img style="max-width:100%" src="candidate_montage.jpg">'
        '<pre>' + html.escape(json.dumps(report, ensure_ascii=False, indent=2)) + '</pre>')
