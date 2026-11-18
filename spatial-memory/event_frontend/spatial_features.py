"""Local CLIP changes with an initial-anchor prior and a causal swept corridor.

The camera is registered to P using background tracks. CLIP spatial tokens are
compared at corresponding locations, rather than pooling an enlarged crop into
one vector. Token responses are a change proxy, not a segmentation/exposure area.
"""
from pathlib import Path
import html
import json
import math
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .detector import sha256_file
from .features import CLIP_SHA256, SPATIAL_STRATEGY, ClipImageFeatures, cluster_frames, top_change_frames
from .video import read_jsonl, save_json, write_jsonl


class ClipSpatialFeatures(ClipImageFeatures):
    """Use the installed ViT's patch outputs, with interpolated position embeddings."""

    def encode_spatial(self, images):
        torch = self.torch
        visual = self.model.visual
        tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(self.device).float() / 255
        mean = tensor.new_tensor((.48145466, .4578275, .40821073))[None, :, None, None]
        std = tensor.new_tensor((.26862954, .26130258, .27577711))[None, :, None, None]
        with torch.inference_mode():
            x = visual.conv1(((tensor - mean) / std).to(visual.conv1.weight.dtype))
            gh, gw = x.shape[-2:]
            x = x.flatten(2).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype)[None, None].expand(len(x), 1, -1)
            x = torch.cat((cls, x), dim=1)
            pos = visual.positional_embedding
            old = math.isqrt(len(pos) - 1)
            patches = pos[1:].float().T.reshape(1, -1, old, old)
            patches = torch.nn.functional.interpolate(patches, size=(gh, gw), mode="bicubic", align_corners=False)
            position = torch.cat((pos[:1].float(), patches.flatten(2).squeeze(0).T), dim=0)
            x = visual.ln_pre(x + position.to(x.dtype))
            x = visual.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
            x = visual.ln_post(x[:, 1:])
            if visual.proj is not None:
                x = x @ visual.proj
            x = torch.nn.functional.normalize(x.float(), dim=-1)
        return x.reshape(len(images), gh, gw, -1).cpu().numpy()


def canvas_size(width, height, long_edge):
    scale = long_edge / max(width, height)
    return max(32, round(width * scale / 32) * 32), max(32, round(height * scale / 32) * 32)


def box_polygon(box, sx=1., sy=1., transform=None):
    if box is None:
        return None
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or values[2] <= values[0] or values[3] <= values[1]:
        return None
    x1, y1, x2, y2 = values * [sx, sy, sx, sy]
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if transform is not None:
        points = (np.c_[points, np.ones(4)] @ np.asarray(transform).T)[:, :2]
    return points


def fill_polygon(mask, polygon):
    if polygon is not None:
        cv2.fillConvexPoly(mask, np.rint(np.clip(polygon, -100000, 100000)).astype(np.int32), 1)


def decay_weights(anchor_mask, path_mask, sigma, path_weight=.75, cutoff_sigma=3.):
    """Unit weight in the initial box; lower corridor weight; Gaussian falloff."""
    def soft(mask):
        if not np.any(mask):
            return np.zeros(mask.shape, dtype=np.float32)
        distance = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)
        result = np.exp(-.5 * (distance / sigma) ** 2)
        result[distance > cutoff_sigma * sigma] = 0
        return result
    return np.maximum(soft(anchor_mask), path_weight * soft(path_mask)).astype(np.float32)


def extend_corridor(mask, previous, current, connect=True):
    """Union successive local hulls, preserving a curved path without future frames."""
    if current is not None:
        fill_polygon(mask, current)
        if previous is not None and connect:
            hull = cv2.convexHull(np.vstack((previous, current)).astype(np.float32)).reshape(-1, 2)
            fill_polygon(mask, hull)
    return mask


def camera_step(previous, current, old_boxes, new_boxes):
    """Estimate previous->current motion; reject tracks on hands and the target."""
    identity = np.eye(3)
    if previous is None:
        return identity, {"valid": True, "inlier_fraction": 1., "reason": "reference_frame"}
    def background(boxes):
        mask = np.full(previous.shape, 255, np.uint8)
        for box in boxes:
            polygon = box_polygon(box)
            if polygon is not None:
                x1, y1 = np.floor(polygon.min(axis=0) - 5).astype(int)
                x2, y2 = np.ceil(polygon.max(axis=0) + 5).astype(int)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
        return mask
    old_mask, new_mask = background(old_boxes), background(new_boxes)
    points = cv2.goodFeaturesToTrack(previous, 220, .015, 8, mask=old_mask)
    failure = {"valid": False, "inlier_fraction": 0., "reason": "insufficient_background_tracks_identity_fallback"}
    if points is None or len(points) < 12:
        return identity, failure
    moved, status, errors = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(21, 21), maxLevel=2)
    if moved is None or status is None:
        return identity, failure
    returned, backward_status, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, winSize=(21, 21), maxLevel=2)
    if returned is None or backward_status is None:
        return identity, failure
    consistent = (status.ravel() == 1) & (backward_status.ravel() == 1)
    consistent &= np.linalg.norm(returned - points, axis=2).ravel() <= 1.5
    consistent &= errors.ravel() <= 15
    a, b = points[consistent].reshape(-1, 2), moved[consistent].reshape(-1, 2)
    finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    a, b = a[finite], b[finite]
    inside = (b[:, 0] >= 0) & (b[:, 0] < current.shape[1]) & (b[:, 1] >= 0) & (b[:, 1] < current.shape[0])
    a, b = a[inside], b[inside]
    keep = new_mask[np.floor(b[:, 1]).astype(int), np.floor(b[:, 0]).astype(int)] > 0
    a, b = a[keep], b[keep]
    if len(a) < 12:
        return identity, failure
    affine, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.)
    quality = float(inliers.mean()) if inliers is not None else 0.
    if affine is None or quality < .5 or not np.isfinite(affine).all():
        return identity, {**failure, "inlier_fraction": quality}
    scale = math.sqrt(abs(float(np.linalg.det(affine[:, :2]))))
    if not .85 <= scale <= 1.18:
        return identity, {**failure, "reason": "implausible_camera_scale_identity_fallback"}
    result = identity.copy()
    result[:2] = affine
    return result, {"valid": True, "inlier_fraction": quality, "track_count": len(a), "reason": "background_bidirectional_lk_ransac"}


def weighted_change(reference, current, weights, valid_fraction, minimum_valid_fraction=.9):
    if reference.shape != current.shape or reference.shape[:2] != weights.shape or weights.shape != valid_fraction.shape:
        raise ValueError("Spatial feature/weight grids differ")
    if not all(np.isfinite(v).all() for v in (reference, current, weights, valid_fraction)) or np.any(weights < 0):
        raise ValueError("Invalid spatial feature or weight values")
    reference_norm = np.linalg.norm(reference, axis=-1, keepdims=True)
    current_norm = np.linalg.norm(current, axis=-1, keepdims=True)
    if np.any(reference_norm <= 1e-12) or np.any(current_norm <= 1e-12):
        raise ValueError("Cannot compare zero spatial features")
    reference = reference / reference_norm
    current = current / current_norm
    difference = np.clip(1 - np.sum(reference * current, axis=-1), 0, 2)
    effective = weights * np.where(valid_fraction >= minimum_valid_fraction, valid_fraction, 0)
    total = float(effective.sum())
    # Missing coverage is explicitly unknown, never silently a zero-change frame.
    score = float(np.sum(effective * difference) / total) if total > 1e-8 else None
    return score, difference.astype(np.float32), effective.astype(np.float32)


def rank_spatial_frames(candidates, detection_run, output, checkpoint, device, config):
    candidates, detection_run, output, checkpoint = map(lambda p: Path(p).resolve(), (candidates, detection_run, output, checkpoint))
    parent = json.loads((candidates / "manifest.json").read_text())
    detection = json.loads((detection_run / "manifest.json").read_text())
    if detection.get("input_scope", detection["source"].get("input_scope")) != "single_interaction_clip":
        raise ValueError("Spatial ranking requires one interaction clip")
    for field, name in (("detection_manifest_sha256", "manifest.json"), ("detections_sha256", "detections.jsonl")):
        if parent[field] != sha256_file(detection_run / name):
            raise ValueError("Candidate/detection identity differs")
    if sha256_file(detection["source"]["video"]) != detection["source"]["video_sha256"] or sha256_file(checkpoint) != CLIP_SHA256:
        raise ValueError("Source video or CLIP weights changed")
    identity = {"sampling_strategy": SPATIAL_STRATEGY, "config": config, "device": device,
                "detection_manifest_sha256": sha256_file(detection_run / "manifest.json"),
                "detections_sha256": sha256_file(detection_run / "detections.jsonl"),
                "parent_manifest_sha256": sha256_file(candidates / "manifest.json"),
                "parent_packets_sha256": sha256_file(candidates / "packets.jsonl"),
                "frame_index_sha256": sha256_file(detection_run / "frame_index.jsonl"),
                "checkpoint_sha256": CLIP_SHA256, "feature_model": "CLIP ViT-B/32 spatial tokens",
                "implementation_sha256": {p: sha256_file(Path(__file__).with_name(p)) for p in ("spatial_features.py", "features.py")}}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Spatial ranking inputs changed; use a new output directory")
    save_json(output / "manifest.json", identity)
    packets = read_jsonl(candidates / "packets.jsonl")
    if len(packets) > 1:
        raise ValueError("Expected at most one selected cluster per interaction")
    ranked, samples, maps, metadata = [], [], [], None
    if packets:
        packet = packets[0]
        p, r = packet["target"]["start_source_frame_id"], packet["target"]["end_source_frame_id"]
        index = read_jsonl(detection_run / "frame_index.jsonl")
        source = {v["source_frame_id"]: v for v in index}
        detections = {v["source_frame_id"]: v for v in read_jsonl(detection_run / "detections.jsonl")}
        anchor = packet["target_boxes"].get(source[p]["frame_id"])
        width, height = source[p]["width"], source[p]["height"]
        cw, ch = canvas_size(width, height, config["spatial_long_edge"])
        sx, sy = cw / width, ch / height
        initial_polygon = box_polygon(anchor, sx, sy)
        if initial_polygon is None:
            raise ValueError("No valid P-frame object anchor")
        anchor_mask = np.zeros((ch, cw), np.uint8)
        fill_polygon(anchor_mask, initial_polygon)
        if not anchor_mask.any():
            raise ValueError("Initial object anchor is outside the image")
        path_mask = anchor_mask.copy()
        sigma = max(1., config["decay_sigma_factor"] * math.sqrt(float(anchor_mask.sum())))
        extractor = ClipSpatialFeatures(checkpoint, device)
        reference_features = None
        previous_gray, old_boxes, cumulative = None, [], np.eye(3)
        previous_target, previous_target_time = initial_polygon, source[p]["relative_seconds"]
        pending_images, pending_rows = [], []

        def flush():
            nonlocal reference_features
            if not pending_images:
                return
            features = extractor.encode_spatial(pending_images)
            if reference_features is None:
                reference_features = features[0]
            for feature, info in zip(features, pending_rows):
                score, difference, effective = weighted_change(reference_features, feature, info.pop("weights"),
                    info.pop("valid_fraction"), config["minimum_valid_patch_fraction"])
                if info["source_frame_id"] == p and score is not None:
                    score, difference = 0., np.zeros_like(difference)
                ranked.append({**info, "feature_change": score, "score_valid": score is not None,
                               "valid_weight_sum": float(effective.sum()), "valid_patch_count": int((effective > 0).sum())})
                maps.append((effective, difference))
            pending_images.clear()
            pending_rows.clear()

        for row, frame in cluster_frames(index, p, r):
            small = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            target_box = packet["target_boxes"].get(row["frame_id"])
            boxes = [h["bbox_xyxy"] for h in detections[row["source_frame_id"]].get("hands", [])]
            if target_box:
                boxes.append(target_box)
            boxes = [(np.asarray(b) * [sx, sy, sx, sy]).tolist() for b in boxes]
            step, quality = camera_step(previous_gray, gray, old_boxes, boxes)
            cumulative = step @ cumulative
            inverse = np.linalg.inv(cumulative)
            aligned = cv2.warpAffine(small, inverse[:2], (cw, ch), flags=cv2.INTER_LINEAR)
            valid = cv2.warpAffine(np.ones((ch, cw), np.float32), inverse[:2], (cw, ch), flags=cv2.INTER_LINEAR)
            polygon = box_polygon(target_box, sx, sy, inverse)
            if polygon is not None:
                gap = row["relative_seconds"] - previous_target_time
                extend_corridor(path_mask, previous_target, polygon, gap <= config["maximum_path_gap_seconds"])
                previous_target, previous_target_time = polygon, row["relative_seconds"]
            weight = decay_weights(anchor_mask, path_mask, sigma, config["path_weight"], config["decay_cutoff_sigma"])
            rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
            info = {"frame_id": row["frame_id"], "source_frame_id": row["source_frame_id"],
                    "relative_seconds": row["relative_seconds"], "camera_to_reference": inverse.tolist(),
                    "camera_registration": quality, "target_polygon_reference": polygon.tolist() if polygon is not None else None,
                    "weights": cv2.resize(weight, (cw // 32, ch // 32), interpolation=cv2.INTER_AREA),
                    "valid_fraction": cv2.resize(valid, (cw // 32, ch // 32), interpolation=cv2.INTER_AREA)}
            pending_images.append(rgb)
            pending_rows.append(info)
            previous_gray, old_boxes = gray, boxes
            if len(pending_images) >= config["batch_size"]:
                flush()
        flush()
        if len(ranked) != r - p + 1:
            raise ValueError("Spatial scores must cover every cluster frame")
        if config["feature_top_k"] + 2 > packet["max_frames_per_packet"]:
            raise ValueError("Feature review candidates exceed packet capacity")
        peaks = top_change_frames([v for v in ranked if v["score_valid"]], config["feature_top_k"], config["minimum_peak_gap_seconds"])
        wanted = set(peaks) | {p, r}
        by_id = {v["source_frame_id"]: (v, i) for i, v in enumerate(ranked)}
        metadata = {"strategy": SPATIAL_STRATEGY, "reference_source_frame_id": p,
                    "reference_frame_id": source[p]["frame_id"], "anchor_bbox_xyxy": anchor,
                    "anchor_coordinate_system": "P_frame_pixels_after_background_camera_registration",
                    "comparison": "weighted mean of local CLIP token cosine distances to P",
                    "feature_model": identity["feature_model"], "checkpoint_sha256": CLIP_SHA256,
                    "canvas_size": [cw, ch], "patch_grid": [ch // 32, cw // 32],
                    "position_embeddings": "bicubic interpolation of official ViT positions",
                    "spatial_token_calibration": "unvalidated dense change proxy; not exposure segmentation",
                    "anchor_weight": 1., "path_weight": config["path_weight"], "decay_sigma_canvas_pixels": sigma,
                    "corridor": "causal union of consecutive tracked-box hulls; no future boxes",
                    "ranked_frame_count": len(ranked), "valid_scored_frames": sum(v["score_valid"] for v in ranked),
                    "camera_identity_fallback_frames": sum(not v["camera_registration"]["valid"] for v in ranked),
                    "top_frame_ids_by_score": peaks, "config": config}
        np.savez_compressed(output / "spatial_maps.npz", source_frame_ids=np.array([v["source_frame_id"] for v in ranked]),
                            weights=np.stack([v[0] for v in maps]), local_changes=np.stack([v[1] for v in maps]))
        images = output / "images"
        images.mkdir(exist_ok=True)
        for row, frame in cluster_frames(index, p, r):
            source_id = row["source_frame_id"]
            if source_id not in wanted:
                continue
            evidence, map_index = by_id[source_id]
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((config["image_max_edge"], config["image_max_edge"]))
            path = images / f"f{source_id:06d}.jpg"
            image.save(path, quality=95)
            preview = images / f"spatial_f{source_id:06d}.jpg"
            render_spatial_preview(frame, evidence, maps[map_index], (cw, ch), initial_polygon, preview)
            samples.append({**row, "image": str(path), "image_sha256": sha256_file(path), "image_header_pixels": 0,
                            "anchor_bbox_xyxy": anchor, "target_bbox_xyxy": packet["target_boxes"].get(row["frame_id"]),
                            "feature_change": evidence["feature_change"], "score_valid": evidence["score_valid"],
                            "spatial_preview": str(preview), "feature_peak_candidate": source_id in peaks,
                            "reason": "spatial_change_peak" if source_id in peaks else "fixed_cluster_endpoint"})
        packets = [{**packet, "frames": samples, "source_range": [p, r],
                    "sampling_strategy": SPATIAL_STRATEGY, "feature_ranking": metadata}]
        print(json.dumps({"stage": "spatial_decay_feature_ranking", "fixed_p": p, "fixed_r": r,
                          "ranked_frames": len(ranked), "top_frame_ids_by_score": peaks}), flush=True)
    for name in ("selected_cluster.json", "clusters.json", "filter_decisions.jsonl", "target_decisions.jsonl", "cluster_timeline.png"):
        if (candidates / name).exists():
            shutil.copyfile(candidates / name, output / name)
    save_json(output / "feature_ranking.json", metadata)
    write_jsonl(output / "feature_scores.jsonl", ranked)
    write_jsonl(output / "candidates.jsonl", samples)
    write_jsonl(output / "packets.jsonl", packets)
    report = {"status": "ready" if packets else "no_valid_cluster", "sampling_strategy": SPATIAL_STRATEGY,
              "feature_ranking": metadata, "selected_frames": len(samples), "packet_count": len(packets), "vlm_used": False}
    save_json(output / "candidate_report.json", report)
    cards = ''.join(f'<figure><img src="images/{Path(v["spatial_preview"]).name}"><figcaption>F{v["source_frame_id"]} · score={v["feature_change"]}</figcaption></figure>' for v in samples)
    (output / "report.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>空间衰减变化评分</title>'
        '<style>body{max-width:1500px;margin:30px auto;font:16px system-ui}img{width:100%}figure{margin:24px 0}pre{white-space:pre-wrap}</style>'
        '<h1>初始物体框＋运动走廊＋外围衰减</h1><p>各图从左到右：原图、配准后的权重、加权局部变化。绿色框为初始框，橙色为当前框。'
        '所有评分仅使用本帧及此前轨迹。黑边无效区域不参与评分；相机配准失败显式记录。未调用 VLM。</p>'
        + cards + '<pre>' + html.escape(json.dumps(metadata, ensure_ascii=False, indent=2)) + '</pre></html>')
    return report


def render_spatial_preview(frame, evidence, maps, size, initial_polygon, output):
    cw, ch = size
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    aligned = cv2.warpAffine(small, np.asarray(evidence["camera_to_reference"])[:2], size)
    weight, change = maps
    panels = [small]
    for values in (weight, weight * change / 2):
        heat = cv2.resize(np.clip(values, 0, 1), size, interpolation=cv2.INTER_NEAREST)
        colors = cv2.applyColorMap(np.uint8(heat * 255), cv2.COLORMAP_TURBO)
        colored = cv2.addWeighted(aligned, .6, colors, .4, 0)
        cv2.polylines(colored, [np.rint(initial_polygon).astype(np.int32)], True, (0, 230, 0), 2)
        polygon = evidence["target_polygon_reference"]
        if polygon is not None:
            cv2.polylines(colored, [np.rint(polygon).astype(np.int32)], True, (0, 150, 255), 2)
        panels.append(colored)
    canvas = Image.new("RGB", (cw * 3, ch + 36), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, ("SOURCE", "SPATIAL WEIGHT", "WEIGHTED LOCAL CHANGE"))):
        canvas.paste(Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)), (i * cw, 32))
        draw.text((i * cw + 8, 8), f'F{evidence["source_frame_id"]} | {label}', fill="black")
    canvas.save(output, quality=94)
