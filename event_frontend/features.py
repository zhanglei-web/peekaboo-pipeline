"""Rank cluster frames by spatially weighted visual change; retain legacy crops."""
from pathlib import Path
import json
import math
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .detector import sha256_file
from .video import read_jsonl, save_json, write_jsonl


STRATEGY = "fixed_initial_object_roi_clip_change"
SPATIAL_STRATEGY = "spatial_decay_local_clip_change"
SUPPORTED_STRATEGIES = (STRATEGY, SPATIAL_STRATEGY)
CLIP_SHA256 = "40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af"


def feature_config(path=None):
    config = json.loads(Path(__file__).with_name("feature_config.json").read_text())
    if path:
        supplied = json.loads(Path(path).read_text())
        if set(supplied) - set(config):
            raise ValueError("Unknown feature configuration keys")
        config.update(supplied)
    for key in ("feature_top_k", "batch_size", "image_max_edge"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    gap = config["minimum_peak_gap_seconds"]
    if not isinstance(gap, (int, float)) or not math.isfinite(gap) or gap < 0:
        raise ValueError("minimum_peak_gap_seconds must be nonnegative")
    if config["strategy"] not in ("fixed_roi", "spatial_decay"):
        raise ValueError("Unknown feature strategy")
    if type(config["spatial_long_edge"]) is not int or not 224 <= config["spatial_long_edge"] <= 896 or config["spatial_long_edge"] % 32:
        raise ValueError("spatial_long_edge must be a multiple of 32 in [224,896]")
    for key in ("decay_sigma_factor", "decay_cutoff_sigma", "path_weight", "minimum_valid_patch_fraction", "maximum_path_gap_seconds"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {key}")
    if config["path_weight"] > 1 or config["minimum_valid_patch_fraction"] > 1:
        raise ValueError("Spatial weights and validity fractions must be at most one")
    return config


def fixed_crop(frame, anchor):
    """The same source-pixel box is used on every frame, before any annotation."""
    height, width = frame.shape[:2]
    x1, y1 = max(0, math.floor(anchor[0])), max(0, math.floor(anchor[1]))
    x2, y2 = min(width, math.ceil(anchor[2])), min(height, math.ceil(anchor[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Initial object box has no pixels inside the source image")
    return Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))


def change_scores(features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0 or not np.isfinite(features).all():
        raise ValueError("Expected finite frame features")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Cannot compare zero image features")
    normalized = features / norms
    scores = np.clip(1 - normalized @ normalized[0], 0, 2)
    scores[0] = 0
    return scores


def top_change_frames(rows, top_k, minimum_gap_seconds):
    """Greedy score ranking with time suppression; endpoints are added separately."""
    selected = []
    for row in sorted(rows, key=lambda r: (-r["feature_change"], r["source_frame_id"])):
        if all(abs(row["relative_seconds"] - old["relative_seconds"]) + 1e-6 >= minimum_gap_seconds
               for old in selected):
            selected.append(row)
            if len(selected) == top_k:
                break
    return [r["source_frame_id"] for r in selected]


class ClipImageFeatures:
    def __init__(self, checkpoint, device):
        import clip
        import torch
        self.torch = torch
        torch.set_num_threads(4)
        self.device = device
        self.model, self.preprocess = clip.load(str(checkpoint), device=device, jit=False)
        self.model.eval()

    def encode(self, images):
        tensor = self.torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with self.torch.inference_mode():
            return self.model.encode_image(tensor).float().cpu().numpy()


def cluster_frames(index, start, end):
    wanted = {r["source_frame_id"]: r for r in index if start <= r["source_frame_id"] <= end}
    if set(wanted) != set(range(start, end + 1)):
        raise ValueError("Cluster has missing source frames in the interaction index")
    capture = cv2.VideoCapture(index[0]["source_video"])
    try:
        for source_id in range(end + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {source_id}")
            if source_id in wanted:
                row = wanted[source_id]
                if frame.shape[:2] != (row["height"], row["width"]):
                    raise ValueError("Source dimensions differ from detection metadata")
                yield row, frame
    finally:
        capture.release()


def rank_cluster_frames(candidates, detection_run, output, checkpoint, device="cuda", config=None):
    config = feature_config() if config is None else config
    if config.get("strategy", "fixed_roi") == "spatial_decay":
        from .spatial_features import rank_spatial_frames
        return rank_spatial_frames(candidates, detection_run, output, checkpoint, device, config)
    candidates, detection_run, output, checkpoint = [Path(p).resolve() for p in (candidates, detection_run, output, checkpoint)]
    parent = json.loads((candidates / "manifest.json").read_text())
    detection = json.loads((detection_run / "manifest.json").read_text())
    if detection.get("input_scope", detection["source"].get("input_scope")) != "single_interaction_clip":
        raise ValueError("Feature ranking requires one interaction clip")
    for field, name in (("detection_manifest_sha256", "manifest.json"), ("detections_sha256", "detections.jsonl")):
        if parent[field] != sha256_file(detection_run / name):
            raise ValueError("Candidate and detection identity differ")
    if sha256_file(detection["source"]["video"]) != detection["source"]["video_sha256"]:
        raise ValueError("Source video changed")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != CLIP_SHA256:
        raise ValueError("Expected the official CLIP ViT-B/32 checkpoint")
    packets = read_jsonl(candidates / "packets.jsonl")
    if len(packets) > 1:
        raise ValueError("One interaction clip must have at most one selected cluster")
    identity = {"sampling_strategy": STRATEGY,
                "detection_manifest_sha256": parent["detection_manifest_sha256"],
                "detections_sha256": parent["detections_sha256"],
                "parent_manifest_sha256": sha256_file(candidates / "manifest.json"),
                "parent_packets_sha256": sha256_file(candidates / "packets.jsonl"),
                "frame_index_sha256": sha256_file(detection_run / "frame_index.jsonl"),
                "feature_model": "CLIP ViT-B/32", "checkpoint_sha256": checkpoint_sha,
                "device": device, "config": config, "implementation_sha256": sha256_file(__file__)}
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != identity:
        raise ValueError("Feature ranking inputs changed; use a new output directory")
    save_json(output / "manifest.json", identity)
    samples, ranked, metadata = [], [], None
    if packets:
        packet = packets[0]
        target = packet["target"]
        p, r = target["start_source_frame_id"], target["end_source_frame_id"]
        index = read_jsonl(detection_run / "frame_index.jsonl")
        source = {row["source_frame_id"]: row for row in index}
        if p not in source or r not in source:
            raise ValueError("Cluster endpoints outside the interaction clip")
        anchor = packet["target_boxes"].get(source[p]["frame_id"])
        if not anchor or len(anchor) != 4 or not all(math.isfinite(v) for v in anchor):
            raise ValueError("No valid initial object box at the selected cluster start")
        if config["feature_top_k"] + 2 > packet["max_frames_per_packet"]:
            raise ValueError("Feature candidates plus P/R exceed the image limit")
        extractor = ClipImageFeatures(checkpoint, device)
        encoded, pending, frame_rows = [], [], []
        for row, frame in cluster_frames(index, p, r):
            pending.append(fixed_crop(frame, anchor))
            frame_rows.append(row)
            if len(pending) == config["batch_size"]:
                encoded.append(extractor.encode(pending))
                pending = []
        if pending:
            encoded.append(extractor.encode(pending))
        scores = change_scores(np.concatenate(encoded))
        ranked = [{"frame_id": row["frame_id"], "source_frame_id": row["source_frame_id"],
                   "relative_seconds": row["relative_seconds"], "feature_change": float(score)}
                  for row, score in zip(frame_rows, scores)]
        peaks = top_change_frames(ranked, config["feature_top_k"], config["minimum_peak_gap_seconds"])
        wanted = set(peaks) | {p, r}
        score_map = {row["source_frame_id"]: row["feature_change"] for row in ranked}
        metadata = {"strategy": STRATEGY, "reference_frame_id": source[p]["frame_id"],
                    "reference_source_frame_id": p, "anchor_bbox_xyxy": anchor,
                    "anchor_coordinate_system": "original_source_pixels_xyxy",
                    "comparison": "1 - cosine(image_feature_t, image_feature_cluster_start)",
                    "feature_model": "CLIP ViT-B/32", "checkpoint_sha256": checkpoint_sha,
                    "ranked_frame_count": len(ranked), "top_frame_ids_by_score": peaks,
                    "config": config}
        images = output / "images"
        images.mkdir(exist_ok=True)
        for row, frame in cluster_frames(index, p, r):
            source_id = row["source_frame_id"]
            if source_id not in wanted:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((config["image_max_edge"], config["image_max_edge"]))
            path = images / f"f{source_id:06d}.jpg"
            image.save(path, quality=95)
            samples.append({**row, "image": str(path), "image_sha256": sha256_file(path),
                            "image_header_pixels": 0, "anchor_bbox_xyxy": anchor,
                            "target_bbox_xyxy": packet["target_boxes"].get(row["frame_id"]),
                            "feature_change": score_map[source_id],
                            "feature_peak_candidate": source_id in peaks,
                            "reason": "feature_change_peak" if source_id in peaks else "fixed_cluster_endpoint"})
        packets = [{**packet, "frames": samples, "source_range": [p, r],
                    "sampling_strategy": STRATEGY, "feature_ranking": metadata}]
        print(json.dumps({"stage": "fixed_roi_feature_ranking", "fixed_p": p, "fixed_r": r,
                          "ranked_frames": len(ranked), "top_frame_ids_by_score": peaks,
                          "vlm_input_frames": sorted(wanted)}), flush=True)
    for name in ("selected_cluster.json", "clusters.json", "filter_decisions.jsonl", "cluster_timeline.png"):
        if (candidates / name).exists():
            shutil.copyfile(candidates / name, output / name)
    save_json(output / "feature_ranking.json", metadata)
    write_jsonl(output / "feature_scores.jsonl", ranked)
    write_jsonl(output / "candidates.jsonl", samples)
    write_jsonl(output / "packets.jsonl", packets)
    report = {"status": "ready" if packets else "no_valid_cluster", "sampling_strategy": STRATEGY,
              "feature_ranking": metadata, "vlm_input_frame_ids": [r["source_frame_id"] for r in samples],
              "selected_frames": len(samples), "packet_count": len(packets), "manually_verified": False}
    save_json(output / "candidate_report.json", report)
    write_feature_review(output, samples, metadata)
    return report


def write_feature_review(output, samples, metadata):
    import html
    tiles = []
    for row in samples:
        with Image.open(row["image"]) as source:
            source = source.convert("RGB")
            box = row["anchor_bbox_xyxy"]
            scale = source.width / row["width"]
            crop = source.crop(tuple(v * scale for v in box))
            crop.thumbnail((260, 260))
            source.thumbnail((520, 330))
            tile = Image.new("RGB", (800, 370), "#e5e7eb")
            tile.paste(source, (0, 32))
            tile.paste(crop, (530, 32))
            draw = ImageDraw.Draw(tile)
            draw.text((8, 8), f'F{row["source_frame_id"]} | change {row["feature_change"]:.4f} | FIXED ROI', fill="black")
            tiles.append(tile)
    if tiles:
        montage = Image.new("RGB", (1600, math.ceil(len(tiles) / 2) * 370), "white")
        for i, tile in enumerate(tiles):
            montage.paste(tile, (i % 2 * 800, i // 2 * 370))
        montage.save(output / "candidate_montage.jpg", quality=94)
    (output / "report.html").write_text('<!doctype html><meta charset="utf-8"><title>固定锚框特征筛选</title>'
        '<h1>簇首帧固定锚框 · CLIP 特征变化候选</h1><p>同一原始像素范围，按相对簇首帧的特征变化排序，时间去重后保留峰值候选。P/R 始终保留。</p>'
        '<img style="max-width:100%" src="candidate_montage.jpg"><pre>' +
        html.escape(json.dumps(metadata, ensure_ascii=False, indent=2)) + '</pre>')
