"""100DOH inference using the official architecture and checkpoint layout."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

import cv2
import numpy as np

from .compat import DESCRIPTION, inclusive_nms, install_roi_compat


CONTACT_STATES = (
    "no_contact", "self_contact", "other_person_contact",
    "portable_object_contact", "stationary_object_contact",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def associate_objects(hands, objects):
    """Official offset endpoint / nearest center rule, only for object contact.

    Upstream magnitude * direction is multiplied by 10000 in original pixels.
    Matching has no learned confidence or geometric validation. No-object and
    non-object-contact states are explicit rather than indexing object -1.
    """
    centers = np.array([[(o["bbox_xyxy"][0] + o["bbox_xyxy"][2]) / 2,
                         (o["bbox_xyxy"][1] + o["bbox_xyxy"][3]) / 2]
                        for o in objects], dtype=np.float64)
    for hand in hands:
        hand.update(matched_object_id=None, matching_distance_px=None,
                    offset_endpoint_xy=None, matching_method="offset_endpoint_nearest_center")
        if hand["contact_state_id"] not in (3, 4):
            hand["matching_status"] = "not_object_contact"
            continue
        box = hand["bbox_xyxy"]
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        magnitude, dx, dy = hand["offset_vector"]
        endpoint = center + magnitude * 10000 * np.array([dx, dy])
        hand["offset_endpoint_xy"] = endpoint.tolist()
        if not objects:
            hand["matching_status"] = "object_missing"
            continue
        distances = np.linalg.norm(centers - endpoint, axis=1)
        index = int(np.argmin(distances))
        hand.update(matched_object_id=objects[index]["object_id"],
                    matching_distance_px=float(distances[index]),
                    matching_status="matched_unverified")
    return hands


class HandObjectDetector:
    def __init__(self, repository, checkpoint, hand_threshold=0.3,
                 object_threshold=0.3, device="cuda", threads=4):
        import torch
        import torchvision
        import yaml
        from easydict import EasyDict

        for value in (hand_threshold, object_threshold):
            if not 0 <= value <= 1:
                raise ValueError("Detection thresholds must be in [0,1]")
        self.repository = Path(repository).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        if not (self.repository / "lib/model/faster_rcnn/resnet.py").is_file():
            raise FileNotFoundError("Official hand_object_detector repository not found")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.set_num_threads(threads)
        torch.manual_seed(3)
        np.random.seed(3)
        self.device = torch.device(device)
        self.hand_threshold, self.object_threshold = hand_threshold, object_threshold
        sys.path.insert(0, str(self.repository / "lib"))
        install_roi_compat()
        from model.utils.config import cfg, _merge_a_into_b
        from model.faster_rcnn.resnet import resnet

        config_path = self.repository / "cfgs/res101.yml"
        _merge_a_into_b(EasyDict(yaml.safe_load(config_path.read_text())), cfg)
        cfg.CUDA = self.device.type == "cuda"
        cfg.USE_GPU_NMS = cfg.CUDA
        self.cfg = cfg
        # Read tensor-only checkpoints; never silently fall back to arbitrary pickle.
        checkpoint_value = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        self.model = resnet(np.asarray(["__background__", "targetobject", "hand"]),
                            101, pretrained=False, class_agnostic=False)
        self.model.create_architecture()
        self.model.load_state_dict(checkpoint_value["model"], strict=True)
        cfg.POOLING_MODE = checkpoint_value.get("pooling_mode", cfg.POOLING_MODE)
        if cfg.POOLING_MODE not in ("align", "pool"):
            raise ValueError(f"Unsupported pooling mode {cfg.POOLING_MODE}")
        self.model.to(self.device).eval()
        commit = subprocess.check_output(
            ["git", "-C", str(self.repository), "rev-parse", "HEAD"], text=True).strip()
        provenance = self.checkpoint.with_suffix(self.checkpoint.suffix + ".provenance.json")
        checkpoint_hash = sha256_file(self.checkpoint)
        ego_mirror_hash = "3444e3b992417cb6adfcd71539ca8c92602c21a3e2fdfff4628dbdbdf8a2dd03"
        self.metadata = {
            "name": "100DOH_hand_object_detector",
            "variant": "handobj_100K+ego_mirror" if checkpoint_hash == ego_mirror_hash else "custom_checkpoint_unknown_variant",
            "repository": str(self.repository), "repository_commit": commit,
            "checkpoint": str(self.checkpoint), "checkpoint_sha256": checkpoint_hash,
            "config_sha256": sha256_file(config_path), "pooling_mode": cfg.POOLING_MODE,
            "hand_threshold": hand_threshold, "object_threshold": object_threshold,
            "nms_threshold": cfg.TEST.NMS, "anchor_scales": cfg.ANCHOR_SCALES,
            "test_short_side": cfg.TEST.SCALES[0], "test_max_side": cfg.TEST.MAX_SIZE,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(self.device),
            "gpu": torch.cuda.get_device_name(self.device) if cfg.CUDA else None,
            "compatibility": DESCRIPTION, "threads": threads,
            "weight_provenance": json.loads(provenance.read_text()) if provenance.exists() else None,
        }

    def predict(self, frame):
        import torch
        from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

        height, width = frame.shape[:2]
        image = frame.astype(np.float32, copy=True)
        image -= self.cfg.PIXEL_MEANS
        scale = float(self.cfg.TEST.SCALES[0]) / min(height, width)
        if np.round(scale * max(height, width)) > self.cfg.TEST.MAX_SIZE:
            scale = float(self.cfg.TEST.MAX_SIZE) / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        info = tensor.new_tensor([[image.shape[0], image.shape[1], scale]])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model(tensor, info, tensor.new_zeros((1, 1, 5)),
                                torch.zeros(1, dtype=torch.long, device=self.device),
                                tensor.new_zeros((1, 1, 5)))
            rois, class_probs, deltas = result[:3]
            extension = result[-1]
            contact_probs = extension[0][0].softmax(dim=2)[0]
            contact_ids = contact_probs.argmax(dim=1)
            offsets = extension[1][0][0]
            right_probs = extension[2][0][0, :, 0].sigmoid()
            if not self.cfg.TEST.BBOX_REG:
                raise RuntimeError("This adapter requires bbox regression")
            deltas = (deltas.reshape(-1, 4) * tensor.new_tensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS)
                      + tensor.new_tensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS)).reshape(1, -1, 12)
            boxes = clip_boxes(bbox_transform_inv(rois[:, :, 1:5], deltas, 1), info, 1)[0] / scale
            scores = class_probs[0]
            hands, objects = [], []
            for class_id, threshold in ((1, self.object_threshold), (2, self.hand_threshold)):
                indices = torch.nonzero(scores[:, class_id] > threshold).flatten()
                if not indices.numel():
                    continue
                cls_boxes = boxes[indices, class_id * 4:(class_id + 1) * 4]
                cls_scores = scores[indices, class_id]
                order = cls_scores.argsort(descending=True)
                keep = inclusive_nms(cls_boxes[order], cls_scores[order], self.cfg.TEST.NMS)
                indices = indices[order[keep]]
                for index in indices.tolist():
                    box = boxes[index, class_id * 4:(class_id + 1) * 4].cpu().tolist()
                    box = [min(max(box[0], 0), width - 1), min(max(box[1], 0), height - 1),
                           min(max(box[2], 0), width - 1), min(max(box[3], 0), height - 1)]
                    if not np.isfinite(box).all() or box[2] < box[0] or box[3] < box[1]:
                        raise ValueError("Invalid detection box")
                    common = {"bbox_xyxy": box, "score": float(scores[index, class_id]),
                              "proposal_id": index}
                    if class_id == 1:
                        objects.append({"object_id": f"o{len(objects)}", **common})
                    else:
                        state = int(contact_ids[index])
                        right = float(right_probs[index])
                        hands.append({"hand_id": f"h{len(hands)}", **common,
                                      "side": "right" if right > 0.5 else "left",
                                      "right_probability": right,
                                      "contact_state_id": state, "contact_state": CONTACT_STATES[state],
                                      "contact_probabilities": contact_probs[index].cpu().tolist(),
                                      "offset_vector": offsets[index].cpu().tolist()})
        associate_objects(hands, objects)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return {
            "hands": hands, "objects": objects,
            "interaction_candidate": any(h["contact_state_id"] in (3, 4) for h in hands),
            "inference_seconds": time.perf_counter() - start,
            "preprocess": {"original_width": width, "original_height": height,
                           "network_width": image.shape[1], "network_height": image.shape[0],
                           "scale": scale, "box_coordinates": "original_image_inclusive_xyxy",
                           "channel_order": "BGR"},
        }
