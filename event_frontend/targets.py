"""Conservative contact-target evidence; never invent a smaller detection box."""
from collections import defaultdict
import copy
import math

import numpy as np

from .objects import area, appearance_distance, center, diagonal, iou


def containment(inner, outer):
    intersection = [max(inner[0], outer[0]), max(inner[1], outer[1]),
                    min(inner[2], outer[2]), min(inner[3], outer[3])]
    return area(intersection) / max(area(inner), 1.)


def point_box_distance(point, box):
    return math.hypot(max(box[0] - point[0], point[0] - box[2], 0),
                      max(box[1] - point[1], point[1] - box[3], 0))


def contact_target_quality(hand, obj, config):
    """A bounded geometric weight, not a calibrated correctness probability.

    Large objects remain eligible. A box is penalized only to the extent that
    its center also disagrees with the predicted contact endpoint. Missing
    endpoint evidence stays unknown rather than claiming a precise target.
    """
    ratio = area(obj["bbox_xyxy"]) / max(area(hand["bbox_xyxy"]), 1.)
    endpoint = hand.get("offset_endpoint_xy")
    if endpoint is None:
        return {"weight": 1., "evidence_available": False,
                "object_to_hand_area_ratio": ratio, "endpoint_center_distance_hand_diagonals": None,
                "extent_ambiguous": None, "method": "missing_contact_endpoint_no_penalty"}
    point = np.asarray(endpoint, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("Invalid hand-object contact endpoint")
    distance = float(np.linalg.norm(center(obj["bbox_xyxy"]) - point) / diagonal(hand["bbox_xyxy"]))
    size_penalty = max(0., 1 - math.sqrt(config["target_quality_area_ratio_reference"] / max(ratio, 1e-9)))
    penalty = min(config["target_quality_maximum_penalty"], size_penalty) * min(1., distance)
    weight = 1 - penalty
    return {"weight": weight, "evidence_available": True,
            "object_to_hand_area_ratio": ratio, "endpoint_center_distance_hand_diagonals": distance,
            "extent_ambiguous": weight < .8, "method": "bounded_size_and_contact_endpoint_penalty"}


def nested_target(hand, objects, config):
    original = objects.get(hand.get("object_id"))
    endpoint = hand.get("offset_endpoint_xy")
    if original is None or endpoint is None:
        return None
    quality = contact_target_quality(hand, original, config)
    if quality["object_to_hand_area_ratio"] <= config["nested_target_hand_area_ratio_max"]:
        return None
    old_distance = float(np.linalg.norm(center(original["bbox_xyxy"]) - endpoint))
    choices = []
    for candidate in objects.values():
        box = candidate["bbox_xyxy"]
        if candidate["object_id"] == original["object_id"] or (
                area(original["bbox_xyxy"]) < config["nested_target_area_ratio_min"] * area(box)):
            continue
        if containment(box, original["bbox_xyxy"]) < config["nested_target_containment_min"]:
            continue
        if area(box) / max(1., area(hand["bbox_xyxy"])) > config["nested_target_hand_area_ratio_max"]:
            continue
        new_distance = float(np.linalg.norm(center(box) - endpoint))
        if new_distance > .8 * old_distance or point_box_distance(endpoint, box) > .25 * diagonal(hand["bbox_xyxy"]):
            continue
        if point_box_distance(center(hand["bbox_xyxy"]), box) > .5 * diagonal(hand["bbox_xyxy"]):
            continue
        choices.append((new_distance / diagonal(hand["bbox_xyxy"]), -candidate.get("score", 0), candidate["object_id"]))
    return min(choices)[-1] if choices else None


def refine_contact_targets(rows, config):
    """Reassociate only repeated, independently detected nested alternatives.

    Support counts distinct frames, not hands. All raw detector boxes, IDs and
    class predictions remain available; a newly contacted unclassified object
    keeps its unknown label. This is an offline, per-interaction evidence pass.
    """
    refined = copy.deepcopy(rows)
    proposals, support = {}, defaultdict(list)
    for row in refined:
        objects = {obj["object_id"]: obj for obj in row["objects"]}
        if row["status"] != "ok":
            continue
        for i, hand in enumerate(row["contacts"]):
            if hand["score"] < config["hand_score_min"]:
                continue
            candidate = nested_target(hand, objects, config)
            if candidate is not None:
                key = (hand.get("object_id"), candidate)
                proposals[row["source_frame_id"], i] = key
                support[key].append(row["source_frame_id"])
    accepted = set()
    support_counts = {}
    for key, ids in support.items():
        groups = []
        for frame_id in sorted(set(ids)):
            if not groups or frame_id - groups[-1][-1] - 1 > config["maximum_missing_contact_frames"]:
                groups.append([])
            groups[-1].append(frame_id)
        for group in groups:
            if len(group) >= config["nested_target_minimum_support_frames"]:
                for frame_id in group:
                    accepted.add((frame_id, key))
                    support_counts[frame_id, key] = len(group)
    decisions = []
    for row in refined:
        objects = {obj["object_id"]: obj for obj in row["objects"]}
        for i, hand in enumerate(row["contacts"]):
            original = hand.get("object_id")
            proposal = proposals.get((row["source_frame_id"], i))
            changed = proposal is not None and (row["source_frame_id"], proposal) in accepted
            if changed:
                hand["object_id"] = proposal[1]
            hand["original_tracked_object_id"] = original
            obj = objects.get(hand.get("object_id"))
            quality = contact_target_quality(hand, obj, config) if obj else None
            decision = {"source_frame_id": row["source_frame_id"], "hand_id": hand["hand_id"],
                        "original_object_id": original, "object_id": hand.get("object_id"),
                        "changed": changed, "proposed_object_id": proposal[1] if proposal else None,
                        "support_frames": support_counts.get((row["source_frame_id"], proposal), 0),
                        "original_bbox_xyxy": objects[original]["bbox_xyxy"] if original in objects else None,
                        "selected_bbox_xyxy": obj["bbox_xyxy"] if obj else None,
                        "target_quality": quality,
                        "reason": "supported_nested_detection" if changed else "insufficient_nested_support" if proposal else "original_association_retained"}
            hand["target_refinement"] = decision
            decisions.append(decision)
    return refined, decisions


def supported_contact_gap(previous, current, rows_by_id, config):
    """Bridge a short contact-ID flicker only with positive continuity evidence.

    Same target ID, similar endpoint boxes/appearance, continuous observed hand,
    and a visible target or containing-box association are all required. No
    interpolated detections or contact counts are created for the gap frames.
    """
    missing = current["source_frame_id"] - previous["source_frame_id"] - 1
    elapsed = current["relative_seconds"] - previous["relative_seconds"]
    if missing < 1 or elapsed <= 0 or elapsed > config["supported_contact_gap_seconds"]:
        return None
    a, b = previous["object"], current["object"]
    overlap = iou(a["bbox_xyxy"], b["bbox_xyxy"])
    visual = appearance_distance(a["appearance"], b["appearance"])
    if overlap < config["supported_contact_gap_iou_min"] or visual > config["supported_contact_gap_appearance_max"]:
        return None
    # A large support box alone is insufficient evidence to bridge a gap.
    for contact in (previous, current):
        if not any(contact_target_quality(h, contact["object"], config)["evidence_available"] and
                   contact_target_quality(h, contact["object"], config)["object_to_hand_area_ratio"] <=
                   config["nested_target_hand_area_ratio_max"] for h in contact["hands"]):
            return None
    hands = previous["hands"]
    evidence = []
    for frame_id in range(previous["source_frame_id"] + 1, current["source_frame_id"] + 1):
        row = rows_by_id.get(frame_id)
        if row is None or row["status"] != "ok":
            return None
        objects = {obj["object_id"]: obj for obj in row["objects"]}
        choices = []
        for hand in row["contacts"]:
            if hand["score"] < config["hand_score_min"] or (
                    hand["contact_probabilities"][hand["contact_state_id"]] < config["contact_probability_min"]):
                continue
            continuity = max((iou(hand["bbox_xyxy"], old["bbox_xyxy"]) for old in hands), default=0)
            obj = objects.get(hand.get("object_id"))
            if continuity < config["supported_contact_gap_hand_iou_min"] or obj is None:
                continue
            same = obj["object_id"] == a["object_id"]
            containing = all(area(obj["bbox_xyxy"]) >= config["supported_contact_gap_containing_area_ratio_min"] * area(target["bbox_xyxy"]) and
                             containment(target["bbox_xyxy"], obj["bbox_xyxy"]) >= config["nested_target_containment_min"]
                             for target in (a, b))
            if same or containing:
                choices.append((hand, "same_target_visible" if same else "containing_box_flicker"))
        if not choices:
            return None
        hands = [h for h, _ in choices]
        if frame_id < current["source_frame_id"]:
            evidence.append({"source_frame_id": frame_id, "reason": choices[0][1]})
    return {"start_source_frame_id": previous["source_frame_id"] + 1,
            "end_source_frame_id": current["source_frame_id"] - 1,
            "missing_contact_frames": missing, "elapsed_seconds": elapsed,
            "endpoint_iou": overlap, "appearance_distance": visual, "evidence": evidence,
            "contact_frames_fabricated": 0}
