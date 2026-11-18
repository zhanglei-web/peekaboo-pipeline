"""One fixed, contextual square crop for each action window, from detections only."""
import numpy as np

from .objects import area


CONTEXT_SETTINGS = {"minimum_hand_score": .5, "maximum_object_image_fraction": .25,
                    "maximum_object_hand_area_ratio": 8., "minimum_short_side_fraction": .6,
                    "padding_fraction": .2, "retained_image_short_side": 640}


def contact_context_view(window, lookup, width, height):
    boxes, observations = [], 0
    for frame in window["sampled_source_frame_ids"]:
        row = lookup[frame]
        if row["status"] != "ok":
            continue
        objects = {o["object_id"]: o for o in row["objects"]}
        hands = [h for h in row["hands"] if h["score"] >= CONTEXT_SETTINGS["minimum_hand_score"]]
        for hand in hands:
            boxes.append(hand["bbox_xyxy"])
            observations += 1
            obj = objects.get(hand.get("matched_object_id"))
            if obj and (area(obj["bbox_xyxy"]) <= CONTEXT_SETTINGS["maximum_object_image_fraction"] * width * height
                        and area(obj["bbox_xyxy"]) <= CONTEXT_SETTINGS["maximum_object_hand_area_ratio"] * max(1., area(hand["bbox_xyxy"]))):
                boxes.append(obj["bbox_xyxy"])
    if not boxes:
        return {"view_bbox_xyxy": None, "view_status": "no_hand_fallback_center", "hand_observations": 0}
    boxes = np.asarray(boxes, dtype=float)
    x1, y1 = boxes[:, :2].min(0)
    x2, y2 = boxes[:, 2:].max(0)
    side = min(min(width, height), max(CONTEXT_SETTINGS["minimum_short_side_fraction"] * min(width, height),
                  (1 + 2 * CONTEXT_SETTINGS["padding_fraction"]) * max(x2 - x1, y2 - y1)))
    side = int(np.ceil(side))
    left = int(np.clip(round((x1 + x2 - side) / 2), 0, width - side))
    top = int(np.clip(round((y1 + y2 - side) / 2), 0, height - side))
    return {"view_bbox_xyxy": [left, top, left + side, top + side],
            "view_status": "fixed_window_contact_context", "hand_observations": observations}


def crop_action_view(image, row, width, height):
    box = row.get("view_bbox_xyxy")
    if box is None:
        return image
    sx, sy = image.width / width, image.height / height
    return image.crop(tuple(round(v * scale) for v, scale in zip(box, (sx, sy, sx, sy))))
