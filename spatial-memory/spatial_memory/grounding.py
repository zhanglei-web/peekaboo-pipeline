"""Ground the operated object and hand exclusions on the actual rectified triplet."""
from pathlib import Path
import sys

import cv2
import numpy as np

from online_annotation.contracts import object_schema
from .io import sha256, write_json


COORD = {"type": "integer", "minimum": 0, "maximum": 1000}
BOX = object_schema({"x_min": COORD, "y_min": COORD, "x_max": COORD, "y_max": COORD})
FRAME = object_schema({"object_box": {"anyOf": [BOX, {"type": "null"}]},
                       "hand_boxes": {"type": "array", "items": BOX, "maxItems": 4}})
SCHEMA = object_schema({
    "object_name": {"type": "string"},
    "scope": {"type": "string", "enum": ["original_support_footprint", "container_interior", "other", "uncertain"]},
    "reason": {"type": "string"},
    "pre": FRAME, "pnr": FRAME, "post": FRAME,
})


def pixel_box(box, width=640, height=480):
    if box is None:
        return None
    x1, y1, x2, y2 = [box[k] for k in ("x_min", "y_min", "x_max", "y_max")]
    if not 0 <= x1 < x2 <= 1000 or not 0 <= y1 < y2 <= 1000:
        raise ValueError("Grounding model supplied an invalid named-coordinate box")
    return [int(round(x1 * width / 1000)), int(round(y1 * height / 1000)),
            int(round(x2 * width / 1000)), int(round(y2 * height / 1000))]


def ground_triplet(model, frames, region, output):
    prompt = (
        "以下三张图是同一视频事件的最终Pre、PNR、Post（按此顺序），已双目校正。"
        "定位同一个被操作物体，并框出每张图里的手。框使用0到1000的具名坐标"
        "x_min,y_min,x_max,y_max，表示左上角和右下角。物体不可见时object_box=null。"
        "每张图的物体框仅包含该物体，不能把承托桌面、整座柜体或手当成小物体。"
        "判断实际检查的空间：original_support_footprint表示物体从初始承托位置移开后露出的原占地；"
        "container_interior表示容器内部；仅操作屏幕、看见或触碰物体不自动证明检查了下面的占地，"
        "无法确认时scope=uncertain。前态应能定位原位置的参考物体。"
        "object_name简短，reason最多100字。只输出给定JSON。"
        f"\n原始区域关联仅供参考：{region['name']}；边界：{region['boundary_description']}。"
    )
    images = [frames[role]["left"] for role in ("pre", "pnr", "post")]
    value, receipt = model.infer(Path(output) / "grounding", prompt, images, SCHEMA, 1600)
    for role in ("pre", "pnr", "post"):
        pixel_box(value[role]["object_box"])
        for box in value[role]["hand_boxes"]:
            pixel_box(box)
    write_json(Path(output) / "grounding.json", {"prediction": value, "receipt": receipt})
    return value


class ObjectMasks:
    def __init__(self, weights, device="cuda:0", pythonpath=None):
        if pythonpath:
            sys.path.insert(0, str(Path(pythonpath).resolve()))
        from ultralytics import FastSAM
        self.model = FastSAM(str(weights))
        self.device = device
        self.weights_sha256 = sha256(weights)

    def compute(self, image_path, grounding, output, hand_mask=None):
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (480, 640):
            raise ValueError("Segmentation requires the 640x480 scoring raster")
        box = pixel_box(grounding["object_box"])
        hands = np.zeros((480, 640), dtype=bool)
        for item in grounding["hand_boxes"]:
            x1, y1, x2, y2 = pixel_box(item)
            hands[max(0, y1-3):min(480, y2+3), max(0, x1-3):min(640, x2+3)] = True
        if hand_mask is not None:
            hands = np.asarray(hand_mask, dtype=bool)
            if hands.shape != (480, 640):
                raise ValueError("Rectified hand exclusion shape is invalid")
        selected = np.zeros_like(hands)
        diagnostic = {"weights_sha256": self.weights_sha256, "object_box": box,
                      "hand_exclusion_source": ("calibrated_100DOH_source_frame_detections" if hand_mask is not None
                                                 else "Qwen3-VL-8B-Instruct_named_boxes")}
        if box is not None:
            prediction = self.model.predict(image, device=self.device, imgsz=640, conf=.25,
                                            iou=.7, retina_masks=True, max_det=300,
                                            verbose=False, save=False)[0]
            if prediction.masks is None:
                raise ValueError("FastSAM found no instance masks")
            masks = prediction.masks.data.detach().cpu().numpy() > .5
            roi = np.zeros_like(hands)
            x1, y1, x2, y2 = box
            roi[y1:y2, x1:x2] = True
            proposals = []
            for i, mask in enumerate(masks):
                area = int(mask.sum())
                intersection = int((mask & roi).sum())
                iou = intersection / max(int((mask | roi).sum()), 1)
                if 50 <= area < mask.size * .55 and iou >= .25 and intersection / area >= .5:
                    proposals.append((iou, i))
            if not proposals:
                raise ValueError("No FastSAM mask matches the operated-object grounding")
            proposals.sort(reverse=True)
            selected = masks[proposals[0][1]] & ~hands
            if selected.sum() < 50:
                raise ValueError("Operated object mask is unavailable outside hand exclusions")
            diagnostic.update(mask_index=proposals[0][1], box_iou=proposals[0][0],
                              proposal_count=len(proposals))
        dynamic = cv2.dilate((selected | hands).astype("uint8"), np.ones((19, 19), np.uint8)) > 0
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        for name, mask in (("object", selected), ("hands", hands), ("dynamic", dynamic)):
            cv2.imwrite(str(output / f"{name}.png"), mask.astype("uint8") * 255)
        diagnostic["object_pixels"] = int(selected.sum())
        write_json(output / "segmentation.json", diagnostic)
        return selected, hands, dynamic
