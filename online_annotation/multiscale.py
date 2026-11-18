"""One complete scene proposal from overview and generic overlapping detail views."""
import copy
import json
from pathlib import Path

from PIL import Image

from .brief import expand, schema_for
from .contracts import validate_proposals
from .model import save_json

ROOT = Path(__file__).resolve().parent


def make_views(selected, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    views = []
    for row in selected:
        source = Image.open(row["image"]).convert("RGB")
        w, h = source.size
        # Three overlapping columns are detail observations, never region boundaries.
        crops = [(0, 0, w, h)]
        detail_w = round(w * .6)
        crops += [(x, 0, x + detail_w, h) for x in (0, (w - detail_w) // 2, w - detail_w)]
        for crop in crops:
            view_id = len(views)
            image = source.crop(crop)
            image.thumbnail((1280, 1280))
            path = output / f"view_{view_id:02d}_f{row['frame_index']:06d}.jpg"
            image.save(path, quality=95)
            views.append({"view_id": view_id, "frame_index": row["frame_index"],
                          "path": str(path.resolve()), "crop_pixels": list(crop),
                          "source_size": [w, h], "size": list(image.size)})
    save_json(output / "views.json", views)
    return views


def expand_views(data, views):
    mapped = {v["view_id"]: v for v in views}
    proposals = copy.deepcopy(expand(data, list(mapped)))
    for item in proposals["parents"] + proposals["regions"]:
        normalized = []
        for evidence in item["evidence"]:
            view = mapped[evidence["frame_index"]]
            left, top, right, bottom = view["crop_pixels"]
            w, h = view["source_size"]
            x1, y1, x2, y2 = evidence["bbox_1000"]
            box = [round((left * 1000 + x1 * (right - left)) / w),
                   round((top * 1000 + y1 * (bottom - top)) / h),
                   round((left * 1000 + x2 * (right - left)) / w),
                   round((top * 1000 + y2 * (bottom - top)) / h)]
            entry = {"frame_index": view["frame_index"], "bbox_1000": box}
            if entry not in normalized:
                normalized.append(entry)
        item["evidence"] = normalized
    validate_proposals(proposals, sorted({v["frame_index"] for v in views}))
    return proposals


def propose(model, output, selected):
    output = Path(output)
    views = make_views(selected, output / "detail_views")
    mapping = "\n".join(
        f"图像{i + 1}：view_id={v['view_id']}，来自原始F{v['frame_index']}，"
        f"裁剪像素范围{v['crop_pixels']}，原图尺寸{v['source_size']}。"
        for i, v in enumerate(views))
    prompt = ((ROOT / "prompts/regions.txt").read_text() + "\n\n"
              + (ROOT / "prompts/brief_transport.txt").read_text() + "\n\n"
              + (ROOT / "prompts/multiscale_regions.txt").read_text()
              + "\n本次视图对应关系：\n" + mapping)
    data, receipt = model.infer(output / "model/regions", prompt,
                               [v["path"] for v in views],
                               schema_for([v["view_id"] for v in views]), 6500)
    proposals = expand_views(data, views)
    save_json(output / "model/regions/expanded_proposals.json", proposals)
    return proposals, receipt
