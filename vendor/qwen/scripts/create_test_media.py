#!/usr/bin/env python3
"""Synthetic fixtures ONLY: no claim about real-workbench recognition quality."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from local_vlm.config import ROOT

def draw_scene(index, total=4):
    img = Image.new("RGB", (960, 640), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "SYNTHETIC PIPELINE TEST", fill="black", font_size=24)
    draw.ellipse((400, 270, 540, 410), fill=(15, 65, 230))
    x = round(120 + index * 600 / (total - 1))
    draw.rectangle((x, 280, x + 110, 390), fill=(230, 25, 25))
    draw.polygon([(470, 90), (410, 200), (530, 200)], fill=(20, 170, 40))
    return img

def main():
    directory = ROOT / "data/synthetic"
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        draw_scene(i).save(directory / f"frame-{i}.png")
    for count, indices in [(2, [0, 3]), (4, [0, 1, 2, 3])]:
        manifest = {"clip_id": f"synthetic_{count}frames", "synthetic": True,
                    "frames": [{"frame_id": f"f{i}", "path": f"frame-{index}.png", "timestamp_s": float(index)} for i, index in enumerate(indices)]}
        (directory / f"frames-{count}.json").write_text(json.dumps(manifest, indent=2))
    path = directory / "synthetic-demo.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (960, 640))
    if not writer.isOpened():
        raise RuntimeError("OpenCV cannot create synthetic MP4")
    for i in range(20):
        writer.write(cv2.cvtColor(np.array(draw_scene(i, 20)), cv2.COLOR_RGB2BGR))
    writer.release()
    print(directory)

if __name__ == "__main__":
    main()
