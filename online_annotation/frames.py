"""Timestamp-limited RGB sampling; no future frames or event text enter selection."""
from bisect import bisect_left
from decimal import Decimal
from pathlib import Path
import hashlib
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def timestamps(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        index, seconds = line.split()[:2]
        rows.append((int(index), int(Decimal(seconds) * 1_000_000_000)))
    if not rows or any(index != i for i, (index, _) in enumerate(rows)):
        raise ValueError("Timestamp rows must use contiguous zero-based source indices")
    if any(b[1] <= a[1] for a, b in zip(rows, rows[1:])):
        raise ValueError("Timestamps must strictly increase")
    return rows


def prefix_rows(rows, seconds):
    if seconds <= 0:
        raise ValueError("Observation duration must be positive")
    cutoff = rows[0][1] + int(Decimal(str(seconds)) * 1_000_000_000)
    return [row for row in rows if row[1] < cutoff], cutoff


def stereo_match(left_ns, right_rows, cutoff_ns, tolerance_ns):
    values = [time for _, time in right_rows]
    offset = bisect_left(values, left_ns)
    candidates = [right_rows[i] for i in (offset - 1, offset) if 0 <= i < len(values)]
    candidates = [row for row in candidates if row[1] < cutoff_ns]
    if not candidates:
        return None
    index, time_ns = min(candidates, key=lambda row: abs(row[1] - left_ns))
    if abs(time_ns - left_ns) > tolerance_ns:
        return None
    return {"frame_index": index, "timestamp_ns": time_ns, "skew_ns": time_ns - left_ns}


def describe_image(frame):
    height, width = frame.shape[:2]
    thumb = cv2.resize(frame, (640, round(height * 640 / width)))
    gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dark = float(np.mean(gray < 12))
    bright = float(np.mean(gray > 243))
    quality = math.log1p(sharpness) * (1 - dark) * (1 - bright)
    # Spatial intensity and colour are only a coarse appearance filter, not pose.
    spatial = cv2.resize(gray, (32, 20)).astype(np.float32).ravel() / 255
    hsv = cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).ravel()
    hist /= max(float(hist.sum()), 1)
    return {
        "sharpness_640px": sharpness, "dark_fraction": dark,
        "bright_fraction": bright, "quality": quality,
        "quality_usable": sharpness >= 8 and dark < 0.85 and bright < 0.85,
    }, (spatial, hist)


def appearance_distance(a, b):
    spatial = float(np.abs(a[0] - b[0]).mean())
    colour = float(np.abs(a[1] - b[1]).sum() / 2)
    return 0.75 * spatial + 0.25 * colour


def sample_scene(session, output, config):
    session, output = Path(session), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    left = timestamps(session / "timestamps/left_timestamps.txt")
    right = timestamps(session / "timestamps/right_timestamps.txt")
    prefix, cutoff = prefix_rows(left, config["observation_seconds"])
    cap = cv2.VideoCapture(str(session / "videos/left.mkv"))
    if not cap.isOpened():
        raise ValueError("Cannot open left video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total != len(left):
        raise ValueError("Decoded video metadata frame count differs from timestamps")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    period_ns = int(1_000_000_000 / config["sample_hz"])
    sampled, features, seen_buckets = [], {}, set()
    try:
        for index, time_ns in prefix:
            ok, frame = cap.read()
            if not ok:
                raise ValueError("Video decode failed within initialization prefix")
            bucket = (time_ns - left[0][1]) // period_ns
            if bucket in seen_buckets:
                continue
            seen_buckets.add(bucket)
            metrics, feature = describe_image(frame)
            filename = f"left_f{index:06d}.jpg"
            if not cv2.imwrite(str(output / filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError("Cannot write sampled image")
            row = {
                "frame_index": index, "timestamp_ns": time_ns,
                "relative_seconds": (time_ns - left[0][1]) / 1e9,
                "image": str((output / filename).resolve()),
                "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                "right_match": stereo_match(time_ns, right, cutoff,
                                           int(config["stereo_tolerance_ms"] * 1e6)),
                **metrics,
            }
            sampled.append(row)
            features[index] = feature
    finally:
        cap.release()
    if not sampled:
        raise ValueError("Observation prefix contains no sampled frames")
    usable = [row for row in sampled if row["quality_usable"]]
    degraded = not usable
    pool = usable or sampled
    # Temporal bins first preserve opportunities for a brief different view.
    bins = {}
    for row in pool:
        key = int(row["relative_seconds"] / config["temporal_bin_seconds"])
        if key not in bins or row["quality"] > bins[key]["quality"]:
            bins[key] = row
    representatives = list(bins.values())
    selected = [max(representatives, key=lambda row: row["quality"])]
    for row in selected:
        row["preselection_reason"] = "highest_quality"
    while len(selected) < config["max_shortlist"]:
        remaining = [row for row in representatives if row not in selected]
        if not remaining:
            break
        distances = [(min(appearance_distance(features[row["frame_index"]],
                          features[chosen["frame_index"]]) for chosen in selected), row)
                     for row in remaining]
        distance, row = max(distances, key=lambda pair: (pair[0], pair[1]["quality"]))
        if distance < config["novelty_threshold"]:
            break
        row["preselection_reason"] = "complementary_appearance"
        row["minimum_appearance_distance"] = distance
        selected.append(row)
    selected.sort(key=lambda row: row["frame_index"])
    audit = {
        "episode_id": session.name, "source": str(session.resolve()),
        "video": {"frame_count": total, "fps": fps, "width": width, "height": height},
        "observation_start_ns": left[0][1], "observation_end_exclusive_ns": cutoff,
        "observation_seconds": config["observation_seconds"],
        "prefix_frames_decoded": len(prefix), "sampled_frames": len(sampled),
        "usable_frames": len(usable), "degraded_quality": degraded,
        "shortlist_frame_indices": [row["frame_index"] for row in selected],
        "stereo_matches": sum(row["right_match"] is not None for row in sampled),
        "stereo_matching": "nearest_timestamp_within_tolerance; never_same_index_assumption",
        "geometry_computed": False,
    }
    return sampled, selected, audit


FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def contact_sheet(rows, path, title, columns=3, width=480):
    columns = max(1, min(columns, len(rows)))
    if len(rows) == 4 and columns == 3:
        columns = 2
    font = ImageFont.truetype(FONT_PATH, 18)
    small = ImageFont.truetype(FONT_PATH, 15)
    rows_count = math.ceil(len(rows) / columns)
    tile_height = 335
    canvas = Image.new("RGB", (columns * width, 74 + rows_count * tile_height), "#f3f6fb")
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), title, font=font, fill="#172c48")
    for i, row in enumerate(rows):
        x, y = (i % columns) * width, 74 + (i // columns) * tile_height
        im = Image.open(row["image"]).convert("RGB")
        im.thumbnail((width - 16, 290))
        canvas.paste(im, (x + 8, y + 28))
        draw.text((x + 10, y), f"F{row['frame_index']:06d}  |  {row['relative_seconds']:.3f}s",
                  font=small, fill="#172c48")
        draw.text((x + 10, y + 314), f"sharpness {row['sharpness_640px']:.1f}", font=small, fill="#526581")
    canvas.save(path)
