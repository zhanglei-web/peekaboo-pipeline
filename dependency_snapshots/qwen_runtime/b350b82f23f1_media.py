import base64
import json
import math
import re
import time
from pathlib import Path
from PIL import Image, ImageOps

def load_manifest(path, max_frames):
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("clip_id"), str) or not data["clip_id"]:
        raise ValueError("manifest needs a nonempty clip_id")
    frames = data.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= max_frames:
        raise ValueError(f"Expected 1..{max_frames} frames; no frames will be silently dropped")
    seen, previous = set(), -math.inf
    for frame in frames:
        fid = frame.get("frame_id")
        if not isinstance(fid, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", fid) or fid in seen:
            raise ValueError("frame_id must be unique and contain only letters, digits, '_' or '-'")
        seen.add(fid)
        ts = frame.get("timestamp_s")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts) or ts < 0 or ts <= previous:
            raise ValueError("timestamp_s must be finite, nonnegative and strictly increasing")
        previous = ts
        p = Path(frame["path"]).expanduser()
        p = p if p.is_absolute() else path.parent / p
        if not p.is_file():
            raise FileNotFoundError(p)
        frame["path"] = str(p.resolve())
    return data

def prepare_image(path, destination, max_edge):
    with Image.open(path) as source:
        original_size = list(source.size)
        oriented = ImageOps.exif_transpose(source)
        oriented_size = list(oriented.size)
        if oriented.mode in ("RGBA", "LA") or "transparency" in oriented.info:
            rgba = oriented.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, "white")
            rgb.paste(rgba, mask=rgba.getchannel("A"))
        else:
            rgb = oriented.convert("RGB")
        rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        rgb.save(destination, "JPEG", quality=95)
        sent_size = list(rgb.size)
    url = "data:image/jpeg;base64," + base64.b64encode(Path(destination).read_bytes()).decode("ascii")
    return url, {"original_size": original_size, "oriented_size": oriented_size, "sent_size": sent_size, "processed_path": str(destination)}

def extract_video(path, output_dir, start, end, num_frames):
    import cv2
    started = time.monotonic()
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Cannot decode video or determine FPS/frame count")
        duration = count / fps
        end = duration if end is None else end
        if not all(math.isfinite(v) for v in [start, end]) or not 0 <= start < end <= duration + 1e-6:
            raise ValueError(f"Expected 0 <= start < end <= duration ({duration:.6f}s)")
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        first = math.ceil(start * fps - 1e-8)
        last = min(count - 1, math.ceil(end * fps - 1e-8) - 1)
        if first > last:
            raise ValueError("The requested interval contains no video frames")
        requested = [round(first + i * (last - first) / max(num_frames - 1, 1)) for i in range(num_frames)]
        indices = list(dict.fromkeys(requested))
        frames = []
        for i, index in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Failed decoding frame {index}")
            actual_index = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if actual_index != index:
                raise ValueError(f"Decoder seek mismatch: requested {index}, actual {actual_index}")
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            # FFmpeg/OpenCV returns decoded PTS for supported streams. Record fallback explicitly.
            use_pts = math.isfinite(pts) and pts >= 0 and (index == 0 or pts > 0)
            timestamp = pts if use_pts else index / fps
            if frames and timestamp <= frames[-1]["timestamp_s"]:
                raise ValueError("Nonmonotonic decoded timestamps; use a normalized constant-FPS input")
            out = output_dir / f"video-frame-{index:06d}.jpg"
            if not cv2.imwrite(str(out), frame):
                raise IOError(f"Cannot write {out}")
            frames.append({"frame_id": f"f{i}", "path": out.name, "frame_index": actual_index,
                           "timestamp_s": timestamp, "timestamp_source": "decoder_pts" if use_pts else "index/fps_fallback"})
        manifest = {"clip_id": path.stem, "synthetic": "synthetic" in path.stem, "frames": frames,
                    "video": str(path), "fps": fps, "frame_count": count, "duration_s": duration,
                    "interval": [start, end], "interval_end_exclusive": True,
                    "sampling_method": "uniform_frame_index (CFR recommended)",
                    "requested_num_frames": num_frames, "actual_num_frames": len(frames),
                    "duplicate_indices_removed": len(requested) - len(indices),
                    "extraction_elapsed_s": time.monotonic() - started}
        dest = output_dir / "frames_manifest.json"
        dest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return dest
    finally:
        cap.release()
