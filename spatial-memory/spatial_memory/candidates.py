"""Export time-stamped event frames for the three-role Qwen decision."""
from pathlib import Path

import cv2

from .io import write_json


def prepare_event(spec, output_dir, *, max_frames=20):
    """Spec comes from an event detector; preserves all declared boundary references.

    Source-frame IDs are zero-based indices into the supplied left video. An optional
    frame_index mapping supplies acquisition timestamps; otherwise video FPS is used.
    """
    start = int(spec["start_source_frame_id"])
    end = int(spec["end_source_frame_id"])
    if start < 0 or end < start:
        raise ValueError("Invalid event source-frame interval")
    video = cv2.VideoCapture(str(spec["left_video"]))
    if not video.isOpened():
        raise FileNotFoundError(spec["left_video"])
    fps = float(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or end >= frame_count:
        video.release()
        raise ValueError("Event interval exceeds video or FPS is invalid")
    timestamps = {int(r["source_frame_id"]): float(r["timestamp_s"])
                  for r in spec.get("frame_index", [])}
    source = "acquisition_index" if timestamps else "video_fps_estimate"
    def time_of(fid):
        if timestamps and fid not in timestamps:
            raise ValueError(f"Source frame {fid} has no acquisition timestamp")
        return timestamps[fid] if timestamps else fid / fps
    radius = round(float(spec.get("context_seconds", 1.0)) * fps)
    anchors = {"event_start": start, "event_end": end}
    for tag, key in (("contact_cluster_start", "contact_cluster_start_source_frame_id"),
                     ("contact_cluster_end", "contact_cluster_end_source_frame_id")):
        if key in spec:
            anchors[tag] = int(spec[key])
    low = max(0, start - radius)
    high = min(frame_count - 1, end + radius)
    protected = set(anchors.values()) | {low, high}
    if not all(low <= f <= high for f in protected):
        video.release()
        raise ValueError("A required reference lies outside declared event context")
    tagged = {fid: {tag for tag, anchor in anchors.items() if anchor == fid}
              for fid in protected}
    for entry in spec.get("important_frames", []):
        fid = int(entry["source_frame_id"])
        if low <= fid <= high:
            tagged.setdefault(fid, set()).add(entry["tag"])
    for i in range(8):
        fid = round(low + (high - low) * i / 7)
        tagged.setdefault(fid, set()).add("temporal_context")
    if len(protected) > max_frames:
        video.release()
        raise ValueError("Frame budget smaller than required boundary/context references")
    # Keep all protected frames, then explicit important frames, then uniform context.
    optional = sorted(set(tagged) - protected,
                      key=lambda f: ("temporal_context" in tagged[f], abs(f - (start + end) / 2), f))
    chosen = sorted(protected | set(optional[:max_frames - len(protected)]))
    output_dir = Path(output_dir)
    image_dir = output_dir / "candidate_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    actions = sorted(spec.get("action_rows", []), key=lambda r: r["source_frame_id"])
    candidates = []
    try:
        for fid in chosen:
            video.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ok, image = video.read()
            if not ok:
                raise ValueError(f"Cannot decode source frame {fid}")
            image_path = image_dir / f"f{fid:08d}.jpg"
            if not cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise OSError(f"Cannot save {image_path}")
            action = min(actions, key=lambda r: abs(r["source_frame_id"] - fid)) if actions else {}
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            candidates.append({"source_frame_id": fid, "timestamp_s": time_of(fid),
                               "image": str(image_path.resolve()),
                               "action": action.get("action_label") or action.get("phase") or "unknown",
                               "action_source_frame_id": action.get("source_frame_id"),
                               "action_confidence": action.get("confidence"),
                               "object_state": action.get("object_state", "unknown"),
                               "sharpness_variance": sharpness,
                               "tags": sorted(tagged[fid])})
    finally:
        video.release()
    result = {"schema_version": "spatial.event_candidates.v1",
              "episode_id": spec["episode_id"], "event_id": spec["event_id"],
              "event_start_time_s": time_of(start), "event_end_time_s": time_of(end),
              "start_source_frame_id": start, "end_source_frame_id": end,
              "timestamp_source": source,
              "required_reference_frame_ids": sorted(set(anchors.values())),
              "candidates": candidates,
              "event_detection_provenance": spec.get("event_detection_provenance")}
    write_json(output_dir / "event_candidates.json", result)
    return result
