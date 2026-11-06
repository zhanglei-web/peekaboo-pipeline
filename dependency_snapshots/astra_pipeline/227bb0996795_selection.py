"""Deterministic image-novelty/temporal sampling; all event triplets remain separate."""
from pathlib import Path

import numpy as np
from PIL import Image

from .io import require


def select_frames(frames, input_root, events, policy):
    require(frames, "no frames to select")
    times = [f["original_video_seconds"] for f in frames]
    require(all(a < b for a, b in zip(times, times[1:])), "selection timestamps are not increasing")
    cap = min(policy["max_frames"], len(frames))
    selected = {0: ["first_bag_frame"], len(frames)-1: ["last_bag_frame"]}
    for event in events["events"]:
        for state in [event["states"][0], event["states"][-1]]:
            t = state["original_video_seconds"]
            if times[0] <= t <= times[-1]:
                i = min(range(len(frames)), key=lambda j: (abs(times[j]-t), j))
                selected.setdefault(i, []).append("near_event_boundary:"+event["event_id"]+":"+state["phase"])
    require(len(selected) <= cap, "keyframe budget cannot cover required event boundaries; increase max_frames")
    while True:
        ordered = sorted(selected)
        gaps = [(times[b]-times[a], a, b) for a, b in zip(ordered, ordered[1:])]
        gap, a, b = max(gaps, default=(0, 0, 0))
        if gap <= policy["max_gap_s"]:
            break
        require(len(selected) < cap and b-a > 1, "keyframe budget/source cannot satisfy max_gap_s")
        midpoint = (times[a]+times[b])/2
        i = min(range(a+1, b), key=lambda j: (abs(times[j]-midpoint), j))
        selected[i] = ["temporal_gap_coverage"]
    features = []
    for frame in frames:
        with Image.open(Path(input_root)/frame["rgb_path"]) as image:
            features.append(np.asarray(image.convert("RGB").resize((96, 54), Image.Resampling.BILINEAR), dtype=np.float32)/255)
    features = np.stack(features)
    adjacent = [0.]+[float(np.abs(features[i]-features[i-1]).mean()) for i in range(1, len(frames))]
    while len(selected) < cap:
        candidates = [i for i in range(len(frames)) if i not in selected and
                      min(abs(times[i]-times[j]) for j in selected) >= policy["min_spacing_s"]]
        if not candidates:
            break
        def novelty(i):
            neighbors = sorted(selected, key=lambda j: (abs(times[i]-times[j]), j))[:2]
            return min(float(np.abs(features[i]-features[j]).mean()) for j in neighbors)
        scored = [(novelty(i), -i, i) for i in candidates]
        score, _, i = max(scored)
        if score < policy["visual_change_threshold"]:
            break
        selected[i] = ["visual_novelty"]
    ordered = sorted(selected)
    report = {"schema": "astra_memory.keyframe_selection.v1", "policy": policy,
        "source_bag_frame_count": len(frames), "selected_bag_frame_count": len(ordered),
        "retained_event_image_count": events["image_count"], "event_images_filtered": False,
        "selected_original_video_frames": [frames[i]["original_video_frame_index"] for i in ordered],
        "max_selected_gap_s": max((times[b]-times[a] for a,b in zip(ordered,ordered[1:])), default=0),
        "coverage_note": "Adaptive samples, not exhaustive frame observation; brief unsampled changes may be absent.",
        "frames": [{"image_ref": f["frame_id"], "original_video_frame_index": f["original_video_frame_index"],
                    "time_s": times[i], "selected": i in selected, "adjacent_rgb_mean_absolute_difference": adjacent[i],
                    "reasons": selected.get(i, ["not_selected_under_visual_temporal_budget"])} for i,f in enumerate(frames)]}
    return [frames[i] for i in ordered], report
