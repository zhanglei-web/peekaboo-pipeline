"""Read-only audit of sqlite ROS2 bags and supplied event triplets."""
from __future__ import annotations

from collections import Counter
import heapq
import io
import json
from pathlib import Path
import re
import sqlite3

from PIL import Image
import yaml

from .io import require, sha

TOPICS = {
    "left": "/preprocess/left/image_rect/compressed",
    "right": "/preprocess/right/image_rect/compressed",
    "left_info": "/preprocess/left/camera_info",
    "right_info": "/preprocess/right/camera_info",
    "depth": "/preprocess/depth/image", "confidence": "/preprocess/depth/confidence",
    "pose": "/preprocess/pose", "tf": "/tf", "metadata": "/preprocess/frame_metadata",
}
REQUIRED = {TOPICS[k] for k in ["left", "right", "left_info", "right_info", "metadata"]}


def bag_files(source):
    source = Path(source).resolve()
    path = source / "metadata.yaml"
    require(path.is_file(), "source_bag must be a ROS2 bag directory containing metadata.yaml")
    meta = yaml.safe_load(path.read_text())["rosbag2_bagfile_information"]
    require(meta["storage_identifier"] == "sqlite3", "this S1 adapter supports sqlite3 ROS2 bags only")
    paths = [(source / name).resolve() for name in meta["relative_file_paths"]]
    require(paths and len(set(paths)) == len(paths), "bag metadata has no files or duplicate file paths")
    require(all(p.is_relative_to(source) and p.is_file() for p in paths), "missing/unsafe bag segment")
    return meta, paths


def open_sqlite(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+"?mode=ro", uri=True)


def audit_bag(source):
    meta, paths = bag_files(source)
    counts, types = Counter(), {}
    for path in paths:
        with open_sqlite(path) as connection:
            for name, kind, serialization, count in connection.execute(
                "SELECT t.name,t.type,t.serialization_format,count(m.id) FROM topics t "
                "LEFT JOIN messages m ON m.topic_id=t.id GROUP BY t.id"):
                require(serialization == "cdr", "only CDR ROS2 messages are supported")
                require(name not in types or types[name] == kind, "topic type differs between bag segments")
                types[name] = kind
                counts[name] += count
    require(REQUIRED <= types.keys(), "bag is missing required stereo/metadata topics")
    require(all(counts[name] > 0 for name in REQUIRED), "required input topic has no messages")
    return {"schema": "astra_memory.input_audit.v1", "source_bag": str(Path(source).resolve()),
        "files": {str(p): sha(p) for p in [Path(source)/"metadata.yaml", *paths]},
        "topic_types": types, "topic_counts": dict(counts),
        "frame_sync_policy": "exact_sensor_and_record_timestamp; reject_ambiguous_groups",
        "geometry_status": "not_certified_by_message_presence", "source_modified": False}


def triplets(directory, episode_id, fps):
    directory = Path(directory).resolve()
    require(directory.is_dir(), "event image directory is missing")
    phases = {"开始": "start", "进行中": "middle", "结束": "end"}
    regex = re.compile(r"^(?P<episode>.+)_第(?P<n>\d+)次_(?P<phase>开始|进行中|结束)_f(?P<frame>\d+)\.jpg$")
    grouped = {}
    for path in sorted(directory.glob("*.jpg")):
        match = regex.match(path.name)
        require(match is not None and match["episode"] == episode_id, f"invalid event filename: {path.name}")
        ordinal, phase, frame = int(match["n"]), phases[match["phase"]], int(match["frame"])
        require(ordinal > 0 and phase not in grouped.setdefault(ordinal, {}), "duplicate event phase")
        with Image.open(path) as image:
            image.verify()
        grouped[ordinal][phase] = {"phase": phase, "original_video_frame_index": frame,
            "original_video_seconds": frame/fps, "image_filename": path.name,
            "image_path": str(path), "image_sha256": sha(path),
            "frame_namespace": "original_video_frame_index_from_filename"}
    require(grouped and sorted(grouped) == list(range(1, len(grouped)+1)), "event ordinals must start at 1 without gaps")
    events = []
    for ordinal, states in sorted(grouped.items()):
        require(set(states) == {"start", "middle", "end"}, f"event {ordinal} lacks a phase image")
        ordered = [states[k] for k in ["start", "middle", "end"]]
        frames = [s["original_video_frame_index"] for s in ordered]
        require(frames[0] < frames[1] < frames[2], "event frame numbers must strictly increase")
        events.append({"event_id": f"{episode_id}:search_triplet_{ordinal:02d}", "ordinal": ordinal,
            "states": ordered, "sample_window_s": [frames[0]/fps, frames[-1]/fps],
            "result": "unknown", "result_source": "awaiting_user_confirmation"})
    return {"schema": "astra_memory.event_image_index.v1", "episode_id": episode_id,
        "directory": str(directory), "event_count": len(events), "image_count": len(events)*3,
        "events": events}


def message_groups(source):
    """Timestamp-ordered stream across split bags; topic IDs may differ per segment."""
    _, paths = bag_files(source)
    connections = []
    try:
        streams = []
        for path in paths:
            connection = open_sqlite(path)
            connections.append(connection)
            # Topic names/types come from each segment rather than shared integer IDs.
            cursor = connection.execute("SELECT m.timestamp,t.name,t.type,m.data FROM messages m "
                "JOIN topics t ON t.id=m.topic_id ORDER BY m.timestamp,m.id")
            streams.append(iter(cursor))
        previous, group = None, {}
        for timestamp, name, kind, data in heapq.merge(*streams, key=lambda row: row[0]):
            if name not in TOPICS.values():
                continue
            if previous is not None and timestamp != previous:
                yield previous, group
                group = {}
            require(name not in group, f"duplicate topic {name} at one timestamp")
            group[name] = (kind, data)
            previous = timestamp
        if group:
            yield previous, group
    finally:
        for connection in connections:
            connection.close()


def confirmation_todo(config, event_index):
    return {"schema": "astra_memory.confirmation_todo.v1", "episode_id": config["episode_id"],
        "target_description": config["target"]["description"], "success_policy": "manual_confirmation_only",
        "full_search_coverage_confirmed": None, "target_reference_image": None,
        "events": [{"event_id": e["event_id"], "ordinal": e["ordinal"],
            "sample_window_s": e["sample_window_s"], "result": "unknown",
            "confirmed_target_instance_id": None, "confirmation_note": None} for e in event_index["events"]],
        "note": "待确认模板，不是已确认事实；填写并明确提交后由后续S3确认入口处理。"}
