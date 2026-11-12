"""Independent ROS message decoding, raw evidence export and approximate geometry."""
from __future__ import annotations

import io
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image

from .inputs import TOPICS, REQUIRED, message_groups
from .io import atomic_bytes, digest, load, read_rows, require, safe_file, save, save_rows, sha


def stamp(header):
    return int(header.stamp.sec)*1_000_000_000 + int(header.stamp.nanosec)


def image_array(msg):
    """Raw single-channel arrays with explicit encoding, stride and endianness."""
    encoding = msg.encoding.lower()
    require(encoding in {"16uc1", "32fc1", "mono8", "8uc1"}, "unsupported depth/confidence encoding")
    code = {"16uc1": "u2", "32fc1": "f4", "mono8": "u1", "8uc1": "u1"}[encoding]
    dtype = np.dtype((">" if msg.is_bigendian else "<")+code)
    require(msg.width > 0 and msg.height > 0, "empty image")
    width_bytes = msg.width*dtype.itemsize
    require(msg.step >= width_bytes and len(msg.data) == msg.height*msg.step, "invalid image stride/byte count")
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :width_bytes].copy()
    return raw.view(dtype).reshape(msg.height, msg.width).astype(dtype.newbyteorder("="))


def save_npy(path, value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    atomic_bytes(path, stream.getvalue())


def write_rgb(directory, name, msg):
    raw = bytes(msg.data)
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format in {"JPEG", "PNG"}, "RGB evidence must be JPEG or PNG")
        width, height = image.size
        extension = "jpg" if image.format == "JPEG" else "png"
        image.verify()
    filename = f"{name}.{extension}"
    atomic_bytes(directory/filename, raw)
    return filename, width, height


def camera_info(msg, width, height, camera_frame):
    require((msg.width, msg.height) == (width, height), "CameraInfo/image resolution mismatch")
    require(msg.header.frame_id == camera_frame, "CameraInfo/image camera frame mismatch")
    k = np.asarray(msg.k, dtype=float).reshape(3, 3)
    require(np.isfinite(k).all() and k[0, 0] > 0 and k[1, 1] > 0, "invalid camera intrinsics")
    return {"frame_id": camera_frame, "width": width, "height": height,
        "k": k.tolist(), "p": list(msg.p), "r": list(msg.r), "d": list(msg.d),
        "distortion_model": msg.distortion_model}


def extract_group(group, timestamp, destination, index, config, decoder):
    require(REQUIRED <= group.keys(), "incomplete exact-timestamp stereo/metadata group")
    messages = {name: decoder(kind, data) for name, (kind, data) in group.items()}
    left = messages[TOPICS["left"]]
    ns = stamp(left.header)
    require(ns == timestamp, "record timestamp differs from image acquisition time; explicit sync adapter required")
    for name, msg in messages.items():
        if hasattr(msg, "header"):
            require(stamp(msg.header) == ns, f"unsynchronized sensor header: {name}")
        if name == TOPICS["tf"]:
            require(all(stamp(t.header) == ns for t in msg.transforms), "TF timestamp differs from image")
    import json
    source_meta = json.loads(messages[TOPICS["metadata"]].data)
    require(source_meta.get("sensor_time_ns") == ns, "frame metadata/sensor timestamp mismatch")
    frame_index = source_meta.get("left_video_frame_idx")
    require(isinstance(frame_index, int) and not isinstance(frame_index, bool) and frame_index >= 0,
            "original video frame index is missing or invalid")
    fid = f'{config["episode_id"]}:bag:{index:06d}'
    folder = destination/"frames"/f"{index:06d}"
    folder.mkdir(parents=True)
    left_name, width, height = write_rgb(folder, "left", left)
    right = messages[TOPICS["right"]]
    right_name, rw, rh = write_rgb(folder, "right", right)
    require((rw, rh) == (width, height), "stereo image resolution mismatch")
    prefix = f"frames/{index:06d}/"
    frame = {"schema": "astra_memory.frame.v1", "frame_id": fid, "bag_keyframe_index": index,
        "original_video_frame_index": frame_index, "original_video_seconds": frame_index/config["fps"],
        "frame_namespace": "original_video_index_from_ros_metadata", "sensor_time_ns": ns,
        "record_time_ns": timestamp, "camera_frame": left.header.frame_id,
        "width": width, "height": height, "rgb_path": prefix+left_name,
        "rgb_sha256": sha(folder/left_name), "right_path": prefix+right_name,
        "right_sha256": sha(folder/right_name), "source_metadata": source_meta,
        "left_camera_info": camera_info(messages[TOPICS["left_info"]], width, height, left.header.frame_id),
        "right_camera_info": camera_info(messages[TOPICS["right_info"]], rw, rh, right.header.frame_id),
        "depth_path": None, "confidence_path": None, "pose": None, "transforms": [],
        "geometry_status": "unavailable"}
    if TOPICS["depth"] in messages:
        msg = messages[TOPICS["depth"]]
        require(msg.header.frame_id == left.header.frame_id, "depth is not aligned to left RGB")
        depth = image_array(msg)
        require(depth.shape == (height, width), "depth/RGB shape mismatch")
        require(msg.encoding.lower() in {"16uc1", "32fc1"}, "invalid depth encoding")
        depth = depth.astype(np.float32)/(1000 if msg.encoding.lower() == "16uc1" else 1)
        save_npy(folder/"depth_m.npy", depth)
        frame.update(depth_path=prefix+"depth_m.npy", depth_unit="m",
            depth_valid_fraction=float(np.mean(np.isfinite(depth) & (depth > 0))),
            geometry_status="approximate_visible_surface_not_independently_verified")
    if TOPICS["confidence"] in messages:
        msg = messages[TOPICS["confidence"]]
        confidence = image_array(msg)
        require(confidence.dtype == np.uint8 and confidence.shape == (height, width) and
                msg.header.frame_id == left.header.frame_id, "invalid confidence image alignment/encoding")
        save_npy(folder/"confidence.npy", confidence)
        frame.update(confidence_path=prefix+"confidence.npy", confidence_scale="raw_uint8_not_calibrated_probability")
    if TOPICS["pose"] in messages:
        msg = messages[TOPICS["pose"]]
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        covariance = list(msg.pose.covariance)
        position, quaternion = [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
        require(all(math.isfinite(x) for x in position+quaternion+covariance), "non-finite pose/covariance")
        frame["pose"] = {"reference_frame": msg.header.frame_id, "position_m": position,
            "quaternion_xyzw": quaternion, "covariance": covariance,
            "quality": "unverified_openvins_estimate_not_ground_truth",
            "max_position_std_m": math.sqrt(max(max(covariance[i], 0) for i in [0, 7, 14]))}
    if TOPICS["tf"] in messages:
        for t in messages[TOPICS["tf"]].transforms:
            tr, q = t.transform.translation, t.transform.rotation
            frame["transforms"].append({"parent": t.header.frame_id, "child": t.child_frame_id,
                "sensor_time_ns": ns, "translation_m": [tr.x, tr.y, tr.z],
                "quaternion_xyzw": [q.x, q.y, q.z, q.w]})
    return frame


def prepare(config, audit, event_index, code_hash):
    output = Path(config["output"])
    dest = output/"input"
    signature = digest({"config": config, "audit": audit, "event_index": event_index, "implementation": code_hash})
    if dest.exists():
        saved = load(dest/"manifest.json")
        require(saved["signature"] == signature, "prepared input/config/code differs; use a new output directory")
        for relative, expected in saved["artifact_sha256"].items():
            path = safe_file(dest, relative)
            require(path.is_file() and sha(path) == expected, "prepared artifact missing or changed")
        return read_rows(dest/"frames.jsonl")
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    classes = {}

    def decoder(kind, data):
        if kind not in classes:
            classes[kind] = get_message(kind)
        return deserialize_message(data, classes[kind])

    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare.", dir=output))
    try:
        frames = []
        for timestamp, group in message_groups(config["source_bag"]):
            frame = extract_group(group, timestamp, staging, len(frames), config, decoder)
            require(not frames or (frame["sensor_time_ns"] > frames[-1]["sensor_time_ns"] and
                frame["original_video_frame_index"] > frames[-1]["original_video_frame_index"]),
                "source sensor/video timestamps are not strictly increasing")
            frames.append(frame)
        require(frames, "no complete input frames")
        source_frame_count = len(frames)
        if config["selection"]["mode"] == "adaptive_keyframes_v1":
            from .selection import select_frames
            save_rows(staging/"all_frames.jsonl", frames)
            frames, selection = select_frames(frames, staging, event_index, config["selection"])
            save(staging/"selection.json", selection)
        save_rows(staging/"frames.jsonl", frames)
        save(staging/"input_audit.json", audit)
        save(staging/"event_images.json", event_index)
        artifact_sha = {str(p.relative_to(staging)): sha(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        save(staging/"manifest.json", {"schema": "astra_memory.prepared_input.v1", "signature": signature,
            "artifact_sha256": artifact_sha, "bag_frame_count": source_frame_count,
            "selected_bag_frame_count": len(frames), "selection_policy": config["selection"],
            "source_video_coverage_s": [frames[0]["original_video_seconds"], frames[-1]["original_video_seconds"]],
            "geometry_quality": "unverified_vio_and_depth; no_world_coordinate_claims"})
        os.replace(staging, dest)
        return frames
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def bbox_geometry(frame, bbox, input_root):
    """BBox-based visible depth estimate, never a physical object center or extent."""
    if not frame["depth_path"]:
        return {"status": "unavailable", "reason": "no_aligned_depth"}
    depth = np.load(safe_file(input_root, frame["depth_path"]), allow_pickle=False)
    x1, y1, x2, y2 = bbox
    w, h = frame["width"], frame["height"]
    # Central half reduces background contamination, without claiming segmentation.
    xa, xb = max(0, int((x1*.75+x2*.25)*w)), min(w, int((x1*.25+x2*.75)*w)+1)
    ya, yb = max(0, int((y1*.75+y2*.25)*h)), min(h, int((y1*.25+y2*.75)*h)+1)
    patch = depth[ya:yb, xa:xb]
    valid = np.isfinite(patch) & (patch > 0)
    if not np.any(valid):
        return {"status": "unavailable", "reason": "no_valid_depth_in_visible_bbox"}
    z = float(np.median(patch[valid]))
    k = frame["left_camera_info"]["k"]
    u, v = (x1+x2)*w/2, (y1+y2)*h/2
    return {"status": "approximate_bbox_visible_surface_camera_only",
        "camera_frame": frame["camera_frame"], "median_depth_m": z,
        "camera_surface_point_m": [(u-k[0][2])*z/k[0][0], (v-k[1][2])*z/k[1][1], z],
        "not_object_center_or_dimensions": True, "world_geometry": "not_computed_from_unverified_pose"}
