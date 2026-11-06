#!/usr/bin/env python3
"""Materialize selected Ego MP4 frames as a rectified stereo dataset.

The script consumes the source-audit frame inventory, the S04 selected source
indices, and the OpenVINS-derived ``world_T_cam0`` samples.  Video PTS is used
only to decode in source order; all output time is copied from the audited
integer-nanosecond sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


SCHEMA_VERSION = 1
PNG_OPTIONS = [cv2.IMWRITE_PNG_COMPRESSION, 1]
PRODUCT_MAX_DEPTH_M = float(np.iinfo(np.uint16).max) / 1000.0
PROFILES: dict[str, tuple[int, int, float]] = {
    "A": (960, 544, 0.00),
    "B": (960, 544, 0.25),
    "C": (960, 544, 0.50),
    "D": (1920, 1088, 0.00),
}
PROFILE_ALIASES = {
    "a": "A",
    "960x544-balance0": "A",
    "960x544-balance0.00": "A",
    "b": "B",
    "960x544-balance0.25": "B",
    "c": "C",
    "960x544-balance0.50": "C",
    "d": "D",
    "1920x1088-balance0": "D",
    "1920x1088-balance0.00": "D",
}
DEFAULT_QUALITY_THRESHOLDS = {
    "stereo_overlap_ratio_min": 0.60,
    "principal_point_delta_max_px": 1.0e-6,
    "baseline_error_max_m": 1.0e-9,
    "feature_inlier_count_min": 20,
    "vertical_residual_median_max_px": 0.5,
    "vertical_residual_p95_max_px": 1.5,
    "positive_disparity_ratio_min": 0.90,
}


@dataclass(frozen=True)
class Calibration:
    K0: np.ndarray
    D0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    image_size: tuple[int, int]
    right_T_left: np.ndarray
    baseline_m: float
    source_path: Path


@dataclass(frozen=True)
class Rectification:
    output_size: tuple[int, int]
    balance: float
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    left_maps: tuple[np.ndarray, np.ndarray]
    right_maps: tuple[np.ndarray, np.ndarray]
    left_valid_ratio: float
    right_valid_ratio: float
    stereo_overlap_ratio: float
    recovered_baseline_m: float


@dataclass(frozen=True)
class FrameRecord:
    source_idx: int
    sensor_time_ns: int
    left_sensor_time_ns: int
    right_sensor_time_ns: int
    left_timestamp_raw: str | None
    right_timestamp_raw: str | None
    left_video_frame_idx: int | None = None
    right_video_frame_idx: int | None = None

    @property
    def left_frame_idx(self) -> int:
        return self.source_idx if self.left_video_frame_idx is None else self.left_video_frame_idx

    @property
    def right_frame_idx(self) -> int:
        return self.source_idx if self.right_video_frame_idx is None else self.right_video_frame_idx


@dataclass(frozen=True)
class PoseRecord:
    source_idx: int
    sensor_time_ns: int
    pose_query_time_ns: int
    time_shift_ns: int
    world_T_cam0: np.ndarray
    pose_covariance: np.ndarray
    openvins_left_bracket_ns: int
    openvins_right_bracket_ns: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially decode selected Ego stereo MP4 frames, jointly "
            "rectify the fisheye pair, and write the prepared-stereo contract."
        )
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--frame-inventory", required=True, type=Path)
    parser.add_argument("--selected-indices", required=True, type=Path)
    parser.add_argument(
        "--selected-frames",
        type=Path,
        help=(
            "Selection metadata JSONL. Defaults to selected_frames.jsonl next "
            "to --selected-indices when present."
        ),
    )
    parser.add_argument("--camera-poses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--calibration",
        type=Path,
        help=(
            "Kalibr camchain YAML. Defaults to "
            "<dataset>/calibrations/camera-camchain.yaml."
        ),
    )
    parser.add_argument(
        "--left-video",
        type=Path,
        help="Defaults to <dataset>/videos/left.mp4.",
    )
    parser.add_argument(
        "--right-video",
        type=Path,
        help="Defaults to <dataset>/videos/right.mp4.",
    )
    parser.add_argument(
        "--profile",
        default="A",
        help=(
            "Rectification profile A/B/C/D. A-C are 960x544 with balance "
            "0/0.25/0.50; D is the 1920x1088 balance-0 reference."
        ),
    )
    parser.add_argument(
        "--quality-config",
        type=Path,
        help=(
            "Explicit quality-policy revision YAML. If omitted, the frozen "
            "Ego_Plan.md v1 thresholds are used."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help=(
            "Uniformly subsample at most this many selected frames while "
            "preserving ascending source order. Intended for smoke runs."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"Missing JSONL input: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        records.append(record)
    if not records:
        raise ValueError(f"No records in {path}")
    return records


def _first(record: dict[str, Any], names: Iterable[str], *, context: str) -> Any:
    for name in names:
        if name in record:
            return record[name]
    raise ValueError(f"{context} is missing one of {tuple(names)!r}")


def _integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer, not bool")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer for {context}: {value!r}") from error
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"Non-integral value for {context}: {value!r}")
    return result


def _matrix(value: Any, shape: tuple[int, int], *, context: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64).reshape(shape)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must have shape {shape}") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{context} contains NaN or Inf")
    return result


def _validate_transform(transform: np.ndarray, *, context: str) -> None:
    if transform.shape != (4, 4):
        raise ValueError(f"{context} must be 4x4")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError(f"{context} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-7):
        raise ValueError(f"{context} rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1.0e-7):
        raise ValueError(f"{context} rotation is not proper")


def load_frame_inventory(path: Path) -> dict[int, FrameRecord]:
    inventory: dict[int, FrameRecord] = {}
    for row, record in enumerate(_read_jsonl(path)):
        context = f"frame inventory row {row}"
        source_idx = _integer(
            _first(record, ("source_idx", "frame_index", "idx"), context=context),
            context=f"{context}.source_idx",
        )
        if source_idx < 0 or source_idx in inventory:
            raise ValueError(f"Duplicate or negative source_idx {source_idx} in {path}")
        retained = bool(record.get("retained", record.get("keep", True)))
        if not retained:
            continue
        left_time = _integer(
            _first(
                record,
                ("left_sensor_time_ns", "cam0_sensor_time_ns", "sensor_time_ns"),
                context=context,
            ),
            context=f"{context}.left_sensor_time_ns",
        )
        right_time = _integer(
            _first(
                record,
                ("right_sensor_time_ns", "cam1_sensor_time_ns", "sensor_time_ns"),
                context=context,
            ),
            context=f"{context}.right_sensor_time_ns",
        )
        sensor_time = _integer(
            record.get("sensor_time_ns", left_time),
            context=f"{context}.sensor_time_ns",
        )
        if sensor_time != left_time:
            raise ValueError(f"{context} sensor_time_ns must equal the left time")
        inventory[source_idx] = FrameRecord(
            source_idx=source_idx,
            left_video_frame_idx=_integer(
                record.get("left_video_frame_idx", source_idx),
                context=f"{context}.left_video_frame_idx",
            ),
            right_video_frame_idx=_integer(
                record.get("right_video_frame_idx", source_idx),
                context=f"{context}.right_video_frame_idx",
            ),
            sensor_time_ns=sensor_time,
            left_sensor_time_ns=left_time,
            right_sensor_time_ns=right_time,
            left_timestamp_raw=(
                str(record["left_timestamp_raw"])
                if record.get("left_timestamp_raw") is not None
                else None
            ),
            right_timestamp_raw=(
                str(record["right_timestamp_raw"])
                if record.get("right_timestamp_raw") is not None
                else None
            ),
        )
    if not inventory:
        raise ValueError("Frame inventory contains no retained stereo frames")
    return inventory


def load_selected_indices(path: Path) -> list[int]:
    if not path.is_file():
        raise ValueError(f"Missing selected-index input: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No selected source indices in {path}")
    if text.startswith("["):
        values = json.loads(text)
        if not isinstance(values, list):
            raise ValueError(f"Expected a JSON array in {path}")
    else:
        values = []
        for line_number, line in enumerate(text.splitlines(), 1):
            clean = line.split("#", 1)[0].strip()
            if not clean:
                continue
            parts = clean.replace(",", " ").split()
            if len(parts) != 1:
                raise ValueError(
                    f"Expected one source index at {path}:{line_number}"
                )
            values.append(parts[0])
    selected = [
        _integer(value, context=f"selected source index {position}")
        for position, value in enumerate(values)
    ]
    if not selected or selected[0] < 0:
        raise ValueError("Selected source indices must be non-empty and non-negative")
    if any(right <= left for left, right in zip(selected, selected[1:])):
        raise ValueError("Selected source indices must be unique and strictly ascending")
    return selected


def load_selection_metadata(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    metadata: dict[int, dict[str, Any]] = {}
    for row, record in enumerate(_read_jsonl(path)):
        context = f"selected frame row {row}"
        source_idx = _integer(
            _first(record, ("source_idx", "frame_index"), context=context),
            context=f"{context}.source_idx",
        )
        reason = record.get("selection_reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"{context} has no selection_reason")
        if source_idx in metadata:
            raise ValueError(f"Duplicate selection metadata source_idx {source_idx}")
        metadata[source_idx] = record
    return metadata


KEYFRAME_SELECTION_METADATA_FIELDS = (
    "selection_reason",
    "selection_reasons",
    "all_selection_reasons",
    "selected_motion_reasons",
    "trigger_source_idx",
    "trigger_selection_reason",
    "selected_source_idx",
    "translation_from_previous_selected_m",
    "rotation_from_previous_selected_deg",
    "gap_from_previous_selected_s",
    "quality_adjusted",
    "quality_score",
    "quality_metrics",
    "quality_window_source_indices",
    "candidate_window",
    "candidate_rejection_reasons",
    "quality_fallback_reason",
    "view_trigger_metrics",
    "view_selection_metrics",
    "frame_roles",
    "role_decisions",
    "use_for_depth",
    "use_for_odometry",
    "use_for_loop",
    "use_for_tsdf",
    "use_for_semantics",
    "map_window_index",
    "map_window_start_ns",
    "map_window_end_ns",
    "candidate_age_to_window_end_s",
)


def compact_keyframe_selection(record: dict[str, Any]) -> dict[str, Any]:
    """Preserve the S04 audit contract through every derived RGB-D dataset."""

    return {
        field: record[field]
        for field in KEYFRAME_SELECTION_METADATA_FIELDS
        if field in record
    }


def uniformly_limit(values: Sequence[int], maximum: int | None) -> list[int]:
    if maximum is None or maximum >= len(values):
        return list(values)
    if maximum <= 0:
        raise ValueError("--max-frames must be positive")
    if maximum == 1:
        return [values[0]]
    positions = np.linspace(0, len(values) - 1, num=maximum)
    indices = np.rint(positions).astype(np.int64)
    # linspace plus rounding is unique when maximum <= len(values), but keep
    # this explicit guard because duplicate smoke frames would break provenance.
    if len(set(int(index) for index in indices)) != maximum:
        raise RuntimeError("Internal uniform-subsampling error")
    return [values[int(index)] for index in indices]


def load_camera_poses(path: Path) -> dict[int, PoseRecord]:
    poses: dict[int, PoseRecord] = {}
    for row, record in enumerate(_read_jsonl(path)):
        context = f"camera pose row {row}"
        source_idx = _integer(
            _first(record, ("source_idx", "source_frame_idx", "frame_index"), context=context),
            context=f"{context}.source_idx",
        )
        sensor_time_ns = _integer(
            _first(record, ("sensor_time_ns", "pose_sensor_time_ns"), context=context),
            context=f"{context}.sensor_time_ns",
        )
        transform = _matrix(
            _first(
                record,
                ("world_T_cam0", "world_T_left_camera", "world_T_camera"),
                context=context,
            ),
            (4, 4),
            context=f"{context}.world_T_cam0",
        )
        _validate_transform(transform, context=f"{context}.world_T_cam0")
        covariance = _matrix(
            _first(
                record,
                ("pose_covariance", "covariance", "covariance_6x6"),
                context=context,
            ),
            (6, 6),
            context=f"{context}.pose_covariance",
        )
        if not np.allclose(covariance, covariance.T, atol=1.0e-8):
            raise ValueError(f"{context}.pose_covariance is not symmetric")
        if float(np.linalg.eigvalsh((covariance + covariance.T) * 0.5).min()) < -1.0e-8:
            raise ValueError(f"{context}.pose_covariance is not positive semidefinite")
        left_bracket = _integer(
            _first(
                record,
                ("openvins_left_bracket_ns", "left_bracket_ns"),
                context=context,
            ),
            context=f"{context}.openvins_left_bracket_ns",
        )
        right_bracket = _integer(
            _first(
                record,
                ("openvins_right_bracket_ns", "right_bracket_ns"),
                context=context,
            ),
            context=f"{context}.openvins_right_bracket_ns",
        )
        pose_query_time_ns = _integer(
            record.get("pose_query_time_ns", sensor_time_ns),
            context=f"{context}.pose_query_time_ns",
        )
        time_shift_ns = _integer(
            record.get("time_shift_ns", pose_query_time_ns - sensor_time_ns),
            context=f"{context}.time_shift_ns",
        )
        if pose_query_time_ns != sensor_time_ns + time_shift_ns:
            raise ValueError(f"{context} pose query time disagrees with camera time shift")
        if left_bracket > pose_query_time_ns or right_bracket < pose_query_time_ns:
            raise ValueError(
                f"{context} OpenVINS brackets do not contain the IMU-clock pose query time"
            )
        if source_idx < 0 or source_idx in poses:
            raise ValueError(f"Duplicate or negative camera-pose source_idx {source_idx}")
        poses[source_idx] = PoseRecord(
            source_idx=source_idx,
            sensor_time_ns=sensor_time_ns,
            pose_query_time_ns=pose_query_time_ns,
            time_shift_ns=time_shift_ns,
            world_T_cam0=transform,
            pose_covariance=covariance,
            openvins_left_bracket_ns=left_bracket,
            openvins_right_bracket_ns=right_bracket,
        )
    return poses


def _camera_matrix(intrinsics: Any, *, context: str) -> np.ndarray:
    values = _matrix(intrinsics, (1, 4), context=context).reshape(-1)
    fx, fy, cx, cy = (float(value) for value in values)
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError(f"{context} focal lengths must be positive")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def load_calibration(path: Path) -> Calibration:
    if not path.is_file():
        raise ValueError(f"Missing stereo calibration: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Invalid Kalibr camchain: {path}")
    try:
        cam0 = document["cam0"]
        cam1 = document["cam1"]
    except KeyError as error:
        raise ValueError("Kalibr camchain must contain cam0 and cam1") from error
    for name, camera in (("cam0", cam0), ("cam1", cam1)):
        if camera.get("camera_model") != "pinhole":
            raise ValueError(f"{name} camera_model must be pinhole")
        if camera.get("distortion_model") != "equidistant":
            raise ValueError(f"{name} distortion_model must be equidistant")
        if len(camera.get("distortion_coeffs", [])) != 4:
            raise ValueError(f"{name} must have four fisheye coefficients")
    K0 = _camera_matrix(cam0.get("intrinsics"), context="cam0.intrinsics")
    K1 = _camera_matrix(cam1.get("intrinsics"), context="cam1.intrinsics")
    D0 = _matrix(cam0.get("distortion_coeffs"), (4, 1), context="cam0.distortion")
    D1 = _matrix(cam1.get("distortion_coeffs"), (4, 1), context="cam1.distortion")
    resolution0 = tuple(int(value) for value in cam0.get("resolution", []))
    resolution1 = tuple(int(value) for value in cam1.get("resolution", []))
    if len(resolution0) != 2 or resolution0 != resolution1 or min(resolution0) <= 0:
        raise ValueError("cam0/cam1 must have the same positive resolution")
    right_T_left = _matrix(
        cam1.get("T_cn_cnm1"), (4, 4), context="cam1.T_cn_cnm1"
    )
    _validate_transform(right_T_left, context="cam1.T_cn_cnm1")
    baseline = float(np.linalg.norm(right_T_left[:3, 3]))
    if not math.isfinite(baseline) or baseline <= 0.0:
        raise ValueError("Stereo baseline must be finite and positive")
    return Calibration(
        K0=K0,
        D0=D0,
        K1=K1,
        D1=D1,
        image_size=(resolution0[0], resolution0[1]),
        right_T_left=right_T_left,
        baseline_m=baseline,
        source_path=path.resolve(),
    )


def resolve_profile(value: str) -> tuple[str, tuple[int, int], float]:
    normalized = value.strip()
    key = normalized.upper() if normalized.upper() in PROFILES else PROFILE_ALIASES.get(normalized.lower())
    if key not in PROFILES:
        raise ValueError(f"Unknown rectification profile {value!r}; expected A, B, C, or D")
    width, height, balance = PROFILES[key]
    return key, (width, height), balance


def build_rectification(
    calibration: Calibration,
    output_size: tuple[int, int],
    balance: float,
) -> Rectification:
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("Rectified output size must be positive")
    if not 0.0 <= balance <= 1.0:
        raise ValueError("Fisheye rectification balance must be in [0, 1]")
    rotation = calibration.right_T_left[:3, :3]
    translation = calibration.right_T_left[:3, 3]
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
        calibration.K0,
        calibration.D0,
        calibration.K1,
        calibration.D1,
        calibration.image_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=output_size,
        balance=float(balance),
        fov_scale=1.0,
    )
    left_maps = cv2.fisheye.initUndistortRectifyMap(
        calibration.K0,
        calibration.D0,
        R1,
        P1[:, :3],
        output_size,
        cv2.CV_32FC1,
    )
    right_maps = cv2.fisheye.initUndistortRectifyMap(
        calibration.K1,
        calibration.D1,
        R2,
        P2[:, :3],
        output_size,
        cv2.CV_32FC1,
    )
    source_valid = np.full(
        (calibration.image_size[1], calibration.image_size[0]), 255, dtype=np.uint8
    )
    left_valid = cv2.remap(
        source_valid,
        left_maps[0],
        left_maps[1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    right_valid = cv2.remap(
        source_valid,
        right_maps[0],
        right_maps[1],
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    recovered_baseline = abs(float(P2[0, 3] / P2[0, 0]))
    return Rectification(
        output_size=output_size,
        balance=balance,
        R1=np.asarray(R1, dtype=np.float64),
        R2=np.asarray(R2, dtype=np.float64),
        P1=np.asarray(P1, dtype=np.float64),
        P2=np.asarray(P2, dtype=np.float64),
        Q=np.asarray(Q, dtype=np.float64),
        left_maps=left_maps,
        right_maps=right_maps,
        left_valid_ratio=float(np.mean(left_valid)),
        right_valid_ratio=float(np.mean(right_valid)),
        stereo_overlap_ratio=float(np.mean(left_valid & right_valid)),
        recovered_baseline_m=recovered_baseline,
    )


def world_T_rectified_left(
    world_T_cam0: np.ndarray, left_original_to_rectified_R: np.ndarray
) -> np.ndarray:
    """Convert an original-left optical pose to the rectified-left frame.

    OpenCV returns ``p_rect = R1 @ p_original``.  A camera pose maps camera
    coordinates into world coordinates, so its right-hand factor is the
    inverse rotation ``original_R_rect = R1.T``.
    """

    source_pose = _matrix(world_T_cam0, (4, 4), context="world_T_cam0")
    _validate_transform(source_pose, context="world_T_cam0")
    R1 = _matrix(
        left_original_to_rectified_R,
        (3, 3),
        context="left_original_to_rectified_R",
    )
    if not np.allclose(R1.T @ R1, np.eye(3), atol=1.0e-7):
        raise ValueError("left_original_to_rectified_R is not orthonormal")
    cam0_T_rect0 = np.eye(4, dtype=np.float64)
    cam0_T_rect0[:3, :3] = R1.T
    result = source_pose @ cam0_T_rect0
    _validate_transform(result, context="world_T_rectified_left")
    return result


def rectify_pair(
    left: np.ndarray, right: np.ndarray, rectification: Rectification
) -> tuple[np.ndarray, np.ndarray]:
    rectified_left = cv2.remap(
        left,
        rectification.left_maps[0],
        rectification.left_maps[1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    rectified_right = cv2.remap(
        right,
        rectification.right_maps[0],
        rectification.right_maps[1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return rectified_left, rectified_right


def _ratio_matches(
    query: np.ndarray | None, train: np.ndarray | None, ratio: float = 0.75
) -> list[cv2.DMatch]:
    if query is None or train is None or len(query) < 2 or len(train) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return [
        first
        for pair in matcher.knnMatch(query, train, k=2)
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < ratio * second.distance
    ]


def epipolar_feature_observations(
    left: np.ndarray, right: np.ndarray
) -> dict[str, Any]:
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=10)
    keypoints_left, descriptors_left = orb.detectAndCompute(gray_left, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(gray_right, None)
    forward = _ratio_matches(descriptors_left, descriptors_right)
    reverse = _ratio_matches(descriptors_right, descriptors_left)
    reverse_pairs = {(match.trainIdx, match.queryIdx) for match in reverse}
    mutual = [
        match
        for match in forward
        if (match.queryIdx, match.trainIdx) in reverse_pairs
    ]
    if len(mutual) < 8:
        return {
            "detected_left": len(keypoints_left),
            "detected_right": len(keypoints_right),
            "mutual_ratio_matches": len(mutual),
            "inlier_matches": 0,
            "vertical_residuals_px": [],
            "disparities_px": [],
        }
    left_points = np.asarray(
        [keypoints_left[match.queryIdx].pt for match in mutual], dtype=np.float64
    )
    right_points = np.asarray(
        [keypoints_right[match.trainIdx].pt for match in mutual], dtype=np.float64
    )
    method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    _, mask = cv2.findFundamentalMat(
        left_points, right_points, method, 1.0, 0.999
    )
    if mask is None:
        inliers = np.zeros(len(mutual), dtype=bool)
    else:
        inliers = np.asarray(mask).reshape(-1).astype(bool)
    inlier_left = left_points[inliers]
    inlier_right = right_points[inliers]
    return {
        "detected_left": len(keypoints_left),
        "detected_right": len(keypoints_right),
        "mutual_ratio_matches": len(mutual),
        "inlier_matches": int(np.count_nonzero(inliers)),
        "vertical_residuals_px": np.abs(
            inlier_left[:, 1] - inlier_right[:, 1]
        ).tolist(),
        "disparities_px": (inlier_left[:, 0] - inlier_right[:, 0]).tolist(),
    }


def summarize_feature_observations(
    observations: Sequence[dict[str, Any]], sample_source_indices: Sequence[int]
) -> dict[str, Any]:
    vertical = np.asarray(
        [value for item in observations for value in item["vertical_residuals_px"]],
        dtype=np.float64,
    )
    disparities = np.asarray(
        [value for item in observations for value in item["disparities_px"]],
        dtype=np.float64,
    )
    if len(vertical):
        median = float(np.median(vertical))
        p95 = float(np.percentile(vertical, 95))
    else:
        median = None
        p95 = None
    positive_ratio = float(np.mean(disparities > 0.0)) if len(disparities) else None
    return {
        "method": "ORB mutual-ratio matches plus fundamental-matrix RANSAC",
        "sample_source_indices": [int(value) for value in sample_source_indices],
        "sample_count": len(observations),
        "detected_left": int(sum(item["detected_left"] for item in observations)),
        "detected_right": int(sum(item["detected_right"] for item in observations)),
        "mutual_ratio_matches": int(
            sum(item["mutual_ratio_matches"] for item in observations)
        ),
        "inlier_matches": int(sum(item["inlier_matches"] for item in observations)),
        "vertical_residual_median_px": median,
        "vertical_residual_p95_px": p95,
        "positive_disparity_ratio": positive_ratio,
    }


def _sample_positions(count: int, maximum: int) -> set[int]:
    if count <= maximum:
        return set(range(count))
    return {
        int(position)
        for position in np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64)
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integrity_entry(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256(path),
    }


def _create_output_tree(parent: Path) -> None:
    for relative in ("rgb", "stereo_right", "pose", "rectification_samples"):
        (parent / relative).mkdir(parents=True, exist_ok=False)


def _write_sample_montage(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    montage = np.concatenate((left, right), axis=1)
    for y in np.linspace(0, montage.shape[0] - 1, 9).astype(int):
        cv2.line(montage, (0, int(y)), (montage.shape[1] - 1, int(y)), (0, 255, 0), 1)
    if not cv2.imwrite(str(path), montage, PNG_OPTIONS):
        raise RuntimeError(f"Failed to write rectification sample {path}")


def load_quality_policy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "revision": "ego_plan_v1",
            "rationale": "Frozen initial thresholds from Ego_Plan.md.",
            "thresholds": dict(DEFAULT_QUALITY_THRESHOLDS),
            "source": None,
        }
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Missing rectification quality config: {resolved}")
    text = resolved.read_text(encoding="utf-8")
    if text.startswith("%YAML:1.0"):
        text = "\n".join(text.splitlines()[1:])
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError("Rectification quality config must be a mapping")
    thresholds = document.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Rectification quality config lacks thresholds")
    missing = sorted(set(DEFAULT_QUALITY_THRESHOLDS) - set(thresholds))
    extra = sorted(set(thresholds) - set(DEFAULT_QUALITY_THRESHOLDS))
    if missing or extra:
        raise ValueError(
            f"Rectification quality threshold keys differ: missing={missing}, extra={extra}"
        )
    normalized: dict[str, float | int] = {}
    for key, default in DEFAULT_QUALITY_THRESHOLDS.items():
        value = thresholds[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Rectification threshold {key} must be numeric")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"Rectification threshold {key} must be finite and non-negative")
        normalized[key] = int(value) if isinstance(default, int) else float(value)
    revision = document.get("revision")
    rationale = document.get("rationale")
    if not isinstance(revision, str) or not revision.strip():
        raise ValueError("Rectification quality config needs a revision")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Rectification quality config needs a rationale")
    return {
        "revision": revision,
        "rationale": rationale,
        "thresholds": normalized,
        "evidence": document.get("evidence"),
        "source": _integrity_entry(resolved),
    }


def _quality_gates(
    rectification: Rectification,
    calibration: Calibration,
    features: dict[str, Any],
    thresholds: Mapping[str, float | int] | None = None,
) -> dict[str, dict[str, Any]]:
    policy = dict(DEFAULT_QUALITY_THRESHOLDS if thresholds is None else thresholds)
    cx_delta = abs(float(rectification.P1[0, 2] - rectification.P2[0, 2]))
    cy_delta = abs(float(rectification.P1[1, 2] - rectification.P2[1, 2]))
    baseline_error = abs(rectification.recovered_baseline_m - calibration.baseline_m)
    median = features["vertical_residual_median_px"]
    p95 = features["vertical_residual_p95_px"]
    positive = features["positive_disparity_ratio"]
    return {
        "stereo_overlap_ratio": {
            "value": rectification.stereo_overlap_ratio,
            "minimum": policy["stereo_overlap_ratio_min"],
            "passed": rectification.stereo_overlap_ratio
            >= policy["stereo_overlap_ratio_min"],
        },
        "zero_disparity_principal_point": {
            "cx_delta_px": cx_delta,
            "cy_delta_px": cy_delta,
            "maximum_px": policy["principal_point_delta_max_px"],
            "passed": max(cx_delta, cy_delta)
            <= policy["principal_point_delta_max_px"],
        },
        "baseline_preservation": {
            "source_baseline_m": calibration.baseline_m,
            "recovered_baseline_m": rectification.recovered_baseline_m,
            "absolute_error_m": baseline_error,
            "maximum_error_m": policy["baseline_error_max_m"],
            "passed": baseline_error <= policy["baseline_error_max_m"],
        },
        "feature_evidence_count": {
            "value": int(features["inlier_matches"]),
            "minimum": policy["feature_inlier_count_min"],
            "passed": int(features["inlier_matches"])
            >= policy["feature_inlier_count_min"],
        },
        "vertical_residual_median_px": {
            "value": median,
            "maximum": policy["vertical_residual_median_max_px"],
            "passed": median is not None
            and float(median) <= policy["vertical_residual_median_max_px"],
        },
        "vertical_residual_p95_px": {
            "value": p95,
            "maximum": policy["vertical_residual_p95_max_px"],
            "passed": p95 is not None
            and float(p95) <= policy["vertical_residual_p95_max_px"],
        },
        "positive_disparity_ratio": {
            "value": positive,
            "minimum": policy["positive_disparity_ratio_min"],
            "passed": positive is not None
            and float(positive) >= policy["positive_disparity_ratio_min"],
        },
    }


def _publish_atomic(temporary: Path, output: Path, overwrite: bool) -> None:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")
    backup: Path | None = None
    try:
        if output.exists():
            backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
            os.replace(output, backup)
        os.replace(temporary, output)
    except BaseException:
        if backup is not None and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if backup is not None:
        if backup.is_dir():
            shutil.rmtree(backup)
        else:
            backup.unlink()


def prepare_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    dataset = args.dataset.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output == dataset:
        raise ValueError("Output must not replace the raw dataset directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite.")

    frame_inventory_path = args.frame_inventory.expanduser().resolve()
    selected_indices_path = args.selected_indices.expanduser().resolve()
    inferred_selection_metadata = selected_indices_path.with_name(
        "selected_frames.jsonl"
    )
    selection_metadata_path = (
        args.selected_frames.expanduser().resolve()
        if args.selected_frames is not None
        else inferred_selection_metadata
        if inferred_selection_metadata.is_file()
        else None
    )
    camera_poses_path = args.camera_poses.expanduser().resolve()
    calibration_path = (
        args.calibration.expanduser().resolve()
        if args.calibration is not None
        else dataset / "calibrations" / "camera-camchain.yaml"
    )
    left_video = (
        args.left_video.expanduser().resolve()
        if args.left_video is not None
        else dataset / "videos" / "left.mp4"
    )
    right_video = (
        args.right_video.expanduser().resolve()
        if args.right_video is not None
        else dataset / "videos" / "right.mp4"
    )
    for label, path in (("left video", left_video), ("right video", right_video)):
        if not path.is_file():
            raise ValueError(f"Missing {label}: {path}")

    inventory = load_frame_inventory(frame_inventory_path)
    selected_all = load_selected_indices(selected_indices_path)
    selected = uniformly_limit(selected_all, args.max_frames)
    selection_metadata = load_selection_metadata(selection_metadata_path)
    poses = load_camera_poses(camera_poses_path)
    for source_idx in selected:
        if source_idx not in inventory:
            raise ValueError(
                f"Selected source_idx {source_idx} is not retained in frame inventory"
            )
        if source_idx not in poses:
            raise ValueError(f"Selected source_idx {source_idx} has no camera pose")
        if selection_metadata and source_idx not in selection_metadata:
            raise ValueError(
                f"Selected source_idx {source_idx} has no selection metadata"
            )
        if poses[source_idx].sensor_time_ns != inventory[source_idx].sensor_time_ns:
            raise ValueError(
                f"Selected source_idx {source_idx} pose/inventory times disagree"
            )
        if inventory[source_idx].left_sensor_time_ns != inventory[source_idx].right_sensor_time_ns:
            raise ValueError(
                f"Selected source_idx {source_idx} left/right times are not identical"
            )
    selected_times = [inventory[index].sensor_time_ns for index in selected]
    if any(right <= left for left, right in zip(selected_times, selected_times[1:])):
        raise ValueError("Selected frame sensor times must be strictly increasing")

    calibration = load_calibration(calibration_path)
    profile_name, output_size, balance = resolve_profile(args.profile)
    rectification = build_rectification(calibration, output_size, balance)
    quality_policy = load_quality_policy(args.quality_config)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    quality_passed = False
    try:
        _create_output_tree(temporary)
        left_capture = cv2.VideoCapture(str(left_video))
        right_capture = cv2.VideoCapture(str(right_video))
        if not left_capture.isOpened() or not right_capture.isOpened():
            raise RuntimeError("Failed to open one or both stereo videos")
        reported_sizes = (
            (
                int(round(left_capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
                int(round(left_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            ),
            (
                int(round(right_capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
                int(round(right_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            ),
        )
        if any(size != calibration.image_size for size in reported_sizes):
            raise ValueError(
                "Video/calibration resolution mismatch: "
                f"videos={reported_sizes}, calibration={calibration.image_size}"
            )

        selected_lookup = {source_idx: row for row, source_idx in enumerate(selected)}
        diagnostic_positions = _sample_positions(len(selected), 20)
        montage_positions = _sample_positions(len(selected), 6)
        observations: list[dict[str, Any]] = []
        observation_source_indices: list[int] = []
        output_frames: list[dict[str, Any]] = []
        rectified_poses: list[np.ndarray] = []
        covariance_records: list[dict[str, Any]] = []
        source_index = 0
        maximum_source_idx = selected[-1]
        try:
            while source_index <= maximum_source_idx:
                left_ok, left = left_capture.read()
                right_ok, right = right_capture.read()
                if not left_ok or not right_ok:
                    raise RuntimeError(
                        "Stereo MP4 decode ended before selected source_idx "
                        f"{maximum_source_idx}; failed at {source_index}"
                    )
                if left.shape[:2] != (
                    calibration.image_size[1],
                    calibration.image_size[0],
                ) or right.shape[:2] != (
                    calibration.image_size[1],
                    calibration.image_size[0],
                ):
                    raise ValueError(f"Unexpected decoded dimensions at {source_index}")
                if source_index in selected_lookup:
                    frame = inventory[source_index]
                    if frame.left_frame_idx != source_index:
                        if not left_capture.set(
                            cv2.CAP_PROP_POS_FRAMES, frame.left_frame_idx
                        ):
                            raise RuntimeError(
                                f"Failed to seek left video to {frame.left_frame_idx}"
                            )
                        left_ok, left = left_capture.read()
                        left_capture.set(cv2.CAP_PROP_POS_FRAMES, source_index + 1)
                        if not left_ok:
                            raise RuntimeError(
                                f"Failed to decode aligned left frame {frame.left_frame_idx}"
                            )
                    if frame.right_frame_idx != source_index:
                        if not right_capture.set(
                            cv2.CAP_PROP_POS_FRAMES, frame.right_frame_idx
                        ):
                            raise RuntimeError(
                                f"Failed to seek right video to {frame.right_frame_idx}"
                            )
                        right_ok, right = right_capture.read()
                        right_capture.set(cv2.CAP_PROP_POS_FRAMES, source_index + 1)
                        if not right_ok:
                            raise RuntimeError(
                                f"Failed to decode aligned right frame {frame.right_frame_idx}"
                            )
                    output_idx = selected_lookup[source_index]
                    rectified_left, rectified_right = rectify_pair(
                        left, right, rectification
                    )
                    left_relative = Path("rgb") / f"{output_idx:08d}.png"
                    right_relative = Path("stereo_right") / f"{output_idx:08d}.png"
                    if not cv2.imwrite(
                        str(temporary / left_relative), rectified_left, PNG_OPTIONS
                    ):
                        raise RuntimeError(f"Failed to write {left_relative}")
                    if not cv2.imwrite(
                        str(temporary / right_relative), rectified_right, PNG_OPTIONS
                    ):
                        raise RuntimeError(f"Failed to write {right_relative}")
                    if output_idx in diagnostic_positions:
                        observations.append(
                            epipolar_feature_observations(
                                rectified_left, rectified_right
                            )
                        )
                        observation_source_indices.append(source_index)
                    if output_idx in montage_positions:
                        _write_sample_montage(
                            temporary
                            / "rectification_samples"
                            / f"source_{source_index:08d}.png",
                            rectified_left,
                            rectified_right,
                        )

                    pose = poses[source_index]
                    rectified_pose = world_T_rectified_left(
                        pose.world_T_cam0, rectification.R1
                    )
                    rectified_poses.append(rectified_pose)
                    output_frames.append(
                        {
                            "idx": output_idx,
                            "source_idx": source_index,
                            "left_video_frame_idx": frame.left_frame_idx,
                            "right_video_frame_idx": frame.right_frame_idx,
                            "pose_row": output_idx,
                            "cam0": str((output / left_relative).resolve()),
                            "cam1": str((output / right_relative).resolve()),
                            "rgb": str((output / left_relative).resolve()),
                            "stereo_right": str((output / right_relative).resolve()),
                            "timestamp": (
                                frame.sensor_time_ns - selected_times[0]
                            )
                            / 1.0e9,
                            "sensor_time_ns": frame.sensor_time_ns,
                            "cam0_sensor_time_ns": frame.left_sensor_time_ns,
                            "cam1_sensor_time_ns": frame.right_sensor_time_ns,
                            "pose_sensor_time_ns": frame.sensor_time_ns,
                            "pose_query_time_ns": pose.pose_query_time_ns,
                            "time_shift_ns": pose.time_shift_ns,
                            "stereo_delta_ms": abs(
                                frame.left_sensor_time_ns - frame.right_sensor_time_ns
                            )
                            / 1.0e6,
                            "openvins_left_bracket_ns": pose.openvins_left_bracket_ns,
                            "openvins_right_bracket_ns": pose.openvins_right_bracket_ns,
                            "left_timestamp_raw": frame.left_timestamp_raw,
                            "right_timestamp_raw": frame.right_timestamp_raw,
                            **(
                                {
                                    "selection_reason": selection_metadata[
                                        source_index
                                    ]["selection_reason"],
                                    "translation_from_previous_selected_m": selection_metadata[
                                        source_index
                                    ].get("translation_from_previous_selected_m"),
                                    "rotation_from_previous_selected_deg": selection_metadata[
                                        source_index
                                    ].get("rotation_from_previous_selected_deg"),
                                    "gap_from_previous_selected_s": selection_metadata[
                                        source_index
                                    ].get("gap_from_previous_selected_s"),
                                    "keyframe_selection": compact_keyframe_selection(
                                        selection_metadata[source_index]
                                    ),
                                    "frame_roles": selection_metadata[source_index].get(
                                        "frame_roles", {}
                                    ),
                                    "role_decisions": selection_metadata[source_index].get(
                                        "role_decisions", {}
                                    ),
                                }
                                if selection_metadata
                                else {}
                            ),
                        }
                    )
                    covariance_records.append(
                        {
                            "idx": output_idx,
                            "source_idx": source_index,
                            "sensor_time_ns": frame.sensor_time_ns,
                            "pose_query_time_ns": pose.pose_query_time_ns,
                            "time_shift_ns": pose.time_shift_ns,
                            "openvins_left_bracket_ns": pose.openvins_left_bracket_ns,
                            "openvins_right_bracket_ns": pose.openvins_right_bracket_ns,
                            "pose_covariance": pose.pose_covariance.tolist(),
                        }
                    )
                source_index += 1
        finally:
            left_capture.release()
            right_capture.release()

        if len(output_frames) != len(selected):
            raise RuntimeError("Decoded output count does not match selected frame count")
        pose_text = "".join(
            " ".join(f"{value:.17g}" for value in pose.reshape(-1)) + "\n"
            for pose in rectified_poses
        )
        (temporary / "pose" / "poses.txt").write_text(pose_text, encoding="utf-8")
        (temporary / "pose" / "pose_timestamps_ns.txt").write_text(
            "".join(f"{time_ns}\n" for time_ns in selected_times), encoding="utf-8"
        )
        (temporary / "pose" / "openvins_covariance.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in covariance_records
            ),
            encoding="utf-8",
        )

        intrinsics = rectification.P1[:, :3]
        camera_info = {
            "schema_version": SCHEMA_VERSION,
            "model": "pinhole",
            "width": output_size[0],
            "height": output_size[1],
            "intrinsics": intrinsics.tolist(),
            "right_intrinsics": rectification.P2[:, :3].tolist(),
            "distortion": [0.0, 0.0, 0.0, 0.0],
            "projection_model": "pinhole_rectified",
            "fx": float(intrinsics[0, 0]),
            "fy": float(intrinsics[1, 1]),
            "cx": float(intrinsics[0, 2]),
            "cy": float(intrinsics[1, 2]),
            "baseline_m": calibration.baseline_m,
            "depth_scale": 1000.0,
            "P1": rectification.P1.tolist(),
            "P2": rectification.P2.tolist(),
            "Q": rectification.Q.tolist(),
        }
        _write_json(temporary / "camera_info.json", camera_info)
        tick_index = {
            "schema_version": SCHEMA_VERSION,
            "dataset_id": dataset.name,
            "source": str(dataset),
            "projection_model": "pinhole",
            "rectification_model": "pinhole_rectified",
            "pose_convention": "world_T_rectified_left_camera",
            "pose_composition": "world_T_cam0 @ inverse(rect0_T_cam0)",
            "width": output_size[0],
            "height": output_size[1],
            "fx": camera_info["fx"],
            "fy": camera_info["fy"],
            "cx": camera_info["cx"],
            "cy": camera_info["cy"],
            "baseline": calibration.baseline_m,
            # The RGB-D product stores uint16 millimetres, whose representable
            # upper bound is 65.535 m.  Consumers such as Hydra still apply
            # their own, tighter mapping range (8 m for this dataset).
            "recommended_max_depth_m": PRODUCT_MAX_DEPTH_M,
            "time_origin_ns": selected_times[0],
            "timebase": {
                "clock": "sensor_time_ns",
                "unit": "ns",
                "timestamp_definition": "(sensor_time_ns - time_origin_ns) / 1e9",
            },
            "pose_time_alignment": {
                "method": "OpenVINS bracketed interpolation at cam0 time plus camera-to-IMU shift",
                "pose_timestamp_file": "pose/pose_timestamps_ns.txt",
                "pose_row_field": "pose_row",
                "camera_timestamp_semantics": "raw cam0 sensor clock",
                "openvins_query_timestamp_semantics": "cam0 sensor time + timeshift_cam_imu",
            },
            "rectification_provenance": {
                "profile": profile_name,
                "balance": balance,
                "left_original_to_rectified_rotation": rectification.R1.tolist(),
                "left_rectification_rotation": rectification.R1.tolist(),
                "right_original_to_rectified_rotation": rectification.R2.tolist(),
            },
            "frames": output_frames,
        }
        _write_json(temporary / "tick_index.json", tick_index)

        feature_report = summarize_feature_observations(
            observations, observation_source_indices
        )
        gates = _quality_gates(
            rectification,
            calibration,
            feature_report,
            quality_policy["thresholds"],
        )
        quality_passed = all(bool(gate["passed"]) for gate in gates.values())
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if quality_passed else "failed_quality_gate",
            "profile": profile_name,
            "profile_parameters": {
                "output_width": output_size[0],
                "output_height": output_size[1],
                "balance": balance,
                "flags": "CALIB_ZERO_DISPARITY",
            },
            "source_image_size": list(calibration.image_size),
            "source_intrinsics_left": calibration.K0.tolist(),
            "source_distortion_left": calibration.D0.reshape(-1).tolist(),
            "source_intrinsics_right": calibration.K1.tolist(),
            "source_distortion_right": calibration.D1.reshape(-1).tolist(),
            "right_T_left": calibration.right_T_left.tolist(),
            "source_baseline_m": calibration.baseline_m,
            "recovered_baseline_m": rectification.recovered_baseline_m,
            "R1_left_original_to_rectified": rectification.R1.tolist(),
            "R2_right_original_to_rectified": rectification.R2.tolist(),
            "P1": rectification.P1.tolist(),
            "P2": rectification.P2.tolist(),
            "Q": rectification.Q.tolist(),
            "valid_area": {
                "left_ratio": rectification.left_valid_ratio,
                "right_ratio": rectification.right_valid_ratio,
                "stereo_overlap_ratio": rectification.stereo_overlap_ratio,
            },
            "feature_epipolar_diagnostic": feature_report,
            "quality_gates": gates,
            "quality_policy": quality_policy,
            "selected_source_frame_count_before_max_frames": len(selected_all),
            "materialized_frame_count": len(selected),
            "max_frames_policy": (
                "uniform_across_selected_span"
                if args.max_frames is not None and args.max_frames < len(selected_all)
                else "all_selected_frames"
            ),
            "opencv_version": cv2.__version__,
        }
        _write_json(temporary / "rectification_report.json", report)
        integrity = {
            "schema_version": SCHEMA_VERSION,
            "inputs": {
                "left_video": _integrity_entry(left_video),
                "right_video": _integrity_entry(right_video),
                "frame_inventory": _integrity_entry(frame_inventory_path),
                "selected_indices": _integrity_entry(selected_indices_path),
                "selection_metadata": (
                    _integrity_entry(selection_metadata_path)
                    if selection_metadata_path is not None
                    else None
                ),
                "camera_poses": _integrity_entry(camera_poses_path),
                "calibration": _integrity_entry(calibration_path),
                "quality_config": quality_policy["source"],
            },
            "selected_source_indices": selected,
            "profile": profile_name,
            "quality_status": report["status"],
        }
        _write_json(temporary / "input_integrity.json", integrity)
        _publish_atomic(temporary, output, bool(args.overwrite))
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    report_path = output / "rectification_report.json"
    if not quality_passed:
        raise RuntimeError(
            "Rectification materialized atomically but failed its quality gates; "
            f"inspect {report_path} and try another --profile"
        )
    return report, output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, output = prepare_dataset(args)
    print(
        f"Prepared {report['materialized_frame_count']} rectified stereo frames "
        f"with profile {report['profile']} at {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
