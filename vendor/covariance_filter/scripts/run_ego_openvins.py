#!/usr/bin/env python3
"""Deterministic offline OpenVINS adapter for Ego stereo/IMU captures.

The module deliberately keeps ROS imports inside :func:`run_ros_replay`.  Its
configuration, timestamp, transform, interpolation, and output-contract helpers
therefore remain testable on hosts without ROS 2.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENVINS_ROOT = Path("/home/user/Code/open_vins")
DEFAULT_DYNAMIC_OVERRIDE = (
    REPOSITORY_ROOT / "config/ego_pipeline/dynamic_initializer_override.yaml"
)
DEFAULT_OPENVINS_BINARY = (
    REPOSITORY_ROOT
    / ".repro/ego_openvins_ws/install/ov_msckf/lib/ov_msckf/run_subscribe_msckf"
)
DEFAULT_RUNTIME_MANIFEST = Path("/opt/daaam-deploy/runtime-manifest.yaml")

UPSTREAM_OPENVINS_COMMIT = "69488123ed9362dd44b6f28e7f4680abbff1442b"
PINNED_OPENVINS_COMMIT = "404f0aebcc4becf2d140dc4bd5047f4db08643bc"
TRAVERSAL_ID = "ego"
LEFT_TOPIC = "/ego/left/image_raw"
RIGHT_TOPIC = "/ego/right/image_raw"
IMU_TOPIC = "/ego/imu"
ODOMETRY_TOPIC = "/ov_msckf/odomimu"
COMMON_TIME_SHIFT_SOURCE = "cam0"

NANOSECONDS_PER_SECOND = 1_000_000_000
OUTPUT_FILENAMES = (
    "openvins_odometry.jsonl",
    "openvins_odometry.jsonl.partial",
    "openvins_stdout.log",
    "openvins_stderr.log",
    "replay_metrics.json",
    "openvins_manifest.json",
    "vio_acceptance.json",
    "camera_pose_samples.jsonl",
    "trajectory_preview.ply",
    "trajectory_preview.png",
    "stage_manifest.json",
    ".complete",
    ".failed.json",
)


def decimal_seconds_to_ns(value: str) -> int:
    """Convert an epoch-seconds decimal string to exact integer nanoseconds."""

    try:
        decimal = Decimal(value.strip())
    except (InvalidOperation, AttributeError) as error:
        raise ValueError(f"invalid decimal timestamp: {value!r}") from error
    scaled = decimal * NANOSECONDS_PER_SECOND
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise ValueError(f"timestamp has sub-nanosecond precision: {value!r}")
    result = int(integral)
    if result <= 0:
        raise ValueError("sensor timestamps must be positive Unix epoch nanoseconds")
    return result


def stamp_to_ns(stamp: Any) -> int:
    """Convert a ROS ``builtin_interfaces/Time``-like object to nanoseconds."""

    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < NANOSECONDS_PER_SECOND:
        raise ValueError("ROS timestamp fields are invalid")
    result = seconds * NANOSECONDS_PER_SECOND + nanoseconds
    if result <= 0:
        raise ValueError("ROS timestamp must be positive absolute nanoseconds")
    return result


def _finite_array(
    value: Any, shape: tuple[int, ...], name: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return result.copy()


def validate_homogeneous_transform(value: Any, name: str) -> np.ndarray:
    """Validate a rigid 4x4 transform and return a defensive copy."""

    transform = _finite_array(value, (4, 4), name)
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-10):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise ValueError(f"{name} rotation is not orthogonal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(determinant, 1.0, abs_tol=1.0e-8):
        raise ValueError(f"{name} rotation determinant is {determinant}, not +1")
    return transform


def invert_transform(value: Any, name: str = "transform") -> np.ndarray:
    transform = validate_homogeneous_transform(value, name)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return result


def quaternion_xyzw_to_matrix(value: Sequence[float]) -> np.ndarray:
    quaternion = _finite_array(value, (4,), "quaternion_xyzw")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion_xyzw must be non-zero")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(value: Any) -> np.ndarray:
    rotation = _finite_array(value, (3, 3), "rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-8):
        raise ValueError("rotation is not orthogonal")
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray(
                [
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def slerp_xyzw(
    first: Sequence[float], second: Sequence[float], alpha: float
) -> np.ndarray:
    """Shortest-arc Hamilton quaternion SLERP."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("SLERP alpha must lie in [0, 1]")
    q0 = _finite_array(first, (4,), "first quaternion")
    q1 = _finite_array(second, (4,), "second quaternion")
    norm0 = float(np.linalg.norm(q0))
    norm1 = float(np.linalg.norm(q1))
    if min(norm0, norm1) <= 1.0e-12:
        raise ValueError("SLERP quaternions must be non-zero")
    q0 /= norm0
    q1 /= norm1
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    result = (
        math.sin((1.0 - alpha) * theta) / sin_theta * q0
        + math.sin(alpha * theta) / sin_theta * q1
    )
    return result / np.linalg.norm(result)


@dataclass(frozen=True)
class CameraFrame:
    source_idx: int
    sensor_time_ns: int
    left_video_frame_idx: int | None = None
    right_video_frame_idx: int | None = None

    @property
    def left_frame_idx(self) -> int:
        return self.source_idx if self.left_video_frame_idx is None else self.left_video_frame_idx

    @property
    def right_frame_idx(self) -> int:
        return self.source_idx if self.right_video_frame_idx is None else self.right_video_frame_idx


@dataclass(frozen=True)
class ImuSample:
    sensor_time_ns: int
    angular_velocity: tuple[float, float, float]
    linear_acceleration: tuple[float, float, float]


@dataclass(frozen=True)
class PoseSample:
    sensor_time_ns: int
    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    pose_covariance: np.ndarray
    covariance_psd_repaired: bool = field(init=False)
    covariance_min_eigenvalue_before: float = field(init=False)
    covariance_maximum_asymmetry: float = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.sensor_time_ns, bool) or self.sensor_time_ns <= 0:
            raise ValueError("pose timestamp must be a positive integer")
        position = _finite_array(self.position_m, (3,), "position_m")
        orientation = _finite_array(
            self.orientation_xyzw, (4,), "orientation_xyzw"
        )
        norm = float(np.linalg.norm(orientation))
        if norm <= 1.0e-12:
            raise ValueError("pose orientation must be non-zero")
        covariance = np.asarray(self.pose_covariance, dtype=np.float64)
        if covariance.shape == (36,):
            covariance = covariance.reshape(6, 6)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("pose covariance must be finite 6x6 or length 36")
        maximum_asymmetry = float(np.max(np.abs(covariance - covariance.T)))
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        minimum_eigenvalue = float(eigenvalues[0])
        spectral_scale = max(float(np.max(np.abs(eigenvalues))), 1.0e-12)
        repair_tolerance = max(1.0e-9, 1.0e-3 * spectral_scale)
        if minimum_eigenvalue < -repair_tolerance:
            raise ValueError(
                "pose covariance is materially indefinite: "
                f"minimum eigenvalue {minimum_eigenvalue:.9g}, "
                f"repair tolerance {repair_tolerance:.9g}"
            )
        repaired = minimum_eigenvalue < 0.0
        if repaired:
            covariance = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            covariance = 0.5 * (covariance + covariance.T)
        object.__setattr__(self, "sensor_time_ns", int(self.sensor_time_ns))
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "orientation_xyzw", orientation / norm)
        object.__setattr__(self, "pose_covariance", covariance)
        object.__setattr__(self, "covariance_psd_repaired", repaired)
        object.__setattr__(
            self, "covariance_min_eigenvalue_before", minimum_eigenvalue
        )
        object.__setattr__(
            self, "covariance_maximum_asymmetry", maximum_asymmetry
        )

    @property
    def world_T_imu(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = quaternion_xyzw_to_matrix(self.orientation_xyzw)
        result[:3, 3] = self.position_m
        return result


@dataclass(frozen=True)
class InterpolatedPose:
    sensor_time_ns: int
    position_m: np.ndarray
    orientation_xyzw: np.ndarray
    pose_covariance: np.ndarray
    left_bracket_ns: int
    right_bracket_ns: int
    alpha: float

    @property
    def world_T_imu(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = quaternion_xyzw_to_matrix(self.orientation_xyzw)
        result[:3, 3] = self.position_m
        return result


def interpolate_pose(
    samples: Sequence[PoseSample],
    sensor_time_ns: int,
    *,
    max_bracket_gap_ns: int = 50_000_000,
) -> InterpolatedPose:
    """Interpolate position/rotation without endpoint clamping or extrapolation."""

    if not samples:
        raise ValueError("cannot interpolate an empty pose stream")
    timestamps = [sample.sensor_time_ns for sample in samples]
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("pose timestamps must increase strictly")
    index = bisect.bisect_left(timestamps, int(sensor_time_ns))
    if index < len(samples) and timestamps[index] == sensor_time_ns:
        sample = samples[index]
        return InterpolatedPose(
            sensor_time_ns=int(sensor_time_ns),
            position_m=sample.position_m.copy(),
            orientation_xyzw=sample.orientation_xyzw.copy(),
            pose_covariance=sample.pose_covariance.copy(),
            left_bracket_ns=sample.sensor_time_ns,
            right_bracket_ns=sample.sensor_time_ns,
            alpha=0.0,
        )
    if index == 0 or index == len(samples):
        raise ValueError("requested pose lies outside OpenVINS temporal coverage")
    left = samples[index - 1]
    right = samples[index]
    rejected_times = getattr(samples, "rejected_time_ns", ())
    if any(left.sensor_time_ns <= t <= right.sensor_time_ns for t in rejected_times):
        raise ValueError("pose interpolation bracket crosses rejected covariance sample")
    gap_ns = right.sensor_time_ns - left.sensor_time_ns
    if gap_ns > max_bracket_gap_ns:
        raise ValueError(
            f"pose interpolation bracket gap {gap_ns} exceeds {max_bracket_gap_ns} ns"
        )
    alpha = (sensor_time_ns - left.sensor_time_ns) / gap_ns
    return InterpolatedPose(
        sensor_time_ns=int(sensor_time_ns),
        position_m=(1.0 - alpha) * left.position_m + alpha * right.position_m,
        orientation_xyzw=slerp_xyzw(
            left.orientation_xyzw, right.orientation_xyzw, alpha
        ),
        pose_covariance=(
            (1.0 - alpha) * left.pose_covariance + alpha * right.pose_covariance
        ),
        left_bracket_ns=left.sensor_time_ns,
        right_bracket_ns=right.sensor_time_ns,
        alpha=float(alpha),
    )


def world_T_camera(world_T_imu: Any, T_imu_cam: Any) -> np.ndarray:
    return validate_homogeneous_transform(
        validate_homogeneous_transform(world_T_imu, "world_T_imu")
        @ validate_homogeneous_transform(T_imu_cam, "T_imu_cam"),
        "world_T_camera",
    )


def apply_rectification_rotation(world_T_cam0: Any, R1: Any) -> np.ndarray:
    """Return ``world_T_rect0`` for OpenCV's ``p_rect0 = R1 @ p_cam0``."""

    world_T_cam0_array = validate_homogeneous_transform(
        world_T_cam0, "world_T_cam0"
    )
    rotation = _finite_array(R1, (3, 3), "R1")
    T_cam0_rect0 = np.eye(4, dtype=np.float64)
    T_cam0_rect0[:3, :3] = rotation.T
    return validate_homogeneous_transform(
        world_T_cam0_array @ T_cam0_rect0, "world_T_rect0"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if text.startswith("%YAML:1.0"):
        text = "\n".join(text.splitlines()[1:])
    document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return document


def load_initialization_overrides(
    profile: str, override_path: Path = DEFAULT_DYNAMIC_OVERRIDE
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load the explicit static/dynamic initializer A/B profile.

    OpenVINS gives declared ROS parameters priority over its estimator YAML.
    The fallback file is therefore small and auditable, while the frozen
    calibration and tracker settings remain in the primary configuration.
    """

    if profile == "static":
        return {}, None
    if profile != "dynamic":
        raise ValueError(f"unsupported initialization profile: {profile}")
    document = _read_yaml(override_path)
    if document.get("profile") != "dynamic":
        raise ValueError(f"dynamic override has the wrong profile: {override_path}")
    overrides = document.get("ros_parameter_overrides")
    if not isinstance(overrides, dict) or not overrides:
        raise ValueError(f"dynamic override has no ROS parameters: {override_path}")
    allowed = {
        "init_dyn_use",
        "init_dyn_mle_opt_calib",
        "init_dyn_mle_max_iter",
        "init_dyn_mle_max_time",
        "init_dyn_mle_max_threads",
        "init_dyn_num_pose",
        "init_dyn_min_deg",
        "init_dyn_inflation_ori",
        "init_dyn_inflation_vel",
        "init_dyn_inflation_bg",
        "init_dyn_inflation_ba",
        "init_dyn_min_rec_cond",
        "init_dyn_bias_g",
        "init_dyn_bias_a",
    }
    forbidden = sorted(set(overrides) - allowed)
    if forbidden:
        raise ValueError(
            f"dynamic override contains non-initializer parameters: {forbidden}"
        )
    if overrides.get("init_dyn_use") is not True:
        raise ValueError("dynamic profile must set init_dyn_use=true")
    provenance = {
        "path": str(override_path.resolve()),
        "sha256": _sha256(override_path),
        "parameters": dict(overrides),
    }
    return dict(overrides), provenance


class _OpenCVDumper(yaml.SafeDumper):
    """Emit the restricted YAML subset accepted by OpenCV FileStorage."""


def _represent_opencv_sequence(
    dumper: yaml.SafeDumper, values: list[Any]
) -> yaml.nodes.SequenceNode:
    # OpenCV 4.6 rejects PyYAML's indentation-less block sequences below a
    # mapping key (``key:\n- value``). Flow sequences are unambiguous for both
    # OpenCV FileStorage and PyYAML, including nested calibration matrices.
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", values, flow_style=True
    )


_OpenCVDumper.add_representer(list, _represent_opencv_sequence)


def _opencv_yaml_text(document: Mapping[str, Any]) -> str:
    body = yaml.dump(
        dict(document),
        Dumper=_OpenCVDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    return "%YAML:1.0\n---\n" + body


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda value: (
                value.item()
                if isinstance(value, np.generic)
                else value.tolist()
                if isinstance(value, np.ndarray)
                else (_ for _ in ()).throw(
                    TypeError(
                        f"Object of type {type(value).__name__} is not JSON serializable"
                    )
                )
            ),
        )
        + "\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def estimator_config_document() -> dict[str, Any]:
    """Return the estimator options frozen by Ego_Plan.md S01."""

    return {
        "verbosity": "INFO",
        "use_fej": True,
        "integration": "rk4",
        "use_stereo": True,
        "max_cameras": 2,
        "calib_cam_extrinsics": False,
        "calib_cam_intrinsics": False,
        "calib_cam_timeoffset": False,
        "calib_imu_intrinsics": False,
        "calib_imu_g_sensitivity": False,
        "max_clones": 11,
        "max_slam": 50,
        "max_slam_in_update": 25,
        "max_msckf_in_update": 40,
        "dt_slam_delay": 1,
        "gravity_mag": 9.81,
        "feat_rep_msckf": "GLOBAL_3D",
        "feat_rep_slam": "ANCHORED_MSCKF_INVERSE_DEPTH",
        "feat_rep_aruco": "ANCHORED_MSCKF_INVERSE_DEPTH",
        "try_zupt": False,
        "zupt_chi2_multipler": 0,
        "zupt_max_velocity": 0.1,
        "zupt_noise_multiplier": 10,
        "zupt_max_disparity": 0.5,
        "zupt_only_at_beginning": False,
        "init_window_time": 2.0,
        "init_imu_thresh": 1.5,
        "init_max_disparity": 10.0,
        "init_max_features": 50,
        "init_dyn_use": False,
        "init_dyn_mle_opt_calib": False,
        "init_dyn_mle_max_iter": 50,
        "init_dyn_mle_max_time": 0.05,
        "init_dyn_mle_max_threads": 6,
        "init_dyn_num_pose": 6,
        "init_dyn_min_deg": 10.0,
        "init_dyn_inflation_ori": 10,
        "init_dyn_inflation_vel": 100,
        "init_dyn_inflation_bg": 10,
        "init_dyn_inflation_ba": 100,
        "init_dyn_min_rec_cond": 1.0e-12,
        "init_dyn_bias_g": [0.0, 0.0, 0.0],
        "init_dyn_bias_a": [0.0, 0.0, 0.0],
        "record_timing_information": False,
        "record_timing_filepath": "/tmp/ego_openvins_timing.txt",
        "save_total_state": False,
        "filepath_est": "/tmp/ego_openvins_estimate.txt",
        "filepath_std": "/tmp/ego_openvins_estimate_std.txt",
        "filepath_gt": "/tmp/ego_openvins_groundtruth.txt",
        "use_klt": True,
        "num_pts": 250,
        "fast_threshold": 20,
        "grid_x": 5,
        "grid_y": 5,
        "min_px_dist": 10,
        "knn_ratio": 0.70,
        "track_frequency": 30.0,
        "downsample_cameras": True,
        "num_opencv_threads": 4,
        "histogram_method": "HISTOGRAM",
        "use_aruco": False,
        "num_aruco": 1024,
        "downsize_aruco": True,
        "up_msckf_sigma_px": 1,
        "up_msckf_chi2_multipler": 1,
        "up_slam_sigma_px": 1,
        "up_slam_chi2_multipler": 1,
        "up_aruco_sigma_px": 1,
        "up_aruco_chi2_multipler": 1,
        "use_mask": False,
        "relative_config_imu": "kalibr_imu_chain.yaml",
        "relative_config_imucam": "kalibr_imucam_chain.yaml",
    }


def build_capture_config_documents(
    source_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build estimator, camera/IMU, and IMU documents from source Kalibr YAML."""

    calibration_dir = source_dir / "calibrations"
    raw_cam = _read_yaml(calibration_dir / "camera_imu-camchain-imucam.yaml")
    raw_imu = _read_yaml(calibration_dir / "camera_imu-imu.yaml")
    if not all(key in raw_cam for key in ("cam0", "cam1")):
        raise ValueError("camera/IMU calibration must contain cam0 and cam1")
    if "imu0" not in raw_imu:
        raise ValueError("IMU calibration must contain imu0")

    T_cam_imu: dict[str, np.ndarray] = {}
    for camera_name in ("cam0", "cam1"):
        T_cam_imu[camera_name] = validate_homogeneous_transform(
            raw_cam[camera_name]["T_cam_imu"], f"{camera_name}.T_cam_imu"
        )
    T_cam1_cam0 = validate_homogeneous_transform(
        raw_cam["cam1"]["T_cn_cnm1"], "cam1.T_cn_cnm1"
    )
    chain_residual = (
        invert_transform(T_cam_imu["cam1"], "cam1.T_cam_imu")
        @ T_cam1_cam0
        @ T_cam_imu["cam0"]
    )
    chain_max_abs_error = float(np.max(np.abs(chain_residual - np.eye(4))))
    if chain_max_abs_error > 1.0e-8:
        raise ValueError(
            "camera chain disagrees with T_cam_imu; maximum residual "
            f"{chain_max_abs_error:.3e}"
        )

    left_shift = float(raw_cam["cam0"]["timeshift_cam_imu"])
    right_shift = float(raw_cam["cam1"]["timeshift_cam_imu"])
    imucam: dict[str, Any] = {}
    topics = {"cam0": LEFT_TOPIC, "cam1": RIGHT_TOPIC}
    overlaps = {"cam0": [1], "cam1": [0]}
    for camera_name in ("cam0", "cam1"):
        raw = raw_cam[camera_name]
        imucam[camera_name] = {
            "T_imu_cam": invert_transform(
                T_cam_imu[camera_name], f"{camera_name}.T_cam_imu"
            ).tolist(),
            "cam_overlaps": overlaps[camera_name],
            "camera_model": str(raw["camera_model"]),
            "distortion_coeffs": [float(value) for value in raw["distortion_coeffs"]],
            "distortion_model": str(raw["distortion_model"]),
            "intrinsics": [float(value) for value in raw["intrinsics"]],
            "resolution": [int(value) for value in raw["resolution"]],
            "rostopic": topics[camera_name],
            # OpenVINS reads cam0 only.  Both entries state the selected common
            # value so the generated artifact cannot imply independent offsets.
            "timeshift_cam_imu": left_shift,
        }

    imu_source = raw_imu["imu0"]
    imu = {
        "imu0": {
            "T_i_b": np.eye(4, dtype=np.float64).tolist(),
            "accelerometer_noise_density": float(
                imu_source["accelerometer_noise_density"]
            ),
            "accelerometer_random_walk": float(
                imu_source["accelerometer_random_walk"]
            ),
            "gyroscope_noise_density": float(
                imu_source["gyroscope_noise_density"]
            ),
            "gyroscope_random_walk": float(
                imu_source["gyroscope_random_walk"]
            ),
            "rostopic": IMU_TOPIC,
            "time_offset": float(imu_source.get("time_offset", 0.0)),
            "update_rate": float(imu_source["update_rate"]),
            "model": "kalibr",
            # Use separate lists so PyYAML does not emit anchors, which are not
            # part of OpenCV FileStorage's intentionally small YAML dialect.
            "Tw": np.eye(3, dtype=np.float64).tolist(),
            "R_IMUtoGYRO": np.eye(3, dtype=np.float64).tolist(),
            "Ta": np.eye(3, dtype=np.float64).tolist(),
            "R_IMUtoACC": np.eye(3, dtype=np.float64).tolist(),
            "Tg": np.zeros((3, 3), dtype=np.float64).tolist(),
        }
    }
    provenance = {
        "common_timeshift_cam_imu_s": left_shift,
        "common_timeshift_source": COMMON_TIME_SHIFT_SOURCE,
        "source_cam0_timeshift_cam_imu_s": left_shift,
        "source_cam1_timeshift_cam_imu_s": right_shift,
        "source_offset_residual_s": right_shift - left_shift,
        "camera_chain_max_abs_error": chain_max_abs_error,
    }
    return estimator_config_document(), imucam, imu, provenance


def generate_capture_configs(source_dir: Path, config_dir: Path) -> dict[str, Any]:
    estimator, imucam, imu, provenance = build_capture_config_documents(source_dir)
    paths = {
        "estimator": config_dir / "estimator_config.yaml",
        "imucam": config_dir / "kalibr_imucam_chain.yaml",
        "imu": config_dir / "kalibr_imu_chain.yaml",
    }
    for key, document in (
        ("estimator", estimator),
        ("imucam", imucam),
        ("imu", imu),
    ):
        _atomic_write_text(paths[key], _opencv_yaml_text(document))
    validation = validate_capture_configs(source_dir, config_dir)
    return {**provenance, **validation}


def validate_capture_configs(source_dir: Path, config_dir: Path) -> dict[str, Any]:
    expected_estimator, _, _, provenance = build_capture_config_documents(source_dir)
    estimator_path = config_dir / "estimator_config.yaml"
    imucam_path = config_dir / "kalibr_imucam_chain.yaml"
    imu_path = config_dir / "kalibr_imu_chain.yaml"
    for path in (estimator_path, imucam_path, imu_path):
        if not path.is_file():
            raise FileNotFoundError(f"required OpenVINS config is missing: {path}")
    estimator = _read_yaml(estimator_path)
    imucam = _read_yaml(imucam_path)
    imu = _read_yaml(imu_path)

    frozen_keys = (
        "use_fej",
        "integration",
        "use_stereo",
        "max_cameras",
        "calib_cam_extrinsics",
        "calib_cam_intrinsics",
        "calib_cam_timeoffset",
        "calib_imu_intrinsics",
        "calib_imu_g_sensitivity",
        "track_frequency",
        "use_klt",
        "use_aruco",
        "try_zupt",
        "num_pts",
        "grid_x",
        "grid_y",
        "fast_threshold",
        "min_px_dist",
        "histogram_method",
        "downsample_cameras",
        "relative_config_imu",
        "relative_config_imucam",
    )
    mismatched = [
        key
        for key in frozen_keys
        if estimator.get(key) != expected_estimator.get(key)
    ]
    if mismatched:
        raise ValueError(f"estimator config violates frozen S01 keys: {mismatched}")

    raw_cam = _read_yaml(
        source_dir / "calibrations/camera_imu-camchain-imucam.yaml"
    )
    for camera_name, topic in (("cam0", LEFT_TOPIC), ("cam1", RIGHT_TOPIC)):
        generated = imucam.get(camera_name)
        if not isinstance(generated, dict):
            raise ValueError(f"generated camera config lacks {camera_name}")
        expected_transform = invert_transform(
            raw_cam[camera_name]["T_cam_imu"],
            f"source {camera_name}.T_cam_imu",
        )
        actual_transform = validate_homogeneous_transform(
            generated["T_imu_cam"], f"generated {camera_name}.T_imu_cam"
        )
        if not np.allclose(actual_transform, expected_transform, atol=1.0e-10):
            raise ValueError(f"generated {camera_name}.T_imu_cam is not source inverse")
        if generated.get("rostopic") != topic:
            raise ValueError(f"generated {camera_name} topic is incorrect")
        if generated.get("distortion_model") != "equidistant":
            raise ValueError(f"generated {camera_name} must retain equidistant model")

    left_shift = float(raw_cam["cam0"]["timeshift_cam_imu"])
    if not math.isclose(
        float(imucam["cam0"]["timeshift_cam_imu"]),
        left_shift,
        abs_tol=1.0e-15,
    ):
        raise ValueError("generated config must use the source cam0 time shift")
    if not math.isclose(
        float(imucam["cam1"]["timeshift_cam_imu"]),
        left_shift,
        abs_tol=1.0e-15,
    ):
        raise ValueError("generated cam1 must declare the selected common cam0 shift")

    generated_imu = imu.get("imu0")
    if not isinstance(generated_imu, dict) or generated_imu.get("rostopic") != IMU_TOPIC:
        raise ValueError("generated IMU topic is incorrect")
    for matrix_name, expected in (
        ("Tw", np.eye(3)),
        ("Ta", np.eye(3)),
        ("R_IMUtoGYRO", np.eye(3)),
        ("R_IMUtoACC", np.eye(3)),
        ("Tg", np.zeros((3, 3))),
    ):
        if not np.allclose(generated_imu.get(matrix_name), expected, atol=1.0e-15):
            raise ValueError(f"generated imu0.{matrix_name} is invalid")

    return {
        **provenance,
        "config_files": {
            path.name: {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in (estimator_path, imucam_path, imu_path)
        },
        "validated": True,
    }


def read_camera_frames(
    source_dir: Path,
    time_shift_s: float,
    *,
    maximum_source_frames: int | None = None,
) -> tuple[list[CameraFrame], dict[str, Any]]:
    """Pair cameras by source index and reject mismatched/invalid timestamps."""

    def read_side(path: Path) -> list[tuple[int, int, str]]:
        rows: list[tuple[int, int, str]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                fields = line.split()
                if len(fields) != 2:
                    raise ValueError(f"{path}:{line_number}: expected index and timestamp")
                index = int(fields[0])
                if index != len(rows):
                    raise ValueError(f"{path}:{line_number}: non-contiguous frame index")
                rows.append((index, decimal_seconds_to_ns(fields[1]), fields[1]))
        if not rows:
            raise ValueError(f"timestamp file is empty: {path}")
        return rows

    left = read_side(source_dir / "timestamps/left_timestamps.txt")
    right = read_side(source_dir / "timestamps/right_timestamps.txt")
    if len(left) != len(right):
        raise ValueError("left/right timestamp files have different row counts")
    source_pair_count = len(left)
    video_without_timestamp_indices: list[int] = []
    if maximum_source_frames is not None:
        trailing_count = len(left) - maximum_source_frames
        if trailing_count > 1:
            raise ValueError(
                "timestamp/video frame counts differ by more than one trailing pair"
            )
        if trailing_count > 0:
            missing_video_indices = list(range(maximum_source_frames, len(left)))
            left = left[:maximum_source_frames]
            right = right[:maximum_source_frames]
        else:
            # Sidecar indices are a validated contiguous prefix, so a longer
            # bilateral MP4 can only contribute a trailing decode-order tail.
            # It has no authoritative absolute timestamps and is excluded;
            # MP4 PTS must never be promoted to sensor time.
            missing_video_indices = []
            video_without_timestamp_indices = list(
                range(len(left), maximum_source_frames)
            )
    else:
        missing_video_indices = []
    imu = read_imu_samples(source_dir / "imu/imu0.csv")
    first_imu_ns = imu[0].sensor_time_ns
    last_imu_ns = imu[-1].sensor_time_ns
    shift_ns = int(round(time_shift_s * NANOSECONDS_PER_SECOND))

    retained: list[CameraFrame] = []
    dropped: dict[str, list[int]] = {
        "stereo_timestamp_mismatch": [],
        "non_monotonic_stereo_timestamp": [],
        "imu_coverage_start": [],
        "imu_coverage_end": [],
        "missing_video_frame": missing_video_indices,
        "missing_timestamp_sidecar": video_without_timestamp_indices,
    }
    previous_ns: int | None = None
    for left_row, right_row in zip(left, right):
        index, left_ns, _ = left_row
        if index != right_row[0] or left_ns != right_row[1]:
            dropped["stereo_timestamp_mismatch"].append(index)
            continue
        if previous_ns is not None and left_ns <= previous_ns:
            dropped["non_monotonic_stereo_timestamp"].append(index)
            continue
        shifted_ns = left_ns + shift_ns
        if shifted_ns < first_imu_ns:
            dropped["imu_coverage_start"].append(index)
            continue
        if shifted_ns > last_imu_ns:
            dropped["imu_coverage_end"].append(index)
            continue
        retained.append(CameraFrame(index, left_ns))
        previous_ns = left_ns
    if not retained:
        raise ValueError("no valid stereo frames remain after time/IMU coverage gates")
    report = {
        "source_pair_count": source_pair_count,
        "source_video_pair_count": maximum_source_frames,
        "decodable_pair_count": len(left),
        "retained_pair_count": len(retained),
        "dropped_by_reason": dropped,
        "common_timeshift_cam_imu_s": time_shift_s,
        "first_retained_sensor_time_ns": retained[0].sensor_time_ns,
        "last_retained_sensor_time_ns": retained[-1].sensor_time_ns,
    }
    return retained, report


def read_camera_frames_from_inventory(path: Path) -> tuple[list[CameraFrame], dict[str, Any]]:
    """Load the audit-authoritative stereo pairing, including side offsets."""

    frames: list[CameraFrame] = []
    source_pair_count = 0
    previous_source_idx: int | None = None
    previous_sensor_time_ns: int | None = None
    previous_left_frame_idx: int | None = None
    previous_right_frame_idx: int | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            source_pair_count += 1
            if not bool(record.get("retained", False)):
                continue
            source_idx = int(record["source_idx"])
            sensor_time_ns = int(record["left_sensor_time_ns"])
            right_sensor_time_ns = int(record["right_sensor_time_ns"])
            left_frame_idx = int(record["left_video_frame_idx"])
            right_frame_idx = int(record["right_video_frame_idx"])
            if sensor_time_ns != right_sensor_time_ns:
                raise ValueError(
                    f"{path}:{line_number}: retained stereo timestamps disagree"
                )
            if previous_source_idx is not None and source_idx <= previous_source_idx:
                raise ValueError(f"{path}:{line_number}: source indices are not increasing")
            if previous_sensor_time_ns is not None and sensor_time_ns <= previous_sensor_time_ns:
                raise ValueError(f"{path}:{line_number}: sensor times are not increasing")
            if previous_left_frame_idx is not None and left_frame_idx <= previous_left_frame_idx:
                raise ValueError(f"{path}:{line_number}: left frame indices are not increasing")
            if previous_right_frame_idx is not None and right_frame_idx <= previous_right_frame_idx:
                raise ValueError(f"{path}:{line_number}: right frame indices are not increasing")
            frames.append(
                CameraFrame(
                    source_idx=source_idx,
                    sensor_time_ns=sensor_time_ns,
                    left_video_frame_idx=left_frame_idx,
                    right_video_frame_idx=right_frame_idx,
                )
            )
            previous_source_idx = source_idx
            previous_sensor_time_ns = sensor_time_ns
            previous_left_frame_idx = left_frame_idx
            previous_right_frame_idx = right_frame_idx
    if not frames:
        raise ValueError(f"frame inventory contains no retained stereo frames: {path}")
    return frames, {
        "source_pair_count": source_pair_count,
        "retained_pair_count": len(frames),
        "pairing_authority": str(path),
        "first_retained_sensor_time_ns": frames[0].sensor_time_ns,
        "last_retained_sensor_time_ns": frames[-1].sensor_time_ns,
        "left_first_video_frame_idx": frames[0].left_frame_idx,
        "left_last_video_frame_idx": frames[-1].left_frame_idx,
        "right_first_video_frame_idx": frames[0].right_frame_idx,
        "right_last_video_frame_idx": frames[-1].right_frame_idx,
    }


def probe_stereo_video_frame_count(source_dir: Path) -> int | None:
    """Return the complete stereo-prefix length for the two source videos.

    A one-frame count difference is recoverable only as a trailing encoder
    flush: replay stops at the shorter stream.  Larger disagreements fail
    closed because an interior camera loss cannot be ruled out.
    """

    left_path = source_dir / "videos/left.mp4"
    right_path = source_dir / "videos/right.mp4"
    if not left_path.is_file() or not right_path.is_file():
        return None
    try:
        import cv2
    except ImportError:
        return None
    counts: list[int] = []
    for path in (left_path, right_path):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"cannot open stereo source video: {path}")
        try:
            count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            capture.release()
        if count <= 0:
            raise ValueError(f"invalid source video frame count: {path}")
        counts.append(count)
    if abs(counts[0] - counts[1]) > 1:
        raise ValueError(f"left/right video frame counts differ: {counts}")
    return min(counts)


def read_imu_samples(path: Path) -> list[ImuSample]:
    samples: list[ImuSample] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "timestamp",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
            "linear_acceleration_x",
            "linear_acceleration_y",
            "linear_acceleration_z",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"IMU CSV lacks required columns: {path}")
        previous_ns: int | None = None
        for row_number, row in enumerate(reader, start=2):
            timestamp_ns = decimal_seconds_to_ns(row["timestamp"])
            if previous_ns is not None and timestamp_ns <= previous_ns:
                raise ValueError(f"{path}:{row_number}: IMU time is not strictly increasing")
            gyro = tuple(float(row[f"angular_velocity_{axis}"]) for axis in "xyz")
            accel = tuple(float(row[f"linear_acceleration_{axis}"]) for axis in "xyz")
            if not np.all(np.isfinite((*gyro, *accel))):
                raise ValueError(f"{path}:{row_number}: IMU contains NaN or Inf")
            samples.append(ImuSample(timestamp_ns, gyro, accel))
            previous_ns = timestamp_ns
    if not samples:
        raise ValueError(f"IMU CSV is empty: {path}")
    return samples


def select_replay_window(
    frames: Sequence[CameraFrame],
    *,
    start_time_s: float = 0.0,
    max_duration_s: float | None = None,
    max_frames: int | None = None,
) -> list[CameraFrame]:
    if not frames:
        raise ValueError("no camera frames are available")
    if start_time_s < 0.0:
        raise ValueError("start_time_s must be non-negative")
    if max_duration_s is not None and max_duration_s <= 0.0:
        raise ValueError("max_duration_s must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    origin_ns = frames[0].sensor_time_ns
    start_ns = origin_ns + int(round(start_time_s * NANOSECONDS_PER_SECOND))
    selected = [frame for frame in frames if frame.sensor_time_ns >= start_ns]
    if not selected:
        raise ValueError("start_time_s lies beyond the camera capture")
    if max_duration_s is not None:
        end_ns = selected[0].sensor_time_ns + int(
            round(max_duration_s * NANOSECONDS_PER_SECOND)
        )
        selected = [frame for frame in selected if frame.sensor_time_ns <= end_ns]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("the requested replay window contains no camera frames")
    return selected


def inspect_openvins_checkout(
    openvins_root: Path,
    *,
    runtime_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify OpenVINS provenance from Git or an immutable runtime manifest.

    Development checkouts retain the stronger Git cleanliness, ancestry, and
    remote checks below. Deployment images intentionally omit VCS object stores;
    there the release-checksummed manifest is the provenance authority.
    """

    git_checkout_usable = (openvins_root / ".git").exists() and (
        subprocess.run(
            ["git", "-C", str(openvins_root), "rev-parse", "HEAD"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    if not git_checkout_usable:
        manifest_path = runtime_manifest_path or Path(
            os.environ.get("DAAAM_RUNTIME_MANIFEST", DEFAULT_RUNTIME_MANIFEST)
        )
        if not manifest_path.is_file():
            raise ValueError(
                f"OpenVINS root is not a Git checkout and the runtime manifest "
                f"is missing: {openvins_root}"
            )
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        try:
            repository = manifest["repositories"]["open_vins"]
            commit = str(repository["commit"])
            url = str(repository["url"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"runtime manifest has no OpenVINS provenance: {manifest_path}"
            ) from error
        if commit != PINNED_OPENVINS_COMMIT:
            raise ValueError(
                f"runtime OpenVINS commit {commit} does not match pinned "
                f"{PINNED_OPENVINS_COMMIT}"
            )
        if "neo128/open_vins" not in url:
            raise ValueError("runtime OpenVINS source must be the neo128/open_vins fork")
        return {
            "root": str(openvins_root.resolve()),
            "commit": commit,
            "describe": f"{commit[:12]}-runtime-image",
            "dirty": False,
            "upstream_base_commit": UPSTREAM_OPENVINS_COMMIT,
            "upstream_merge_base": UPSTREAM_OPENVINS_COMMIT,
            "fork_commits_ahead": 1,
            "remotes": {
                "origin": {"fetch": url, "push": url},
                "upstream": {
                    "fetch": "https://github.com/rpng/open_vins.git",
                    "push": "https://github.com/rpng/open_vins.git",
                },
            },
            "provenance": {
                "authority": "immutable_runtime_manifest",
                "manifest": str(manifest_path.resolve()),
            },
        }

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(openvins_root), *arguments],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()

    commit = git("rev-parse", "HEAD")
    dirty_lines = git("status", "--porcelain").splitlines()
    if commit != PINNED_OPENVINS_COMMIT:
        raise ValueError(
            f"OpenVINS commit {commit} does not match pinned {PINNED_OPENVINS_COMMIT}"
        )
    if dirty_lines:
        raise ValueError("OpenVINS checkout must be clean for a reproducible run")
    merge_base = git("merge-base", commit, UPSTREAM_OPENVINS_COMMIT)
    if merge_base != UPSTREAM_OPENVINS_COMMIT:
        raise ValueError(
            "Pinned OpenVINS fork commit does not descend from the audited upstream base"
        )
    remotes: dict[str, dict[str, str]] = {}
    for line in git("remote", "-v").splitlines():
        fields = line.split()
        if len(fields) >= 3:
            remote_name, url, operation = fields[:3]
            remotes.setdefault(remote_name, {})[operation.strip("()")] = url
    origin_urls = set(remotes.get("origin", {}).values())
    upstream_urls = set(remotes.get("upstream", {}).values())
    if not any("neo128/open_vins" in url for url in origin_urls):
        raise ValueError("OpenVINS origin must be the writable neo128/open_vins fork")
    if not any("rpng/open_vins" in url for url in upstream_urls):
        raise ValueError("OpenVINS upstream must be rpng/open_vins")
    return {
        "root": str(openvins_root.resolve()),
        "commit": commit,
        "describe": git("describe", "--always", "--dirty"),
        "dirty": False,
        "upstream_base_commit": UPSTREAM_OPENVINS_COMMIT,
        "upstream_merge_base": merge_base,
        "fork_commits_ahead": int(
            git("rev-list", "--count", f"{UPSTREAM_OPENVINS_COMMIT}..{commit}")
        ),
        "remotes": remotes,
    }


def odometry_record_from_message(
    message: Any, *, traversal_id: str = TRAVERSAL_ID
) -> dict[str, Any]:
    sensor_time_ns = stamp_to_ns(message.header.stamp)
    covariance = [float(value) for value in message.pose.covariance]
    if len(covariance) != 36 or not np.all(np.isfinite(covariance)):
        raise ValueError("OpenVINS pose covariance must contain 36 finite values")
    position = [
        float(message.pose.pose.position.x),
        float(message.pose.pose.position.y),
        float(message.pose.pose.position.z),
    ]
    orientation = [
        float(message.pose.pose.orientation.x),
        float(message.pose.pose.orientation.y),
        float(message.pose.pose.orientation.z),
        float(message.pose.pose.orientation.w),
    ]
    if not np.all(np.isfinite((*position, *orientation))):
        raise ValueError("OpenVINS pose contains NaN or Inf")
    if float(np.linalg.norm(orientation)) <= 1.0e-12:
        raise ValueError("OpenVINS pose quaternion is zero")
    frame_id = str(message.header.frame_id)
    child_frame_id = str(message.child_frame_id)
    if frame_id != "global" or child_frame_id != "imu":
        raise ValueError(
            f"unexpected OpenVINS frames: {frame_id!r} -> {child_frame_id!r}"
        )
    return {
        "sensor_time_ns": sensor_time_ns,
        "pose_time_ns": sensor_time_ns,
        "frame_id": frame_id,
        "child_frame_id": child_frame_id,
        "position_m": position,
        "orientation_xyzw": orientation,
        "pose_covariance": covariance,
        "lookup_semantics": "message_stamp",
        "source_topic": ODOMETRY_TOPIC,
        "traversal_id": traversal_id,
    }


class AtomicJsonlWriter:
    """One-write-per-record O_APPEND writer, atomically renamed on completion."""

    def __init__(self, final_path: Path) -> None:
        self.final_path = final_path
        self.partial_path = final_path.with_name(final_path.name + ".partial")
        self._descriptor = os.open(
            self.partial_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o644,
        )
        self._lock = threading.Lock()

    def write(self, record: Mapping[str, Any]) -> None:
        payload = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        with self._lock:
            written = os.write(self._descriptor, payload)
            if written != len(payload):
                raise OSError("short JSONL append")

    def complete(self) -> None:
        with self._lock:
            if self._descriptor >= 0:
                os.fsync(self._descriptor)
                os.close(self._descriptor)
                self._descriptor = -1
        os.replace(self.partial_path, self.final_path)

    def abort(self) -> None:
        with self._lock:
            if self._descriptor >= 0:
                os.close(self._descriptor)
                self._descriptor = -1


def wait_for_input_subscribers(
    publishers: Mapping[str, Any],
    process: Any,
    timeout_s: float,
    *,
    spin_once: Callable[[], None] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    monotonic_fn: Callable[[], float] = time.monotonic,
) -> dict[str, int]:
    """Wait until every input publisher has at least one matched subscriber."""

    deadline = monotonic_fn() + timeout_s
    while True:
        counts = {
            name: int(publisher.get_subscription_count())
            for name, publisher in publishers.items()
        }
        if counts and all(value > 0 for value in counts.values()):
            return counts
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"OpenVINS exited with code {return_code} before subscriber handshake"
            )
        if monotonic_fn() >= deadline:
            missing = [name for name, count in counts.items() if count == 0]
            raise TimeoutError(f"OpenVINS input subscriber handshake timed out: {missing}")
        if spin_once is not None:
            spin_once()
        sleep_fn(0.05)


class ReplayClock:
    def __init__(self, first_sensor_time_ns: int, rate: float) -> None:
        if rate <= 0.0:
            raise ValueError("replay rate must be positive")
        self.first_sensor_time_ns = first_sensor_time_ns
        self.rate = rate
        self.started_monotonic = time.monotonic()

    def wait(self, sensor_time_ns: int) -> None:
        target = self.started_monotonic + (
            sensor_time_ns - self.first_sensor_time_ns
        ) / NANOSECONDS_PER_SECOND / self.rate
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(remaining, 0.02))


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_FILENAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"OpenVINS output already exists in {output_dir}; use --overwrite"
        )
    if overwrite:
        for path in existing:
            if path.is_dir():
                raise ValueError(f"refusing to remove unexpected output directory: {path}")
            path.unlink()


def _terminate_process(process: subprocess.Popen[Any], timeout_s: float = 10.0) -> int:
    if process.poll() is not None:
        return int(process.returncode)
    process.send_signal(signal.SIGINT)
    try:
        return int(process.wait(timeout=timeout_s))
    except subprocess.TimeoutExpired:
        process.terminate()
    try:
        return int(process.wait(timeout=5.0))
    except subprocess.TimeoutExpired:
        process.kill()
        return int(process.wait(timeout=5.0))


def run_ros_replay(
    *,
    source_dir: Path,
    output_dir: Path,
    config_dir: Path,
    binary: Path,
    frames: Sequence[CameraFrame],
    imu_samples: Sequence[ImuSample],
    replay_rate: float,
    handshake_timeout_s: float,
    drain_timeout_s: float,
    drain_quiet_s: float,
    imu_preroll_s: float,
    imu_tail_s: float,
    ros_parameter_overrides: Mapping[str, Any] | None = None,
    left_video_path: Path | None = None,
    right_video_path: Path | None = None,
    left_rectification_maps: tuple[np.ndarray, np.ndarray] | None = None,
    right_rectification_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Launch OpenVINS, replay decoded stereo/IMU data, and record odometry."""

    try:
        import cv2
        import rclpy
        from nav_msgs.msg import Odometry
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, Imu
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 replay requires /usr/bin/python3 after sourcing "
            "/opt/ros/jazzy/setup.bash"
        ) from error

    left_path = left_video_path or source_dir / "videos/left.mp4"
    right_path = right_video_path or source_dir / "videos/right.mp4"
    for path in (left_path, right_path, binary):
        if not path.is_file():
            raise FileNotFoundError(path)

    class ReplayNode(Node):
        def __init__(self, writer: AtomicJsonlWriter) -> None:
            super().__init__("daaam_ego_openvins_replay")
            qos = QoSProfile(
                depth=100,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.left_publisher = self.create_publisher(Image, LEFT_TOPIC, qos)
            self.right_publisher = self.create_publisher(Image, RIGHT_TOPIC, qos)
            self.imu_publisher = self.create_publisher(Imu, IMU_TOPIC, qos)
            self.create_subscription(Odometry, ODOMETRY_TOPIC, self.on_odometry, qos)
            self.writer = writer
            self.failure: Exception | None = None
            self.odometry_count = 0
            self.first_odometry_ns: int | None = None
            self.last_odometry_ns: int | None = None
            self.last_odometry_monotonic: float | None = None

        def on_odometry(self, message: Any) -> None:
            try:
                record = odometry_record_from_message(
                    message, traversal_id=source_dir.name
                )
                timestamp_ns = int(record["sensor_time_ns"])
                if self.last_odometry_ns is not None and timestamp_ns <= self.last_odometry_ns:
                    raise ValueError("OpenVINS odometry timestamps are not strictly increasing")
                self.writer.write(record)
                self.odometry_count += 1
                self.first_odometry_ns = self.first_odometry_ns or timestamp_ns
                self.last_odometry_ns = timestamp_ns
                self.last_odometry_monotonic = time.monotonic()
            except Exception as error:  # callback boundary
                self.failure = error

    def set_stamp(message: Any, timestamp_ns: int) -> None:
        message.header.stamp.sec = timestamp_ns // NANOSECONDS_PER_SECOND
        message.header.stamp.nanosec = timestamp_ns % NANOSECONDS_PER_SECOND

    def image_message(frame: np.ndarray, timestamp_ns: int, frame_id: str) -> Any:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError(f"unsupported decoded frame shape: {frame.shape}")
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        message = Image()
        set_stamp(message, timestamp_ns)
        message.header.frame_id = frame_id
        message.height, message.width = gray.shape
        message.encoding = "mono8"
        message.is_bigendian = False
        message.step = int(message.width)
        message.data = gray.tobytes()
        return message

    def imu_message(sample: ImuSample) -> Any:
        message = Imu()
        set_stamp(message, sample.sensor_time_ns)
        message.header.frame_id = "imu"
        message.orientation_covariance[0] = -1.0
        message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z = sample.angular_velocity
        message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z = sample.linear_acceleration
        return message

    writer = AtomicJsonlWriter(output_dir / "openvins_odometry.jsonl")
    stdout_stream = (output_dir / "openvins_stdout.log").open("xb")
    stderr_stream = (output_dir / "openvins_stderr.log").open("xb")
    command = [
        str(binary),
        "--ros-args",
        "-r",
        "__ns:=/ov_msckf",
        "-p",
        f"config_path:={config_dir / 'estimator_config.yaml'}",
    ]
    for name, value in sorted((ros_parameter_overrides or {}).items()):
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float, str)):
            rendered = str(value)
        else:
            raise ValueError(
                f"unsupported ROS parameter override type for {name}: "
                f"{type(value).__name__}"
            )
        command.extend(("-p", f"{name}:={rendered}"))
    process: subprocess.Popen[Any] | None = None
    node: ReplayNode | None = None
    executor: Any = None
    spin_thread: threading.Thread | None = None
    left_capture: Any = None
    right_capture: Any = None
    metrics: dict[str, Any] = {
        "command": command,
        "camera_pair_count": 0,
        "imu_count": 0,
        "odometry_count": 0,
        "handshake_counts": {},
        "drain_completed": False,
    }
    started = time.monotonic()
    try:
        rclpy.init(args=None)
        node = ReplayNode(writer)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        process = subprocess.Popen(
            command,
            stdout=stdout_stream,
            stderr=stderr_stream,
            cwd=str(output_dir),
        )
        publishers = {
            "left": node.left_publisher,
            "right": node.right_publisher,
            "imu": node.imu_publisher,
        }
        metrics["handshake_counts"] = wait_for_input_subscribers(
            publishers, process, handshake_timeout_s
        )

        first_camera_ns = frames[0].sensor_time_ns
        final_camera_ns = frames[-1].sensor_time_ns
        imu_start_ns = first_camera_ns - int(round(imu_preroll_s * NANOSECONDS_PER_SECOND))
        imu_end_ns = final_camera_ns + int(round(imu_tail_s * NANOSECONDS_PER_SECOND))
        selected_imu = [
            sample
            for sample in imu_samples
            if imu_start_ns <= sample.sensor_time_ns <= imu_end_ns
        ]
        if not selected_imu or selected_imu[0].sensor_time_ns > first_camera_ns:
            raise ValueError("IMU replay window does not cover the first camera frame")
        replay_clock = ReplayClock(
            min(selected_imu[0].sensor_time_ns, first_camera_ns), replay_rate
        )

        left_capture = cv2.VideoCapture(str(left_path))
        right_capture = cv2.VideoCapture(str(right_path))
        if not left_capture.isOpened() or not right_capture.isOpened():
            raise RuntimeError("failed to open left/right MP4")
        imu_index = 0
        next_left_frame_idx = 0
        next_right_frame_idx = 0
        for camera in frames:
            if camera.left_frame_idx < next_left_frame_idx:
                raise RuntimeError("left video frame indices are not increasing")
            if camera.right_frame_idx < next_right_frame_idx:
                raise RuntimeError("right video frame indices are not increasing")
            left_image = None
            while next_left_frame_idx <= camera.left_frame_idx:
                left_ok, left_image = left_capture.read()
                if not left_ok:
                    raise RuntimeError(
                        "left decode failed at video frame index "
                        f"{next_left_frame_idx} for source index {camera.source_idx}"
                    )
                next_left_frame_idx += 1
            right_image = None
            while next_right_frame_idx <= camera.right_frame_idx:
                right_ok, right_image = right_capture.read()
                if not right_ok:
                    raise RuntimeError(
                        "right decode failed at video frame index "
                        f"{next_right_frame_idx} for source index {camera.source_idx}"
                    )
                next_right_frame_idx += 1
            assert left_image is not None and right_image is not None
            if left_rectification_maps is not None:
                left_image = cv2.remap(
                    left_image,
                    left_rectification_maps[0],
                    left_rectification_maps[1],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            if right_rectification_maps is not None:
                right_image = cv2.remap(
                    right_image,
                    right_rectification_maps[0],
                    right_rectification_maps[1],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            while (
                imu_index < len(selected_imu)
                and selected_imu[imu_index].sensor_time_ns <= camera.sensor_time_ns
            ):
                sample = selected_imu[imu_index]
                replay_clock.wait(sample.sensor_time_ns)
                node.imu_publisher.publish(imu_message(sample))
                metrics["imu_count"] += 1
                imu_index += 1
                if node.failure is not None:
                    raise node.failure
            replay_clock.wait(camera.sensor_time_ns)
            node.left_publisher.publish(image_message(left_image, camera.sensor_time_ns, "cam0"))
            node.right_publisher.publish(image_message(right_image, camera.sensor_time_ns, "cam1"))
            metrics["camera_pair_count"] += 1
            if process.poll() is not None:
                raise RuntimeError(f"OpenVINS exited early with code {process.returncode}")
            if node.failure is not None:
                raise node.failure

        while imu_index < len(selected_imu):
            sample = selected_imu[imu_index]
            replay_clock.wait(sample.sensor_time_ns)
            node.imu_publisher.publish(imu_message(sample))
            metrics["imu_count"] += 1
            imu_index += 1

        drain_deadline = time.monotonic() + drain_timeout_s
        while time.monotonic() < drain_deadline:
            if node.failure is not None:
                raise node.failure
            if process.poll() is not None:
                raise RuntimeError(f"OpenVINS exited during drain with code {process.returncode}")
            if (
                node.odometry_count > 0
                and node.last_odometry_monotonic is not None
                and time.monotonic() - node.last_odometry_monotonic >= drain_quiet_s
            ):
                metrics["drain_completed"] = True
                break
            time.sleep(0.05)
        if not metrics["drain_completed"]:
            if node.odometry_count == 0:
                raise RuntimeError("OpenVINS produced no odometry (initialization failed)")
            raise TimeoutError("OpenVINS estimator did not become quiet before drain timeout")
        metrics["odometry_count"] = node.odometry_count
        metrics["first_odometry_sensor_time_ns"] = node.first_odometry_ns
        metrics["last_odometry_sensor_time_ns"] = node.last_odometry_ns
        writer.complete()
    except Exception:
        writer.abort()
        raise
    finally:
        if left_capture is not None:
            left_capture.release()
        if right_capture is not None:
            right_capture.release()
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if process is not None:
            metrics["openvins_exit_code"] = _terminate_process(process)
        stdout_stream.close()
        stderr_stream.close()
        metrics["wall_duration_s"] = time.monotonic() - started
    return metrics


def load_pose_samples(path: Path) -> list[PoseSample]:
    class ValidatedPoseSamples(list):
        pass
    samples = ValidatedPoseSamples()
    samples.rejected_time_ns = []
    rejected = []
    previous_input_timestamp = None
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if record.get("frame_id") != "global" or record.get("child_frame_id") != "imu":
                    raise ValueError("expected global -> imu")
                timestamp_ns = int(record["sensor_time_ns"])
                if int(record["pose_time_ns"]) != timestamp_ns:
                    raise ValueError("pose_time_ns must equal sensor_time_ns")
                if record.get("lookup_semantics") != "message_stamp":
                    raise ValueError("lookup_semantics must be message_stamp")
                if previous_input_timestamp is not None and timestamp_ns <= previous_input_timestamp:
                    raise ValueError("all input timestamps must increase strictly")
                previous_input_timestamp = timestamp_ns
                try:
                    sample = PoseSample(
                        timestamp_ns,
                        np.asarray(record["position_m"], dtype=np.float64),
                        np.asarray(record["orientation_xyzw"], dtype=np.float64),
                        np.asarray(record["pose_covariance"], dtype=np.float64),
                    )
                except ValueError as covariance_error:
                    if not str(covariance_error).startswith("pose covariance is materially indefinite"):
                        raise
                    samples.rejected_time_ns.append(timestamp_ns)
                    rejected.append({"line_number": line_number, "sensor_time_ns": timestamp_ns,
                        "reason": str(covariance_error), "original_line_sha256": hashlib.sha256(line.encode()).hexdigest()})
                    continue
                if samples and sample.sensor_time_ns <= samples[-1].sensor_time_ns:
                    raise ValueError("timestamps must increase strictly")
                samples.append(sample)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid odometry JSONL line {line_number}: {error}") from error
    report = {"schema": "ego.rejected_odometry_covariances.v1", "source_path": str(path),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "accepted_count": len(samples),
        "rejected_count": len(rejected), "rejected": rejected,
        "policy": "unchanged_PSD_tolerance; invalid_samples_excluded; no_interpolation_across_rejected_samples"}
    path.with_name("rejected_odometry_covariances.json").write_text(json.dumps(report, indent=2) + "\n")
    if not samples:
        raise ValueError(f"odometry JSONL contains no poses: {path}")
    return samples


def write_pose_products(
    *,
    odometry_path: Path,
    camera_frames: Sequence[CameraFrame],
    T_imu_cam0: Any,
    timeshift_cam_imu_s: float,
    output_dir: Path,
    max_bracket_gap_ns: int = 50_000_000,
) -> dict[str, Any]:
    samples = load_pose_samples(odometry_path)
    T_imu_cam0_array = validate_homogeneous_transform(T_imu_cam0, "T_imu_cam0")
    if not math.isfinite(timeshift_cam_imu_s):
        raise ValueError("timeshift_cam_imu_s must be finite")
    time_shift_ns = int(round(timeshift_cam_imu_s * NANOSECONDS_PER_SECOND))
    camera_path = output_dir / "camera_pose_samples.jsonl"
    temporary_path = camera_path.with_name(camera_path.name + ".tmp")
    accepted = 0
    rejected: list[dict[str, Any]] = []
    positions: list[np.ndarray] = []
    with temporary_path.open("x", encoding="utf-8") as stream:
        for frame in camera_frames:
            # OpenVINS publishes odomimu from IMU callbacks, so its message
            # stamps live on the IMU clock.  Kalibr/OpenVINS define
            # t_imu = t_cam + timeshift_cam_imu.  Querying at raw t_cam would
            # introduce a systematic 47.8 ms error for this capture.
            pose_query_time_ns = frame.sensor_time_ns + time_shift_ns
            try:
                interpolated = interpolate_pose(
                    samples,
                    pose_query_time_ns,
                    max_bracket_gap_ns=max_bracket_gap_ns,
                )
            except ValueError as error:
                rejected.append(
                    {
                        "source_idx": frame.source_idx,
                        "sensor_time_ns": frame.sensor_time_ns,
                        "reason": str(error),
                    }
                )
                continue
            transform = world_T_camera(interpolated.world_T_imu, T_imu_cam0_array)
            orientation = matrix_to_quaternion_xyzw(transform[:3, :3])
            record = {
                "source_idx": frame.source_idx,
                "left_video_frame_idx": frame.left_frame_idx,
                "right_video_frame_idx": frame.right_frame_idx,
                "sensor_time_ns": frame.sensor_time_ns,
                "pose_sensor_time_ns": frame.sensor_time_ns,
                "pose_query_time_ns": pose_query_time_ns,
                "time_shift_ns": time_shift_ns,
                "timeshift_cam_imu_ns": time_shift_ns,
                "frame_id": "global",
                "child_frame_id": "cam0",
                "position_m": transform[:3, 3].tolist(),
                "orientation_xyzw": orientation.tolist(),
                "world_T_cam0": transform.tolist(),
                "pose_covariance": interpolated.pose_covariance.tolist(),
                "openvins_left_bracket_ns": interpolated.left_bracket_ns,
                "openvins_right_bracket_ns": interpolated.right_bracket_ns,
                "interpolation_alpha": interpolated.alpha,
                "lookup_semantics": "bracketed_linear_slerp",
            }
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            accepted += 1
            positions.append(transform[:3, 3].copy())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, camera_path)

    timestamps = np.asarray([sample.sensor_time_ns for sample in samples], dtype=np.int64)
    gaps_s = np.diff(timestamps).astype(np.float64) / NANOSECONDS_PER_SECOND
    duration_s = (timestamps[-1] - timestamps[0]) / NANOSECONDS_PER_SECOND
    rate_hz = (len(samples) - 1) / duration_s if duration_s > 0.0 else 0.0
    covariance_diagonals = np.asarray(
        [np.diag(sample.pose_covariance) for sample in samples]
    )
    covariance_repaired = [sample for sample in samples if sample.covariance_psd_repaired]
    coverage = accepted / len(camera_frames) if camera_frames else 0.0
    metrics = {
        "odometry_sample_count": len(samples),
        "odometry_duration_s": duration_s,
        "odometry_rate_hz": rate_hz,
        "odometry_max_gap_s": float(np.max(gaps_s)) if len(gaps_s) else None,
        "camera_frame_count": len(camera_frames),
        "camera_pose_count": accepted,
        "camera_pose_coverage": coverage,
        "timeshift_cam_imu_s": timeshift_cam_imu_s,
        "time_shift_ns": time_shift_ns,
        "timeshift_cam_imu_ns": time_shift_ns,
        "maximum_position_std_m": float(
            np.sqrt(np.maximum(covariance_diagonals[:, :3], 0.0)).max()
        ),
        "maximum_rotation_std_deg": float(
            np.rad2deg(np.sqrt(np.maximum(covariance_diagonals[:, 3:], 0.0))).max()
        ),
        "covariance_psd_repair_count": len(covariance_repaired),
        "covariance_psd_repair_ratio": len(covariance_repaired) / len(samples),
        "minimum_covariance_eigenvalue_before_repair": min(
            sample.covariance_min_eigenvalue_before for sample in samples
        ),
        "maximum_covariance_asymmetry": max(
            sample.covariance_maximum_asymmetry for sample in samples
        ),
        "covariance_psd_repair_policy": {
            "method": "symmetric_eigendecomposition_clip_negative_to_zero",
            "maximum_negative_eigenvalue_ratio": 1.0e-3,
            "absolute_floor": 1.0e-9,
            "materially_indefinite_fails": True,
        },
    }
    gates = {
        "camera_pose_coverage": coverage >= 0.90,
        "odometry_rate": rate_hz >= 20.0,
        "odometry_max_gap": bool(len(gaps_s)) and float(np.max(gaps_s)) <= 0.10,
        "position_std": metrics["maximum_position_std_m"] <= 1.0,
        "rotation_std": metrics["maximum_rotation_std_deg"] <= 30.0,
    }
    report = {
        "schema_version": 1,
        "passed": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "rejected_camera_frames": rejected,
        "no_ground_truth": True,
    }
    _atomic_write_json(output_dir / "vio_acceptance.json", report)
    write_trajectory_previews(positions, output_dir)
    return report


def write_trajectory_previews(positions: Sequence[np.ndarray], output_dir: Path) -> None:
    ply_path = output_dir / "trajectory_preview.ply"
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(positions)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend("{:.9g} {:.9g} {:.9g}".format(*position) for position in positions)
    _atomic_write_text(ply_path, "\n".join(lines) + "\n")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        array = np.asarray(positions, dtype=np.float64)
        figure, axis = plt.subplots(figsize=(7, 7), dpi=120)
        if len(array):
            axis.plot(array[:, 0], array[:, 1], linewidth=1.0)
            axis.scatter(array[0, 0], array[0, 1], c="green", s=20, label="start")
            axis.scatter(array[-1, 0], array[-1, 1], c="red", s=20, label="end")
            axis.legend()
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        temporary = output_dir / "trajectory_preview.tmp.png"
        figure.savefig(temporary)
        plt.close(figure)
        os.replace(temporary, output_dir / "trajectory_preview.png")
    except ImportError:
        # PLY is the dependency-free canonical preview; PNG is best effort on
        # minimal ROS installations.
        pass


def _source_inventory(source_dir: Path) -> dict[str, Any]:
    relative_paths = (
        "videos/left.mp4",
        "videos/right.mp4",
        "timestamps/left_timestamps.txt",
        "timestamps/right_timestamps.txt",
        "imu/imu0.csv",
        "calibrations/camera_imu-camchain-imucam.yaml",
        "calibrations/camera_imu-imu.yaml",
    )
    result: dict[str, Any] = {}
    for relative in relative_paths:
        path = source_dir / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Ego OpenVINS configs and replay stereo MP4 + IMU over ROS 2"
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--frame-inventory",
        type=Path,
        help="Audit-authoritative retained stereo pairs and per-side video indices.",
    )
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--openvins-root", type=Path, default=DEFAULT_OPENVINS_ROOT)
    parser.add_argument("--openvins-binary", type=Path, default=DEFAULT_OPENVINS_BINARY)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--regenerate-config", action="store_true")
    parser.add_argument("--config-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--start-time-s", type=float, default=0.0)
    parser.add_argument(
        "--max-duration-s", "--duration-s", dest="max_duration_s", type=float
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--replay-rate", type=float, default=1.0)
    parser.add_argument("--handshake-timeout-s", type=float, default=30.0)
    parser.add_argument("--drain-timeout-s", type=float, default=15.0)
    parser.add_argument("--drain-quiet-s", type=float, default=1.0)
    parser.add_argument("--imu-preroll-s", type=float, default=0.25)
    parser.add_argument("--imu-tail-s", type=float, default=0.25)
    parser.add_argument(
        "--initialization-profile",
        choices=("static", "dynamic"),
        default="static",
        help="Use the frozen primary initializer or the explicit dynamic A/B fallback.",
    )
    parser.add_argument(
        "--dynamic-override",
        type=Path,
        default=DEFAULT_DYNAMIC_OVERRIDE,
        help="Audited ROS-parameter override YAML used only by the dynamic profile.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir.resolve()
    config_dir = args.config_dir.resolve()
    openvins_root = args.openvins_root.resolve()
    binary = args.openvins_binary.resolve()
    if args.replay_rate <= 0.0:
        raise ValueError("--replay-rate must be positive")
    if min(
        args.handshake_timeout_s,
        args.drain_timeout_s,
        args.drain_quiet_s,
        args.imu_preroll_s,
        args.imu_tail_s,
    ) < 0.0:
        raise ValueError("timeouts and IMU margins must be non-negative")

    if args.regenerate_config or not (
        config_dir / "estimator_config.yaml"
    ).is_file():
        config_validation = generate_capture_configs(source_dir, config_dir)
    else:
        config_validation = validate_capture_configs(source_dir, config_dir)
    checkout = inspect_openvins_checkout(openvins_root)
    imucam = _read_yaml(config_dir / "kalibr_imucam_chain.yaml")
    time_shift_s = float(imucam["cam0"]["timeshift_cam_imu"])
    if args.frame_inventory is not None:
        frames, frame_report = read_camera_frames_from_inventory(
            args.frame_inventory.resolve()
        )
    else:
        frames, frame_report = read_camera_frames(
            source_dir,
            time_shift_s,
            maximum_source_frames=probe_stereo_video_frame_count(source_dir),
        )
    selected_frames = select_replay_window(
        frames,
        start_time_s=args.start_time_s,
        max_duration_s=args.max_duration_s,
        max_frames=args.max_frames,
    )
    imu_samples = read_imu_samples(source_dir / "imu/imu0.csv")
    initialization_overrides, initialization_override_provenance = (
        load_initialization_overrides(
            args.initialization_profile, args.dynamic_override.resolve()
        )
    )
    plan = {
        "schema_version": 1,
        "dry_run": bool(args.dry_run),
        "source_dir": str(source_dir),
        "config_dir": str(config_dir),
        "output_dir": str(args.output_dir.resolve()) if args.output_dir else None,
        "openvins": checkout,
        "openvins_binary": {"path": str(binary), "exists": binary.is_file()},
        "config_validation": config_validation,
        "frame_report": frame_report,
        "selected_camera_pair_count": len(selected_frames),
        "selected_first_source_idx": selected_frames[0].source_idx,
        "selected_last_source_idx": selected_frames[-1].source_idx,
        "selected_duration_s": (
            selected_frames[-1].sensor_time_ns - selected_frames[0].sensor_time_ns
        )
        / NANOSECONDS_PER_SECOND,
        "replay_rate": args.replay_rate,
        "initialization_profile": args.initialization_profile,
        "initialization_override": initialization_override_provenance,
        "would_execute_ros": not args.dry_run and not args.config_only,
    }
    if args.dry_run or args.config_only:
        print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run or --config-only")
    if not binary.is_file():
        raise FileNotFoundError(
            f"isolated OpenVINS binary is missing: {binary}; build it per Ego_Plan.md"
        )

    output_dir = args.output_dir.resolve()
    _prepare_output_dir(output_dir, args.overwrite)
    manifest = {
        **plan,
        "dry_run": False,
        "source_inventory": _source_inventory(source_dir),
        "status": "running",
        "started_unix_ns": time.time_ns(),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "python_executable": sys.executable,
    }
    _atomic_write_json(output_dir / "openvins_manifest.json", manifest)
    try:
        replay_metrics = run_ros_replay(
            source_dir=source_dir,
            output_dir=output_dir,
            config_dir=config_dir,
            binary=binary,
            frames=selected_frames,
            imu_samples=imu_samples,
            replay_rate=args.replay_rate,
            handshake_timeout_s=args.handshake_timeout_s,
            drain_timeout_s=args.drain_timeout_s,
            drain_quiet_s=args.drain_quiet_s,
            imu_preroll_s=args.imu_preroll_s,
            imu_tail_s=args.imu_tail_s,
            ros_parameter_overrides=initialization_overrides,
        )
        _atomic_write_json(output_dir / "replay_metrics.json", replay_metrics)
        acceptance = write_pose_products(
            odometry_path=output_dir / "openvins_odometry.jsonl",
            camera_frames=selected_frames,
            T_imu_cam0=imucam["cam0"]["T_imu_cam"],
            timeshift_cam_imu_s=time_shift_s,
            output_dir=output_dir,
        )
        if not acceptance["passed"]:
            raise RuntimeError("OpenVINS VIO acceptance gates failed")
        manifest.update(
            {
                "status": "complete",
                "finished_unix_ns": time.time_ns(),
                "outputs": {
                    name: {
                        "sha256": _sha256(output_dir / name),
                        "size_bytes": (output_dir / name).stat().st_size,
                    }
                    for name in (
                        "openvins_odometry.jsonl",
                        "replay_metrics.json",
                        "vio_acceptance.json",
                        "camera_pose_samples.jsonl",
                        "trajectory_preview.ply",
                    )
                },
            }
        )
        _atomic_write_json(output_dir / "openvins_manifest.json", manifest)
        _atomic_write_json(output_dir / "stage_manifest.json", manifest)
        _atomic_write_text(output_dir / ".complete", "complete\n")
        return 0
    except Exception as error:
        failure = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "failed_unix_ns": time.time_ns(),
        }
        _atomic_write_json(output_dir / ".failed.json", failure)
        manifest.update(failure)
        _atomic_write_json(output_dir / "openvins_manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
