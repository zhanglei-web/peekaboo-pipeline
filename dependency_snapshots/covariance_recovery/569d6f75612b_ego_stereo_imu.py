"""Strict source auditing for Ego MP4 stereo/IMU recordings.

The raw timestamp sidecars are the sensor clock authority.  MP4 timestamps are
retained only as decode-order evidence; they are never promoted to absolute
sensor time.  This module deliberately keeps the S00 audit independent from
ROS so that malformed source data fails before an OpenVINS replay is started.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


NANOSECONDS_PER_SECOND = Decimal(1_000_000_000)
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
SCHEMA_VERSION = 1

LEFT_VIDEO = Path("videos/left.mp4")
RIGHT_VIDEO = Path("videos/right.mp4")
LEFT_TIMESTAMPS = Path("timestamps/left_timestamps.txt")
RIGHT_TIMESTAMPS = Path("timestamps/right_timestamps.txt")
IMU_CSV = Path("imu/imu0.csv")
CAMCHAIN_YAML = Path("calibrations/camera-camchain.yaml")
IMUCAM_YAML = Path("calibrations/camera_imu-camchain-imucam.yaml")
IMU_YAML = Path("calibrations/camera_imu-imu.yaml")
META_JSON = Path("meta.json")

REQUIRED_SOURCE_FILES = (
    LEFT_VIDEO,
    RIGHT_VIDEO,
    LEFT_TIMESTAMPS,
    RIGHT_TIMESTAMPS,
    IMU_CSV,
    CAMCHAIN_YAML,
    IMUCAM_YAML,
    IMU_YAML,
    META_JSON,
)

OUTPUT_FILENAMES = (
    "source_audit.json",
    "frame_inventory.jsonl",
    "sensor_events.jsonl",
    "calibration_normalized.json",
    "source_manifest.json",
)

DROP_STEREO_TIMESTAMP_MISMATCH = "stereo_timestamp_mismatch"
DROP_NON_MONOTONIC_LEFT = "non_monotonic_left_timestamp"
DROP_NON_MONOTONIC_RIGHT = "non_monotonic_right_timestamp"
DROP_IMU_COVERAGE_START = "imu_coverage_start"
DROP_IMU_COVERAGE_END = "imu_coverage_end"
DROP_MISSING_VIDEO_FRAME = "missing_video_frame"
DROP_MISSING_TIMESTAMP = "missing_timestamp_sidecar"
DROP_STEREO_OFFSET_BOUNDARY = "stereo_global_offset_boundary"

# The capture process writes a timestamp row before the corresponding encoded
# frame is durably committed.  A clean shutdown should keep both counts equal,
# but an interrupted/final encoder flush can therefore leave exactly one
# trailing, bilateral timestamp pair without an MP4 frame.  Only that narrow
# case is recoverable: interior gaps, unilateral extras, or larger truncations
# remain hard failures.
MAX_TRAILING_TIMESTAMP_PAIRS_WITHOUT_VIDEO = 1

# Independent MP4 encoders can flush one camera a frame later than the other at
# shutdown.  The only safe recovery is to keep the common decoded prefix and
# mark the final, incomplete stereo source index as missing video.  Larger
# disagreements remain a hard failure because they may indicate an interior
# loss that decode-order evidence cannot localize.
MAX_TRAILING_STEREO_VIDEO_COUNT_DELTA = 1

# Some capture sessions contain a sidecar indexing defect where one camera's
# complete timestamp stream is displaced by one encoded frame.  Recover only a
# single, global integer offset with overwhelming exact-timestamp evidence;
# this is deliberately not per-frame nearest-neighbour timestamp matching.
MAX_GLOBAL_STEREO_FRAME_OFFSET = 2
MIN_GLOBAL_STEREO_OFFSET_OVERLAP = 30
MIN_GLOBAL_STEREO_OFFSET_EXACT_RATIO = 0.95
MAX_GLOBAL_STEREO_OFFSET_RUNNER_UP_RATIO = 0.20

# ``frames/`` is an optional decode-order materialization of the MP4 inputs.
# It is deliberately excluded from the frozen source manifest: including it
# would make the source identity depend on whether the optional extraction
# stage happened to run before S00, and would hash the same pixels twice.
DERIVED_MATERIALIZATION_ROOTS = frozenset({"frames"})


class EgoAuditError(ValueError):
    """Raised when a source-audit hard gate fails."""


@dataclass(frozen=True)
class TimestampRecord:
    """One exact timestamp-sidecar row."""

    source_idx: int
    raw_seconds: str
    sensor_time_ns: int


@dataclass(frozen=True)
class ImuSample:
    """One IMU CSV row; orientation is retained only as ignored evidence."""

    source_row: int
    raw_seconds: str
    sensor_time_ns: int
    orientation_xyzw: tuple[float, float, float, float]
    angular_velocity_rad_s: tuple[float, float, float]
    linear_acceleration_m_s2: tuple[float, float, float]


@dataclass(frozen=True)
class VideoFrameEvidence:
    """MP4 frame-order evidence reported by ffprobe."""

    frame_idx: int
    pts: int
    pts_time_s: str | None
    duration_time_s: str | None


@dataclass(frozen=True)
class VideoProbe:
    """A normalized video summary and its per-frame MP4 evidence."""

    summary: dict[str, Any]
    frames: tuple[VideoFrameEvidence, ...]


@dataclass(frozen=True)
class AuditResult:
    """All S00 records, ready for atomic materialization."""

    source_audit: dict[str, Any]
    frame_inventory: tuple[dict[str, Any], ...]
    sensor_events: tuple[dict[str, Any], ...]
    calibration_normalized: dict[str, Any]
    source_manifest: dict[str, Any]


def decimal_seconds_to_ns(value: str | Decimal | int, *, field: str = "timestamp") -> int:
    """Convert decimal seconds to integer nanoseconds without a float round-trip.

    Sub-nanosecond values are rejected instead of rounded.  Rejecting ``float``
    inputs is intentional: by the time a value is a binary float, exact source
    decimal provenance has already been lost.
    """

    if isinstance(value, bool) or isinstance(value, float):
        raise EgoAuditError(f"{field} must be parsed from decimal text, not float")
    try:
        decimal_value = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise EgoAuditError(f"Invalid decimal {field}: {value!r}") from error
    if not decimal_value.is_finite():
        raise EgoAuditError(f"{field} must be finite: {value!r}")
    scaled = decimal_value * NANOSECONDS_PER_SECOND
    integral = scaled.to_integral_value()
    if scaled != integral:
        raise EgoAuditError(
            f"{field} has sub-nanosecond precision and cannot be represented exactly: "
            f"{value!r}"
        )
    result = int(integral)
    if result < INT64_MIN or result > INT64_MAX:
        raise EgoAuditError(f"{field} is outside signed int64 nanoseconds: {value!r}")
    return result


def _decimal_text(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise EgoAuditError(f"{field} must be a decimal number")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise EgoAuditError(f"Invalid decimal {field}: {value!r}") from error
    if not result.is_finite():
        raise EgoAuditError(f"{field} must be finite")
    return result


def parse_timestamp_file(path: Path | str) -> list[TimestampRecord]:
    """Parse an index/timestamp sidecar and require contiguous source indices."""

    source = Path(path)
    if not source.is_file():
        raise EgoAuditError(f"Missing timestamp file: {source}")
    records: list[TimestampRecord] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                raise EgoAuditError(f"Blank timestamp row in {source} line {line_number}")
            fields = line.split()
            if len(fields) != 2:
                raise EgoAuditError(
                    f"Expected '<frame_index> <seconds>' in {source} line {line_number}"
                )
            try:
                source_idx = int(fields[0], 10)
            except ValueError as error:
                raise EgoAuditError(
                    f"Invalid frame index in {source} line {line_number}: {fields[0]!r}"
                ) from error
            expected_idx = len(records)
            if source_idx != expected_idx:
                raise EgoAuditError(
                    f"Non-contiguous frame index in {source} line {line_number}: "
                    f"expected {expected_idx}, got {source_idx}"
                )
            timestamp_ns = decimal_seconds_to_ns(
                fields[1], field=f"{source.name} line {line_number} timestamp"
            )
            if timestamp_ns <= 0:
                raise EgoAuditError(
                    f"Timestamp must be positive in {source} line {line_number}"
                )
            records.append(TimestampRecord(source_idx, fields[1], timestamp_ns))
    if not records:
        raise EgoAuditError(f"Timestamp file is empty: {source}")
    return records


def timestamp_stream_metrics(records: Sequence[TimestampRecord]) -> dict[str, Any]:
    """Describe raw timestamp anomalies without deciding which stereo frame to drop."""

    if not records:
        raise EgoAuditError("Timestamp metrics require at least one record")
    deltas = [
        records[index].sensor_time_ns - records[index - 1].sensor_time_ns
        for index in range(1, len(records))
    ]
    transitions = [
        {
            "previous_source_idx": records[index - 1].source_idx,
            "current_source_idx": records[index].source_idx,
            "delta_ns": deltas[index - 1],
            "kind": "duplicate" if deltas[index - 1] == 0 else "decreasing",
        }
        for index in range(1, len(records))
        if deltas[index - 1] <= 0
    ]
    positive = [value for value in deltas if value > 0]
    return {
        "count": len(records),
        "first_sensor_time_ns": records[0].sensor_time_ns,
        "last_sensor_time_ns": records[-1].sensor_time_ns,
        "strictly_increasing": not transitions,
        "non_increasing_transitions": transitions,
        "positive_delta_ns": _numeric_summary(positive),
    }


def _parse_finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EgoAuditError(f"Invalid numeric {field}: {value!r}") from error
    if not math.isfinite(result):
        raise EgoAuditError(f"{field} must be finite: {value!r}")
    return result


def parse_imu_csv(path: Path | str) -> list[ImuSample]:
    """Parse the IMU stream, retaining exact timestamps and finite vectors."""

    source = Path(path)
    if not source.is_file():
        raise EgoAuditError(f"Missing IMU CSV: {source}")
    orientation_fields = tuple(f"orientation_{axis}" for axis in "xyzw")
    gyro_fields = tuple(f"angular_velocity_{axis}" for axis in "xyz")
    acceleration_fields = tuple(f"linear_acceleration_{axis}" for axis in "xyz")
    required = ("timestamp",) + orientation_fields + gyro_fields + acceleration_fields
    samples: list[ImuSample] = []
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise EgoAuditError(f"IMU CSV has no header: {source}")
        missing = [name for name in required if name not in reader.fieldnames]
        if missing:
            raise EgoAuditError(f"IMU CSV is missing columns {missing}: {source}")
        for source_row, row in enumerate(reader):
            line_number = source_row + 2
            raw_time = row.get("timestamp")
            if raw_time is None or not raw_time.strip():
                raise EgoAuditError(
                    f"Missing IMU timestamp in {source} line {line_number}"
                )
            timestamp_ns = decimal_seconds_to_ns(
                raw_time.strip(), field=f"IMU line {line_number} timestamp"
            )
            if timestamp_ns <= 0:
                raise EgoAuditError(
                    f"IMU timestamp must be positive in {source} line {line_number}"
                )
            if samples and timestamp_ns <= samples[-1].sensor_time_ns:
                raise EgoAuditError(
                    f"IMU timestamps must increase strictly in {source}; "
                    f"line {line_number} is not after the previous sample"
                )
            orientation = tuple(
                _parse_finite_float(row[name], field=f"IMU line {line_number} {name}")
                for name in orientation_fields
            )
            gyro = tuple(
                _parse_finite_float(row[name], field=f"IMU line {line_number} {name}")
                for name in gyro_fields
            )
            acceleration = tuple(
                _parse_finite_float(row[name], field=f"IMU line {line_number} {name}")
                for name in acceleration_fields
            )
            samples.append(
                ImuSample(
                    source_row=source_row,
                    raw_seconds=raw_time.strip(),
                    sensor_time_ns=timestamp_ns,
                    orientation_xyzw=orientation,  # type: ignore[arg-type]
                    angular_velocity_rad_s=gyro,  # type: ignore[arg-type]
                    linear_acceleration_m_s2=acceleration,  # type: ignore[arg-type]
                )
            )
    if not samples:
        raise EgoAuditError(f"IMU CSV contains no samples: {source}")
    return samples


def _quantile(sorted_values: Sequence[float | int], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def _numeric_summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": float(ordered[0]),
        "median": _quantile(ordered, 0.5),
        "p99": _quantile(ordered, 0.99),
        "max": float(ordered[-1]),
    }


def imu_metrics(samples: Sequence[ImuSample]) -> dict[str, Any]:
    """Compute sampling and broad physical-unit sanity checks."""

    if not samples:
        raise EgoAuditError("IMU metrics require at least one sample")
    deltas_ns = [
        samples[index].sensor_time_ns - samples[index - 1].sensor_time_ns
        for index in range(1, len(samples))
    ]
    gyro_norms = [math.sqrt(sum(value * value for value in sample.angular_velocity_rad_s)) for sample in samples]
    acceleration_norms = [
        math.sqrt(sum(value * value for value in sample.linear_acceleration_m_s2))
        for sample in samples
    ]
    orientation_all_zero = all(
        all(value == 0.0 for value in sample.orientation_xyzw) for sample in samples
    )
    mean_delta_ns = sum(deltas_ns) / len(deltas_ns) if deltas_ns else None
    mean_rate_hz = (1_000_000_000.0 / mean_delta_ns) if mean_delta_ns else None
    gyro_summary = _numeric_summary(gyro_norms)
    acceleration_summary = _numeric_summary(acceleration_norms)
    # These deliberately generous hard gates catch unit swaps and corrupted rows,
    # while allowing vigorous handheld/head-mounted motion.
    if gyro_summary["max"] is not None and gyro_summary["max"] > 100.0:
        raise EgoAuditError(
            "IMU angular velocity exceeds 100 rad/s; verify units or source integrity"
        )
    if acceleration_summary["max"] is not None and acceleration_summary["max"] > 200.0:
        raise EgoAuditError(
            "IMU acceleration exceeds 200 m/s^2; verify units or source integrity"
        )
    median_acceleration = acceleration_summary["median"]
    if median_acceleration is not None and not 1.0 <= median_acceleration <= 30.0:
        raise EgoAuditError(
            "Median acceleration magnitude is implausible for SI m/s^2 units: "
            f"{median_acceleration:.6g}"
        )
    return {
        "count": len(samples),
        "first_sensor_time_ns": samples[0].sensor_time_ns,
        "last_sensor_time_ns": samples[-1].sensor_time_ns,
        "strictly_increasing": True,
        "delta_ns": _numeric_summary(deltas_ns),
        "mean_rate_hz": mean_rate_hz,
        "orientation_all_zero": orientation_all_zero,
        "orientation_policy": "ignored; ROS orientation_covariance[0] must be -1",
        "angular_velocity_norm_rad_s": gyro_summary,
        "linear_acceleration_norm_m_s2": acceleration_summary,
    }


def _load_yaml(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
        raw_scalars = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    except (OSError, yaml.YAMLError) as error:
        raise EgoAuditError(f"Cannot parse YAML calibration {path}: {error}") from error
    if not isinstance(parsed, dict) or not isinstance(raw_scalars, dict):
        raise EgoAuditError(f"Calibration YAML must contain a mapping: {path}")
    return parsed, raw_scalars


def _matrix4(value: Any, *, field: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise EgoAuditError(f"{field} must be a numeric 4x4 matrix") from error
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise EgoAuditError(f"{field} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-12, rtol=0.0):
        raise EgoAuditError(f"{field} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1.0e-9 or abs(determinant - 1.0) > 1.0e-9:
        raise EgoAuditError(
            f"{field} rotation is invalid (orthogonality error "
            f"{orthogonality_error:.3g}, determinant {determinant:.12g})"
        )
    return matrix


def _camera_parameters(camera: Mapping[str, Any], *, field: str) -> dict[str, Any]:
    if camera.get("camera_model") != "pinhole":
        raise EgoAuditError(f"{field}.camera_model must be 'pinhole'")
    if camera.get("distortion_model") != "equidistant":
        raise EgoAuditError(f"{field}.distortion_model must be 'equidistant'")
    intrinsics = [_parse_finite_float(value, field=f"{field}.intrinsics") for value in camera.get("intrinsics", [])]
    distortion = [
        _parse_finite_float(value, field=f"{field}.distortion_coeffs")
        for value in camera.get("distortion_coeffs", [])
    ]
    resolution = camera.get("resolution")
    if len(intrinsics) != 4 or intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
        raise EgoAuditError(f"{field}.intrinsics must be [fx, fy, cx, cy] with positive focal lengths")
    if len(distortion) != 4:
        raise EgoAuditError(f"{field}.distortion_coeffs must contain four equidistant coefficients")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in resolution
        )
    ):
        raise EgoAuditError(f"{field}.resolution must contain two positive integers")
    return {
        "camera_model": "pinhole",
        "distortion_model": "equidistant",
        "intrinsics": intrinsics,
        "distortion_coeffs": distortion,
        "resolution": [int(resolution[0]), int(resolution[1])],
        "rostopic": str(camera.get("rostopic", "")),
    }


def parse_calibration(dataset_root: Path | str) -> dict[str, Any]:
    """Load and cross-check Kalibr camera/IMU calibrations."""

    root = Path(dataset_root)
    camchain, _ = _load_yaml(root / CAMCHAIN_YAML)
    imucam, imucam_raw = _load_yaml(root / IMUCAM_YAML)
    imu_yaml, _ = _load_yaml(root / IMU_YAML)
    for camera_id in ("cam0", "cam1"):
        if not isinstance(camchain.get(camera_id), dict) or not isinstance(imucam.get(camera_id), dict):
            raise EgoAuditError(f"Both camera calibrations must define {camera_id}")

    camchain_left = _camera_parameters(camchain["cam0"], field="camera-camchain.cam0")
    camchain_right = _camera_parameters(camchain["cam1"], field="camera-camchain.cam1")
    imucam_left = _camera_parameters(imucam["cam0"], field="imucam.cam0")
    imucam_right = _camera_parameters(imucam["cam1"], field="imucam.cam1")
    if camchain_left != imucam_left or camchain_right != imucam_right:
        raise EgoAuditError("Camera intrinsics/models differ between camchain and imucam calibration")

    stereo_camchain = _matrix4(
        camchain["cam1"].get("T_cn_cnm1"), field="camera-camchain.cam1.T_cn_cnm1"
    )
    stereo_imucam = _matrix4(
        imucam["cam1"].get("T_cn_cnm1"), field="imucam.cam1.T_cn_cnm1"
    )
    cam0_T_imu = _matrix4(imucam["cam0"].get("T_cam_imu"), field="imucam.cam0.T_cam_imu")
    cam1_T_imu = _matrix4(imucam["cam1"].get("T_cam_imu"), field="imucam.cam1.T_cam_imu")
    chain_difference = float(np.max(np.abs(stereo_camchain - stereo_imucam)))
    derived_stereo = cam1_T_imu @ np.linalg.inv(cam0_T_imu)
    imu_chain_difference = float(np.max(np.abs(derived_stereo - stereo_imucam)))
    identity_residual = np.linalg.inv(cam1_T_imu) @ stereo_imucam @ cam0_T_imu
    identity_error = float(np.max(np.abs(identity_residual - np.eye(4))))
    if chain_difference > 1.0e-9:
        raise EgoAuditError(
            "Stereo extrinsic differs between camchain and imucam calibration: "
            f"max error {chain_difference:.3g}"
        )
    if imu_chain_difference > 1.0e-9 or identity_error > 1.0e-9:
        raise EgoAuditError(
            "T_cam_imu and T_cam1_cam0 are inconsistent: "
            f"derived max error {imu_chain_difference:.3g}, identity error {identity_error:.3g}"
        )

    try:
        left_shift_raw = imucam_raw["cam0"]["timeshift_cam_imu"]
        right_shift_raw = imucam_raw["cam1"]["timeshift_cam_imu"]
    except (KeyError, TypeError) as error:
        raise EgoAuditError("imucam calibration is missing camera/IMU time shifts") from error
    left_shift = _decimal_text(left_shift_raw, field="cam0.timeshift_cam_imu")
    right_shift = _decimal_text(right_shift_raw, field="cam1.timeshift_cam_imu")

    imu0 = imu_yaml.get("imu0")
    if not isinstance(imu0, dict):
        raise EgoAuditError("IMU calibration must define imu0")
    imu_T_body = _matrix4(imu0.get("T_i_b"), field="imu0.T_i_b")
    imu_fields = (
        "accelerometer_noise_density",
        "accelerometer_random_walk",
        "gyroscope_noise_density",
        "gyroscope_random_walk",
        "update_rate",
    )
    imu_parameters = {
        key: _parse_finite_float(imu0.get(key), field=f"imu0.{key}") for key in imu_fields
    }
    if any(value <= 0.0 for value in imu_parameters.values()):
        raise EgoAuditError("IMU noise, random-walk, and update-rate parameters must be positive")

    translation = stereo_imucam[:3, 3]
    baseline = float(np.linalg.norm(translation))
    if not 0.01 <= baseline <= 1.0:
        raise EgoAuditError(f"Stereo baseline is implausible: {baseline:.12g} m")
    trace_value = float(np.trace(stereo_imucam[:3, :3]))
    rotation_angle_deg = math.degrees(math.acos(float(np.clip((trace_value - 1.0) * 0.5, -1.0, 1.0))))

    cameras: dict[str, Any] = {}
    for camera_id, name, parameters, camera_T_imu, shift_raw, shift in (
        ("cam0", "left", imucam_left, cam0_T_imu, str(left_shift_raw), left_shift),
        ("cam1", "right", imucam_right, cam1_T_imu, str(right_shift_raw), right_shift),
    ):
        cameras[name] = {
            "kalibr_id": camera_id,
            **parameters,
            "T_cam_imu": camera_T_imu.tolist(),
            "T_imu_cam": np.linalg.inv(camera_T_imu).tolist(),
            "timeshift_cam_imu_s": shift_raw,
            "timeshift_cam_imu_ns_decimal": str(shift * NANOSECONDS_PER_SECOND),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "conventions": {
            "T_a_b": "maps coordinates from frame b into frame a",
            "camera_time_offset": "t_imu = t_cam + timeshift_cam_imu",
            "openvins_extrinsic": "T_imu_cam = inverse(T_cam_imu)",
            "camera_order": {"cam0": "left", "cam1": "right"},
        },
        "cameras": cameras,
        "stereo": {
            "T_cam1_cam0": stereo_imucam.tolist(),
            "baseline_m": baseline,
            "relative_rotation_deg": rotation_angle_deg,
            "camchain_vs_imucam_max_abs_error": chain_difference,
            "camera_imu_derived_max_abs_error": imu_chain_difference,
            "identity_gate_max_abs_error": identity_error,
            "right_minus_left_timeshift_s": str(right_shift - left_shift),
            "right_minus_left_timeshift_ms": float((right_shift - left_shift) * Decimal(1000)),
            "common_openvins_timeshift_source": "left/cam0",
        },
        "imu": {
            "T_imu_body": imu_T_body.tolist(),
            **imu_parameters,
            "model": str(imu0.get("model", "")),
            "rostopic": str(imu0.get("rostopic", "")),
            "time_offset_s": _parse_finite_float(imu0.get("time_offset", 0.0), field="imu0.time_offset"),
        },
        "source_files": {
            "camera_chain": CAMCHAIN_YAML.as_posix(),
            "camera_imu_chain": IMUCAM_YAML.as_posix(),
            "imu": IMU_YAML.as_posix(),
        },
    }


def _run_json_command(command: Sequence[str], *, purpose: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as error:
        raise EgoAuditError(f"Unable to run {purpose}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic"
        raise EgoAuditError(f"{purpose} failed ({result.returncode}): {detail}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise EgoAuditError(f"{purpose} returned invalid JSON") from error
    if not isinstance(value, dict):
        raise EgoAuditError(f"{purpose} did not return a JSON object")
    return value


def tool_version(executable: str) -> str:
    resolved = shutil.which(executable) if os.sep not in executable else executable
    if not resolved:
        raise EgoAuditError(f"Required executable not found: {executable}")
    try:
        result = subprocess.run(
            [str(resolved), "-version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as error:
        raise EgoAuditError(f"Unable to run {executable} -version: {error}") from error
    if result.returncode != 0:
        raise EgoAuditError(f"Unable to query {executable} version")
    return result.stdout.splitlines()[0].strip()


def _fraction(value: Any, *, field: str) -> Fraction:
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise EgoAuditError(f"Invalid rational {field}: {value!r}") from error
    if result <= 0:
        raise EgoAuditError(f"{field} must be positive: {value!r}")
    return result


def probe_video(path: Path | str, *, ffprobe_executable: str = "ffprobe") -> VideoProbe:
    """Run stream metadata and full sequential frame-decode audits with ffprobe."""

    source = Path(path)
    if not source.is_file():
        raise EgoAuditError(f"Missing video: {source}")
    resolved = shutil.which(ffprobe_executable) if os.sep not in ffprobe_executable else ffprobe_executable
    if not resolved:
        raise EgoAuditError(f"Required executable not found: {ffprobe_executable}")
    metadata = _run_json_command(
        [
            str(resolved),
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(source),
        ],
        purpose=f"ffprobe metadata audit for {source}",
    )
    streams = metadata.get("streams")
    if not isinstance(streams, list):
        raise EgoAuditError(f"ffprobe returned no stream list for {source}")
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise EgoAuditError(
            f"Expected exactly one video stream in {source}, found {len(video_streams)}"
        )
    stream = video_streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as error:
        raise EgoAuditError(f"ffprobe returned invalid video dimensions for {source}") from error
    if width <= 0 or height <= 0:
        raise EgoAuditError(f"Video dimensions must be positive for {source}")
    average_rate = _fraction(stream.get("avg_frame_rate"), field=f"{source} avg_frame_rate")
    nominal_rate = _fraction(stream.get("r_frame_rate"), field=f"{source} r_frame_rate")

    decoded = _run_json_command(
        [
            str(resolved),
            "-v",
            "error",
            "-err_detect",
            "explode",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            (
                "frame=media_type,best_effort_timestamp,best_effort_timestamp_time,"
                "pkt_duration_time,duration_time,width,height"
            ),
            "-of",
            "json",
            str(source),
        ],
        purpose=f"sequential frame decode audit for {source}",
    )
    raw_frames = decoded.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise EgoAuditError(f"Sequential decode produced no video frames for {source}")
    frames: list[VideoFrameEvidence] = []
    previous_pts: int | None = None
    for frame_idx, frame in enumerate(raw_frames):
        if frame.get("media_type", "video") != "video":
            continue
        try:
            pts = int(frame["best_effort_timestamp"])
            frame_width = int(frame.get("width", width))
            frame_height = int(frame.get("height", height))
        except (KeyError, TypeError, ValueError) as error:
            raise EgoAuditError(
                f"Frame {frame_idx} in {source} lacks valid decode/PTS evidence"
            ) from error
        if frame_width != width or frame_height != height:
            raise EgoAuditError(
                f"Decoded frame {frame_idx} in {source} changed dimensions to "
                f"{frame_width}x{frame_height}"
            )
        if previous_pts is not None and pts <= previous_pts:
            raise EgoAuditError(
                f"MP4 presentation timestamps are not strictly increasing in {source} "
                f"at decoded frame {frame_idx}"
            )
        previous_pts = pts
        frames.append(
            VideoFrameEvidence(
                frame_idx=len(frames),
                pts=pts,
                pts_time_s=(
                    str(frame["best_effort_timestamp_time"])
                    if frame.get("best_effort_timestamp_time") is not None
                    else None
                ),
                duration_time_s=(
                    str(frame.get("pkt_duration_time", frame.get("duration_time")))
                    if frame.get("pkt_duration_time", frame.get("duration_time")) is not None
                    else None
                ),
            )
        )
    if len(frames) != len(raw_frames):
        raise EgoAuditError(f"Unexpected non-video frame records while decoding {source}")

    for field in ("nb_frames", "nb_read_frames"):
        value = stream.get(field)
        if value not in (None, "N/A") and int(value) != len(frames):
            raise EgoAuditError(
                f"{field}={value} disagrees with {len(frames)} decoded frames in {source}"
            )
    format_info = metadata.get("format") if isinstance(metadata.get("format"), dict) else {}
    summary = {
        "path": str(source.resolve()),
        "codec_name": str(stream.get("codec_name", "")),
        "codec_long_name": str(stream.get("codec_long_name", "")),
        "pixel_format": str(stream.get("pix_fmt", "")),
        "width": width,
        "height": height,
        "average_frame_rate": str(average_rate),
        "average_frame_rate_hz": float(average_rate),
        "nominal_frame_rate": str(nominal_rate),
        "nominal_frame_rate_hz": float(nominal_rate),
        "time_base": str(stream.get("time_base", "")),
        "declared_frame_count": (
            None if stream.get("nb_frames") in (None, "N/A") else int(stream["nb_frames"])
        ),
        "counted_frame_count": (
            None
            if stream.get("nb_read_frames") in (None, "N/A")
            else int(stream["nb_read_frames"])
        ),
        "decoded_frame_count": len(frames),
        "duration_s": (
            None
            if format_info.get("duration") in (None, "N/A")
            else str(format_info.get("duration"))
        ),
        "pts_strictly_increasing": True,
        "first_pts": frames[0].pts,
        "last_pts": frames[-1].pts,
        "sequential_decode_status": "complete",
    }
    return VideoProbe(summary=summary, frames=tuple(frames))


def detect_global_stereo_frame_offset(
    left: Sequence[TimestampRecord],
    right: Sequence[TimestampRecord],
    *,
    maximum_offset: int = MAX_GLOBAL_STEREO_FRAME_OFFSET,
) -> dict[str, Any]:
    """Detect an auditable constant ``right_index - left_index`` offset.

    An offset is applied only when at least 95% of a sufficiently large
    overlap has exactly equal authoritative timestamps and every competing
    offset is weak.  Otherwise pairing remains at offset zero and ordinary
    mismatch gates fail closed.
    """

    if maximum_offset < 0:
        raise ValueError("maximum_offset must be non-negative")
    candidates: list[dict[str, Any]] = []
    for offset in range(-maximum_offset, maximum_offset + 1):
        left_start = max(0, -offset)
        left_stop = min(len(left), len(right) - offset)
        overlap = max(0, left_stop - left_start)
        exact = sum(
            left[left_index].sensor_time_ns
            == right[left_index + offset].sensor_time_ns
            for left_index in range(left_start, left_stop)
        )
        candidates.append(
            {
                "right_index_minus_left_index": offset,
                "overlap_count": overlap,
                "exact_timestamp_count": exact,
                "exact_timestamp_ratio": (exact / overlap if overlap else 0.0),
            }
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            float(item["exact_timestamp_ratio"]),
            int(item["exact_timestamp_count"]),
            -abs(int(item["right_index_minus_left_index"])),
        ),
        reverse=True,
    )
    best = ranked[0]
    runner_up_ratio = float(ranked[1]["exact_timestamp_ratio"]) if len(ranked) > 1 else 0.0
    accepted = (
        int(best["overlap_count"]) >= MIN_GLOBAL_STEREO_OFFSET_OVERLAP
        and float(best["exact_timestamp_ratio"])
        >= MIN_GLOBAL_STEREO_OFFSET_EXACT_RATIO
        and runner_up_ratio <= MAX_GLOBAL_STEREO_OFFSET_RUNNER_UP_RATIO
    )
    chosen_offset = int(best["right_index_minus_left_index"]) if accepted else 0
    return {
        "applied": bool(accepted and chosen_offset != 0),
        "accepted": bool(accepted),
        "right_index_minus_left_index": chosen_offset,
        "best_candidate": best,
        "runner_up_exact_timestamp_ratio": runner_up_ratio,
        "candidates": sorted(
            candidates, key=lambda item: int(item["right_index_minus_left_index"])
        ),
        "policy": (
            "one global integer frame offset; exact timestamp equality; "
            "no per-frame nearest-neighbour matching"
        ),
    }


def pair_stereo_frames(
    left: Sequence[TimestampRecord],
    right: Sequence[TimestampRecord],
    *,
    left_video: VideoProbe,
    right_video: VideoProbe,
    imu_first_time_ns: int,
    imu_last_time_ns: int,
    common_timeshift_cam_imu_s: Decimal,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair by source index, discard bad pairs, then enforce retained monotonicity."""

    counts = {
        "left_timestamps": len(left),
        "right_timestamps": len(right),
        "left_decoded_frames": len(left_video.frames),
        "right_decoded_frames": len(right_video.frames),
    }
    if len(left) != len(right):
        raise EgoAuditError(f"Stereo source counts disagree: {counts}")
    left_decoded_count = len(left_video.frames)
    right_decoded_count = len(right_video.frames)
    if abs(left_decoded_count - right_decoded_count) > MAX_TRAILING_STEREO_VIDEO_COUNT_DELTA:
        raise EgoAuditError(f"Stereo source counts disagree: {counts}")
    # Timestamp sidecars are the absolute-time authority.  A bilaterally
    # decodable MP4 tail without timestamp rows can be localized unambiguously
    # because sidecar indices are required to be the contiguous prefix
    # 0..N-1.  Keep that timestamped common prefix and exclude the MP4-only
    # tail; never synthesize absolute timestamps from MP4 PTS.
    decoded_pair_count = min(len(left), left_decoded_count, right_decoded_count)
    trailing_timestamp_pair_count = len(left) - decoded_pair_count
    if not 0 <= trailing_timestamp_pair_count <= MAX_TRAILING_TIMESTAMP_PAIRS_WITHOUT_VIDEO:
        raise EgoAuditError(f"Stereo source counts disagree: {counts}")
    orphan_timestamp_indices = list(range(decoded_pair_count, len(left)))
    inventory: list[dict[str, Any]] = []
    previous_retained_left_ns: int | None = None
    previous_retained_right_ns: int | None = None
    reason_counts: dict[str, int] = {}
    mismatch_indices: list[int] = []
    for position in range(decoded_pair_count):
        if left[position].source_idx != position or right[position].source_idx != position:
            raise EgoAuditError(f"Stereo sidecar/source index disagreement at position {position}")
        if (
            left_video.frames[position].frame_idx != position
            or right_video.frames[position].frame_idx != position
        ):
            raise EgoAuditError(f"Decoded-frame index disagreement at position {position}")
    alignment = detect_global_stereo_frame_offset(
        left[:decoded_pair_count], right[:decoded_pair_count]
    )
    right_index_offset = int(alignment["right_index_minus_left_index"])
    offset_boundary_indices: list[int] = []
    for position in range(decoded_pair_count):
        left_record = left[position]
        right_position = position + right_index_offset
        right_available = 0 <= right_position < decoded_pair_count
        right_record = right[right_position] if right_available else None
        left_frame = left_video.frames[position]
        right_frame = right_video.frames[right_position] if right_available else None
        reasons: list[str] = []
        if not right_available:
            reasons.append(DROP_STEREO_OFFSET_BOUNDARY)
            offset_boundary_indices.append(position)
        elif left_record.sensor_time_ns != right_record.sensor_time_ns:
            reasons.append(DROP_STEREO_TIMESTAMP_MISMATCH)
            mismatch_indices.append(position)
        else:
            if (
                previous_retained_left_ns is not None
                and left_record.sensor_time_ns <= previous_retained_left_ns
            ):
                reasons.append(DROP_NON_MONOTONIC_LEFT)
            if (
                previous_retained_right_ns is not None
                and right_record is not None
                and right_record.sensor_time_ns <= previous_retained_right_ns
            ):
                reasons.append(DROP_NON_MONOTONIC_RIGHT)

        shifted_ns = Decimal(left_record.sensor_time_ns) + (
            common_timeshift_cam_imu_s * NANOSECONDS_PER_SECOND
        )
        if shifted_ns < Decimal(imu_first_time_ns):
            reasons.append(DROP_IMU_COVERAGE_START)
        elif shifted_ns > Decimal(imu_last_time_ns):
            reasons.append(DROP_IMU_COVERAGE_END)
        retained = not reasons
        if retained:
            assert right_record is not None
            previous_retained_left_ns = left_record.sensor_time_ns
            previous_retained_right_ns = right_record.sensor_time_ns
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        inventory.append(
            {
                "source_idx": position,
                "left_video_frame_idx": position,
                "right_video_frame_idx": right_position if right_available else None,
                "left_timestamp_raw": left_record.raw_seconds,
                "right_timestamp_raw": (
                    right_record.raw_seconds if right_record is not None else None
                ),
                "left_sensor_time_ns": left_record.sensor_time_ns,
                "right_sensor_time_ns": (
                    right_record.sensor_time_ns if right_record is not None else None
                ),
                "stereo_timestamp_delta_ns": (
                    left_record.sensor_time_ns - right_record.sensor_time_ns
                    if right_record is not None
                    else None
                ),
                "left_mp4_pts": left_frame.pts,
                "right_mp4_pts": right_frame.pts if right_frame is not None else None,
                "left_mp4_pts_time_s": left_frame.pts_time_s,
                "right_mp4_pts_time_s": (
                    right_frame.pts_time_s if right_frame is not None else None
                ),
                "common_shifted_imu_time_ns_decimal": str(shifted_ns),
                "retained": retained,
                "drop_reasons": reasons,
            }
        )
    retained_count = sum(bool(row["retained"]) for row in inventory)
    if orphan_timestamp_indices:
        reason_counts[DROP_MISSING_VIDEO_FRAME] = len(orphan_timestamp_indices)
    common_video_tail_without_timestamps = list(
        range(len(left), min(left_decoded_count, right_decoded_count))
    )
    return inventory, {
        "source_pair_count": len(left),
        "timestamp_pair_count": len(left),
        "decodable_pair_count": decoded_pair_count,
        "retained_pair_count": retained_count,
        "dropped_pair_count": len(left) - retained_count,
        "drop_reason_counts": dict(sorted(reason_counts.items())),
        "trailing_timestamp_pairs_without_video": orphan_timestamp_indices,
        "trailing_video_frame_pairs_without_timestamp": (
            common_video_tail_without_timestamps
        ),
        "excluded_video_frame_reason_counts": (
            {DROP_MISSING_TIMESTAMP: len(common_video_tail_without_timestamps)}
            if common_video_tail_without_timestamps
            else {}
        ),
        "trailing_unpaired_video_frame_indices": {
            "left": list(range(decoded_pair_count, left_decoded_count)),
            "right": list(range(decoded_pair_count, right_decoded_count)),
        },
        "global_frame_alignment": alignment,
        "global_offset_boundary_indices": offset_boundary_indices,
        "stereo_timestamp_mismatch_indices": mismatch_indices,
        "pairing_policy": (
            "exact timestamp after optional evidence-gated global integer frame offset; "
            "per-frame nearest-timestamp rematching forbidden"
        ),
        "monotonic_policy": "strictly increasing after mismatch/coverage filtering",
        "common_timeshift_cam_imu_s": str(common_timeshift_cam_imu_s),
    }


def build_sensor_events(
    inventory: Sequence[Mapping[str, Any]], samples: Sequence[ImuSample]
) -> list[dict[str, Any]]:
    """Create one time-ordered replay inventory for valid stereo pairs and IMU."""

    sortable: list[tuple[int, int, int, dict[str, Any]]] = []
    for sample in samples:
        event = {
            "event_type": "imu",
            "sensor_time_ns": sample.sensor_time_ns,
            "source_row": sample.source_row,
            "timestamp_raw": sample.raw_seconds,
            "angular_velocity_rad_s": list(sample.angular_velocity_rad_s),
            "linear_acceleration_m_s2": list(sample.linear_acceleration_m_s2),
            "orientation_xyzw_ignored": list(sample.orientation_xyzw),
            "orientation_valid": False,
        }
        sortable.append((sample.sensor_time_ns, 0, sample.source_row, event))
    for row in inventory:
        if not bool(row["retained"]):
            continue
        source_idx = int(row["source_idx"])
        event = {
            "event_type": "stereo_pair",
            "sensor_time_ns": int(row["left_sensor_time_ns"]),
            "source_idx": source_idx,
            "timestamp_raw": str(row["left_timestamp_raw"]),
            "left_video_frame_idx": int(row["left_video_frame_idx"]),
            "right_video_frame_idx": int(row["right_video_frame_idx"]),
            "left_mp4_pts": int(row["left_mp4_pts"]),
            "right_mp4_pts": int(row["right_mp4_pts"]),
            "left_mp4_pts_time_s": row["left_mp4_pts_time_s"],
            "right_mp4_pts_time_s": row["right_mp4_pts_time_s"],
        }
        sortable.append((event["sensor_time_ns"], 1, source_idx, event))
    sortable.sort(key=lambda item: item[:3])
    events: list[dict[str, Any]] = []
    for event_idx, (_, _, _, event) in enumerate(sortable):
        event["event_idx"] = event_idx
        events.append(event)
    return events


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(dataset_root: Path | str) -> dict[str, Any]:
    """Hash every raw source file, including explicitly unused sensor streams."""

    root = Path(dataset_root).resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in DERIVED_MATERIALIZATION_ROOTS:
            continue
        if path.is_symlink():
            raise EgoAuditError(f"Source symlinks are not allowed in the frozen manifest: {path}")
        if not path.is_file():
            continue
        stat = path.stat()
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    if not rows:
        raise EgoAuditError(f"Dataset contains no source files: {root}")
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
                "utf-8"
            )
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(root),
        "file_count": len(rows),
        "files": rows,
        "inventory_sha256": digest.hexdigest(),
        "excluded_derived_materialization_roots": sorted(
            DERIVED_MATERIALIZATION_ROOTS
        ),
        "unused_by_pipeline": [
            "videos/bright.mp4",
            "videos/bleft.mp4",
            "timestamps/bright_timestamps.txt",
            "timestamps/bleft_timestamps.txt",
            "audio/audio.wav",
        ],
    }


def _load_meta(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EgoAuditError(f"Cannot parse metadata JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise EgoAuditError(f"Metadata JSON must contain an object: {path}")
    return value


def _metadata_checks(
    meta: Mapping[str, Any],
    *,
    frame_count: int,
    imu_count: int,
    timestamp_count: int | None = None,
    decoded_frame_count: int | None = None,
) -> tuple[list[str], dict[str, Any]]:
    warnings: list[str] = []
    timestamp_count = frame_count if timestamp_count is None else timestamp_count
    decoded_frame_count = frame_count if decoded_frame_count is None else decoded_frame_count
    topics = meta.get("topics")
    if not isinstance(topics, dict):
        raise EgoAuditError("meta.json must contain a topics mapping")
    per_topic_written: dict[str, int] = {}
    for side in ("left", "right"):
        topic = topics.get(side)
        if not isinstance(topic, dict):
            raise EgoAuditError(f"meta.json is missing topics.{side}")
        try:
            written = int(topic["written_frames"])
        except (KeyError, TypeError, ValueError) as error:
            raise EgoAuditError(f"meta.json topics.{side}.written_frames is invalid") from error
        per_topic_written[side] = written
        if written not in {decoded_frame_count, timestamp_count}:
            raise EgoAuditError(
                f"meta.json topics.{side}.written_frames={written} disagrees with "
                f"decoded count {decoded_frame_count} and timestamp count {timestamp_count}"
            )
        if written != decoded_frame_count:
            warnings.append(
                f"meta.json topics.{side}.written_frames={written} matches timestamp "
                f"rows but exceeds decoded video frames {decoded_frame_count}; trailing "
                "timestamp-without-video evidence is retained"
            )
    if len(set(per_topic_written.values())) != 1:
        raise EgoAuditError(
            f"meta.json left/right written frame counts disagree: {per_topic_written}"
        )
    imu_meta = meta.get("imu")
    try:
        written_imu_samples = (
            int(imu_meta["written_samples"]) if isinstance(imu_meta, dict) else -1
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EgoAuditError("meta.json imu.written_samples must be an integer") from error
    imu_count_mismatch: dict[str, int] | None = None
    if written_imu_samples != imu_count:
        # The CSV is replayable sensor evidence; meta.json is only a
        # recorder-side counter. Parsing has already enforced finite values
        # and strict timestamp monotonicity, while stereo pairing separately
        # removes camera frames outside actual IMU coverage. Preserve the
        # disagreement instead of inventing samples or rejecting the valid
        # common interval.
        imu_count_mismatch = {
            "metadata": written_imu_samples,
            "parsed_csv": imu_count,
        }
        warnings.append(
            f"meta.json imu.written_samples={written_imu_samples} disagrees with "
            f"parsed IMU CSV count {imu_count}; parsed CSV evidence and its actual "
            "time coverage take precedence"
        )
    global_mismatches: dict[str, Any] = {}
    for field in ("synced_frames", "total_written_frames"):
        if field not in meta:
            continue
        try:
            value = int(meta[field])
        except (TypeError, ValueError) as error:
            raise EgoAuditError(f"meta.json {field} must be an integer") from error
        if value != decoded_frame_count:
            global_mismatches[field] = {
                "metadata": value,
                "actual": decoded_frame_count,
            }
            warnings.append(
                f"meta.json {field}={value} conflicts with per-topic written/decode count "
                f"{decoded_frame_count}; per-topic and decoded evidence take precedence"
            )
    if decoded_frame_count != timestamp_count:
        warnings.append(
            f"decoded stereo videos contain {decoded_frame_count} frames but timestamp "
            f"sidecars contain {timestamp_count} rows; only the timestamped common prefix "
            "is eligible for replay"
        )
    return warnings, {
        "per_topic_written_frame_count": per_topic_written["left"],
        "decoded_frame_count": decoded_frame_count,
        "timestamped_stereo_pair_count": frame_count,
        "timestamp_row_count": timestamp_count,
        "parsed_imu_sample_count": imu_count,
        "metadata_written_imu_sample_count": written_imu_samples,
        "imu_sample_count_mismatch": imu_count_mismatch,
        "global_frame_count_mismatches": global_mismatches,
    }


def _validate_required_files(root: Path) -> None:
    if not root.is_dir():
        raise EgoAuditError(f"Dataset root is not a directory: {root}")
    missing = [path.as_posix() for path in REQUIRED_SOURCE_FILES if not (root / path).is_file()]
    if missing:
        raise EgoAuditError(f"Dataset is missing required source files: {missing}")


def run_source_audit(
    dataset_root: Path | str,
    *,
    ffprobe_executable: str = "ffprobe",
) -> AuditResult:
    """Execute all S00 read-only checks and return materializable artifacts."""

    root = Path(dataset_root).resolve()
    _validate_required_files(root)
    left_timestamps = parse_timestamp_file(root / LEFT_TIMESTAMPS)
    right_timestamps = parse_timestamp_file(root / RIGHT_TIMESTAMPS)
    imu_samples = parse_imu_csv(root / IMU_CSV)
    imu_report = imu_metrics(imu_samples)
    calibration = parse_calibration(root)
    left_video = probe_video(root / LEFT_VIDEO, ffprobe_executable=ffprobe_executable)
    right_video = probe_video(root / RIGHT_VIDEO, ffprobe_executable=ffprobe_executable)

    comparable_video_fields = (
        "width",
        "height",
        "average_frame_rate",
        "nominal_frame_rate",
    )
    disagreements = {
        field: {"left": left_video.summary[field], "right": right_video.summary[field]}
        for field in comparable_video_fields
        if left_video.summary[field] != right_video.summary[field]
    }
    if disagreements:
        raise EgoAuditError(f"Left/right video stream metadata disagree: {disagreements}")

    left_resolution = calibration["cameras"]["left"]["resolution"]
    right_resolution = calibration["cameras"]["right"]["resolution"]
    decoded_resolution = [left_video.summary["width"], left_video.summary["height"]]
    if left_resolution != decoded_resolution or right_resolution != decoded_resolution:
        raise EgoAuditError(
            "Calibration and decoded video resolutions disagree: "
            f"calibration left={left_resolution}, right={right_resolution}, "
            f"decoded={decoded_resolution}"
        )

    left_shift = _decimal_text(
        calibration["cameras"]["left"]["timeshift_cam_imu_s"],
        field="left timeshift_cam_imu",
    )
    frame_inventory, pairing_report = pair_stereo_frames(
        left_timestamps,
        right_timestamps,
        left_video=left_video,
        right_video=right_video,
        imu_first_time_ns=imu_samples[0].sensor_time_ns,
        imu_last_time_ns=imu_samples[-1].sensor_time_ns,
        common_timeshift_cam_imu_s=left_shift,
    )
    events = build_sensor_events(frame_inventory, imu_samples)
    meta = _load_meta(root / META_JSON)
    warnings, metadata_report = _metadata_checks(
        meta,
        frame_count=len(frame_inventory),
        imu_count=len(imu_samples),
        timestamp_count=len(left_timestamps),
        decoded_frame_count=len(left_video.frames),
    )
    source_manifest = build_source_manifest(root)
    ffprobe_version = tool_version(ffprobe_executable)
    created_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "passed": True,
        "created_at_utc": created_at,
        "dataset_root": str(root),
        "hard_failures": [],
        "warnings": warnings,
        "policy": {
            "absolute_time_authority": "timestamp sidecar decimal text",
            "internal_time": "signed int64 Unix epoch nanoseconds",
            "mp4_pts_usage": "decode order/integrity evidence only",
            "stereo_pairing": (
                "exact timestamp after optional evidence-gated global integer frame "
                "offset; per-frame nearest-timestamp rematching forbidden"
            ),
            "imu_orientation": "ignored because the source quaternion is invalid/all-zero",
        },
        "videos": {
            "left": left_video.summary,
            "right": right_video.summary,
        },
        "timestamps": {
            "left": timestamp_stream_metrics(left_timestamps),
            "right": timestamp_stream_metrics(right_timestamps),
        },
        "imu": imu_report,
        "calibration": {
            "baseline_m": calibration["stereo"]["baseline_m"],
            "relative_rotation_deg": calibration["stereo"]["relative_rotation_deg"],
            "camera_imu_derived_max_abs_error": calibration["stereo"][
                "camera_imu_derived_max_abs_error"
            ],
            "common_timeshift_cam_imu_s": calibration["cameras"]["left"][
                "timeshift_cam_imu_s"
            ],
            "right_minus_left_timeshift_ms": calibration["stereo"][
                "right_minus_left_timeshift_ms"
            ],
        },
        "stereo_pairing": pairing_report,
        "metadata": metadata_report,
        "source_manifest": {
            "file_count": source_manifest["file_count"],
            "inventory_sha256": source_manifest["inventory_sha256"],
        },
        "sensor_events": {
            "count": len(events),
            "imu_count": len(imu_samples),
            "stereo_pair_count": pairing_report["retained_pair_count"],
            "strictly_time_sorted": all(
                int(events[index]["sensor_time_ns"])
                <= int(events[index + 1]["sensor_time_ns"])
                for index in range(len(events) - 1)
            ),
        },
        "tools": {
            "ffprobe": ffprobe_version,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyyaml": yaml.__version__,
        },
        "artifacts": {name: name for name in OUTPUT_FILENAMES},
    }
    return AuditResult(
        source_audit=audit,
        frame_inventory=tuple(frame_inventory),
        sensor_events=tuple(events),
        calibration_normalized=calibration,
        source_manifest=source_manifest,
    )


def _atomic_text_writer(path: Path, chunks: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_json(path: Path | str, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    _atomic_text_writer(Path(path), [payload])


def atomic_write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> None:
    def chunks() -> Iterable[str]:
        for row in rows:
            yield json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"

    _atomic_text_writer(Path(path), chunks())


def materialize_audit(result: AuditResult, output_dir: Path | str, *, overwrite: bool = False) -> None:
    """Atomically write every S00 artifact, with source_audit as completion marker."""

    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise EgoAuditError(f"Audit output is not a directory: {output}")
    existing = [output / name for name in OUTPUT_FILENAMES if (output / name).exists()]
    if existing and not overwrite:
        raise EgoAuditError(
            "Audit artifacts already exist; pass overwrite=True to replace the known files: "
            + ", ".join(str(path) for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "calibration_normalized.json", result.calibration_normalized)
    atomic_write_json(output / "source_manifest.json", result.source_manifest)
    atomic_write_jsonl(output / "frame_inventory.jsonl", result.frame_inventory)
    atomic_write_jsonl(output / "sensor_events.jsonl", result.sensor_events)
    # Written last so downstream stages can treat this file as a completion marker.
    atomic_write_json(output / "source_audit.json", result.source_audit)


def audit_and_materialize(
    dataset_root: Path | str,
    output_dir: Path | str,
    *,
    ffprobe_executable: str = "ffprobe",
    overwrite: bool = False,
) -> AuditResult:
    root = Path(dataset_root).resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise EgoAuditError(
            "Audit output must be outside the immutable raw dataset directory: "
            f"{output}"
        )
    result = run_source_audit(root, ffprobe_executable=ffprobe_executable)
    materialize_audit(result, output, overwrite=overwrite)
    return result


__all__ = [
    "AuditResult",
    "EgoAuditError",
    "ImuSample",
    "TimestampRecord",
    "VideoFrameEvidence",
    "VideoProbe",
    "atomic_write_json",
    "atomic_write_jsonl",
    "audit_and_materialize",
    "build_sensor_events",
    "build_source_manifest",
    "decimal_seconds_to_ns",
    "detect_global_stereo_frame_offset",
    "imu_metrics",
    "materialize_audit",
    "pair_stereo_frames",
    "parse_calibration",
    "parse_imu_csv",
    "parse_timestamp_file",
    "probe_video",
    "run_source_audit",
    "timestamp_stream_metrics",
]
