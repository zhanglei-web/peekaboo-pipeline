#!/usr/bin/env python3
"""Select auditable, role-aware Ego keyframes from full-rate camera poses."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import asdict, dataclass
from decimal import Decimal
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class SelectionConfig:
    soft_translation_m: float = 0.06
    soft_rotation_deg: float = 5.0
    hard_translation_m: float = 0.15
    hard_rotation_deg: float = 12.0
    minimum_interval_s: float = 0.15
    maximum_gap_s: float = 0.25
    minimum_median_pixel_motion_px: float = 6.0
    maximum_view_overlap_ratio: float = 0.90
    minimum_new_visible_ratio: float = 0.10
    minimum_depth_support_ratio: float = 0.20

    def validate(self) -> None:
        if (
            self.soft_translation_m <= 0.0
            or self.soft_rotation_deg <= 0.0
            or self.hard_translation_m < self.soft_translation_m
            or self.hard_rotation_deg < self.soft_rotation_deg
            or self.minimum_interval_s < 0.0
            or self.maximum_gap_s <= 0.0
            or self.minimum_interval_s >= self.maximum_gap_s
            or self.minimum_median_pixel_motion_px <= 0.0
            or not 0.0 < self.maximum_view_overlap_ratio <= 1.0
            or not 0.0 <= self.minimum_new_visible_ratio < 1.0
            or not 0.0 <= self.minimum_depth_support_ratio <= 1.0
        ):
            raise ValueError("invalid keyframe selection thresholds")


@dataclass(frozen=True)
class FixedRateSelectionConfig:
    """Hard gates for at most one content-bearing map keyframe per tick."""

    map_rate_hz: float = 1.0
    candidate_tail_s: float = 0.35
    minimum_quality_score: float = 0.25
    minimum_stereo_matchability: float = 0.20
    minimum_stereo_match_count: int = 50
    maximum_position_std_m: float = 0.25
    maximum_rotation_std_deg: float = 8.0
    minimum_temporal_change_score: float = 0.035
    minimum_cumulative_change_score: float = 0.10

    def validate(self) -> None:
        period_s = 1.0 / self.map_rate_hz if self.map_rate_hz > 0.0 else 0.0
        if (
            not math.isfinite(self.map_rate_hz)
            or self.map_rate_hz <= 0.0
            or not math.isfinite(self.candidate_tail_s)
            or self.candidate_tail_s <= 0.0
            or self.candidate_tail_s > period_s
            or not 0.0 <= self.minimum_quality_score <= 1.0
            or not 0.0 <= self.minimum_stereo_matchability <= 1.0
            or self.minimum_stereo_match_count < 1
            or self.maximum_position_std_m <= 0.0
            or self.maximum_rotation_std_deg <= 0.0
            or self.minimum_temporal_change_score <= 0.0
            or self.minimum_cumulative_change_score <= 0.0
        ):
            raise ValueError("invalid fixed-rate keyframe settings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-poses", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--soft-translation-m", type=float, default=0.06)
    parser.add_argument("--soft-rotation-deg", type=float, default=5.0)
    parser.add_argument("--hard-translation-m", type=float, default=0.15)
    parser.add_argument("--hard-rotation-deg", type=float, default=12.0)
    parser.add_argument("--minimum-interval-s", type=float, default=0.15)
    parser.add_argument("--maximum-gap-s", type=float, default=0.25)
    parser.add_argument("--minimum-median-pixel-motion-px", type=float, default=6.0)
    parser.add_argument("--maximum-view-overlap-ratio", type=float, default=0.90)
    parser.add_argument("--minimum-new-visible-ratio", type=float, default=0.10)
    parser.add_argument("--minimum-depth-support-ratio", type=float, default=0.20)
    parser.add_argument("--quality-left-video", type=Path)
    parser.add_argument("--quality-right-video", type=Path)
    parser.add_argument("--quality-calibration", type=Path)
    parser.add_argument("--imu-csv", type=Path)
    parser.add_argument("--quality-window-radius-frames", type=int, default=0)
    parser.add_argument("--quality-analysis-width", type=int, default=480)
    parser.add_argument("--minimum-quality-improvement", type=float, default=0.015)
    parser.add_argument(
        "--fixed-map-rate-hz",
        type=float,
        help="Select exactly one causal map keyframe per fixed-rate sensor window.",
    )
    parser.add_argument("--fixed-candidate-tail-s", type=float, default=0.35)
    parser.add_argument("--fixed-minimum-quality-score", type=float, default=0.25)
    parser.add_argument("--fixed-minimum-stereo-matchability", type=float, default=0.20)
    parser.add_argument("--fixed-minimum-stereo-match-count", type=int, default=50)
    parser.add_argument("--fixed-maximum-position-std-m", type=float, default=0.25)
    parser.add_argument("--fixed-maximum-rotation-std-deg", type=float, default=8.0)
    parser.add_argument(
        "--fixed-minimum-temporal-change-score", type=float, default=0.035
    )
    parser.add_argument(
        "--fixed-minimum-cumulative-change-score", type=float, default=0.10
    )
    parser.add_argument(
        "--preserve-legacy-trigger-count",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep the legacy motion selector's frame budget while using the "
            "view gate to place those frames."
        ),
    )
    parser.add_argument(
        "--prior-bundle",
        type=Path,
        help=(
            "Optional daaam.ego_prior_bundle.v1 file. Real camera samples inside "
            "episode windows become user-definition candidates."
        ),
    )
    parser.add_argument("--prior-pre-roll-s", type=float, default=2.0)
    parser.add_argument("--prior-post-roll-s", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSONL line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(f"camera pose line {line_number} is not an object")
            records.append(value)
    if not records:
        raise ValueError(f"camera pose stream is empty: {path}")
    return records


def load_prior_event_intervals(
    path: Path, *, pre_roll_s: float, post_roll_s: float
) -> tuple[tuple[int, int], ...]:
    """Load only episode time ranges; no absent image is synthesized."""

    if pre_roll_s < 0.0 or post_roll_s < 0.0:
        raise ValueError("prior event pre/post roll must be nonnegative")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema") != (
        "daaam.ego_prior_bundle.v1"
    ):
        raise ValueError("unsupported prior bundle schema")
    episodes = document.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError("prior bundle episodes must be a list")
    pre_ns = round(pre_roll_s * 1.0e9)
    post_ns = round(post_roll_s * 1.0e9)
    intervals = []
    for index, value in enumerate(episodes):
        if not isinstance(value, dict) or not isinstance(value.get("episode"), dict):
            raise ValueError(f"prior bundle episode {index} is invalid")
        episode = value["episode"]
        try:
            start_ns = int(episode["start_sensor_time_ns"])
            end_ns = int(episode["end_sensor_time_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"prior bundle episode {index} time is invalid") from exc
        if start_ns <= 0 or end_ns < start_ns:
            raise ValueError(f"prior bundle episode {index} time is invalid")
        intervals.append((max(1, start_ns - pre_ns), end_ns + post_ns))
    return tuple(sorted(set(intervals)))


def record_source_index(record: dict[str, Any]) -> int:
    for name in ("source_idx", "source_index", "frame_index"):
        if name in record:
            return int(record[name])
    raise ValueError("camera pose record has no source index")


def record_pose(record: dict[str, Any]) -> np.ndarray:
    for name in ("world_T_cam0", "world_T_camera", "world_T_left"):
        if name in record:
            pose = np.asarray(record[name], dtype=np.float64)
            break
    else:
        raise ValueError("camera pose record has no world_T_cam0")
    if (
        pose.shape != (4, 4)
        or not np.all(np.isfinite(pose))
        or not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1.0e-9)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1.0e-5)
        or not math.isclose(float(np.linalg.det(pose[:3, :3])), 1.0, abs_tol=1.0e-5)
    ):
        raise ValueError("world_T_cam0 must be a finite rigid homogeneous transform")
    return pose


def record_time(record: dict[str, Any]) -> int:
    try:
        value = int(record["sensor_time_ns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("camera pose record has invalid sensor_time_ns") from error
    if value <= 0:
        raise ValueError("sensor_time_ns must be positive")
    return value


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def selection_reason(
    *,
    translation_m: float,
    rotation_deg: float,
    gap_s: float,
    config: SelectionConfig,
) -> str | None:
    """Legacy metric gate retained for API compatibility and safety audits."""

    if translation_m >= config.hard_translation_m:
        return "hard_translation"
    if rotation_deg >= config.hard_rotation_deg:
        return "hard_rotation"
    if gap_s >= config.maximum_gap_s:
        return "maximum_gap"
    if gap_s < config.minimum_interval_s:
        return None
    if translation_m >= config.soft_translation_m:
        return "soft_translation"
    if rotation_deg >= config.soft_rotation_deg:
        return "soft_rotation"
    return None


def _all_metric_reasons(
    translation_m: float,
    rotation_deg: float,
    gap_s: float,
    config: SelectionConfig,
) -> list[str]:
    reasons = []
    if translation_m >= config.hard_translation_m:
        reasons.append("hard_translation")
    if rotation_deg >= config.hard_rotation_deg:
        reasons.append("hard_rotation")
    if gap_s >= config.maximum_gap_s:
        reasons.append("maximum_gap")
    if gap_s >= config.minimum_interval_s:
        if translation_m >= config.soft_translation_m:
            reasons.append("soft_translation")
        if rotation_deg >= config.soft_rotation_deg:
            reasons.append("soft_rotation")
    return reasons


def _normalize_records(
    records: Iterable[dict[str, Any]],
) -> list[tuple[dict[str, Any], int, int, np.ndarray]]:
    normalized = [
        (record, record_source_index(record), record_time(record), record_pose(record))
        for record in records
    ]
    if not normalized:
        raise ValueError("no camera poses to select")
    if any(right[1] <= left[1] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("camera pose source indices must increase strictly")
    if any(right[2] <= left[2] for left, right in zip(normalized, normalized[1:])):
        raise ValueError("camera pose timestamps must increase strictly")
    return normalized


def select_keyframes(
    records: Iterable[dict[str, Any]], config: SelectionConfig
) -> list[dict[str, Any]]:
    """Legacy pose gate used when image-space evidence is not supplied."""

    config.validate()
    normalized = _normalize_records(records)
    selected: list[dict[str, Any]] = []

    def append(
        item: tuple[dict[str, Any], int, int, np.ndarray],
        reason: str,
        translation_m: float,
        rotation_deg: float,
        gap_s: float,
    ) -> None:
        record, source_index, timestamp, _ = item
        output = dict(record)
        all_reasons = _all_metric_reasons(
            translation_m, rotation_deg, gap_s, config
        ) or [reason]
        output.update(
            {
                "idx": len(selected),
                "pose_row": len(selected),
                "source_idx": source_index,
                "sensor_time_ns": timestamp,
                "selection_reason": reason,
                "selection_reasons": all_reasons,
                "all_selection_reasons": all_reasons,
                "translation_from_previous_selected_m": translation_m,
                "rotation_from_previous_selected_deg": rotation_deg,
                "gap_from_previous_selected_s": gap_s,
            }
        )
        selected.append(output)

    append(normalized[0], "first", 0.0, 0.0, 0.0)
    last = normalized[0]
    selected_sources = {last[1]}
    for item in normalized[1:]:
        translation_m = float(np.linalg.norm(item[3][:3, 3] - last[3][:3, 3]))
        rotation_deg = rotation_distance_deg(last[3], item[3])
        gap_s = (item[2] - last[2]) / 1.0e9
        reason = selection_reason(
            translation_m=translation_m,
            rotation_deg=rotation_deg,
            gap_s=gap_s,
            config=config,
        )
        if reason is None:
            continue
        append(item, reason, translation_m, rotation_deg, gap_s)
        last = item
        selected_sources.add(item[1])
    final = normalized[-1]
    if final[1] not in selected_sources:
        translation_m = float(np.linalg.norm(final[3][:3, 3] - last[3][:3, 3]))
        rotation_deg = rotation_distance_deg(last[3], final[3])
        gap_s = (final[2] - last[2]) / 1.0e9
        append(final, "last", translation_m, rotation_deg, gap_s)
    return selected


def load_analysis_camera(path: Path, analysis_width: int) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text()) or {}
    rectified = raw.get("rectified", {})
    if (
        not rectified
        and isinstance(raw.get("cam0"), dict)
        and isinstance(raw.get("cam1"), dict)
    ):
        left = raw["cam0"]
        right = raw["cam1"]
        resolution = tuple(int(value) for value in left.get("resolution", ()))
        if (
            resolution != tuple(int(value) for value in right.get("resolution", ()))
            or len(resolution) != 2
            or left.get("camera_model") != "pinhole"
            or right.get("camera_model") != "pinhole"
            or left.get("distortion_model") != "equidistant"
            or right.get("distortion_model") != "equidistant"
            or analysis_width <= 0
        ):
            raise ValueError(f"invalid raw Kalibr stereo calibration: {path}")

        def matrix(camera: dict[str, Any]) -> np.ndarray:
            fx, fy, cx, cy = (float(value) for value in camera["intrinsics"])
            return np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

        left_K = matrix(left)
        right_K = matrix(right)
        left_D = np.asarray(left["distortion_coeffs"], dtype=np.float64)
        right_D = np.asarray(right["distortion_coeffs"], dtype=np.float64)
        right_T_left = np.asarray(right.get("T_cn_cnm1"), dtype=np.float64)
        if (
            left_D.shape != (4,)
            or right_D.shape != (4,)
            or right_T_left.shape != (4, 4)
        ):
            raise ValueError(f"invalid raw Kalibr stereo calibration: {path}")
        analysis_size = (
            int(analysis_width),
            max(1, int(round(resolution[1] * analysis_width / resolution[0]))),
        )
        R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(
            left_K,
            left_D,
            right_K,
            right_D,
            resolution,
            right_T_left[:3, :3],
            right_T_left[:3, 3],
            flags=cv2.CALIB_ZERO_DISPARITY,
            newImageSize=analysis_size,
            balance=0.0,
            fov_scale=1.0,
        )
        left_maps = cv2.fisheye.initUndistortRectifyMap(
            left_K,
            left_D,
            R1,
            P1[:, :3],
            analysis_size,
            cv2.CV_32FC1,
        )
        right_maps = cv2.fisheye.initUndistortRectifyMap(
            right_K,
            right_D,
            R2,
            P2[:, :3],
            analysis_size,
            cv2.CV_32FC1,
        )
        baseline_m = abs(float(P2[0, 3] / P2[0, 0]))
        if not np.isfinite(P1).all() or baseline_m <= 0.0:
            raise ValueError(f"invalid raw Kalibr rectification: {path}")
        return {
            "intrinsics": P1[:, :3],
            "original_to_rectified_rotation": R1,
            "baseline_m": baseline_m,
            "analysis_width": analysis_size[0],
            "analysis_height": analysis_size[1],
            "source": str(path.resolve()),
            "source_model": "kalibr_pinhole_equidistant_raw",
            "left_rectification_maps": left_maps,
            "right_rectification_maps": right_maps,
        }
    image_size = rectified.get("image_size")
    intrinsics = np.asarray(rectified.get("K1_rect"), dtype=np.float64)
    rotation = np.asarray(rectified.get("R1"), dtype=np.float64)
    baseline_m = abs(float(raw.get("stereo_params", {}).get("baseline_m", 0.0)))
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or intrinsics.shape != (3, 3)
        or rotation.shape != (3, 3)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-6)
        or baseline_m <= 0.0
        or analysis_width <= 0
    ):
        raise ValueError(f"invalid rectified camera calibration: {path}")
    scale = analysis_width / int(image_size[0])
    scaled = intrinsics.copy()
    scaled[0, :] *= scale
    scaled[1, :] *= scale
    scaled[2, :] = [0.0, 0.0, 1.0]
    crop = raw.get("crop_roi", {})
    scaled[0, 2] -= float(crop.get("x_min", 0.0)) * scale
    scaled[1, 2] -= float(crop.get("y_min", 0.0)) * scale
    analysis_height = round(
        (float(crop.get("y_max", image_size[1])) - float(crop.get("y_min", 0.0)))
        * scale
    )
    return {
        "intrinsics": scaled,
        "original_to_rectified_rotation": rotation,
        "baseline_m": baseline_m,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "source": str(path.resolve()),
    }


def _analysis_gray(image: np.ndarray, analysis_width: int) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("quality video yielded a non-BGR frame")
    if image.shape[1] > analysis_width:
        scale = analysis_width / image.shape[1]
        image = cv2.resize(
            image,
            (analysis_width, max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _visual_quality(image: np.ndarray, analysis_width: int) -> dict[str, float]:
    gray = _analysis_gray(image, analysis_width)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "sharpness": float(np.var(laplacian)),
        "texture": float(np.mean(cv2.magnitude(sobel_x, sobel_y))),
        "clipped_ratio": float(
            np.count_nonzero((gray <= 5) | (gray >= 250)) / gray.size
        ),
        "mean_luma": float(np.mean(gray)),
    }


def _stereo_match_quality(
    left_image: np.ndarray, right_image: np.ndarray, analysis_width: int
) -> dict[str, float | int]:
    left = _analysis_gray(left_image, analysis_width)
    right = _analysis_gray(right_image, analysis_width)
    detector = cv2.ORB_create(nfeatures=500, fastThreshold=12)
    left_points, left_descriptors = detector.detectAndCompute(left, None)
    right_points, right_descriptors = detector.detectAndCompute(right, None)
    if left_descriptors is None or right_descriptors is None:
        return {
            "stereo_keypoints_min": min(len(left_points), len(right_points)),
            "stereo_match_count": 0,
            "stereo_matchable_ratio": 0.0,
            "stereo_median_disparity_px": 0.0,
        }
    disparities = []
    for pair in cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(
        left_descriptors, right_descriptors, k=2
    ):
        if len(pair) != 2 or pair[0].distance >= 0.80 * pair[1].distance:
            continue
        left_xy = left_points[pair[0].queryIdx].pt
        right_xy = right_points[pair[0].trainIdx].pt
        disparity = left_xy[0] - right_xy[0]
        if abs(left_xy[1] - right_xy[1]) <= 2.0 and 0.25 <= disparity < left.shape[1]:
            disparities.append(disparity)
    keypoints_min = min(len(left_points), len(right_points))
    return {
        "stereo_keypoints_min": keypoints_min,
        "stereo_match_count": len(disparities),
        "stereo_matchable_ratio": len(disparities) / max(1, keypoints_min),
        "stereo_median_disparity_px": float(np.median(disparities))
        if disparities
        else 0.0,
    }


def load_imu_angular_velocity(path: Path) -> tuple[list[int], list[float]]:
    times_ns: list[int] = []
    angular_velocity: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "timestamp",
            "angular_velocity_x",
            "angular_velocity_y",
            "angular_velocity_z",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"IMU CSV is missing angular-velocity columns: {path}")
        for row_number, row in enumerate(reader, 2):
            try:
                timestamp_ns = int(Decimal(row["timestamp"]) * Decimal(1_000_000_000))
                vector = np.asarray(
                    [
                        float(row["angular_velocity_x"]),
                        float(row["angular_velocity_y"]),
                        float(row["angular_velocity_z"]),
                    ],
                    dtype=np.float64,
                )
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid IMU CSV row {row_number}") from error
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"non-finite IMU angular velocity at row {row_number}")
            if times_ns and timestamp_ns <= times_ns[-1]:
                raise ValueError("IMU timestamps must increase strictly")
            times_ns.append(timestamp_ns)
            angular_velocity.append(float(np.linalg.norm(vector)))
    if not times_ns:
        raise ValueError(f"IMU CSV is empty: {path}")
    return times_ns, angular_velocity


def _interpolated_imu_angular_velocity(
    timestamp_ns: int, times_ns: list[int], values: list[float]
) -> tuple[float, float]:
    right = bisect_left(times_ns, timestamp_ns)
    if right == 0:
        return values[0], (times_ns[0] - timestamp_ns) / 1.0e6
    if right == len(times_ns):
        return values[-1], (timestamp_ns - times_ns[-1]) / 1.0e6
    left = right - 1
    alpha = (timestamp_ns - times_ns[left]) / (times_ns[right] - times_ns[left])
    value = (1.0 - alpha) * values[left] + alpha * values[right]
    bracket_distance_ms = (
        max(timestamp_ns - times_ns[left], times_ns[right] - timestamp_ns) / 1.0e6
    )
    return float(value), float(bracket_distance_ms)


def _pose_covariance_trace(record: dict[str, Any]) -> float:
    covariance = np.asarray(record.get("pose_covariance", []), dtype=np.float64)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        return 0.0
    return max(0.0, float(np.trace(covariance)))


def _pose_angular_velocity_deg_s(
    normalized: list[tuple[dict[str, Any], int, int, np.ndarray]], index: int
) -> float:
    if len(normalized) < 2:
        return 0.0
    left = max(0, index - 1)
    right = min(len(normalized) - 1, index + 1)
    duration_s = (normalized[right][2] - normalized[left][2]) / 1.0e9
    return (
        rotation_distance_deg(normalized[left][3], normalized[right][3]) / duration_s
        if duration_s > 0.0
        else 0.0
    )


def score_quality_candidates(
    records: list[dict[str, Any]],
    source_indices: set[int],
    left_video: Path,
    right_video: Path,
    imu_csv: Path,
    analysis_width: int,
    camera: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Decode full-rate stereo once and score every requested pose frame."""

    normalized = _normalize_records(records)
    by_source = {item[1]: (position, item) for position, item in enumerate(normalized)}
    if source_indices - set(by_source):
        raise ValueError("quality candidates contain frames without OpenVINS poses")
    imu_times, imu_values = load_imu_angular_velocity(imu_csv)
    left_capture = cv2.VideoCapture(str(left_video))
    right_capture = cv2.VideoCapture(str(right_video))
    if not left_capture.isOpened() or not right_capture.isOpened():
        left_capture.release()
        right_capture.release()
        raise RuntimeError("failed to open both rectified quality videos")
    scores: dict[int, dict[str, Any]] = {}
    previous_change_image: np.ndarray | None = None
    try:
        next_left_frame_idx = 0
        next_right_frame_idx = 0
        for source_index in sorted(source_indices):
            record_position, (record, _, _, _) = by_source[source_index]
            left_frame_idx = int(record.get("left_video_frame_idx", source_index))
            right_frame_idx = int(record.get("right_video_frame_idx", source_index))
            if (
                left_frame_idx < next_left_frame_idx
                or right_frame_idx < next_right_frame_idx
            ):
                raise RuntimeError("quality video frame indices are not increasing")
            left_image = None
            while next_left_frame_idx <= left_frame_idx:
                left_ok, left_image = left_capture.read()
                if not left_ok:
                    raise RuntimeError(
                        f"left quality video ended at frame {next_left_frame_idx}"
                    )
                next_left_frame_idx += 1
            right_image = None
            while next_right_frame_idx <= right_frame_idx:
                right_ok, right_image = right_capture.read()
                if not right_ok:
                    raise RuntimeError(
                        f"right quality video ended at frame {next_right_frame_idx}"
                    )
                next_right_frame_idx += 1
            assert left_image is not None and right_image is not None
            if "left_rectification_maps" in camera:
                left_image = cv2.remap(
                    left_image,
                    camera["left_rectification_maps"][0],
                    camera["left_rectification_maps"][1],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                right_image = cv2.remap(
                    right_image,
                    camera["right_rectification_maps"][0],
                    camera["right_rectification_maps"][1],
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
            change_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            change_height = max(
                1,
                int(round(change_gray.shape[0] * 160 / change_gray.shape[1])),
            )
            change_image = (
                cv2.resize(
                    change_gray, (160, change_height), interpolation=cv2.INTER_AREA
                ).astype(np.float32)
                / 255.0
            )
            change_image = np.clip(
                (change_image - float(np.mean(change_image)))
                / max(0.05, float(np.std(change_image))),
                -3.0,
                3.0,
            )
            temporal_change_score = (
                float(np.mean(np.abs(change_image - previous_change_image)) / 6.0)
                if previous_change_image is not None
                and previous_change_image.shape == change_image.shape
                else 0.0
            )
            previous_change_image = change_image
            left = _visual_quality(left_image, analysis_width)
            right = _visual_quality(right_image, analysis_width)
            stereo = _stereo_match_quality(left_image, right_image, analysis_width)
            sharpness = min(left["sharpness"], right["sharpness"])
            texture = min(left["texture"], right["texture"])
            clipped_ratio = max(left["clipped_ratio"], right["clipped_ratio"])
            exposure_balance = min(
                1.0 - abs(left["mean_luma"] - 127.5) / 127.5,
                1.0 - abs(right["mean_luma"] - 127.5) / 127.5,
            )
            covariance_trace = _pose_covariance_trace(record)
            imu_omega, imu_bracket_ms = _interpolated_imu_angular_velocity(
                record_time(record), imu_times, imu_values
            )
            query_ns = int(record.get("pose_query_time_ns", record_time(record)))
            bracket_left = record.get(
                "pose_left_bracket_ns", record.get("openvins_left_bracket_ns")
            )
            bracket_right = record.get(
                "pose_right_bracket_ns", record.get("openvins_right_bracket_ns")
            )
            bracket_valid = (
                bracket_left is not None
                and bracket_right is not None
                and int(bracket_left) <= query_ns <= int(bracket_right)
            )
            bracket_distance_ms = (
                max(query_ns - int(bracket_left), int(bracket_right) - query_ns) / 1.0e6
                if bracket_valid
                else None
            )
            reported_pose_confidence = record.get("pose_confidence")
            covariance_pose_quality = 1.0 / (1.0 + 250.0 * covariance_trace)
            if reported_pose_confidence is None:
                pose_quality = covariance_pose_quality
                pose_quality_source = "covariance"
            else:
                confidence = max(0.0, min(100.0, float(reported_pose_confidence)))
                pose_quality = confidence / 100.0
                pose_quality_source = "reported_tracking_confidence"
            components = {
                "sharpness": sharpness / (sharpness + 100.0),
                "texture": texture / (texture + 20.0),
                "stereo_matchability": min(
                    1.0, float(stereo["stereo_matchable_ratio"]) / 0.55
                ),
                "exposure": max(
                    0.0, min(1.0, (1.0 - 2.0 * clipped_ratio) * exposure_balance)
                ),
                "imu_angular": 1.0 / (1.0 + imu_omega / 2.0),
                "pose": pose_quality,
                "pose_bracket": (
                    1.0 / (1.0 + bracket_distance_ms / 5.0)
                    if bracket_distance_ms is not None
                    else 0.0
                ),
            }
            score = (
                0.25 * components["sharpness"]
                + 0.10 * components["texture"]
                + 0.20 * components["stereo_matchability"]
                + 0.10 * components["exposure"]
                + 0.10 * components["imu_angular"]
                + 0.15 * components["pose"]
                + 0.10 * components["pose_bracket"]
            )
            disparity_px = float(stereo["stereo_median_disparity_px"])
            scores[source_index] = {
                "source_idx": source_index,
                "score": float(score),
                "sharpness": sharpness,
                "texture": texture,
                "clipped_ratio": clipped_ratio,
                "exposure_balance": exposure_balance,
                **stereo,
                "stereo_median_depth_m": (
                    float(camera["intrinsics"][0, 0])
                    * float(camera["baseline_m"])
                    / disparity_px
                    if disparity_px > 0.0
                    else None
                ),
                "imu_angular_velocity_rad_s": imu_omega,
                "imu_interpolation_bracket_distance_ms": imu_bracket_ms,
                "pose_covariance_trace": covariance_trace,
                "pose_quality_source": pose_quality_source,
                "reported_pose_confidence": reported_pose_confidence,
                "pose_angular_velocity_deg_s": _pose_angular_velocity_deg_s(
                    normalized, record_position
                ),
                "pose_interpolation_bracket_distance_ms": bracket_distance_ms,
                "pose_interpolation_bracket_valid": bracket_valid,
                "temporal_change_score": temporal_change_score,
                "components": components,
            }
    finally:
        left_capture.release()
        right_capture.release()
    missing = source_indices - set(scores)
    if missing:
        raise RuntimeError(
            f"quality videos ended before source frames: {sorted(missing)[:5]}"
        )
    return scores


def estimate_view_geometry(
    previous_record: dict[str, Any],
    current_record: dict[str, Any],
    quality: dict[str, Any],
    camera: dict[str, Any],
) -> dict[str, Any]:
    intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64)
    rotation = np.asarray(camera["original_to_rectified_rotation"], dtype=np.float64)
    width = int(camera["analysis_width"])
    height = int(camera["analysis_height"])
    depth_m = min(20.0, max(0.3, float(quality.get("stereo_median_depth_m") or 3.0)))
    pixels = np.asarray(
        [
            (x, y, 1.0)
            for y in np.linspace(0.08 * height, 0.92 * height, 7)
            for x in np.linspace(0.08 * width, 0.92 * width, 12)
        ]
    )
    rays = (np.linalg.inv(intrinsics) @ pixels.T).T
    original_T_rectified = np.eye(4)
    original_T_rectified[:3, :3] = rotation.T
    world_T_previous = record_pose(previous_record) @ original_T_rectified
    world_T_current = record_pose(current_record) @ original_T_rectified
    current_T_previous = np.linalg.inv(world_T_current) @ world_T_previous
    points = (current_T_previous[:3, :3] @ (rays * depth_m).T).T
    points += current_T_previous[:3, 3]
    positive = points[:, 2] > 1.0e-3
    projected = np.full((len(points), 2), np.nan)
    projected[positive] = (intrinsics @ points[positive].T).T[:, :2] / points[
        positive, 2:3
    ]
    inside = (
        positive
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )
    displacement = np.linalg.norm(projected[positive] - pixels[positive, :2], axis=1)
    overlap = float(np.count_nonzero(inside) / len(points))
    return {
        "median_pixel_motion_px": float(np.median(displacement))
        if displacement.size
        else 1.0e9,
        "view_overlap_ratio": overlap,
        "new_visible_ratio": 1.0 - overlap,
        "depth_support_ratio": float(quality.get("stereo_matchable_ratio", 0.0)),
        "representative_depth_m": depth_m,
        "representative_depth_source": (
            "stereo_sparse_matches"
            if quality.get("stereo_median_depth_m") is not None
            else "fallback_3m"
        ),
    }


def _adaptive_reasons(
    translation_m: float,
    rotation_deg: float,
    gap_s: float,
    view: dict[str, Any],
    config: SelectionConfig,
) -> list[str]:
    reasons: list[str] = []
    if gap_s >= config.maximum_gap_s:
        reasons.append("maximum_gap")
    if translation_m >= config.hard_translation_m:
        reasons.append("hard_translation_safety")
    if rotation_deg >= config.hard_rotation_deg:
        reasons.append("hard_rotation_safety")
    if gap_s >= config.minimum_interval_s:
        supported = view["depth_support_ratio"] >= config.minimum_depth_support_ratio
        if (
            supported
            and view["median_pixel_motion_px"] >= config.minimum_median_pixel_motion_px
        ):
            reasons.append("view_pixel_motion")
        if (
            supported
            and view["view_overlap_ratio"] <= config.maximum_view_overlap_ratio
        ):
            reasons.append("view_overlap")
        if supported and view["new_visible_ratio"] >= config.minimum_new_visible_ratio:
            reasons.append("new_visible_area")
        if not supported and translation_m >= config.soft_translation_m:
            reasons.append("soft_translation_unobservable_safety")
        if not supported and rotation_deg >= config.soft_rotation_deg:
            reasons.append("soft_rotation_unobservable_safety")
    return reasons


def adaptive_selection_reason(
    *,
    translation_m: float,
    rotation_deg: float,
    gap_s: float,
    view: dict[str, Any],
    config: SelectionConfig,
) -> str | None:
    """Expose the deterministic view-first priority for tests and audits."""

    reasons = _adaptive_reasons(translation_m, rotation_deg, gap_s, view, config)
    priority = (
        "maximum_gap",
        "hard_translation_safety",
        "hard_rotation_safety",
        "view_pixel_motion",
        "view_overlap",
        "new_visible_area",
        "soft_translation_unobservable_safety",
        "soft_rotation_unobservable_safety",
    )
    return next((name for name in priority if name in reasons), None)


def select_adaptive_keyframes(
    records: list[dict[str, Any]],
    quality: dict[int, dict[str, Any]],
    camera: dict[str, Any],
    config: SelectionConfig,
) -> list[dict[str, Any]]:
    """Trigger intervals from view change, with metric pose/time safety fallbacks."""

    config.validate()
    _normalize_records(records)
    selected: list[dict[str, Any]] = []

    def append(
        record: dict[str, Any], reason: str, reasons: list[str], view: Any
    ) -> None:
        if selected:
            previous = selected[-1]
            translation_m = float(
                np.linalg.norm(
                    record_pose(record)[:3, 3] - record_pose(previous)[:3, 3]
                )
            )
            rotation_deg = rotation_distance_deg(
                record_pose(previous), record_pose(record)
            )
            gap_s = (record_time(record) - record_time(previous)) / 1.0e9
        else:
            translation_m = rotation_deg = gap_s = 0.0
        output = dict(record)
        output.update(
            {
                "idx": len(selected),
                "pose_row": len(selected),
                "selection_reason": reason,
                "selection_reasons": reasons,
                "all_selection_reasons": reasons,
                "translation_from_previous_selected_m": translation_m,
                "rotation_from_previous_selected_deg": rotation_deg,
                "gap_from_previous_selected_s": gap_s,
                "view_trigger_metrics": view,
            }
        )
        selected.append(output)

    append(records[0], "first", ["first"], None)
    for record in records[1:]:
        previous = selected[-1]
        translation_m = float(
            np.linalg.norm(record_pose(record)[:3, 3] - record_pose(previous)[:3, 3])
        )
        rotation_deg = rotation_distance_deg(record_pose(previous), record_pose(record))
        gap_s = (record_time(record) - record_time(previous)) / 1.0e9
        view = estimate_view_geometry(
            previous, record, quality[record_source_index(record)], camera
        )
        reasons = _adaptive_reasons(translation_m, rotation_deg, gap_s, view, config)
        if not reasons:
            continue
        reason = adaptive_selection_reason(
            translation_m=translation_m,
            rotation_deg=rotation_deg,
            gap_s=gap_s,
            view=view,
            config=config,
        )
        assert reason is not None
        append(record, reason, reasons, view)
    if record_source_index(selected[-1]) != record_source_index(records[-1]):
        append(records[-1], "last", ["last"], None)
    return selected


def normalize_trigger_count(
    records: list[dict[str, Any]],
    triggers: list[dict[str, Any]],
    quality: dict[int, dict[str, Any]],
    camera: dict[str, Any],
    config: SelectionConfig,
    target_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match a stable frame budget without reverting to metre-only placement."""

    if not 2 <= target_count <= len(records):
        raise ValueError("trigger target count is outside source cardinality")
    source_to_record = {record_source_index(record): record for record in records}
    ordered_sources = [record_source_index(record) for record in records]
    selected_sources = [record_source_index(trigger) for trigger in triggers]
    reasons = {
        record_source_index(trigger): list(
            trigger.get("all_selection_reasons", [trigger["selection_reason"]])
        )
        for trigger in triggers
    }
    locked = {
        "first",
        "last",
        "maximum_gap",
        "hard_translation_safety",
        "hard_rotation_safety",
    }
    removed_sources: list[int] = []
    maximum_discrete_gap_s = config.maximum_gap_s + float(
        np.median(
            [
                (record_time(right) - record_time(left)) / 1.0e9
                for left, right in zip(records, records[1:])
            ]
        )
    )
    while len(selected_sources) > target_count:
        removable = []
        for position in range(1, len(selected_sources) - 1):
            source = selected_sources[position]
            if locked.intersection(reasons[source]):
                continue
            previous_source = selected_sources[position - 1]
            next_source = selected_sources[position + 1]
            merged_gap_s = (
                record_time(source_to_record[next_source])
                - record_time(source_to_record[previous_source])
            ) / 1.0e9
            if merged_gap_s > maximum_discrete_gap_s:
                continue
            removable.append((float(quality[source]["score"]), position, source))
        if not removable:
            break
        _, position, source = min(removable)
        selected_sources.pop(position)
        reasons.pop(source)
        removed_sources.append(source)

    added_sources: list[int] = []
    source_position = {source: index for index, source in enumerate(ordered_sources)}
    while len(selected_sources) < target_count:
        interval_choices = []
        for interval in range(len(selected_sources) - 1):
            left_source = selected_sources[interval]
            right_source = selected_sources[interval + 1]
            left_position = source_position[left_source]
            right_position = source_position[right_source]
            candidates = ordered_sources[left_position + 1 : right_position]
            if not candidates:
                continue
            left_time = record_time(source_to_record[left_source])
            right_time = record_time(source_to_record[right_source])
            duration_s = (right_time - left_time) / 1.0e9
            best_candidate = None
            for source in candidates:
                current_time = record_time(source_to_record[source])
                balance = min(
                    current_time - left_time, right_time - current_time
                ) / max(1, right_time - left_time)
                view = estimate_view_geometry(
                    source_to_record[left_source],
                    source_to_record[source],
                    quality[source],
                    camera,
                )
                novelty = min(
                    1.0,
                    float(view["median_pixel_motion_px"])
                    / max(1.0, config.minimum_median_pixel_motion_px),
                )
                priority = (
                    0.65 * float(quality[source]["score"])
                    + 0.20 * balance
                    + 0.15 * novelty
                )
                candidate = (priority, source, view)
                if best_candidate is None or candidate[0] > best_candidate[0]:
                    best_candidate = candidate
            if best_candidate is not None:
                interval_choices.append((duration_s, *best_candidate, interval))
        if not interval_choices:
            break
        _, _, source, view, interval = max(
            interval_choices, key=lambda item: (item[0], item[1])
        )
        selected_sources.insert(interval + 1, source)
        reasons[source] = ["target_count_quality_fill"]
        added_sources.append(source)

    normalized: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for source in selected_sources:
        record = dict(source_to_record[source])
        if previous is None:
            translation_m = rotation_deg = gap_s = 0.0
            view = None
        else:
            translation_m = float(
                np.linalg.norm(
                    record_pose(record)[:3, 3] - record_pose(previous)[:3, 3]
                )
            )
            rotation_deg = rotation_distance_deg(
                record_pose(previous), record_pose(record)
            )
            gap_s = (record_time(record) - record_time(previous)) / 1.0e9
            view = estimate_view_geometry(previous, record, quality[source], camera)
        record.update(
            {
                "idx": len(normalized),
                "pose_row": len(normalized),
                "selection_reason": reasons[source][0],
                "selection_reasons": reasons[source],
                "all_selection_reasons": reasons[source],
                "translation_from_previous_selected_m": translation_m,
                "rotation_from_previous_selected_deg": rotation_deg,
                "gap_from_previous_selected_s": gap_s,
                "view_trigger_metrics": view,
            }
        )
        normalized.append(record)
        previous = record
    return normalized, {
        "enabled": True,
        "target_count": target_count,
        "input_adaptive_count": len(triggers),
        "output_count": len(normalized),
        "target_met": len(normalized) == target_count,
        "added_count": len(added_sources),
        "removed_count": len(removed_sources),
        "added_source_indices": added_sources,
        "removed_source_indices": removed_sources,
    }


def initial_frame_roles(
    quality: dict[str, Any], view: dict[str, Any] | None
) -> tuple[dict[str, bool], dict[str, Any]]:
    components = quality.get("components", {})
    sharpness = float(quality.get("sharpness", 0.0))
    texture = float(quality.get("texture", 0.0))
    stereo_ratio = float(quality.get("stereo_matchable_ratio", 0.0))
    clipped_ratio = float(quality.get("clipped_ratio", 1.0))
    imu_omega = float(quality.get("imu_angular_velocity_rad_s", float("inf")))
    pose = float(components.get("pose", 0.0))
    bracket = float(components.get("pose_bracket", 0.0))
    keypoints = int(quality.get("stereo_keypoints_min", 0))
    overlap = float(view.get("view_overlap_ratio", 0.75)) if view else 0.75
    pixel_motion = float(view.get("median_pixel_motion_px", 6.0)) if view else 6.0
    failed = {
        "use_for_depth": [
            name
            for name, passed in (
                ("sharpness", sharpness >= 800.0),
                ("stereo_matchability", stereo_ratio >= 0.38),
                ("exposure_clipping", clipped_ratio <= 0.20),
                ("imu_angular_velocity", imu_omega <= 2.0),
            )
            if not passed
        ],
        "use_for_odometry": [
            name
            for name, passed in (
                ("sharpness", sharpness >= 800.0),
                ("texture", texture >= 55.0),
                ("stereo_matchability", stereo_ratio >= 0.38),
                ("stereo_keypoints", keypoints >= 30),
                ("pose_covariance", pose >= 0.35),
                ("pose_bracket", bracket >= 0.20),
                ("imu_angular_velocity", imu_omega <= 2.0),
                ("view_overlap", 0.35 <= overlap <= 1.0),
                ("pixel_motion", 1.5 <= pixel_motion <= 120.0),
            )
            if not passed
        ],
        "use_for_loop": [
            name
            for name, passed in (
                ("sharpness", sharpness >= 900.0),
                ("texture", texture >= 60.0),
                ("exposure_clipping", clipped_ratio <= 0.20),
                ("visual_distinctiveness", keypoints >= 80),
            )
            if not passed
        ],
        "use_for_tsdf": [
            name
            for name, passed in (
                ("sharpness", sharpness >= 800.0),
                ("stereo_matchability", stereo_ratio >= 0.38),
                ("exposure_clipping", clipped_ratio <= 0.20),
            )
            if not passed
        ],
        "use_for_semantics": [
            name
            for name, passed in (
                ("sharpness", sharpness >= 700.0),
                ("texture", texture >= 45.0),
                ("exposure_clipping", clipped_ratio <= 0.20),
                ("imu_angular_velocity", imu_omega <= 2.5),
            )
            if not passed
        ],
    }
    roles = {name: not reasons for name, reasons in failed.items()}
    return roles, {
        "schema": "daaam.keyframe_roles.v1",
        "decision_stage": "S04_provisional",
        "failed_quality_checks": failed,
        "pending_downstream_checks": {
            "use_for_odometry": ["rgbd_overlap", "3d_inlier_ratio", "disparity_range"],
            "use_for_loop": ["revisit_distinctiveness"],
            "use_for_tsdf": [
                "left_right_consistency",
                "temporal_depth_consistency",
                "static_valid_ratio",
            ],
            "use_for_semantics": [
                "object_visible_area",
                "semantic_observation_complementarity",
            ],
        },
    }


def _pose_uncertainty(record: dict[str, Any]) -> tuple[float, float]:
    covariance = np.asarray(record.get("pose_covariance"), dtype=np.float64)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        return float("inf"), float("inf")
    diagonal = np.maximum(np.diag(covariance), 0.0)
    return (
        float(np.sqrt(np.max(diagonal[:3]))),
        float(np.degrees(np.sqrt(np.max(diagonal[3:])))),
    )


def fixed_rate_candidate_rejections(
    record: dict[str, Any],
    quality: dict[str, Any] | None,
    config: FixedRateSelectionConfig,
) -> tuple[list[str], dict[str, bool], dict[str, Any]]:
    """Return auditable hard-gate failures for a formal map-tick candidate."""

    if quality is None:
        return ["missing_quality_score"], {}, {}
    rejections: list[str] = []
    score = float(quality.get("score", float("nan")))
    stereo = float(quality.get("stereo_matchable_ratio", float("nan")))
    stereo_match_count = float(quality.get("stereo_match_count", float("nan")))
    if not math.isfinite(score) or score < config.minimum_quality_score:
        rejections.append("image_quality_below_minimum")
    stereo_evidence_valid = (
        math.isfinite(stereo) and stereo >= config.minimum_stereo_matchability
    ) or (
        math.isfinite(stereo_match_count)
        and stereo_match_count >= config.minimum_stereo_match_count
    )
    if not stereo_evidence_valid:
        rejections.append("stereo_matchability_below_minimum")
    if quality.get("pose_interpolation_bracket_valid") is not True:
        rejections.append("pose_bracket_invalid")
    position_std_m, rotation_std_deg = _pose_uncertainty(record)
    if position_std_m > config.maximum_position_std_m:
        rejections.append("position_uncertainty_above_maximum")
    if rotation_std_deg > config.maximum_rotation_std_deg:
        rejections.append("rotation_uncertainty_above_maximum")
    advisory_roles, advisory_decisions = initial_frame_roles(quality, None)
    content_bearing = not rejections
    roles = {
        "use_for_depth": content_bearing,
        "use_for_odometry": advisory_roles["use_for_odometry"],
        "use_for_loop": advisory_roles["use_for_loop"],
        "use_for_tsdf": content_bearing,
        "use_for_semantics": content_bearing,
    }
    role_decisions = {
        "schema": "daaam.keyframe_roles.v1",
        "decision_stage": "fixed_rate_hard_gate",
        "hard_gate_rejections": list(rejections),
        "thresholds": {
            "minimum_quality_score": config.minimum_quality_score,
            "minimum_stereo_matchability": config.minimum_stereo_matchability,
            "minimum_stereo_match_count": config.minimum_stereo_match_count,
            "maximum_position_std_m": config.maximum_position_std_m,
            "maximum_rotation_std_deg": config.maximum_rotation_std_deg,
        },
        "legacy_role_advisory": advisory_decisions,
        "pending_downstream_checks": {
            "use_for_tsdf": [
                "dense_depth_validity",
                "left_right_consistency",
                "temporal_depth_consistency",
                "static_valid_ratio",
            ],
            "use_for_semantics": ["nonzero_label_evidence"],
        },
    }
    return rejections, roles, role_decisions


def fixed_rate_content_reasons(
    previous_record: dict[str, Any] | None,
    current_record: dict[str, Any],
    quality: dict[str, Any],
    camera: dict[str, Any] | None,
    selection_config: SelectionConfig,
    fixed_config: FixedRateSelectionConfig,
    *,
    cumulative_change_score: float,
    semantic_event: bool = False,
) -> list[str]:
    """Return only evidence that scene/map content changed.

    Elapsed time is intentionally not evidence.  This is the key distinction
    between a 1 Hz upper bound and artificial 1 Hz padding of a static scene.
    """

    if previous_record is None:
        return ["first_frame", *(["user_definition"] if semantic_event else [])]
    translation_m = float(
        np.linalg.norm(
            record_pose(current_record)[:3, 3] - record_pose(previous_record)[:3, 3]
        )
    )
    rotation_deg = rotation_distance_deg(
        record_pose(previous_record), record_pose(current_record)
    )
    gap_s = (record_time(current_record) - record_time(previous_record)) / 1.0e9
    reasons: list[str] = []
    if semantic_event:
        reasons.append("user_definition")
    if translation_m >= selection_config.hard_translation_m:
        reasons.append("hard_translation")
    elif gap_s >= selection_config.minimum_interval_s and (
        translation_m >= selection_config.soft_translation_m
    ):
        reasons.append("soft_translation")
    if rotation_deg >= selection_config.hard_rotation_deg:
        reasons.append("hard_rotation")
    elif gap_s >= selection_config.minimum_interval_s and (
        rotation_deg >= selection_config.soft_rotation_deg
    ):
        reasons.append("soft_rotation")
    if camera is not None:
        view = estimate_view_geometry(previous_record, current_record, quality, camera)
        if (
            float(view["median_pixel_motion_px"])
            >= selection_config.minimum_median_pixel_motion_px
        ):
            reasons.append("view_pixel_motion")
        if (
            float(view["depth_support_ratio"])
            >= selection_config.minimum_depth_support_ratio
            and float(view["view_overlap_ratio"])
            <= selection_config.maximum_view_overlap_ratio
        ):
            reasons.append("view_overlap_change")
        if (
            float(view["new_visible_ratio"])
            >= selection_config.minimum_new_visible_ratio
        ):
            reasons.append("new_visible_area")
    temporal_change = float(quality.get("temporal_change_score", 0.0))
    if temporal_change >= fixed_config.minimum_temporal_change_score:
        reasons.append("visual_change")
    elif cumulative_change_score >= fixed_config.minimum_cumulative_change_score:
        reasons.append("cumulative_visual_change")
    return list(dict.fromkeys(reasons))


def select_fixed_rate_keyframes(
    records: list[dict[str, Any]],
    quality: dict[int, dict[str, Any]],
    camera: dict[str, Any] | None,
    selection_config: SelectionConfig,
    fixed_config: FixedRateSelectionConfig,
    semantic_event_intervals: tuple[tuple[int, int], ...] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select at most one quality-best changed frame per causal sensor window.

    The window tail bounds observation staleness. A quality-valid frame from
    the whole window is used only when the tail contains no valid candidate;
    a window with no quality-valid candidate is missed.  A valid but unchanged
    window is explicitly suppressed and does not enter the heavy pipeline.
    """

    selection_config.validate()
    fixed_config.validate()
    normalized = _normalize_records(records)
    period_ns = int(round(1.0e9 / fixed_config.map_rate_hz))
    tail_ns = int(round(fixed_config.candidate_tail_s * 1.0e9))
    anchor_ns = normalized[0][2]
    windows: dict[int, list[dict[str, Any]]] = {}
    for record, _, timestamp, _ in normalized:
        window_index = (timestamp - anchor_ns) // period_ns
        windows.setdefault(int(window_index), []).append(record)
    window_count = max(windows) + 1
    selected: list[dict[str, Any]] = []
    missed_windows: list[dict[str, Any]] = []
    suppressed_static_windows: list[dict[str, Any]] = []
    fallback_windows: list[int] = []
    rejected_reason_counts: dict[str, int] = {}
    candidate_ages_s: list[float] = []
    previous_record: dict[str, Any] | None = None

    for window_index in range(window_count):
        window_start_ns = anchor_ns + window_index * period_ns
        window_end_ns = window_start_ns + period_ns
        records_in_window = windows.get(window_index, [])
        hard_valid: list[tuple[dict[str, Any], dict[str, bool], dict[str, Any]]] = []
        rejection_by_candidate: dict[str, list[str]] = {}
        for record in records_in_window:
            source = record_source_index(record)
            rejections, roles, role_decisions = fixed_rate_candidate_rejections(
                record, quality.get(source), fixed_config
            )
            rejection_by_candidate[str(source)] = rejections
            for reason in rejections:
                rejected_reason_counts[reason] = (
                    rejected_reason_counts.get(reason, 0) + 1
                )
            if not rejections:
                hard_valid.append((record, roles, role_decisions))
        recent = [
            item
            for item in hard_valid
            if record_time(item[0]) >= window_end_ns - tail_ns
        ]
        eligible = recent or hard_valid
        if not eligible:
            missed_windows.append(
                {
                    "map_window_index": window_index,
                    "window_start_ns": window_start_ns,
                    "window_end_ns": window_end_ns,
                    "source_frame_count": len(records_in_window),
                    "candidate_rejection_reasons": rejection_by_candidate,
                }
            )
            continue
        cumulative_change_score = sum(
            max(
                0.0,
                float(
                    quality.get(record_source_index(record), {}).get(
                        "temporal_change_score", 0.0
                    )
                ),
            )
            for record in records_in_window
        )
        content_eligible = [
            (
                record,
                roles,
                role_decisions,
                fixed_rate_content_reasons(
                    previous_record,
                    record,
                    quality[record_source_index(record)],
                    camera,
                    selection_config,
                    fixed_config,
                    cumulative_change_score=cumulative_change_score,
                    semantic_event=any(
                        start_ns <= record_time(record) <= end_ns
                        for start_ns, end_ns in semantic_event_intervals
                    ),
                ),
            )
            for record, roles, role_decisions in eligible
        ]
        content_eligible = [item for item in content_eligible if item[3]]
        if not content_eligible:
            suppressed_static_windows.append(
                {
                    "map_window_index": window_index,
                    "window_start_ns": window_start_ns,
                    "window_end_ns": window_end_ns,
                    "source_frame_count": len(records_in_window),
                    "reason": "static_scene_no_dispatch",
                    "cumulative_visual_change_score": cumulative_change_score,
                }
            )
            continue
        if not recent:
            fallback_windows.append(window_index)
        chosen_record, chosen_roles, chosen_role_decisions, content_reasons = max(
            content_eligible,
            key=lambda item: (
                float(quality[record_source_index(item[0])]["score"]),
                record_time(item[0]),
            ),
        )
        chosen_source = record_source_index(chosen_record)
        chosen_quality = quality[chosen_source]
        selected_view = (
            estimate_view_geometry(
                previous_record, chosen_record, chosen_quality, camera
            )
            if previous_record is not None and camera is not None
            else None
        )
        advisory_roles, _ = initial_frame_roles(chosen_quality, selected_view)
        roles = {
            **chosen_roles,
            "use_for_odometry": advisory_roles["use_for_odometry"],
            "use_for_loop": advisory_roles["use_for_loop"],
        }
        role_decisions = chosen_role_decisions
        if previous_record is None:
            translation_m = rotation_deg = gap_s = 0.0
        else:
            translation_m = float(
                np.linalg.norm(
                    record_pose(chosen_record)[:3, 3]
                    - record_pose(previous_record)[:3, 3]
                )
            )
            rotation_deg = rotation_distance_deg(
                record_pose(previous_record), record_pose(chosen_record)
            )
            gap_s = (record_time(chosen_record) - record_time(previous_record)) / 1.0e9
        candidate_age_s = max(0.0, (window_end_ns - record_time(chosen_record)) / 1.0e9)
        candidate_ages_s.append(candidate_age_s)
        all_reasons = ["content_rate_slot", "quality_best_in_window"]
        all_reasons.append(
            "recent_window_tail" if recent else "full_window_quality_fallback"
        )
        motion_reasons = content_reasons
        output = dict(chosen_record)
        output.update(
            {
                "idx": len(selected),
                "pose_row": len(selected),
                "source_idx": chosen_source,
                "sensor_time_ns": record_time(chosen_record),
                "selection_reason": (
                    "user_definition"
                    if "user_definition" in motion_reasons
                    else "content_change"
                ),
                "selection_reasons": [*all_reasons, *motion_reasons],
                "all_selection_reasons": [*all_reasons, *motion_reasons],
                "selected_motion_reasons": motion_reasons,
                "trigger_source_idx": chosen_source,
                "trigger_selection_reason": "content_rate_slot",
                "selected_source_idx": chosen_source,
                "quality_adjusted": False,
                "quality_score": float(chosen_quality["score"]),
                "quality_metrics": {
                    key: value
                    for key, value in chosen_quality.items()
                    if key not in {"source_idx", "score"}
                },
                "candidate_window": [
                    record_source_index(record) for record in records_in_window
                ],
                "quality_window_source_indices": [
                    record_source_index(record) for record, _, _ in eligible
                ],
                "candidate_rejection_reasons": rejection_by_candidate,
                "quality_fallback_reason": (
                    "no_valid_candidate_in_recent_window_tail" if not recent else None
                ),
                "view_selection_metrics": selected_view,
                "frame_roles": {"schema": "daaam.keyframe_roles.v1", **roles},
                "role_decisions": role_decisions,
                **roles,
                "translation_from_previous_selected_m": translation_m,
                "rotation_from_previous_selected_deg": rotation_deg,
                "gap_from_previous_selected_s": gap_s,
                "map_window_index": window_index,
                "map_window_start_ns": window_start_ns,
                "map_window_end_ns": window_end_ns,
                "candidate_age_to_window_end_s": candidate_age_s,
                "cumulative_visual_change_score": cumulative_change_score,
            }
        )
        selected.append(output)
        previous_record = chosen_record

    return selected, {
        "schema": "daaam.fixed_rate_keyframes.v1",
        "map_rate_hz": fixed_config.map_rate_hz,
        "period_s": 1.0 / fixed_config.map_rate_hz,
        "anchor_sensor_time_ns": anchor_ns,
        "window_count": window_count,
        "successful_window_count": len(selected),
        "missed_window_count": len(missed_windows),
        "missed_windows": missed_windows,
        "suppressed_static_window_count": len(suppressed_static_windows),
        "suppressed_static_windows": suppressed_static_windows,
        "candidate_tail_s": fixed_config.candidate_tail_s,
        "semantic_event_interval_count": len(semantic_event_intervals),
        "user_definition_keyframe_count": sum(
            item.get("selection_reason") == "user_definition" for item in selected
        ),
        "tail_fallback_window_count": len(fallback_windows),
        "tail_fallback_windows": fallback_windows,
        "candidate_age_p95_s": (
            float(np.percentile(candidate_ages_s, 95)) if candidate_ages_s else None
        ),
        "candidate_age_max_s": max(candidate_ages_s) if candidate_ages_s else None,
        "candidate_rejection_reason_counts": rejected_reason_counts,
    }


def refine_keyframes_by_quality(
    records: list[dict[str, Any]],
    triggers: list[dict[str, Any]],
    quality: dict[int, dict[str, Any]],
    config: SelectionConfig,
    radius_frames: int,
    camera: dict[str, Any] | None = None,
    minimum_quality_improvement: float = 0.0,
) -> list[dict[str, Any]]:
    """Replace each trigger by one quality-best frame in its ±N-frame interval."""

    if radius_frames < 0:
        raise ValueError("quality_window_radius_frames cannot be negative")
    if minimum_quality_improvement < 0.0:
        raise ValueError("minimum_quality_improvement cannot be negative")
    if radius_frames == 0:
        return triggers
    by_source = {record_source_index(record): record for record in records}
    ordered_sources = sorted(by_source)
    position_by_source = {source: index for index, source in enumerate(ordered_sources)}
    maximum_discrete_gap_s = config.maximum_gap_s
    refined: list[dict[str, Any]] = []
    previous_record: dict[str, Any] | None = None
    previous_source: int | None = None
    for trigger_index, trigger in enumerate(triggers):
        trigger_source = record_source_index(trigger)
        trigger_position = position_by_source[trigger_source]
        next_trigger_source = (
            record_source_index(triggers[trigger_index + 1])
            if trigger_index + 1 < len(triggers)
            else None
        )
        candidate_window = ordered_sources[
            max(0, trigger_position - radius_frames) : trigger_position
            + radius_frames
            + 1
        ]
        rejection_by_candidate: dict[str, list[str]] = {}
        eligible: list[tuple[float, int, float, float, float]] = []
        for source in candidate_window:
            candidate = by_source[source]
            rejections: list[str] = []
            if previous_source is not None and source <= previous_source:
                rejections.append("violates_strict_order")
            if next_trigger_source is not None and source >= next_trigger_source:
                rejections.append("collides_with_next_trigger_interval")
            if (
                str(trigger["selection_reason"]) in {"first", "last"}
                and source != trigger_source
            ):
                rejections.append("endpoint_fixed")
            if previous_record is None:
                translation_m = rotation_deg = gap_s = 0.0
            else:
                translation_m = float(
                    np.linalg.norm(
                        record_pose(candidate)[:3, 3]
                        - record_pose(previous_record)[:3, 3]
                    )
                )
                rotation_deg = rotation_distance_deg(
                    record_pose(previous_record), record_pose(candidate)
                )
                gap_s = (record_time(candidate) - record_time(previous_record)) / 1.0e9
                if gap_s > maximum_discrete_gap_s:
                    rejections.append("previous_gap_exceeds_maximum")
            if next_trigger_source is not None:
                next_gap_s = (
                    record_time(by_source[next_trigger_source]) - record_time(candidate)
                ) / 1.0e9
                if next_gap_s > maximum_discrete_gap_s:
                    rejections.append("next_trigger_gap_exceeds_maximum")
            if source not in quality:
                rejections.append("missing_quality_score")
            rejection_by_candidate[str(source)] = rejections
            if not rejections:
                eligible.append(
                    (
                        float(quality[source]["score"]),
                        source,
                        translation_m,
                        rotation_deg,
                        gap_s,
                    )
                )
        fallback_reason = None
        if eligible:
            chosen_tuple = max(
                eligible,
                key=lambda item: (item[0], -abs(item[1] - trigger_source)),
            )
            original_tuple = next(
                (item for item in eligible if item[1] == trigger_source), None
            )
            if (
                original_tuple is not None
                and chosen_tuple[1] != trigger_source
                and chosen_tuple[0] < original_tuple[0] + minimum_quality_improvement
            ):
                rejection_by_candidate[str(chosen_tuple[1])].append(
                    "quality_gain_below_minimum"
                )
                chosen_tuple = original_tuple
            _, chosen_source, translation_m, rotation_deg, gap_s = chosen_tuple
            for item in eligible:
                if (
                    item[1] != chosen_source
                    and not rejection_by_candidate[str(item[1])]
                ):
                    rejection_by_candidate[str(item[1])].append(
                        "lower_quality_score_than_selected"
                    )
        else:
            chosen_source = trigger_source
            chosen = by_source[chosen_source]
            if previous_record is None:
                translation_m = rotation_deg = gap_s = 0.0
            else:
                translation_m = float(
                    np.linalg.norm(
                        record_pose(chosen)[:3, 3] - record_pose(previous_record)[:3, 3]
                    )
                )
                rotation_deg = rotation_distance_deg(
                    record_pose(previous_record), record_pose(chosen)
                )
                gap_s = (record_time(chosen) - record_time(previous_record)) / 1.0e9
            fallback_reason = "no_interval_candidate_satisfied_order_and_gap_contract"
        chosen_record = dict(by_source[chosen_source])
        selected_view = (
            estimate_view_geometry(
                previous_record, chosen_record, quality[chosen_source], camera
            )
            if previous_record is not None and camera is not None
            else None
        )
        roles, role_decisions = initial_frame_roles(
            quality[chosen_source], selected_view
        )
        all_reasons = list(
            dict.fromkeys(
                [
                    *trigger.get(
                        "all_selection_reasons", [trigger["selection_reason"]]
                    ),
                    f"{trigger['selection_reason']}_interval_coverage",
                    "trigger_interval_coverage",
                ]
            )
        )
        chosen_record.update(
            {
                "idx": len(refined),
                "pose_row": len(refined),
                "source_idx": chosen_source,
                "sensor_time_ns": record_time(chosen_record),
                "selection_reason": trigger["selection_reason"],
                "selection_reasons": all_reasons,
                "all_selection_reasons": all_reasons,
                "selected_motion_reasons": _all_metric_reasons(
                    translation_m, rotation_deg, gap_s, config
                ),
                "trigger_source_idx": trigger_source,
                "trigger_selection_reason": trigger["selection_reason"],
                "selected_source_idx": chosen_source,
                "quality_adjusted": chosen_source != trigger_source,
                "quality_score": float(quality[chosen_source]["score"]),
                "quality_metrics": {
                    key: value
                    for key, value in quality[chosen_source].items()
                    if key not in {"source_idx", "score"}
                },
                "candidate_window": candidate_window,
                "quality_window_source_indices": candidate_window,
                "candidate_rejection_reasons": rejection_by_candidate,
                "quality_fallback_reason": fallback_reason,
                "view_selection_metrics": selected_view,
                "frame_roles": {"schema": "daaam.keyframe_roles.v1", **roles},
                "role_decisions": role_decisions,
                **roles,
                "translation_from_previous_selected_m": translation_m,
                "rotation_from_previous_selected_deg": rotation_deg,
                "gap_from_previous_selected_s": gap_s,
            }
        )
        refined.append(chosen_record)
        previous_record = chosen_record
        previous_source = chosen_source
    if len(refined) != len(triggers):
        raise RuntimeError("quality refinement changed trigger interval cardinality")
    return refined


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def prepare_output(path: Path, overwrite: bool) -> None:
    expected = {
        "selected_frames.jsonl",
        "selected_source_indices.txt",
        "selection_report.json",
        "quality_candidates.jsonl",
    }
    if path.exists():
        entries = {item.name for item in path.iterdir()}
        if entries - expected or (entries and not overwrite):
            raise FileExistsError(
                f"refusing to replace existing/unexpected output: {path}"
            )
        for name in entries:
            (path / name).unlink()
    else:
        path.mkdir(parents=True)


def main() -> None:
    args = parse_args()
    config = SelectionConfig(
        soft_translation_m=args.soft_translation_m,
        soft_rotation_deg=args.soft_rotation_deg,
        hard_translation_m=args.hard_translation_m,
        hard_rotation_deg=args.hard_rotation_deg,
        minimum_interval_s=args.minimum_interval_s,
        maximum_gap_s=args.maximum_gap_s,
        minimum_median_pixel_motion_px=args.minimum_median_pixel_motion_px,
        maximum_view_overlap_ratio=args.maximum_view_overlap_ratio,
        minimum_new_visible_ratio=args.minimum_new_visible_ratio,
        minimum_depth_support_ratio=args.minimum_depth_support_ratio,
    )
    records = load_jsonl(args.camera_poses.resolve())
    quality_enabled = args.quality_window_radius_frames > 0
    quality: dict[int, dict[str, Any]] = {}
    camera = None
    fixed_rate_selection = None
    semantic_event_intervals = (
        ()
        if args.prior_bundle is None
        else load_prior_event_intervals(
            args.prior_bundle.expanduser().resolve(),
            pre_roll_s=args.prior_pre_roll_s,
            post_roll_s=args.prior_post_roll_s,
        )
    )
    if quality_enabled:
        required = (
            args.quality_left_video,
            args.quality_right_video,
            args.quality_calibration,
            args.imu_csv,
        )
        if any(value is None for value in required):
            raise ValueError(
                "quality selection requires both videos, calibration, and IMU CSV"
            )
        camera = load_analysis_camera(
            args.quality_calibration.resolve(), args.quality_analysis_width
        )
        all_sources = {record_source_index(record) for record in records}
        quality = score_quality_candidates(
            records,
            all_sources,
            args.quality_left_video.resolve(),
            args.quality_right_video.resolve(),
            args.imu_csv.resolve(),
            args.quality_analysis_width,
            camera,
        )
        if args.fixed_map_rate_hz is not None:
            selected, fixed_rate_selection = select_fixed_rate_keyframes(
                records,
                quality,
                camera,
                config,
                FixedRateSelectionConfig(
                    map_rate_hz=args.fixed_map_rate_hz,
                    candidate_tail_s=args.fixed_candidate_tail_s,
                    minimum_quality_score=args.fixed_minimum_quality_score,
                    minimum_stereo_matchability=(
                        args.fixed_minimum_stereo_matchability
                    ),
                    minimum_stereo_match_count=(args.fixed_minimum_stereo_match_count),
                    maximum_position_std_m=args.fixed_maximum_position_std_m,
                    maximum_rotation_std_deg=(args.fixed_maximum_rotation_std_deg),
                    minimum_temporal_change_score=(
                        args.fixed_minimum_temporal_change_score
                    ),
                    minimum_cumulative_change_score=(
                        args.fixed_minimum_cumulative_change_score
                    ),
                ),
                semantic_event_intervals=semantic_event_intervals,
            )
            if fixed_rate_selection["missed_window_count"]:
                raise RuntimeError(
                    "fixed-rate selector found no content-bearing candidate in "
                    f"{fixed_rate_selection['missed_window_count']} map windows"
                )
            triggers = selected
            legacy_count_normalization = {"enabled": False}
            trigger_mode = "fixed_rate_causal_quality_window"
        else:
            triggers = select_adaptive_keyframes(records, quality, camera, config)
            legacy_count_normalization = {"enabled": False}
            if args.preserve_legacy_trigger_count:
                legacy_target = len(select_keyframes(records, config))
                triggers, legacy_count_normalization = normalize_trigger_count(
                    records,
                    triggers,
                    quality,
                    camera,
                    config,
                    legacy_target,
                )
            selected = refine_keyframes_by_quality(
                records,
                triggers,
                quality,
                config,
                args.quality_window_radius_frames,
                camera,
                args.minimum_quality_improvement,
            )
            trigger_mode = "view_change_primary_pose_time_safety"
    else:
        triggers = select_keyframes(records, config)
        selected = triggers
        trigger_mode = "legacy_pose_thresholds"
        legacy_count_normalization = {"enabled": False}
    output = args.output.resolve()
    prepare_output(output, args.overwrite)
    atomic_write_text(
        output / "selected_frames.jsonl",
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in selected),
    )
    atomic_write_text(
        output / "selected_source_indices.txt",
        "".join(f"{record_source_index(record)}\n" for record in selected),
    )
    atomic_write_text(
        output / "quality_candidates.jsonl",
        "".join(
            json.dumps(quality[source], allow_nan=False) + "\n"
            for source in sorted(quality)
        ),
    )
    duration_s = (record_time(records[-1]) - record_time(records[0])) / 1.0e9
    reason_counts: dict[str, int] = {}
    for record in selected:
        reason = str(record["selection_reason"])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    report = {
        "schema": "daaam.ego_motion_quality_keyframes.v3",
        "camera_poses": str(args.camera_poses.resolve()),
        "config": asdict(config),
        "trigger_mode": trigger_mode,
        "source_frame_count": len(records),
        "trigger_frame_count": len(triggers),
        "selected_frame_count": len(selected),
        "cardinality_preserved_after_quality_selection": len(selected) == len(triggers),
        "legacy_count_normalization": legacy_count_normalization,
        "reduction_ratio": len(selected) / len(records),
        "source_duration_s": duration_s,
        "effective_rate_hz": (
            len(selected) / duration_s
            if fixed_rate_selection is not None and duration_s
            else (len(selected) - 1) / duration_s
            if duration_s
            else 0.0
        ),
        "interval_rate_hz": (
            (len(selected) - 1)
            / ((record_time(selected[-1]) - record_time(selected[0])) / 1.0e9)
            if len(selected) >= 2
            else 0.0
        ),
        "fixed_rate_selection": fixed_rate_selection,
        "selection_reasons": reason_counts,
        "frame_roles": {
            name: sum(bool(record.get(f"use_for_{name}")) for record in selected)
            for name in ("depth", "odometry", "loop", "tsdf", "semantics")
        },
        "quality_selection": {
            "enabled": quality_enabled,
            "left_video": str(args.quality_left_video.resolve())
            if quality_enabled
            else None,
            "right_video": str(args.quality_right_video.resolve())
            if quality_enabled
            else None,
            "calibration": str(args.quality_calibration.resolve())
            if quality_enabled
            else None,
            "imu_csv": str(args.imu_csv.resolve()) if quality_enabled else None,
            "window_radius_frames": args.quality_window_radius_frames,
            "analysis_width": args.quality_analysis_width,
            "minimum_quality_improvement": args.minimum_quality_improvement,
            "candidate_count": len(quality),
            "adjusted_frame_count": sum(
                bool(record.get("quality_adjusted")) for record in selected
            ),
            "fallback_count": sum(
                record.get("quality_fallback_reason") is not None for record in selected
            ),
            "mean_selected_score": (
                float(np.mean([record["quality_score"] for record in selected]))
                if quality_enabled
                else None
            ),
            "mean_trigger_score": (
                float(
                    np.mean(
                        [
                            quality[record_source_index(trigger)]["score"]
                            for trigger in triggers
                        ]
                    )
                )
                if quality_enabled
                else None
            ),
        },
        "first_source_idx": record_source_index(selected[0]),
        "last_source_idx": record_source_index(selected[-1]),
    }
    atomic_write_text(
        output / "selection_report.json",
        json.dumps(report, indent=2, allow_nan=False) + "\n",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
