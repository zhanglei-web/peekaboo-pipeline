#!/usr/bin/env python3
"""Persistent Fast-FoundationStereo JSON-lines worker for realtime mapping."""

from __future__ import annotations
from pathlib import Path
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / 'repository.marker').is_file())

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

from daaam.depth.confidence import (  # noqa: E402
    compute_left_right_confidence,
    disparity_to_metric_depth,
)
from daaam.realtime.gpu import SharedGpuCoordinator  # noqa: E402
from run_fast_foundation_stereo_depth import (  # noqa: E402
    load_model,
    prepare_pair,
    timed_inference,
    verify_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--max-disp", type=int, default=192)
    parser.add_argument(
        "--volume-builder", choices=("triton", "pytorch1"), default="triton"
    )
    parser.add_argument(
        "--confidence-mode", choices=("left-right", "validity"), default="left-right"
    )
    parser.add_argument("--lr-absolute-tolerance-px", type=float, default=0.75)
    parser.add_argument("--lr-relative-tolerance", type=float, default=0.03)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--gpu-lock-path", type=Path)
    parser.add_argument("--gpu-activity-path", type=Path)
    args = parser.parse_args(argv)
    if min(args.width, args.height, args.iters, args.max_disp) <= 0:
        parser.error("width, height, iters and max-disp must be positive")
    if args.warmup < 0:
        parser.error("warmup cannot be negative")
    return args


def _atomic_imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary), image):
        raise IOError(f"failed to write {temporary}")
    temporary.replace(path)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class FastInferenceWorker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo.expanduser().resolve()
        self.checkpoint = args.checkpoint.expanduser().resolve()
        self.gpu = SharedGpuCoordinator(
            lock_path=args.gpu_lock_path,
            activity_path=args.gpu_activity_path,
        )
        self.artifacts = verify_artifacts(self.repo, self.checkpoint)
        with self.gpu.lease():
            self.model, self.model_info = load_model(
                self.repo,
                self.checkpoint,
                args.max_disp,
                args.iters,
            )
        from core.utils.utils import InputPadder

        self.InputPadder = InputPadder
        self.warmup_seconds = self._warmup(args.warmup)

    def _infer_tensors(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        confidence_mode: str,
        iterations: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float]]:
        inference_iterations = self.args.iters if iterations is None else int(iterations)
        if not 0 < inference_iterations <= self.args.iters:
            raise ValueError(
                f"request iterations must be in [1, {self.args.iters}]"
            )
        left_tensor, left_timing = timed_inference(
            self.model,
            left,
            right,
            inference_iterations,
            self.args.volume_builder,
        )
        right_tensor = None
        right_timing = {"wall_seconds": 0.0, "cuda_event_seconds": 0.0}
        if confidence_mode == "left-right":
            right_tensor, right_timing = timed_inference(
                self.model,
                torch.flip(right, dims=[3]),
                torch.flip(left, dims=[3]),
                inference_iterations,
                self.args.volume_builder,
            )
        return left_tensor, right_tensor, {
            "left_model_wall": left_timing["wall_seconds"],
            "left_model_cuda_event": left_timing["cuda_event_seconds"],
            "right_model_wall": (
                right_timing["wall_seconds"] if right_tensor is not None else None
            ),
            "right_model_cuda_event": (
                right_timing["cuda_event_seconds"] if right_tensor is not None else None
            ),
            "dual_model_wall": (
                left_timing["wall_seconds"] + right_timing["wall_seconds"]
            ),
            "dual_model_cuda_event": (
                left_timing["cuda_event_seconds"]
                + right_timing["cuda_event_seconds"]
            ),
        }

    def _warmup(self, iterations: int) -> float:
        if iterations == 0:
            return 0.0
        dummy = np.zeros((self.args.height, self.args.width, 3), dtype=np.uint8)
        left, right, _ = prepare_pair(dummy, dummy, self.InputPadder)
        started = time.perf_counter()
        with self.gpu.lease():
            for _ in range(iterations):
                self._infer_tensors(left, right, confidence_mode="validity")
        return time.perf_counter() - started

    def infer(self, request: dict) -> dict:
        started = time.perf_counter()
        self.gpu.touch_activity()
        confidence_mode = str(request.get("confidence_mode", self.args.confidence_mode))
        if confidence_mode not in {"left-right", "validity"}:
            raise ValueError(f"unsupported confidence mode: {confidence_mode}")
        iterations = int(request.get("iterations", self.args.iters))
        if not 0 < iterations <= self.args.iters:
            raise ValueError(
                f"request iterations must be in [1, {self.args.iters}]"
            )
        left_bgr = cv2.imread(str(request["left_path"]), cv2.IMREAD_COLOR)
        right_bgr = cv2.imread(str(request["right_path"]), cv2.IMREAD_COLOR)
        expected_shape = (self.args.height, self.args.width, 3)
        if (
            left_bgr is None
            or right_bgr is None
            or left_bgr.shape != expected_shape
            or right_bgr.shape != expected_shape
        ):
            raise ValueError(
                f"Fast-FoundationStereo request must contain {expected_shape} images"
            )
        preprocess_started = time.perf_counter()
        left, right, padder = prepare_pair(left_bgr, right_bgr, self.InputPadder)
        preprocess_seconds = time.perf_counter() - preprocess_started
        with self.gpu.lease():
            left_tensor, right_tensor, timing = self._infer_tensors(
                left,
                right,
                confidence_mode=confidence_mode,
                iterations=iterations,
            )
        postprocess_started = time.perf_counter()
        left_disparity = (
            padder.unpad(left_tensor.float()).cpu().numpy().squeeze()
        )
        left_disparity = np.clip(left_disparity, 0.0, None).astype(
            np.float32, copy=False
        )
        if right_tensor is not None:
            right_disparity = (
                padder.unpad(torch.flip(right_tensor, dims=[3]).float())
                .cpu()
                .numpy()
                .squeeze()
            )
            right_disparity = np.clip(right_disparity, 0.0, None).astype(
                np.float32, copy=False
            )
            stereo = compute_left_right_confidence(
                left_disparity,
                right_disparity,
                absolute_tolerance_px=self.args.lr_absolute_tolerance_px,
                relative_tolerance=self.args.lr_relative_tolerance,
            )
            valid = stereo.consistent_mask
            confidence = stereo.confidence
            occlusion = stereo.occlusion_mask
            consistency = stereo.metrics["left_right_consistency"]
        else:
            valid = np.isfinite(left_disparity) & (left_disparity > 0.0)
            confidence = valid.astype(np.float32)
            occlusion = np.zeros_like(valid)
            consistency = None
        depth = disparity_to_metric_depth(
            left_disparity,
            focal_length_px=float(request["fx"]),
            baseline_m=float(request["baseline_m"]),
            maximum_depth_m=float(request["maximum_depth_m"]),
            valid_mask=valid,
        )
        output_dir = Path(request["output_dir"])
        frame_index = int(request["frame_idx"])
        frame_name = f"{frame_index:08d}.png"
        paths = {
            "depth_path": output_dir / "depth" / frame_name,
            "confidence_path": output_dir / "depth_confidence" / frame_name,
            "consistency_path": output_dir / "depth_consistency" / frame_name,
            "occlusion_path": output_dir / "depth_occlusion" / frame_name,
            "metadata_path": output_dir
            / "depth_metadata"
            / f"{frame_index:08d}.json",
        }
        products = {
            "depth_path": np.rint(depth * 1000.0).astype(np.uint16),
            "confidence_path": np.rint(np.clip(confidence, 0.0, 1.0) * 255.0).astype(
                np.uint8
            ),
            "consistency_path": valid.astype(np.uint8) * 255,
            "occlusion_path": occlusion.astype(np.uint8) * 255,
        }
        for name, product in products.items():
            _atomic_imwrite(paths[name], product)
        valid_depth = depth > 0.0
        postprocess_seconds = time.perf_counter() - postprocess_started
        metadata = {
            "schema": "daaam.fast_foundation_stereo_worker_frame.v1",
            "frame_idx": frame_index,
            "sensor_time_ns": int(request["sensor_time_ns"]),
            "confidence_mode": confidence_mode,
            "left_right_verified": confidence_mode == "left-right",
            "valid_ratio": float(valid_depth.mean()),
            "left_right_consistency": consistency,
            "mean_confidence": (
                float(np.mean(confidence[valid_depth])) if np.any(valid_depth) else 0.0
            ),
            "raw_positive_disparity_ratio": float(
                np.mean(np.isfinite(left_disparity) & (left_disparity > 0.0))
            ),
            "settings": {
                "width": self.args.width,
                "height": self.args.height,
                "iterations": iterations,
                "maximum_disparity_px": self.args.max_disp,
                "volume_builder": self.args.volume_builder,
                "precision": "fp16",
                "adaptive_profile": request.get("adaptive_profile"),
            },
            "timing_seconds": {
                "preprocess_wall": preprocess_seconds,
                **timing,
                "postprocess_and_write_wall": postprocess_seconds,
                "end_to_end_wall": time.perf_counter() - started,
            },
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_worker_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
        }
        _atomic_json(paths["metadata_path"], metadata)
        return {
            "status": "ok",
            "request_id": request["request_id"],
            **metadata,
            **{name: str(path) for name, path in paths.items()},
        }

    def ready_message(self) -> dict:
        return {
            "status": "ready",
            "backend": "fast-foundation-stereo",
            "artifacts": self.artifacts,
            "model": self.model_info,
            "settings": {
                "width": self.args.width,
                "height": self.args.height,
                "iterations": self.args.iters,
                "maximum_disparity_px": self.args.max_disp,
                "volume_builder": self.args.volume_builder,
                "default_confidence_mode": self.args.confidence_mode,
                "warmup_iterations": self.args.warmup,
                "warmup_seconds": self.warmup_seconds,
            },
        }


def emit(message: dict) -> None:
    print(json.dumps(message, allow_nan=False), flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    worker = FastInferenceWorker(args)
    emit(worker.ready_message())
    for line in sys.stdin:
        request: dict = {}
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                emit({"status": "stopped"})
                break
            if request.get("command") != "infer":
                raise ValueError("unknown worker command")
            emit(worker.infer(request))
        except Exception as error:
            if "out of memory" in str(error).lower():
                torch.cuda.empty_cache()
            emit(
                {
                    "status": "error",
                    "request_id": request.get("request_id"),
                    "error": repr(error),
                }
            )


if __name__ == '__main__':
    import sys as _freeze_sys
    from pathlib import Path as _FreezePath
    _freeze_sys.path.insert(0, str(_FreezePath(__file__).resolve().parents[1] / 'src'))
    from daaam.memory.legacy_gate import require_legacy_opt_in
    require_legacy_opt_in('fast_foundation_stereo_worker.py')
if __name__ == "__main__":
    main()
