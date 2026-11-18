"""Timestamp-paired calibrated stereo and Fast-FoundationStereo depth (3.md)."""
from decimal import Decimal
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

from .io import sha256, write_json


def timestamps(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        index, stamp = line.split()[:2]
        result[int(index)] = int(Decimal(stamp) * 1_000_000_000)
    if not result or any(b <= a for a, b in zip(result.values(), list(result.values())[1:])):
        raise ValueError("Invalid acquisition timestamps")
    return result


def ds_project(points, intrinsics):
    xi, alpha, fx, fy, cx, cy = intrinsics
    z = points[..., 2] + xi * np.linalg.norm(points, axis=-1)
    length = np.sqrt(points[..., 0] ** 2 + points[..., 1] ** 2 + z ** 2)
    denominator = alpha * length + (1 - alpha) * z
    return np.stack((fx * points[..., 0] / denominator + cx,
                     fy * points[..., 1] / denominator + cy), axis=-1)


class StereoSequence:
    def __init__(self, session):
        self.session = Path(session)
        self.calibration_file = self.session / "calibrations/camera-camchain.yaml"
        calibration = yaml.safe_load(self.calibration_file.read_text())
        if any(calibration[c]["camera_model"] != "ds" for c in ("cam0", "cam1")):
            raise ValueError("This rectifier requires calibrated double-sphere cameras")
        transform = np.asarray(calibration["cam1"]["T_cn_cnm1"], dtype=float)
        K = np.array([[280., 0., 319.5], [0., 280., 239.5], [0., 0., 1.]])
        R0, R1, P0, P1, *_ = cv2.stereoRectify(
            K, np.zeros(5), K, np.zeros(5), (640, 480), transform[:3, :3],
            transform[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
        self.K = P0[:3, :3]
        self.R0 = R0
        self.baseline_m = abs(float(P1[0, 3] / P1[0, 0]))
        self.maps = []
        yy, xx = np.indices((960, 1280))
        # Pixel-center preserving 2x inference raster, same scoring camera.
        for name, R in (("cam0", R0), ("cam1", R1)):
            rays = np.stack((((xx + .5) / 2 - .5 - self.K[0, 2]) / self.K[0, 0],
                             ((yy + .5) / 2 - .5 - self.K[1, 2]) / self.K[1, 1],
                             np.ones_like(xx)), axis=-1) @ R
            uv = ds_project(rays, calibration[name]["intrinsics"]).astype("float32")
            self.maps.append((uv[..., 0], uv[..., 1]))
        self.left_times = timestamps(self.session / "timestamps/left_timestamps.txt")
        self.right_times = timestamps(self.session / "timestamps/right_timestamps.txt")

    def prepare(self, selection, output_dir, tolerance_ns=2_000_000):
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        requested = {selection[role] for role in ("pre", "pnr", "post")}
        if len(requested) != 3:
            raise ValueError("Need three distinct selected source frames")
        right_ids = np.array(list(self.right_times))
        right_ns = np.array(list(self.right_times.values()), dtype=np.int64)
        pairs = {}
        for fid in requested:
            when = self.left_times[fid]
            index = int(np.argmin(np.abs(right_ns - when)))
            if abs(int(right_ns[index]) - when) > tolerance_ns:
                raise ValueError(f"Selected source frame {fid} has no synchronized right image")
            pairs[fid] = int(right_ids[index])
        def decode(camera, ids):
            cap = cv2.VideoCapture(str(self.session / f"videos/{camera}.mkv"))
            found = {}
            try:
                for fid in range(max(ids) + 1):
                    if not cap.grab():
                        break
                    if fid in ids:
                        ok, frame = cap.retrieve()
                        if ok:
                            found[fid] = frame
            finally:
                cap.release()
            if set(found) != set(ids):
                raise ValueError("Selected source image cannot be decoded")
            return found
        left = decode("left", requested)
        right = decode("right", set(pairs.values()))
        frames = {}
        for role in ("pre", "pnr", "post"):
            fid = selection[role]
            a = cv2.remap(left[fid], *self.maps[0], cv2.INTER_LINEAR)
            b = cv2.remap(right[pairs[fid]], *self.maps[1], cv2.INTER_LINEAR)
            paths = {}
            for name, value in (("left_hires", a), ("right_hires", b),
                                ("left", cv2.resize(a, (640, 480), interpolation=cv2.INTER_AREA))):
                path = output_dir / f"{role}_{name}.png"
                if not cv2.imwrite(str(path), value):
                    raise OSError(path)
                paths[name] = str(path)
            frames[role] = {**paths, "source_frame_id": fid, "right_source_frame_id": pairs[fid],
                            "timestamp_ns": self.left_times[fid],
                            "stereo_skew_ns": self.right_times[pairs[fid]] - self.left_times[fid]}
        write_json(output_dir / "stereo.json", {
            "calibration_sha256": sha256(self.calibration_file), "K": self.K.tolist(),
            "baseline_m": self.baseline_m, "frames": frames,
            "inference_size": [1280, 960], "scoring_size": [640, 480]})
        return frames

    def hand_exclusions(self, selection, detections_path, output):
        import json
        wanted = {selection[role] for role in ("pre", "pnr", "post")}
        rows = {}
        for line in Path(detections_path).read_text().splitlines():
            row = json.loads(line)
            if row["source_frame_id"] in wanted:
                rows[row["source_frame_id"]] = row
        masks = {}
        yy, xx = np.indices((480, 640))
        u, v = (2 * xx + .5).astype("float32"), (2 * yy + .5).astype("float32")
        mapx = cv2.remap(self.maps[0][0], u, v, cv2.INTER_LINEAR)
        mapy = cv2.remap(self.maps[0][1], u, v, cv2.INTER_LINEAR)
        for role in ("pre", "pnr", "post"):
            row = rows[selection[role]]
            if row["status"] != "ok":
                raise ValueError("Selected frame's hand detections are unavailable")
            mask = np.zeros((480, 640), dtype=bool)
            for hand in row["hands"]:
                if hand["score"] < .3:
                    continue
                x1, y1, x2, y2 = hand["bbox_xyxy"]
                mask |= (mapx >= x1) & (mapx <= x2) & (mapy >= y1) & (mapy <= y2)
            masks[role] = cv2.dilate(mask.astype("uint8"), np.ones((7, 7), np.uint8)) > 0
        write_json(output, {"detections_sha256": sha256(detections_path),
                            "selected_source_frame_ids": sorted(wanted),
                            "source": "100DOH_original_camera_boxes_rectified_with_calibration"})
        return masks

    def support_normal_hint(self, source_frame_id):
        """Use measured pre-frame IMU specific force only when locally stable."""
        import csv
        calibration_path = self.session / "calibrations/camera_imu-camchain-imucam.yaml"
        imu_path = self.session / "imu/imu0.csv"
        if not calibration_path.exists() or not imu_path.exists():
            return None
        calibration = yaml.safe_load(calibration_path.read_text())["cam0"]
        when = self.left_times[source_frame_id] / 1e9 + calibration["timeshift_cam_imu"]
        with imu_path.open() as stream:
            rows = [r for r in csv.DictReader(stream) if when - .1 <= float(r["timestamp"]) <= when]
        if len(rows) < 8:
            return None
        acceleration = np.array([[float(r["linear_acceleration_" + axis]) for axis in "xyz"] for r in rows])
        lengths = np.linalg.norm(acceleration, axis=1)
        mean = acceleration.mean(axis=0)
        norm = np.linalg.norm(mean)
        if not 8.0 <= norm <= 11.5 or np.std(lengths) > 1.0 or np.any(lengths < 1):
            return None
        direction = mean / norm
        spread = float(np.degrees(np.quantile(np.arccos(np.clip(acceleration / lengths[:, None] @ direction, -1, 1)), .9)))
        if spread > 8:
            return None
        normal = self.R0 @ np.array(calibration["T_cam_imu"])[:3, :3] @ direction
        normal /= np.linalg.norm(normal)
        return {"normal": normal.tolist(), "max_angle_deg": 25.,
                "source": "preceding_100ms_specific_force_with_calibrated_imu_to_rectified_camera",
                "sample_count": len(rows), "acceleration_norm_m_s2": float(norm),
                "direction_spread_p90_deg": spread, "calibration_sha256": sha256(calibration_path)}


class FastStereoDepth:
    def __init__(self, repository, checkpoint, device="cuda"):
        import torch
        sys.path.insert(0, str(Path(repository).resolve()))
        from core.utils.utils import InputPadder
        self.torch, self.Padder, self.device = torch, InputPadder, device
        torch.set_num_threads(2)
        self.model = torch.load(checkpoint, map_location="cpu", weights_only=False).to(device).eval()
        self.model.args.valid_iters = 4
        self.model.args.max_disp = 192
        self.checkpoint_sha256 = sha256(checkpoint)

    def _run(self, a, b):
        torch = self.torch
        x = torch.from_numpy(a.copy()).to(self.device).float().permute(2, 0, 1)[None]
        y = torch.from_numpy(b.copy()).to(self.device).float().permute(2, 0, 1)[None]
        pad = self.Padder(x.shape, divis_by=32, force_square=False)
        x, y = pad.pad(x, y)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                   enabled=str(self.device).startswith("cuda")):
            disparity = self.model.forward(x, y, iters=4, test_mode=True,
                                           optimize_build_volume="pytorch1")
        return pad.unpad(disparity.float()).cpu().numpy().squeeze()

    def compute(self, frame, K, baseline_m, output):
        start = time.perf_counter()
        a = cv2.cvtColor(cv2.imread(frame["left_hires"]), cv2.COLOR_BGR2RGB)
        b = cv2.cvtColor(cv2.imread(frame["right_hires"]), cv2.COLOR_BGR2RGB)
        dl = self._run(a, b)
        dr = self._run(b[:, ::-1], a[:, ::-1])[:, ::-1]
        yy, xx = np.indices((480, 640))
        u, v = (2 * xx + .5).astype("float32"), (2 * yy + .5).astype("float32")
        left = cv2.remap(dl, u, v, cv2.INTER_LINEAR)
        right = cv2.remap(dr, u - left, v, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=-999)
        residual = abs(left - right) / 2
        disparity = left / 2
        depth = np.divide(K[0, 0] * baseline_m, disparity,
                          out=np.zeros_like(disparity), where=disparity > 0)
        valid = ((disparity > 0) & (u - left >= 0) & np.isfinite(depth)
                 & (residual <= 1.5) & (depth > .15) & (depth < 4.0))
        np.savez_compressed(output, depth=depth, valid=valid, disparity=disparity,
                            lr_residual=residual, source_frame_id=frame["source_frame_id"])
        return {"model": "Fast-FoundationStereo", "checkpoint_sha256": self.checkpoint_sha256,
                "iterations": 4, "left_right_tolerance_scoring_px": 1.5,
                "valid_fraction": float(valid.mean()), "seconds": time.perf_counter() - start}
