from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import json

import cv2
import numpy as np

from spatial_memory.geometry_compiler import static_check
from spatial_memory.stereo import FastStereoDepth, StereoSequence


class GeometryGenerationTests(unittest.TestCase):
    def test_support_hint_uses_calibrated_preceding_imu_and_rejects_large_acceleration(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calibrations").mkdir()
            (root / "imu").mkdir()
            calibration = {"cam0": {"timeshift_cam_imu": .1, "T_cam_imu": np.eye(4).tolist()}}
            (root / "calibrations/camera_imu-camchain-imucam.yaml").write_text(json.dumps(calibration))
            sequence = StereoSequence.__new__(StereoSequence)
            sequence.session, sequence.left_times, sequence.R0 = root, {1: 10_000_000_000}, np.eye(3)
            for value, expected in ((9.81, True), (16., False)):
                header = "timestamp,linear_acceleration_x,linear_acceleration_y,linear_acceleration_z\n"
                samples = "".join(f"{10.01+i*.004},0,0,{value}\n" for i in range(20))
                (root / "imu/imu0.csv").write_text(header + samples)
                hint = sequence.support_normal_hint(1)
                self.assertEqual(hint is not None, expected)
                if hint:
                    np.testing.assert_allclose(hint["normal"], [0, 0, 1])

    def test_inference_disparity_is_scaled_to_scoring_camera(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raster = np.zeros((960, 1280, 3), np.uint8)
            left, right = root / "left.png", root / "right.png"
            cv2.imwrite(str(left), raster)
            cv2.imwrite(str(right), raster)
            engine = FastStereoDepth.__new__(FastStereoDepth)
            engine.checkpoint_sha256 = "synthetic-test-no-model-inference"
            engine._run = lambda a, b: np.full((960, 1280), 40., np.float32)
            engine.compute({"left_hires": str(left), "right_hires": str(right), "source_frame_id": 77},
                           np.array([[280., 0, 320.], [0, 280., 240.], [0, 0, 1.]]), .1, root / "depth.npz")
            with np.load(root / "depth.npz") as result:
                # 40 px in 1280 raster = 20 px at scoring fx=280: z=280*0.1/20.
                self.assertAlmostEqual(float(np.median(result["depth"])), 1.4, places=5)
                self.assertFalse(result["valid"][:, :20].any())
                self.assertTrue(result["valid"][:, 20:].all())
                self.assertEqual(int(result["source_frame_id"]), 77)

    def test_static_depth_conflict_prevents_pose_acceptance(self):
        K = np.array([[100., 0, 50.], [0, 100., 50.], [0, 0, 1.]])
        x, y = np.meshgrid(np.arange(-.2, .21, .02), np.arange(-.2, .21, .02))
        points = np.column_stack((x.ravel(), y.ravel(), np.ones(x.size)))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.npz"
            dynamic = np.zeros((100, 100), bool)
            for depth_value, accepted in ((1.0, True), (.9, False)):
                np.savez(path, K=K, T=np.eye(4), pose_valid=True,
                         depth=np.full((100, 100), depth_value), valid=np.ones((100, 100), bool))
                self.assertEqual(static_check(points, path, dynamic)["accepted"], accepted)


if __name__ == "__main__":
    unittest.main()
