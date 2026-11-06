"""Exact camera projection adapters; native Kalibr lengths here are metres."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path

import cv2
import numpy as np
import yaml

from .common import legacy, transform_units


def load_calibration(source):
    source = Path(source)
    chain = yaml.safe_load((source / 'calibrations/camera-camchain.yaml').read_text())
    cameras = yaml.safe_load((source / 'calibrations/camera_imu-camchain-imucam.yaml').read_text())
    imu = yaml.safe_load((source / 'calibrations/camera_imu-imu.yaml').read_text())['imu0']
    validate = legacy('run_ego_openvins').validate_homogeneous_transform
    for name in ('cam0', 'cam1'):
        c = cameras[name]
        for key in ('camera_model', 'intrinsics', 'distortion_model', 'distortion_coeffs', 'resolution'):
            if c[key] != chain[name][key]:
                raise ValueError(f'{name}.{key}: calibration files disagree')
        params = np.asarray(c['intrinsics'], float)
        if not np.isfinite(params).all() or len(c['resolution']) != 2 or min(c['resolution']) <= 0:
            raise ValueError('invalid camera calibration')
        if c['camera_model'] == 'ds':
            if (params.shape != (6,) or not 0 <= params[1] < 1 or min(params[2:4]) <= 0
                    or c['distortion_model'] != 'none' or c['distortion_coeffs']):
                raise ValueError('DS requires [xi,alpha,fx,fy,cx,cy] and no extra distortion')
        elif c['camera_model'] == 'pinhole':
            if params.shape != (4,) or min(params[:2]) <= 0 or c['distortion_model'] not in ('radtan', 'equidistant'):
                raise ValueError('unsupported pinhole calibration')
            if len(c['distortion_coeffs']) != 4:
                raise ValueError('expected four distortion parameters')
        else:
            raise ValueError(f'unsupported camera model {c["camera_model"]}')
        validate(c['T_cam_imu'], name)
    T = validate(cameras['cam1']['T_cn_cnm1'], 'T_cam1_cam0')
    if not np.allclose(T, chain['cam1']['T_cn_cnm1'], atol=1e-9, rtol=0):
        raise ValueError('stereo calibration files disagree')
    if not np.allclose(T @ cameras['cam0']['T_cam_imu'], cameras['cam1']['T_cam_imu'], atol=1e-8, rtol=0):
        raise ValueError('stereo and IMU extrinsics disagree')
    if cameras['cam0']['resolution'] != cameras['cam1']['resolution'] or np.linalg.norm(T[:3, 3]) <= 0:
        raise ValueError('invalid stereo pair')
    for field in ('accelerometer_noise_density', 'accelerometer_random_walk', 'gyroscope_noise_density',
                  'gyroscope_random_walk', 'update_rate'):
        if not np.isfinite(imu[field]) or imu[field] <= 0:
            raise ValueError(f'invalid IMU {field}')
    shift = int((Decimal(str(cameras['cam0']['timeshift_cam_imu'])) * 10**9)
                .to_integral_value(rounding=ROUND_HALF_EVEN))
    return {'cameras': cameras, 'imu': imu, 'right_T_left': T, 'time_shift_ns': shift}


def public_calibration(cal):
    return {'length_unit': 'mm', 'transform_convention': 'T_A_B maps B into A',
            'time_relation': 't_imu = t_cam + time_shift_ns', 'time_shift_ns': cal['time_shift_ns'],
            'offset_rounding': 'nearest integer ns, ties to even; image timestamps unchanged',
            'cameras': {name: {**{k: c[k] for k in ('camera_model', 'distortion_model', 'distortion_coeffs', 'intrinsics', 'resolution')},
                               'T_camera_imu_mm': transform_units(c['T_cam_imu'], 1000).tolist(),
                               'timeshift_cam_imu_s': str(c['timeshift_cam_imu'])}
                        for name, c in cal['cameras'].items() if name in ('cam0', 'cam1')},
            'T_right_left_mm': transform_units(cal['right_T_left'], 1000).tolist(),
            'baseline_mm': float(np.linalg.norm(cal['right_T_left'][:3, 3]) * 1000)}


def virtual_K(size, horizontal_fov_deg):
    width, height = map(int, size)
    if min(width, height) <= 0 or not 10 < horizontal_fov_deg < 170:
        raise ValueError('invalid virtual pinhole size/FOV')
    focal = width / (2 * np.tan(np.deg2rad(horizontal_fov_deg) / 2))
    return np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1.]])


def project(camera, rays):
    """Project native-camera rays; invalid/behind-model points become NaN."""
    rays = np.asarray(rays, dtype=float)
    shape = rays.shape[:-1]
    points = rays.reshape(-1, 3)
    if camera['camera_model'] == 'ds':
        xi, alpha, fx, fy, cx, cy = camera['intrinsics']
        x, y, z = points.T
        d1 = np.linalg.norm(points, axis=1)
        zxi = xi*d1 + z
        d2 = np.sqrt(x*x+y*y+zxi*zxi)
        denominator = alpha*d2+(1-alpha)*zxi
        w1 = alpha/(1-alpha) if alpha <= .5 else (1-alpha)/alpha
        w2 = (w1+xi)/np.sqrt(2*w1*xi+xi*xi+1)
        valid = (d1 > 0) & (z > -w2*d1) & (denominator > 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = np.stack([fx*x/denominator+cx, fy*y/denominator+cy], axis=-1)
        uv[~valid] = np.nan
    else:
        fx, fy, cx, cy = camera['intrinsics']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        D = np.asarray(camera['distortion_coeffs'], float)
        if camera['distortion_model'] == 'equidistant':
            uv, _ = cv2.fisheye.projectPoints(points.reshape(1, -1, 3), np.zeros(3), np.zeros(3), K, D)
        else:
            uv, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), K, D)
        uv = uv.reshape(-1, 2)
        uv[points[:, 2] <= 0] = np.nan
    return uv.reshape(*shape, 2)


def unproject(camera, pixels):
    """Convert native pixels to unit camera rays; pixels outside the model are NaN."""
    pixels = np.asarray(pixels, dtype=float)
    if pixels.shape[-1] != 2:
        raise ValueError('pixels must have a final dimension of 2')
    shape = pixels.shape[:-1]
    uv = pixels.reshape(-1, 2)
    if camera['camera_model'] == 'ds':
        xi, alpha, fx, fy, cx, cy = camera['intrinsics']
        mx, my = (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy
        radius2 = mx*mx + my*my
        radicand = 1 - (2*alpha - 1)*radius2
        valid = np.isfinite(uv).all(axis=1) & (radicand >= 0)
        root = np.sqrt(np.maximum(radicand, 0))
        denominator = alpha*root + 1-alpha
        mz = (1-alpha*alpha*radius2) / denominator
        root_ray = mz*mz + radius2
        k_radicand = mz*mz + (1-xi*xi)*radius2
        valid &= (denominator > 1e-12) & (root_ray > 1e-12) & (k_radicand >= 0)
        k = (mz*xi + np.sqrt(np.maximum(k_radicand, 0))) / np.maximum(root_ray, 1e-12)
        rays = np.stack([k*mx, k*my, k*mz-xi], axis=-1)
    else:
        fx, fy, cx, cy = camera['intrinsics']
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        D = np.asarray(camera['distortion_coeffs'], float)
        points = uv.reshape(-1, 1, 2)
        if camera['distortion_model'] == 'equidistant':
            rays = cv2.fisheye.undistortPoints(points, K, D).reshape(-1, 2)
        else:
            rays = cv2.undistortPoints(points, K, D).reshape(-1, 2)
        rays = np.column_stack([rays, np.ones(len(rays))])
        valid = np.isfinite(rays).all(axis=1)
    norms = np.linalg.norm(rays, axis=1)
    valid &= norms > 1e-12
    rays[valid] /= norms[valid, None]
    rays[~valid] = np.nan
    return rays.reshape(*shape, 3)


def remap_for_camera(camera, K, size, original_to_virtual=None):
    width, height = size
    yy, xx = np.mgrid[:height, :width]
    rays_virtual = np.stack([xx, yy, np.ones_like(xx)], axis=-1) @ np.linalg.inv(K).T
    rotation = np.eye(3) if original_to_virtual is None else original_to_virtual
    rays_native = rays_virtual @ rotation
    uv = project(camera, rays_native)
    uv[~np.isfinite(uv)] = -1
    maps = (uv[..., 0].astype(np.float32), uv[..., 1].astype(np.float32))
    w, h = camera['resolution']
    valid = (maps[0] >= 0) & (maps[0] < w-1) & (maps[1] >= 0) & (maps[1] < h-1)
    return maps, valid


class StereoGeometry:
    def __init__(self, size, K, R1, R2, P1, P2, left_maps, right_maps, valid, baseline_m):
        self.size, self.K = size, K
        self.R1, self.R2, self.P1_m, self.P2_m = R1, R2, P1, P2
        self.left_maps, self.right_maps, self.valid = left_maps, right_maps, valid
        self.baseline_m = baseline_m

    def document(self):
        P1, P2 = self.P1_m.copy(), self.P2_m.copy()
        P1[:, 3] *= 1000
        P2[:, 3] *= 1000
        return {'schema': 'ego.rectified_camera.v1', 'length_unit': 'mm', 'projection_model': 'pinhole',
                'width': self.size[0], 'height': self.size[1], 'K': self.K.tolist(),
                'P_left_mm': P1.tolist(), 'P_right_mm': P2.tolist(),
                'R_original_to_rectified_left': self.R1.tolist(), 'R_original_to_rectified_right': self.R2.tolist(),
                'baseline_mm': self.baseline_m*1000, 'valid_overlap_ratio': float(self.valid.mean()),
                'depth_encoding': '16UC1', 'depth_unit': 'mm', 'invalid_depth': 0,
                'confidence_encoding': 'mono8', 'confidence_scale': 255, 'confidence_range': [0, 1]}


def stereo_geometry(cal, size=(960, 544), fov_deg=100.):
    size = tuple(map(int, size))
    initial_K = virtual_K(size, fov_deg)
    T = cal['right_T_left']
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(initial_K, np.zeros(4), initial_K, np.zeros(4), size,
                                               T[:3, :3], T[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    if abs(P2[1, 3]) > 1e-9 or P2[0, 3] >= 0:
        raise ValueError('depth backend requires right camera to the right, horizontal positive disparity')
    maps0, valid0 = remap_for_camera(cal['cameras']['cam0'], P1[:, :3], size, R1)
    maps1, valid1 = remap_for_camera(cal['cameras']['cam1'], P2[:, :3], size, R2)
    baseline = -P2[0, 3]/P2[0, 0]
    if not np.isclose(baseline, np.linalg.norm(T[:3, 3]), atol=1e-9):
        raise ValueError('rectified baseline inconsistent')
    return StereoGeometry(size, P1[:, :3], R1, R2, P1, P2, maps0, maps1, valid0 & valid1, baseline)
