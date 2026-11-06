from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import (camera_covariance, covariance_units, execute_stage, legacy,
                     read_json, read_rows, write_json, write_rows)
from .geometry import load_calibration, remap_for_camera, virtual_K


def run(source, audit, output, *, binary, width=960, height=600, fov_deg=100.,
        replay_rate=1., initialization_profile='dynamic', force=False, resume=False):
    source, audit, binary = Path(source).resolve(), Path(audit).resolve(), Path(binary).resolve()
    inventory = read_rows(audit/'frame_inventory.jsonl')
    frames_rows = [row for row in inventory if row['retained']]
    if not frames_rows:
        raise ValueError('audit has no camera pairs with shifted IMU coverage')
    calibration = load_calibration(source)
    ov = legacy('run_ego_openvins')
    initialization_overrides, initialization_provenance = ov.load_initialization_overrides(
        initialization_profile
    )
    K = virtual_K((width, height), fov_deg)
    source_calibration = calibration['cameras']
    left_maps, left_valid = remap_for_camera(source_calibration['cam0'], K, (width, height))
    right_maps, right_valid = remap_for_camera(source_calibration['cam1'], K, (width, height))
    common_valid = left_valid & right_valid
    if float(common_valid.mean()) < 0.35:
        raise ValueError(f'virtual pinhole valid stereo region too small: {common_valid.mean():.3f}')
    config = {'source': str(source), 'audit': str(audit), 'openvins_binary': str(binary),
              'virtual_camera': {'width': width, 'height': height, 'horizontal_fov_deg': fov_deg,
                                 'K': K.tolist(), 'model': 'pinhole', 'distortion_model': 'radtan'},
              'replay_rate': replay_rate, 'time_shift_ns': calibration['time_shift_ns'],
              'initialization_profile': initialization_profile,
              'initialization_override': initialization_provenance}
    dependencies = [audit]

    def action(out):
        config_dir, native_dir = out/'config', out/'native'
        config_dir.mkdir(); native_dir.mkdir()
        estimator, imucam, imu_doc, provenance = ov.build_capture_config_documents(source)
        for name in ('cam0', 'cam1'):
            imucam[name]['camera_model'] = 'pinhole'
            imucam[name]['distortion_model'] = 'radtan'
            imucam[name]['distortion_coeffs'] = [0., 0., 0., 0.]
            imucam[name]['intrinsics'] = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
            imucam[name]['resolution'] = [width, height]
        for filename, document in (('estimator_config.yaml', estimator),
                                   ('kalibr_imucam_chain.yaml', imucam),
                                   ('kalibr_imu_chain.yaml', imu_doc)):
            (config_dir/filename).write_text(ov._opencv_yaml_text(document))
        native_frames = [ov.CameraFrame(int(row['source_idx']), int(row['sensor_time_ns']),
                                         int(row['left_video_frame_idx']), int(row['right_video_frame_idx']))
                         for row in frames_rows]
        imu_samples = ov.read_imu_samples(source/'imu/imu0.csv')
        replay = ov.run_ros_replay(source_dir=source, output_dir=native_dir,
                                   config_dir=config_dir, binary=binary, frames=native_frames,
                                   imu_samples=imu_samples, replay_rate=replay_rate,
                                   handshake_timeout_s=30., drain_timeout_s=30., drain_quiet_s=.5,
                                   imu_preroll_s=.05, imu_tail_s=.05,
                                   ros_parameter_overrides=initialization_overrides,
                                   left_video_path=source/'videos/left.mkv',
                                   right_video_path=source/'videos/right.mkv',
                                   left_rectification_maps=left_maps, right_rectification_maps=right_maps)
        ov.write_pose_products(
            odometry_path=native_dir/'openvins_odometry.jsonl', camera_frames=native_frames,
            T_imu_cam0=np.linalg.inv(source_calibration['cam0']['T_cam_imu']),
            timeshift_cam_imu_s=float(source_calibration['cam0']['timeshift_cam_imu']),
            output_dir=native_dir)
        # The legacy writer normalizes NumPy scalars in the saved report.
        native_pose_report = read_json(native_dir/'vio_acceptance.json')
        if not native_pose_report['passed']:
            failed_gates = [name for name, passed in native_pose_report['gates'].items() if not passed]
            print(f'vio: WARNING acceptance gates failed: {failed_gates}; '
                  'only frames with bracketed poses will reach downstream stages; '
                  'see native/vio_acceptance.json', flush=True)
        pose_samples = ov.load_pose_samples(native_dir/'openvins_odometry.jsonl')
        out_rows = []
        for row in read_rows(native_dir/'camera_pose_samples.jsonl'):
            transform = np.asarray(row['world_T_cam0'], float)
            query_ns = int(row['pose_query_time_ns'])
            interpolated = ov.interpolate_pose(pose_samples, query_ns, max_bracket_gap_ns=50_000_000)
            cov = camera_covariance(interpolated.world_T_imu,
                                    np.linalg.inv(source_calibration['cam0']['T_cam_imu']),
                                    interpolated.pose_covariance)
            transform_mm = transform.copy(); transform_mm[:3, 3] *= 1000.
            out_rows.append({'schema': 'ego.pose.v1', 'length_unit': 'mm',
                             'source_idx': int(row['source_idx']),
                             'left_video_frame_idx': int(row['left_video_frame_idx']),
                             'right_video_frame_idx': int(row['right_video_frame_idx']),
                             'sensor_time_ns': int(row['sensor_time_ns']),
                             'reference_time_ns': int(row['sensor_time_ns']),
                             'pose_query_time_ns': query_ns,
                             'processing_completed_time_ns': None,
                             'frame_id': 'vio_world', 'child_frame_id': 'cam0',
                             'position_mm': transform_mm[:3, 3].tolist(),
                             'orientation_xyzw': row['orientation_xyzw'],
                             'world_T_cam0_mm': transform_mm.tolist(),
                             'pose_covariance_mm_rad': covariance_units(cov, 1000.).tolist(),
                             'openvins_left_bracket_ns': int(row['openvins_left_bracket_ns']),
                             'openvins_right_bracket_ns': int(row['openvins_right_bracket_ns']),
                             'interpolation_alpha': float(row['interpolation_alpha']),
                             'lookup_semantics': 'bracketed_linear_slerp'})
        if not out_rows:
            raise RuntimeError('OpenVINS returned no bracketed camera poses')
        write_rows(out/'pose_samples_mm.jsonl', out_rows)
        write_json(out/'vio_report.json', {'replay': replay, 'pose_acceptance': native_pose_report,
                   'pose_samples': len(out_rows), 'input_stereo_pairs': len(frames_rows),
                   'initialization_profile': initialization_profile,
                   'initialization_override': initialization_provenance,
                   'virtual_stereo_valid_ratio': float(common_valid.mean()),
                   'native_odometry_path': 'native/openvins_odometry.jsonl',
                   'camera_pose_uncertainty': 'lever-arm transformed then converted to mm^2/rad'})
        write_json(out/'virtual_camera.json', {'width': width, 'height': height, 'K': K.tolist(),
                   'camera_model': 'pinhole', 'distortion_model': 'radtan',
                   'source_camera_model': [source_calibration['cam0']['camera_model'],
                                           source_calibration['cam1']['camera_model']],
                   'valid_stereo_ratio': float(common_valid.mean()),
                   'openvins_calibration_source': 'Kalibr T_imu_cam extrinsics with DS ray remapping'})
        return {'camera_pose_count': len(out_rows), 'replay_camera_pairs': replay['camera_pair_count'],
                'openvins_odometry_count': replay['odometry_count'],
                'pose_acceptance_passed': native_pose_report['passed']}

    return execute_stage('vio', config, output, dependencies, action, resume=resume, force=force)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Run OpenVINS on DS-undistorted virtual pinhole stereo images.')
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--openvins-binary', required=True, type=Path)
    parser.add_argument('--width', type=int, default=960); parser.add_argument('--height', type=int, default=600)
    parser.add_argument('--fov-deg', type=float, default=100.); parser.add_argument('--replay-rate', type=float, default=1.)
    parser.add_argument('--initialization-profile', choices=('static', 'dynamic'), default='dynamic')
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    run(args.source, args.audit, args.output, binary=args.openvins_binary, width=args.width,
        height=args.height, fov_deg=args.fov_deg, replay_rate=args.replay_rate,
        initialization_profile=args.initialization_profile,
        force=args.force, resume=args.resume)


if __name__ == '__main__':
    main()
