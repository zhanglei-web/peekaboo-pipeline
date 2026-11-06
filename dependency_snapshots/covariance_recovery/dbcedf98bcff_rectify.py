from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import VideoReader, execute_stage, legacy, read_rows, write_json, write_rows
from .geometry import load_calibration, stereo_geometry


def run(source, audit, selection, vio, output, *, width=960, height=600, fov_deg=100., force=False, resume=False):
    source, audit, selection, vio = map(lambda p: Path(p).resolve(), (source, audit, selection, vio))
    selected = read_rows(selection/'selected_frames.jsonl')
    calibration = load_calibration(source)
    geometry = stereo_geometry(calibration, (width, height), fov_deg)
    if float(geometry.valid.mean()) < .25:
        raise ValueError(f'rectified shared camera coverage is too low: {geometry.valid.mean():.3f}')
    config = {'source': str(source), 'audit': str(audit), 'selection': str(selection), 'vio': str(vio),
              'width': width, 'height': height, 'horizontal_fov_deg': fov_deg,
              'model': 'double-sphere to rectified virtual pinhole'}

    def action(out):
        left_video, right_video = VideoReader(source/'videos/left.mkv'), VideoReader(source/'videos/right.mkv')
        audit_rows = {int(r['source_idx']): r for r in read_rows(audit/'frame_inventory.jsonl')}
        pose_rows = {int(r['source_idx']): r for r in read_rows(vio/'pose_samples_mm.jsonl')}
        frames, poses = [], []
        try:
            for i, keyframe in enumerate(selected):
                source_idx = int(keyframe['source_idx'])
                original = audit_rows[source_idx]
                if not original['retained'] or source_idx not in pose_rows:
                    raise ValueError(f'selected source frame {source_idx} missing audited pair or pose')
                left_raw = left_video.get(int(original['left_video_frame_idx']))
                right_raw = right_video.get(int(original['right_video_frame_idx']))
                left = cv2.remap(left_raw, *geometry.left_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                right = cv2.remap(right_raw, *geometry.right_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                stem = f'{i:08d}'
                left_path, right_path = out/'left_rect'/f'{stem}.png', out/'right_rect'/f'{stem}.png'
                left_path.parent.mkdir(parents=True, exist_ok=True); right_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(left_path), left, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise IOError(f'cannot write {left_path}')
                if not cv2.imwrite(str(right_path), right, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise IOError(f'cannot write {right_path}')
                base_pose = np.asarray(pose_rows[source_idx]['world_T_cam0_mm'], float)
                cam0_T_rect = np.eye(4); cam0_T_rect[:3, :3] = geometry.R1.T
                world_T_rect = base_pose @ cam0_T_rect
                q = legacy('run_ego_openvins').matrix_to_quaternion_xyzw(world_T_rect[:3, :3])
                frame = {'schema': 'ego.frame.v1', 'keyframe_index': i,
                         'frame_id': 'cam0_rect', 'source_idx': source_idx,
                         'left_source_idx': int(original['left_source_idx']),
                         'right_source_idx': int(original['right_source_idx']),
                         'left_video_frame_idx': int(original['left_video_frame_idx']),
                         'right_video_frame_idx': int(original['right_video_frame_idx']),
                         'sensor_time_ns': int(original['sensor_time_ns']),
                         'reference_time_ns': int(original['reference_time_ns']),
                         'processing_completed_time_ns': None,
                         'left_image_path': str(left_path.relative_to(out)),
                         'right_image_path': str(right_path.relative_to(out)),
                         'image_encoding': 'bgr8', 'rectified_left_frame': 'cam0_rect',
                         'selection_reason': keyframe['selection_reason']}
                frames.append(frame)
                poses.append({'schema': 'ego.pose.v1', 'length_unit': 'mm',
                              'source_idx': source_idx, 'keyframe_index': i,
                              'sensor_time_ns': int(original['sensor_time_ns']),
                              'reference_time_ns': int(original['reference_time_ns']),
                              'pose_query_time_ns': int(pose_rows[source_idx]['pose_query_time_ns']),
                              'processing_completed_time_ns': None,
                              'frame_id': 'vio_world', 'child_frame_id': 'cam0_rect',
                              'position_mm': world_T_rect[:3, 3].tolist(),
                              'orientation_xyzw': q.tolist(),
                              'world_T_camera_mm': world_T_rect.tolist(),
                              'pose_covariance_mm_rad': pose_rows[source_idx]['pose_covariance_mm_rad'],
                              'openvins_left_bracket_ns': pose_rows[source_idx]['openvins_left_bracket_ns'],
                              'openvins_right_bracket_ns': pose_rows[source_idx]['openvins_right_bracket_ns'],
                              'lookup_semantics': pose_rows[source_idx]['lookup_semantics']})
        finally:
            left_video.close(); right_video.close()
        write_json(out/'rectified_camera.json', geometry.document())
        write_rows(out/'frame_manifest.jsonl', frames)
        write_rows(out/'pose_samples_mm.jsonl', poses)
        write_json(out/'rectification_report.json', {'frame_count': len(frames),
                   'shared_valid_ratio': float(geometry.valid.mean()), 'baseline_mm': geometry.baseline_m*1000.,
                   'left_map_valid_ratio': float(np.mean((geometry.left_maps[0]>=0)&(geometry.left_maps[1]>=0))),
                   'right_map_valid_ratio': float(np.mean((geometry.right_maps[0]>=0)&(geometry.right_maps[1]>=0))),
                   'rectified_pose_convention': 'world_T_cam0 @ inverse(R1); camera center unchanged'})
        return {'rectified_frames': len(frames), 'width': width, 'height': height,
                'shared_valid_ratio': float(geometry.valid.mean()), 'baseline_mm': geometry.baseline_m*1000.}

    return execute_stage('rectify', config, output, [audit, selection, vio], action, resume=resume, force=force)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Rectify DS stereo images and rotate matching poses.')
    parser.add_argument('--source', required=True, type=Path); parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path); parser.add_argument('--vio', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path); parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=600); parser.add_argument('--fov-deg', type=float, default=100.)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--force', action='store_true')
    a = parser.parse_args(argv)
    run(a.source, a.audit, a.selection, a.vio, a.output, width=a.width, height=a.height,
        fov_deg=a.fov_deg, resume=a.resume, force=a.force)


if __name__ == '__main__':
    main()
