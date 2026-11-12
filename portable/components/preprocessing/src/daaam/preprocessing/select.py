from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import covariance_units, execute_stage, legacy, read_rows, write_json, write_rows


def run(audit, vio, output, *, soft_translation_mm=60., hard_translation_mm=150.,
        soft_rotation_deg=5., hard_rotation_deg=12., min_interval_s=.15, max_gap_s=.25,
        force=False, resume=False):
    audit, vio = Path(audit).resolve(), Path(vio).resolve()
    frames = {int(r['source_idx']): r for r in read_rows(audit/'frame_inventory.jsonl') if r['retained']}
    poses = read_rows(vio/'pose_samples_mm.jsonl')
    if not poses:
        raise ValueError('VIO stage contains no pose samples')
    config = {'soft_translation_mm': soft_translation_mm, 'hard_translation_mm': hard_translation_mm,
              'soft_rotation_deg': soft_rotation_deg, 'hard_rotation_deg': hard_rotation_deg,
              'minimum_interval_s': min_interval_s, 'maximum_gap_s': max_gap_s,
              'algorithm': 'repository motion keyframe selector'}

    def action(out):
        selector = legacy('select_ego_keyframes')
        native = []
        for row in poses:
            source_idx = int(row['source_idx'])
            if source_idx not in frames:
                continue
            transform = np.asarray(row['world_T_cam0_mm'], float).copy()
            transform[:3, 3] /= 1000.
            native.append({**row, 'world_T_cam0': transform.tolist(),
                           'position_m': (transform[:3, 3]).tolist(),
                           'pose_covariance': covariance_units(row['pose_covariance_mm_rad'], .001).tolist(),
                           'left_video_frame_idx': int(frames[source_idx]['left_video_frame_idx']),
                           'right_video_frame_idx': int(frames[source_idx]['right_video_frame_idx'])})
        native.sort(key=lambda x: (int(x['sensor_time_ns']), int(x['source_idx'])))
        thresholds = selector.SelectionConfig(
            soft_translation_m=soft_translation_mm/1000., hard_translation_m=hard_translation_mm/1000.,
            soft_rotation_deg=soft_rotation_deg, hard_rotation_deg=hard_rotation_deg,
            minimum_interval_s=min_interval_s, maximum_gap_s=max_gap_s)
        chosen = selector.select_keyframes(native, thresholds)
        public = []
        pose_by_source = {int(row['source_idx']): row for row in poses}
        for index, record in enumerate(chosen):
            original = pose_by_source[int(record['source_idx'])]
            frame = frames[int(record['source_idx'])]
            public.append({**original, 'keyframe_index': index,
                           'left_video_frame_idx': int(frame['left_video_frame_idx']),
                           'right_video_frame_idx': int(frame['right_video_frame_idx']),
                           'selection_reason': record['selection_reason'],
                           'selection_reasons': record['selection_reasons'],
                           'translation_from_previous_selected_mm': float(record['translation_from_previous_selected_m']*1000),
                           'rotation_from_previous_selected_deg': record['rotation_from_previous_selected_deg'],
                           'gap_from_previous_selected_s': record['gap_from_previous_selected_s']})
        if not public:
            raise RuntimeError('keyframe selection produced no frames')
        write_rows(out/'selected_frames.jsonl', public)
        write_json(out/'selection_report.json', {'input_pose_count': len(native), 'selected_count': len(public),
                   'source_indices': [int(x['source_idx']) for x in public],
                   'thresholds': config, 'online_causal': True,
                   'selector': 'scripts/select_ego_keyframes.py:select_keyframes'})
        return {'input_pose_count': len(native), 'selected_count': len(public)}

    return execute_stage('select', config, output, [audit, vio], action, resume=resume, force=force)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Select keyframes with the repository pose-motion selector.')
    parser.add_argument('--audit', required=True, type=Path); parser.add_argument('--vio', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--soft-translation-mm', type=float, default=60.); parser.add_argument('--hard-translation-mm', type=float, default=150.)
    parser.add_argument('--soft-rotation-deg', type=float, default=5.); parser.add_argument('--hard-rotation-deg', type=float, default=12.)
    parser.add_argument('--min-interval-s', type=float, default=.15); parser.add_argument('--max-gap-s', type=float, default=.25)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--force', action='store_true')
    a = parser.parse_args(argv)
    run(a.audit, a.vio, a.output, soft_translation_mm=a.soft_translation_mm,
        hard_translation_mm=a.hard_translation_mm, soft_rotation_deg=a.soft_rotation_deg,
        hard_rotation_deg=a.hard_rotation_deg, min_interval_s=a.min_interval_s,
        max_gap_s=a.max_gap_s, resume=a.resume, force=a.force)


if __name__ == '__main__':
    main()
