from __future__ import annotations

from pathlib import Path

import cv2

from daaam.datasets.ego_stereo_imu import parse_imu_csv, parse_timestamp_file

from .common import execute_stage, read_json, sha256, write_json, write_rows
from .geometry import load_calibration, public_calibration


def _decode_count(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'cannot open video: {path}')
    count, shape = 0, None
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f'invalid decoded frame {count}: {path}')
            current = list(image.shape[:2])
            if shape is not None and current != shape:
                raise ValueError(f'video resolution changes at frame {count}: {path}')
            shape = current
            count += 1
    finally:
        capture.release()
    return count, shape


def run(source, output, *, force=False, resume=False, left_video=None, right_video=None):
    source = Path(source).expanduser().resolve()
    videos = {'left': Path(left_video or source/'videos/left.mkv'),
              'right': Path(right_video or source/'videos/right.mkv')}
    ts = {side: parse_timestamp_file(source/'timestamps'/f'{side}_timestamps.txt') for side in videos}
    calibration = load_calibration(source)
    imu = parse_imu_csv(source/'imu/imu0.csv')
    if not imu:
        raise ValueError('IMU CSV has no samples')
    counts = {side: _decode_count(path) for side, path in videos.items()}
    for side in videos:
        if counts[side][0] != len(ts[side]):
            raise ValueError(f'{side}: decoded {counts[side][0]} frames but sidecar has {len(ts[side])} rows')
        if counts[side][1] != list(calibration['cameras']['cam0']['resolution'][::-1]):
            raise ValueError(f'{side}: decoded frame dimensions disagree with camera calibration')

    by_time = {}
    for side in ('left', 'right'):
        index = {}
        for row in ts[side]:
            if row.sensor_time_ns in index:
                raise ValueError(f'duplicate {side} timestamp {row.sensor_time_ns}')
            index[row.sensor_time_ns] = row
        by_time[side] = index
    matches, unmatched_left = [], []
    for row in ts['left']:
        other = by_time['right'].get(row.sensor_time_ns)
        if other is None:
            unmatched_left.append(row.source_idx)
            continue
        query_ns = row.sensor_time_ns + calibration['time_shift_ns']
        reasons = []
        if query_ns < imu[0].sensor_time_ns:
            reasons.append('imu_time_shift_before_coverage')
        if query_ns > imu[-1].sensor_time_ns:
            reasons.append('imu_time_shift_after_coverage')
        matches.append({'frame_id': f'pair-{len(matches):08d}',
                        'source_idx': row.source_idx,
                        'left_source_idx': row.source_idx, 'right_source_idx': other.source_idx,
                        'left_video_frame_idx': row.source_idx, 'right_video_frame_idx': other.source_idx,
                        'left_sensor_time_ns': row.sensor_time_ns,
                        'right_sensor_time_ns': other.sensor_time_ns,
                        'sensor_time_ns': row.sensor_time_ns,
                        'reference_time_ns': row.sensor_time_ns,
                        'pose_query_time_ns': query_ns,
                        'retained': not reasons, 'drop_reasons': reasons})
    left_times = set(by_time['left'])
    unmatched_right = [row.source_idx for row in ts['right'] if row.sensor_time_ns not in left_times]
    if len(matches) < 2:
        raise ValueError('fewer than two exact timestamp stereo matches')

    relevant = [*videos.values(), source/'timestamps/left_timestamps.txt', source/'timestamps/right_timestamps.txt',
                source/'imu/imu0.csv', source/'meta.json', *sorted((source/'calibrations').glob('*.yaml'))]
    input_hashes = {str(p): sha256(p) for p in relevant}
    metadata = read_json(source/'meta.json') if (source/'meta.json').exists() else {}
    config = {'source': str(source), 'videos': {k: str(v) for k, v in videos.items()},
              'pairing': 'exact_integer_timestamp_join', 'length_unit': 'mm', 'time_unit': 'ns'}

    def action(out):
        write_rows(out/'frame_inventory.jsonl', matches)
        write_json(out/'calibration_mm.json', public_calibration(calibration))
        report = {'schema': 'ego.audit.v1', 'input_sha256': input_hashes,
                  'decoded_video_frames': {k: v[0] for k, v in counts.items()},
                  'decoded_resolution_wh': {k: [v[1][1], v[1][0]] for k, v in counts.items()},
                  'timestamp_rows': {k: len(v) for k, v in ts.items()},
                  'exact_timestamp_pairs': len(matches),
                  'eligible_pairs': sum(r['retained'] for r in matches),
                  'unmatched_left_source_indices': unmatched_left,
                  'unmatched_right_source_indices': unmatched_right,
                  'imu_samples': len(imu), 'imu_time_range_ns': [imu[0].sensor_time_ns, imu[-1].sensor_time_ns],
                  'imu_orientation_used': False,
                  'synchronization_status': 'unknown' if metadata.get('sync_mode') == 'none' else 'not_verified',
                  'metadata_sync_mode': metadata.get('sync_mode'),
                  'processing_time_is_separate': True}
        write_json(out/'audit_report.json', report)
        return report

    return execute_stage('audit', config, output, [], action, resume=resume, force=force, extra=input_hashes)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Audit stereo/IMU sources and exact-join image timestamps.')
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--left-video', type=Path)
    parser.add_argument('--right-video', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    run(args.source, args.output, force=args.force, resume=args.resume,
        left_video=args.left_video, right_video=args.right_video)


if __name__ == '__main__':
    main()
