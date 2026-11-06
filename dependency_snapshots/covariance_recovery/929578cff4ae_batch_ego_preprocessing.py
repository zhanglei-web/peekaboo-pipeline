#!/usr/bin/env python3
"""Run the existing preprocessing pipeline and export bags under capture names."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from daaam.preprocessing.common import read_json, read_rows, sha256, write_json


REQUIRED = (
    'videos/left.mkv', 'videos/right.mkv',
    'timestamps/left_timestamps.txt', 'timestamps/right_timestamps.txt',
    'imu/imu0.csv', 'calibrations/camera-camchain.yaml',
    'calibrations/camera_imu-camchain-imucam.yaml',
    'calibrations/camera_imu-imu.yaml', 'meta.json',
)


def verify_bag(bag_path: Path, output: Path) -> dict:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    selected = read_rows(output / 'select/selected_frames.jsonl')
    expected = {int(row['sensor_time_ns']) for row in selected}
    if not expected or len(expected) != len(selected):
        raise ValueError('selected frames must have nonempty, unique timestamps')
    for stage, filename in (
        ('rectify', 'frame_manifest.jsonl'), ('rectify', 'pose_samples_mm.jsonl'),
        ('depth', 'depth_manifest.jsonl'),
    ):
        rows = read_rows(output / stage / filename)
        if len(rows) != len(expected) or {int(row['sensor_time_ns']) for row in rows} != expected:
            raise ValueError(f'{stage}/{filename}: acquisition timestamps differ')
    report = read_json(output / 'ros_output/ros_output_report.json')
    camera = read_json(output / 'rectify/rectified_camera.json')
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_path), storage_id='sqlite3'),
        rosbag2_py.ConverterOptions('cdr', 'cdr'),
    )
    types = {topic.name: topic.type for topic in reader.get_all_topics_and_types()}
    if types != report['topics'] or len(types) != 9:
        raise ValueError(f'unexpected bag topics: {types}')
    classes = {topic: get_message(kind) for topic, kind in types.items()}
    counts, stamps = Counter(), defaultdict(set)
    while reader.has_next():
        topic, raw, timestamp = reader.read_next()
        if timestamp not in expected:
            raise ValueError(f'{topic}: unexpected bag timestamp {timestamp}')
        msg = deserialize_message(raw, classes[topic])
        counts[topic] += 1
        stamps[topic].add(timestamp)
        headers = ([transform.header for transform in msg.transforms]
                   if types[topic] == 'tf2_msgs/msg/TFMessage'
                   else [msg.header] if hasattr(msg, 'header') else [])
        if any(header.stamp.sec * 10**9 + header.stamp.nanosec != timestamp
               for header in headers):
            raise ValueError(f'{topic}: header stamp differs from acquisition time')
        if topic.endswith('/compressed'):
            if types[topic] != 'sensor_msgs/msg/CompressedImage' or 'jpeg' not in msg.format:
                raise ValueError(f'{topic}: expected JPEG CompressedImage')
        if topic.endswith('/depth/image'):
            if (msg.encoding != '16UC1' or
                    (msg.width, msg.height) != (camera['width'], camera['height'])):
                raise ValueError(f'{topic}: invalid depth encoding or dimensions')
        if topic.endswith('/depth/confidence') and msg.encoding != 'mono8':
            raise ValueError(f'{topic}: expected mono8 confidence')
    for topic in types:
        if counts[topic] != len(expected) or stamps[topic] != expected:
            raise ValueError(f'{topic}: missing or duplicated frames')
    return {
        'frames': len(expected), 'messages': sum(counts.values()),
        'topic_message_counts': dict(counts),
        'first_sensor_time_ns': min(expected), 'last_sensor_time_ns': max(expected),
        'timestamp_alignment_verified': True,
    }


def export_bag(output: Path, bag_root: Path, name: str) -> tuple[Path, dict]:
    source = output / 'ros_output/rosbag'
    metadata = yaml.safe_load((source / 'metadata.yaml').read_text())
    info = metadata['rosbag2_bagfile_information']
    renames = {old: f'{name}_{index}{Path(old).suffix}'
               for index, old in enumerate(info['relative_file_paths'])}
    target = bag_root / name
    if target.exists():
        if all((target / new).is_file() and sha256(source / old) == sha256(target / new)
               for old, new in renames.items()):
            return target, verify_bag(target, output)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=bag_root))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        for old, new in renames.items():
            (temporary / old).rename(temporary / new)
        info['relative_file_paths'] = list(renames.values())
        for entry in info.get('files', []):
            entry['path'] = renames[entry['path']]
        # This writer can double the per-file count at close; use the verified count.
        if len(info.get('files', [])) == 1:
            info['files'][0]['message_count'] = info['message_count']
        (temporary / 'metadata.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))
        verification = verify_bag(temporary, output)
        if target.exists():
            target.rename(target.with_name(f'{name}.previous-{time.time_ns()}'))
        temporary.rename(target)
        return target, verification
    except BaseException:
        print(f'Export diagnostics retained in {temporary}', file=sys.stderr, flush=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--bag-root', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path, help='existing pipeline config template')
    parser.add_argument('--names', nargs='+', help='optional subset of exact capture directory names')
    parser.add_argument('--ros-domain-id', type=int, default=81)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--force', action='store_true')
    args = parser.parse_args()
    input_root, output_root, bag_root = (p.expanduser().resolve() for p in
                                        (args.input_root, args.output_root, args.bag_root))
    captures = sorted(p for p in input_root.iterdir() if p.is_dir() and (p / 'meta.json').is_file())
    if args.names:
        by_name = {p.name: p for p in captures}
        missing = set(args.names) - by_name.keys()
        if missing:
            parser.error(f'capture directories not found: {sorted(missing)}')
        captures = [by_name[name] for name in dict.fromkeys(args.names)]
    if not captures:
        parser.error('no captures found')
    for capture in captures:
        missing = [relative for relative in REQUIRED if not (capture / relative).is_file()]
        if missing:
            parser.error(f'{capture.name}: missing inputs: {missing}')
    template = read_json(args.config)
    for field in ('openvins_binary',):
        if not Path(template[field]).is_absolute():
            parser.error(f'template {field} must be an absolute path')
    for field in ('repo', 'checkpoint', 'python'):
        if not Path(template['depth'][field]).is_absolute():
            parser.error(f'template depth.{field} must be an absolute path')
    output_root.mkdir(parents=True, exist_ok=True)
    bag_root.mkdir(parents=True, exist_ok=True)
    config_root = output_root / 'configs'
    config_root.mkdir(exist_ok=True)
    results_path = bag_root / 'batch_report.json'
    results = read_json(results_path) if results_path.exists() else {'schema': 'ego.batch.v1', 'datasets': {}}
    failures = 0
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    env['ROS_DOMAIN_ID'] = str(args.ros_domain_id)
    for index, capture in enumerate(captures, 1):
        output = output_root / capture.name
        config = copy.deepcopy(template)
        config.update(source=str(capture), output=str(output))
        config.pop('audit', None)  # Video overrides in the template must not leak across captures.
        config['ros_output'].update(enabled=True, mode='bag')
        config_path = config_root / f'{capture.name}.json'
        write_json(config_path, config)
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / f'batch-{time.time_ns()}.log'
        record = {'name': capture.name, 'source': str(capture), 'output': str(output),
                  'config': str(config_path), 'log': str(log_path), 'status': 'running'}
        results['datasets'][capture.name] = record
        write_json(results_path, results)
        print(f'[{index}/{len(captures)}] START {capture.name}', flush=True)
        try:
            command = [sys.executable, '-m', 'daaam.preprocessing.runner', '--config',
                       str(config_path), '--ros-mode', 'bag']
            if args.resume:
                command.append('--resume')
            if args.force:
                command.append('--force')
            with log_path.open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            bag, verification = export_bag(output, bag_root, capture.name)
            acceptance = read_json(output / 'vio/native/vio_acceptance.json')
            audit = read_json(output / 'audit/audit_report.json')
            record.update(status='complete', bag=str(bag), verification=verification,
                          original_frames=audit['decoded_video_frames'],
                          paired_frames=audit['exact_timestamp_pairs'],
                          audited_frames=audit['eligible_pairs'],
                          pose_frames=acceptance['metrics']['camera_pose_count'],
                          vio_acceptance_passed=acceptance['passed'],
                          vio_failed_gates=[key for key, value in acceptance['gates'].items() if not value],
                          bag_size_bytes=sum(p.stat().st_size for p in bag.iterdir() if p.is_file()))
            print(f'[{index}/{len(captures)}] COMPLETE {capture.name}: '
                  f'{verification["frames"]} frames -> {bag}', flush=True)
        except Exception as error:
            failures += 1
            record.update(status='failed', error=str(error))
            with log_path.open('a') as log:
                traceback.print_exc(file=log)
            print(f'[{index}/{len(captures)}] FAILED {capture.name}: {error}; log={log_path}', flush=True)
        finally:
            record['finished_time_ns'] = time.time_ns()
            write_json(results_path, results)
    print(f'Finished: {len(captures) - failures}/{len(captures)} complete; report={results_path}', flush=True)
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
