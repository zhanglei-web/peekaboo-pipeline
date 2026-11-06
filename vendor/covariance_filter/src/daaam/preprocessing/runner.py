from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import audit, depth, rectify, ros_output, select, vio


STAGES = ('audit', 'vio', 'select', 'rectify', 'depth', 'ros-output')


def _path(value, base, name):
    if not value:
        raise ValueError(f'configuration field {name!r} is required')
    rendered = os.path.expandvars(str(value))
    if '$' in rendered:
        raise ValueError(f'unexpanded environment variable in {name}: {value}')
    path = Path(rendered).expanduser()
    return (path if path.is_absolute() else base/path).resolve()


def run(config_path, *, through='ros-output', ros_mode=None, force=False, resume=False):
    config_path = Path(config_path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    base = config_path.parent
    source = _path(config.get('source'), base, 'source')
    output = _path(config.get('output'), base, 'output')
    if through not in STAGES:
        raise ValueError(f'unknown final stage {through!r}')
    if force and resume:
        raise ValueError('--force and --resume cannot be combined')

    geometry = config.get('geometry', {})
    width, height = int(geometry.get('width', 960)), int(geometry.get('height', 600))
    fov_deg = float(geometry.get('horizontal_fov_deg', 100.))
    audit_dir, vio_dir = output/'audit', output/'vio'
    select_dir, rectify_dir = output/'select', output/'rectify'
    depth_dir, ros_dir = output/'depth', output/'ros_output'

    audit_config = config.get('audit', {})
    audit.run(source, audit_dir, force=force, resume=resume,
              left_video=_path(audit_config['left_video'], base, 'audit.left_video')
              if audit_config.get('left_video') else None,
              right_video=_path(audit_config['right_video'], base, 'audit.right_video')
              if audit_config.get('right_video') else None)
    if through == 'audit':
        return output

    binary = _path(config.get('openvins_binary'), base, 'openvins_binary')
    vio_config = config.get('vio', {})
    vio.run(source, audit_dir, vio_dir, binary=binary, width=width, height=height,
            fov_deg=fov_deg, replay_rate=float(vio_config.get('replay_rate', 1.)),
            initialization_profile=vio_config.get('initialization_profile', 'dynamic'),
            force=force, resume=resume)
    if through == 'vio':
        return output

    select_config = config.get('selection', {})
    select.run(audit_dir, vio_dir, select_dir,
               soft_translation_mm=float(select_config.get('soft_translation_mm', 60.)),
               hard_translation_mm=float(select_config.get('hard_translation_mm', 150.)),
               soft_rotation_deg=float(select_config.get('soft_rotation_deg', 5.)),
               hard_rotation_deg=float(select_config.get('hard_rotation_deg', 12.)),
               min_interval_s=float(select_config.get('min_interval_s', .15)),
               max_gap_s=float(select_config.get('max_gap_s', .25)),
               force=force, resume=resume)
    if through == 'select':
        return output

    rectify.run(source, audit_dir, select_dir, vio_dir, rectify_dir, width=width,
                height=height, fov_deg=fov_deg, force=force, resume=resume)
    if through == 'rectify':
        return output

    depth_config = config.get('depth', {})
    depth_python = os.path.expandvars(str(depth_config.get('python', sys.executable)))
    if '$' in depth_python:
        raise ValueError(f'unexpanded environment variable in depth.python: {depth_python}')
    depth.run(rectify_dir, depth_dir,
              repo=_path(depth_config.get('repo'), base, 'depth.repo'),
              checkpoint=_path(depth_config.get('checkpoint'), base, 'depth.checkpoint'),
              python=depth_python,
              iterations=int(depth_config.get('iterations', 8)),
              max_disp=int(depth_config.get('max_disp', 416)),
              volume_builder=depth_config.get('volume_builder', 'triton'),
              confidence_mode=depth_config.get('confidence_mode', 'left-right'),
              max_depth_mm=int(depth_config.get('max_depth_mm', 65535)),
              warmup=int(depth_config.get('warmup', 2)),
              worker_timeout_s=float(depth_config.get('worker_timeout_s', 600.)),
              force=force, resume=resume)
    if through == 'depth':
        return output

    ros_config = config.get('ros_output', {})
    selected_mode = ros_mode or ros_config.get('mode', 'bag')
    if selected_mode == 'none' or not ros_config.get('enabled', True):
        print('ros-output: skipped (disabled by config or --ros-mode none)', flush=True)
        return output
    ros_output.run(rectify_dir, depth_dir, ros_dir, mode=selected_mode,
                   prefix=ros_config.get('prefix', '/preprocess'),
                   image_format=ros_config.get('image_format', 'jpeg'),
                   jpeg_quality=int(ros_config.get('jpeg_quality', 90)),
                   publish_rate=float(ros_config.get('publish_rate', 1.)),
                   storage_id=ros_config.get('storage_id', 'sqlite3'),
                   force=force, resume=resume)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the modular stereo/IMU preprocessing pipeline.')
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--through', choices=STAGES, default='ros-output',
                        help='stop after this stage; each module is also runnable on its own')
    parser.add_argument('--ros-mode', choices=('bag', 'topic', 'both', 'none'),
                        help='override configured ROS output mode')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', action='store_true')
    group.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    run(args.config, through=args.through, ros_mode=args.ros_mode,
        force=args.force, resume=args.resume)


if __name__ == '__main__':
    main()
