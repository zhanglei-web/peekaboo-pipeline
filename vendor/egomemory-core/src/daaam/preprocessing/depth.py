from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

from .common import execute_stage, read_json, read_rows, sha256, write_json, write_rows


def _read_worker_message(process, timeout_s):
    ready, _, _ = select.select([process.stdout], [], [], timeout_s)
    if not ready:
        raise TimeoutError(f'depth worker did not respond within {timeout_s}s')
    line = process.stdout.readline()
    if not line:
        error = process.stderr.read()
        raise RuntimeError(f'depth worker exited ({process.poll()}): {error[-4000:]}')
    return json.loads(line)


def run(rectify, output, *, repo, checkpoint, python=sys.executable, iterations=8,
        max_disp=416, volume_builder='triton', confidence_mode='left-right',
        max_depth_mm=65535, warmup=2, worker_timeout_s=600., force=False, resume=False):
    rectify, repo, checkpoint = map(lambda p: Path(p).expanduser().resolve(), (rectify, repo, checkpoint))
    if not 0 < max_depth_mm <= 65535:
        raise ValueError('uint16 depth output requires max_depth_mm in [1,65535]')
    if not (repo.is_dir() and checkpoint.is_file()):
        raise FileNotFoundError('Fast-FoundationStereo repo/checkpoint missing')
    rectified = read_json(rectify/'rectified_camera.json')
    frames = read_rows(rectify/'frame_manifest.jsonl')
    if not frames:
        raise ValueError('rectification has no images')
    config = {'rectify': str(rectify), 'repo': str(repo), 'checkpoint': str(checkpoint),
              'repo_sha256': sha256(repo/'core/igev_stereo.py') if (repo/'core/igev_stereo.py').is_file() else None,
              'checkpoint_sha256': sha256(checkpoint), 'python': str(python),
              'iterations': iterations, 'max_disp_px': max_disp, 'volume_builder': volume_builder,
              'confidence_mode': confidence_mode, 'max_depth_mm': max_depth_mm, 'warmup': warmup}

    def action(out):
        worker_script = Path(__file__).resolve().parents[3]/'scripts/fast_foundation_stereo_worker.py'
        command = [str(python), str(worker_script), '--repo', str(repo), '--checkpoint', str(checkpoint),
                   '--width', str(rectified['width']), '--height', str(rectified['height']),
                   '--iters', str(iterations), '--max-disp', str(max_disp), '--volume-builder', volume_builder,
                   '--confidence-mode', confidence_mode, '--warmup', str(warmup)]
        env = os.environ.copy(); env['DAAAM_ENABLE_ISOLATED_LEGACY'] = '1'
        log = (out/'worker.log').open('w')
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=log, text=True, bufsize=1, env=env)
        manifests = []
        try:
            ready = _read_worker_message(process, worker_timeout_s)
            if ready.get('status') != 'ready':
                raise RuntimeError(f'depth worker failed to initialize: {ready}')
            for row in frames:
                started_ns = time.time_ns()
                request_id = f"{row['keyframe_index']}:{row['sensor_time_ns']}"
                request = {'command': 'infer', 'request_id': request_id,
                           'frame_idx': int(row['keyframe_index']), 'sensor_time_ns': int(row['sensor_time_ns']),
                           'left_path': str(rectify/row['left_image_path']),
                           'right_path': str(rectify/row['right_image_path']),
                           'fx': float(rectified['K'][0][0]), 'baseline_m': float(rectified['baseline_mm'])/1000.,
                           'maximum_depth_m': max_depth_mm/1000., 'output_dir': str(out)}
                process.stdin.write(json.dumps(request, allow_nan=False)+'\n'); process.stdin.flush()
                result = _read_worker_message(process, worker_timeout_s)
                if result.get('status') != 'ok' or result.get('request_id') != request_id:
                    raise RuntimeError(f'depth inference failed for {request_id}: {result}')
                completed_ns = time.time_ns()
                manifests.append({'schema': 'ego.depth.v1', 'keyframe_index': int(row['keyframe_index']),
                                  'source_idx': int(row['source_idx']), 'sensor_time_ns': int(row['sensor_time_ns']),
                                  'reference_time_ns': int(row['reference_time_ns']),
                                  'processing_started_time_ns': started_ns,
                                  'processing_completed_time_ns': completed_ns,
                                  'depth_path': str(Path(result['depth_path']).relative_to(out)),
                                  'confidence_path': str(Path(result['confidence_path']).relative_to(out)),
                                  'depth_unit': 'mm', 'encoding': '16UC1', 'invalid_value': 0,
                                  'confidence_encoding': 'mono8', 'confidence_range': [0, 255],
                                  'valid_ratio': result['valid_ratio'], 'mean_confidence': result['mean_confidence'],
                                  'confidence_mode': result['confidence_mode']})
            process.stdin.write(json.dumps({'command': 'shutdown'})+'\n'); process.stdin.flush()
            _read_worker_message(process, 30.)
            code = process.wait(timeout=30.)
            if code:
                raise RuntimeError(f'depth worker exit code {code}')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            log.write(f"command={command!r}\nreturncode={process.returncode}\n")
            log.close()
        write_rows(out/'depth_manifest.jsonl', manifests)
        write_json(out/'depth_report.json', {'frames': len(manifests), 'model': ready.get('model'),
                   'artifacts': ready.get('artifacts'), 'settings': config,
                   'depth_scale': 'uint16 millimetres; zero denotes invalid',
                   'confidence_scale': 'uint8 [0,255] representing [0,1]'})
        return {'depth_frames': len(manifests), 'model': ready.get('model')}

    return execute_stage('depth', config, output, [rectify], action, resume=resume, force=force,
                         extra={'checkpoint_sha256': config['checkpoint_sha256']})


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Run pinned Fast-FoundationStereo on rectified keyframes.')
    parser.add_argument('--rectify', required=True, type=Path); parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--repo', required=True, type=Path); parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--python', default=sys.executable); parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--max-disp', type=int, default=416); parser.add_argument('--volume-builder', choices=('triton','pytorch1'), default='triton')
    parser.add_argument('--confidence-mode', choices=('left-right','validity'), default='left-right')
    parser.add_argument('--max-depth-mm', type=int, default=65535); parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--worker-timeout-s', type=float, default=600.)
    parser.add_argument('--resume', action='store_true'); parser.add_argument('--force', action='store_true')
    a = parser.parse_args(argv)
    run(a.rectify, a.output, repo=a.repo, checkpoint=a.checkpoint, python=a.python,
        iterations=a.iterations, max_disp=a.max_disp, volume_builder=a.volume_builder,
        confidence_mode=a.confidence_mode, max_depth_mm=a.max_depth_mm, warmup=a.warmup,
        worker_timeout_s=a.worker_timeout_s, resume=a.resume, force=a.force)


if __name__ == '__main__':
    main()
