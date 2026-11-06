from __future__ import annotations

import hashlib
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 'ego.preprocess.v1'
STAGES = ('audit', 'vio', 'select', 'rectify', 'depth')


def legacy(name):
    """Reuse repository helpers without invoking their isolated legacy CLIs."""
    scripts = str(ROOT / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module(name)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def transform_units(matrix, factor):
    result = np.array(matrix, dtype=float, copy=True)
    legacy('run_ego_openvins').validate_homogeneous_transform(result, 'transform')
    result[:3, 3] *= factor
    return result


def covariance_units(covariance, factor):
    result = np.asarray(covariance, dtype=float).reshape(6, 6)
    scale = np.diag([factor] * 3 + [1.] * 3)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite covariance')
    return scale @ result @ scale.T


def camera_covariance(world_T_imu, T_imu_camera, covariance):
    """Propagate a world-frame pose covariance through the camera lever arm."""
    pose = np.asarray(world_T_imu, dtype=float)
    extrinsic = np.asarray(T_imu_camera, dtype=float)
    covariance = np.asarray(covariance, dtype=float).reshape(6, 6)
    if pose.shape != (4, 4) or extrinsic.shape != (4, 4):
        raise ValueError('pose transforms must be 4x4')
    if not np.isfinite(pose).all() or not np.isfinite(extrinsic).all() or not np.isfinite(covariance).all():
        raise ValueError('pose covariance inputs must be finite')
    arm_world = pose[:3, :3] @ extrinsic[:3, 3]
    x, y, z = arm_world
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    jacobian = np.eye(6)
    jacobian[:3, 3:] = -skew
    result = jacobian @ covariance @ jacobian.T
    return (result + result.T) * 0.5


def tree_fingerprint(path):
    path = Path(path)
    return {str(p.relative_to(path)): sha256(p) for p in sorted(path.rglob('*'))
            if p.is_file() and p.name != 'stage.json'}


def verify_stage(path):
    path = Path(path)
    state = read_json(path / 'stage.json')
    if state.get('status') != 'complete' or state['artifacts'] != tree_fingerprint(path):
        raise ValueError(f'stage incomplete or artifacts changed: {path}; use --force to regenerate')
    return state


def execute_stage(name, config, output, dependencies, action, *, resume=False, force=False, extra=None):
    """Publish complete directories atomically; failed runs remain inspectable."""
    output = Path(output)
    deps = {str(Path(dep).resolve()): verify_stage(dep) for dep in dependencies}
    code = {str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / 'src/daaam/preprocessing').glob('*.py'))}
    for filename in ('run_ego_openvins.py', 'select_ego_keyframes.py',
                     'prepare_ego_stereo_dataset.py', 'fast_foundation_stereo_worker.py',
                     'run_fast_foundation_stereo_depth.py'):
        code['scripts/' + filename] = sha256(ROOT / 'scripts' / filename)
    signature = digest_json({'name': name, 'config': config, 'deps': deps, 'code': code, 'extra': extra})
    if output.exists():
        if resume and not force:
            state = verify_stage(output)
            if state['signature'] != signature:
                raise ValueError(f'{name}: inputs/config/code changed; use --force')
            print(f'{name}: validated existing output {output}', flush=True)
            return state
        if not force:
            raise FileExistsError(f'{output} exists; use --resume or --force')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=output.parent))
    started = time.time_ns()
    try:
        result = action(temporary)
        state = {'schema': SCHEMA, 'stage': name, 'status': 'complete', 'signature': signature,
                 'started_time_ns': started, 'completed_time_ns': time.time_ns(),
                 'config': config, 'result': result, 'artifacts': tree_fingerprint(temporary)}
        write_json(temporary / 'stage.json', state)
        if output.exists():
            backup = output.with_name(output.name + f'.previous-{started}')
            output.rename(backup)
        temporary.rename(output)
        print(f'{name}: complete {output}', flush=True)
        return state
    except BaseException as error:
        write_json(temporary / 'failure.json', {'stage': name, 'error': repr(error), 'started_time_ns': started})
        print(f'{name}: failed; diagnostics retained in {temporary}', file=sys.stderr, flush=True)
        raise


class VideoReader:
    """Decode independently and monotonically; never seek by nominal frame rate."""
    def __init__(self, path):
        import cv2
        self.capture = cv2.VideoCapture(str(path))
        self.index = -1
        if not self.capture.isOpened():
            raise ValueError(f'cannot open video {path}')

    def get(self, index):
        if not isinstance(index, int) or index <= self.index:
            raise ValueError('video indices must increase strictly')
        image = None
        while self.index < index:
            ok, image = self.capture.read()
            self.index += 1
            if not ok:
                raise ValueError(f'video decode ended before frame {index}')
        return image

    def close(self):
        self.capture.release()
