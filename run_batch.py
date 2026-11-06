#!/usr/bin/env python3
"""Recursive stereo/IMU batch: two CPU lanes and one GPU lane, existing algorithms."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import collections
import fcntl
import json
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
CORE = Path('/home/baai/zl/egomemory_ws/src/egomemory-core')
sys.path.insert(0, str(CORE / 'scripts'))
from batch_ego_preprocessing import export_bag, verify_bag

RECOVERY_SOURCE = Path('/home/baai/zl/egomemory_ws/outputs/todo_search_episodes_20261103/frozen_runners/covariance_filter')


def covariance_recovery(record):
    """Only a diagnosed indefinite covariance permits this isolated recovery."""
    path = Path(record.get('log', '/nonexistent'))
    diagnostic = path.read_text(errors='replace') if path.is_file() else ''
    if record.get('runner') or 'pose covariance is materially indefinite' not in diagnostic:
        return False
    runner = ROOT / 'frozen_runners/covariance_filter'
    if not runner.exists():
        # Called at startup before worker threads, or serialized by recovery.lock.
        shutil.copytree(RECOVERY_SOURCE, runner)
    policy = json.loads((runner/'recovery_policy.json').read_text())
    helper = runner/'scripts/run_ego_openvins.py'
    if hashlib.sha256(helper.read_bytes()).hexdigest() != policy['modified_helper_sha256']:
        raise ValueError('Recovery helper differs from audited policy')
    original = {k: record.get(k) for k in ('output', 'config', 'status', 'error', 'log')}
    output = ROOT.parent/'preprocessed_recovery'/record['id']
    config = json.loads(Path(record['config']).read_text())
    config['output'] = str(output)
    effective = ROOT/'recovery_configs'/(record['id']+'.json')
    save(effective, config)
    record.update(original_attempt=original, output=str(output), effective_config=str(effective),
                  runner=str(runner), status='queued', recovery_started_at=now(),
                  recovery_policy=policy['policy'])
    save(ROOT/'recovery_audit'/(record['id']+'.json'), record)
    return True


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


class Batch:
    def __init__(self):
        self.inventory = json.loads((ROOT / 'inventory.json').read_text())
        previous = ROOT / 'state.json'
        self.state = json.loads(previous.read_text()) if previous.exists() else {
            'records': {r['id']: deepcopy(r) for r in self.inventory['records']}, 'created_at': now()}
        for record in self.state['records'].values():
            if record['status'] == 'cpu_failed':
                covariance_recovery(record)
        self.lock = threading.Lock()
        self.processes = {}
        self.stop = threading.Event()
        self.state.update(pid=os.getpid(), status='running', started_at=now(), runner_version='11.4.same_preprocessing_parameters.v1')
        self.state.pop('finished_at', None)

    def update(self, record_id=None, **changes):
        with self.lock:
            if record_id is None:
                self.state.update(changes)
            else:
                self.state['records'][record_id].update(changes, updated_at=now())
            self.state['updated_at'] = now()
            self.state['counts'] = dict(collections.Counter(r['status'] for r in self.state['records'].values()))
            save(ROOT / 'state.json', self.state)

    def command(self, record, through, lane):
        config = Path(record.get('effective_config', record['config']))
        environment = os.environ.copy()
        runner = Path(record.get('runner', str(CORE)))
        environment['PYTHONPATH'] = str(runner / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
        environment['ROS_DOMAIN_ID'] = str(111 + lane)
        environment['OMP_NUM_THREADS'] = '2'
        environment['OPENBLAS_NUM_THREADS'] = '1'
        environment['MKL_NUM_THREADS'] = '2'
        environment['PYTHONUNBUFFERED'] = '1'
        command = [sys.executable, '-u', '-m', 'daaam.preprocessing.runner', '--config', str(config),
                   '--through', through, '--ros-mode', 'bag', '--resume']
        logs = ROOT / 'logs' / record['id']
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / (through + '-' + str(time.time_ns()) + '.log')
        with path.open('w') as log:
            if self.stop.is_set():
                return -15, path
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                    env=environment, start_new_session=True)
            with self.lock:
                self.processes[proc.pid] = proc
            self.update(record['id'], worker_pid=proc.pid, log=str(path))
            code = proc.wait()
            with self.lock:
                self.processes.pop(proc.pid, None)
        return code, path

    def cpu(self, record, lane):
        self.update(record['id'], status='cpu_running', cpu_started_at=now(), cpu_lane=lane)
        try:
            if record['missing_inputs']:
                raise ValueError('Missing required input files')
            code, log = self.command(record, 'rectify', lane)
            if code != 0:
                with (ROOT/'recovery.lock').open('a') as recovery_lock:
                    fcntl.flock(recovery_lock, fcntl.LOCK_EX)
                    with self.lock:
                        recovered = covariance_recovery(record) if not self.stop.is_set() else False
                if recovered:
                    self.update(record['id'], **{k:v for k,v in record.items() if k != 'id'})
                    code, log = self.command(record, 'rectify', lane)
                if code != 0:
                    raise RuntimeError('CPU pipeline exited ' + str(code) + '; see ' + str(log))
            output = Path(record['output'])
            report = json.loads((output / 'select/selection_report.json').read_text())
            vio = json.loads((output / 'vio/native/vio_acceptance.json').read_text())
            self.update(record['id'], status='waiting_gpu', cpu_finished_at=now(),
                        selected_frames=report['selected_count'], vio_acceptance_passed=vio['passed'],
                        vio_failed_gates=[k for k, v in vio['gates'].items() if not v])
            print('CPU_READY', record['id'], report['selected_count'], flush=True)
            return record
        except Exception as exc:
            self.update(record['id'], status='stopped' if self.stop.is_set() else 'cpu_failed',
                        error=str(exc), error_type=type(exc).__name__)
            print('CPU_FAILED', record['id'], str(exc), flush=True)
            return None

    def gpu(self, record):
        self.update(record['id'], status='depth_running', depth_started_at=now())
        try:
            code, log = self.command(record, 'ros-output', 2)
            if code != 0:
                raise RuntimeError('Depth/ROS pipeline exited ' + str(code) + '; see ' + str(log))
            self.update(record['id'], status='verifying')
            bag = Path(record['bag'])
            bag.parent.mkdir(parents=True, exist_ok=True)
            exported, verification = export_bag(Path(record['output']), bag.parent, bag.name)
            self.update(record['id'], status='complete', completed_at=now(), bag=str(exported),
                        verification=verification,
                        bag_size_bytes=sum(p.stat().st_size for p in exported.iterdir() if p.is_file()))
            print('BAG_COMPLETE', record['id'], verification['frames'], str(exported), flush=True)
        except Exception as exc:
            self.update(record['id'], status='stopped' if self.stop.is_set() else 'depth_or_export_failed',
                        error=str(exc), error_type=type(exc).__name__)
            print('GPU_FAILED', record['id'], str(exc), flush=True)

    @staticmethod
    def gpu_free():
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                    text=True, capture_output=True, timeout=10, check=True)
            return int(result.stdout.splitlines()[0])
        except (OSError, ValueError, subprocess.SubprocessError):
            return None

    def terminate(self, *_):
        self.stop.set()
        with self.lock:
            active = list(self.processes.values())
        for proc in active:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def run(self):
        self.update()
        pending = []
        ready = collections.deque()
        for record in sorted(self.state['records'].values(), key=lambda r: (r.get('duration_s') or 0, r['id'])):
            # Never silently rerun a failed capture. Its diagnostics remain reviewable.
            if record['status'] == 'complete':
                verify_bag(Path(record['bag']), Path(record['output']))
                continue
            if record['status'] in {'cpu_failed', 'depth_or_export_failed'}:
                continue
            if (Path(record['output']) / 'rectify/stage.json').is_file():
                ready.append(record)
                self.update(record['id'], status='waiting_gpu')
            else:
                pending.append(record)
        cpu_tasks = {}
        gpu_task = None
        last_poll = 0
        with ThreadPoolExecutor(max_workers=2) as cpu_pool, ThreadPoolExecutor(max_workers=1) as gpu_pool:
            while pending or cpu_tasks or ready or gpu_task:
                if self.stop.is_set():
                    break
                free_lanes = [i for i in range(2) if i not in [entry[1] for entry in cpu_tasks.values()]]
                for lane in free_lanes:
                    if pending:
                        record = pending.pop(0)
                        cpu_tasks[cpu_pool.submit(self.cpu, record, lane)] = (record, lane)
                for future in list(cpu_tasks):
                    if future.done():
                        record = future.result()
                        cpu_tasks.pop(future)
                        if record is not None:
                            ready.append(record)
                if gpu_task and gpu_task.done():
                    gpu_task.result()
                    gpu_task = None
                if time.monotonic() - last_poll >= 15:
                    free = self.gpu_free()
                    self.update(gpu_free_mib=free, cpu_active=len(cpu_tasks), gpu_active=int(gpu_task is not None),
                                gpu_waiting_count=len(ready), cpu_queued_count=len(pending),
                                gpu_status='running' if gpu_task else 'waiting_for_free_memory' if ready else 'idle')
                    last_poll = time.monotonic()
                    if (ready and not gpu_task and free is not None and free >= self.inventory['gpu_min_free_mib']
                            and (ROOT/'gpu_release_authorized.json').exists()):
                        gpu_task = gpu_pool.submit(self.gpu, ready.popleft())
                self.stop.wait(1)
        counts = collections.Counter(r['status'] for r in self.state['records'].values())
        self.update(status='stopped_by_user' if self.stop.is_set() else
                    'complete' if counts['complete'] == self.inventory['record_count'] else 'finished_with_failures',
                    finished_at=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with (ROOT / 'batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = Batch()
        signal.signal(signal.SIGTERM, batch.terminate)
        signal.signal(signal.SIGINT, batch.terminate)
        batch.run()


if __name__ == '__main__':
    main()
