"""Host GPU ownership covers depth jobs and the full local model lifetime."""
from contextlib import contextmanager
from pathlib import Path
import os
import socket
import subprocess
import sys
import threading
import time
import psutil
import requests
from .common import QWEN, alive, atomic, identity, lock, now, read

class ExternalResourceBusy(RuntimeError): pass

def check_bind(host,port):
    # Mirror the HTTP server's listener policy: TIME_WAIT from our previous
    # graceful shutdown must not look like a live foreign listener.
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        s.bind((host,port))

def gpu_snapshot():
    result = subprocess.run(['nvidia-smi','--query-gpu=index,memory.free',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free = {int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in result.stdout.splitlines())}
    processes = subprocess.run(['nvidia-smi','--query-compute-apps=pid',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    pids = [int(line.strip()) for line in processes.stdout.splitlines() if line.strip().isdigit()]
    return free,pids

class GPU:
    def __init__(self, config, root):
        self.config = config; self.root = Path(root)
        self.mutex = threading.Lock(); self.proc = None; self.record = None

    @contextmanager
    def lease(self, kind, minimum_free):
        with self.mutex, lock(self.config['lock_path']):
            free,pids = gpu_snapshot()
            if pids or free[self.config.get('device',0)] < minimum_free:
                raise ExternalResourceBusy('GPU occupied; no foreign process will be stopped')
            atomic(self.root/'gpu_owner.json', {**identity(), 'kind':kind, 'acquired_at':now(),
                   'snapshot_free_mib':free, 'lease_scope':'held OS lock; expiry is not release'})
            try: yield
            finally:
                # A surviving process keeps the whole pipeline in a blocked state;
                # no later job can treat an expired timestamp as GPU release.
                free,pids = gpu_snapshot()
                atomic(self.root/'gpu_owner.json', {**identity(), 'kind':'released' if not pids else 'external_or_unreleased',
                       'at':now(), 'remaining_compute_pids':pids, 'snapshot_free_mib':free})

    @contextmanager
    def local_server(self, profile):
        with self.lease('local_qwen_residency',self.config['local_min_free_mib']):
            sys.path.insert(0,str(QWEN/'src'))
            from local_vlm.server import command,environment
            cfg = read(self.config['local_server_config']); host = cfg['server']['host']; port = cfg['server']['port']
            if cfg['server']['alias'] != profile['model'] or profile['base_url'] != f'http://{host}:{port}/v1':
                raise ValueError('Local endpoint and server configuration differ')
            # Check the entire existing project process namespace, not just GPU
            # utilization. Never attach to, stop or take ownership of an old service.
            for p in psutil.process_iter(['name','cmdline']):
                if 'llama-server' in (p.info['name'] or ''):
                    raise ExternalResourceBusy('An existing llama-server is not owned by 11-5')
            check_bind(host,port)
            cmd = command(cfg); self.root.mkdir(parents=True,exist_ok=True)
            with (self.root/'local_server.log').open('a') as log:
                self.proc = subprocess.Popen(cmd,cwd=QWEN,env=environment(),stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            self.record = identity(self.proc.pid)
            atomic(self.root/'local_server_identity.json',self.record)
            session = requests.Session(); session.trust_env = False
            try:
                deadline = time.monotonic()+cfg['server']['startup_timeout']
                while time.monotonic() < deadline:
                    if self.proc.poll() is not None: raise RuntimeError('Owned local server exited during startup')
                    try:
                        r = session.get(f'http://{host}:{port}/health',timeout=2)
                        if r.ok and r.json().get('status') == 'ok': break
                    except (requests.RequestException,ValueError): pass
                    time.sleep(1)
                else: raise TimeoutError('Owned local server startup timeout')
                yield
            finally:
                if self.proc.poll() is None:
                    if not alive(self.record): raise RuntimeError('Server process identity changed; refusing signal')
                    # Caller has no active request when exiting this context. If
                    # a timed-out HTTP call might still run, /slots prevents unload.
                    slots = session.get(f'http://{host}:{port}/slots',timeout=10,allow_redirects=False)
                    if not slots.ok or any(s.get('is_processing') for s in slots.json()):
                        raise RuntimeError('Local server not confirmed idle; retained; depth remains blocked')
                    self.proc.terminate()
                    try: self.proc.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        raise RuntimeError('Owned server did not exit; no forced kill; depth blocked') from None
                atomic(self.root/'local_server_stopped.json',{'identity':self.record,'stopped_at':now()})
