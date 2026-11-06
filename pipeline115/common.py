"""Atomic receipts, content identities and operating-system ownership locks."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import fcntl
import hashlib
import json
import os
import tempfile
import psutil

ROOT = Path(__file__).resolve().parents[1]
CORE = Path('/home/baai/zl/egomemory_ws/src/egomemory-core')
QWEN = Path('/home/baai/zl/Qwen3.8-27B')
RECOVERY = Path('/home/baai/zl/egomemory_ws/outputs/todo_search_episodes_20261103/frozen_runners/covariance_filter')

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def bind(path, value):
    """Never reinterpret old successful evidence after configuration changes."""
    path = Path(path)
    if path.exists():
        if read(path) != value: raise ValueError('Immutable input changed: '+str(path))
    else: atomic(path, value)

def identity(pid=None):
    p = psutil.Process(pid or os.getpid())
    return {'pid':p.pid, 'create_time':p.create_time(), 'command':p.cmdline()}

def alive(record):
    try:
        p = psutil.Process(record['pid'])
        return (abs(p.create_time()-record['create_time']) < .01 and
                p.cmdline() == record['command'] and p.status() != psutil.STATUS_ZOMBIE)
    except (psutil.Error, KeyError): return False

@contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream

def code_identity():
    files = [ROOT/'pipeline_11_5.py', ROOT/'annotate_batch.py', ROOT/'annotation_contract.py',
             ROOT/'audit_inputs.py', ROOT/'export_confirmed.py', ROOT/'dependencies.lock.json']
    files += sorted((ROOT/'pipeline115').glob('*.py'))
    files += sorted((QWEN/'src/local_vlm').glob('*.py'))
    return {str(p):sha(p) for p in files}

def verify_dependencies():
    failed = [r['path'] for r in read(ROOT/'dependencies.lock.json')['code']
              if not Path(r['path']).is_file() or sha(r['path']) != r['sha256']]
    if failed: raise ValueError('Frozen dependencies changed: '+repr(failed))

def tree(path):
    return {str(p.resolve()):sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}
