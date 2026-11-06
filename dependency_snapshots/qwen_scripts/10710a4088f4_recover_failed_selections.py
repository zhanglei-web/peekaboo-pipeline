"""Prefill only previously failed Astra selections while the original GPU batch runs."""
import argparse
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys

import psutil
from export_research_fields import atomic, read

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output.resolve()
    lock = (out/'.selection_recovery.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = psutil.Process()
    state = {'status':'running', 'pid':process.pid, 'create_time':process.create_time(),
             'command':process.cmdline(), 'episodes':{}}
    eligible = set(read(out/'research_dataset_protocol.json')['episode_ids'])
    initial = read(out/'batch_status.json')
    candidates = [eid for eid,v in initial['episodes'].items()
                  if eid in eligible and v.get('status')=='failed' and v.get('stage')=='selection']
    def update():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        atomic(out/'selection_recovery_status.json',state)
    update()
    try:
        for eid in candidates:
            current = read(out/'batch_status.json')
            # The normal supervisor owns subsequent batch retries. Finish an in-flight
            # selection, but do not start more after the original batch has drained.
            if current['status'] != 'running':
                break
            folder = out/'episodes'/eid
            if (folder/'selection/selected_frames.json').exists():
                state['episodes'][eid] = {'status':'already_cached'}; update(); continue
            if current['episodes'][eid].get('status') != 'failed':
                continue
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            evidence = out/'retry_evidence'/('selection_prefill_'+stamp)/eid
            evidence.mkdir(parents=True)
            for file in (folder/'selection').rglob('*'):
                if file.is_file() and file.suffix in ['.json','.txt','.log']:
                    destination = evidence/file.relative_to(folder/'selection')
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(file,destination)
            state['episodes'][eid] = {'status':'running','evidence':str(evidence)}; update()
            command = [sys.executable,str(ROOT/'scripts/select_scene_keyframes.py'),
                       '--input',str(folder/'input'),'--output',str(folder/'selection')]
            with (folder/'selection.log').open('a') as log:
                result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            state['episodes'][eid].update(status='cached' if result.returncode==0 else 'failed',
                                          returncode=result.returncode)
            update()
        state['status'] = 'finished'; update()
    except BaseException as exc:
        state.update(status='needs_attention',error_type=type(exc).__name__); update()
        raise


if __name__ == '__main__':
    main()
