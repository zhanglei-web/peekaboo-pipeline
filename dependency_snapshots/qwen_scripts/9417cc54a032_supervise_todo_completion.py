"""Wait for the active batch, then resume incomplete episodes without changing annotations."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import jsonschema
import psutil

from export_research_fields import atomic, read, sha, verify

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def matching_process(record, script, out):
    try:
        p = psutil.Process(record['pid'])
        if abs(p.create_time()-record['create_time']) > .01 or p.status() == psutil.STATUS_ZOMBIE:
            return None
        command = p.cmdline()
        if str(ROOT/'scripts'/script) not in command or str(out) not in command:
            raise RuntimeError('Recorded process identity mismatch')
        return p
    except psutil.NoSuchProcess:
        return None


def preserve(out, status, attempt):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    evidence = out/'retry_evidence'/f'{stamp}_attempt_{attempt:02d}'
    evidence.mkdir(parents=True)
    for name in ['batch_status.json', 'launch.json', 'timing_summary_target_v1.json',
                 'batch.log', 'research_export_status.json']:
        if (out/name).exists():
            shutil.copy2(out/name, evidence/name)
    for eid, entry in status['episodes'].items():
        if entry.get('status') == 'complete':
            continue
        folder = out/'episodes'/eid
        for source in folder.rglob('*'):
            if source.is_file() and source.suffix in ['.json', '.jsonl', '.txt', '.log']:
                target = evidence/'episodes'/eid/source.relative_to(folder)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    return str(evidence)


def validate_final(out):
    ids = read(out/'research_dataset_protocol.json')['episode_ids']
    schema = read(out/'research_target.schema.json')
    validator = jsonschema.Draft202012Validator(schema)
    for eid in ids:
        path = out/'research_json'/(eid+'.json')
        receipt = read(out/'episodes'/eid/'research_fields_complete.json')
        if (receipt['sha256'] != sha(path) or receipt['source_sha256'] != sha(out/'json'/(eid+'.json'))
                or receipt['protocol_sha256'] != sha(out/'research_dataset_protocol.json')):
            raise ValueError('Research completion hash mismatch: '+eid)
        doc = read(path)
        validator.validate(doc)
        verify(doc)
    return len(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    lock = (out/'.completion_supervisor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = psutil.Process()
    state = {'status': 'watching_current_batch', 'started_at': now(), 'retry_rounds': 0,
             'stagnant_rounds': 0, 'pid': process.pid, 'create_time': process.create_time(),
             'command': process.cmdline(), 'last_completed_count': read(out/'batch_status.json').get('completed_count', 0)}

    def update(**values):
        state.update(values, updated_at=now())
        atomic(out/'completion_supervisor_status.json', state)

    def launch(script, logfile, extra=()):
        command = [sys.executable, '-u', str(ROOT/'scripts'/script), '--output', str(out), *extra]
        with (out/logfile).open('a') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        time.sleep(2)
        if child.poll() is not None:
            raise RuntimeError(f'{script} exited at startup: {child.returncode}')
        return child.pid

    try:
        update()
        while True:
            status = read(out/'batch_status.json')
            export = read(out/'research_export_status.json')
            if export['status'] == 'needs_review':
                update(status='needs_attention', reason='Research exporter reported validation or protected-output conflict')
                return
            if export['status'] == 'complete':
                count = validate_final(out)
                update(status='complete', validated_research_outputs=count, finished_at=now())
                return
            exporter = matching_process(read(out/'research_export_launch.json'), 'export_research_fields.py', out)
            if exporter is None:
                update(exporter_restarted_pid=launch('export_research_fields.py', 'research_export.log', ['--watch']))
            batch = matching_process(read(out/'launch.json'), 'run_scene_event_batch.py', out)
            if batch is not None:
                update(status='watching_current_batch', active_batch_pid=batch.pid,
                       visual_completed_count=status.get('completed_count', 0),
                       research_completed_count=export['completed_count'])
                time.sleep(20)
                continue
            if status['status'] == 'complete':
                time.sleep(20)  # Allow the exporter to consume the last completed visual output.
                continue
            completed = status.get('completed_count', 0)
            stagnant = state['stagnant_rounds'] + 1 if completed <= state['last_completed_count'] else 0
            if stagnant >= 3:
                update(status='needs_attention', stagnant_rounds=stagnant,
                       reason='Three retry rounds made no progress; investigate preserved evidence before further requests')
                return
            attempt = state['retry_rounds'] + 1
            evidence = preserve(out, status, attempt)
            update(status='retry_backoff', retry_rounds=attempt, stagnant_rounds=stagnant,
                   last_completed_count=completed, evidence=evidence)
            time.sleep(min(30 * attempt, 300))
            # Re-check launch identity after waiting, so a separate recovery is adopted rather than duplicated.
            if matching_process(read(out/'launch.json'), 'run_scene_event_batch.py', out) is None:
                pid = launch('run_scene_event_batch.py', 'batch.log')
                update(status='watching_current_batch', active_batch_pid=pid)
    except BaseException as exc:
        update(status='needs_attention', reason=repr(exc), finished_at=now())
        raise


if __name__ == '__main__':
    main()
