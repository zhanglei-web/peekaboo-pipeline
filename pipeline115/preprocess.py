"""Unchanged CPU/depth runners with isolated, policy-checked recovery."""
from copy import deepcopy
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time
from .common import CORE, RECOVERY, atomic, bind, identity, now, read, sha

def command(config, through, root, lane, runner=CORE):
    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(Path(runner)/'src')+os.pathsep+env.get('PYTHONPATH',''),
        ROS_DOMAIN_ID=str(141+lane),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    cmd = [sys.executable,'-u','-m','daaam.preprocessing.runner','--config',str(config),
           '--through',through,'--ros-mode','none','--resume']
    path = root/(through+'-'+str(time.time_ns())+'.log')
    with path.open('w') as log:
        proc = subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(root/'worker.json',{**identity(proc.pid),'through':through,'log':str(path)})
        try: code = proc.wait()
        except BaseException:
            # Do not orphan VIO/depth children when an operator interrupts.
            import signal
            os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=60); raise
    return code,path

def configure(record, root):
    cfg = deepcopy(read(record['config']))
    configured_source=Path(cfg['source'])
    if not configured_source.is_absolute():configured_source=Path(record['config']).parent/configured_source
    if configured_source.resolve()!=Path(record['source']).resolve():
        raise ValueError('Preprocessing config belongs to another capture')
    cfg['output'] = str(Path(root)/'preprocessed'/record['id'])
    cfg['ros_output'] = {**cfg.get('ros_output',{}),'enabled':False,'mode':'none'}
    path = Path(root)/'configs'/(record['id']+'.json'); bind(path,cfg)
    return path

def cpu(record, root, lane):
    root = Path(root); effective = configure(record,root); logs = root/'logs'/record['id']
    recovery_file = logs/'recovery.json'; runner = CORE
    if recovery_file.exists():
        audit = read(recovery_file); effective = Path(audit['effective_config']); runner = Path(audit['runner'])
        if sha(runner/'scripts/run_ego_openvins.py') != audit['policy']['modified_helper_sha256']:
            raise ValueError('Recovery runner changed')
    code,diagnostic = command(effective,'rectify',logs,lane,runner)
    if code and runner == CORE and 'pose covariance is materially indefinite' in diagnostic.read_text(errors='replace'):
        policy = read(RECOVERY/'recovery_policy.json')
        if policy['policy'] != 'exclude_invalid_covariance; preserve_original_threshold; prohibit_interpolation_across_rejections':
            raise ValueError('Unexpected covariance recovery policy')
        if sha(RECOVERY/'scripts/run_ego_openvins.py') != policy['modified_helper_sha256']:
            raise ValueError('Frozen covariance helper changed')
        if sha(CORE/'scripts/run_ego_openvins.py') != policy['original_helper_sha256']:
            raise ValueError('Original covariance helper changed')
        runner = root/'frozen_runners'/record['id']
        if not runner.exists(): shutil.copytree(RECOVERY,runner)
        cfg = read(effective); cfg['output'] = str(root/'preprocessed_recovery'/record['id'])
        recovery_config = root/'recovery_configs'/(record['id']+'.json'); bind(recovery_config,cfg)
        atomic(recovery_file,{'original_config':str(effective),'original_log':str(diagnostic),
            'original_exit_code':code,'effective_config':str(recovery_config),'runner':str(runner),
            'policy':policy,'at':now()})
        effective = recovery_config; code,diagnostic = command(effective,'rectify',logs,lane,runner)
    if code: raise RuntimeError('CPU preprocessing failed; inspect '+str(diagnostic))
    return {'config':str(effective),'runner':str(runner),'output':read(effective)['output']}

def depth(prepared, root, episode_id):
    code,path = command(prepared['config'],'depth',Path(root)/'logs'/episode_id,2,prepared['runner'])
    if code: raise RuntimeError('Depth preprocessing failed; inspect '+str(path))
    return prepared['output']
