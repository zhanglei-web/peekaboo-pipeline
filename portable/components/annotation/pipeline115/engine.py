"""Bounded episode pipeline with persistent stages and a single GPU controller."""
from pathlib import Path
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / 'repository.marker').is_file())
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import psutil
from .common import ROOT, atomic, bind, code_identity, digest, identity, lock, now, read, sha, tree, verify_dependencies
from . import direct, preprocess
from . import dispatch
from .resources import ExternalResourceBusy, GPU

STEPS = ('cpu','depth','direct','astra','scene','events','temporal','assemble')

def load_config(path):
    cfg = read(path)
    if cfg['version'] != '11-5': raise ValueError('Expected 11-5 configuration')
    output = Path(cfg['output_root']).resolve()
    protected = [ROOT.parent,Path(str(REPOSITORY / 'legacy/zl/todo/11.3')),Path(str(REPOSITORY / 'legacy/11-4-81')),Path(str(REPOSITORY / 'legacy/11-5-29'))]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError('New output root must not overlap historical data')
    cfg['output_root'] = str(output)
    for key,value in cfg['limits'].items():
        if not isinstance(value,(int,float)) or isinstance(value,bool) or value <= 0:
            raise ValueError('Expected positive resource limit: '+key)
    if cfg['limits']['cpu_workers'] > 2: raise ValueError('Initial CPU concurrency limit is two')
    if cfg['limits']['depth_group_size'] > cfg['limits']['queue_episodes']:
        raise ValueError('Depth group cannot exceed bounded queue')
    dispatch.validate_config(cfg)
    profiles = [(stage, cfg['routes'][stage]) for stage in dispatch.MODEL_STAGES]
    if dispatch.episode_mode(cfg):
        profiles += [(stage,key) for key in dispatch.endpoint_keys(cfg) for stage in dispatch.MODEL_STAGES]
    for stage,key in profiles:
        profile = cfg['endpoints'][key]
        if profile['placement'] not in ('local_gpu','remote'): raise ValueError('Unknown endpoint placement')
        allowed={'name','base_url','model','context_tokens','placement','api_key_env','quantization','timeout_s'}
        if set(profile)-allowed: raise ValueError('Unsupported endpoint fields; credentials only through api_key_env')
        if profile['context_tokens'] <= {'scene':10000,'events':6144,'temporal':3072}[stage]:
            raise ValueError('Unchanged '+stage+' output budget exceeds endpoint context')
    local = [p for p in cfg['endpoints'].values() if p['placement']=='local_gpu']
    if len(local)>1: raise ValueError('One local server supported; independent URLs do not imply independent GPUs')
    if any(k in str(cfg).lower() for k in ('bearer ', 'api_key_value')):
        raise ValueError('Credentials must be environment variables, not config values')
    return cfg

def audit(row):
    from audit_inputs import check
    from .location_facts import verify
    verify(row)
    if row.get('problems'): raise ValueError('Input mapping has unresolved problems')
    checks = check(row)
    value = deepcopy(row); value.pop('source_bag',None); value.pop('preprocessed',None)
    value['raw_video_image_checks'] = checks
    value['source_file_hashes'] = tree(row['source'])
    if any(x['decoded_minus_filename'] != 0 for x in checks):
        value['frame_index_warning'] = 'Filename labels differ from decoded indices; original labels and label/fps times preserved. See input audit.'
    return value

def worker(stage,cfg,row_path,log_root):
    log_root.mkdir(parents=True,exist_ok=True)
    path = log_root/(stage+'-'+str(time.time_ns())+'.log')
    cmd = [sys.executable,'-u','-m','pipeline115.worker',stage,str(cfg),str(row_path)]
    env = os.environ.copy(); env['PYTHONPATH'] = str(ROOT)+os.pathsep+env.get('PYTHONPATH','')
    with path.open('w') as log:
        proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        atomic(log_root/'worker.json',{**identity(proc.pid),'stage':stage,'log':str(path)})
        try: code = proc.wait()
        except BaseException:
            import signal
            os.killpg(proc.pid,signal.SIGTERM); proc.wait(timeout=60); raise
    if code == 75: return False
    if code: raise RuntimeError(stage+' failed; inspect '+str(path))
    return True

class Engine:
    def __init__(self,cfg,only=None,reuse=None,retry_failed=False):
        self.cfg = cfg; self.root = Path(cfg['output_root']); self.root.mkdir(parents=True,exist_ok=True)
        annotations = read(cfg['annotation_inventory'])['episodes']
        records = {r['id']:r for r in read(cfg['preprocessing_inventory'])['records']}
        if only:
            unknown = set(only)-{r['episode_id'] for r in annotations}
            if unknown: raise ValueError('Unknown episodes: '+repr(unknown))
            annotations = [r for r in annotations if r['episode_id'] in only]
        if not annotations: raise ValueError('No episodes selected')
        for row in annotations:
            eid=row['episode_id']
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',eid) or '..' in eid:
                raise ValueError('Unsafe episode ID')
            source=Path(row['source']).resolve()
            if source==self.root or source in self.root.parents or self.root in source.parents:
                raise ValueError('Output root overlaps original capture')
        self.rows = {r['episode_id']:r for r in annotations}; self.records = records
        self.reuse = read(reuse)['records'] if reuse else None
        self.mutex = threading.RLock(); self.stop = threading.Event()
        manifest = {'config':cfg,'code':code_identity(),'astra_config_sha256':sha(cfg['astra_config']),
            'local_server_config_sha256':sha(cfg['gpu']['local_server_config']),
            'preprocessing_configs':{eid:sha(records[eid]['config']) for eid in self.rows},
            'inventory_sha256':sha(cfg['annotation_inventory']),
            'preprocessing_inventory_sha256':sha(cfg['preprocessing_inventory']),
            'facts_sha256':sha(cfg['facts']) if cfg.get('facts') else None,
            'episode_ids':list(self.rows),'reuse_preprocessed_state_sha256':sha(reuse) if reuse else None}
        legacy = self.root/'legacy_completed_episodes.json'
        if legacy.exists(): manifest['legacy_completed_episodes_sha256'] = sha(legacy)
        legacy_configs=self.root/'legacy_episode_configs.json'
        if legacy_configs.exists(): manifest['legacy_episode_configs_sha256']=sha(legacy_configs)
        bind(self.root/'run_manifest.json',manifest); bind(self.root/'config.json',cfg)
        self.state_path = self.root/'state.json'
        self.state = read(self.state_path) if self.state_path.exists() else {'episodes':{eid:{'next':'cpu','status':'queued','stages':{}} for eid in self.rows}}
        for eid,s in self.state['episodes'].items():
            if s['status'] == 'running':
                # OS root lock has already been acquired; verify no orphan stage
                # subprocess before allowing another writer to this episode.
                from .common import alive
                for p in (self.root/'logs'/eid).glob('worker.json'):
                    if alive(read(p)): raise RuntimeError('Prior stage subprocess remains alive: '+eid)
                s['status'] = 'queued'
            if retry_failed and s['status'] in ('needs_repair','waiting_external_resource'):
                s['status'] = 'queued'; s.pop('error',None)
            for receipt_path in sorted((self.root/'receipts'/eid).glob('*.json')):
                receipt = read(receipt_path)
                for file,h in receipt['outputs'].items():
                    if sha(file) != h: raise ValueError('Completed stage output changed: '+file)
            # Recover a crash between durable receipt publication and state
            # commit without resubmitting an already successful stage.
            while s['next'] in STEPS:
                checkpoint=self.root/'receipts'/eid/(s['next']+'.json')
                if not checkpoint.exists(): break
                receipt=read(checkpoint)
                if receipt['status']!='passed' or receipt['stage']!=s['next'] or receipt['episode_id']!=eid:
                    raise ValueError('Invalid stage checkpoint')
                s['stages'][s['next']]=receipt
                index=STEPS.index(s['next'])+1
                s.update(next=STEPS[index] if index<len(STEPS) else 'complete',
                         status='queued' if index<len(STEPS) else 'complete')
            if dispatch.episode_mode(cfg):
                owner = dispatch.recover_assignment(self.root,cfg,eid)
                if owner:
                    if s.get('owner',owner['endpoint']) != owner['endpoint']:
                        raise ValueError('State owner differs from durable assignment: '+eid)
                    s['owner'] = owner['endpoint']
                elif s.get('owner') or (s['status']!='complete' and
                        (s['next'] in ('events','temporal','assemble') or
                         any((self.root/'annotations/episodes'/eid/(stage+'_profile.json')).exists()
                             for stage in dispatch.MODEL_STAGES))):
                    raise ValueError('Cannot silently transfer partial model work to episode dispatch: '+eid)
        if retry_failed:
            self.state.pop('resource_error',None)
        self.state.update(identity(),updated_at=now())
        if not self.state.get('resource_error'):
            self.state['status']='running'
        self.gpu = GPU(cfg['gpu'],self.root/'resources')
        self.save()

    def save(self):
        with self.mutex:
            self.state['counts'] = dict(Counter(s['status'] for s in self.state['episodes'].values()))
            self.state['updated_at'] = now(); atomic(self.state_path,self.state)

    def pressure(self):
        limits = self.cfg['limits']
        if psutil.virtual_memory().available < limits['min_available_ram_mib']*1024**2: return 'available_ram'
        if shutil.disk_usage(self.root).free < limits['min_free_disk_mib']*1024**2: return 'free_disk'
        # Byte limit covers retained direct-frame working packages. Disk-free gate
        # covers preprocessing retained separately; no cleanup of evidence.
        size = sum(p.stat().st_size for eid,s in self.state['episodes'].items()
                   if s['status'] in ('queued','running') and s['next']!='cpu'
                   for p in (self.root/'annotations/episodes'/eid).rglob('*') if p.is_file())
        if size >= limits['queue_bytes']: return 'retained_package_bytes'
        return None

    def candidate(self,kind):
        """Called under mutex. Prefer an endpoint's current episode over new work."""
        if self.stop.is_set(): return None
        episodes = self.state['episodes']
        active = sum(s['status'] in ('queued','running') and (s['next']!='cpu' or s['status']=='running')
                     for s in episodes.values())
        whole = dispatch.episode_mode(self.cfg)
        key = None
        if whole and kind == 'local':
            key = next((k for k in dispatch.endpoint_keys(self.cfg)
                        if self.cfg['endpoints'][k]['placement']=='local_gpu'),None)
            if key is None: return None
        elif whole and kind.startswith('remote:'): key=kind.split(':',1)[1]
        owned = {eid for eid,s in episodes.items() if key and s.get('owner')==key
                 and s['status'] in ('queued','running')}
        candidates = sorted(episodes.items(),key=lambda item:0 if item[0] in owned else 1)
        for eid,s in candidates:
            if s['status'] != 'queued': continue
            stage = s['next']
            if kind == 'cpu' and (stage!='cpu' or active >= self.cfg['limits']['queue_episodes'] or self.pressure()): continue
            if kind == 'depth' and stage!='depth': continue
            if kind == 'astra' and stage!='astra': continue
            if kind == 'io' and stage not in ('direct','assemble'): continue
            if kind == 'local' or kind.startswith('remote:'):
                if stage not in dispatch.MODEL_STAGES: continue
                if whole:
                    if s.get('owner'):
                        if s['owner'] != key: continue
                    elif stage!='scene' or owned: continue
                else:
                    route=self.cfg['routes'][stage];placement=self.cfg['endpoints'][route]['placement']
                    if kind=='local' and placement!='local_gpu': continue
                    if kind.startswith('remote:') and (placement!='remote' or route!=kind.split(':',1)[1]): continue
            return eid,stage,key
        return None

    def claim(self,kind):
        with self.mutex:
            found=self.candidate(kind)
            if not found: return None
            eid,stage,key=found;s=self.state['episodes'][eid]
            if key:
                owner=dispatch.assign(self.root,self.cfg,eid,key)
                s['owner']=owner['endpoint']
            s.update(status='running',started_at=now()); self.save(); return eid,stage

    def pending(self,kind):
        with self.mutex:
            if kind=='local': return self.candidate('local') is not None
            for s in self.state['episodes'].values():
                if s['status'] not in ('queued','running'): continue
                stage=s['next']
                if kind=='depth' and stage=='depth': return True
            return False

    def output_files(self,eid,stage):
        base=self.root/'annotations/episodes'/eid
        names={'cpu':[self.root/'rows'/(eid+'.json'),self.root/'prepared'/(eid+'.json')],
            'depth':[self.root/'prepared'/(eid+'.json')],
            'direct':[base/'input/ready.json'],
            'astra':[base/'selection/selected_frames.json'],
            'scene':[base/'scene.json'],'events':[base/'event_payloads.json'],
            'temporal':[base/'visual_source.json',base/'temporal.json',base/'parent_graph.json'],
            'assemble':[self.root/'annotations/json'/(eid+'.json'),base/'complete.json']}
        result={str(p):sha(p) for p in names[stage]}
        # Protect every stage-specific cache artifact, not just its summary.
        for name in {'astra':['selection'],'scene':['selection/qwen_scene_v2'],
                     'events':['events'],'temporal':['temporal']}.get(stage,[]):
            artifacts=tree(base/name)
            if stage=='astra':
                # Later scene inference legitimately adds these files under
                # selection; they belong to the scene receipt, not Astra's.
                artifacts={p:h for p,h in artifacts.items()
                           if 'qwen_scene_v2' not in Path(p).parts and Path(p).name!='scene_memory.json'}
            result.update(artifacts)
        if stage=='assemble':
            result.update(tree(self.root/'annotations/exports'/eid))
            result.update(tree(self.root/'annotations/confirmed_exports'/eid))
            confirmed=self.root/'annotations/confirmed_json'/(eid+'.json')
            if confirmed.exists(): result[str(confirmed)]=sha(confirmed)
            confirmed_receipt=self.root/'annotations/confirmed_receipts'/(eid+'.json')
            if confirmed_receipt.exists(): result[str(confirmed_receipt)]=sha(confirmed_receipt)
        return result

    def execute(self,job,lane=0):
        eid,stage=job; tick=time.perf_counter(); row_path=self.root/'rows'/(eid+'.json')
        try:
            if stage=='cpu':
                row=audit(self.rows[eid]); bind(row_path,row)
                if Path(self.records[eid]['source']).resolve()!=Path(row['source']).resolve():
                    raise ValueError('Preprocessing inventory capture differs from annotation capture')
                if self.reuse:
                    old=self.reuse[eid]
                    if old['status']!='complete' or Path(old['source']).resolve()!=Path(row['source']).resolve():
                        raise ValueError('Preprocessed reuse is incomplete or belongs to another capture')
                    direct.stage_records(old['output'])
                    prepared={'output':old['output'],'read_only_reuse':True}
                else: prepared=preprocess.cpu(self.records[eid],self.root,lane)
                bind(self.root/'prepared'/(eid+'.json'),prepared)
            elif stage=='depth':
                prepared=read(self.root/'prepared'/(eid+'.json'))
                if not prepared.get('read_only_reuse'): preprocess.depth(prepared,self.root,eid)
                direct.stage_records(prepared['output'])
            elif stage=='direct':
                row=read(row_path); prepared=read(self.root/'prepared'/(eid+'.json'))
                direct.build(row,prepared['output'],self.root/'annotations/episodes'/eid/'input')
            else:
                worker_cfg=self.root/'config.json'
                owner=None
                if dispatch.episode_mode(self.cfg) and stage in (*dispatch.MODEL_STAGES,'assemble'):
                    owner=dispatch.receipt_metadata(self.root,self.cfg,eid)
                    worker_cfg=dispatch.config_path(self.root,eid)
                with lock(self.root/'episode_locks'/(eid+'.lock')):
                    if not worker(stage,worker_cfg,row_path,self.root/'logs'/eid):
                        with self.mutex: self.state['episodes'][eid]['status']='queued'; self.save()
                        return
            receipt={'stage':stage,'episode_id':eid,'status':'passed','seconds':time.perf_counter()-tick,
                'completed_at':now(),'outputs':self.output_files(eid,stage)}
            if dispatch.episode_mode(self.cfg) and stage in (*dispatch.MODEL_STAGES,'assemble'):
                receipt['dispatch']=owner
                for path in (dispatch.assignment_path(self.root,eid),dispatch.config_path(self.root,eid)):
                    receipt['outputs'][str(path)]=sha(path)
            atomic(self.root/'receipts'/eid/(stage+'.json'),receipt)
            with self.mutex:
                s=self.state['episodes'][eid]; s['stages'][stage]=receipt
                index=STEPS.index(stage)+1
                s.update(status='complete' if index==len(STEPS) else 'queued',next=STEPS[index] if index<len(STEPS) else 'complete')
                self.save()
        except Exception as exc:
            with self.mutex:
                s=self.state['episodes'][eid]; s.update(status='needs_repair',error=str(exc),failed_stage=stage)
                atomic(self.root/'logs'/eid/('failure-'+str(time.time_ns())+'.json'),{'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()})
                self.save()

    def gpu_loop(self):
        group=0; budget=self.cfg['limits']['local_residency_budget_s']
        while not self.stop.is_set():
            job=None
            try:
                if self.pending('depth') and (group<self.cfg['limits']['depth_group_size'] or not self.pending('local')):
                    job=self.claim('depth')
                    if job:
                        prepared=self.root/'prepared'/(job[0]+'.json')
                        if read(prepared).get('read_only_reuse'): self.execute(job)
                        else:
                            with self.gpu.lease('depth',self.cfg['gpu']['depth_min_free_mib']): self.execute(job)
                        group+=1; continue
                if self.pending('local'):
                    profile=next(p for p in self.cfg['endpoints'].values() if p['placement']=='local_gpu')
                    (self.root/'yield_local').unlink(missing_ok=True)
                    with self.gpu.local_server(profile):
                        started=time.monotonic()
                        while not self.stop.is_set():
                            if self.pending('depth') and time.monotonic()-started>=budget: break
                            job=self.claim('local')
                            if not job: break
                            # Main loop writes yield_local at a request boundary
                            # once depth is waiting past the residency budget.
                            self.local_started=started
                            self.execute(job); self.local_started=None
                            job=None
                    group=0; continue
                time.sleep(.2)
            except Exception as exc:
                with self.mutex:
                    if job:
                        self.state['episodes'][job[0]].update(status='waiting_external_resource',error=str(exc))
                    # A retained/unresponsive local process is never reclaimed.
                    self.state.update(resource_error=str(exc),status='waiting_external_resource')
                    self.save(); self.stop.set(); return

    def run(self):
        import signal
        if self.state.get('resource_error'): return self.state
        handlers={}
        def request_stop(signum,frame):
            (self.root/'stop_requested').touch(); self.stop.set()
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT,signal.SIGTERM):
                handlers[sig]=signal.signal(sig,request_stop)
        (self.root/'stop_requested').unlink(missing_ok=True)
        from annotation_contract import build_contract
        from .location_facts import contract
        bind(self.root/'annotations/target.schema.json',contract(build_contract()))
        self.local_started=None
        gpu_thread=threading.Thread(target=self.gpu_loop,name='gpu-controller'); gpu_thread.start()
        try:
            remote_keys=[key for key in dispatch.endpoint_keys(self.cfg)
                         if self.cfg['endpoints'][key]['placement']=='remote']
            with ThreadPoolExecutor(max_workers=self.cfg['limits']['cpu_workers']+2+len(remote_keys)) as pool:
                jobs={}
                while not self.stop.is_set():
                    if self.local_started and self.pending('depth') and time.monotonic()-self.local_started >= self.cfg['limits']['local_residency_budget_s']:
                        (self.root/'yield_local').touch()
                    remote_kinds=[('remote:'+key,1) for key in remote_keys]
                    for kind,count in [('cpu',self.cfg['limits']['cpu_workers']),('astra',1),*remote_kinds,('io',1)]:
                        occupied={lane for k,lane in jobs.values() if k==kind}
                        for lane in (i for i in range(count) if i not in occupied):
                            job=self.claim(kind)
                            if job: jobs[pool.submit(self.execute,job,lane)]=(kind,lane)
                    if jobs:
                        done,_=wait(jobs,timeout=.3,return_when=FIRST_COMPLETED)
                        for future in done: future.result(); jobs.pop(future)
                    else: time.sleep(.2)
                    with self.mutex:
                        pending=[s for s in self.state['episodes'].values() if s['status'] in ('queued','running')]
                        if not pending: break
                        if all(s['next']=='cpu' and s['status']=='queued' for s in pending) and not jobs:
                            reason=self.pressure()
                            if reason: self.state.update(status='waiting_external_resource',resource_error='Backpressure: '+reason); break
        finally:
            self.stop.set(); gpu_thread.join()
            for sig,handler in handlers.items():signal.signal(sig,handler)
        if not self.state.get('resource_error'):
            if all(s['status']=='complete' for s in self.state['episodes'].values()):
                self.state['status']='complete'
            elif (self.root/'stop_requested').exists():
                self.state['status']='stopped'
            else:
                self.state['status']='needs_repair'
        self.state['finished_at']=now(); self.save(); return self.state
