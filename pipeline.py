#!/usr/bin/env python3
"""Terminal interface for the pinned 11-4 dataset pipeline on this host."""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
import psutil
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'search_episode_outputs'

def read(p,default=None):return json.loads(p.read_text()) if p.exists() else default
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def alive(record):
 try:
  p=psutil.Process(record['pid']);return p.create_time()==record['create_time'] and p.cmdline()==record['command'] and p.status()!=psutil.STATUS_ZOMBIE
 except (KeyError,TypeError,psutil.Error):return False

def locked(path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return False
  except BlockingIOError:return True

def status():
 prep=read(ROOT/'state.json',{});ann=read(OUT/'batch_status.json',{});orch=read(ROOT/'orchestrator_state.json',{});con=read(OUT/'confirmed_status.json',{})
 return {'version':(ROOT/'VERSION').read_text().strip(),'preprocessing':prep.get('counts',{}),'annotation_completed':ann.get('completed_count',0),'annotation_status':ann.get('status'),'confirmed_completed':con.get('completed_count',0),'confirmed_status':con.get('status'),'orchestrator_live':alive(orch),'orchestrator_stage':orch.get('stage'),'annotation_live':alive(read(OUT/'launch.json',{})),'terminal_worker_live':alive(read(ROOT/'terminal_worker.json',{})),'pending':[{'episode_id':k,**v} for k,v in ann.get('episodes',{}).items() if v.get('status')!='complete']}

def check(require_key=False):
 errors=[];lock=read(ROOT/'dependencies.lock.json')
 for e in lock['code']:
  if not Path(e['path']).is_file() or sha(e['path'])!=e['sha256']:errors.append('Dependency changed/missing: '+e['path'])
  if sha(ROOT/e['snapshot'])!=e['sha256']:errors.append('Committed snapshot changed: '+e['snapshot'])
 for e in lock['external_assets']:
  p=Path(e['path'])
  if not p.is_file() or (p.stat().st_size,p.stat().st_mtime_ns)!=(e['size'],e['mtime_ns']):errors.append('External asset changed/missing: '+str(p))
 for p in ['/opt/ros/jazzy/setup.bash','/home/baai/zl/egomemory_ws/.venv/bin/python','/usr/bin/python3']:
  if not Path(p).is_file():errors.append('Required runtime missing: '+p)
 if require_key and not os.environ.get('ZYAPI_API_KEY'):errors.append('ZYAPI_API_KEY is not set; value is never logged')
 if not (ROOT/'annotation_inventory.json').exists():errors.append('Batch inventory missing; this release expects the existing 11.4 batch')
 import orchestrate
 orchestrate.check_sources();orchestrate.check_paused()
 if errors:raise RuntimeError('\n'.join(errors))
 return {'status':'passed','code_files_verified':len(lock['code']),'external_assets_checked':len(lock['external_assets']),'key_available':bool(os.environ.get('ZYAPI_API_KEY'))}

def validate(reconcile=False):
 import jsonschema
 from annotation_contract import validate_potential
 sys.path.insert(0,'/home/baai/zl/end_version')
 from validate_final_annotation_sample import audit_temporal,assert_target_fields
 from export_confirmed import verify,atomic
 inv=read(ROOT/'annotation_inventory.json');schema=read(OUT/'target.schema.json');facts=read(ROOT/'dataset_facts.json');done=[];missing=[];errors=[]
 for row in inv['episodes']:
  eid=row['episode_id'];f=OUT/'json'/(eid+'.json');rec=OUT/'episodes'/eid/'complete.json'
  if not rec.exists():missing.append(eid);continue
  try:
   assert sha(f)==read(rec)['sha256'];d=read(f);jsonschema.validate(d,schema);validate_potential(d);audit_temporal(d)
   assert len(d['inspections'])==row['event_count'];assert d['metadata']['target']['description']==row['target']
   for event,src in zip(d['search_event_states']['events'],row['event_index']['events'],strict=True):
    assert event['event_id']==src['event_id']
    for a,b in zip(event['states'],src['states'],strict=True):
     for k in ['image_path','image_sha256','original_video_frame_index','phase']:assert a[k]==b[k]
     assert sha(a['image_path'])==a['image_sha256']
   for name,key in [(eid+'.json','episode'),('experiment_records.json','experiment')]:
    template='cp_0011_session_20261029_115034.json' if key=='episode' else name
    assert_target_fields(read(Path('/home/baai/zl/end_version')/template),read(OUT/'exports'/eid/name),key)
   confirmed=OUT/'confirmed_json'/(eid+'.json')
   if confirmed.exists():
    c=read(confirmed);jsonschema.validate(c,schema);verify(c,facts);validate_potential(c);audit_temporal(c)
    receipt=read(OUT/'confirmed_receipts'/(eid+'.json'));assert receipt['source_sha256']==sha(f) and receipt['confirmation_sha256']==sha(ROOT/'dataset_facts.json')
    for path,h in receipt['outputs'].items():assert sha(path)==h
   done.append(eid)
  except Exception as e:errors.append({'episode_id':eid,'error':repr(e)})
 result={'status':'passed' if len(done)==len(inv['episodes']) and not errors else 'incomplete','validated':len(done),'expected':len(inv['episodes']),'missing':missing,'errors':errors}
 if reconcile:
  if locked(ROOT/'orchestrator.lock') or locked(OUT/'batch.lock'):raise RuntimeError('Active writer; cannot reconcile live status')
  bs=read(OUT/'batch_status.json',{'episodes':{}})
  for eid in done:
   bs['episodes'].setdefault(eid,{}).update(status='complete',stage='validated_and_published',output=str(OUT/'json'/(eid+'.json')));bs['episodes'][eid].pop('error',None)
  bs.update(completed_count=len(done),status='complete' if result['status']=='passed' else 'needs_repair');atomic(OUT/'batch_status.json',bs)
  atomic(ROOT/'terminal_validation.json',result)
  if result['status']=='passed':
   atomic(ROOT/'final_validation.json',{'status':'passed','episodes':len(done),'events':sum(x['event_count'] for x in inv['episodes']),'outputs':str(OUT/'json'),'basis':'Full CLI validation of schema, receipts, hashes, event evidence, temporal relations and target exports; historical failure logs preserved','semantic_human_reviewed':False})
 return result

def launch(script,log,command=None):
 with (ROOT/log).open('a') as out:
  child=subprocess.Popen(command or ['/usr/bin/python3','-u',str(ROOT/script)],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 time.sleep(.1)
 if child.poll() is not None and child.returncode:raise RuntimeError('Failed to start '+script+'; see '+log)
 return child

def work():
 from export_confirmed import atomic,run_once
 with (ROOT/'terminal_worker.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  me=psutil.Process();atomic(ROOT/'terminal_worker.json',{'pid':me.pid,'create_time':me.create_time(),'command':me.cmdline()})
  check(require_key=True)
  # Respect already-running jobs, including the old orchestrator during handoff.
  while locked(ROOT/'orchestrator.lock') or locked(OUT/'batch.lock'):time.sleep(15)
  report=validate(reconcile=True)
  if report['status']!='passed':
   pre=read(ROOT/'state.json',{})
   if pre.get('counts',{}).get('complete',0)!=29:
    failed=[k for k,v in pre.get('records',{}).items() if v['status'] in ['cpu_failed','depth_or_export_failed']]
    if failed:raise RuntimeError('Diagnose failed preprocessing before resume: '+str(failed))
    if not locked(ROOT/'batch.lock'):
     c=launch('run_batch.py','batch.log',['bash',str(ROOT/'run.sh')]);p=psutil.Process(c.pid)
     for _ in range(30):
      if str(ROOT/'run_batch.py') in p.cmdline():break
      time.sleep(.1)
     atomic(ROOT/'launcher.json',{'pid':p.pid,'create_time':p.create_time(),'command':p.cmdline()})
   audit=read(ROOT/'input_audit.json',{})
   if audit.get('status')!='passed':
    if audit.get('status')=='needs_repair':raise RuntimeError('Input audit requires repair before continuing')
    if not locked(ROOT/'input_audit.lock'):launch('audit_inputs.py','input_audit.log')
   c=launch('orchestrate.py','orchestrator.log')
   while c.poll() is None:time.sleep(15)
   report=validate(reconcile=True)
  confirmation=run_once()
  atomic(ROOT/'terminal_run_result.json',{'annotation':report,'human_confirmed_output':confirmation})
  if report['status']!='passed' or confirmation['status']!='complete':raise RuntimeError('Incomplete; inspect terminal_run_result.json')
  print(json.dumps({'status':'complete','output':confirmation['output']},ensure_ascii=False),flush=True)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['status','check','run','resume','validate','export-confirmed','_worker']);p.add_argument('--reconcile',action='store_true',help='Rebuild batch status from verified completion receipts, only with no active writer');a=p.parse_args()
 if a.command=='status':result=status()
 elif a.command=='check':result=check()
 elif a.command=='validate':result=validate(a.reconcile)
 elif a.command=='export-confirmed':
  from export_confirmed import main as export_main
  sys.argv=[sys.argv[0]];return export_main()
 elif a.command=='_worker':return work()
 else:
  check(require_key=True)
  with (ROOT/'terminal_start.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   if locked(ROOT/'terminal_worker.lock'):result={'status':'already_running',**status()}
   else:
    c=launch('pipeline.py','terminal.log',['/usr/bin/python3','-u',str(ROOT/'pipeline.py'),'_worker']);result={'status':'started','pid':c.pid,'log':str(ROOT/'terminal.log')}
 print(json.dumps(result,ensure_ascii=False,indent=2))
 if result.get('status') in ['incomplete','needs_repair']:raise SystemExit(2)
if __name__=='__main__':
 try:main()
 except Exception as e:print(str(e),file=sys.stderr);raise SystemExit(1)
