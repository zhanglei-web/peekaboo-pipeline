"""Finish all preprocessing, then run the prefetched annotation batch and release GPU."""
from pathlib import Path
import json,sys,time,fcntl,subprocess,traceback
import psutil,requests
ROOT=Path(__file__).resolve().parent
QWEN=Path('/home/baai/zl/Qwen3.8-27B')
OUT=ROOT.parent/'search_episode_outputs'
sys.path.insert(0,str(QWEN/'scripts'))
from run_scene_event_batch import atomic,sha,now

def read(p):return json.loads(p.read_text())
def identity(record):
 try:
  p=psutil.Process(record['pid']);return p.create_time()==record['create_time'] and p.cmdline()==record['command'] and p.status()!=psutil.STATUS_ZOMBIE
 except psutil.NoSuchProcess:return False

def check_paused():
 for r in read(ROOT/'gpu_release_authorized.json')['paused_clients_preserved']:
  if not identity(r) or psutil.Process(r['pid']).status()!=psutil.STATUS_STOPPED:raise RuntimeError('Paused worker identity/status changed')

def check_sources():
 for p in (ROOT/'source_manifests').glob('*.json'):
  m=read(p)
  for f in m['files']:
   st=Path(f['path']).stat()
   if st.st_size!=f['size'] or st.st_mtime_ns!=f['mtime_ns']:raise RuntimeError('Original input changed: '+f['path'])
  if sha(Path(m['markdown']))!=m['markdown_sha256']:raise RuntimeError('Input Markdown changed')
 for path,h in read(ROOT/'protected_outputs.json').items():
  if sha(Path(path))!=h:raise RuntimeError('Prior output/sample changed: '+path)

def main():
 with (ROOT/'orchestrator.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  me=psutil.Process();state={'pid':me.pid,'create_time':me.create_time(),'command':me.cmdline(),'started_at':now(),'status':'running','stage':'preprocessing'};owned=None
  def update(**values):state.update(values,updated_at=now());atomic(ROOT/'orchestrator_state.json',state)
  try:
   check_paused();update()
   while True:
    batch=read(ROOT/'state.json');update(stage='preprocessing',preprocessing_counts=batch.get('counts',{}))
    if batch.get('status')=='complete' and batch.get('counts',{}).get('complete')==29:break
    if not identity(read(ROOT/'launcher.json')):raise RuntimeError('Preprocessor exited before all 29 bags passed; inspect batch.log and state.json')
    time.sleep(15)
   while True:
    audit=read(ROOT/'input_audit.json')
    if audit['status']=='passed':break
    if audit['status']=='needs_repair':raise RuntimeError('Raw image content audit failed; inspect input_audit.json')
    update(stage='waiting_for_input_audit');time.sleep(15)
   check_sources();check_paused()
   # A service owned by another job is never stopped or reconfigured.
   while any(p.info['name']=='llama-server' for p in psutil.process_iter(['name'])):
    update(stage='waiting_for_other_model_service');time.sleep(15)
   update(stage='starting_local_model')
   with (ROOT/'server_start.log').open('a') as log:subprocess.run([str(QWEN/'scripts/start_server.sh'),'--config',str(ROOT/'local_qwen.server.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
   owned=read(QWEN/'logs/server.pid.json');atomic(ROOT/'server_identity.json',owned)
   if not identity(owned):raise RuntimeError('Server identity mismatch')
   check_paused();update(stage='annotating')
   with (ROOT/'annotation.log').open('a') as log:child=subprocess.Popen(['/usr/bin/python3','-u',str(ROOT/'annotate_batch.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   proc=psutil.Process(child.pid);update(annotation_process={'pid':proc.pid,'create_time':proc.create_time(),'command':proc.cmdline()})
   while child.poll() is None:
    if (OUT/'batch_status.json').exists():
     bs=read(OUT/'batch_status.json');update(annotation_completed=bs.get('completed_count',0),annotation_episodes=bs['episodes'])
    time.sleep(15)
   if child.returncode:raise RuntimeError('Annotation process exited '+str(child.returncode)+'; inspect retained per-episode errors')
   check_sources();check_paused()
   bs=read(OUT/'batch_status.json');assert bs['completed_count']==29
   for row in read(ROOT/'annotation_inventory.json')['episodes']:
    receipt=read(OUT/'episodes'/row['episode_id']/'complete.json');assert receipt['status']=='passed' and sha(OUT/'json'/(row['episode_id']+'.json'))==receipt['sha256']
   atomic(ROOT/'final_validation.json',{'status':'passed','episodes':29,'events':196,'outputs':str(OUT/'json'),'finished_at':now(),'originals_and_sample_unchanged':True,'semantic_human_reviewed':False})
   update(status='complete',stage='complete',annotation_completed=29,finished_at=now())
  except Exception as exc:
   update(status='needs_repair',error=repr(exc));(ROOT/'orchestrator_failure.txt').write_text(traceback.format_exc())
  finally:
   if owned:
    try:
     if not identity(owned):raise RuntimeError('Cannot stop service: identity changed')
     sess=requests.Session();sess.trust_env=False
     for _ in range(3):
      slots=sess.get('http://127.0.0.1:8080/slots',timeout=10).json()
      if not slots or any(s['is_processing'] for s in slots):raise RuntimeError('Service is busy; preserved for active request')
      time.sleep(1)
     subprocess.run([str(QWEN/'scripts/stop_server.sh')],check=True);update(service_stopped=True)
    except Exception as exc:update(service_stop_error=repr(exc))
if __name__=='__main__':main()
