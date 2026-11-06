from pathlib import Path
import sys,json,re,copy,psutil
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
sys.path.insert(0,'/home/baai/zl/Qwen3.8-27B/scripts')
import run_scene_event_batch as p

def field(text,name):
 m=re.search(r'^'+re.escape(name)+r'\s*[：:]\s*(.*?)\s*$',text,re.M)
 return m[1] if m else None

def main():
 if (ROOT/'inventory.json').exists():raise FileExistsError('Preserve existing inventory')
 raws={};mds={}
 for bundle in ['已标完数据','11.4已标注']:
  for f in (BASE/bundle).rglob('meta.json'):
   if '_session_' in f.parent.name:
    assert f.parent.name not in raws
    raws[f.parent.name]=f.parent
  for f in (BASE/bundle).rglob('*三帧.md'):
   assert f.parent.name not in mds
   mds[f.parent.name]=f
 assert len(raws)==29 and set(raws)==set(mds)
 template=json.loads(Path('/home/baai/zl/egomemory_ws/src/egomemory-core/config/ego_preprocessing/demo.json').read_text())
 # Exact same parameters as the validated previous preprocessing batch.
 prev=json.loads(Path('/home/baai/zl/todo/11.3/processing/inventory.json').read_text())
 template=json.loads(Path(prev['records'][0]['config']).read_text())
 records=[];episodes=[]
 for eid,raw in sorted(raws.items()):
  md=mds[eid];text=md.read_text(encoding='utf-8-sig');meta=json.loads((raw/'meta.json').read_text())
  assert raw.relative_to(BASE).parts[0]==md.relative_to(BASE).parts[0]
  scene=field(text,'场景');target=field(text,'要找的物体');assert scene and target
  if raw.relative_to(BASE).parts[0]=='已标完数据':assert raw.parent.parent.name==scene and raw.parent.name==target
  fps=float(re.search(r'帧率\s*(\d+(?:\.\d+)?)\s*帧/秒',text)[1]);assert fps==meta['fps']
  idx=p.triplets(md.parent/'left',eid,fps)
  right=p.triplets(md.parent/'right',eid,fps)
  table=[[int(x) for x in m] for m in re.findall(r'^\|\s*第(\d+)次\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|',text,re.M)]
  for index in [idx,right]:assert table==[[e['ordinal'],*[s['original_video_frame_index'] for s in e['states']]] for e in index['events']]
  assert max(f for e in table for f in e[1:])<meta['topics']['left']['written_frames']
  required=['meta.json','videos/left.mkv','videos/right.mkv','imu/imu0.csv','timestamps/left_timestamps.txt','timestamps/right_timestamps.txt','calibrations/camera-camchain.yaml','calibrations/camera_imu-camchain-imucam.yaml','calibrations/camera_imu-imu.yaml']
  missing=[f for f in required if not (raw/f).is_file()];assert not missing,(eid,missing)
  out=BASE/'preprocessed'/eid;bag=BASE/'rosbags'/eid;config=ROOT/'configs'/(eid+'.json')
  cfg=copy.deepcopy(template);cfg.update(source=str(raw),output=str(out));p.atomic(config,cfg)
  records.append({'id':eid,'name':eid,'group':scene,'source':str(raw),'output':str(out),'bag':str(bag),'config':str(config),'duration_s':meta['duration_sec'],'status':'queued','missing_inputs':[],'meta_sha256':p.sha(raw/'meta.json')})
  episodes.append({'episode_id':eid,'source':str(raw),'source_bag':str(bag),'preprocessed':str(out),'scene_group':scene,'target':target,'fps':fps,'target_markdown':str(md),'target_markdown_sha256':p.sha(md),'event_images':str(md.parent/'left'),'event_count':idx['event_count'],'event_index':idx,'right_event_index':right,'source_frame_count':meta['topics']['left']['written_frames'],'metadata_sha256':p.sha(raw/'meta.json'),'source_annotation_metadata':{k:field(text,k) for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']},'problems':[]})
  p.atomic(ROOT/'source_manifests'/(eid+'.json'),{'files':[{'path':str(f),'size':f.stat().st_size,'mtime_ns':f.stat().st_mtime_ns} for f in sorted(raw.rglob('*')) if f.is_file()],'metadata_sha256':p.sha(raw/'meta.json'),'markdown':str(md),'markdown_sha256':p.sha(md)})
 inv={k:v for k,v in prev.items() if k!='records'};inv.update(input_root=str(BASE),output_root=str(BASE/'preprocessed'),bag_root=str(BASE/'rosbags'),record_count=29,records=records)
 p.atomic(ROOT/'inventory.json',inv)
 p.atomic(ROOT/'annotation_inventory.json',{'total':29,'ready_count':29,'blocked_count':0,'events':sum(r['event_count'] for r in episodes),'episodes':episodes})
 paused=[]
 for proc in psutil.process_iter(['pid','status','cmdline','create_time']):
  if proc.info['status']==psutil.STATUS_STOPPED and any('Qwen3.8-27B' in a for a in proc.info['cmdline'] or []):paused.append({'pid':proc.pid,'create_time':proc.create_time(),'command':proc.cmdline()})
 assert not any(x.info['name']=='llama-server' for x in psutil.process_iter(['name']))
 p.atomic(ROOT/'gpu_release_authorized.json',{'authorization':'User requested processing these 29 captures; no local model service is running.','checked_at':p.now(),'paused_clients_preserved':paused})
 # Preserve the original runner and its fixed recovery policy.
 s=Path('/home/baai/zl/todo/11.3/processing/run_batch.py').read_text().replace("runner_version='11.3.recovery_and_gpu_handoff.v1'","runner_version='11.4.same_preprocessing_parameters.v1'")
 (ROOT/'run_batch.py').write_text(s)
 print(json.dumps({'captures':len(records),'events':sum(x['event_count'] for x in episodes),'paused':len(paused)},ensure_ascii=False))
if __name__=='__main__':main()
