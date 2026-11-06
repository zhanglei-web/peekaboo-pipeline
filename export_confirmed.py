"""Project human-confirmed dataset outcomes; preserve inference and original outputs."""
from pathlib import Path
from copy import deepcopy
import json,sys,fcntl,time,argparse,hashlib
import jsonschema
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'search_episode_outputs'
sys.path.insert(0,'/home/baai/zl/end_version')
from export_final_annotation_sample import export
from validate_final_annotation_sample import audit_temporal,assert_target_fields
from annotation_contract import validate_potential

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def transform(source,facts):
 eid=source['metadata']['episode_id']
 if eid not in facts['scope']['episode_ids']:raise ValueError('Episode not in user-confirmed scope')
 if facts['source']!='explicit_user_message' or facts['facts']!={'each_episode_has_exactly_one_found_event':True,'found_event_is_last_in_supplied_event_order':True,'all_earlier_events_not_found':True}:raise ValueError('Unexpected confirmation facts')
 d=deepcopy(source);events=d['search_event_states']['events'];n=len(events)
 if not n or [e['ordinal'] for e in events]!=list(range(1,n+1)):raise ValueError('Noncontiguous event order')
 inspections={i['event_id']:i for i in d['inspections']}
 if set(inspections)!={e['event_id'] for e in events}:raise ValueError('Event/inspection mapping differs')
 origin=facts['fact_id']
 d['metadata']['only_final_event_found_target_confirmed']=True
 for idx,e in enumerate(events):
  found=idx==n-1
  # Never rewrite visual descriptions, temporal updates or evidence identities.
  if 'visual_outcome_before_protocol' not in e:e['visual_outcome_before_protocol']=deepcopy(e['outcome'])
  e['result']='found' if found else 'not_found';e['result_source']=origin
  e['outcome']={'target_found':found,'target_identity_confirmed':e['visual_outcome_before_protocol']['target_identity_confirmed'],
   'reason':'Dataset owner confirms that only the final supplied search event found the target; this event '+('is the final event.' if found else 'occurred before the final event.')+' This human outcome label does not establish a visually verified instance, exact location or success frame.'}
  inspections[e['event_id']]['result']=e['result'];inspections[e['event_id']]['result_source']=origin
 d['episode_outcome'].update(result='found',success_event_id=events[-1]['event_id'],success_event_order=events[-1]['ordinal'],result_source=origin)
 # Unspecified location, frame, temporal boundaries and revisit fields are preserved.
 d['derived_statistics'].update(found_event_count=1,events_before_success=n-1,events_through_success=n,
  note='Human-confirmed outcomes for this enumerated dataset: all earlier events not_found, final event found. Not-found does not establish effective exclusion; revisit counts require cross-event region identity.')
 record=d['README']['annotation_record'];record['label_source']='astra_local_qwen_visual_annotation_plus_explicit_human_dataset_outcomes'
 d['README']['rules'].append('For episodes in dataset_facts.json only, the dataset owner confirms earlier searches did not find the target and the last search did. Preserve original visual interpretations; do not infer no-revisit, precise found-frame, or hidden unit identity.')
 verify(d,facts)
 return d

def verify(d,facts):
 assert d['metadata']['episode_id'] in facts['scope']['episode_ids']
 es=d['search_event_states']['events'];ins={i['event_id']:i for i in d['inspections']}
 assert d['metadata']['only_final_event_found_target_confirmed'] is True
 for i,e in enumerate(es):
  expected='found' if i==len(es)-1 else 'not_found'
  assert e['result']==expected and ins[e['event_id']]['result']==expected
  assert e['outcome']['target_found']==(i==len(es)-1)
  assert e['result_source']==facts['fact_id']==ins[e['event_id']]['result_source']
 assert d['episode_outcome']['result']=='found'
 assert d['episode_outcome']['success_event_id']==es[-1]['event_id']
 assert d['episode_outcome']['success_event_order']==len(es)
 assert d['derived_statistics']['found_event_count']==1
 assert d['derived_statistics']['events_before_success']==len(es)-1
 assert d['derived_statistics']['events_through_success']==len(es)

def run_once(out=OUT,facts_path=ROOT/'dataset_facts.json'):
 facts=read(facts_path);schema=read(out/'target.schema.json');done=[];pending=[];failures=[]
 for eid in facts['scope']['episode_ids']:
  try:
   receipt=out/'episodes'/eid/'complete.json';src=out/'json'/(eid+'.json')
   if not receipt.exists():pending.append(eid);continue
   source_hash=sha(src)
   if read(receipt)['sha256']!=source_hash:raise ValueError('Input completion receipt/hash mismatch')
   source=read(src);jsonschema.validate(source,schema);validate_potential(source);audit_temporal(source)
   d=transform(source,facts);dest=out/'confirmed_json'/(eid+'.json');rec=out/'confirmed_receipts'/(eid+'.json')
   d['README']['annotation_record'].update(manifest=str(facts_path),completion=str(rec),scene_ref=eid+'.json#/scene_memory')
   jsonschema.validate(d,schema);validate_potential(d);audit_temporal(d)
   assert d['scene_memory']==source['scene_memory'] and d['hidden']==source['hidden'] and d['priors']==source['priors']
   for a,b in zip(d['search_event_states']['events'],source['search_event_states']['events'],strict=True):
    for key in ['states','event_summary','observed_changes','search_region','potential_search_regions','spatial_changes','spatial_change_annotation','is_revisit','region_visit_index','region_total_visits','visual_outcome_before_protocol']:assert a[key]==b[key],key
   a,b=export(d);ref=Path('/home/baai/zl/end_version')
   assert_target_fields(read(ref/'cp_0011_session_20261029_115034.json'),a,'episode');assert_target_fields(read(ref/'experiment_records.json'),b,'experiment')
   a['scene_ref']='../../confirmed_json/'+eid+'.json#/scene_memory'
   outputs={dest:d,out/'confirmed_exports'/eid/(eid+'.json'):a,out/'confirmed_exports'/eid/'experiment_records.json':b}
   for path,value in outputs.items():
    if path.exists() and read(path)!=value:raise ValueError('Refusing to overwrite changed confirmed output: '+str(path))
   for path,value in outputs.items():
    if not path.exists():atomic(path,value)
   atomic(rec,{'status':'passed','source_json':str(src),'source_sha256':source_hash,'confirmation_file':str(facts_path),'confirmation_sha256':sha(facts_path),'schema_sha256':sha(out/'target.schema.json'),'output_sha256':sha(dest),'outputs':{str(p):sha(p) for p in outputs},'events':len(d['inspections']),'model_calls_for_outcome_projection':0})
   done.append(eid)
  except Exception as exc:failures.append({'episode_id':eid,'error':repr(exc)})
 report={'status':'complete' if len(done)==len(facts['scope']['episode_ids']) and not failures else 'needs_repair' if failures else 'waiting_for_source','expected':len(facts['scope']['episode_ids']),'completed_count':len(done),'completed':done,'pending':pending,'failures':failures,'facts_sha256':sha(facts_path),'output':str(out/'confirmed_json')}
 atomic(out/'confirmed_status.json',report);return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');args=p.parse_args()
 with (OUT/'confirmed_export.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  while True:
   r=run_once();print(json.dumps({k:v for k,v in r.items() if k not in ['completed']},ensure_ascii=False),flush=True)
   if r['status'] in ['complete','needs_repair'] or not args.wait:break
   time.sleep(15)
  if r['status']=='needs_repair':raise SystemExit(1)
if __name__=='__main__':main()
