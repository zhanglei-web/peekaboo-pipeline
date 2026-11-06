"""One preparation lane overlaps serial local scene/event/temporal inference."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json,sys,time,fcntl,traceback
import jsonschema
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'search_episode_outputs'
sys.path.insert(0,'/home/baai/zl/Qwen3.8-27B/scripts')
sys.path.insert(0,'/home/baai/zl/end_version')
import run_scene_event_batch as pipe
from batch_temporal_trees import annotate_event,output_schema,validate_output,VERSION as TEMP_VERSION
from temporal_spatial_v2 import initialize_temporal_graph,nested_temporal_tree
from build_final_annotation_sample import extend,preserved_values
from validate_final_annotation_sample import audit_temporal,assert_target_fields
from export_final_annotation_sample import export
from annotation_contract import build_contract,potential_schema,add_unknown_research,attach_potential,validate_potential
from PIL import Image,ImageDraw,ImageFont


def read(p):return json.loads(p.read_text())

def event_job(event,folder,target):
 folder.mkdir(parents=True,exist_ok=True)
 width,height,banner=960,600,40
 canvas=Image.new('RGB',(width*3,height+banner),'white');draw=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
 for i,state in enumerate(event['states']):
  p=Path(state['image_path']);assert pipe.sha(p)==state['image_sha256']
  with Image.open(p) as im:canvas.paste(im.convert('RGB').resize((width,height)),(i*width,banner))
  draw.text((i*width+8,5),f"{state['phase']} | f{state['original_video_frame_index']} | {state['original_video_seconds']:.2f}s",font=font,fill='black')
 path=folder/'triplet.jpg';canvas.save(path,quality=95,subsampling=0);details=[]
 for i,state in enumerate(event['states']):
  detail=folder/(state['phase']+'_detail.jpg');canvas.crop((i*width,0,(i+1)*width,height+banner)).save(detail,quality=95,subsampling=0);details.append(detail)
 ev=pipe.full_contract()['properties']['search_event_states']['properties']['events']['items'];schema=pipe.detailed_schema(ev)
 schema['properties']['potential_search_regions']=potential_schema([s['original_video_frame_index'] for s in event['states']]);schema['required'].append('potential_search_regions')
 prompt=(pipe.ROOT/'prompts/search_event_detailed.zh.txt').read_text()+'\n'+(pipe.ROOT/'prompts/search_event_target_context.zh.txt').read_text()
 prompt+='\nThe search target from the supplied event Markdown is: '+json.dumps(target,ensure_ascii=False)
 prompt+='\nThe three individual images show the same start/middle/end panels in detail, not additional moments. Respect their original frame numbers. Distinguish finger contact from nearby objects; uncertainty must remain explicit. Each frame description must independently state its visible hand action and object states. Do not infer continuous actions, intentions or target success.'
 prompt+='\nAdditional field potential_search_regions: list the distinct spatial areas that this event could plausibly be inspecting, supported by the supplied three frames (for example under a lifted object, inside an opened container, or the specific shelf portion exposed by an action). For each give description, visual_basis, evidence_frames using only the supplied frame numbers, and uncertainty. Separate visible evidence from the interpretation as a possible search region. Do not list generic places merely because the target might fit, unrelated background areas, or assert that all candidates were actually searched. Do not replace the observed search_region. Use [] if no candidate area can be supported; explain the limitation in uncertainty. Do not infer cross-event region identity or success. FINAL LANGUAGE REQUIREMENT: write all generated descriptions in English, overriding earlier language instructions; preserve proper target names and phase enum values.'
 data=pipe.local_call(folder,prompt,[path,*details],schema,6144)
 assert [s['phase'] for s in data['states']]==['start','middle','end']
 return data


def process(row,folder,update):
 eid=row['episode_id'];tick=time.perf_counter();stages={}
 def stage(name,fn):
  update(eid,status='running',stage=name);t=time.perf_counter();v=fn();stages[name]=time.perf_counter()-t;pipe.atomic(folder/'stage_times.json',stages);return v
 scene=stage('scene',lambda:pipe.scene_job(folder/'selection'))
 payloads=[stage('event_%02d'%e['ordinal'],lambda e=e:event_job(e,folder/'events'/('event_%02d'%e['ordinal']),row['target'])) for e in row['event_index']['events']]
 source=pipe.export_episode(row,scene,payloads);pipe.atomic(folder/'visual_source.json',source)
 graph=initialize_temporal_graph(scene);temporal=deepcopy(source)
 for e in temporal['search_event_states']['events']:
  graph,changes,uncertainties,cached=stage('temporal_%02d'%e['ordinal'],lambda e=e:annotate_event(source,e,graph,folder/'temporal'/('event_%02d'%e['ordinal'])))
  e['spatial_changes']=changes;e['spatial_change_annotation']={'status':'local_qwen_visual_inference_unreviewed','uncertainties':uncertainties,'source':'original_event_triplet_images','description_language':'en'}
 temporal['schema']=TEMP_VERSION;temporal['scene_memory']['spatial_structure']=nested_temporal_tree(graph)
 validate_output(source,temporal,output_schema(source['schema']))
 pipe.atomic(folder/'temporal.json',temporal);pipe.atomic(folder/'parent_graph.json',graph)
 finish(row,folder,source,payloads,temporal,graph,stages,tick,update)

def finish(row,folder,source,payloads,temporal,graph,stages,tick,update=None):
 eid=row['episode_id']
 if update is None:update=lambda *args,**kwargs:None
 research=add_unknown_research(source,row);attach_potential(research,payloads)
 source_path=folder/'source_with_research_fields.json';pipe.atomic(source_path,research)
 final=extend(research,source_path,temporal,graph,folder/'temporal.json')
 record=final['README']['annotation_record'];record.update(label_source='fresh_astra_local_qwen_with_potential_regions',manifest=str(ROOT/'annotation_inventory.json'),completion=str(folder/'complete.json'),scene_ref=eid+'.json#/scene_memory',new_model_calls=len(list(folder.rglob('response.json'))))
 final['README']['format_extension']['version']='additive_research_temporal.potential_regions.v1'
 final['README']['format_extension']['legacy_revisit_policy']='Cross-event place identity and revisit counts remain unknown; event-local IDs must not imply distinct physical places.'
 final['README']['completeness']['missing']['spatial_temporal_review']=['Model visual annotations are unreviewed. Sample frame boundaries do not establish exact continuous action times.']+[e['event_id']+': '+u for e in final['search_event_states']['events'] for u in e['spatial_change_annotation']['uncertainties']]
 final['README']['format_extension']['identity_policy'] += ' Potential search regions are event-local hypotheses, not confirmed intentions or a complete inventory of places.'
 if row.get('frame_index_warning'):final['README']['completeness']['missing']['spatial_temporal_review'].append(row['frame_index_warning'])
 final['scene_memory']['spatial_structure']['temporal_status']='fresh_model_annotations_linked_unreviewed'
 issues=preserved_values(research,final)
 if issues:raise ValueError('Source fields changed during assembly: '+repr(issues))
 jsonschema.validate(final,read(OUT/'target.schema.json'));validate_potential(final);audit=audit_temporal(final)
 assert final['metadata']['target']['description']==row['target']
 assert len(final['search_event_states']['events'])==row['event_count']
 assert pipe.sha(Path(row['target_markdown']))==row['target_markdown_sha256']
 for event,expected in zip(final['search_event_states']['events'],row['event_index']['events'],strict=True):
  for a,b in zip(event['states'],expected['states'],strict=True):
   for key in ['image_path','image_sha256','original_video_frame_index','phase']:assert a[key]==b[key]
   assert pipe.sha(Path(a['image_path']))==a['image_sha256']
 a,b=export(final)
 ref=Path('/home/baai/zl/end_version')
 assert_target_fields(read(ref/'cp_0011_session_20261029_115034.json'),a,'episode');assert_target_fields(read(ref/'experiment_records.json'),b,'experiment')
 output=OUT/'json'/(eid+'.json');pipe.atomic(output,final);assert read(output)==final
 a['scene_ref']='../../json/'+eid+'.json#/scene_memory'
 pipe.atomic(OUT/'exports'/eid/(eid+'.json'),a);pipe.atomic(OUT/'exports'/eid/'experiment_records.json',b)
 receipt={'status':'passed','episode_id':eid,'event_count':row['event_count'],'potential_region_count':sum(len(e['potential_search_regions']) for e in final['search_event_states']['events']),'sha256':pipe.sha(output),'target_markdown_sha256':row['target_markdown_sha256'],'schema_sha256':pipe.sha(OUT/'target.schema.json'),'temporal_audit':audit,'local_processing_seconds':time.perf_counter()-tick,'stages':stages,'completed_at':pipe.now(),'semantic_human_reviewed':False}
 pipe.atomic(folder/'complete.json',receipt);update(eid,status='complete',output=str(output),receipt=str(folder/'complete.json'))


def main():
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'batch.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  import psutil
  me=psutil.Process();pipe.atomic(OUT/'launch.json',{'pid':me.pid,'create_time':me.create_time(),'command':me.cmdline()})
  inv=read(ROOT/'annotation_inventory.json');pre=read(ROOT/'state.json')
  assert pre['counts'].get('complete')==29,'All bags must pass verification before annotation'
  rows=inv['episodes'];audit=read(ROOT/'input_audit.json');assert audit['status']=='passed'
  for row in rows:
   check=audit['episodes'][row['episode_id']];assert check['status']=='passed'
   row['raw_video_image_checks']=check['checks']
   if check['frame_index_drift_count']:row['frame_index_warning']='Some filename frame labels differ from sequential decoded frame indices. Supplied labels and label/fps times are preserved; they are not claimed as calibrated acquisition times. See input_audit.json.'
   state=pre['records'][row['episode_id']];row['source_bag']=state['bag'];row['preprocessed']=state['output']
  pipe.atomic(OUT/'inventory.json',inv);pipe.atomic(OUT/'target.schema.json',build_contract())
  state=read(OUT/'batch_status.json') if (OUT/'batch_status.json').exists() else {'episodes':{},'started_at':pipe.now()}
  def update(eid,**changes):
   state['episodes'].setdefault(eid,{}).update(changes,updated_at=pipe.now());state['completed_count']=sum(e.get('status')=='complete' for e in state['episodes'].values());state['status']='running';pipe.atomic(OUT/'batch_status.json',state)
  pending=[]
  for row in rows:
   folder=OUT/'episodes'/row['episode_id'];receipt=folder/'complete.json';output=OUT/'json'/(row['episode_id']+'.json')
   if receipt.exists():
    assert pipe.sha(output)==read(receipt)['sha256'];jsonschema.validate(read(output),read(OUT/'target.schema.json'));validate_potential(read(output));update(row['episode_id'],status='complete',output=str(output))
   else:pending.append(row)
  # At most one next episode is being prepared; no concurrent local Qwen calls.
  with ThreadPoolExecutor(max_workers=1) as prep:
   future=prep.submit(pipe.prepare_episode,pending[0],OUT) if pending else None
   for index,row in enumerate(pending):
    try:folder=future.result();preparation_error=None
    except Exception as exc:preparation_error=exc
    future=prep.submit(pipe.prepare_episode,pending[index+1],OUT) if index+1<len(pending) else None
    try:
     if preparation_error:raise preparation_error
     process(row,folder,update)
    except Exception as exc:
     update(row['episode_id'],status='needs_repair',error=repr(exc));(OUT/'episodes'/row['episode_id']/'failure_traceback.txt').write_text(traceback.format_exc())
  state['status']='complete' if state.get('completed_count',0)==29 else 'needs_repair';state['finished_at']=pipe.now();pipe.atomic(OUT/'batch_status.json',state)
  if state['status']!='complete':raise RuntimeError('Some episodes need repair; inspect retained diagnostics')
if __name__=='__main__':main()
