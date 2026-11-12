"""11.4 additive sample contract with unknown research labels and potential regions."""
from pathlib import Path
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / 'repository.marker').is_file())
from copy import deepcopy
from pathlib import Path
import json,sys
sys.path.insert(0,str(REPOSITORY / 'components/additive'))
from final_sample_schema import build,add

VERSION='spatial_memory.search_episode.additive_research_temporal.potential_regions.v1'
TEXT={'type':'string','minLength':1}
NULLTEXT={'type':['string','null']}
SOURCE='user_supplied_markdown_and_visual_inference_unreviewed'

def obj(props):return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}

def potential_schema(frames=None,exported=False):
 f={'type':'integer','minimum':0}
 if frames is not None:f['enum']=frames
 props={'description':deepcopy(TEXT),'visual_basis':deepcopy(TEXT),
        'evidence_frames':{'type':'array','minItems':1,'uniqueItems':True,'items':f},
        'uncertainty':deepcopy(TEXT)}
 if exported:props.update(region_id=deepcopy(TEXT),evidence_state_ids={'type':'array','minItems':1,'uniqueItems':True,'items':deepcopy(TEXT)},annotation_status={'const':'potential_search_region_inferred_not_confirmed_intention'})
 return {'type':'array','items':obj(props)}

def build_contract():
 s=build();p=s['properties'];p['schema']={'const':VERSION}
 meta=p['metadata']['properties']
 meta['pre_search_information_received']={'type':['boolean','null']}
 meta['no_revisits_confirmed']={'type':['boolean','null']}
 for k in ['pre_search_information_source','research_fields_source']:meta[k]=deepcopy(NULLTEXT)
 add(p['metadata'],{'source_annotation_metadata':obj({k:deepcopy(NULLTEXT) for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']}),
   'source_annotation_markdown':deepcopy(TEXT),'source_annotation_markdown_sha256':deepcopy(TEXT)})
 p['history']={'anyOf':[{'type':'null'},deepcopy(p['history'])]}
 h=p['hidden']['properties'];h['target_location']={'anyOf':[{'type':'null'},h['target_location']]};h['source']=deepcopy(NULLTEXT)
 for node in [p['search_event_states']['properties']['events']['items'],p['inspections']['items']]:
  q=node['properties']
  q['region_visit_index']={'type':['integer','null'],'minimum':1};q['region_total_visits']={'type':['integer','null'],'minimum':1}
  q['is_revisit']={'type':['boolean','null']};q['region_identity_source']=deepcopy(NULLTEXT)
  q['result_source']=deepcopy(NULLTEXT);q['result']={'enum':['unknown','found','not_found']}
 unit=p['search_units']['properties']['list']['items']['properties'];unit['visit_count']={'type':['integer','null'],'minimum':1};unit['visit_count_source']=deepcopy(NULLTEXT)
 p['episode_outcome']['properties']['result_source']=deepcopy(NULLTEXT)
 d=p['derived_statistics']['properties'];d['revisit_count']={'type':['integer','null'],'minimum':0};d['revisit_count_source']=deepcopy(NULLTEXT)
 d['region_visit_counts']={'type':['array','null'],'items':obj({'unit_id':deepcopy(TEXT),'description':deepcopy(TEXT),'count':{'type':['integer','null'],'minimum':1}})}
 ev=p['search_event_states']['properties']['events']['items'];add(ev,{'potential_search_regions':potential_schema(exported=True)})
 import jsonschema
 jsonschema.Draft202012Validator.check_schema(s)
 return s

def add_unknown_research(source,row):
 d=deepcopy(source);d['schema']=VERSION
 m=d['metadata'];meta=row['source_annotation_metadata']
 for key in ['participant_id','home_id','encoding_condition','target_instance_id','target_category']:m[key]=None
 m.update(participant_id=meta['采集人员'],scene_id=meta['场景'] or m['scene_id'],
  pre_search_information_received=None,
  pre_search_information_source='Participant pre-search information is not confirmed; the Markdown exposure flag is retained separately without equating it to information received',research_fields_source=SOURCE,
  pending_metadata_fields=['home_id','encoding_condition','target_instance_id','target_category'],no_revisits_confirmed=None,
  source_annotation_metadata=meta,source_annotation_markdown=row['target_markdown'],source_annotation_markdown_sha256=row['target_markdown_sha256'])
 d['history']=None
 d['hidden']={'target_location':None,'target_instance_id':None,'source':SOURCE,'rule':'Raw location descriptions are retained in metadata.source_annotation_metadata; no confirmed unit identity or success event is inferred from them.'}
 for event,inspection in zip(d['search_event_states']['events'],d['inspections'],strict=True):
  shared={'search_region_id':inspection['unit_ids'][0],'region_visit_index':None,'region_total_visits':None,'is_revisit':None,'region_identity_source':'event_local_region_not_cross_event_identity','validity':None,'validity_status':'pending_definition','inspect_cost_s':None}
  event.update(shared,visual_outcome_before_protocol=deepcopy(event['outcome']),result='unknown',result_source='no_explicit_success_event_confirmation')
  inspection.update(shared,search_region=event['search_region'])
 for unit in d['search_units']['list']:unit.update(visit_count=None,visit_count_source='cross_event_region_identity_not_verified')
 d['episode_outcome']['result_source']='no_explicit_success_event_confirmation'
 d['derived_statistics'].update(revisit_count=None,revisit_count_source='cross_event_region_identity_not_verified',region_visit_counts=None)
 return d

def attach_potential(doc,payloads):
 for event,data in zip(doc['search_event_states']['events'],payloads,strict=True):
  mapping={s['original_video_frame_index']:s['state_id'] for s in event['states']}
  event['potential_search_regions']=[]
  for n,region in enumerate(data['potential_search_regions'],1):
   if not set(region['evidence_frames'])<=set(mapping):raise ValueError('Potential region references another event frame')
   event['potential_search_regions'].append({**deepcopy(region),'region_id':event['event_id']+f':potential_region_{n:02d}',
     'evidence_state_ids':[mapping[f] for f in region['evidence_frames']],
     'annotation_status':'potential_search_region_inferred_not_confirmed_intention'})

def validate_potential(doc):
 for e in doc['search_event_states']['events']:
  mapping={s['original_video_frame_index']:s['state_id'] for s in e['states']}
  ids=[]
  for r in e['potential_search_regions']:
   assert r['evidence_state_ids']==[mapping[f] for f in r['evidence_frames']]
   ids.append(r['region_id'])
  assert len(ids)==len(set(ids))
