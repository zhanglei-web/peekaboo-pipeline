"""Release regression tests using isolated temporary outputs; no model/GPU use."""
from pathlib import Path
from copy import deepcopy
import json,tempfile,time,sys
import jsonschema
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import annotate_batch as a
from annotation_contract import build_contract
from export_confirmed import transform,verify

def main():
 baseline=Path('/home/baai/zl/end_version/performance_checks/fresh_single_20261104_1624')
 row=a.read(baseline/'manifest.json')['input'];eid=row['episode_id'];row['source_annotation_metadata']={k:None for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']}
 source=a.read(baseline/'fresh_visual_source.json');temporal=a.read(baseline/'fresh_temporal.json');graph=a.read(baseline/'parent_graph.json')
 payloads=[{'potential_search_regions':[]} for e in row['event_index']['events']];updates=[]
 with tempfile.TemporaryDirectory() as tmp:
  out=Path(tmp);folder=out/'episodes'/eid;folder.mkdir(parents=True);a.pipe.atomic(folder/'temporal.json',temporal);a.pipe.atomic(out/'target.schema.json',build_contract());old_out=a.OUT;a.OUT=out
  try:a.finish(row,folder,source,payloads,temporal,graph,{},time.perf_counter(),lambda *args,**kw:updates.append((args,kw)))
  finally:a.OUT=old_out
  assert updates[-1][1]['status']=='complete','Completion callback lost'
  d=a.read(out/'json'/(eid+'.json'));assert d['metadata']['provenance']==source['metadata']['provenance'];assert d['README']['rules']==source['README']['rules']
  facts=a.read(ROOT/'dataset_facts.json');facts['scope']['episode_ids']=[eid] # Synthetic in-memory scope only; production manifest never changed.
  projected=transform(d,facts);verify(projected,facts);jsonschema.validate(projected,build_contract())
  assert transform(projected,facts)['search_event_states']==projected['search_event_states']
  assert d['episode_outcome']['result']=='unknown'
  assert projected['episode_outcome']['final_location'] is None
  assert projected['episode_outcome']['found_evidence_frames'] is None
  assert projected['metadata']['no_revisits_confirmed'] is None
  assert projected['scene_memory']==d['scene_memory']
  for x,y in zip(d['search_event_states']['events'],projected['search_event_states']['events']):
   assert x['states']==y['states'];assert x['potential_search_regions']==y['potential_search_regions'];assert x['visual_outcome_before_protocol']==y['visual_outcome_before_protocol'];assert y['is_revisit'] is None
  facts['scope']['episode_ids']=[]
  try:transform(d,facts)
  except ValueError:pass
  else:raise AssertionError('Out-of-scope episode accepted')
 print('PASS: actual final assembly, source preservation, completion callback, confirmed outcomes, no fabricated location/revisit/frame, out-of-scope rejection')
if __name__=='__main__':main()
