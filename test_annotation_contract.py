"""Offline contract and evidence-link regression checks; no inference outputs published."""
from pathlib import Path
from copy import deepcopy
import sys,json,tempfile
import jsonschema
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,'/home/baai/zl/end_version')
from annotation_contract import *
from build_final_annotation_sample import extend,preserved_values
from validate_final_annotation_sample import audit_temporal,assert_target_fields
from export_final_annotation_sample import export

def main():
 old=Path('/home/baai/zl/end_version/performance_checks/fresh_single_20261104_1624')
 source=json.loads((old/'fresh_visual_source.json').read_text());temporal=json.loads((old/'fresh_temporal.json').read_text());graph=json.loads((old/'parent_graph.json').read_text());row=json.loads((old/'manifest.json').read_text())['input']
 row['source_annotation_metadata']={k:None for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']}
 d=add_unknown_research(source,row)
 data=[{'potential_search_regions':[{'description':'Synthetic contract test only','visual_basis':'Synthetic contract test only','evidence_frames':[e['states'][0]['original_video_frame_index']],'uncertainty':'Synthetic fixture, not an annotation'}]} for e in d['search_event_states']['events']]
 attach_potential(d,data)
 with tempfile.TemporaryDirectory() as t:
  f=Path(t)/'fixture.json';f.write_text(json.dumps(d));final=extend(d,f,temporal,graph,old/'fresh_temporal.json')
 schema=build_contract();jsonschema.validate(final,schema);validate_potential(final);audit_temporal(final);assert not preserved_values(d,final)
 a,b=export(final)
 assert_target_fields(json.loads(Path('/home/baai/zl/end_version/cp_0011_session_20261029_115034.json').read_text()),a,'episode')
 assert_target_fields(json.loads(Path('/home/baai/zl/end_version/experiment_records.json').read_text()),b,'experiment')
 broken=deepcopy(final);del broken['search_event_states']['events'][0]['potential_search_regions']
 try:jsonschema.validate(broken,schema)
 except jsonschema.ValidationError:pass
 else:raise AssertionError('Missing field accepted')
 broken=deepcopy(final);broken['search_event_states']['events'][0]['potential_search_regions'][0]['evidence_frames']=[99999999]
 try:validate_potential(broken)
 except (AssertionError,KeyError):pass
 else:raise AssertionError('Wrong event frame accepted')
 assert final['episode_outcome']['result']=='unknown' and final['metadata']['no_revisits_confirmed'] is None
 assert all(e['is_revisit'] is None and e['region_total_visits'] is None and e['result']=='unknown' for e in final['search_event_states']['events'])
 for name in ['annotation_contract.py','annotate_batch.py','audit_inputs.py']:compile((ROOT/name).read_text(),name,'exec')
 print('PASS: strict additive format, unknown labels, target exports, temporal audit, required potential region field and event-frame rejection')
if __name__=='__main__':main()
