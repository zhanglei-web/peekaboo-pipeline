from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from frame_anchors.common import sha, atomic, inside
from frame_anchors.cli import DEFAULTS, run
from frame_anchors.contract import plan, augment, assert_preserved
from frame_anchors.geometry import ground_frame, interior_anchor
from frame_anchors.artifacts import save_masks, load_masks, validate


def candidate(cid='scene:o01'):
    return dict(candidate_id=cid,reference_kind='scene_object',source_ref=cid.split(':')[-1],
                source_label='cup',scene_entity_candidates=['o01'],is_frame_focus=False,prompt='cup')


def frame(candidates=None):
    return dict(width=20,height=20,state_id='s1',image_sha256='test',candidates=candidates or [candidate()])


def evidence():
    masks=np.zeros((1,20,20),dtype=bool); masks[0,4:16,4:16]=True
    return {'cup':[{'box':[4,4,16,16],'score':.9}]},masks


def test_ring_anchor_and_disconnected_largest_component():
    mask=np.zeros((20,20),dtype=bool);mask[2:18,2:18]=True;mask[5:15,5:15]=False
    a=interior_anchor(mask);x,y=a['pixel_xy'];assert mask[int(y),int(x)]
    mask=np.zeros((20,20),dtype=bool);mask[0,0]=True;mask[10:18,10:18]=True
    a=interior_anchor(mask);assert a['pixel_xy'][0]>=10 and a['connected_components']==2
    a=interior_anchor(np.ones((20,20),bool));assert a['boundary_clearance_px']==10


def test_identity_conflict_is_retained():
    ds,ms=evidence();result=ground_frame(frame([candidate(),candidate('entity:local1')]),ds,ms,DEFAULTS)
    assert len(result['objects'])==2
    for obj in result['objects']:
        assert 'competing_source_references' in obj['review_flags']
        assert obj['track_id'] is None and not obj['human_reviewed']
        assert obj['apparent_size']['metric_dimensions'] is None


@pytest.mark.parametrize('ambiguous',[False,True])
def test_explicit_mask_fallback(ambiguous):
    ds,ms=evidence();ms=np.repeat(ms,2,axis=0) if ambiguous else np.zeros((0,20,20),bool)
    obj=ground_frame(frame(),ds,ms,DEFAULTS)['objects'][0]
    assert obj['anchor']['pixel_xy']==[10,10]
    assert obj['anchor']['inside_segmented_region'] is None
    assert obj['segmentation_status']==('ambiguous_masks' if ambiguous else 'no_matching_mask')


def test_clipping_nonfinite_and_wrong_mask_size():
    ds,ms=evidence();ds['cup'][0]['box']=[-100,-50,21,30]
    obj=ground_frame(frame(),ds,ms,DEFAULTS)['objects'][0]
    assert obj['bbox_xyxy_pixels']==[0,0,20,20]
    assert 'image_boundary_contact_possible_truncation' in obj['review_flags']
    ds['cup'][0]['score']=float('nan')
    with pytest.raises(ValueError):ground_frame(frame(),ds,ms,DEFAULTS)
    with pytest.raises(ValueError):ground_frame(frame(),{},np.zeros((0,10,10),bool),DEFAULTS)


def test_not_detected_does_not_claim_absent():
    result=ground_frame(frame(),{},np.zeros((0,20,20),bool),DEFAULTS)
    assert result['objects']==[]
    assert result['candidate_outcomes'][0]['status']=='not_detected'
    assert result['candidate_outcomes'][0]['absence_confirmed'] is False


def test_multiple_instances_not_merged_and_limit_flag():
    ds={'cup':[{'box':[1,1,7,7],'score':.9},{'box':[12,12,18,18],'score':.8}]}
    ms=np.zeros((0,20,20),bool)
    result=ground_frame(frame(),ds,ms,DEFAULTS)
    assert len(result['objects'])==2
    assert all('multiple_instances_for_source_reference' in o['review_flags'] for o in result['objects'])
    result=ground_frame(frame(),ds,ms,{**DEFAULTS,'max_instances_per_prompt':1})
    assert result['candidate_outcomes'][0]['instance_limit_reached']


@pytest.fixture
def document(tmp_path):
    source=tmp_path/'input';source.mkdir();image=source/'image.png';Image.new('RGB',(20,20),'white').save(image)
    states=[dict(state_id='s'+str(i),phase=phase,image_path=str(image),image_sha256=sha(image),
                 original_video_frame_index=i,focus_object_states=[{'entity_ref':'local1'}])
            for i,phase in enumerate(['start','middle','end'])]
    return {'schema':'test.v1','unrelated':{'value':[1,'unchanged']},
        'scene_memory':{'objects':[{'id':'o01','identity':{'name':'杯子'}}]},
        'search_event_states':{'entity_reference_catalog':{'local1':{'label':'cup','memory_entity_candidates':[]}},
                               'events':[{'event_id':'event1','states':states}]}}


def test_contract_preserves_values_and_checks_conflicts(document):
    prompts={'scene:o01':'cup'};frames=plan(document,prompts)
    assert len(frames)==3
    result=augment(document,{f['state_id']:{} for f in frames},{})
    assert_preserved(document,result)
    result['unrelated']['value'][0]=2
    with pytest.raises(ValueError):assert_preserved(document,result)
    with pytest.raises(ValueError):plan(document,{**prompts,'scene:unknown':'other'})
    bad=deepcopy(document);bad['search_event_states']['events'][0]['states'][0]['image_sha256']='wrong'
    with pytest.raises(ValueError):plan(bad,prompts)
    bad=deepcopy(document);bad['search_event_states']['events'][0]['states'][1]['state_id']='s0'
    with pytest.raises(ValueError):plan(bad,prompts)
    bad=deepcopy(document);bad['search_event_states']['events'][0]['states'][0]['focus_object_states'][0]['entity_ref']='missing'
    with pytest.raises(ValueError):plan(bad,prompts)


def test_mask_archive_and_path_guard(tmp_path):
    for masks in [np.zeros((0,20,20),bool),evidence()[1]]:
        save_masks(tmp_path/'m.npz',masks)
        assert np.array_equal(load_masks(tmp_path/'m.npz'),masks)
    with pytest.raises(ValueError):inside(tmp_path,'../escape')


def test_review_lists_undetected_candidates(tmp_path):
    from frame_anchors.artifacts import make_review
    f={**frame(),'frame_index':1}
    result=ground_frame(f,{},np.zeros((0,20,20),bool),DEFAULTS)
    make_review(tmp_path,[f],{'s1':result})
    assert 'scene:o01' in (tmp_path/'review.html').read_text()


def test_end_to_end_mock_and_tamper_detection(document,tmp_path,monkeypatch):
    import frame_anchors.models as adapters
    source=tmp_path/'input';atomic(source/'source.json',document);atomic(source/'prompts.json',{'scene:o01':'cup'})
    weights=tmp_path/'models';weights.mkdir()
    for name in ['config.json','model.safetensors','preprocessor_config.json','tokenizer_config.json','tokenizer.json','fastsam.pt']:
        (weights/name).write_text('mock-only')
    cfg={**DEFAULTS,'input_json':str(source/'source.json'),'prompt_map':str(source/'prompts.json'),
         'output_dir':str(tmp_path/'output'),'grounding_dino_path':str(weights),'fastsam_weights':str(weights/'fastsam.pt')}
    class FakeModels:
        calls=0
        def __init__(self,cfg):pass
        def infer(self,image,prompts):
            FakeModels.calls+=1;ds,ms=evidence()
            return ds,ms,dict(detection_seconds=.01,segmentation_seconds=.01,mask_limit_reached=False)
    monkeypatch.setattr('frame_anchors.cli.importlib.metadata.version',lambda name:'mock-test-only')
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    monkeypatch.setattr(adapters,'Models',FakeModels)
    monkeypatch.setattr(adapters,'gpu_lease',lambda cfg:nullcontext())
    run(cfg);assert FakeModels.calls==3
    out=Path(cfg['output_dir']);report=validate(out)
    assert report['frames']==3 and report['semantic_human_reviewed'] is False
    run(cfg);assert FakeModels.calls==3
    with pytest.raises(ValueError):run({**cfg,'detection_threshold':.4})
    payload=json.loads((out/'anchors.sidecar.json').read_text());payload['frames'].pop('s0');atomic(out/'anchors.sidecar.json',payload)
    with pytest.raises(ValueError):validate(out)


def test_failure_record_only_after_manifest_acceptance(tmp_path):
    from frame_anchors.cli import report_failures
    with pytest.raises(RuntimeError):
        with report_failures(tmp_path):raise RuntimeError('fake model failure')
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['status']=='failed' and state['error']=='fake model failure'
