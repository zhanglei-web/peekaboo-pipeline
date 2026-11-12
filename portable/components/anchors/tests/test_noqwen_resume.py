import importlib.util
from pathlib import Path
from copy import deepcopy
from contextlib import nullcontext
import json
import pytest
from frame_anchors.common import atomic,read,sha


def script(name):
    path=Path(__file__).resolve().parents[1]/'scripts'/name
    spec=importlib.util.spec_from_file_location(name.replace('.','_'),path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_longest_noun_matches_and_no_body_or_unseen_target():
    m=script('prepare_resume50.py')
    text='白色笔记本电脑和黑色鼠标垫上有白色鼠标。未见其他疑似黑色U盘。'
    assert [m.prompt(x) for x in m.mentions(text)]==['white laptop','black mouse pad','white mouse']
    text='A white laptop and black mouse pad sit on a desk. No yellow tag is visible. A hand holds various small items.'
    assert [m.prompt(x) for x in m.mentions(text)]==['white laptop','black mouse pad','desk']


def test_goal_brackets_uncertainty_and_packaging_depiction():
    m=script('prepare_resume50.py')
    assert not m.mentions('Searching for [headphones]. No headphones are visible.')
    assert not m.mentions('It is unclear whether a yellow tag is visible.')
    assert not m.mentions('an image of a power drill')
    assert [m.prompt(x) for x in m.mentions('a cardboard box with a drill graphic')]==['cardboard box']
    assert [m.prompt(x) for x in m.mentions('A hand holds black headphones.')]==['black headphones']


def test_source_bound_mapping_does_not_expand_all_scene_objects():
    m=script('prepare_resume50.py')
    event={'event_id':'event1','event_summary':'A black mouse is next to a white laptop.',
        'states':[{'phase':x,'focus_object_states':[{'entity_ref':'local1'}]} for x in ['start','middle','end']]}
    doc={'scene_memory':{'objects':[{'id':'o1','identity':{'name':'white laptop'}},
                                  {'id':'o2','identity':{'name':'red basket'}}]},
         'search_event_states':{'entity_reference_catalog':{'local1':{'label':'black mouse'}}}}
    selected=m.build(doc,event)
    assert {c['candidate_id'] for c in selected['candidates']}=={'entity:local1','scene:o1'}
    assert all(c['description_evidence'][0]['quote'] in event['event_summary'] for c in selected['candidates'])


def test_compact_check_preservation_and_geometry(tmp_path):
    runner=script('run_noqwen_batches.py')
    original={'search_event_states':{'events':[{'states':[{'phase':'start','keep':[1,2]}]}]}}
    source=tmp_path/'source.json';out=tmp_path/'out.json';atomic(source,original)
    compact=deepcopy(original);state=compact['search_event_states']['events'][0]['states'][0]
    state['object_anchors']=[{'name':'cup','ref':'scene:o1','anchor_xy_normalized':None,
                            'size_wh_normalized':None,'status':'not_detected'}]
    atomic(out,compact);assert runner.compact_check(source,out)==1
    state['object_anchors'][0]['anchor_xy_normalized']=[.2,.3];atomic(out,compact)
    with pytest.raises(ValueError,match='Undetected'):runner.compact_check(source,out)
    state['object_anchors']=[];state['keep'][0]=3;atomic(out,compact)
    with pytest.raises(ValueError,match='Original'):runner.compact_check(source,out)


def test_conflicting_record_is_skipped_and_next_record_completes(tmp_path,monkeypatch):
    runner=script('run_noqwen_batches.py')
    rows=[]
    for i in range(2):
        source=tmp_path/f'input{i}.json';atomic(source,{'search_event_states':{'events':[]}})
        rows.append({'group':'g','record_id':str(i),'source_json':str(source),'source_sha256':sha(source),
                     'event_count':0,'frame_count':0})
        atomic(tmp_path/'selections'/'g'/f'{i}.json',{'policy':{'ready_for_grounding':True,
            'selection_review_status':'source_bound_lexicon_not_human_semantic_reviewed'},'fail':i==0})
    atomic(tmp_path/'input_inventory.json',{'records':rows,'errors':[]})
    monkeypatch.setattr(runner,'plan',lambda doc,mapping,scope: (_ for _ in ()).throw(ValueError('test conflict')) if mapping['fail'] else [])
    monkeypatch.setattr(runner,'gpu_lease',lambda cfg:nullcontext())
    monkeypatch.setattr(runner,'Models',lambda cfg:object())
    calls=[]
    monkeypatch.setattr(runner,'run',lambda cfg,**kw:calls.append(cfg['input_json']))
    def export(detailed,compact):
        compact.mkdir(parents=True);out=compact/'out.json';atomic(out,{'search_event_states':{'events':[]}})
        return {'compact_json':str(out),'compact_sha256':sha(out),'source_sha256':rows[1]['source_sha256'],'status':'passed'}
    monkeypatch.setattr(runner,'load_exporter',lambda:export)
    atomic(tmp_path/'model_config.json',{'grounding_dino_path':str(tmp_path/'dino'),'fastsam_weights':str(tmp_path/'fastsam.pt'),'gpu_lock':str(tmp_path/'gpu.lock')})
    runner.run_batch(tmp_path)
    state=read(tmp_path/'batch_state.json')
    assert calls==[rows[1]['source_json']]
    assert set(state['completed'])=={'g/1'}
    assert set(state['failures'])=={'g/0'}
    assert state['status']=='incomplete'
    final=read(tmp_path/'final_validation.json')
    assert final['unprocessed']==['g/0'] and final['completed']==1
