from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
from PIL import Image
from frame_anchors.common import digest,sha
from frame_anchors.contract import plan
from frame_anchors.geometry import ground_frame
from frame_anchors.cli import DEFAULTS


@pytest.fixture
def sample(tmp_path):
    path=tmp_path/'frame.png';Image.new('RGB',(20,20)).save(path)
    doc={'scene_memory':{'objects':[{'id':key,'identity':{'name':name}} for key,name in [('o1','桌子'),('o2','无关灯')]]},
         'search_event_states':{'entity_reference_catalog':{'e1:cup':{'label':'cup','memory_entity_candidates':[]}},
         'events':[{'event_id':'e1','event_summary':'A cup rests on a white desk. A small dark device is nearby.',
          'states':[dict(state_id='s'+str(i),phase=phase,image_path=str(path),image_sha256=sha(path),
                         original_video_frame_index=i,focus_object_states=[{'entity_ref':'e1:cup'}] if i==2 else [])
                    for i,phase in enumerate(['start','middle','end'])]}]}}
    candidates=[]
    for cid,kind,ref,prompt,quote in [('entity:e1:cup','event_entity','e1:cup','cup','cup'),
        ('scene:o1','scene_object','o1','white desk','white desk'),
        ('mention:e1:device','description_mention',None,'small dark device','small dark device')]:
        candidates.append(dict(candidate_id=cid,reference_kind=kind,source_ref=ref,prompt=prompt,label=prompt,
            role='mentioned_object',description_evidence=[{'path':'event_summary','quote':quote}]))
    mapping={'schema':'ego.event_description_candidates.v1','source_document_digest':digest(doc),
             'events':{'e1':{'candidates':candidates,'excluded_mentions':[]}}}
    return doc,mapping


def test_event_union_all_three_frames_without_unmentioned_scene_objects(sample):
    doc,mapping=sample;frames=plan(doc,mapping,'event_description')
    assert len(frames)==3
    for f in frames:
        assert {c['candidate_id'] for c in f['candidates']}=={'entity:e1:cup','scene:o1','mention:e1:device'}
    assert not frames[0]['candidates'][0]['is_frame_focus']
    assert frames[2]['candidates'][0]['is_frame_focus']
    result=ground_frame(frames[0],{'cup':[{'box':[1,1,10,10],'score':.9}]},np.zeros((0,20,20),bool),DEFAULTS)
    assert result['objects'][0]['description_evidence']==mapping['events']['e1']['candidates'][0]['description_evidence']
    assert result['objects'][0]['selection_scope']=='event_description_mentions'
    assert len(result['candidate_outcomes'])==3


def test_chinese_original_label_uses_explicit_english_mapping(sample):
    doc,mapping=sample
    doc['search_event_states']['entity_reference_catalog']['local1' if 'local1' in doc['search_event_states']['entity_reference_catalog'] else 'e1:cup']['label']='杯子'
    mapping['source_document_digest']=digest(doc)
    frames=plan(doc,mapping,'event_description')
    assert frames[0]['candidates'][0]['prompt']=='cup'
    assert frames[0]['candidates'][0]['source_label']=='杯子'


@pytest.mark.parametrize('mutation',['wrong_quote','wrong_source','wrong_event','duplicate','cross_event','wrong_field'])
def test_invalid_selection_rejected(sample,mutation):
    doc,mapping=sample;m=deepcopy(mapping);c=m['events']['e1']['candidates'][0]
    if mutation=='wrong_quote':c['description_evidence'][0]['quote']='unmentioned lamp'
    if mutation=='wrong_source':m['source_document_digest']='wrong'
    if mutation=='wrong_event':m['events']['other']=m['events'].pop('e1')
    if mutation=='duplicate':m['events']['e1']['candidates'].append(deepcopy(c))
    if mutation=='cross_event':c['source_ref']='e2:cup'
    if mutation=='wrong_field':c['description_evidence'][0]['path']='event_id'
    with pytest.raises(ValueError):plan(doc,m,'event_description')
