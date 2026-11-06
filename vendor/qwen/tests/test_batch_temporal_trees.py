import copy
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from batch_temporal_trees import merge_observations, output_schema, validate_output, VERSION
from temporal_spatial_v2 import initialize_temporal_graph, nested_temporal_tree


def test_same_frame_event_can_replace_reference_with_audit():
    scene={'spatial_structure':{'root_id':'root','nodes':[{'id':'root','name':'root','type':'region','children':[
        {'id':'a','name':'object','type':'object','object_ref':'a','children':[],
         'placement':{'predicate':'within_region','evidence_frames':['f10']}}
    ]}],'unlocated_nodes':[]}}
    graph=initialize_temporal_graph(scene)
    new,changes,audit=merge_observations(graph,[{'node_id':'a','change_type':'unparent',
        'change_frame':10,'parent_id':None,'predicate':'unknown','relative_position':None}], 'event',[10,15,20])
    assert new['nodes']['a']['parent_relations'][0]['parent_id'] is None
    assert audit[0]['reference_relation']['parent_id']=='root'
    assert graph['nodes']['a']['parent_relations'][0]['parent_id']=='root'


def test_both_source_variants_preserve_all_existing_labels():
    source=Path('/home/baai/11-4-81/标注信息')
    for name in ('cp_0007_session_20261026_122149.json','nightstand__cs_0001_session_20261030_183921.json'):
        path=source/name
        if not path.exists():pytest.skip('Local fixture unavailable')
        doc=json.loads(path.read_text());out=copy.deepcopy(doc)
        out['schema']=VERSION
        out['scene_memory']['spatial_structure']=nested_temporal_tree(initialize_temporal_graph(doc['scene_memory']))
        for e in out['search_event_states']['events']:
            e['spatial_changes']=[]
            e['spatial_change_annotation']={'status':'local_qwen_visual_inference_unreviewed','uncertainties':[],
                'source':'original_event_triplet_images','description_language':'en'}
        validate_output(doc,out,output_schema(doc['schema']))
        out['metadata']['only_final_event_found_target_confirmed']=not doc['metadata']['only_final_event_found_target_confirmed']
        with pytest.raises(ValueError,match='changed'):
            validate_output(doc,out,output_schema(doc['schema']))


def test_future_scene_reference_does_not_overwrite_event_state():
    from temporal_spatial_v2 import _active_relation
    scene={'spatial_structure':{'root_id':'root','nodes':[{'id':'root','name':'root','type':'region','children':[
        {'id':'a','name':'object','type':'object','object_ref':'a','children':[],
         'placement':{'predicate':'within_region','evidence_frames':['f132']}}
    ]}],'unlocated_nodes':[]}}
    graph=initialize_temporal_graph(scene)
    new,changes,audit=merge_observations(graph,[{'node_id':'a','change_type':'unparent',
        'change_frame':109,'parent_id':None,'predicate':'unknown','relative_position':None},
        {'node_id':'a','change_type':'establish','change_frame':191,'parent_id':'root',
         'predicate':'within_region','relative_position':None}], 'event',[109,133,191])
    assert _active_relation(new['nodes']['a'],133)['parent_id'] is None
    assert _active_relation(new['nodes']['a'],191)['parent_id']=='root'
    assert audit[0]['reference_relation']['valid_from_frame']==132


def test_reconfirmation_is_not_an_invented_position_change():
    scene={'spatial_structure':{'root_id':'root','nodes':[{'id':'root','name':'root','type':'region','children':[
        {'id':'a','name':'object','type':'object','object_ref':'a','children':[],
         'placement':{'predicate':'within_region','evidence_frames':['f10'],'description':'center'}}
    ]}],'unlocated_nodes':[]}}
    graph=initialize_temporal_graph(scene)
    new,changes,audit=merge_observations(graph,[{'node_id':'a','change_type':'establish',
        'change_frame':20,'parent_id':'root','predicate':'within_region','relative_position':'middle'}], 'event',[20,25,30])
    assert new==graph
    assert changes==[]
    assert audit[0]['reason']=='reconfirmed_relation_without_supported_change'
