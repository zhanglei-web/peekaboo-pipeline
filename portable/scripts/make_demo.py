#!/usr/bin/env python3
"""Generate synthetic schema-valid JSON and images; never run or imitate model inference."""
from pathlib import Path
import argparse
from copy import deepcopy
import json
import sys
from pipeline import activate,write,sha
activate()
from PIL import Image,ImageDraw
from generate_selected_scene import assemble
from run_scene_event_batch import export_episode
from temporal_spatial_v2 import initialize_temporal_graph,nested_temporal_tree
from annotation_contract import build_contract,add_unknown_research,attach_potential
from build_final_annotation_sample import extend
from validate_final_annotation_sample import audit_temporal
import jsonschema


def generate(root):
    root=Path(root).resolve()
    if root.exists() and any(root.iterdir()):raise ValueError('Demo directory must be new or empty')
    evidence=root/'evidence';evidence.mkdir(parents=True,exist_ok=True)
    eid='synthetic_episode_001';event_id=eid+':search_triplet_01';states=[]
    for i,(phase,frame) in enumerate(zip(['start','middle','end'],[1,4,7])):
        im=Image.new('RGB',(96,64),'white');draw=ImageDraw.Draw(im)
        draw.rectangle((0,40,95,63),fill='tan');draw.rectangle((20+i*4,20,40+i*4,42),fill='blue')
        p=evidence/(phase+'.png');im.save(p)
        states.append({'phase':phase,'original_video_frame_index':frame,'original_video_seconds':frame/30,
            'image_filename':p.name,'image_path':str(p),'image_sha256':sha(p),'frame_namespace':'original_video_frame_index_from_filename'})
    md=evidence/'event.md';md.write_text('SYNTHETIC TEST ONLY\n要找的物体：blue cup\n帧率 30 帧/秒\n| 第1次 | 1 | 4 | 7 |\n')
    row={'episode_id':eid,'fps':30,'target':'blue cup','event_images':str(evidence),'target_markdown':str(md),
         'target_markdown_sha256':sha(md),'event_count':1,
         'event_index':{'events':[{'event_id':event_id,'ordinal':1,'states':states,'sample_window_s':[1/30,7/30]}]},
         'source_annotation_metadata':{k:None for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']}}
    row['source_annotation_metadata']['要找的物体']='blue cup'
    scene=assemble({'workspace_name':'synthetic scene','objects':[{'id':'o01','name':'blue cup','category':'cup',
        'entity_type':'individual','shape':'rectangle in synthetic image','colors':['blue'],'commonsense':['Synthetic fixture, not an observed real object.'],
        'affordances':{},'parent':'region:workspace','predicate':'within_region','placement':'center', 'frames':[1,4,7]}],
        'relations':[],'identity_candidates':[],'uncertainty':['Synthetic fixture: no real inference or semantic acceptance.']})['scene_memory']
    payload={'event_summary':'A blue cup is visible.','observed_changes':['The blue cup changes image position.'],
        'search_region':'blue cup','uncertainty':'Synthetic example; no human search occurred.',
        'states':[{'phase':s['phase'],'state_description':'A blue cup is visible.',
            'hand_action':'No hand is shown.','focus_object_states':[{'label':'blue cup','observed_state':'visible'}],
            'visibility_and_occlusion':'Blue cup is fully visible in the synthetic drawing.'} for s in states],
        'potential_search_regions':[]}
    source=export_episode(row,scene,[payload]);research=add_unknown_research(source,row);attach_potential(research,[payload])
    intermediate=root/'intermediate';intermediate.mkdir();source_path=intermediate/'source.json';write(source_path,research)
    graph=initialize_temporal_graph(scene);temporal=deepcopy(source)
    temporal['scene_memory']['spatial_structure']=nested_temporal_tree(graph)
    for event in temporal['search_event_states']['events']:
        event['spatial_changes']=[];event['spatial_change_annotation']={'status':'local_qwen_visual_inference_unreviewed',
            'uncertainties':['Synthetic fixture; no Qwen inference was performed.'],'source':'original_event_triplet_images','description_language':'en'}
    temporal_path=intermediate/'temporal.json';write(temporal_path,temporal)
    final=extend(research,source_path,temporal,graph,temporal_path)
    final['README']['annotation_record']['label_source']='synthetic_test_fixture_no_model_calls'
    final['README']['annotation_record']['new_model_calls']=0
    jsonschema.validate(final,build_contract());audit_temporal(final)
    write(root/'json'/f'{eid}.json',final)
    write(root/'anchor_sources.json',{'groups':[{'name':'synthetic','directory':'json'}]})
    print(json.dumps({'status':'created','synthetic':True,'model_calls':0,'root':str(root)},ensure_ascii=False))
    return final

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    generate(p.parse_args().output)
