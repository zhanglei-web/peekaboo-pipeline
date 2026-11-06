"""Strict schema for the approved additive format, preserving original contracts."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import jsonschema

sys.path.insert(0, '/home/baai/zl/Qwen3.8-27B/scripts')
from run_scene_event_batch import full_contract
from export_research_fields import schema_for
from temporal_spatial_v2 import temporal_graph_schema, event_spatial_change_schema
from batch_temporal_trees import ANNOTATION_SCHEMA

ROOT = Path(__file__).resolve().parent
TEXT = {'type':'string'}
NULL_TEXT = {'type':['string','null']}
NULL_NUM = {'type':['number','null']}


def shape(value):
    if isinstance(value, dict):
        props={k:shape(v) for k,v in value.items()}
        return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
    if isinstance(value,list):
        return {'type':'array','items':shape(value[0]) if value else {}}
    if value is None:return {'type':'null'}
    return {'type': 'boolean' if isinstance(value,bool) else 'integer' if isinstance(value,int) else 'number' if isinstance(value,float) else 'string'}


def add(schema, fields):
    schema['properties'].update(deepcopy(fields))
    schema['required']=list(dict.fromkeys(schema.get('required',[])+list(fields)))


def build():
    sample=json.loads((ROOT/'final_annotation_sample.json').read_text())
    s=schema_for(full_contract())
    add(s['properties']['metadata'],{k:NULL_NUM if k.endswith('_s') else NULL_TEXT for k in [
        'target_description_to_participant','mobility_tier','video_duration_s','search_start_unit',
        'search_start_s','search_end_s','boundary_sampling_interval_s']})
    add(s['properties']['search_units']['properties']['list']['items'],{
        'room':NULL_TEXT,'parent_id':NULL_TEXT,'inspect_cost_s':NULL_NUM,
        'inspect_cost_trials_s':{'anyOf':[{'type':'null'},{'type':'array','items':{'type':'number','minimum':0}}]},
        'volume_l':NULL_NUM,'surface_vs_enclosed':NULL_TEXT})
    add(s['properties']['inspections']['items'],{
        'validity':{'type':['string','number','null']},'start_s':NULL_NUM,'end_s':NULL_NUM,
        'obstruction':NULL_TEXT,'revisit_kind':NULL_TEXT})
    add(s['properties']['episode_outcome'],{
        'found_evidence_frames':{'anyOf':[{'type':'null'},{'type':'array','items':TEXT}]},
        'stop_reason':NULL_TEXT,'duration_s':NULL_NUM})
    add(s['properties']['hidden'],{
        'original_unit_id':NULL_TEXT,'relocation_source':NULL_TEXT,'relocation_evidence_state':NULL_TEXT,
        'evidence_coverage':NULL_NUM,'evidence_confidence':NULL_NUM,'intervention_time_h':NULL_NUM})
    counts={'type':'object','additionalProperties':{'type':['number','null']}}
    block={'type':'object','properties':{'counts':counts},'required':['counts'],'additionalProperties':False}
    add(s,{'priors':{'type':'object','properties':{
        'habit_prior':block,'cohabitant_counts':deepcopy(block),
        'semantic_scores':{'type':'object','properties':{'source':NULL_TEXT,'scores':counts},
                           'required':['source','scores'],'additionalProperties':False}},
        'required':['habit_prior','cohabitant_counts','semantic_scores'],'additionalProperties':False}})
    readme=s['properties']['README']
    for name in ['format_extension','annotation_record','completeness']:
        add(readme,{name:shape(sample['README'][name])})
    record=readme['properties']['annotation_record']['properties']
    for name in ['manifest','completion','home_id_basis','temporal_annotation_json']:
        record[name]=deepcopy(NULL_TEXT)
    record['model_evidence_available_after_s']=deepcopy(NULL_NUM)
    structure=s['properties']['scene_memory']['properties']['spatial_structure']
    add(structure,{'timebase':shape(sample['scene_memory']['spatial_structure']['timebase']),
                   'temporal_status':TEXT})
    relation=deepcopy(temporal_graph_schema()['properties']['nodes']['additionalProperties']['properties']['parent_relations']['items'])
    # Evidence keys are present only when the corresponding boundary has an event.
    relation['properties'].update(start_evidence_state_id=deepcopy(TEXT),end_evidence_state_id=deepcopy(TEXT))
    add(s['$defs']['node'],{'parent_relations':{'anyOf':[{'type':'null'},{'type':'array','items':relation}]}})
    changes=event_spatial_change_schema(None)
    add(changes['items'],{'evidence_state_id':TEXT})
    ev=s['properties']['search_event_states']['properties']['events']['items']
    add(ev,{'spatial_changes':{'anyOf':[{'type':'null'},changes]},
            'spatial_change_annotation':{'anyOf':[deepcopy(ANNOTATION_SCHEMA),
                {'type':'object','properties':{'status':{'const':'not_annotated'}},'required':['status'],'additionalProperties':False}]}})
    jsonschema.Draft202012Validator.check_schema(s)
    jsonschema.validate(sample,s)
    return s
