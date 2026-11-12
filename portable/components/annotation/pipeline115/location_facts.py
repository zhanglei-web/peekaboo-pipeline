"""Preserve owner-supplied location descriptions without inventing unit identities."""
from copy import deepcopy
from pathlib import Path
import re
import jsonschema
from .common import sha

def field(text,name):
    matches=re.findall(r'^'+re.escape(name)+r'\s*[：:]\s*(.*?)\s*$',text,re.M)
    if len(matches)!=1: raise ValueError('Missing or ambiguous Markdown field: '+name)
    return matches[0]

def object_schema(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

def schema():
    text={'type':'string','minLength':1}; nullable={'type':['string','null']}
    return object_schema({
        'schema':{'const':'owner_supplied_episode_locations.v1'},
        'episode_id':text,'scene_id':text,'target_name':text,
        'initial_location':object_schema({
            'status':{'enum':['recorded_filename','not_provided','unresolved_alias']},
            'description':nullable,'source_file':nullable,'source_sha256':nullable,
            'filename':nullable,'target_name_in_filename':nullable,
            'matching_rule':text,'raw_markdown_value':nullable}),
        'actual_location':object_schema({
            'description':text,'source_file':text,'source_sha256':text,
            'field_name':{'const':'物体最后所在位置'},
            'source_type':{'const':'owner_confirmed_markdown'},
            'temporal_scope':{'const':'episode_level_record_no_within_episode_relocation_time'},
            'associated_event_ids':{'type':'array','minItems':1,'uniqueItems':True,'items':text},
            'search_unit_id':{'type':'null'},'relocation_time_s':{'type':'null'}}),
        'interpretation':text})

def contract(base):
    value=deepcopy(base)
    value['properties']['metadata']['properties']['dataset_location_facts']=schema()
    from .input_exceptions import schema as exception_schema
    value['properties']['metadata']['properties']['input_evidence_exception']=exception_schema()
    return value

def verify(row):
    from .input_exceptions import verify as verify_exception
    verify_exception(row)
    facts=row.get('dataset_location_facts')
    if facts is None:return
    jsonschema.validate(facts,schema())
    if (facts['episode_id'],facts['scene_id'],facts['target_name'])!=(row['episode_id'],row['scene_group'],row['target']):
        raise ValueError('Location facts belong to another episode/scene/target')
    actual=facts['actual_location'];md=Path(row['target_markdown'])
    if actual['source_file']!=str(md) or actual['source_sha256']!=row['target_markdown_sha256'] or sha(md)!=actual['source_sha256']:
        raise ValueError('Location Markdown identity changed')
    text=md.read_text(encoding='utf-8-sig')
    if actual['description']!=field(text,'物体最后所在位置'):
        raise ValueError('Actual location differs from original Markdown')
    if actual['associated_event_ids']!=[e['event_id'] for e in row['event_index']['events']]:
        raise ValueError('Location episode/event association changed')
    initial=facts['initial_location']
    if initial['raw_markdown_value']!=field(text,'物体原始位置'):
        raise ValueError('Initial location source metadata changed')
    if initial['status']=='recorded_filename':
        photo=Path(initial['source_file']);prefix=row['scene_group']+initial['target_name_in_filename']
        if initial['matching_rule']=='same scene directory; exact target filename without scene prefix':
            prefix=initial['target_name_in_filename']
        if photo.parent.name!=row['scene_group'] or photo.name!=initial['filename'] or sha(photo)!=initial['source_sha256']:
            raise ValueError('Initial photo identity/scene changed')
        if not photo.stem.startswith(prefix) or photo.stem[len(prefix):].lstrip('—-_ ' )!=initial['description']:
            raise ValueError('Initial location differs from source filename')
    elif any(initial[k] is not None for k in ('description','source_file','source_sha256','filename','target_name_in_filename')):
        raise ValueError('Unresolved initial location must remain unspecified')

def attach(doc,row):
    verify(row)
    if (doc['metadata']['episode_id'],doc['metadata']['target']['description'])!=(row['episode_id'],row['target']):
        raise ValueError('Cannot attach location facts to a different annotated episode/target')
    if row.get('input_evidence_exception'):
        doc['metadata']['input_evidence_exception']=deepcopy(row['input_evidence_exception'])
        doc['README']['rules'].append('The owner explicitly approved annotation using the complete left-eye triplets despite one unavailable right-eye frame beyond the end of its original video. No substitute frame was used; metadata.input_evidence_exception retains exact scope and source hashes.')
    if row.get('dataset_location_facts'):
        doc['metadata']['dataset_location_facts']=deepcopy(row['dataset_location_facts'])
        doc['README']['rules'].append('Initial location descriptions come from the same-scene reference photo filename; actual placement descriptions come from the episode Markdown as confirmed by the dataset owner. These records do not establish a visual tree node, search unit identity, relocation time, or search-event outcome.')
    return doc

def project_exports(doc,episode,experiment):
    facts=doc['metadata'].get('dataset_location_facts')
    if facts:
        for output in (episode,experiment):
            output['hidden']['location_descriptions']=deepcopy(facts)
    if doc['metadata'].get('input_evidence_exception'):
        for output in (episode,experiment):
            output.setdefault('metadata',{})['input_evidence_exception']=deepcopy(doc['metadata']['input_evidence_exception'])
    return episode,experiment
