"""Annotate direct-parent changes from existing triplets; preserve all source labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import threading
import time

import jsonschema
from PIL import Image, ImageDraw, ImageFont
import psutil

from run_scene_event_batch import local_call, full_contract, atomic, sha, now, ROOT
from export_research_fields import schema_for, verify
from temporal_spatial_v2 import (initialize_temporal_graph, build_spatial_change_prompt,
    event_spatial_change_schema, object_schema, apply_spatial_changes, _active_relation,
    refresh_child_relations, validate_temporal_graph, nested_temporal_tree,
    validate_temporal_tree, temporal_tree_schema)

VERSION = 'spatial_memory.search_episode.temporal_tree_batch.v1'
TEXT = {'type': 'string', 'minLength': 1}
LOCK = threading.Lock()
ANNOTATION_SCHEMA = object_schema({
    'status': {'const': 'local_qwen_visual_inference_unreviewed'},
    'uncertainties': {'type': 'array', 'items': TEXT},
    'source': {'const': 'original_event_triplet_images'},
    'description_language': {'const': 'en'},
})


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def output_schema(source_version):
    base = full_contract()
    if source_version == 'spatial_memory.search_episode.research_fields.v1':
        base = schema_for(base)
    elif source_version != 'spatial_memory.search_episode.p0_p1_2.v1':
        raise ValueError('Unsupported source schema')
    tree = temporal_tree_schema()
    base.setdefault('$defs', {}).update(tree.pop('$defs'))
    base['properties']['schema'] = {'const': VERSION}
    base['properties']['scene_memory']['properties']['spatial_structure'] = tree
    ev = base['properties']['search_event_states']['properties']['events']['items']
    ev['properties']['spatial_changes'] = event_spatial_change_schema(None)
    ev['properties']['spatial_change_annotation'] = ANNOTATION_SCHEMA
    ev['required'] += ['spatial_changes', 'spatial_change_annotation']
    return base


def merge_observations(graph, changes, event_id, frames):
    """Absolute observations, with update types derived from current graph state.

    Same-frame event evidence may supersede a mixed-view scene reference; keep
    that reference in the merge audit. Conflicting event observations are rejected.
    """
    jsonschema.validate(changes, event_spatial_change_schema(frames))
    updated = deepcopy(graph)
    normalized, audit = [], []
    # A mixed-view reference is a fallback initialization, not a future event.
    # Once an object has a time-specific event observation, do not let a later
    # provisional scene anchor silently overwrite the observed state.
    first_observations = {}
    for change in changes:
        nid = change['node_id']
        first_observations[nid] = min(first_observations.get(nid, change['change_frame']), change['change_frame'])
    for nid, first in first_observations.items():
        if nid not in updated['nodes'] or nid == updated['root_id']:
            raise ValueError('Unknown/root node in observation')
        edges = updated['nodes'][nid]['parent_relations']
        removed = [r for r in edges if r['boundary_status'] == 'scene_reference_observation_unreviewed'
                   and r['valid_from_frame'] >= first]
        for relation in removed:
            edges.remove(relation)
            audit.append({'reason':'event_observation_supersedes_provisional_scene_reference',
                          'node_id':nid,'reference_relation':deepcopy(relation),'event_id':event_id})
    refresh_child_relations(updated)
    groups = {}
    for change in sorted(changes, key=lambda c:c['change_frame']):
        groups.setdefault(change['change_frame'], []).append(change)
    for frame, items in groups.items():
        batch = []
        for raw in items:
            change = deepcopy(raw)
            nid = change['node_id']
            if nid not in updated['nodes'] or nid == updated['root_id']:
                raise ValueError('Unknown/root node in observation')
            current = _active_relation(updated['nodes'][nid], frame)
            if current and current['valid_from_frame'] == frame:
                same = all(current[k] == change[k] for k in ('parent_id', 'predicate', 'relative_position'))
                if not same and current['boundary_status'] == 'scene_reference_observation_unreviewed':
                    audit.append({'reason':'event_observation_supersedes_same_frame_scene_reference',
                                  'node_id':nid, 'reference_relation':deepcopy(current), 'event_id':event_id})
                    updated['nodes'][nid]['parent_relations'].remove(current)
                    current = _active_relation(updated['nodes'][nid], frame)
            if current and current['valid_from_frame'] < frame:
                same_parent = (current['parent_id'], current['predicate']) == (change['parent_id'], change['predicate'])
                same_position = current['relative_position'] == change['relative_position']
                if same_parent and (same_position or raw['change_type'] == 'establish'):
                    audit.append({'reason':'reconfirmed_relation_without_supported_change',
                                  'node_id':nid,'frame':frame,'observation':deepcopy(raw)})
                    continue
            if change['parent_id'] is None:
                if change['predicate'] != 'unknown' or change['relative_position'] is not None:
                    raise ValueError('Unknown parent requires unknown predicate and null relative_position')
                kind = 'unparent'
            elif current is None or current['parent_id'] is None:
                kind = 'establish'
            elif current['parent_id'] != change['parent_id']:
                kind = 'reparent'
            else:
                kind = 'position_change'
            if kind != raw['change_type']:
                audit.append({'reason':'update_type_derived_from_actual_state', 'node_id':nid,
                              'frame':frame,'model_type':raw['change_type'],'applied_type':kind})
            change['change_type'] = kind
            batch.append(change)
        refresh_child_relations(updated)
        updated = apply_spatial_changes(updated, batch, event_id, frames)
        normalized.extend(batch)
    return updated, normalized, audit


def prepare_images(event, folder):
    folder.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB',(2880,640),'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    for i, state in enumerate(event['states']):
        p = Path(state['image_path'])
        if sha(p) != state['image_sha256']:
            raise ValueError('Event image hash mismatch: '+str(p))
        with Image.open(p) as image:
            canvas.paste(image.convert('RGB').resize((960,600)),(i*960,40))
        draw.text((i*960+8,5),f"{state['phase']} | f{state['original_video_frame_index']}",fill='black',font=font)
    paths = [folder/'triplet.jpg']
    canvas.save(paths[0],quality=95,subsampling=0)
    for i, state in enumerate(event['states']):
        p = folder/(state['phase']+'_detail.jpg')
        canvas.crop((i*960,0,(i+1)*960,640)).save(p,quality=95,subsampling=0)
        paths.append(p)
    return paths


def annotate_event(doc, event, graph, folder):
    frames = [s['original_video_frame_index'] for s in event['states']]
    changes = event_spatial_change_schema(frames)
    props = changes['items']['properties']
    props['node_id']['enum'] = [nid for nid in graph['nodes'] if nid != graph['root_id']]
    props['parent_id']['enum'] = [None,*graph['nodes']]
    response_schema = object_schema({
        'frame_support_observations': {'type':'array','minItems':3,'maxItems':3,
            'items': object_schema({'frame':{'type':'integer','enum':frames}, 'observation':TEXT})},
        'spatial_changes': changes, 'uncertainties': {'type':'array','items':TEXT}})
    prompt = ('Annotate direct-parent and parent-relative position observations for ALL directly manipulated catalog objects in these three images, regardless of what the person is searching for. This is scene-state tracking, not target tracking. A moved laptop, tablet, mouse, container, or other catalog object MUST be considered even when unrelated to the search target. '
        'FIRST describe in frame_support_observations the visible physical support of the directly manipulated objects at EACH of the three frames, including whether hands merely touch a resting object or visibly hold it clear of its support. Use short factual English sentences. THEN derive compact spatial_changes consistent with those observations and short English uncertainties. Existing event prose and outcome labels '
        'will be preserved separately; do not rewrite them or infer target success. An empty list means no '
        'supported update, not proof that nothing moved. Do not classify touch alone as removal. '
        'Moving a container changes the container parent, not the direct parent of objects still inside it. NEVER unparent contents merely because their container is carried by a hand; those contents remain inside the same container unless visibly removed. Establish a missing inside relation if the contents are identifiable. For each directly manipulated catalog object, establish its visible parent at the first panel when '
        'missing or inconsistent, even when it never moves across these three frames. No motion does not mean no initial parent observation. Then report only supported subsequent changes. Only use supplied catalog IDs. '
        'If identity cannot be matched, explain uncertainty rather than guessing. '
        'No coordinates, neighbors, full graphs or unchanged background inventories. Relative position must describe the direct parent only, such as left side of the tabletop, never left of another object. '
        'Never assert a known support while explaining that support is assumed or occluded. If removal is not visible and support is uncertain, omit the observation and state uncertainty. Use relative_position=null when exact parent-relative position is uncertain. Hand contact alone does not establish a position change. Do not invent left/center/front changes from camera motion. Object appearance and identities are reused estimates, not ground truth.\n')
    prompt += build_spatial_change_prompt(graph, frames)
    prompt += '\nCompact appearance catalog: '+json.dumps([
        {'id':o['spatial_node_id'],'appearance':o.get('appearance')} for o in doc['scene_memory']['objects']
    ],ensure_ascii=False,separators=(',',':'))
    cache_key = digest({'prompt':prompt,'schema':response_schema,
        'images':[s['image_sha256'] for s in event['states']], 'graph':graph})
    cached = folder/'accepted.json'
    if cached.exists():
        result = json.loads(cached.read_text())
        if result['input_key'] != cache_key:
            raise ValueError('Accepted temporal cache has different input')
        updated, normalized, audit = merge_observations(graph,result['raw_changes'],event['event_id'],frames)
        return updated, normalized, result['uncertainties'], True
    paths = prepare_images(event,folder)
    error = None
    # Attempt folders are immutable evidence, including malformed/failed responses.
    next_attempt = 1 + max([int(p.name.split('_')[1]) for p in folder.glob('attempt_*') if p.is_dir()] or [0])
    for attempt in range(next_attempt,next_attempt+3):
        attempt_dir = folder/f'attempt_{attempt:03d}'
        attempt_prompt = prompt
        if error:
            attempt_prompt += '\nThe previous response failed consistency validation: '+str(error)[:1000]+'. Reinspect the images and output supported absolute relations; do not fabricate an update merely to satisfy validation.'
        try:
            result = local_call(attempt_dir,attempt_prompt,paths,response_schema,3072)
            if [o['frame'] for o in result['frame_support_observations']] != frames:
                raise ValueError('Frame support observations must follow all three supplied frames in order')
            updated, normalized, audit = merge_observations(graph,result['spatial_changes'],event['event_id'],frames)
            atomic(cached,{'input_key':cache_key,'raw_changes':result['spatial_changes'],
                'spatial_changes':normalized,'uncertainties':result['uncertainties'],
                'merge_audit':audit,'frame_support_observations':result['frame_support_observations'],
                'accepted_attempt':attempt,'accepted_at':now(),
                'response_sha256':sha(attempt_dir/'response.json')})
            return updated,normalized,result['uncertainties'],False
        except Exception as exc:
            error = exc
            atomic(attempt_dir/'failure.json',{'error':repr(exc),'at':now()})
            if 'Connection' in type(exc).__name__ or 'Timeout' in type(exc).__name__:
                time.sleep(10)
    raise RuntimeError(f'Event failed three attempts: {event["event_id"]}: {error}')


def assert_preserved(source, output):
    a, b = deepcopy(source), deepcopy(output)
    for d in (a,b):
        d.pop('schema'); d.pop('README')
        d['scene_memory'].pop('spatial_structure')
        for e in d['search_event_states']['events']:
            e.pop('spatial_changes',None);e.pop('spatial_change_annotation',None)
    if a != b:
        raise ValueError('Source metadata, object profiles, events or research labels changed')


def validate_output(source, doc, schema):
    jsonschema.validate(doc,schema)
    validate_temporal_tree(doc['scene_memory']['spatial_structure'])
    assert_preserved(source,doc)
    if source['schema'] == 'spatial_memory.search_episode.research_fields.v1':
        verify(doc)


def process_episode(src, output, evidence, state, publish):
    eid = src.stem
    folder = evidence/'episodes'/eid
    folder.mkdir(parents=True,exist_ok=True)
    frozen = evidence/'source_json'/src.name
    if sha(src) != sha(frozen):
        raise ValueError('Original source modified since preflight: '+eid)
    doc = json.loads(frozen.read_text())
    schema = output_schema(doc['schema'])
    schema_path = evidence/'schemas'/(doc['schema']+'.json')
    # Schemas are published once during preflight, avoiding shared writes.
    marker = folder/'complete.json'; dest = output/src.name
    if marker.exists():
        receipt=json.loads(marker.read_text())
        if receipt['source_sha256']!=sha(frozen) or receipt['output_sha256']!=sha(dest) or receipt['schema_sha256']!=sha(schema_path):
            raise ValueError('Completion receipt mismatch')
        validate_output(doc,json.loads(dest.read_text()),schema)
        publish(eid,{'status':'complete','events':len(doc['search_event_states']['events']),'cached_episode':True})
        return
    started=now();tick=time.perf_counter()
    graph=initialize_temporal_graph(doc['scene_memory'])
    atomic(folder/'reference_graph.json',graph)
    result=deepcopy(doc); cached_count=0;update_count=0
    for index,event in enumerate(result['search_event_states']['events'],1):
        publish(eid,{'status':'running','event_index':index,'events':len(result['search_event_states']['events'])})
        graph,changes,uncertainties,cached=annotate_event(doc,event,graph,folder/f'event_{index:02d}')
        cached_count+=int(cached);update_count+=len(changes)
        event['spatial_changes']=changes
        event['spatial_change_annotation']={'status':'local_qwen_visual_inference_unreviewed',
            'uncertainties':uncertainties,'source':'original_event_triplet_images','description_language':'en'}
        publish(eid,{'status':'running','event_index':index,'completed_events':index,
                     'events':len(result['search_event_states']['events'])})
    result['schema']=VERSION
    result['scene_memory']['spatial_structure']=nested_temporal_tree(graph)
    result['README']['scope']=[s for s in result['README']['scope'] if s!='物体历史及最后观测按用户要求删除']
    result['README']['scope'].append('Time-bounded direct-parent relationships in a nested historical tree; original object profiles and event prose retained.')
    result['README']['rules'].append('Nested children are temporal occurrences of stable object IDs, not simultaneous duplicates. Node frame intervals are clipped by ancestor paths; placement intervals retain direct-parent evidence. Unknown-parent periods are unlocated. All new relations remain unreviewed VLM inferences. Existing outcome and research labels are unchanged.')
    validate_output(doc,result,schema)
    if sha(src)!=sha(frozen):raise ValueError('Source changed while processing')
    atomic(dest,result);atomic(folder/'parent_graph.json',graph)
    timing={'started_at':started,'finished_at':now(),'wall_seconds':time.perf_counter()-tick,
            'event_count':len(result['search_event_states']['events']),'cached_events':cached_count,
            'relation_updates':update_count,'note':'Episode wall time includes service queueing and CPU assembly. Concurrent episode/request durations must not be summed as batch wall time.'}
    atomic(folder/'timing.json',timing)
    atomic(marker,{'episode_id':eid,'source_sha256':sha(frozen),'output_sha256':sha(dest),
                   'schema_sha256':sha(schema_path),'parent_graph_sha256':sha(folder/'parent_graph.json'),
                   'finished_at':now(),'semantic_status':'unreviewed'})
    publish(eid,{'status':'complete','events':timing['event_count'],'completed_events':timing['event_count'],
                 'relation_updates':update_count,'wall_seconds':timing['wall_seconds']})


def run(source,output,evidence,only=None):
    output.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
    lock=(evidence/'.batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files=sorted(source.glob('*.json'))
    if only:files=[p for p in files if p.stem in only]
    if not files:raise ValueError('No source JSON files')
    manifest=[]
    for p in files:
        doc=json.loads(p.read_text());eid=p.stem
        if doc['metadata']['episode_id']!=eid:raise ValueError('Episode filename mismatch')
        base=full_contract()
        if doc['schema']=='spatial_memory.search_episode.research_fields.v1':base=schema_for(base)
        jsonschema.validate(doc,base);initialize_temporal_graph(doc['scene_memory'])
        images=[]
        for ev in doc['search_event_states']['events']:
            for s in ev['states']:
                if sha(Path(s['image_path']))!=s['image_sha256']:raise ValueError('Image mismatch')
                images.append({'path':s['image_path'],'sha256':s['image_sha256'],'frame':s['original_video_frame_index']})
        manifest.append({'episode_id':eid,'path':str(p),'sha256':sha(p),'source_schema':doc['schema'],
                         'event_count':len(doc['search_event_states']['events']),'images':images})
        frozen=evidence/'source_json'/p.name
        if frozen.exists():
            if sha(frozen)!=sha(p):raise ValueError('Frozen source mismatch')
        else:
            frozen.parent.mkdir(parents=True,exist_ok=True);frozen.write_bytes(p.read_bytes())
        atomic(evidence/'schemas'/(doc['schema']+'.json'),output_schema(doc['schema']))
    manifest_data={'episodes':manifest,'source_directory':str(source),'output_directory':str(output),
        'code_sha256':{p:sha(ROOT/p) for p in ('scripts/batch_temporal_trees.py','scripts/temporal_spatial_v2.py','prompts/spatial_relation_changes.en.txt')}}
    manifest_path=evidence/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest_data:
        raise ValueError('Batch manifest changed; use a separate evidence directory')
    atomic(manifest_path,manifest_data)
    for name in manifest_data['code_sha256']:
        target=evidence/'implementation'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    servers=[p.info for p in psutil.process_iter(['pid','create_time','cmdline'])
             if p.info['cmdline'] and Path(p.info['cmdline'][0]).name=='llama-server']
    atomic(evidence/'model_processes.json',{'checked_at':now(),'processes':servers})
    me=psutil.Process();atomic(evidence/'launch.json',{'pid':me.pid,'create_time':me.create_time(),'command':me.cmdline(),'started_at':now()})
    state={'status':'running','started_at':now(),'episode_count':len(files),
           'event_count':sum(r['event_count'] for r in manifest),'completed_count':0,'episodes':{}}
    tick=time.perf_counter()
    def publish(eid,info):
        with LOCK:
            old=state['episodes'].get(eid,{})
            if info.get('status')=='complete':old.pop('error',None)
            old.update(info);state['episodes'][eid]=old
            state['completed_count']=sum(x.get('status')=='complete' for x in state['episodes'].values())
            state['updated_at']=now();atomic(evidence/'batch_status.json',state)
    pending=files
    for round_index in range(3):
        state['retry_round']=round_index
        failed=[]
        with ThreadPoolExecutor(max_workers=2) as pool:
            tasks={pool.submit(process_episode,p,output,evidence,state,publish):p for p in pending}
            for task in as_completed(tasks):
                p=tasks[task]
                try:task.result()
                except Exception as exc:
                    failed.append(p);publish(p.stem,{'status':'failed','error':repr(exc)})
                    print(p.stem,repr(exc),flush=True)
        if not failed:break
        pending=failed
    state['status']='complete' if state['completed_count']==len(files) else 'needs_attention'
    state['finished_at']=now();state['wall_seconds']=time.perf_counter()-tick
    state['episodes_per_hour']=state['completed_count']*3600/state['wall_seconds']
    atomic(evidence/'batch_status.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='episodes'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--evidence',required=True,type=Path)
    parser.add_argument('--only',nargs='+')
    args=parser.parse_args();run(args.source,args.output,args.evidence,args.only)
