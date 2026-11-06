"""Default new-data pipeline: nested temporal trees, one VLM request per event."""
import fcntl
import json
from pathlib import Path
import time

import jsonschema
import psutil

from run_scene_event_batch import (inventory, prepare_episode, scene_job, export_episode,
                                   atomic, sha, now)
from run_temporal_episode import annotate_event
from temporal_spatial_v2 import initialize_temporal_graph, nested_temporal_tree
from batch_temporal_trees import (VERSION, output_schema, merge_observations,
                                 validate_output)


def run(todo, output, inventory_only=False, limit=None):
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    schema_path=output/'target.schema.json'
    if schema_path.exists():
        old=json.loads(schema_path.read_text())
        if old.get('properties',{}).get('schema',{}).get('const')!=VERSION:
            raise ValueError('Existing output uses a legacy schema. Choose a new output directory; do not overwrite historical evidence.')
    lock=(output/'.batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=inventory(todo,output)
    schema=output_schema('spatial_memory.search_episode.p0_p1_2.v1')
    atomic(schema_path,schema)
    ready=[r for r in rows if not r['problems']]
    if limit:ready=ready[:limit]
    if inventory_only:
        print(json.dumps({'ready':len(ready),'blocked':sum(bool(r['problems']) for r in rows),'output_schema':VERSION}));return
    me=psutil.Process();started=now();tick=time.perf_counter()
    atomic(output/'launch.json',{'pid':me.pid,'create_time':me.create_time(),'command':me.cmdline(),
                                'started_at':started,'annotation_version':VERSION})
    state={'status':'running','started_at':started,'episode_count':len(ready),'completed_count':0,
           'annotation_version':VERSION,'episodes':{}}
    def update(eid,**values):
        state['episodes'].setdefault(eid,{}).update(values)
        state['completed_count']=sum(r.get('status')=='complete' for r in state['episodes'].values())
        state['updated_at']=now();atomic(output/'batch_status.json',state)
    for row in ready:
        eid=row['episode_id'];folder=output/'episodes'/eid;dest=output/'json'/(eid+'.json')
        marker=folder/'temporal_tree_complete.json'
        try:
            if marker.exists():
                receipt=json.loads(marker.read_text());doc=json.loads(dest.read_text())
                if (receipt['sha256']!=sha(dest) or receipt['schema_sha256']!=sha(schema_path)
                    or receipt['target_markdown_sha256']!=row['target_markdown_sha256']
                    or receipt['source_sha256']!=sha(folder/'visual_source.json')
                    or receipt['parent_graph_sha256']!=sha(folder/'parent_graph.json')):
                    raise ValueError('Temporal completion provenance mismatch')
                source=json.loads((folder/'visual_source.json').read_text())
                validate_output(source,doc,schema);update(eid,status='complete',cache_reused=True);continue
            update(eid,status='preparing')
            episode_tick=time.perf_counter();folder=prepare_episode(row,output)
            scene=scene_job(folder/'selection')
            graph=initialize_temporal_graph(scene)
            atomic(folder/'reference_graph.json',graph)
            data=[];changes=[];merge_audits=[]
            # Generate descriptions and compact parent changes together. No second event call.
            for event in row['event_index']['events']:
                update(eid,status='event_annotation',event_ordinal=event['ordinal'])
                event_folder=folder/'events_temporal'/f"event_{event['ordinal']:02d}"
                error=None
                for attempt in range(1,4):
                    try:
                        payload=annotate_event(event,event_folder/f'attempt_{attempt:03d}',graph,row['target'],correction=error)
                        frames=[s['original_video_frame_index'] for s in event['states']]
                        new_graph,normalized,audit=merge_observations(graph,payload['spatial_changes'],event['event_id'],frames)
                        break
                    except Exception as exc:
                        error=repr(exc);atomic(event_folder/f'failure_{attempt:03d}.json',{'error':error,'at':now()})
                else:raise RuntimeError(error)
                graph=new_graph;data.append(payload);changes.append(normalized);merge_audits.extend(audit)
            source=export_episode(row,scene,data)
            atomic(folder/'visual_source.json',source)
            # The raw static scene is evidence only. The delivered JSON is always temporal.
            doc=json.loads(json.dumps(source));doc['schema']=VERSION
            doc['scene_memory']['spatial_structure']=nested_temporal_tree(graph)
            for event,updates,payload in zip(doc['search_event_states']['events'],changes,data,strict=True):
                event['spatial_changes']=updates
                event['spatial_change_annotation']={'status':'local_qwen_visual_inference_unreviewed',
                    'uncertainties':[payload['uncertainty']] if payload.get('uncertainty') else [],
                    'source':'original_event_triplet_images','description_language':'en'}
            doc['README']['scope']=[s for s in doc['README']['scope'] if s!='物体历史及最后观测按用户要求删除']
            doc['README']['scope'].append('Nested tree with original-video-frame direct-parent intervals.')
            doc['README']['rules'].append('Temporal children are historical occurrences of stable IDs; filter by frame. Unknown-parent periods are unlocated. New event and relation descriptions are English. Visual semantics remain unreviewed.')
            validate_output(source,doc,schema)
            atomic(dest,doc);atomic(folder/'parent_graph.json',graph);atomic(folder/'merge_audit.json',merge_audits)
            atomic(folder/'temporal_timing.json',{'wall_seconds':time.perf_counter()-episode_tick,
                'event_count':len(data),'finished_at':now(),'note':'Actual episode wall time, including cache reuse and scheduling. No sum of concurrent request durations.'})
            atomic(marker,{'sha256':sha(dest),'schema_sha256':sha(schema_path),
                'source_sha256':sha(folder/'visual_source.json'),'parent_graph_sha256':sha(folder/'parent_graph.json'),
                'target':row['target'],'target_markdown_sha256':row['target_markdown_sha256'],'finished_at':now()})
            update(eid,status='complete',output=str(dest))
        except Exception as exc:
            update(eid,status='failed',error=repr(exc))
    state['status']='complete' if state['completed_count']==len(ready) else 'needs_attention'
    state['finished_at']=now();state['wall_seconds']=time.perf_counter()-tick
    atomic(output/'batch_status.json',state)
    print(json.dumps({k:v for k,v in state.items() if k!='episodes'}))
    return state
