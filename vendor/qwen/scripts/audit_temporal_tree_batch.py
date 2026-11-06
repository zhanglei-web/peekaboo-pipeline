"""Independently verify and summarize a completed temporal-tree batch."""
import argparse
import json
from pathlib import Path
import statistics

from batch_temporal_trees import output_schema, validate_output, merge_observations
from temporal_spatial_v2 import initialize_temporal_graph, nested_temporal_tree
from run_scene_event_batch import atomic, sha, now


def audit(source, output, evidence):
    manifest=json.loads((evidence/'manifest.json').read_text())
    rows=[];errors=[]
    for record in manifest['episodes']:
        eid=record['episode_id'];folder=evidence/'episodes'/eid
        try:
            original=source/(eid+'.json');frozen=evidence/'source_json'/(eid+'.json');dest=output/(eid+'.json')
            receipt=json.loads((folder/'complete.json').read_text())
            assert sha(original)==sha(frozen)==record['sha256']==receipt['source_sha256'],'source hash'
            assert sha(dest)==receipt['output_sha256'],'output hash'
            doc=json.loads(frozen.read_text());result=json.loads(dest.read_text())
            schema_path=evidence/'schemas'/(doc['schema']+'.json')
            assert sha(schema_path)==receipt['schema_sha256'],'schema hash'
            schema=json.loads(schema_path.read_text())
            assert schema==output_schema(doc['schema']),'schema implementation'
            validate_output(doc,result,schema)
            graph=initialize_temporal_graph(doc['scene_memory']);uncertainty_count=0;attempts=0;updates=0
            for index, (source_event,event) in enumerate(zip(doc['search_event_states']['events'],result['search_event_states']['events'],strict=True),1):
                event_folder=folder/f'event_{index:02d}'
                accepted=json.loads((event_folder/'accepted.json').read_text())
                frames=[s['original_video_frame_index'] for s in source_event['states']]
                graph,changes,merge_audit=merge_observations(graph,accepted['raw_changes'],event['event_id'],frames)
                assert changes==event['spatial_changes']==accepted['spatial_changes'],'applied observations'
                assert merge_audit==accepted['merge_audit'],'merge audit'
                response=event_folder/f"attempt_{accepted['accepted_attempt']:03d}"/'response.json'
                assert sha(response)==accepted['response_sha256'],'accepted response hash'
                raw=json.loads(response.read_text())
                assert raw['model']=='local-vlm' and raw['choices'][0]['finish_reason']=='stop','model completion'
                content=json.loads(raw['choices'][0]['message']['content'])
                assert content['spatial_changes']==accepted['raw_changes'],'raw observations'
                assert content['frame_support_observations']==accepted['frame_support_observations'],'frame support evidence'
                assert [o['frame'] for o in content['frame_support_observations']]==frames,'support frame ordering'
                assert content['uncertainties']==event['spatial_change_annotation']['uncertainties'],'uncertainties'
                for s in source_event['states']:
                    assert sha(Path(s['image_path']))==s['image_sha256'],'image hash'
                uncertainty_count+=len(accepted['uncertainties']);updates+=len(changes)
                attempts+=len(list(event_folder.glob('attempt_*')))
            assert graph==json.loads((folder/'parent_graph.json').read_text()),'assembled canonical graph'
            assert sha(folder/'parent_graph.json')==receipt['parent_graph_sha256'],'graph hash'
            assert nested_temporal_tree(graph)==result['scene_memory']['spatial_structure'],'assembled tree'
            timing=json.loads((folder/'timing.json').read_text())
            rows.append({'episode_id':eid,'events':len(doc['search_event_states']['events']),
                'relation_updates':updates,'uncertainty_notes':uncertainty_count,'attempts':attempts,
                'episode_wall_seconds':timing['wall_seconds'],'cached_events':timing['cached_events'],
                'source_sha256':receipt['source_sha256'],'output_sha256':receipt['output_sha256']})
        except Exception as exc:errors.append({'episode_id':eid,'error':repr(exc)})
    expected={r['episode_id'] for r in manifest['episodes']}
    extra={p.stem for p in output.glob('*.json')}-expected
    if extra:errors.append({'extra_outputs':sorted(extra)})
    batch=json.loads((evidence/'batch_status.json').read_text())
    report={'audited_at':now(),'expected_count':len(expected),'validated_count':len(rows),
        'status':'complete' if len(rows)==len(expected) and not errors else 'incomplete',
        'event_count':sum(r['events'] for r in rows),'relation_update_count':sum(r['relation_updates'] for r in rows),
        'uncertainty_notes':sum(r['uncertainty_notes'] for r in rows),'errors':errors,
        'batch_wall_seconds':batch.get('wall_seconds'),'episodes_per_hour':batch.get('episodes_per_hour'),
        'timing_scope':'Batch wall time includes queueing, cache reuse and retries. Concurrent episode/request durations are not summed. Pilot timings remain separately recorded.',
        'mean_episode_wall_seconds':statistics.mean(r['episode_wall_seconds'] for r in rows) if rows else None,
        'semantic_status':'Structural, provenance and preservation checks passed for validated rows; visual relations remain unreviewed model inferences.',
        'rows':rows}
    atomic(evidence/'completion_audit.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','output','evidence'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();r=audit(a.source,a.output,a.evidence)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},ensure_ascii=False,indent=2))
