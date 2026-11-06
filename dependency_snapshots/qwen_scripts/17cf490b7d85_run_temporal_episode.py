"""Independent opt-in temporal revision, preserving all original visual/research evidence.

Reuses the stable scene catalog; one VLM call per event produces description and
compact direct-parent updates together. Events can run in parallel against that
catalog, followed by CPU assembly ordered by original frame number.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import time

import jsonschema
from PIL import Image, ImageDraw, ImageFont

from export_research_fields import read, sha, atomic, transform, schema_for, verify
from run_scene_event_batch import local_call, export_episode, full_contract, ROOT, now
from benchmark_event_montage import detailed_schema
from temporal_spatial_v2 import (initialize_temporal_graph, extend_event_schema,
    build_spatial_change_prompt, apply_spatial_changes, temporal_graph_schema,
    event_spatial_change_schema, snapshot_at_frame, validate_temporal_graph,
    temporal_tree_schema, nested_temporal_tree, validate_temporal_tree)

VERSION = 'spatial_memory.search_episode.research_temporal_tree.v1'


def final_schema(base):
    result = deepcopy(base)
    result['properties']['schema'] = {'const': VERSION}
    tree_schema = temporal_tree_schema()
    result.setdefault('$defs', {}).update(tree_schema.pop('$defs'))
    result['properties']['scene_memory']['properties']['spatial_structure'] = tree_schema
    event = result['properties']['search_event_states']['properties']['events']['items']
    event['properties']['spatial_changes'] = event_spatial_change_schema(None)
    event['required'] = list(dict.fromkeys(event['required'] + ['spatial_changes']))
    return result


def annotate_event(event, output, graph, target, correction=None):
    output.mkdir(parents=True, exist_ok=True)
    canvas = Image.new('RGB', (2880, 640), 'white')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
    frames = []
    details = []
    for index, state in enumerate(event['states']):
        path = Path(state['image_path'])
        if sha(path) != state['image_sha256']:
            raise ValueError('Input image hash changed')
        frame = state['original_video_frame_index']
        frames.append(frame)
        with Image.open(path) as image:
            canvas.paste(image.convert('RGB').resize((960, 600)), (index*960, 40))
        draw.text((index*960+8, 5), f"{state['phase']} | f{frame} | {state['original_video_seconds']:.2f}s", fill='black', font=font)
    image = output/'triplet.jpg'
    canvas.save(image, quality=95, subsampling=0)
    for index,state in enumerate(event['states']):
        detail = output/(state['phase']+'_detail.jpg')
        canvas.crop((index*960, 0, (index+1)*960, 640)).save(detail, quality=95, subsampling=0)
        details.append(detail)
    event_base = full_contract()['properties']['search_event_states']['properties']['events']['items']
    contract = extend_event_schema(detailed_schema(event_base), frames)
    props = contract['properties']['spatial_changes']['items']['properties']
    props['node_id']['enum'] = [nid for nid in graph['nodes'] if nid != graph['root_id']]
    props['parent_id']['enum'] = [None, *graph['nodes']]
    prompt = (ROOT/'prompts/search_event_temporal.en.txt').read_text()
    prompt += '\nSearch target: '+json.dumps(target, ensure_ascii=False)+'\n'
    prompt += build_spatial_change_prompt(graph, frames)
    prompt += '\nTrack ALL manipulated catalog objects, regardless of the search target. Touching a resting object is not lifting it. If a container is moved, its contents remain children of that container unless visibly removed. Establish missing visible initial parents even when no motion occurs; do not infer exact support or displacement through occlusion or camera motion.'
    if correction:
        prompt += '\nPrevious output failed consistency validation: '+correction[:1000]+'. Reinspect the images; report supported relations without inventing changes to satisfy validation.'
    data = local_call(output, prompt, [image, *details], contract, 6144)
    if [s['phase'] for s in data['states']] != ['start', 'middle', 'end']:
        raise ValueError('Event phase mismatch')
    return data


def run(source, eid, out, concurrency=1, prepare_only=False):
    if out.resolve() == source.resolve() or source.resolve() in out.resolve().parents:
        raise ValueError('Temporal outputs must be separate from the running source batch')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.temporal.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = now(); tick = time.perf_counter()
    row = next(r for r in read(source/'inventory.json')['episodes'] if r['episode_id'] == eid and not r['problems'])
    src = source/'json'/(eid+'.json')
    marker = read(source/'episodes'/eid/'target_context_complete.json')
    if marker['sha256'] != sha(src) or marker['target'] != row['target'] or marker['target_markdown_sha256'] != sha(Path(row['target_markdown'])):
        raise ValueError('Source visual completion marker or target provenance mismatch')
    protocol = source/'research_dataset_protocol.json'
    if eid not in read(protocol)['episode_ids']:
        raise ValueError('Episode outside confirmed research protocol')
    original = read(src)
    jsonschema.validate(original, read(source/'target.schema.json'))
    scene = original['scene_memory']
    graph = initialize_temporal_graph(scene)
    schema = final_schema(schema_for(read(source/'target.schema.json')))
    jsonschema.Draft202012Validator.check_schema(schema)
    atomic(out/'target.schema.json', schema)
    manifest = {'episode_id': eid, 'source_json': str(src), 'source_sha256': sha(src),
                'protocol_sha256': sha(protocol), 'scene_catalog_reused': True,
                'new_event_description_language': 'en', 'legacy_scene_text_language': 'unchanged',
                'semantic_review_status': 'required', 'frame_namespace': 'original_video_frame_index',
                'implementation_sha256': {name: sha(ROOT/name) for name in [
                    'scripts/run_temporal_episode.py', 'scripts/temporal_spatial_v2.py',
                    'prompts/search_event_temporal.en.txt', 'prompts/spatial_relation_changes.en.txt']}}
    prior = out/'source.json'
    if prior.exists() and read(prior) != manifest:
        raise ValueError('Source differs from existing temporal run')
    atomic(prior, manifest)
    atomic(out/'reference_graph.json', graph)
    if prepare_only:
        atomic(out/'status.json', {'status': 'prepared', 'episode_id': eid, 'model_calls': 0})
        return
    atomic(out/'status.json', {'status': 'running', 'episode_id': eid, 'started_at': started})
    events = row['event_index']['events']
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            tasks = [executor.submit(annotate_event, e, out/'events'/f"event_{e['ordinal']:02d}", graph, row['target']) for e in events]
            data = [task.result() for task in tasks]
        # Across-event generation is independent; apply observations in original-frame order.
        updates = []
        for event,payload in zip(events, data, strict=True):
            frames = [s['original_video_frame_index'] for s in event['states']]
            for change in payload['spatial_changes']:
                updates.append((change['change_frame'], event['event_id'], frames, change))
        # Keep an event's same-frame edits atomic when they rearrange a hierarchy.
        groups = {}
        for frame,event_id,frames,change in sorted(updates, key=lambda u:(u[0],u[1])):
            groups.setdefault((frame,event_id), (frames, []))[1].append(change)
        from batch_temporal_trees import merge_observations
        normalized_by_event = {}
        merge_audit = []
        for (frame,event_id),(frames,changes) in groups.items():
            graph, normalized, audit = merge_observations(graph, changes, event_id, frames)
            normalized_by_event.setdefault(event_id, []).extend(normalized)
            merge_audit.extend(audit)
        atomic(out/'merge_audit.json', merge_audit)
        validate_temporal_graph(graph)
        doc = transform(export_episode(row, scene, data))
        doc['schema'] = VERSION
        doc['scene_memory']['spatial_structure'] = nested_temporal_tree(graph)
        validate_temporal_tree(doc['scene_memory']['spatial_structure'])
        atomic(out/'parent_graph.json', graph)
        for event,payload in zip(doc['search_event_states']['events'], data, strict=True):
            event['spatial_changes'] = normalized_by_event.get(event['event_id'], [])
        doc['README']['rules'].append('Nested children retain historical parent placements and frame intervals; repeated IDs reference the same object. Branch intervals are clipped to ancestor lifetimes; placement bounds retain direct-parent evidence. Parent-side trees are CPU derived. Temporal parent intervals are observation-bounded estimates. Existing scene profile text is reused. New event descriptions are English.')
        jsonschema.validate(doc, schema); verify(doc)
        atomic(out/(eid+'.json'), doc)
        sample_frames = sorted({s['original_video_frame_index'] for e in events for s in e['states']})
        atomic(out/'snapshots.json', [snapshot_at_frame(graph, f) for f in sample_frames])
        elapsed = time.perf_counter()-tick
        timings = [read(out/'events'/f"event_{e['ordinal']:02d}"/'timing.json') for e in events]
        atomic(out/'timing.json', {'started_at': started, 'finished_at': now(), 'wall_seconds': elapsed,
            'event_count': len(events), 'concurrency': concurrency,
            'cached_events': sum(t['started_at'] < started for t in timings),
            'request_timings': timings, 'note': 'Wall time includes queueing alongside original batch; overlapping request durations are not summed.'})
        atomic(out/'complete.json', {**manifest, 'output_sha256': sha(out/(eid+'.json')), 'schema_sha256': sha(out/'target.schema.json'), 'finished_at': now()})
        atomic(out/'status.json', {'status': 'complete', 'event_count': len(events), 'wall_seconds': elapsed,
                                 'relation_update_count': len(updates), 'semantic_review_status': 'required'})
    except Exception as exc:
        atomic(out/'status.json', {'status': 'failed', 'error': repr(exc), 'finished_at': now()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-batch', type=Path, required=True)
    parser.add_argument('--episode', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, choices=[1, 2], default=1)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.source_batch, args.episode, args.output, args.concurrency, args.prepare_only)


if __name__ == '__main__':
    main()
