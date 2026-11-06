"""Re-export a completed temporal episode as nested trees without model calls."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import jsonschema

from export_research_fields import read, sha, atomic, verify
from run_scene_event_batch import ROOT, now
from run_temporal_episode import VERSION, final_schema
from temporal_spatial_v2 import nested_temporal_tree, validate_temporal_tree


def export_tree(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('Use a separate output directory to preserve source evidence')
    receipt = read(source/'complete.json')
    eid = receipt['episode_id']
    src = source/(eid+'.json')
    source_schema = source/'target.schema.json'
    if sha(src) != receipt['output_sha256'] or sha(source_schema) != receipt['schema_sha256']:
        raise ValueError('Source completion hashes do not match')
    doc = read(src)
    if doc['schema'] != 'spatial_memory.search_episode.research_temporal.v1':
        raise ValueError('Expected a completed temporal graph episode')
    old_schema = read(source_schema)
    jsonschema.validate(doc, old_schema)
    verify(doc)
    graph = doc['scene_memory']['spatial_structure']
    tick, started = time.perf_counter(), now()
    result = deepcopy(doc)
    result['schema'] = VERSION
    result['scene_memory']['spatial_structure'] = nested_temporal_tree(graph)
    result['README']['rules'].append(
        'Nested children preserve historical parent placements. Repeated IDs denote the same object. '
        'Node intervals are clipped along the ancestor path; placement intervals retain original '
        'direct-parent evidence. Unknown-parent periods are under unlocated_nodes. '
        'This tree export reuses existing annotations without new model calls.')
    # Reuse the episode contract; final_schema adds the tree definition and event field.
    schema = final_schema(old_schema)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(result, schema)
    validate_temporal_tree(result['scene_memory']['spatial_structure'])
    verify(result)
    assert result['scene_memory']['objects'] == doc['scene_memory']['objects']
    assert result['search_event_states'] == doc['search_event_states']
    # Require a new directory so a failed/repeated conversion cannot replace evidence.
    output.mkdir(parents=True, exist_ok=False)
    atomic(output/(eid+'.json'), result)
    atomic(output/'target.schema.json', schema)
    atomic(output/'parent_graph.json', graph)
    for name in ('semantic_review.json', 'snapshots.json'):
        if (source/name).exists():
            atomic(output/name, read(source/name))
    manifest = {'episode_id': eid, 'source_directory': str(source),
                'source_json': str(src), 'source_sha256': sha(src),
                'source_schema_sha256': sha(source_schema),
                'source_receipt_sha256': sha(source/'complete.json'),
                'model_calls': 0, 'annotations_reused': True,
                'semantic_review_status': 'not_accepted_limits_inherited',
                'implementation_sha256': {name: sha(ROOT/name) for name in (
                    'scripts/temporal_spatial_v2.py', 'scripts/run_temporal_episode.py',
                    'scripts/export_temporal_tree.py')}}
    atomic(output/'source.json', manifest)
    atomic(output/'timing.json', {'started_at': started, 'finished_at': now(),
        'wall_seconds': time.perf_counter()-tick, 'model_calls': 0,
        'scope': 'CPU tree conversion and validation only; no new visual generation.',
        'original_generation_timing': str(source/'timing.json'),
        'original_generation_timing_sha256': sha(source/'timing.json')})
    atomic(output/'complete.json', {**manifest, 'finished_at': now(),
        'output_sha256': sha(output/(eid+'.json')), 'schema_sha256': sha(output/'target.schema.json'),
        'parent_graph_sha256': sha(output/'parent_graph.json')})
    atomic(output/'status.json', {'status': 'complete', 'episode_id': eid, 'model_calls': 0,
                                  'semantic_review_status': 'not_accepted_limits_inherited'})
    return output/(eid+'.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(export_tree(args.source, args.output))


if __name__ == '__main__':
    main()
