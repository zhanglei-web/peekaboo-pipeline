"""Audit additive preservation, event/frame links and both target projections."""
from pathlib import Path
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / 'repository.marker').is_file())
import copy
import hashlib
import json
from pathlib import Path

from build_final_annotation_sample import extend, preserved_values, tree_nodes
from export_final_annotation_sample import export
from query_sample_spatial_tree import snapshot, snapshot_tree


ROOT = Path(__file__).parent
DATASET = Path(str(REPOSITORY / 'legacy/11-4-81'))


def assert_target_fields(reference, actual, path=''):
    if actual is None:
        return
    if isinstance(reference, dict):
        assert isinstance(actual, dict), path
        if path == 'experiment.units' or path.endswith('.counts') or path.endswith('.scores'):
            if reference:
                for value in actual.values():
                    assert_target_fields(next(iter(reference.values())), value, path + '.<unit>')
        else:
            for key, value in reference.items():
                assert key in actual, path + '.' + key
                assert_target_fields(value, actual[key], path + '.' + key)
    elif isinstance(reference, list):
        assert isinstance(actual, list), path
        if reference:
            for value in actual:
                assert_target_fields(reference[0], value, path + '[]')


def audit_temporal(d):
    structure = d['scene_memory']['spatial_structure']
    nodes = {n['id']: n for n in tree_nodes(structure)}
    events = {e['event_id']: e for e in d['search_event_states']['events']}
    relations = {}
    boundaries = {0}
    for node in nodes.values():
        previous_end = None
        for index, relation in enumerate(node['parent_relations'] or []):
            a, b = relation['valid_from_frame'], relation['valid_to_frame_exclusive']
            assert a >= 0 and (b is None or b > a)
            assert index == 0 or (previous_end is not None and a >= previous_end)
            previous_end = b
            assert relation['relation_id'] not in relations
            relations[relation['relation_id']] = (node['id'], relation)
            assert relation['parent_id'] is None or relation['parent_id'] in nodes
            boundaries.add(a)
            if a > 0:
                boundaries.add(a - 1)
            if b is not None:
                boundaries.update([b - 1, b])
            for event_key, frame, evidence_key in [
                ('started_by_event_id', a, 'start_evidence_state_id'),
                ('ended_by_event_id', b, 'end_evidence_state_id')
            ]:
                event_id = relation[event_key]
                if event_id is not None:
                    assert event_id in events
                    matches = [s for s in events[event_id]['states']
                               if s['state_id'] == relation[evidence_key]]
                    assert len(matches) == 1
                    assert matches[0]['original_video_frame_index'] == frame
    changes = 0
    for event in events.values():
        for change in event['spatial_changes'] or []:
            changes += 1
            assert change['node_id'] in nodes
            matches = [s for s in event['states'] if s['state_id'] == change['evidence_state_id']]
            assert len(matches) == 1 and matches[0]['original_video_frame_index'] == change['change_frame']
            active = [r for r in nodes[change['node_id']]['parent_relations']
                      if r['valid_from_frame'] == change['change_frame'] and
                      r['started_by_event_id'] == event['event_id']]
            assert len(active) == 1
            assert active[0]['parent_id'] == change['parent_id']
            assert active[0]['predicate'] == change['predicate']
            boundaries.add(change['change_frame'])
    object_ids = {o['id'] for o in d['scene_memory']['objects']}
    for frame in boundaries:
        rows = snapshot(d, frame)
        assert {r['node_id'] for r in rows} == object_ids
        tree = snapshot_tree(d, frame)
        seen = []
        def walk(items):
            for item in items:
                if item['node_id'] != structure['root_id']:
                    seen.append(item['node_id'])
                walk(item['children'])
        walk(tree['nodes'])
        walk(tree['unlocated_nodes'])
        assert len(seen) == len(set(seen)) and set(seen) == object_ids
    return {'direct_parent_relations': len(relations), 'linked_event_changes': changes,
            'boundary_frame_queries_verified': len(boundaries)}


def main():
    sample_path = ROOT / 'final_annotation_sample.json'
    sample = json.loads(sample_path.read_text())
    original_path = Path(sample['README']['annotation_record']['source_annotation_json'])
    original = json.loads(original_path.read_text())
    assert not preserved_values(original, sample)
    assert set(sample) - set(original) == {'priors'}
    temporal_result = audit_temporal(sample)
    image_count = 0
    for event in sample['search_event_states']['events']:
        for state in event['states']:
            assert hashlib.sha256(Path(state['image_path']).read_bytes()).hexdigest() == state['image_sha256']
            image_count += 1
    target_a = json.loads((ROOT / 'episode_export.shape.json').read_text())
    target_b = json.loads((ROOT / 'experiment_export.shape.json').read_text())
    episode, experiment = export(sample)
    assert_target_fields(target_a, episode, 'episode')
    assert_target_fields(target_b, experiment, 'experiment')
    assert episode['episode_outcome']['success_order'] == 3
    assert all(i['start_s'] is None for i in episode['inspections'])
    bad = copy.deepcopy(sample)
    bad['inspections'][0]['unit_ids'] *= 2
    try:
        export(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Multi-unit events must not be silently collapsed')

    audits = []
    temporal_count = 0
    for source_path in sorted((DATASET / '标注信息').glob('*.json')):
        source = json.loads(source_path.read_text())
        temporal_path = DATASET / '标注信息_时序树' / source_path.name
        graph_path = DATASET / '时序树处理记录/full/episodes' / source_path.stem / 'parent_graph.json'
        temporal = json.loads(temporal_path.read_text()) if temporal_path.is_file() else None
        graph = json.loads(graph_path.read_text()) if temporal is not None else None
        candidate = extend(source, source_path, temporal, graph,
                           temporal_path if temporal is not None else None)
        lost = preserved_values(source, candidate)
        assert not lost, (source_path.name, lost)
        a, b = export(candidate)
        assert_target_fields(target_a, a, 'episode')
        assert_target_fields(target_b, b, 'experiment')
        if temporal is not None:
            audit_temporal(candidate)
            temporal_count += 1
        audits.append({'episode_id': source_path.stem,
                       'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                       'preserved_source_values': True,
                       'target_field_projection': 'passed',
                       'temporal_links_verified': temporal is not None,
                       'output_written': False})
    assert len(audits) == 81
    manifest_path = sorted((DATASET / '三帧迁移记录').glob('*/manifest.json'))[-1]
    for path, expected in json.loads(manifest_path.read_text())['annotation_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    report = {
        'status': 'passed', 'sample_design': 'additive_extension_of_original_81_format',
        'sample_preserved_source_values': True, 'removed_or_replaced_source_values': [],
        'original_children_placement_cross_relations_and_evidence_preserved': True,
        'single_stored_spatial_hierarchy': True,
        'temporal_parent_lists_inside_original_tree_nodes': True,
        'all_target_fields_projectable': True,
        'experiment_values_optional_as_requested': True,
        'image_hashes_verified': image_count,
        'sample_temporal_validation': temporal_result,
        'semantic_model_predictions_independently_confirmed': False,
        'original_81_jsons_audited_in_memory': len(audits),
        'original_81_jsons_unchanged': True,
        'existing_temporal_episodes_link_checked': temporal_count,
        'remaining_episodes_without_completed_temporal_annotations': len(audits) - temporal_count,
        'new_model_calls': 0, 'batch_output_files_written': 0, 'per_episode': audits
    }
    episode['scene_ref'] = '../final_annotation_sample.json#/scene_memory'
    for name, value in [(sample['metadata']['episode_id'] + '.json', episode),
                        ('experiment_export.shape.json', experiment)]:
        (ROOT / 'final_sample_exports' / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'final_sample_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'per_episode'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
