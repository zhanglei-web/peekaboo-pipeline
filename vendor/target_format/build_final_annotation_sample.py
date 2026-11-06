"""Add research fields and temporal links without replacing any source JSON value."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_nodes(structure):
    def descend(nodes):
        for node in nodes:
            yield node
            yield from descend(node['children'])
    for key in ['nodes', 'context_nodes', 'unlocated_nodes']:
        yield from descend(structure.get(key, []))


def preserved_values(source, extended, path=''):
    """Return every removed or changed source value, including list order."""
    issues = []
    if isinstance(source, dict):
        if not isinstance(extended, dict):
            return [path + ': type changed']
        for key, value in source.items():
            p = path + '/' + key
            if key not in extended:
                issues.append(p + ': removed')
            else:
                issues.extend(preserved_values(value, extended[key], p))
    elif isinstance(source, list):
        if not isinstance(extended, list) or len(source) != len(extended):
            return [path + ': list changed']
        for index, (a, b) in enumerate(zip(source, extended)):
            issues.extend(preserved_values(a, b, path + '/' + str(index)))
    elif type(source) is not type(extended) or source != extended:
        issues.append(path + ': value changed')
    return issues


def extend(source, source_path, temporal=None, graph=None, temporal_path=None):
    d = copy.deepcopy(source)
    metadata = d['metadata']
    for key in ['participant_id', 'home_id', 'encoding_condition', 'target_instance_id',
                'target_category', 'target_description_to_participant', 'mobility_tier',
                'video_duration_s', 'search_start_unit', 'search_start_s', 'search_end_s',
                'boundary_sampling_interval_s']:
        metadata.setdefault(key, None)
    units = d['search_units']['list']
    for unit in units:
        for key in ['room', 'parent_id', 'inspect_cost_s', 'inspect_cost_trials_s',
                    'volume_l', 'surface_vs_enclosed']:
            unit.setdefault(key, None)
    for inspection in d['inspections']:
        for key in ['validity', 'start_s', 'end_s', 'obstruction', 'revisit_kind']:
            inspection.setdefault(key, None)
    d['episode_outcome'].setdefault('found_evidence_frames', None)
    d['episode_outcome'].setdefault('stop_reason', None)
    d['episode_outcome'].setdefault('duration_s', None)
    d.setdefault('history', None)
    d.setdefault('hidden', {'target_location': None})
    for key in ['original_unit_id', 'relocation_source', 'relocation_evidence_state',
                'evidence_coverage', 'evidence_confidence', 'intervention_time_h']:
        d['hidden'].setdefault(key, None)
    unit_ids = [u['unit_id'] for u in units]
    d.setdefault('priors', {
        'habit_prior': {'counts': dict.fromkeys(unit_ids)},
        'cohabitant_counts': {'counts': dict.fromkeys(unit_ids)},
        'semantic_scores': {'source': None, 'scores': dict.fromkeys(unit_ids)}
    })
    readme = d['README']
    readme['format_extension'] = {
        'version': 'spatial_memory.search_episode.additive_research_temporal.v1.draft',
        'policy': '原字段、原值、原列表顺序及原树形children全部保留，只增加字段。原schema标识基础格式，本字段标识扩展约定。',
        'legacy_revisit_policy': '原无重访字段作为旧版记录原样保留，尚未按同一区域再次搜索规则核对，不能当作当前已确认结论。',
        'temporal_policy': 'children和placement保留原参考布局；各树节点parent_relations描述随时间变化的直接父关系。时间查询按parent_relations重建树，不把参考children误作每时刻布局。',
        'null_policy': '新增实验类字段允许null。null parent_relations表示未做时序标注；空数组仅表示根节点没有父级。',
        'identity_policy': '事件局部实体、全局物体、候选搜索区域分别使用自己的ID，未经确认不强行互换。'
    }
    readme['annotation_record'] = {
        'annotation_profile': 'search_core', 'label_source': 'existing_astra_local_qwen_annotation',
        'label_status': 'draft', 'source_annotation_json': str(source_path),
        'source_annotation_sha256': digest(source_path),
        'temporal_annotation_json': str(temporal_path) if temporal_path else None,
        'manifest': None, 'completion': None, 'home_id_basis': None,
        'scene_ref': 'final_annotation_sample.json#/scene_memory',
        'model_evidence_available_after_s': None, 'new_model_calls': 0
    }
    readme['completeness'] = {
        'video_labels_complete': False, 'spec_complete': False,
        'required_fields_present': True, 'experiment_values_required_for_template': False,
        'temporal_links_verified': False,
        'missing': {
            'experiment_records': [
                'participant_id', 'encoding_condition', 'target_description_to_participant',
                'relocation_records', 'habit_and_cohabitant_counts', 'unit_measurements'
            ],
            'unit_measurements': ['search_units.list[].inspect_cost_trials_s',
                                  'search_units.list[].inspect_cost_s',
                                  'search_units.list[].volume_l'],
            'pending_annotations': ['candidate_region_identity_and_revisits',
                                    'continuous_action_boundaries', 'inspection_validity_and_obstruction'],
            'spatial_temporal_review': []
        }
    }
    structure = d['scene_memory']['spatial_structure']
    structure['timebase'] = {
        'unit': 'original_video_frame_index', 'index_base': 0,
        'fps': metadata['fps'], 'interval_convention': '[valid_from_frame, valid_to_frame_exclusive)',
        'seconds_formula': 'original_video_frame_index / fps'
    }
    structure['temporal_status'] = 'not_annotated'
    nodes = list(tree_nodes(structure))
    node_map = {node['id']: node for node in nodes}
    if len(node_map) != len(nodes):
        raise ValueError('Source reference hierarchy has duplicate stable node IDs')
    for node in nodes:
        node.setdefault('parent_relations', [] if node['id'] == structure['root_id'] else None)
    events = {e['event_id']: e for e in d['search_event_states']['events']}
    for event in events.values():
        event.setdefault('spatial_changes', None)
        event.setdefault('spatial_change_annotation', {'status': 'not_annotated'})
    if temporal is not None:
        if graph is None or temporal['metadata']['episode_id'] != metadata['episode_id']:
            raise ValueError('Temporal graph and matching episode are required')
        if temporal['scene_memory']['objects'] != d['scene_memory']['objects']:
            raise ValueError('Object catalogs differ; IDs cannot be silently reused')
        if set(graph['nodes']) != set(node_map):
            raise ValueError('Temporal graph stable IDs do not match the original tree')
        source_events = {e['event_id']: e for e in temporal['search_event_states']['events']}
        if set(events) != set(source_events):
            raise ValueError('Temporal event IDs differ')
        for event_id, event in events.items():
            original_event = source_events[event_id]
            if [s['original_video_frame_index'] for s in event['states']] != [s['original_video_frame_index'] for s in original_event['states']]:
                raise ValueError('Temporal evidence frames differ')
            for key in ['spatial_changes', 'spatial_change_annotation']:
                event[key] = copy.deepcopy(original_event[key])
            for change in event['spatial_changes']:
                if change['node_id'] not in node_map:
                    raise ValueError('Unresolved temporal node')
                state = next((s for s in event['states'] if s['original_video_frame_index'] == change['change_frame']), None)
                if state is None:
                    raise ValueError('Change is not supported by an event sample frame')
                change['evidence_state_id'] = state['state_id']
        for node_id, node in node_map.items():
            relations = copy.deepcopy(graph['nodes'][node_id]['parent_relations'])
            for relation in relations:
                for boundary, field in [('start', 'started_by_event_id'), ('end', 'ended_by_event_id')]:
                    event_id = relation[field]
                    if event_id is None:
                        continue
                    frame = relation['valid_from_frame'] if boundary == 'start' else relation['valid_to_frame_exclusive']
                    if event_id not in events:
                        raise ValueError('Unresolved temporal event')
                    state = next((s for s in events[event_id]['states'] if s['original_video_frame_index'] == frame), None)
                    if state is None:
                        raise ValueError('Boundary is not linked to an event sample')
                    relation[boundary + '_evidence_state_id'] = state['state_id']
            node['parent_relations'] = relations
        structure['temporal_status'] = 'existing_model_annotations_linked_unreviewed'
        readme['annotation_record']['temporal_annotation_sha256'] = digest(temporal_path)
        readme['completeness']['temporal_links_verified'] = True
        readme['completeness']['missing']['spatial_temporal_review'] = [
            '事件2的平板、事件4的小物体尚未进入全局物体目录，不能把spatial_changes=[]解读为无动作。',
            '鼠标f116的父关系未知与原文字描述存在分歧，不能据此声称已确认拿起。',
            '已有时序区间为采样证据估计；边界关联已核验，物理变化瞬间和帧间持续仍未独立确认。',
            '原左右等参考关系及证据帧原样保留，但未增加未经证实的连续有效区间。'
        ]
    issues = preserved_values(source, d)
    if issues:
        raise ValueError('Source information changed: ' + str(issues))
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--temporal', type=Path)
    parser.add_argument('--graph', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.source.resolve():
        raise ValueError('The sample must not replace the source annotation')
    source = json.loads(args.source.read_text())
    temporal = json.loads(args.temporal.read_text()) if args.temporal else None
    graph = json.loads(args.graph.read_text()) if args.graph else None
    d = extend(source, args.source, temporal, graph, args.temporal)
    args.output.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
