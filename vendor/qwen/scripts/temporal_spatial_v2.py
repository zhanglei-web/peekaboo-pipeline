"""Frame-bounded direct-parent relations; no imports or mutations of the active batch."""
from copy import deepcopy
import json
from pathlib import Path

import jsonschema

RELATION_TYPES = ['on', 'inside', 'mounted_on', 'hanging_on', 'within_region', 'unknown']
PROMPT_PATH = Path(__file__).resolve().parents[1] / 'prompts/spatial_relation_changes.en.txt'
VERSION = 'ego.scene.temporal_parent_graph.v1'


def object_schema(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def event_spatial_change_schema(visible_frames):
    text = {'type': 'string', 'minLength': 1}
    frame = {'type': 'integer', 'minimum': 0}
    if visible_frames is not None:
        if not visible_frames:
            raise ValueError('Event must provide source-video frame indices')
        frame['enum'] = sorted(set(visible_frames))
    return {'type': 'array', 'items': object_schema({
        'node_id': text,
        'change_type': {'enum': ['establish', 'reparent', 'unparent', 'position_change']},
        'change_frame': frame,
        'parent_id': {'type': ['string', 'null'], 'minLength': 1},
        'predicate': {'enum': RELATION_TYPES},
        'relative_position': {'type': ['string', 'null'], 'minLength': 1},
    })}


def extend_event_schema(event_schema, visible_frames):
    schema = deepcopy(event_schema)
    if 'spatial_changes' in schema['properties']:
        raise ValueError('Event schema already contains spatial_changes')
    schema['properties']['spatial_changes'] = event_spatial_change_schema(visible_frames)
    schema['required'] = list(schema['required']) + ['spatial_changes']
    return schema


def temporal_graph_schema():
    text = {'type': 'string', 'minLength': 1}
    nullable_text = {'type': ['string', 'null'], 'minLength': 1}
    bounds = {'valid_from_frame': {'type': 'integer', 'minimum': 0},
              'valid_to_frame_exclusive': {'type': ['integer', 'null'], 'minimum': 0}}
    edge = object_schema({'relation_id': text, 'parent_id': nullable_text,
        'predicate': {'enum': RELATION_TYPES}, 'relative_position': nullable_text,
        **bounds, 'started_by_event_id': nullable_text, 'ended_by_event_id': nullable_text,
        'boundary_status': {'enum': ['scene_reference_observation_unreviewed',
                                    'first_observed_in_event_unreviewed']}})
    child = object_schema({'relation_id': text, 'child_id': text, **bounds,
                          'started_by_event_id': nullable_text, 'ended_by_event_id': nullable_text})
    node = object_schema({'id': text, 'name': text, 'type': text, 'object_ref': nullable_text,
                         'parent_relations': {'type': 'array', 'items': edge},
                         'child_relations': {'type': 'array', 'items': child}})
    return object_schema({'schema': {'const': VERSION}, 'root_id': text,
        'timebase': object_schema({'unit': {'const': 'original_video_frame_index'},
                                  'index_base': {'const': 0},
                                  'interval_convention': {'const': '[valid_from_frame, valid_to_frame_exclusive)'}}),
        'interpretation': {'const': 'Observation-bounded estimates; persistence between observations is inferred. Null end is open within this recording, not confirmed forever. Absence of an edge means unknown.'},
        'nodes': {'type': 'object', 'minProperties': 1, 'additionalProperties': node}})


def _flatten_scene(scene_memory):
    spatial = scene_memory['spatial_structure']
    nodes, parents, placements = {}, {}, {}

    def visit(node, parent_id=None):
        nid = node['id']
        if nid in nodes:
            raise ValueError(f'Duplicate spatial node ID: {nid}')
        nodes[nid] = {k: deepcopy(node[k]) for k in ['id', 'name', 'type']}
        nodes[nid].update(object_ref=node.get('object_ref'), parent_relations=[], child_relations=[])
        parents[nid], placements[nid] = parent_id, node.get('placement') or {}
        for child in node.get('children', []):
            visit(child, nid)

    for node in spatial.get('nodes', []):
        visit(node)
    for node in spatial.get('unlocated_nodes', []):
        visit(node)
    return nodes, parents, placements, spatial['root_id']


def _active_relation(node, frame):
    active = [r for r in node['parent_relations'] if r['valid_from_frame'] <= frame and
              (r['valid_to_frame_exclusive'] is None or frame < r['valid_to_frame_exclusive'])]
    if len(active) > 1:
        raise ValueError(f'Overlapping parent relations for {node["id"]} at frame {frame}')
    return active[0] if active else None


def refresh_child_relations(graph):
    for node in graph['nodes'].values():
        node['child_relations'] = []
    for child_id, child in graph['nodes'].items():
        for index, relation in enumerate(child['parent_relations'], 1):
            relation['relation_id'] = f'{child_id}:parent:{index:03d}'
            parent_id = relation['parent_id']
            if parent_id in graph['nodes']:
                graph['nodes'][parent_id]['child_relations'].append({
                    'relation_id': relation['relation_id'], 'child_id': child_id,
                    **{k: relation[k] for k in ['valid_from_frame', 'valid_to_frame_exclusive',
                                                'started_by_event_id', 'ended_by_event_id']}})


def initialize_temporal_graph(scene_memory, first_frame=None):
    """Anchor existing reference placements ONLY to their actual supporting frames.

    first_frame is retained for caller compatibility; it never supplies missing evidence.
    A mixed-view scene is not retroactively asserted to be the initial scene snapshot.
    """
    nodes, parents, placements, root_id = _flatten_scene(scene_memory)
    graph = {'schema': VERSION, 'root_id': root_id,
             'timebase': {'unit': 'original_video_frame_index', 'index_base': 0,
                          'interval_convention': '[valid_from_frame, valid_to_frame_exclusive)'},
             'interpretation': temporal_graph_schema()['properties']['interpretation']['const'],
             'nodes': nodes}
    for nid, parent in parents.items():
        p = placements[nid]
        frames = [int(f[1:]) for f in p.get('evidence_frames', [])
                  if isinstance(f, str) and f.startswith('f') and f[1:].isdigit()]
        if parent is None or not frames:
            continue
        if p.get('predicate') not in RELATION_TYPES:
            raise ValueError(f'Unsupported parent predicate: {p.get("predicate")}')
        nodes[nid]['parent_relations'].append({
            'relation_id': f'{nid}:parent:001', 'parent_id': parent,
            'predicate': p['predicate'], 'relative_position': p.get('description'),
            'valid_from_frame': min(frames), 'valid_to_frame_exclusive': None,
            'started_by_event_id': None, 'ended_by_event_id': None,
            'boundary_status': 'scene_reference_observation_unreviewed'})
    refresh_child_relations(graph)
    validate_temporal_graph(graph)
    return graph


def build_spatial_change_prompt(graph, event_frames):
    if not event_frames:
        raise ValueError('Event must provide source-video frame indices')
    catalog = []
    for nid, node in graph['nodes'].items():
        active = _active_relation(node, min(event_frames))
        catalog.append({'node_id': nid, 'name': node['name'],
                        'parent_id': active['parent_id'] if active else None,
                        'predicate': active['predicate'] if active else None,
                        'relative_position': active['relative_position'] if active else None})
    return (PROMPT_PATH.read_text() + '\nEvent source-video frame indices: ' +
            json.dumps(sorted(set(event_frames))) +
            '\nStable node catalog. Parent values are unreviewed reference estimates at the first frame, not ground truth: ' +
            json.dumps(catalog, ensure_ascii=False, separators=(',', ':')))


def _check_acyclic(graph, frame):
    for nid, node in graph['nodes'].items():
        seen = {nid}
        r = _active_relation(node, frame)
        parent = r['parent_id'] if r else None
        while parent is not None:
            if parent in seen:
                raise ValueError(f'Parent cycle at frame {frame}: {nid}')
            seen.add(parent)
            r = _active_relation(graph['nodes'][parent], frame)
            parent = r['parent_id'] if r else None


def apply_spatial_changes(graph, changes, event_id, event_frame_bounds):
    """Atomic event update. Include all same-frame edits before cycle validation."""
    jsonschema.validate(changes, event_spatial_change_schema(event_frame_bounds))
    updated = deepcopy(graph)
    seen = set()
    for change in sorted(changes, key=lambda x: x['change_frame']):
        nid, frame, parent = change['node_id'], change['change_frame'], change['parent_id']
        if nid not in updated['nodes'] or nid == updated['root_id']:
            raise ValueError(f'Unknown or root node_id: {nid}')
        if (nid, frame) in seen:
            raise ValueError('Multiple updates for the same node/frame')
        seen.add((nid, frame))
        if parent is not None and (parent not in updated['nodes'] or parent == nid):
            raise ValueError(f'Invalid parent_id: {parent}')
        kind = change['change_type']
        if kind == 'unparent':
            if parent is not None or change['predicate'] != 'unknown':
                raise ValueError('unparent requires null parent and unknown predicate')
        elif parent is None:
            raise ValueError('Known parent required except for unparent')
        edges = updated['nodes'][nid]['parent_relations']
        current = _active_relation(updated['nodes'][nid], frame)
        if kind == 'position_change' and (current is None or current['parent_id'] != parent):
            raise ValueError('position_change must preserve the current parent')
        if current and current['valid_from_frame'] == frame:
            if (current['parent_id'], current['predicate'], current['relative_position']) == (parent, change['predicate'], change['relative_position']):
                continue
            raise ValueError('Conflicting observations at the same node/frame')
        following = next((r for r in edges if r['valid_from_frame'] > frame), None)
        if current:
            current['valid_to_frame_exclusive'] = frame
            current['ended_by_event_id'] = event_id
        edges.append({'relation_id': 'pending', 'parent_id': parent,
                      'predicate': change['predicate'], 'relative_position': change['relative_position'],
                      'valid_from_frame': frame,
                      'valid_to_frame_exclusive': following['valid_from_frame'] if following else None,
                      'started_by_event_id': event_id,
                      'ended_by_event_id': following['started_by_event_id'] if following else None,
                      'boundary_status': 'first_observed_in_event_unreviewed'})
        edges.sort(key=lambda r: r['valid_from_frame'])
    refresh_child_relations(updated)
    validate_temporal_graph(updated)
    return updated


def validate_temporal_graph(graph):
    jsonschema.validate(graph, temporal_graph_schema())
    nodes = graph['nodes']
    if graph['root_id'] not in nodes or nodes[graph['root_id']]['parent_relations']:
        raise ValueError('Root missing or has a parent')
    transition_frames = set()
    for nid, node in nodes.items():
        if node['id'] != nid:
            raise ValueError('Node ID and dictionary key differ')
        previous = None
        for r in node['parent_relations']:
            start, end, parent = r['valid_from_frame'], r['valid_to_frame_exclusive'], r['parent_id']
            if parent is not None and (parent not in nodes or parent == nid):
                raise ValueError(f'Invalid parent reference for {nid}')
            if end is not None and end <= start:
                raise ValueError(f'Invalid temporal interval for {nid}')
            if previous and (previous['valid_to_frame_exclusive'] is None or start < previous['valid_to_frame_exclusive']):
                raise ValueError(f'Overlapping temporal intervals for {nid}')
            previous = r
            transition_frames.add(start)
            if end is not None:
                transition_frames.add(end)
    for frame in transition_frames:
        _check_acyclic(graph, frame)
    expected = deepcopy(graph)
    refresh_child_relations(expected)
    if expected != graph:
        raise ValueError('Generated relation IDs or parent-side child intervals do not match canonical relations')
    return True


def snapshot_at_frame(graph, frame):
    """Resolve the tree at a source-video frame, without repeating object profiles."""
    validate_temporal_graph(graph)
    def node_at(nid):
        node = graph['nodes'][nid]
        result = {'id': nid, 'name': node['name'], 'children': []}
        relation = _active_relation(node, frame)
        if relation:
            result['placement'] = deepcopy(relation)
        for edge in node['child_relations']:
            if edge['valid_from_frame'] <= frame and (edge['valid_to_frame_exclusive'] is None or frame < edge['valid_to_frame_exclusive']):
                result['children'].append(node_at(edge['child_id']))
        return result
    unlocated = [nid for nid,n in graph['nodes'].items() if nid != graph['root_id']
                 and ((_active_relation(n, frame) or {}).get('parent_id') is None)]
    return {'frame': frame, 'root': node_at(graph['root_id']),
            'unlocated_nodes': [node_at(nid) for nid in unlocated]}


TREE_VERSION = 'ego.scene.temporal_parent_tree.v1'
TREE_INTERPRETATION = (
    'Nested historical tree, not a simultaneous snapshot. Repeated IDs refer to the '
    'same object. Node intervals are intersections along the ancestor path; placement '
    'intervals retain the original direct-parent evidence bounds. Unknown-parent '
    'periods are under unlocated_nodes. Persistence is inferred; null end is open '
    'within this recording, not confirmed forever.')


def temporal_tree_schema():
    graph_schema = temporal_graph_schema()
    props = graph_schema['properties']
    edge = props['nodes']['additionalProperties']['properties']['parent_relations']['items']
    text = {'type': 'string', 'minLength': 1}
    node = object_schema({
        'id': text, 'name': text, 'type': text,
        'object_ref': {'type': ['string', 'null'], 'minLength': 1},
        'valid_from_frame': {'type': 'integer', 'minimum': 0},
        'valid_to_frame_exclusive': {'type': ['integer', 'null'], 'minimum': 0},
        'placement': {'anyOf': [deepcopy(edge), {'type': 'null'}]},
        'children': {'type': 'array', 'items': {'$ref': '#/$defs/temporal_tree_node'}},
    })
    result = object_schema({
        'schema': {'const': TREE_VERSION}, 'root_id': text,
        'timebase': deepcopy(props['timebase']),
        'interpretation': {'const': TREE_INTERPRETATION},
        'nodes': {'type': 'array', 'minItems': 1, 'maxItems': 1,
                  'items': {'$ref': '#/$defs/temporal_tree_node'}},
        'unlocated_nodes': {'type': 'array',
                           'items': {'$ref': '#/$defs/temporal_tree_node'}},
    })
    result['$defs'] = {'temporal_tree_node': node}
    return result


def nested_temporal_tree(graph):
    """CPU-only historical tree. Clip occurrences, never alter evidence boundaries."""
    validate_temporal_graph(graph)

    def intersect(a, b, c, d):
        start = max(a, c)
        ends = [x for x in (b, d) if x is not None]
        end = min(ends) if ends else None
        return (start, end) if end is None or start < end else None

    def occurrence(nid, start, end, placement):
        source = graph['nodes'][nid]
        result = {k: deepcopy(source[k]) for k in ('id', 'name', 'type', 'object_ref')}
        result.update(valid_from_frame=start, valid_to_frame_exclusive=end,
                      placement=deepcopy(placement), children=[])
        for edge in source['child_relations']:
            bounds = intersect(start, end, edge['valid_from_frame'], edge['valid_to_frame_exclusive'])
            if bounds is None:
                continue
            child = graph['nodes'][edge['child_id']]
            relation = next(r for r in child['parent_relations'] if r['relation_id'] == edge['relation_id'])
            result['children'].append(occurrence(child['id'], *bounds, relation))
        return result

    unlocated = []
    for nid, node in graph['nodes'].items():
        if nid == graph['root_id']:
            continue
        cursor = 0
        for relation in node['parent_relations']:
            start, end = relation['valid_from_frame'], relation['valid_to_frame_exclusive']
            if cursor < start:
                unlocated.append(occurrence(nid, cursor, start, None))
            if relation['parent_id'] is None:
                unlocated.append(occurrence(nid, start, end, relation))
            cursor = end
            if cursor is None:
                break
        if cursor is not None:
            unlocated.append(occurrence(nid, cursor, None, None))
    return {'schema': TREE_VERSION, 'root_id': graph['root_id'],
            'timebase': deepcopy(graph['timebase']), 'interpretation': TREE_INTERPRETATION,
            'nodes': [occurrence(graph['root_id'], 0, None, None)],
            'unlocated_nodes': unlocated}


def validate_temporal_tree(tree):
    """Reconstruct canonical edges and require a complete, correctly clipped tree."""
    jsonschema.validate(tree, temporal_tree_schema())
    nodes, relations = {}, {}

    def visit(item):
        nid = item['id']
        profile = {k: item[k] for k in ('id', 'name', 'type', 'object_ref')}
        if nid in nodes and profile != {k: nodes[nid][k] for k in profile}:
            raise ValueError('Repeated node IDs have inconsistent identity fields')
        nodes.setdefault(nid, {**profile, 'parent_relations': [], 'child_relations': []})
        placement = item['placement']
        if placement:
            rid = placement['relation_id']
            record = (nid, placement)
            if rid in relations and relations[rid] != record:
                raise ValueError('Repeated relation IDs have inconsistent evidence')
            if rid not in relations:
                relations[rid] = record
                nodes[nid]['parent_relations'].append(deepcopy(placement))
        for child in item['children']:
            visit(child)

    for node in tree['nodes'] + tree['unlocated_nodes']:
        visit(node)
    for node in nodes.values():
        node['parent_relations'].sort(key=lambda r: r['valid_from_frame'])
    graph = {'schema': VERSION, 'root_id': tree['root_id'], 'timebase': tree['timebase'],
             'interpretation': temporal_graph_schema()['properties']['interpretation']['const'],
             'nodes': nodes}
    refresh_child_relations(graph)
    validate_temporal_graph(graph)
    expected = nested_temporal_tree(graph)
    # Sibling order has no spatial meaning; normalize it for exact coverage checking.
    def ordered(value):
        if isinstance(value, dict):
            return {k: ordered(v) for k, v in value.items()}
        if isinstance(value, list):
            return sorted((ordered(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
        return value
    if ordered(expected) != ordered(tree):
        raise ValueError('Nested tree placement, interval or coverage mismatch')
    return True
