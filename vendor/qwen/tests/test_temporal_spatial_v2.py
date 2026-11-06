import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from temporal_spatial_v2 import (  # noqa: E402
    apply_spatial_changes,
    build_spatial_change_prompt,
    event_spatial_change_schema,
    initialize_temporal_graph,
    validate_temporal_graph,
)


@pytest.fixture
def base_graph():
    scene = {'spatial_structure': {
        'root_id': 'workspace',
        'nodes': [{'id': 'workspace', 'name': 'Workspace', 'type': 'region', 'children': [
            {'id': 'M', 'name': 'Board', 'type': 'object', 'object_ref': 'M', 'placement': {'predicate': 'within_region', 'evidence_frames': ['f10']}, 'children': [
                {'id': 'A', 'name': 'Tool', 'type': 'object', 'object_ref': 'A', 'children': [],
                 'placement': {'predicate': 'hanging_on', 'description': 'lower left', 'evidence_frames': ['f10']}}
            ]},
            {'id': 'N', 'name': 'Table', 'type': 'object', 'object_ref': 'N', 'placement': {'predicate': 'within_region', 'evidence_frames': ['f10']}, 'children': []}
        ]}], 'unlocated_nodes': []
    }}
    return initialize_temporal_graph(scene, 10)


def test_reparent_closes_old_interval_and_derives_parent_children(base_graph):
    graph = apply_spatial_changes(base_graph, [{
        'node_id': 'A', 'change_type': 'reparent', 'change_frame': 20,
        'parent_id': 'N', 'predicate': 'on', 'relative_position': 'lower center'
    }], 'event_01', [15, 20, 25])
    edges = graph['nodes']['A']['parent_relations']
    assert [(x['parent_id'], x['valid_from_frame'], x['valid_to_frame_exclusive']) for x in edges] == [
        ('M', 10, 20), ('N', 20, None)
    ]
    assert graph['nodes']['M']['child_relations'][0]['valid_to_frame_exclusive'] == 20
    assert graph['nodes']['N']['child_relations'][0]['valid_from_frame'] == 20
    assert validate_temporal_graph(graph)


def test_position_change_keeps_parent(base_graph):
    graph = apply_spatial_changes(base_graph, [{
        'node_id': 'A', 'change_type': 'position_change', 'change_frame': 20,
        'parent_id': 'M', 'predicate': 'hanging_on', 'relative_position': 'upper left'
    }], 'event_02', [15, 20, 25])
    edges = graph['nodes']['A']['parent_relations']
    assert [x['parent_id'] for x in edges] == ['M', 'M']
    assert edges[0]['valid_to_frame_exclusive'] == edges[1]['valid_from_frame'] == 20


def test_rejects_unknown_frame_and_cycles(base_graph):
    change = {'node_id': 'A', 'change_type': 'reparent', 'change_frame': 21,
              'parent_id': 'N', 'predicate': 'on', 'relative_position': 'left'}
    with pytest.raises(Exception, match='not one of'):
        apply_spatial_changes(base_graph, [change], 'event_01', [15, 20, 25])
    cycle = {'node_id': 'M', 'change_type': 'reparent', 'change_frame': 20,
             'parent_id': 'A', 'predicate': 'inside', 'relative_position': 'inside'}
    with pytest.raises(ValueError, match='cycle'):
        apply_spatial_changes(base_graph, [cycle], 'event_02', [15, 20, 25])


def test_spatial_delta_schema_uses_only_source_frames():
    schema = event_spatial_change_schema([12, 24, 36])
    valid = {'node_id': 'A', 'change_type': 'reparent', 'change_frame': 24,
             'parent_id': 'N', 'predicate': 'on', 'relative_position': 'lower shelf'}
    import jsonschema
    jsonschema.validate([valid], schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate([{**valid, 'change_frame': 25}], schema)


def test_prompt_includes_compact_ids_and_active_parent(base_graph):
    prompt = build_spatial_change_prompt(base_graph, [15, 20, 25])
    assert 'original source-video frame index' in prompt or 'original source-video frame' in prompt
    assert '"node_id":"A"' in prompt
    assert '"parent_id":"M"' in prompt


def test_unparent_then_restore_retains_unknown_interval(base_graph):
    graph = apply_spatial_changes(base_graph, [{
        'node_id': 'A', 'change_type': 'unparent', 'change_frame': 20,
        'parent_id': None, 'predicate': 'unknown', 'relative_position': None,
    }], 'event_01', [15,20,25])
    graph = apply_spatial_changes(graph, [{
        'node_id': 'A', 'change_type': 'establish', 'change_frame': 30,
        'parent_id': 'N', 'predicate': 'on', 'relative_position': 'center',
    }], 'event_02', [26,30,35])
    assert [(r['parent_id'],r['valid_from_frame'],r['valid_to_frame_exclusive'])
            for r in graph['nodes']['A']['parent_relations']] == [('M',10,20),(None,20,30),('N',30,None)]


def test_event_before_reference_does_not_overlap_future_relation(base_graph):
    graph = apply_spatial_changes(base_graph, [{
        'node_id': 'A', 'change_type': 'establish', 'change_frame': 5,
        'parent_id': 'N', 'predicate': 'on', 'relative_position': 'center',
    }], 'early_event', [3,5,8])
    assert graph['nodes']['A']['parent_relations'][0]['valid_to_frame_exclusive'] == 10
    assert validate_temporal_graph(graph)


def test_same_frame_restructuring_is_atomic(base_graph):
    changes = [
        {'node_id':'M','change_type':'reparent','change_frame':20,'parent_id':'A',
         'predicate':'inside','relative_position':None},
        {'node_id':'A','change_type':'reparent','change_frame':20,'parent_id':'workspace',
         'predicate':'within_region','relative_position':None},
    ]
    assert validate_temporal_graph(apply_spatial_changes(base_graph, changes, 'event', [15,20,25]))


def test_no_evidence_means_no_invented_initial_placement():
    scene = {'spatial_structure': {'root_id':'root','nodes':[
        {'id':'root','name':'Root','type':'region','children':[
            {'id':'a','name':'Unknown','type':'object','object_ref':'a','children':[]}
        ]}], 'unlocated_nodes':[]}}
    graph = initialize_temporal_graph(scene, 0)
    assert graph['nodes']['a']['parent_relations'] == []


def test_snapshot_boundary_and_no_legacy_neighbor_edges(base_graph):
    from temporal_spatial_v2 import snapshot_at_frame
    graph = apply_spatial_changes(base_graph, [{
        'node_id':'A','change_type':'reparent','change_frame':20,
        'parent_id':'N','predicate':'on','relative_position':'center'}], 'event', [15,20,25])
    before = snapshot_at_frame(graph,19)['root']['children']
    after = snapshot_at_frame(graph,20)['root']['children']
    assert before[0]['children'][0]['id'] == 'A'
    assert after[0]['children'] == [] and after[1]['children'][0]['id'] == 'A'
    assert all('placement' not in n and 'relations' not in n for n in graph['nodes'].values())


def test_rejects_open_overlap_and_corrupted_child_index(base_graph):
    graph = copy.deepcopy(base_graph)
    graph['nodes']['A']['parent_relations'].append(copy.deepcopy(graph['nodes']['A']['parent_relations'][0]))
    with pytest.raises(ValueError, match='Overlapping'):
        validate_temporal_graph(graph)
    graph = copy.deepcopy(base_graph)
    graph['nodes']['M']['child_relations'] = []
    with pytest.raises(ValueError, match='parent-side'):
        validate_temporal_graph(graph)


def walk_tree(tree):
    def visit(node):
        yield node
        for child in node['children']:
            yield from visit(child)
    return [n for root in tree['nodes'] + tree['unlocated_nodes'] for n in visit(root)]


def test_nested_tree_preserves_both_historical_parents(base_graph):
    from temporal_spatial_v2 import nested_temporal_tree, validate_temporal_tree
    graph = apply_spatial_changes(base_graph, [{
        'node_id': 'A', 'change_type': 'reparent', 'change_frame': 20,
        'parent_id': 'N', 'predicate': 'on', 'relative_position': 'center',
    }], 'event', [15,20,25])
    tree = nested_temporal_tree(graph)
    assert validate_temporal_tree(tree)
    parents = {n['id']: n for n in tree['nodes'][0]['children']}
    old, new = parents['M']['children'][0], parents['N']['children'][0]
    assert old['id'] == new['id'] == 'A'
    assert (old['valid_from_frame'], old['valid_to_frame_exclusive']) == (10,20)
    assert (new['valid_from_frame'], new['valid_to_frame_exclusive']) == (20,None)
    assert new['placement']['parent_id'] == 'N'
    # The unknown pre-observation period never invents a parent at frame 0.
    assert any(n['id'] == 'A' and n['placement'] is None and
               n['valid_to_frame_exclusive'] == 10 for n in tree['unlocated_nodes'])


def test_nested_tree_splits_same_parent_position_and_unknown(base_graph):
    from temporal_spatial_v2 import nested_temporal_tree, validate_temporal_tree
    graph = apply_spatial_changes(base_graph, [{
        'node_id': 'A', 'change_type': 'position_change', 'change_frame': 20,
        'parent_id': 'M', 'predicate': 'hanging_on', 'relative_position': 'upper left',
    }, {
        'node_id': 'A', 'change_type': 'unparent', 'change_frame': 25,
        'parent_id': None, 'predicate': 'unknown', 'relative_position': None,
    }], 'event', [15,20,25])
    tree = nested_temporal_tree(graph)
    assert validate_temporal_tree(tree)
    children = tree['nodes'][0]['children'][0]['children']
    assert [n['placement']['relative_position'] for n in children] == ['lower left','upper left']
    assert [n['valid_to_frame_exclusive'] for n in children] == [20,25]
    assert any(n['id'] == 'A' and n['valid_from_frame'] == 25 and
               n['placement']['parent_id'] is None for n in tree['unlocated_nodes'])


def test_ancestor_move_clips_descendants_without_new_evidence(base_graph):
    from temporal_spatial_v2 import nested_temporal_tree, validate_temporal_tree
    graph = apply_spatial_changes(base_graph, [{
        'node_id': 'M', 'change_type': 'reparent', 'change_frame': 20,
        'parent_id': 'N', 'predicate': 'on', 'relative_position': 'center',
    }], 'event', [15,20,25])
    tree = nested_temporal_tree(graph)
    assert validate_temporal_tree(tree)
    appearances = [n for n in walk_tree(tree) if n['id'] == 'A' and n['placement']]
    assert sorted(n['valid_from_frame'] for n in appearances) == [10,20]
    assert all(n['placement']['valid_from_frame'] == 10 and
               n['placement']['valid_to_frame_exclusive'] is None for n in appearances)
    bad = copy.deepcopy(tree)
    bad['nodes'][0]['children'][0]['children'][0]['valid_to_frame_exclusive'] = None
    with pytest.raises(ValueError, match='mismatch'):
        validate_temporal_tree(bad)


def test_historical_reverse_ancestry_has_no_infinite_recursion(base_graph):
    from temporal_spatial_v2 import nested_temporal_tree, validate_temporal_tree
    graph = apply_spatial_changes(base_graph, [
        {'node_id':'M','change_type':'reparent','change_frame':20,'parent_id':'A',
         'predicate':'inside','relative_position':None},
        {'node_id':'A','change_type':'reparent','change_frame':20,'parent_id':'workspace',
         'predicate':'within_region','relative_position':None},
    ], 'event', [15,20,25])
    assert validate_temporal_tree(nested_temporal_tree(graph))


def test_nested_tree_cannot_drop_a_known_interval(base_graph):
    from temporal_spatial_v2 import nested_temporal_tree, validate_temporal_tree
    tree = nested_temporal_tree(base_graph)
    # Removing the whole supported relation leaves an incomplete unknown history.
    tree['nodes'][0]['children'][0]['children'] = []
    with pytest.raises(ValueError, match='mismatch'):
        validate_temporal_tree(tree)
