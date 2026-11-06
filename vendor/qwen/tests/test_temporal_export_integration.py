"""Integration against the locally retained real episode, without any model calls."""
import json
from pathlib import Path
import sys

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from export_research_fields import transform, schema_for, verify
from run_temporal_episode import final_schema, VERSION
from temporal_spatial_v2 import initialize_temporal_graph, apply_spatial_changes, nested_temporal_tree, validate_temporal_tree


def test_full_research_contract_preserves_objects_and_protocol():
    source = ROOT/'outputs/todo_astra_tf_qwen'
    path = source/'json/cp_0007_session_20261026_122149.json'
    if not path.exists():
        pytest.skip('Local real-data integration fixture is unavailable')
    original = json.loads(path.read_text())
    schema = final_schema(schema_for(json.loads((source/'target.schema.json').read_text())))
    doc = transform(original)
    graph = initialize_temporal_graph(original['scene_memory'])
    # Only exercise assembly semantics: this is not a claimed visual annotation.
    graph = apply_spatial_changes(graph, [{'node_id':'o08', 'change_type':'unparent',
        'change_frame':133, 'parent_id':None, 'predicate':'unknown',
        'relative_position':None}], 'integration_test', [109,133,191])
    doc['schema'] = VERSION
    doc['scene_memory']['spatial_structure'] = nested_temporal_tree(graph)
    assert validate_temporal_tree(doc['scene_memory']['spatial_structure'])
    for event in doc['search_event_states']['events']:
        event['spatial_changes'] = []
    jsonschema.validate(doc, schema)
    verify(doc)
    assert doc['scene_memory']['objects'] == original['scene_memory']['objects']
    assert [e['result'] for e in doc['search_event_states']['events']] == ['not_found','not_found','found']
    assert doc['metadata']['pre_search_information_received'] is False


def test_reexport_real_temporal_episode_without_model_calls(tmp_path):
    from export_temporal_tree import export_tree
    from export_research_fields import sha
    from temporal_spatial_v2 import snapshot_at_frame
    source = ROOT/'outputs/temporal_v2_trial_122149_r2'
    if not (source/'complete.json').exists():
        pytest.skip('Local completed temporal fixture is unavailable')
    before = {p: sha(p) for p in source.rglob('*.json')}
    output = tmp_path/'tree'
    path = export_tree(source, output)
    result = json.loads(path.read_text())
    original = json.loads((source/path.name).read_text())
    tree = result['scene_memory']['spatial_structure']
    graph = original['scene_memory']['spatial_structure']
    assert isinstance(tree['nodes'], list)
    assert result['scene_memory']['objects'] == original['scene_memory']['objects']
    assert result['search_event_states'] == original['search_event_states']
    assert {p: sha(p) for p in before} == before
    receipt = json.loads((output/'complete.json').read_text())
    assert receipt['model_calls'] == 0
    assert receipt['output_sha256'] == sha(path)
    assert validate_temporal_tree(tree)

    def active_nodes(roots, frame, temporal):
        found = {}
        def visit(node, parent=None):
            if temporal and not (node['valid_from_frame'] <= frame and
                (node['valid_to_frame_exclusive'] is None or frame < node['valid_to_frame_exclusive'])):
                return
            assert node['id'] not in found, 'Same object appears twice at one frame'
            found[node['id']] = (parent, node.get('placement'))
            for child in node['children']:
                visit(child, node['id'])
        for root in roots:
            visit(root)
        return found

    frames = {0}
    for node in graph['nodes'].values():
        for edge in node['parent_relations']:
            for bound in (edge['valid_from_frame'], edge['valid_to_frame_exclusive']):
                if bound is not None:
                    frames.update((max(0,bound-1),bound,bound+1))
    for frame in sorted(frames):
        snapshot = snapshot_at_frame(graph, frame)
        actual = active_nodes(tree['nodes'] + tree['unlocated_nodes'], frame, True)
        expected = active_nodes([snapshot['root']] + snapshot['unlocated_nodes'], frame, False)
        assert actual == expected
        assert set(actual) == set(graph['nodes'])
    with pytest.raises(FileExistsError):
        export_tree(source, output)
