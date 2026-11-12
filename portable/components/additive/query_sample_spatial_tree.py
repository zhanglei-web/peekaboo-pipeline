"""Query the recorded temporal tree estimates at an original-video frame."""
import argparse
import json
from pathlib import Path

from build_final_annotation_sample import tree_nodes


def snapshot(annotation, frame):
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError('frame must be a nonnegative integer')
    tree = annotation['scene_memory']['spatial_structure']
    if 'timebase' not in tree:
        raise ValueError('No temporal extension found in the reference hierarchy')
    result = []
    for node in tree_nodes(tree):
        if not node.get('object_ref'):
            continue
        active = [r for r in node['parent_relations'] or []
                  if frame >= r['valid_from_frame'] and
                  (r['valid_to_frame_exclusive'] is None or frame < r['valid_to_frame_exclusive'])]
        if len(active) > 1:
            raise ValueError('Overlapping temporal parent relations for ' + node['id'])
        relation = active[0] if active else None
        result.append({
            'node_id': node['id'], 'object_ref': node['object_ref'],
            'parent_id': relation['parent_id'] if relation else None,
            'predicate': relation['predicate'] if relation else 'unknown',
            'relative_position': relation['relative_position'] if relation else None,
            'relation_id': relation['relation_id'] if relation else None,
            'status': 'recorded_unreviewed_estimate' if relation else 'unknown',
            'persistence_between_observations': 'inferred' if relation else None
        })
    links = {r['node_id']: r['parent_id'] for r in result}
    for row in result:
        visited = set()
        node_id = row['node_id']
        while node_id in links and links[node_id] is not None:
            if node_id in visited:
                raise ValueError('Temporal parent cycle at frame ' + str(frame))
            visited.add(node_id)
            node_id = links[node_id]
        row['workspace_path_known'] = node_id == tree['root_id']
    return sorted(result, key=lambda r: r['node_id'])


def snapshot_tree(annotation, frame):
    rows = snapshot(annotation, frame)
    root_id = annotation['scene_memory']['spatial_structure']['root_id']
    nodes = {r['node_id']: dict(r, children=[]) for r in rows}
    root = {'node_id': root_id, 'children': []}
    unknown = []
    for row in rows:
        node = nodes[row['node_id']]
        if row['parent_id'] == root_id:
            root['children'].append(node)
        elif row['parent_id'] in nodes:
            nodes[row['parent_id']]['children'].append(node)
        else:
            unknown.append(node)
    return {'nodes': [root], 'unlocated_nodes': unknown}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--frame', type=int, required=True)
    args = parser.parse_args()
    d = json.loads(args.source.read_text())
    print(json.dumps({'frame': args.frame, 'time_s': args.frame / d['metadata']['fps'],
                      'scope': 'recorded estimates, not independently verified physical truth',
                      'tree': snapshot_tree(d, args.frame)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
