#!/usr/bin/env python3
"""Packaged frozen-contract and semantic validator; no model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_schema():
    path = HERE / 'resources/search_episode.schema.json'
    lock = json.loads((HERE / 'resources/contract.lock.json').read_text())
    require(digest(path) == lock['schema_sha256'], 'Schema hash differs from frozen contract')
    return json.loads(path.read_text())


def unique_index(items, key, label):
    result = {}
    for item in items:
        require(item[key] not in result, f'duplicate {label}: {item[key]}')
        result[item[key]] = item
    return result


def validate_contract(doc, *, verify_images=False, confirmation_ledger=None, image_directory_override=None):
    errors = sorted(Draft202012Validator(load_schema()).iter_errors(doc),
                    key=lambda e: str(list(e.absolute_path)))
    require(not errors, 'Schema: ' + '; '.join(
        f'{"/".join(map(str, e.absolute_path))}: {e.message}' for e in errors[:8]))
    metadata = doc['metadata']
    objects = unique_index(doc['scene_memory']['objects'], 'id', 'object')
    graph = doc['scene_memory']['spatial_structure']
    nodes, parents, edges = {}, {}, {}

    def walk(node, parent):
        nid = node['id']
        require(nid not in nodes, f'duplicate spatial node/cycle: {nid}')
        nodes[nid], parents[nid] = node, parent
        for child in node['children']:
            walk(child, nid)

    for collection in ['nodes', 'context_nodes', 'unlocated_nodes']:
        for n in graph[collection]:
            walk(n, None)
    require(graph['root_id'] in nodes, 'unknown root_id')
    require(parents[graph['root_id']] is None, 'workspace root must not have a parent')
    for nid, n in nodes.items():
        if nid != graph['root_id']:
            require(n.get('object_ref') in objects, f'unresolved object_ref: {nid}')
        if 'placement' in n:
            require(parents[nid] is not None, f'placement without parent: {nid}')
        for field in ['placement', 'additional_parent_relations', 'relations', 'cross_level_relations', 'associations']:
            values = [n[field]] if field == 'placement' and field in n else n.get(field, [])
            for edge in values:
                rid = edge.get('id')
                if rid:
                    require(rid not in edges, f'duplicate relation ID: {rid}')
                    edges[rid] = edge
                if 'target_id' in edge:
                    require(edge['target_id'] in nodes, f'unresolved relation target: {rid}')
                if field == 'relations':
                    require(parents[nid] is not None and parents[nid] == parents[edge['target_id']],
                            f'relations must connect siblings: {rid}')
                summary = edge.get('observation_summary')
                if summary:
                    require(summary['first_seen_s'] <= summary['last_seen_s'], f'reversed evidence range: {rid}')
    for oid, o in objects.items():
        require(o['spatial_node_id'] in nodes, f'unresolved spatial_node_id: {oid}')
        require(nodes[o['spatial_node_id']].get('object_ref') == oid,
                f'object/node reference mismatch: {oid}')
        for loc in o['last_observation']['location'] or []:
            require(loc['reference_id'] in objects, f'unresolved location reference: {oid}')
        times = [c['time_s'] for c in o['historical_observation']['state_changes']]
        require(times == sorted(times), f'out-of-order history: {oid}')
    for candidate in graph['identity_candidates']:
        require(set(candidate['entity_ids']) <= objects.keys(), 'unresolved identity candidate')
    units = unique_index(doc['search_units']['list'], 'unit_id', 'unit')
    for uid, u in units.items():
        require(u['zone_id'] == metadata['zone_id'], f'unit/zone mismatch: {uid}')
        require(set(u['source_entity_ids']) <= objects.keys(), f'unresolved source_entity_ids: {uid}')
    ev_section = doc['search_event_states']
    events = unique_index(ev_section['events'], 'event_id', 'event')
    require(ev_section['event_count'] == len(events), 'event_count mismatch')
    fps = ev_section['frame_numbering']['original_video_fps']
    require(math.isclose(fps, metadata['fps']), 'inconsistent fps')
    require(ev_section['frame_numbering']['original_video_frame_index_base'] == 0, 'only zero-based frames supported')
    catalog = ev_section['entity_reference_catalog']
    for candidate in catalog.values():
        require(set(candidate['memory_entity_candidates']) <= objects.keys(), 'unresolved event entity candidate')
    require([e['ordinal'] for e in events.values()] == list(range(1, len(events) + 1)), 'event ordinal mismatch')
    states, image_paths = {}, set()
    image_dir = Path(ev_section['image_directory']).resolve()
    relocated_images = Path(image_directory_override).resolve() if image_directory_override is not None else None
    for eid, event in events.items():
        require([s['phase'] for s in event['states']] == ['start', 'middle', 'end'], f'phase order: {eid}')
        frames = [s['original_video_frame_index'] for s in event['states']]
        require(frames[0] < frames[1] < frames[2], f'non-increasing frames: {eid}')
        require(event['supplied_image_interval_seconds'] == [frames[0]/fps, frames[-1]/fps], f'event image window: {eid}')
        same_bounds = len(event['previous_json_interval_seconds']) == 2 and all(math.isclose(a, b, abs_tol=1e-6) for a, b in zip(
            event['supplied_image_interval_seconds'], event['previous_json_interval_seconds']))
        require(event['interval_alignment']['status'] == ('previous_interval_unavailable' if not event['previous_json_interval_seconds'] else 'matching_bounds' if same_bounds else 'different_bounds'),
                f'interval_alignment mismatch: {eid}')
        for s, expected_label in zip(event['states'], ['开始', '进行中', '结束']):
            require(s['state_id'] not in states, 'duplicate state_id')
            states[s['state_id']] = (eid, s)
            require(s['phase_label'] == expected_label, f'phase_label mismatch: {eid}')
            frame = s['original_video_frame_index']
            match = re.search(r'_f(\d+)\.jpg$', s['image_filename'])
            require(match is not None and int(match[1]) == frame, f'filename/frame mismatch: {s["state_id"]}')
            require(s['frame_label'] == f'f{frame}', 'frame_label mismatch')
            require(math.isclose(s['original_video_seconds'], frame/fps, abs_tol=1e-6), 'frame/time mismatch')
            require(event['phase_frame_indices'][s['phase']] == frame, 'phase_frame_indices mismatch')
            path = Path(s['image_path']).resolve()
            require(path.name == s['image_filename'] and path.parent == image_dir, 'image path/directory mismatch')
            require(str(path) not in image_paths, 'image used more than once')
            image_paths.add(str(path))
            if verify_images:
                actual = relocated_images / s['image_filename'] if relocated_images is not None else path
                require(actual.is_file() and digest(actual) == s['image_sha256'], f'image hash mismatch: {actual}')
            for focus in s['focus_object_states']:
                require(focus['entity_ref'] in catalog, 'unresolved focus entity_ref')
    require(ev_section['image_count'] == len(states), 'image_count mismatch')
    inspections = unique_index(doc['inspections'], 'event_id', 'inspection')
    require(inspections.keys() == events.keys(), 'inspections/events are not one-to-one')
    require([i['order'] for i in inspections.values()] == list(range(1, len(events) + 1)), 'inspection order mismatch')
    for eid, i in inspections.items():
        require(set(i['unit_ids']) <= units.keys(), f'unresolved inspection unit: {eid}')
        require(i['order'] == events[eid]['ordinal'], 'inspection/event order mismatch')
        require(i['sample_window_s'] == events[eid]['supplied_image_interval_seconds'], 'inspection sample_window_s mismatch')
        require(i['source_region_text'] == events[eid]['search_region'], 'source_region_text differs from snapshot')
        require(i['action_summary'] == events[eid]['event_summary'], 'action_summary differs from snapshot')
        expected = 'single_unit_event' if len(i['unit_ids']) == 1 else 'multi_unit_event'
        require(i['granularity'] == expected, 'inspection granularity mismatch')
    target = metadata['target']['confirmed_instance_id']
    require(target is None or target in objects, 'unresolved confirmed_instance_id')
    outcome = doc['episode_outcome']
    success_id = outcome['success_event_id']
    if outcome['result'] == 'found':
        require(success_id in inspections and inspections[success_id]['result'] == 'found', 'success_event_id is not a found event')
        require(outcome['success_event_order'] == inspections[success_id]['order'], 'success_event_order mismatch')
        tid = outcome['target_instance_id']
        require(tid is None or tid in objects, 'unresolved target_instance_id')
        require(target is None or tid == target, 'target identity mismatch')
    else:
        require(all(outcome[k] is None for k in ['success_event_id', 'success_event_order', 'target_instance_id', 'final_location']),
                'unsuccessful/unknown outcome must not retain success fields')
    final = outcome['final_location']
    if final:
        require(final['unit_id'] in units and final['unit_id'] in inspections[success_id]['unit_ids'], 'final location unit is not in successful inspection')
        require(final['evidence_state_id'] in states, 'unresolved final evidence state')
        eid, s = states[final['evidence_state_id']]
        require(eid == success_id and final['sample_time_s'] == s['original_video_seconds'], 'final location evidence/time mismatch')
    found = [i for i in inspections.values() if i['result'] == 'found']
    if metadata['only_final_event_found_target_confirmed']:
        require(len(found) == 1 and found[0]['order'] == len(events), 'only-final-found confirmation conflicts with results')
    stats = doc['derived_statistics']
    expected_stats = dict(search_event_count=len(events), found_event_count=len(found),
                          search_unit_catalog_count=len(units),
                          events_before_success=outcome['success_event_order']-1 if success_id else None,
                          events_through_success=outcome['success_event_order'] if success_id else None)
    for key, expected in expected_stats.items():
        require(stats[key] == expected, f'derived statistic mismatch: {key}')
    claims = [eid for eid, i in inspections.items() if 'user_confirmation' in i['result_source']]
    claims += ['metadata'] if (metadata['full_search_coverage_confirmed'] or
                              metadata['only_final_event_found_target_confirmed'] or
                              'user_confirmation' in metadata['target']['identity_source'] or
                              'user_confirmation' in metadata['search_scope_source']) else []
    if confirmation_ledger is not None:
        require(confirmation_ledger['episode_id'] == metadata['episode_id'], 'confirmation ledger episode mismatch')
        # A ledger must bind the exact output; it is an attestation, not a way to infer confirmations.
        payload = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
        require(confirmation_ledger['output_content_sha256'] == hashlib.sha256(payload).hexdigest(),
                'confirmation ledger output binding mismatch')
        require(set(claims) <= set(confirmation_ledger['confirmed_claim_refs']), 'missing confirmation provenance')
    return {'status': 'contract_passed', 'episode_id': metadata['episode_id'],
            'objects': len(objects), 'search_units': len(units), 'events': len(events), 'images': len(states),
            'image_hashes_checked': verify_images,
            'image_directory_override': str(relocated_images) if relocated_images is not None else None,
            'confirmation_provenance': 'attestation_ledger_checked' if confirmation_ledger is not None else
                'not_independently_verified' if claims else 'no_user_confirmation_claims',
            'semantic_accuracy': 'not_certified_by_contract_validation',
            'end_to_end_pipeline_executed': False}

