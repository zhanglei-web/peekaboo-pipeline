"""Export the two target structures from the existing-style source annotation.

Unknown values stay unknown. No model inference or video decoding is performed.
"""
import argparse
import copy
import json
import os
from pathlib import Path


METADATA_FIELDS = [
    'episode_id', 'participant_id', 'home_id', 'zone_id', 'encoding_condition',
    'target_instance_id', 'target_category', 'fps', 'time_axis', 'video_duration_s',
    'target_description_to_participant', 'mobility_tier', 'search_start_unit',
    'search_start_s', 'search_end_s', 'boundary_sampling_interval_s'
]
EXPERIMENT_METADATA_FIELDS = [
    'participant_id', 'home_id', 'encoding_condition',
    'target_description_to_participant'
]
HIDDEN_FIELDS = [
    'true_unit_id', 'original_unit_id', 'relocation_source',
    'relocation_evidence_state', 'evidence_coverage', 'evidence_confidence',
    'intervention_time_h'
]
UNIT_FIELDS = [
    'unit_id', 'label', 'boundary', 'is_container', 'room', 'parent_id',
    'inspect_cost_s', 'volume_l', 'surface_vs_enclosed'
]


def export(annotation):
    d = copy.deepcopy(annotation)
    metadata = d['metadata']
    units = d['search_units']['list']
    unit_ids = {u['unit_id'] for u in units}
    if len(unit_ids) != len(units):
        raise ValueError('Duplicate unit_id in search_units.list')
    events = {e['event_id']: e for e in d['search_event_states']['events']}
    if len(events) != len(d['search_event_states']['events']):
        raise ValueError('Duplicate event_id')
    for name in ['habit_prior', 'cohabitant_counts']:
        if set(d['priors'][name]['counts']) != unit_ids:
            raise ValueError('Prior unit IDs differ from the unit catalog: ' + name)
    if set(d['priors']['semantic_scores']['scores']) != unit_ids:
        raise ValueError('Semantic score unit IDs differ from the unit catalog')
    inspections = []
    source_order = [i['order'] for i in d['inspections']]
    if source_order != sorted(set(source_order)):
        raise ValueError('Inspection orders must be unique and chronological')
    for order, i in enumerate(d['inspections']):
        if i['event_id'] not in events:
            raise ValueError('Inspection does not reference an existing event')
        ids = i['unit_ids']
        if len(ids) != 1:
            raise ValueError('This single-unit target adapter requires explicitly segmented per-unit inspections; cannot silently collapse a multi-unit event')
        if ids[0] not in unit_ids:
            raise ValueError('Unknown inspected unit_id')
        event = events[i['event_id']]
        frames = ['left_f%06d' % s['original_video_frame_index'] for s in event['states']]
        if 'evidence_frames' in i and i['evidence_frames'] != frames:
            raise ValueError('Inspection evidence disagrees with event states')
        inspections.append({
            'event_id': i['event_id'], 'order': order,
            'sample_window_s': i['sample_window_s'], 'unit_id': ids[0],
            'validity': i['validity'], 'result': i['result'],
            'obstruction': i['obstruction'], 'evidence_frames': frames,
            'start_s': i['start_s'], 'end_s': i['end_s'],
            'revisit_kind': i['revisit_kind'],
            'source_region_text': i['source_region_text'], 'note': i['action_summary']
        })
    if set(events) != {i['event_id'] for i in inspections}:
        raise ValueError('Event/inspection mapping is incomplete')
    source = d['README']['annotation_record']
    outcome = d['episode_outcome']
    success_id = outcome['success_event_id']
    if success_id is not None and success_id not in events:
        raise ValueError('Unknown success_event_id')
    success_order = next((i['order'] for i in inspections if i['event_id'] == success_id), None)
    found_unit = outcome['final_location']['unit_id'] if outcome['final_location'] else None
    if found_unit is not None and found_unit not in unit_ids:
        raise ValueError('Unknown found_unit_id')
    hidden_source = d.get('hidden')
    hidden = ({
        k: ((hidden_source['target_location']['unit_id'] if hidden_source.get('target_location') else None)
            if k == 'true_unit_id' else hidden_source[k])
        for k in HIDDEN_FIELDS
    } if hidden_source is not None else None)
    if hidden is not None:
        for name in ['true_unit_id', 'original_unit_id']:
            if hidden[name] is not None and hidden[name] not in unit_ids:
                raise ValueError('Unknown hidden unit ID: ' + name)
    for h in d['history'] or []:
        if h['unit_id'] not in unit_ids:
            raise ValueError('Unknown historical unit ID')
    priors = {
        'habit_prior': {'counts': d['priors']['habit_prior']['counts']},
        'cohabitant_counts': {'counts': d['priors']['cohabitant_counts']['counts']},
        'semantic_scores': {k: d['priors']['semantic_scores'][k] for k in ['source', 'scores']}
    }
    episode = {
        'schema': 'peekaboo.episode.annotation.v1',
        'annotation_profile': source['annotation_profile'],
        'label_source': source['label_source'], 'label_status': source['label_status'],
        'metadata': {k: metadata[k] for k in METADATA_FIELDS},
        'units': [{k: u[k] for k in UNIT_FIELDS} for u in units],
        'inspections': inspections,
        'episode_outcome': {
            'result': outcome['result'], 'success_order': success_order,
            'found_unit_id': found_unit, 'found_evidence_frames': outcome['found_evidence_frames'],
            'stop_reason': outcome['stop_reason'], 'duration_s': outcome['duration_s']
        },
        'history': d['history'], 'hidden': hidden,
        'model_evidence_available_after_s': source['model_evidence_available_after_s'],
        'source': {k: source[k] for k in ['manifest', 'completion', 'home_id_basis']},
        'priors': priors, 'scene_ref': source['scene_ref'],
        'completeness': d['README']['completeness']
    }
    experiment = {
        'metadata': {k: metadata[k] for k in EXPERIMENT_METADATA_FIELDS},
        'history': d['history'], 'hidden': copy.deepcopy(hidden),
        'priors': {k: copy.deepcopy(priors[k]) for k in ['habit_prior', 'cohabitant_counts']},
        'units': {u['unit_id']: {k: u[k] for k in ['inspect_cost_trials_s', 'volume_l']} for u in units}
    }
    return episode, experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.source.read_text())
    episode, experiment = export(d)
    # Keep the scene in the source annotation and link it by a relative JSON reference.
    # Do not overwrite the source or the two original target examples.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in [d['metadata']['episode_id'] + '.json', 'experiment_records.json']:
        output = args.output_dir / filename
        if output.exists():
            raise FileExistsError(output)
        if output.resolve() == args.source.resolve():
            raise ValueError('Output directory must differ from the source directory')
    episode['scene_ref'] = os.path.relpath(args.source.resolve(), args.output_dir.resolve()) + '#/scene_memory'
    for filename, value in [
        (d['metadata']['episode_id'] + '.json', episode),
        ('experiment_records.json', experiment)
    ]:
        output = args.output_dir / filename
        output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        print(output)


if __name__ == '__main__':
    main()
