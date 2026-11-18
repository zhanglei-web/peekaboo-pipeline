"""Independently verify a completed three-recording full pipeline run.

Usage: python scripts/verify_full_run.py /path/to/completed_run
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import sys

source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source))
from spatial_memory.memory import replay

root = Path(sys.argv[1]).resolve()
read = lambda p: json.loads(p.read_text())
manifest = read(root / 'run_manifest.json')
assert manifest['model'] == 'Qwen3-VL-8B-Instruct'
assert manifest['historical_prediction_caches_used'] is False
assert manifest['raw_recordings_processed_in_full'] is True
for name, digest in manifest['source_hashes'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest, name
summary = read(root / 'full_summary.json')
assert len(summary['recordings']) == len(manifest['sessions']) == 3
numeric = lambda q: isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) and 0 <= q <= 1
results, total_methods, total_failures, status_counts = [], Counter(), Counter(), Counter()
for session in manifest['sessions']:
    out = root / Path(session).name
    catalog = read(out / 'stage1/catalog.v0.json')
    detector = read(out / 'detection/detection_report.json')
    assert detector['computed_frames'] == detector['expected_frames']
    assert detector['reused_frames'] == 0
    assert detector['status_counts'] == {'ok': detector['expected_frames']}
    predictions = read(out / 'events/predictions.json')['predictions']
    records, methods, scores, triples = [], Counter(), [], []
    for prediction in predictions:
        eid = prediction['event_id']
        folder = out / 'event_runs' / eid
        selection = read(folder / 'final_selection.json')
        candidates = read(folder / 'final_candidates.json')
        record = read(folder / 'scoring/committed_event.json')
        assert selection['model'] == manifest['model']
        ids = [selection[r] for r in ('pre', 'pnr', 'post')]
        assert ids[0] < ids[1] < ids[2]
        assert set(ids) <= {c['source_frame_id'] for c in candidates['candidates']}
        status_counts[selection['status']] += 1
        assert record['event_id'] == eid
        assert record['assessments']
        for a in record['assessments']:
            assert numeric(a['q'])
            assert a.get('estimation_method')
            if a['status'] == 'valid':
                assert a['score_type'] == 'geometric_coverage'
            methods[a['estimation_method']] += 1
            if a.get('fallback_reason'):
                total_failures[a['fallback_reason']] += 1
        assert numeric(record['q_event_estimate'])
        assert abs(record['q_event_estimate'] - sum(a['q'] for a in record['assessments']) / len(record['assessments'])) < 1e-12
        records.append(record)
        scores.append(record['q_event_estimate'])
        triples.append({'event_id': eid, 'frames': ids, 'selection_status': selection['status'],
                        'q_event': record['q_event_estimate'],
                        'assessments': [{'q': a['q'], 'type': a['score_type'], 'method': a['estimation_method']} for a in record['assessments']]})
    memory = read(out / 'memory.json')
    fresh = replay(catalog, records)
    for key in ['units', 'object_summary', 'macro_summary', 'applied_events']:
        assert fresh[key] == memory[key], (out.name, key)
    assert all(numeric(u['Q']) for u in memory['units'].values())
    for unit in memory['units'].values():
        values = [h['q'] for h in unit['history'] if numeric(h.get('q'))]
        assert unit['Q'] == max([0.] + values)
        valid = [h['q'] for h in unit['history'] if h['status'] == 'valid' and h['score_type'] == 'geometric_coverage']
        assert unit['Q_geometric'] == (max(valid) if valid else None)
    saved = read(out / 'recording_summary.json')
    assert saved['predicted_events'] == saved['committed_events'] == saved['events_with_numeric_score'] == len(predictions)
    assert saved['score_method_counts'] == dict(methods)
    trace = read(out / 'event_stage_trace.json')
    assert len(trace) == 4 * len(predictions)
    assert all(t['exit_code'] == 0 for t in trace)
    assert all(t['exit_code'] == 0 for t in read(out / 'stage_trace.json'))
    total_methods.update(methods)
    results.append({'recording': out.name, 'frames': detector['expected_frames'], 'events': len(records),
                    'numeric_events': len(scores), 'initial_regions': len(catalog['regions']),
                    'final_units': len(memory['units']), 'score_method_counts': dict(methods),
                    'full_memory_replay_matches': True, 'triplets_and_scores': triples})
report = {'model': manifest['model'], 'quantization': manifest['weight_quantization'],
          'source_and_resource_hashes_verified': len(manifest['source_hashes']),
          'full_raw_frame_count': sum(r['frames'] for r in results),
          'total_events': sum(r['events'] for r in results), 'historical_prediction_reused_frames': 0,
          'score_method_counts': dict(total_methods), 'selection_status_counts': dict(status_counts),
          'strict_geometry_failure_counts': dict(total_failures),
          'elapsed_seconds': summary['elapsed_seconds'], 'recordings': results}
(root / 'independent_verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'recordings'}, ensure_ascii=False, indent=2))
