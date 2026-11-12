"""Deterministic research-field export; never calls a model or alters source output."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import time

import jsonschema
import psutil

VERSION = 'spatial_memory.search_episode.research_fields.v1'
SOURCE = 'explicit_user_dataset_protocol_2026_11_03'


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def obj(properties):
    return {'type': 'object', 'properties': properties,
            'required': list(properties), 'additionalProperties': False}


def extend(node, properties):
    node['properties'].update(properties)
    node['required'] = list(dict.fromkeys(node['required'] + list(properties)))


def schema_for(base):
    schema = copy.deepcopy(base)
    text = {'type': 'string', 'minLength': 1}
    pending = {'type': ['string', 'null']}
    protocol = {'const': SOURCE}
    location = obj({'unit_id': text, 'description': text, 'source_event_id': text,
                    'location_granularity': {'const': 'search_region'},
                    'description_source': {'const': 'qwen_search_region_unreviewed'}})
    extend(schema, {
        'hidden': obj({'target_location': location, 'target_instance_id': pending,
                       'source': protocol, 'rule': text}),
        'history': {'type': 'array', 'maxItems': 0},
    })
    schema['properties']['schema'] = {'const': VERSION}
    extend(schema['properties']['metadata'], {
        **{key: pending for key in ['participant_id', 'home_id', 'encoding_condition',
                                   'target_instance_id', 'target_category']},
        'pre_search_information_received': {'const': False},
        'pre_search_information_source': protocol,
        'research_fields_source': protocol,
        'pending_metadata_fields': {'type': 'array', 'items': text},
        'no_revisits_confirmed': {'const': True},
    })
    event = schema['properties']['search_event_states']['properties']['events']['items']
    old_outcome = copy.deepcopy(event['properties']['outcome'])
    shared = {
        'search_region_id': text, 'region_visit_index': {'const': 1},
        'region_total_visits': {'const': 1}, 'is_revisit': {'const': False},
        'region_identity_source': protocol,
        'validity': {'type': ['string', 'number', 'null']},
        'validity_status': {'enum': ['pending_definition', 'annotated']},
        'inspect_cost_s': {'type': ['number', 'null'], 'minimum': 0},
    }
    extend(event, {**copy.deepcopy(shared), 'result': {'enum': ['found', 'not_found']},
                   'result_source': protocol, 'visual_outcome_before_protocol': old_outcome})
    extend(schema['properties']['inspections']['items'], {
        **copy.deepcopy(shared), 'search_region': text,
    })
    extend(schema['properties']['search_units']['properties']['list']['items'], {
        'visit_count': {'const': 1}, 'visit_count_source': protocol,
    })
    final_location = schema['properties']['episode_outcome']['properties']['final_location']['anyOf'][0]
    for key in ['evidence_state_id', 'sample_time_s']:
        original = final_location['properties'][key]
        final_location['properties'][key] = {'anyOf': [original, {'type': 'null'}]}
    extend(schema['properties']['episode_outcome'], {'result_source': protocol})
    extend(schema['properties']['derived_statistics'], {
        'revisit_count': {'const': 0}, 'revisit_count_source': protocol,
        'region_visit_counts': {'type': 'array', 'items': obj({
            'unit_id': text, 'description': text, 'count': {'const': 1}})},
    })
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def transform(source):
    doc = copy.deepcopy(source)
    if source['schema'] == VERSION:
        return doc
    events = doc['search_event_states']['events']
    inspections = doc['inspections']
    if not events or len(events) != len(inspections):
        raise ValueError('Expected matching nonempty events and inspections')
    if [e['ordinal'] for e in events] != list(range(1, len(events) + 1)):
        raise ValueError('Event order must be contiguous')
    units = {u['unit_id']: u for u in doc['search_units']['list']}
    if len(units) != len(events):
        raise ValueError('No-revisit protocol requires one distinct region per event')
    doc['schema'] = VERSION
    meta = doc['metadata']
    fields = ['participant_id', 'home_id', 'encoding_condition', 'target_instance_id', 'target_category']
    meta.update({key: None for key in fields})
    meta.update(pre_search_information_received=False, pre_search_information_source=SOURCE,
                research_fields_source=SOURCE, pending_metadata_fields=fields,
                no_revisits_confirmed=True, only_final_event_found_target_confirmed=True)
    doc['history'] = []
    counts = []
    seen = set()
    for index, (event, inspection) in enumerate(zip(events, inspections, strict=True)):
        if inspection['event_id'] != event['event_id'] or inspection['order'] != event['ordinal']:
            raise ValueError('Event/inspection identity mismatch')
        if len(inspection['unit_ids']) != 1:
            raise ValueError('Expected one search region per supplied event')
        uid = inspection['unit_ids'][0]
        if uid not in units or uid in seen:
            raise ValueError('Missing or repeated region ID under no-revisit protocol')
        seen.add(uid)
        region = event['search_region']
        found = index == len(events) - 1
        shared = {'search_region_id': uid, 'region_visit_index': 1, 'region_total_visits': 1,
                  'is_revisit': False, 'region_identity_source': SOURCE,
                  'validity': None, 'validity_status': 'pending_definition', 'inspect_cost_s': None}
        event['visual_outcome_before_protocol'] = copy.deepcopy(event['outcome'])
        event.update(shared, result='found' if found else 'not_found', result_source=SOURCE)
        event['search_region_status'] = 'user_confirmed_search_event_qwen_region_description_unreviewed'
        event['outcome'] = {'target_found': found, 'target_identity_confirmed': found,
                            'reason': '用户确认本批仅最后一次搜索找到目标；当前事件' +
                            ('为最后一次，确认找到。实例ID待补；非模型视觉确认。' if found else '并非最后一次，确认未找到；不代表有效排除。')}
        inspection.update(shared, search_region=region, result=event['result'], result_source=SOURCE)
        units[uid].update(visit_count=1, visit_count_source=SOURCE)
        counts.append({'unit_id': uid, 'description': region, 'count': 1})
    last = events[-1]
    location = {'unit_id': last['search_region_id'], 'description': last['search_region'],
                'source_event_id': last['event_id'], 'location_granularity': 'search_region',
                'description_source': 'qwen_search_region_unreviewed'}
    doc['hidden'] = {'target_location': location, 'target_instance_id': None, 'source': SOURCE,
                     'rule': '每条最后一次搜索区域即真实藏匿区域；此前未找到，仅最后一次找到。具体区域文字来自模型，未确认区域内精确坐标。'}
    doc['episode_outcome'].update(result='found', success_event_id=last['event_id'],
        success_event_order=last['ordinal'], result_source=SOURCE,
        final_location={'unit_id': last['search_region_id'], 'description': last['search_region'],
            'location_granularity': 'search_region', 'source': SOURCE,
            'evidence_state_id': None, 'sample_time_s': None,
            'note': '真值来自用户采集规则，关联最后一次搜索区域；无单帧视觉确认或精确命中时间，物体实例ID待补。'})
    doc['derived_statistics'].update(found_event_count=1, events_before_success=len(events)-1,
        events_through_success=len(events), revisit_count=0, revisit_count_source=SOURCE,
        region_visit_counts=counts,
        note='按用户规则：最后一次命中、此前未找到、本批无重访。validity未定义；not_found不等于有效排除。')
    doc['search_units'].update(catalog_status='distinct_event_regions_by_user_no_revisit_protocol',
        coverage_note='本批按用户确认无重访建立不同搜索区域；区域文字由模型生成，不保证场景空间穷尽或精确实例匹配。')
    doc['search_event_states']['provenance']['knowledge_scope'] = (
        '视觉描述由Qwen生成，采集规则由用户确认；二者分别保留来源。搜索结果以result/outcome为准，原视觉判断保存在visual_outcome_before_protocol。')
    doc['README']['rules'] = [rule for rule in doc['README']['rules'] if '九个顶层字段' not in rule]
    doc['README']['rules'] += [
        'research_fields.v1在无历史版结构上增加hidden、空曝光history及研究字段。',
        '曝光history为空且pre_search_information_received=false表示用户确认无搜索前信息，不表示缺失记录。',
        '人因字段和搜索充分程度待补，null不等于否定或零；不得用未找到直接推导有效排除。',
        'hidden及result为事后真值；预测搜索决策时应作为标签隔离，不作为事前可用信息。',
    ]
    return doc


def verify(doc):
    events = doc['search_event_states']['events']
    assert sum(e['result'] == 'found' for e in events) == 1
    assert events[-1]['result'] == 'found'
    assert all(e['result'] == 'not_found' for e in events[:-1])
    assert doc['hidden']['target_location']['unit_id'] == events[-1]['search_region_id']
    assert doc['episode_outcome']['success_event_id'] == events[-1]['event_id']
    assert doc['derived_statistics']['events_before_success'] == len(events)-1
    assert all(e['validity'] is None and e['region_total_visits'] == 1 for e in events)
    assert all(i['result'] == e['result'] for i, e in zip(doc['inspections'], events, strict=True))


def run_once(out):
    base = read(out/'target.schema.json')
    schema = schema_for(base)
    atomic(out/'research_target.schema.json', schema)
    protocol_path = out/'research_dataset_protocol.json'
    protocol = read(protocol_path)
    if protocol['source'] != SOURCE or protocol['rules'] != {
            'only_last_event_found': True, 'last_search_region_is_hidden_location': True,
            'pre_search_information_received': False, 'no_revisits': True}:
        raise ValueError('Dataset protocol changed; explicit review required')
    allowed = set(protocol['episode_ids'])
    inventory = {r['episode_id']: r for r in read(out/'inventory.json')['episodes']
                 if not r['problems'] and r['episode_id'] in allowed}
    if set(inventory) != allowed:
        raise ValueError('Protocol episode inventory changed')
    status = {'updated_at': now(), 'schema': VERSION, 'eligible_episodes': len(inventory),
              'completed_count': 0, 'episodes': {}, 'errors': []}
    for eid, row in inventory.items():
        source = out/'json'/(eid+'.json')
        base_marker = out/'episodes'/eid/'target_context_complete.json'
        if not source.exists() or not base_marker.exists():
            continue
        try:
            source_hash = sha(source)
            marker = read(base_marker)
            if marker['sha256'] != source_hash:
                continue  # Batch is between atomic output and completion-marker writes.
            if marker['target'] != row['target'] or marker['target_markdown_sha256'] != sha(Path(row['target_markdown'])):
                raise ValueError('Target provenance mismatch')
            output = out/'research_json'/(eid+'.json')
            receipt = out/'episodes'/eid/'research_fields_complete.json'
            previous = read(receipt) if receipt.exists() else {}
            if (output.exists() and previous.get('source_sha256') == source_hash
                    and previous.get('sha256') == sha(output)
                    and previous.get('protocol_sha256') == sha(protocol_path)):
                doc = read(output)
            else:
                if output.exists():
                    raise ValueError('Research output changed; preserve manual edits for review')
                raw = read(source)
                jsonschema.validate(raw, base)
                doc = transform(raw)
                jsonschema.validate(doc, schema)
                verify(doc)
                atomic(output, doc)
                atomic(receipt, {'source_sha256': source_hash, 'sha256': sha(output),
                    'protocol_sha256': sha(protocol_path),
                    'schema': VERSION, 'finished_at': now(), 'protocol_source': SOURCE})
            jsonschema.validate(doc, schema)
            verify(doc)
            status['episodes'][eid] = {'status': 'complete', 'output': str(output),
                                      'event_count': len(doc['inspections'])}
        except Exception as exc:
            status['errors'].append({'episode_id': eid, 'error': repr(exc)})
    status['completed_count'] = len(status['episodes'])
    status['status'] = 'complete' if status['completed_count'] == len(inventory) and not status['errors'] else 'waiting_for_source'
    if status['errors']:
        status['status'] = 'needs_review'
    atomic(out/'research_export_status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    lock = (out/'.research_export.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    proc = psutil.Process()
    atomic(out/'research_export_launch.json', {'pid': proc.pid, 'create_time': proc.create_time(),
        'command': proc.cmdline(), 'started_at': now(), 'schema': VERSION})
    while True:
        status = run_once(out)
        if not args.watch or status['status'] in ['complete', 'needs_review']:
            print(json.dumps(status, ensure_ascii=False), flush=True)
            break
        time.sleep(20)


if __name__ == '__main__':
    main()
