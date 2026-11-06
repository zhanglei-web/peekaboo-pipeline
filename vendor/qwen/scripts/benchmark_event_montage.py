"""Isolated triplet-montage benchmark; export the existing event-section schema.

The reference supplies only the contract and timing/identity metadata. Its visual
descriptions, object catalog and outcome labels are never sent to the model.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import re
import time

import jsonschema
from PIL import Image, ImageDraw, ImageFont, ImageOps
import psutil
import requests

ROOT = Path(__file__).resolve().parents[1]
DEMO = Path('/home/baai/zl/demo')
PHASES = [('start', '开始'), ('middle', '进行中'), ('end', '结束')]
PROMPT = '''这是一段搜索事件的三帧横向拼图，从左到右为开始、进行中、结束。
仅根据图像观察，不推测视线、意图或未采样动作。比较手部接触和物体状态。
只输出紧凑JSON：r=操作区域；s=按三帧顺序各一句“手部动作+相关物体状态”；
d=三帧间主要可见变化；u=关键不确定性。全部用简短中文。
r不超过8字，s每项不超过12字，d不超过16字，u不超过10字。
看不清就说不明，不确定取下或放回就不要断言。不要解释、换行或列举背景物体。'''
SHORT_SCHEMA = {
    'type': 'object',
    'properties': {
        'r': {'type': 'string', 'minLength': 1},
        's': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}, 'minItems': 3, 'maxItems': 3},
        'd': {'type': 'string', 'minLength': 1},
        'u': {'type': 'string', 'minLength': 1},
    },
    'required': ['r', 's', 'd', 'u'], 'additionalProperties': False,
}
EVENT_PROMPT = '''观察一张三帧拼图，左/中/右是同一搜索事件的开始/进行中/结束，顶部有原视频帧号。
一次描述完整事件。仅依据这三帧，不补出帧间动作，不推断意图或搜索成功。
输出紧凑JSON：r操作区域(8字内)；e连贯的事件描述(45字内，串联开始、中间、结束及变化)；
s三个时刻的手部与物体状态(按左中右顺序，各12字内)；d主要可见变化(12字内)；u关键不确定性(12字内)。
看不清物体类别就用颜色/形状描述，勿猜物体用途。接触不等于拿起；手或视角移动不等于物体移动。
只有图像明确支持才说取下、放回、离板或移走。e、s、d、u必须相互一致。
中文简洁表述，不重复背景，不输出解释。'''
EVENT_SCHEMA = json.loads(json.dumps(SHORT_SCHEMA))
EVENT_SCHEMA['properties']['e'] = {'type': 'string', 'minLength': 1}
EVENT_SCHEMA['required'].append('e')


def detailed_schema(event_schema):
    state = event_schema['properties']['states']['items']['properties']
    properties = {key: event_schema['properties'][key] for key in (
        'search_region', 'event_summary', 'observed_changes')}
    frame = {key: state[key] for key in (
        'phase', 'state_description', 'hand_action', 'visibility_and_occlusion')}
    frame['focus_object_states'] = {
        'type': 'array', 'maxItems': 3, 'items': {
            'type': 'object', 'properties': {
                'label': {'type': 'string', 'minLength': 1},
                'observed_state': {'type': 'string', 'minLength': 1}},
            'required': ['label', 'observed_state'], 'additionalProperties': False}}
    properties['states'] = {'type': 'array', 'minItems': 3, 'maxItems': 3, 'items': {
        'type': 'object', 'properties': frame, 'required': list(frame), 'additionalProperties': False}}
    properties['uncertainty'] = {'type': 'string', 'minLength': 1}
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def assemble_detailed(payload, sources, reference_event):
    if [state['phase'] for state in payload['states']] != [phase for phase, _ in PHASES]:
        raise ValueError('Detailed output phases do not match image order')
    normalized = {'r': payload['search_region'], 'e': payload['event_summary'],
                  'd': payload['event_summary'], 'u': payload['uncertainty'],
                  's': [state['state_description'] for state in payload['states']]}
    event, _ = assemble(normalized, sources, reference_event)
    event['observed_changes'] = payload['observed_changes']
    catalog, labels = {}, {}
    for dest, state in zip(event['states'], payload['states'], strict=True):
        for key in ('state_description', 'hand_action', 'visibility_and_occlusion'):
            dest[key] = state[key]
        dest['focus_object_states'] = []
        dest['evidence_status'] = 'local_qwen_review_of_resized_three_frame_montage_detailed_annotation'
        for obj in state['focus_object_states']:
            label = obj['label']
            if label not in labels:
                labels[label] = f"event_{event['ordinal']:02d}_entity_{len(labels)+1:02d}"
            entity = labels[label]
            catalog[entity] = {'label': label, 'memory_entity_candidates': [],
                               'link_status': 'event_local_label_reference_only; cross_frame_identity_unverified; scene_memory_not_linked'}
            dest['focus_object_states'].append({'entity_ref': entity, 'observed_state': obj['observed_state']})
    return event, catalog


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_server(record):
    process = psutil.Process(record['pid'])
    if abs(process.create_time() - record['create_time']) >= .01 or process.cmdline() != record['command']:
        raise RuntimeError('Project server identity changed')
    if process.status() == psutil.STATUS_STOPPED:
        raise RuntimeError('Server is stopped; resume only for an explicitly authorized test')


def montage(ordinal, destination, panel_width):
    sources, panels = [], []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    for phase, label in PHASES:
        matches = list((DEMO / '5三帧图像').glob(f'5_第{ordinal}次_{label}_f*.jpg'))
        if len(matches) != 1:
            raise ValueError(f'Expected one source for event {ordinal}/{phase}')
        path = matches[0]
        frame = int(re.search(r'_f(\d+)\.jpg$', path.name).group(1))
        with Image.open(path) as original:
            im = ImageOps.exif_transpose(original).convert('RGB')
            height = round(im.height * panel_width / im.width)
            im = im.resize((panel_width, height), Image.Resampling.LANCZOS)
        panel = Image.new('RGB', (panel_width, height + 36), 'white')
        panel.paste(im, (0, 36))
        ImageDraw.Draw(panel).text((10, 6), f'{phase.upper()} | f{frame} | {frame / 30:.1f}s', font=font, fill='black')
        panels.append(panel)
        sources.append({'phase': phase, 'phase_label': label, 'frame': frame,
                        'path': str(path), 'sha256': sha(path)})
    combined = Image.new('RGB', (panel_width * 3, max(p.height for p in panels)), 'white')
    for index, panel in enumerate(panels):
        combined.paste(panel, (index * panel_width, 0))
    combined.save(destination, quality=90)
    return sources, combined.size


def assemble(short, sources, reference_event):
    ordinal = reference_event['ordinal']
    event_id = reference_event['event_id']
    entity = f'event_{ordinal:02d}_operation_region'
    states = []
    for source, sentence in zip(sources, short['s'], strict=True):
        states.append({
            'original_video_frame_index': source['frame'],
            'state_description': sentence,
            'hand_action': sentence,
            'focus_object_states': [{'entity_ref': entity, 'observed_state': sentence}],
            'visibility_and_occlusion': '本次快速标注未逐帧单独评估遮挡；事件级不确定性：' + short['u'],
            'state_id': event_id + ':' + source['phase'], 'phase': source['phase'],
            'phase_label': source['phase_label'], 'frame_label': 'f' + str(source['frame']),
            'frame_namespace': 'original_video_frame_index_from_filename',
            'original_video_seconds': source['frame'] / 30,
            'image_filename': Path(source['path']).name, 'image_path': source['path'],
            'image_sha256': source['sha256'],
            'evidence_status': 'local_qwen_review_of_resized_three_frame_montage_brief_annotation',
        })
    bounds = [sources[0]['frame'] / 30, sources[-1]['frame'] / 30]
    old_bounds = reference_event['previous_json_interval_seconds']
    event = {
        'ordinal': ordinal, 'search_region': short['r'], 'event_summary': short.get('e', short['d']),
        'observed_changes': [short['d']], 'event_id': event_id,
        'source_human_interval_id': reference_event['source_human_interval_id'],
        'search_region_status': 'visual_interpretation_of_hand_contact_and_view; not_verified_gaze_or_intention',
        'states': states, 'phase_frame_indices': {s['phase']: s['frame'] for s in sources},
        'supplied_image_interval_seconds': bounds, 'previous_json_interval_seconds': old_bounds,
        'interval_alignment': {
            'status': 'matching_bounds' if bounds == old_bounds else 'different_bounds',
            'basis_for_this_record': 'user_supplied_image_filename_frame_indices',
            'note': '帧号由文件名提取；既有区间仅用于元数据对照。',
        },
        'outcome': {'target_found': None, 'target_identity_confirmed': False,
                    'reason': '本次三帧快速标注未验证目标实例或搜索成功；' + short['u']},
    }
    catalog = {entity: {'label': short['r'], 'memory_entity_candidates': [],
                       'link_status': 'event_local_operation_region_only; scene_memory_identity_not_evaluated'}}
    return event, catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--events', nargs='+', type=int, default=[1])
    parser.add_argument('--panel-width', type=int, default=512)
    parser.add_argument('--mode', choices=['brief', 'event', 'detailed'], default='brief')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = json.loads((ROOT / 'logs/server.pid.json').read_text())
    verify_server(record)
    reference_path = DEMO / 'search_episode_5.json'
    reference = json.loads(reference_path.read_text())
    schema_path = ROOT / 'integrations/astra_pipeline/src/astra_memory/resources/search_episode.schema.json'
    schema = json.loads(schema_path.read_text())['properties']['search_event_states']
    save(out / 'search_event_states.schema.json', schema)
    prompt = EVENT_PROMPT if args.mode == 'event' else PROMPT
    output_schema = EVENT_SCHEMA if args.mode == 'event' else SHORT_SCHEMA
    if args.mode == 'detailed':
        prompt = (ROOT / 'prompts/search_event_detailed.zh.txt').read_text()
        output_schema = detailed_schema(schema['properties']['events']['items'])
    save(out / 'model_output.schema.json', output_schema)
    save(out / 'server_snapshot.json', record)
    (out / 'prompt.txt').write_text(prompt)
    report = {
        'status': 'running', 'events': [], 'model': record['config']['server']['model'],
        'mode': args.mode,
        'model_lock_sha256': sha(ROOT / 'configs/model.lock.json'),
        'reference_sha256': sha(reference_path), 'contract_sha256': sha(schema_path),
        'input_mode': 'one_horizontal_full_frame_montage_per_event',
        'panel_width': args.panel_width, 'sequential_requests': True, 'cache_prompt': False,
        'thinking': False, 'reference_visual_answers_supplied_to_model': False,
        'annotation_scope': ('detailed_per_frame_scene_action_object_states_occlusion; event_summary_changes; no_scene_memory_identity_linking_or_success_confirmation'
                             if args.mode == 'detailed' else 'brief_only; per-frame description/hand action/region-state share one observation; no separate occlusion review, object identity linking or success confirmation'),
        'timing_scope': 'event_elapsed_s includes montage, request and validated assembly; api_elapsed_s is HTTP round-trip; one-time startup/setup and artifact writes excluded',
    }
    session = requests.Session()
    session.trust_env = False
    events, catalog = [], {}
    batch_start = time.perf_counter()
    try:
        for ordinal in args.events:
            verify_server(record)
            slots = session.get('http://127.0.0.1:8080/slots', timeout=10)
            slots.raise_for_status()
            if any(slot['is_processing'] for slot in slots.json()):
                raise RuntimeError('Other inference is active; refusing a contaminated latency measurement')
            event_start = time.perf_counter()
            path = out / f'event_{ordinal:02d}'
            path.mkdir()
            image_path = path / 'triplet.jpg'
            sources, dimensions = montage(ordinal, image_path, args.panel_width)
            save(path / 'inputs.json', {'sources': sources, 'montage_size': dimensions, 'montage_sha256': sha(image_path)})
            body = {
                'model': 'local-vlm',
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(image_path.read_bytes()).decode()}},
                ]}],
                'max_tokens': 2048 if args.mode == 'detailed' else 384,
                'temperature': 0, 'seed': 42, 'cache_prompt': False,
                'chat_template_kwargs': {'enable_thinking': False}, 'reasoning_format': 'deepseek',
                'response_format': {'type': 'json_schema', 'json_schema': {'name': 'brief_event', 'strict': True, 'schema': output_schema}},
            }
            # Request budget: fixed prompt, one capped image, bounded output;
            # no history or reference scene answers enter this request.
            printable = json.loads(json.dumps(body))
            printable['messages'][0]['content'][1]['image_url']['url'] = '[local montage JPEG; inputs.json records exact hash]'
            save(path / 'request.json', printable)
            started = time.perf_counter()
            response = session.post('http://127.0.0.1:8080/v1/chat/completions', json=body, timeout=180, allow_redirects=False)
            elapsed = time.perf_counter() - started
            (path / 'response.json').write_text(response.text)
            response.raise_for_status()
            raw = response.json()
            if raw['model'] != 'local-vlm' or raw['choices'][0]['finish_reason'] != 'stop':
                raise RuntimeError('Wrong model or incomplete output')
            short = json.loads(raw['choices'][0]['message']['content'])
            jsonschema.Draft202012Validator(output_schema).validate(short)
            ref = next(e for e in reference['search_event_states']['events'] if e['ordinal'] == ordinal)
            event, entities = (assemble_detailed if args.mode == 'detailed' else assemble)(short, sources, ref)
            jsonschema.Draft202012Validator(schema['properties']['events']['items']).validate(event)
            assert {o['entity_ref'] for s in event['states'] for o in s['focus_object_states']} <= entities.keys()
            event_elapsed = time.perf_counter() - event_start
            save(path / 'brief.json', short)
            save(path / 'event.json', event)
            stats = {'ordinal': ordinal, 'api_elapsed_s': elapsed, 'event_elapsed_s': event_elapsed,
                     'under_10_seconds': event_elapsed <= 10, 'usage': raw.get('usage'), 'timings': raw.get('timings'),
                     'schema_valid': True, 'finish_reason': raw['choices'][0]['finish_reason']}
            save(path / 'timing.json', stats)
            events.append(event)
            catalog.update(entities)
            report['events'].append(stats)
            save(out / 'report.json', report)
            print(json.dumps({'ordinal': ordinal, 'elapsed_s': round(event_elapsed, 3), 'brief': short, 'tokens': raw.get('usage')}, ensure_ascii=False), flush=True)
        section = {
            'schema': 'ego.search_event_triplet_states.v1', 'image_directory': str(DEMO / '5三帧图像'),
            'event_count': len(events), 'image_count': 3 * len(events),
            'frame_numbering': {
                'source': 'filename suffix _f<number>.jpg', 'original_video_fps': 30,
                'original_video_frame_index_base': 0, 'fps_source': 'demo reference metadata; 30 fps',
                'time_formula': 'original_video_frame_index / 30',
                'namespace_note': '原视频帧号来自图像文件名，不等同于场景记忆采样帧ID。',
            },
            'provenance': {
                'description_source': '本地Qwen独立观察三帧拼图后生成简短描述；未提供参考JSON中的视觉答案。',
                'coverage': '每事件仅三个缩小的静态采样；状态、手部动作和操作区状态复用同一句简述，未逐帧单独评估遮挡；不等同于详细标注。',
                'knowledge_scope': '仅测试搜索事件部分；区域为事件局部引用，无场景记忆实体链接；未确认目标身份及搜索成功。',
            },
            'entity_reference_catalog': catalog, 'events': events,
        }
        if args.mode == 'detailed':
            section['provenance'].update(
                description_source='本地Qwen独立观察带帧号三帧拼图，一次生成逐帧详细状态和事件整体；未提供参考JSON的视觉答案。',
                coverage='每事件仅三个缩小的静态采样；场景、手部动作、关注物体状态及遮挡分别标注，不涵盖帧间全部动作。',
                knowledge_scope='仅搜索事件部分；物体引用按事件内同名标签归组，不代表身份已验证；未与场景记忆链接，未确认目标或搜索成功。')
        jsonschema.Draft202012Validator(schema).validate(section)
        save(out / 'search_event_states.json', section)
        report.update(status='passed', section_schema_valid=True, batch_elapsed_s=time.perf_counter() - batch_start,
                      mean_event_s=sum(e['event_elapsed_s'] for e in report['events']) / len(events),
                      max_event_s=max(e['event_elapsed_s'] for e in report['events']),
                      all_events_under_10_seconds=all(e['under_10_seconds'] for e in report['events']))
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save(out / 'report.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'events'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
