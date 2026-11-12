"""TF-assisted Astra keyframes + local Qwen scene/events; independent resumable batch."""
from pathlib import Path
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / 'repository.marker').is_file())
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import io
import json
import math
from pathlib import Path
import re
import signal
import statistics
import sqlite3
import subprocess
import sys
import threading
import time

import jsonschema
from PIL import Image, ImageDraw, ImageFont
import psutil
import requests

from select_scene_keyframes import ROOT, save, sha
from generate_selected_scene import PROMPT, model_schema, contract, assemble, verify_processes, SCHEMA_PATH
from benchmark_event_montage import detailed_schema

sys.path.insert(0,str(ROOT/'integrations/astra_pipeline/src'))
from astra_memory.inputs import TOPICS, bag_files, triplets

OLD_BATCH=Path(str(REPOSITORY / 'legacy/zl/egomemory_ws/outputs/todo_search_episodes_20261103'))
PHASE_LABEL={'start':'开始','middle':'进行中','end':'结束'}
STOP=threading.Event()


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    save(tmp,data)
    tmp.replace(path)


def full_contract():
    original=json.loads(SCHEMA_PATH.read_text())
    original['properties']['scene_memory']=contract()['properties']['scene_memory']
    return original


def inventory(todo,out):
    rows=[]
    old=json.loads((OLD_BATCH/'inventory.json').read_text()) if (OLD_BATCH/'inventory.json').exists() else {'episodes':[]}
    known={r['episode_id']:r for r in old['episodes']}
    for directory in sorted((todo/'rosbag').iterdir()):
        if not directory.is_dir():continue
        episode=directory.name
        row={'episode_id':episode,'problems':[],'bag_root':str(directory)}
        bags=list(directory.rglob('metadata.yaml'))
        if len(bags)!=1:
            row['problems'].append('expected_one_complete_rosbag_found_'+str(len(bags)))
        else:row['source_bag']=str(bags[0].parent)
        event=todo/'special_frames'/episode
        correction=OLD_BATCH/'event_input_corrections'/(episode+'.json')
        if correction.exists():
            c=json.loads(correction.read_text())
            if c['source']!='explicit_user_message' or c['episode_id']!=episode:
                raise ValueError('Invalid existing event correction')
            for a in c['artifacts']:
                if sha(Path(a['original_path']))!=a['original_sha256'] or sha(Path(a['view_path']))!=a['view_sha256']:
                    raise ValueError('Existing event correction hash mismatch')
            event=Path(c['event_view'])
            row['event_input_correction']={'path':str(correction),'sha256':sha(correction)}
        docs=list(event.glob('*.md'))
        if len(docs)!=1:
            row['problems'].append('missing_unique_event_markdown')
        else:
            text=docs[0].read_text(encoding='utf-8-sig')
            title=re.search(r'^#\s+'+re.escape(episode)+r'\s+查找(.+?)三帧\s*$',text,re.M)
            fps=re.search(r'帧率\s*(\d+(?:\.\d+)?)\s*帧/秒',text)
            if not title or not fps:
                row['problems'].append('missing_target_or_fps')
            else:
                target=title[1].strip()
                if target=='和黑色数据线':target='黑色数据线'
                row.update(target=target,fps=float(fps[1]),target_markdown=str(docs[0]),target_markdown_sha256=sha(docs[0]))
                image_dirs=[p for p in event.iterdir() if p.is_dir() and any(p.glob('*.jpg'))]
                try:
                    if len(image_dirs)!=1:raise ValueError('expected_one_image_directory')
                    index=triplets(image_dirs[0],episode,row['fps'])
                    table=[[int(v) for v in m] for m in re.findall(r'^\|\s*第(\d+)次\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|',text,re.M)]
                    actual=[[e['ordinal'],*[s['original_video_frame_index'] for s in e['states']]] for e in index['events']]
                    if table!=actual:raise ValueError('markdown_filename_mismatch')
                    raw_count=known.get(episode,{}).get('source_frame_count')
                    if raw_count and max(f for e in actual for f in e[1:])>=raw_count:raise ValueError('event_frame_outside_video')
                    row.update(event_images=str(image_dirs[0]),event_count=index['event_count'],event_index=index)
                except Exception as exc:row['problems'].append(str(exc))
        row['status']='ready' if not row['problems'] else 'input_blocked'
        rows.append(row)
    result={'created_at':now(),'rosbag_directory_count':len(rows),'ready_count':sum(not r['problems'] for r in rows),
            'episodes':rows,'scope':'directories currently present under todo/rosbag only'}
    atomic(out/'inventory.json',result)
    atomic(out/'missing_inputs.json',[r for r in rows if r['problems']])
    return rows


def extract_rgb_tf(row,folder):
    """Read only synchronized RGB/metadata/TF/pose; don't recompute VIO/depth."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    folder.mkdir(parents=True,exist_ok=True)
    manifest=folder/'all_frames.jsonl'
    if manifest.exists():
        records=[json.loads(line) for line in manifest.read_text().splitlines()]
        if not records:raise ValueError('Empty saved frame manifest')
        for r in records:
            if sha(folder/r['rgb_path'])!=r['rgb_sha256']:raise ValueError('Saved RGB changed')
        return records
    _,paths=bag_files(row['source_bag'])
    names=[TOPICS[k] for k in ['left','metadata','tf','pose']]
    classes={}
    grouped={}
    records=[]
    def flush(timestamp,items):
        if TOPICS['left'] not in items or TOPICS['metadata'] not in items:
            raise ValueError('Unsynchronized RGB/metadata')
        left,metadata=items[TOPICS['left']],json.loads(items[TOPICS['metadata']].data)
        frame=metadata['left_video_frame_idx']
        ns=int(left.header.stamp.sec)*10**9+int(left.header.stamp.nanosec)
        if ns!=timestamp or metadata['sensor_time_ns']!=ns:raise ValueError('Sensor/record time mismatch')
        path=folder/f'f{frame:07d}.jpg'
        raw=bytes(left.data)
        with Image.open(io.BytesIO(raw)) as im:
            width,height=im.size
            if im.format!='JPEG':raise ValueError('Expected JPEG compressed RGB')
            im.verify()
        path.write_bytes(raw)
        transforms=[]
        for t in getattr(items.get(TOPICS['tf']),'transforms',[]):
            tr,q=t.transform.translation,t.transform.rotation
            transforms.append({'parent':t.header.frame_id,'child':t.child_frame_id,
                'sensor_time_ns':int(t.header.stamp.sec)*10**9+int(t.header.stamp.nanosec),
                'translation_m':[tr.x,tr.y,tr.z],'quaternion_xyzw':[q.x,q.y,q.z,q.w]})
        pose=None
        if TOPICS['pose'] in items:
            m=items[TOPICS['pose']];cov=list(m.pose.covariance)
            valid=all(math.isfinite(v) for v in cov)
            pose={'max_position_std_m':math.sqrt(max(0,*[cov[i] for i in [0,7,14]])) if valid else None,
                  'quality':'unverified_VIO_covariance_valid' if valid else 'invalid_covariance_TF_visual_prior_only'}
        records.append({'frame_id':f"{row['episode_id']}:bag:{len(records):06d}",'bag_keyframe_index':len(records),
            'original_video_frame_index':frame,'original_video_seconds':frame/row['fps'],
            'rgb_path':path.name,'rgb_sha256':sha(path),'width':width,'height':height,
            'sensor_time_ns':ns,'camera_frame':left.header.frame_id,'transforms':transforms,'pose':pose})
    # Merge across segments by timestamp before flushing, without reading depth blobs.
    import heapq
    connections=[]
    try:
        cursors=[]
        for path in paths:
            conn=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True);connections.append(conn)
            cursors.append(iter(conn.execute('SELECT m.timestamp,t.name,t.type,m.data FROM messages m JOIN topics t ON t.id=m.topic_id WHERE t.name IN (?,?,?,?) ORDER BY m.timestamp,m.id',names)))
        current=None
        for timestamp,name,kind,raw in heapq.merge(*cursors,key=lambda x:x[0]):
            if current is not None and current!=timestamp:
                flush(current,grouped);grouped={}
            if name in grouped:raise ValueError('Duplicate timestamp/topic')
            if kind not in classes:classes[kind]=get_message(kind)
            grouped[name]=deserialize_message(raw,classes[kind]);current=timestamp
        if grouped:flush(current,grouped)
    finally:
        for conn in connections:conn.close()
    indices=[r['original_video_frame_index'] for r in records]
    if not indices or indices!=sorted(set(indices)):raise ValueError('Nonmonotonic/empty frame sequence')
    temp=manifest.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in records));temp.replace(manifest)
    atomic(folder/'source_manifest.json',{'source_bag':row['source_bag'],
        'metadata_sha256':sha(Path(row['source_bag'])/'metadata.yaml'),
        'segments':[{'path':str(p),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths],
        'frame_count':len(records),'geometry_scope':'TF camera prior only; depth not decoded; no metric object localization'})
    return records


def local_call(folder,prompt,images,schema,max_tokens=10000):
    folder.mkdir(parents=True,exist_ok=True)
    input_manifest={'prompt':prompt,'images':[{'path':str(p),'sha256':sha(p)} for p in images],
                    'schema':schema,'model':'local-vlm','max_tokens':max_tokens}
    if (folder/'result.json').exists():
        if json.loads((folder/'inputs.json').read_text())!=input_manifest:raise ValueError('Cached request changed')
        result=json.loads((folder/'result.json').read_text());jsonschema.validate(result,schema);return result
    save(folder/'inputs.json',input_manifest)
    body={'model':'local-vlm','messages':[{'role':'user','content':[{'type':'text','text':prompt}]+[
        {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(p.read_bytes()).decode()}} for p in images]}],
        'max_tokens':max_tokens,'temperature':0,'seed':42,'cache_prompt':False,'chat_template_kwargs':{'enable_thinking':False},
        'response_format':{'type':'json_schema','json_schema':{'name':'annotation','strict':True,'schema':schema}}}
    session=requests.Session();session.trust_env=False
    tick=time.perf_counter();started_at=now()
    response=session.post('http://127.0.0.1:8080/v1/chat/completions',json=body,timeout=1200,allow_redirects=False)
    (folder/'response.json').write_text(response.text)
    response.raise_for_status();raw=response.json()
    save(folder/'timing.json',{'started_at':started_at,'finished_at':now(),'seconds':time.perf_counter()-tick,
        'scope':'HTTP wall time including server scheduling; may overlap another request',
        'usage':raw.get('usage'),'timings':raw.get('timings')})
    if raw['model']!='local-vlm' or raw['choices'][0]['finish_reason']!='stop':raise ValueError('Incomplete/wrong-model response')
    result=json.loads(raw['choices'][0]['message']['content']);jsonschema.validate(result,schema)
    save(folder/'result.json',result)
    return result


def scene_job(folder):
    selection=json.loads((folder/'selected_frames.json').read_text())
    frames=[x['original_video_frame_index'] for x in selection['frames']]
    images=[folder/'selected_montage.jpg',*sorted((folder/'qwen_detail_strips').glob('*.jpg'))]
    checklist=selection['selection']['selected']
    prompt=PROMPT+'\n允许的原视频帧号：'+str(frames)+'\n筛帧器提供的待核查覆盖清单（不是已确认事实，必须用图像验证）：'+json.dumps(checklist,ensure_ascii=False)
    prompt+='\n输出前逐项核对覆盖清单。特别检查：可见台面等支承结构是否独立建档；容器是否归属正确支承面；容器内部可见内容是否列为子项；不能辨认的堆叠内容用group_or_partial。只把没有可见直接父物体的物体放在工作区根节点。不要遗漏小件，也不要把看不见的东西补进去。'
    data=local_call(folder/'qwen_scene_v2',prompt,images,model_schema(frames))
    scene=assemble(data);jsonschema.validate(scene,contract());save(folder/'scene_memory.json',scene)
    return scene['scene_memory']


def event_job(event,folder,target=None):
    folder.mkdir(parents=True,exist_ok=True)
    width,height,banner=960,600,40
    canvas=Image.new('RGB',(width*3,height+banner),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
    for i,state in enumerate(event['states']):
        path=Path(state['image_path'])
        if sha(path)!=state['image_sha256']:raise ValueError('Event image changed')
        with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((width,height)),(i*width,banner))
        draw.text((i*width+8,5),f"{state['phase']} | f{state['original_video_frame_index']} | {state['original_video_seconds']:.2f}s",font=font,fill='black')
    path=folder/'triplet.jpg';canvas.save(path,quality=95,subsampling=0)
    detail_paths=[]
    for i,state in enumerate(event['states']):
        detail=folder/(state['phase']+'_detail.jpg')
        canvas.crop((i*width,0,(i+1)*width,height+banner)).save(detail,quality=95,subsampling=0)
        detail_paths.append(detail)
    event_schema=json.loads(SCHEMA_PATH.read_text())['properties']['search_event_states']['properties']['events']['items']
    prompt=(ROOT/'prompts/search_event_detailed.zh.txt').read_text()
    if target is not None:
        prompt+='\n'+(ROOT/'prompts/search_event_target_context.zh.txt').read_text()
        prompt+='\n本次任务的搜索目标（来自该数据的三帧说明文件）：'+json.dumps(target,ensure_ascii=False)
    prompt+='\n随后三张图是同一拼图开始、中间、结束面板的高清版本，不是额外时刻。以每张真实帧号为准。手指接触点与相邻物体必须仔细区分；无法区分就描述工具簇附近，不要强行命名接触对象。不要写“准备取下”“似乎想”“动作持续”等不可从静态帧确认的意图或连续过程；每个中间和结束状态要独立描述该帧可见细节，不仅写“同上”。'
    data=local_call(folder,prompt,[path,*detail_paths],detailed_schema(event_schema),6144)
    if [s['phase'] for s in data['states']]!=['start','middle','end']:raise ValueError('Event phase mismatch')
    return data


def export_episode(row,scene,event_data):
    catalog={};events=[];units=[];inspections=[]
    episode=row['episode_id'];zone='zone:'+episode
    for source,data in zip(row['event_index']['events'],event_data,strict=True):
        eid=source['event_id'];ordinal=source['ordinal'];states=[];labels={}
        for src,annotation in zip(source['states'],data['states'],strict=True):
            state={**src,'phase_label':PHASE_LABEL[src['phase']],'state_id':eid+':'+src['phase'],
                   'frame_label':'f'+str(src['original_video_frame_index']),
                   'evidence_status':'local_qwen_detailed_triplet_annotation_unreviewed'}
            for key in ['state_description','hand_action','visibility_and_occlusion']:state[key]=annotation[key]
            state['focus_object_states']=[]
            for focus in annotation['focus_object_states']:
                label=focus['label']
                if label not in labels:labels[label]=f'{eid}:entity_{len(labels)+1:02d}'
                ref=labels[label]
                candidates=[o['id'] for o in scene['objects'] if o['identity']['name']==label]
                catalog[ref]={'label':label,'memory_entity_candidates':candidates,
                    'link_status':'exact_label_candidate_not_verified_instance' if candidates else 'event_local_entity_unlinked_to_scene'}
                state['focus_object_states'].append({'entity_ref':ref,'observed_state':focus['observed_state']})
            states.append(state)
        window=source['sample_window_s']
        event={'event_id':eid,'ordinal':ordinal,'event_summary':data['event_summary'],'observed_changes':data['observed_changes'],
            'search_region':data['search_region'],'search_region_status':'visible_hand_interaction_region_not_verified_intention',
            'source_human_interval_id':None,'states':states,'phase_frame_indices':{s['phase']:s['original_video_frame_index'] for s in states},
            'supplied_image_interval_seconds':window,'previous_json_interval_seconds':[],
            'interval_alignment':{'status':'not_compared','basis_for_this_record':'user_supplied_image_filename_frame_indices','note':'直接使用用户事件三帧，不引用旧JSON区间。'},
            'outcome':{'target_found':None,'target_identity_confirmed':False,'reason':'尚无目标身份或成功确认；'+data['uncertainty']}}
        events.append(event)
        unit=f'{episode}:inspection_{ordinal:02d}'
        units.append({'unit_id':unit,'zone_id':zone,'label':data['search_region'],'type':'inspection_area','is_container':None,
            'boundary':'事件三帧中实际接触/操作所涉及的局部范围；'+data['search_region'],
            'source_entity_ids':[],'entity_link_basis':'事件局部操作区域；未强行关联为场景中的特定物体实例。'})
        inspections.append({'event_id':eid,'order':ordinal,'unit_ids':[unit],'granularity':'single_unit_event',
            'action_summary':data['event_summary'],'sample_window_s':window,'source_region_text':data['search_region'],
            'result':'unknown','result_source':'no_explicit_target_success_confirmation',
            'unit_link_basis':'one_event_local_inspection_area; may include several touched objects; not a unique physical container'})
    section={'schema':'ego.search_event_triplet_states.v1','image_directory':row['event_images'],'event_count':len(events),'image_count':3*len(events),
        'frame_numbering':{'source':'filename suffix _f<number>.jpg','original_video_fps':int(row['fps']),'original_video_frame_index_base':0,
            'fps_source':'source event markdown','time_formula':'original_video_frame_index / fps','namespace_note':'原视频帧号，与rosbag采样序号不同。'},
        'provenance':{'description_source':'本地Qwen根据三帧说明文件中的搜索目标，按详细提示词一次读取每个事件的带帧号三帧拼图。',
            'coverage':'仅前中后三个采样时刻，不补造连续动作。','knowledge_scope':'详细事件描述；同名场景物体只作为候选关联，搜索成功未知。'},
        'entity_reference_catalog':catalog,'events':events}
    doc={'schema':'spatial_memory.search_episode.p0_p1_2.v1',
        'README':{'purpose':'Astra结合图像和TF筛帧，Qwen生成空间、物体档案和详细事件描述。',
            'scope':['场景参考布局及物体外观/常识/容纳潜力','每个事件三帧详细描述','物体历史及最后观测按用户要求删除'],
            'rules':['保持参考JSON的九个顶层字段；物体档案使用无历史契约。','常识及潜在容纳能力不作为已观察事实。','物体数量未穷尽、未知结果不臆断。']},
        'metadata':{'episode_id':episode,'scene_id':'scene_'+episode,'zone_id':zone,'fps':row['fps'],'time_axis':'original_video_seconds',
            'search_scope':'所给rosbag及事件图像覆盖的局部工作区','search_scope_source':'provided_evidence_only',
            'full_search_coverage_confirmed':False,'only_final_event_found_target_confirmed':False,
            'target':{'description':row['target'],'category':None,'confirmed_instance_id':None,'identity_source':'user_supplied_markdown_target_name_only'},
            'provenance':'source_bag='+row.get('source_bag','demo_prepared_frames')+'; Astra筛帧，TF仅作未经独立验证的相机视角先验；Qwen生成语义；程序装配与校验。'},
        'scene_memory':scene,'search_event_states':section,
        'search_units':{'scope':'事件局部检查区域','catalog_status':'event_local_areas_not_verified_unique_places',
            'coverage_note':'检查区域不保证完全覆盖场景；同一实物区域在不同事件可重复出现。','id_policy':'episode_and_event_ordinal','list':units},
        'inspections':inspections,'episode_outcome':{'result':'unknown','success_event_id':None,'success_event_order':None,'target_instance_id':None,'final_location':None},
        'derived_statistics':{'counting_unit':'original_search_event','search_event_count':len(events),'search_unit_catalog_count':len(units),
            'found_event_count':0,'events_before_success':None,'events_through_success':None,'note':'found_event_count仅统计已确认成功事件；0不表示搜索失败。'}}
    jsonschema.Draft202012Validator(full_contract()).validate(doc)
    assert len(events)==row['event_count'] and sum(len(e['states']) for e in events)==3*row['event_count']
    assert all(f['entity_ref'] in catalog for e in events for s in e['states'] for f in s['focus_object_states'])
    assert all(set(c['memory_entity_candidates'])<={o['id'] for o in scene['objects']} for c in catalog.values())
    return doc


def prepare_episode(row,out):
    folder=out/'episodes'/row['episode_id']
    folder.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter();started_at=now()
    preparation_cache_hit=(folder/'selection/selected_frames.json').exists()
    extract_rgb_tf(row,folder/'input');extraction_s=time.perf_counter()-tick
    selection=folder/'selection'
    if not (selection/'selected_frames.json').exists():
        command=[sys.executable,str(ROOT/'scripts/select_scene_keyframes.py'),'--input',str(folder/'input'),'--output',str(selection)]
        with (folder/'selection.log').open('a') as log:
            for attempt in range(2):
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==0:break
                if attempt==1:raise subprocess.CalledProcessError(result.returncode,command)
    if not (folder/'prepare_timing.json').exists():
        atomic(folder/'prepare_timing.json',{'started_at':started_at,'finished_at':now(),
            'seconds':time.perf_counter()-tick,'extraction_seconds':extraction_s,
            'selection_seconds':time.perf_counter()-tick-extraction_s,'preparation_cache_hit':preparation_cache_hit})
    atomic(folder/'prepare_timing_target_v1.json',{'started_at':started_at,'finished_at':now(),
        'seconds':time.perf_counter()-tick,'extraction_seconds':extraction_s,
        'selection_seconds':time.perf_counter()-tick-extraction_s,'preparation_cache_hit':preparation_cache_hit})
    return folder


def record_episode_timing(row,folder,out,local_started,local_seconds):
    preparation=json.loads((folder/'prepare_timing_target_v1.json').read_text())
    scene=json.loads((folder/'selection/qwen_scene_v2/timing.json').read_text())
    events=[{'ordinal':e['ordinal'],**json.loads((folder/'events_target_v1'/f"event_{e['ordinal']:02d}"/'timing.json').read_text())} for e in row['event_index']['events']]
    astra=[json.loads(p.read_text()) for p in sorted((folder/'selection').glob('astra_*/timing.json'))]
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(preparation['started_at'])).total_seconds()
    scene_reused=scene['started_at']<local_started
    cached_events=sum(e['started_at']<local_started for e in events)
    report={'episode_id':row['episode_id'],'annotation_version':'target_context_v1','finished_at':now(),'event_count':len(events),
        'reannotation':(out/'json_before_target_context'/(row['episode_id']+'.json')).exists(),
        'scene_cache_reused':scene_reused,'cached_event_count':cached_events,
        'prepare':preparation,'astra_calls':astra,'qwen_scene':scene,'qwen_events':events,
        'local_stage_started_at':local_started,'local_stage_wall_seconds':local_seconds,
        'end_to_end_wall_seconds':elapsed,'mean_event_request_seconds':statistics.mean(e['seconds'] for e in events),
        'note':'This attempt wall time includes prefetch wait, excludes original cached preparation/scene time. Request histories may be reused and overlap; never sum them as wall time.'}
    atomic(folder/'episode_timing_target_v1.json',report)
    reports=[json.loads(p.read_text()) for p in sorted((out/'episodes').glob('*/episode_timing_target_v1.json'))]
    times=[r['end_to_end_wall_seconds'] for r in reports]
    local_times=[r['local_stage_wall_seconds'] for r in reports]
    summary={'updated_at':now(),'completed_episodes':len(reports),'mean_end_to_end_seconds':statistics.mean(times),
        'median_end_to_end_seconds':statistics.median(times),'mean_local_stage_seconds':statistics.mean(local_times),
        'episodes':[{'episode_id':r['episode_id'],'events':r['event_count'],
            'total_seconds':r['end_to_end_wall_seconds'],'local_seconds':r['local_stage_wall_seconds']} for r in reports],
        'note':'Target-context attempts only; old-version timing_summary.json is preserved. Reannotation reuses scene/preparation. Do not sum concurrent request times.'}
    session_start=json.loads((out/'launch.json').read_text())['started_at']
    current=[r for r in reports if r['local_stage_started_at']>=session_start]
    wall=(datetime.now(timezone.utc)-datetime.fromisoformat(session_start)).total_seconds()
    summary['current_run']={'started_at':session_start,'wall_seconds':wall,'completed_episodes':len(current),
        'events':sum(r['event_count'] for r in current),'episodes_per_hour':len(current)*3600/wall,
        'events_per_minute':sum(r['event_count'] for r in current)*60/wall,
        'note':'Completed output throughput across this process lifetime, including cache reuse; not raw model generation speed.'}
    atomic(out/'timing_summary_target_v1.json',summary)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--todo',type=Path,default=Path(str(REPOSITORY / 'legacy/zl/todo')))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--inventory-only',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--legacy-static',action='store_true',help='Explicit historical reproduction only; default delivery uses nested temporal trees')
    args=parser.parse_args()
    if not args.legacy_static:
        from run_scene_temporal_batch import run
        return run(args.todo,args.output,args.inventory_only,args.limit)
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'.batch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    rows=inventory(args.todo,out);schema=full_contract();atomic(out/'target.schema.json',schema)
    if args.inventory_only:
        print(json.dumps({'ready':sum(not r['problems'] for r in rows),'blocked':sum(bool(r['problems']) for r in rows)}));return
    ready=[r for r in rows if not r['problems']]
    if args.limit:ready=ready[:args.limit]
    state={'status':'running','annotation_version':'target_context_v1','started_at':now(),'episode_count':len(ready),
        'episodes':{},'local_concurrency':2,'astra_selection_concurrency':1}
    def update(eid,**values):
        state['episodes'].setdefault(eid,{}).update(values)
        state['updated_at']=now();state['completed_count']=sum(v.get('status')=='complete' for v in state['episodes'].values())
        atomic(out/'batch_status.json',state)
    record=json.loads((ROOT/'logs/server.pid.json').read_text());server,workers=verify_processes(record)
    stopped=server.status()==psutil.STATUS_STOPPED
    if stopped:server.send_signal(signal.SIGCONT)
    atomic(out/'launch.json',{'pid':psutil.Process().pid,'create_time':psutil.Process().create_time(),'server_pid':server.pid,
        'workers':workers,'started_at':now(),'annotation_version':'target_context_v1'})
    try:
        pending=[]
        archive=out/'json_before_target_context'
        for row in ready:
            output=out/'json'/(row['episode_id']+'.json')
            marker=out/'episodes'/row['episode_id']/'target_context_complete.json'
            saved=json.loads(marker.read_text()) if marker.exists() else {}
            if (output.exists() and saved.get('sha256')==sha(output)
                    and saved.get('target')==row['target'] and saved.get('target_markdown_sha256')==row['target_markdown_sha256']):
                jsonschema.validate(json.loads(output.read_text()),schema);update(row['episode_id'],status='complete',output=str(output));continue
            update(row['episode_id'],status='pending_target_reannotation' if output.exists() else 'pending',
                previous_output_available=output.exists())
            pending.append(row)
        with ThreadPoolExecutor(max_workers=1) as selection_pool,ThreadPoolExecutor(max_workers=2) as local_pool:
            future=selection_pool.submit(prepare_episode,pending[0],out) if pending else None
            for index,row in enumerate(pending):
                eid=row['episode_id'];update(eid,status='astra_selection',stage_started_at=now())
                try:folder=future.result()
                except Exception as exc:
                    update(eid,status='failed',stage='selection',error_type=type(exc).__name__)
                    future=selection_pool.submit(prepare_episode,pending[index+1],out) if index+1<len(pending) else None
                    continue
                future=selection_pool.submit(prepare_episode,pending[index+1],out) if index+1<len(pending) else None
                update(eid,status='qwen_scene_and_events',stage_started_at=now())
                local_started=now();local_tick=time.perf_counter()
                tasks={local_pool.submit(scene_job,folder/'selection'):('scene',None)}
                for event in row['event_index']['events']:
                    tasks[local_pool.submit(event_job,event,folder/'events_target_v1'/f"event_{event['ordinal']:02d}",row['target'])]=('event',event['ordinal'])
                scene=None;events={};errors=[]
                for task in as_completed(tasks):
                    kind,ordinal=tasks[task]
                    try:
                        result=task.result()
                        if kind=='scene':scene=result
                        else:events[ordinal]=result
                    except Exception as exc:errors.append({'stage':kind,'ordinal':ordinal,'error_type':type(exc).__name__})
                    update(eid,events_completed=len(events),scene_complete=scene is not None,errors=errors)
                if errors: update(eid,status='failed');continue
                try:
                    doc=export_episode(row,scene,[events[e['ordinal']] for e in row['event_index']['events']])
                    output=out/'json'/(eid+'.json')
                    if output.exists() and not (archive/(eid+'.json')).exists():
                        archive.mkdir(parents=True,exist_ok=True)
                        atomic(archive/(eid+'.json'),json.loads(output.read_text()))
                    atomic(output,doc)
                    timing=record_episode_timing(row,folder,out,local_started,time.perf_counter()-local_tick)
                    atomic(folder/'target_context_complete.json',{'sha256':sha(output),'target':row['target'],
                        'target_markdown_sha256':row['target_markdown_sha256'],'finished_at':now()})
                    update(eid,status='complete',output=str(output),sha256=sha(output),finished_at=now(),
                        end_to_end_wall_seconds=timing['end_to_end_wall_seconds'],local_stage_wall_seconds=timing['local_stage_wall_seconds'])
                except Exception as exc:update(eid,status='failed',stage='assembly',error_type=type(exc).__name__)
        state['status']='complete' if all(state['episodes'].get(r['episode_id'],{}).get('status')=='complete' for r in ready) else 'finished_with_failures'
    except BaseException as exc:
        state.update(status='interrupted_or_failed',error_type=type(exc).__name__)
        raise
    finally:
        if stopped:
            server,workers=verify_processes(record);server.send_signal(signal.SIGSTOP)
            atomic(out/'pause_restored.json',{'server_pid':server.pid,'workers':workers,'at':now()})
        state['finished_at']=now();atomic(out/'batch_status.json',state)


if __name__=='__main__':
    main()
