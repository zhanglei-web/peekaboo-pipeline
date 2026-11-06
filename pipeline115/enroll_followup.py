"""Enroll independent input subsets; skip ambiguous evidence with explicit reasons."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import re
import sys
from .common import ROOT,QWEN,read,sha,atomic,bind,now
from .enroll_batch import REQUIRED
from .location_facts import field,verify


def initial_photo(dataset,scene,target,raw_value):
    value={'status':'not_provided','description':None,'source_file':None,'source_sha256':None,
           'filename':None,'target_name_in_filename':None,
           'matching_rule':'Markdown explicitly records no supplied initial location','raw_markdown_value':raw_value}
    if raw_value=='无':return value
    folder=dataset/'原始位置照片'/scene
    if not folder.is_dir():raise ValueError('Missing same-scene initial photo directory')
    matches=[]
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in ('.jpg','.jpeg','.png'):continue
        for prefix,rule in [(scene+target,'same scene; exact target name'),
                            (target,'same scene directory; exact target filename without scene prefix')]:
            if p.stem.startswith(prefix):
                description=p.stem[len(prefix):].lstrip('—-_ ')
                # An explicit descriptive initial location is an independent check
                # against accidentally matching a longer, similarly named target.
                if description and (raw_value=='1' or description==raw_value):
                    matches.append((p,description,rule))
                break
    if len(matches)!=1:raise ValueError('Missing/ambiguous same-scene exact-target initial photo or location mismatch')
    photo,description,rule=matches[0]
    value.update(status='recorded_filename',description=description,source_file=str(photo),
                 source_sha256=sha(photo),filename=photo.name,target_name_in_filename=target,matching_rule=rule)
    return value


def build_row(dataset,raw,md):
    sys.path.insert(0,str(QWEN/'scripts'))
    from run_scene_event_batch import triplets
    eid=raw.name;text=md.read_text(encoding='utf-8-sig');meta=read(raw/'meta.json')
    metadata={k:field(text,k) for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']}
    scene=metadata['场景'];target=metadata['要找的物体']
    if md.relative_to(dataset/'标注数据').parts[:2]!=(scene,target):raise ValueError('Scene/target path differs from Markdown')
    match=re.search(r'帧率\s*(\d+(?:\.\d+)?)\s*帧/秒',text)
    if not match:raise ValueError('Missing FPS')
    fps=float(match[1])
    if fps!=meta['fps']:raise ValueError('FPS differs from capture metadata')
    table=[[int(x) for x in m] for m in re.findall(r'^\|\s*第(\d+)次\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|',text,re.M)]
    indexes=[triplets(md.parent/eye,eid,fps) for eye in ('left','right')]
    for index,eye in zip(indexes,('left','right')):
        actual=[[e['ordinal'],*[s['original_video_frame_index'] for s in e['states']]] for e in index['events']]
        if table!=actual:raise ValueError('Markdown/image label mismatch: '+eye)
        if not table or max(f for e in table for f in e[1:])>=meta['topics'][eye]['written_frames']:
            raise ValueError('Triplet exceeds original '+eye+' frame bounds')
    if any(not (raw/rel).is_file() for rel in REQUIRED):raise ValueError('Missing required raw capture file')
    if not metadata['物体最后所在位置'] or metadata['物体最后所在位置'] in ('未知','不清楚'):
        raise ValueError('Actual placement description is unspecified')
    row={'episode_id':eid,'source':str(raw),'scene_group':scene,'target':target,'fps':fps,
         'target_markdown':str(md),'target_markdown_sha256':sha(md),'event_images':str(md.parent/'left'),
         'event_count':indexes[0]['event_count'],'event_index':indexes[0],'right_event_index':indexes[1],
         'source_frame_count':meta['topics']['left']['written_frames'],'metadata_sha256':sha(raw/'meta.json'),
         'source_annotation_metadata':metadata,'problems':[]}
    row['dataset_location_facts']={'schema':'owner_supplied_episode_locations.v1','episode_id':eid,
        'scene_id':scene,'target_name':target,'initial_location':initial_photo(dataset,scene,target,metadata['物体原始位置']),
        'actual_location':{'description':metadata['物体最后所在位置'],'source_file':str(md),'source_sha256':sha(md),
            'field_name':'物体最后所在位置','source_type':'owner_confirmed_markdown',
            'temporal_scope':'episode_level_record_no_within_episode_relocation_time',
            'associated_event_ids':[e['event_id'] for e in indexes[0]['events']],
            'search_unit_id':None,'relocation_time_s':None},
        'interpretation':'Owner requests the same annotation requirements as the previous batch: initial location from same-scene photo filename, actual placement from episode Markdown. No invented node identity or relocation timing.'}
    verify(row);return row,meta


def enroll_subset(dataset,output,base_config):
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output==dataset or dataset in output.parents:raise ValueError('Output overlaps source')
    raws={p.name:p for p in (dataset/'baai_ego_task').iterdir() if p.is_dir()}
    mds={}
    for p in sorted((dataset/'标注数据').rglob('*_三帧.md')):mds.setdefault(p.parent.name,[]).append(p)
    template=read(read(ROOT/'inventory.json')['records'][0]['config'])
    episodes=[];records=[];skips=[]
    for eid in sorted(set(raws)|set(mds)):
        try:
            if eid not in raws or len(mds.get(eid,[]))!=1:raise ValueError('Missing/duplicate raw or Markdown identity')
            row,meta=build_row(dataset,raws[eid],mds[eid][0])
            cfg=deepcopy(template);cfg.update(source=row['source'],output=str(output/'run001/preprocessed'/eid))
            cfg['ros_output']={**cfg.get('ros_output',{}),'enabled':False,'mode':'none'}
            path=output/'input/configs'/(eid+'.json');bind(path,cfg)
            records.append({'id':eid,'name':eid,'group':row['scene_group'],'source':row['source'],'config':str(path),
                            'output':cfg['output'],'duration_s':meta['duration_sec'],'meta_sha256':row['metadata_sha256']})
            episodes.append(row)
        except (ValueError,KeyError,IndexError,OSError,TypeError) as exc:
            skips.append({'episode_id':eid,'source':str(raws[eid]) if eid in raws else None,
                          'markdown_files':[str(p) for p in mds.get(eid,[])],
                          'status':'skipped_input_question','reason':str(exc),'error_type':type(exc).__name__})
    bind(output/'input/annotation_inventory.json',{'total':len(raws),'ready_count':len(episodes),'blocked_count':len(skips),
         'events':sum(r['event_count'] for r in episodes),'episodes':episodes})
    bind(output/'input/preprocessing_inventory.json',{'records':records,'record_count':len(records),'input_root':str(dataset)})
    facts=read(ROOT/'dataset_facts.json');facts.update(version='11-5',fact_id='followup138_same_requirements_20261105',
        user_statement='完成当前72条后处理11-5标注目录下138组；要求和上边任务一致；有疑问跳过该条。',
        interpretation='Explicit same-requirements instruction applies prior earlier-not-found/last-found dataset facts only to the enumerated new subset.')
    facts['scope']={'dataset':str(dataset),'episode_ids':[r['episode_id'] for r in episodes],
                    'episode_count':len(episodes),'search_event_count':sum(r['event_count'] for r in episodes),
                    'scope_note':'Same-requirements authorization for this enumerated subset; not a universal future-data rule.'}
    bind(output/'input/dataset_facts.json',facts)
    report={'source':str(dataset),'raw_count':len(raws),'markdown_count':sum(map(len,mds.values())),
            'ready_count':len(episodes),'skipped_count':len(skips),'skipped':skips,
            'event_count':sum(r['event_count'] for r in episodes),'scene_counts':dict(Counter(r['scene_group'] for r in episodes)),
            'scope':'Structural enrollment; mandatory original-video/image content verification remains in CPU stage.',
            'policy':'Input ambiguity or incomplete evidence is skipped; no inherited right-frame exception or inferred aliases.'}
    bind(output/'input/enrollment_report.json',report)
    cfg=deepcopy(base_config);cfg.update(output_root=str(output/'run001'),annotation_inventory=str(output/'input/annotation_inventory.json'),
        preprocessing_inventory=str(output/'input/preprocessing_inventory.json'),facts=str(output/'input/dataset_facts.json'))
    bind(output/'config.json',cfg)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-config',type=Path,required=True);args=p.parse_args()
    print(enroll_subset(args.dataset,args.output,read(args.base_config)))

if __name__=='__main__':main()
