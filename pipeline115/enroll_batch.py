"""Enroll the exact remaining captures, preserving reference photos and Markdown facts."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse
import re
import sys
from .common import ROOT,QWEN,read,sha,atomic,bind,now
from .location_facts import field,verify

REQUIRED=['meta.json','videos/left.mkv','videos/right.mkv','imu/imu0.csv',
          'timestamps/left_timestamps.txt','timestamps/right_timestamps.txt',
          'calibrations/camera-camchain.yaml','calibrations/camera_imu-camchain-imucam.yaml',
          'calibrations/camera_imu-imu.yaml']

def enroll(dataset,prior_path,output,aliases,expected,exception_path=None):
    sys.path.insert(0,str(QWEN/'scripts'))
    from run_scene_event_batch import triplets
    dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output==dataset or dataset in output.parents:raise ValueError('Enrollment must be outside source data')
    prior=read(prior_path)['episodes'];old={r['episode_id']:r for r in prior}
    raws={p.name:p for p in (dataset/'baai_ego_task').iterdir() if p.is_dir()}
    mds={}
    for p in sorted((dataset/'标注数据').rglob('*_三帧.md')):
        if p.parent.name in mds:raise ValueError('Duplicate episode Markdown: '+p.parent.name)
        mds[p.parent.name]=p
    if set(raws)!=set(mds) or not set(old)<=set(raws):raise ValueError('Raw/Markdown/prior episode identity mismatch')
    remaining=sorted(set(raws)-set(old))
    if len(remaining)!=expected:raise ValueError('Unexpected remaining count: '+str(len(remaining)))
    excluded=[]
    for eid,row in sorted(old.items()):
        old_meta=read(Path(row['source'])/'meta.json');new_meta=read(raws[eid]/'meta.json')
        if old_meta['data_id']!=new_meta['data_id']:raise ValueError('Prior capture identity changed: '+eid)
        hashes={}
        for rel in REQUIRED:
            a=sha(Path(row['source'])/rel);b=sha(raws[eid]/rel)
            if a!=b:raise ValueError('Previously processed raw input differs: '+eid+'/'+rel)
            hashes[rel]=b
        complete=ROOT.parent/'search_episode_outputs/episodes'/eid/'complete.json'
        result=ROOT.parent/'search_episode_outputs/json'/(eid+'.json')
        if read(complete)['sha256']!=sha(result):raise ValueError('Prior completion hash mismatch: '+eid)
        excluded.append({'episode_id':eid,'capture_data_id':new_meta['data_id'],
                         'matched_required_raw_hashes':hashes,'prior_output':str(result),'prior_output_sha256':sha(result),
                         'reason':'same capture and previously completed output verified'})
    old_pre=read(ROOT/'inventory.json')
    template=read(old_pre['records'][0]['config']);records=[];episodes=[];warnings=[]
    for eid in remaining:
        raw=raws[eid];md=mds[eid];text=md.read_text(encoding='utf-8-sig');meta=read(raw/'meta.json')
        metadata={k:field(text,k) for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']}
        scene=metadata['场景'];target=metadata['要找的物体']
        if md.relative_to(dataset/'标注数据').parts[:2]!=(scene,target):raise ValueError('Scene/target path disagrees with Markdown')
        fps=float(re.search(r'帧率\s*(\d+(?:\.\d+)?)\s*帧/秒',text)[1])
        if fps!=meta['fps']:raise ValueError('FPS mismatch')
        table=[[int(x) for x in match] for match in re.findall(r'^\|\s*第(\d+)次\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|',text,re.M)]
        indexes=[triplets(md.parent/'left',eid,fps)];exception=None
        try:indexes.append(triplets(md.parent/'right',eid,fps))
        except ValueError:
            if exception_path is None:raise
            from .input_exceptions import partial_right
            right,exception=partial_right(raw,md,indexes[0],exception_path);indexes.append(right)
        for index in indexes[:1] if exception else indexes:
            if table!=[[e['ordinal'],*[s['original_video_frame_index'] for s in e['states']]] for e in index['events']]:
                raise ValueError('Triplet table/image labels differ: '+eid)
        if not table or max(f for e in table for f in e[1:])>=meta['topics']['left']['written_frames']:
            raise ValueError('Invalid frame count/bounds: '+eid)
        if any(not (raw/rel).is_file() for rel in REQUIRED):raise ValueError('Missing raw input: '+eid)
        cfg=deepcopy(template);cfg.update(source=str(raw),output=str(output/'run001/preprocessed'/eid))
        cfg['ros_output']={**cfg.get('ros_output',{}),'enabled':False,'mode':'none'}
        config=output/'input/configs'/(eid+'.json');bind(config,cfg)
        records.append({'id':eid,'name':eid,'group':scene,'source':str(raw),'config':str(config),
                        'output':cfg['output'],'duration_s':meta['duration_sec'],'meta_sha256':sha(raw/'meta.json')})
        row={'episode_id':eid,'source':str(raw),'scene_group':scene,'target':target,'fps':fps,
             'target_markdown':str(md),'target_markdown_sha256':sha(md),'event_images':str(md.parent/'left'),
             'event_count':indexes[0]['event_count'],'event_index':indexes[0],'right_event_index':indexes[1],
             'source_frame_count':meta['topics']['left']['written_frames'],'metadata_sha256':sha(raw/'meta.json'),
             'source_annotation_metadata':metadata,'problems':[]}
        if exception:
            row['input_evidence_exception']=exception
            warnings.append(exception)
        photo_target=aliases.get(scene,{}).get(target,target)
        photos=sorted((dataset/'原始位置照片'/scene).glob(scene+photo_target+'*.jpg'))
        initial={'status':'not_provided','description':None,'source_file':None,'source_sha256':None,
                 'filename':None,'target_name_in_filename':None,'matching_rule':'Markdown explicitly records no supplied initial location',
                 'raw_markdown_value':metadata['物体原始位置']}
        if metadata['物体原始位置']!='无':
            if len(photos)!=1:raise ValueError('Need unambiguous same-scene initial photo: '+scene+'/'+target)
            photo=photos[0];description=photo.stem[len(scene+photo_target):].lstrip('—-_ ')
            if not description:raise ValueError('Initial photo filename lacks location')
            initial.update(status='recorded_filename',description=description,source_file=str(photo),source_sha256=sha(photo),
                           filename=photo.name,target_name_in_filename=photo_target,
                           matching_rule='same scene; '+('explicit alias '+target+' -> '+photo_target if photo_target!=target else 'exact target name'))
        else:warnings.append({'episode_id':eid,'kind':'initial_location_explicitly_not_provided'})
        row['dataset_location_facts']={'schema':'owner_supplied_episode_locations.v1','episode_id':eid,'scene_id':scene,'target_name':target,
            'initial_location':initial,'actual_location':{'description':metadata['物体最后所在位置'],
                'source_file':str(md),'source_sha256':sha(md),'field_name':'物体最后所在位置','source_type':'owner_confirmed_markdown',
                'temporal_scope':'episode_level_record_no_within_episode_relocation_time',
                'associated_event_ids':[e['event_id'] for e in indexes[0]['events']],
                'search_unit_id':None,'relocation_time_s':None},
            'interpretation':'Owner confirms the photo filename describes the initial location and episode Markdown describes actual placement. Preserve raw language. Do not infer a tree node, search-unit ID, relocation time or search outcome; MD triplets are inspections within one episode.'}
        verify(row);episodes.append(row)
    bind(output/'input/preprocessing_inventory.json',{'records':records,'record_count':len(records),'input_root':str(dataset)})
    bind(output/'input/annotation_inventory.json',{'total':len(episodes),'ready_count':len(episodes),'blocked_count':0,
         'events':sum(r['event_count'] for r in episodes),'episodes':episodes})
    report={'source':str(dataset),'raw_count':len(raws),'prior_count':len(excluded),'remaining_count':len(episodes),
        'events':sum(r['event_count'] for r in episodes),'scene_counts':dict(Counter(r['scene_group'] for r in episodes)),
        'initial_location_counts':dict(Counter(r['dataset_location_facts']['initial_location']['status'] for r in episodes)),
        'aliases':aliases,'excluded':excluded,'warnings':warnings,
        'scope':'Enrollment/labels/hash verification; actual video-content matching remains a mandatory pipeline CPU-stage gate'}
    bind(output/'input/enrollment_report.json',report)
    config=read(ROOT/'pipeline115.example.json');config.update(output_root=str(output/'run001'),
        annotation_inventory=str(output/'input/annotation_inventory.json'),
        preprocessing_inventory=str(output/'input/preprocessing_inventory.json'),
        facts=str(output/'input/dataset_facts.json') if (output/'input/dataset_facts.json').exists() else None)
    bind(output/'config.json',config)
    return {k:v for k,v in report.items() if k not in ('excluded','warnings')}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-inventory',type=Path,default=ROOT/'annotation_inventory.json')
    parser.add_argument('--aliases',type=Path,required=True);parser.add_argument('--expected',type=int,required=True)
    parser.add_argument('--input-exception',type=Path)
    args=parser.parse_args()
    import json
    print(json.dumps(enroll(args.dataset,args.prior_inventory,args.output,read(args.aliases),args.expected,args.input_exception),ensure_ascii=False))

if __name__=='__main__':main()
