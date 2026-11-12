#!/usr/bin/env python3
"""Portable entry point; prepare performs no model inference or network requests."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATHS = [ROOT/'components/annotation', ROOT/'components/additive',
         ROOT/'components/semantic/scripts', ROOT/'components/semantic/src',
         ROOT/'components/semantic/integrations/astra_pipeline/src',
         ROOT/'components/preprocessing/src', ROOT/'components/anchors']

def environment():
    env=os.environ.copy()
    env['PYTHONPATH']=os.pathsep.join(map(str,PATHS))+os.pathsep+env.get('PYTHONPATH','')
    env.setdefault('DAAAM_OUTPUT_DIR',str(ROOT/'runs/daaam'))
    return env

def activate():
    sys.path[:0]=list(map(str,PATHS))
    os.environ.update(environment())

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if path.exists() and path.read_text()!=content:raise ValueError('Refusing to overwrite different file: '+str(path))
    if not path.exists():path.write_text(content)

def resolve(value,base,*,follow_symlinks=True):
    expanded=os.path.expandvars(str(value))
    if '$' in expanded or '<' in expanded:raise ValueError('Unresolved path placeholder: '+str(value))
    p=Path(expanded).expanduser()
    absolute=p if p.is_absolute() else Path(base)/p
    return absolute.resolve() if follow_symlinks else Path(os.path.abspath(absolute))

def safe_id(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',value) or '..' in value:
        raise ValueError('Unsafe record ID: '+repr(value))
    return value

def disjoint(output,inputs):
    for source in inputs:
        source=Path(source).resolve()
        if output==source or source in output.parents or output in source.parents:
            raise ValueError('Output overlaps input: '+str(source))

def prepare(manifest_path,output):
    """Enroll explicit captures and manually selected image triplets for pipeline115."""
    activate()
    from astra_memory.inputs import triplets
    from pipeline115.common import verify_dependencies
    verify_dependencies()
    manifest_path=Path(manifest_path).resolve();base=manifest_path.parent;m=read(manifest_path)
    output=Path(output).resolve()
    disjoint(output,[manifest_path,ROOT/'components',ROOT/'configs'])
    rows=[];records=[];configs=[];ids=set()
    for item in m['episodes']:
        eid=safe_id(item['episode_id'])
        if eid in ids:raise ValueError('Duplicate episode ID: '+eid)
        ids.add(eid)
        raw=resolve(item['capture'],base);md=resolve(item['markdown'],base)
        left=resolve(item['left_triplets'],base);right=resolve(item['right_triplets'],base)
        disjoint(output,[raw,md,left,right])
        metadata=read(raw/'meta.json');fps=float(item['fps'])
        if fps<=0 or fps!=float(metadata['fps']):raise ValueError('FPS differs from capture metadata')
        text=md.read_text(encoding='utf-8-sig')
        label_fps=re.search(r'帧率\s*(\d+(?:\.\d+)?)\s*帧/秒',text)
        if label_fps is None or float(label_fps[1])!=fps:raise ValueError('Markdown must record matching FPS')
        indexes=[triplets(left,eid,fps),triplets(right,eid,fps)]
        table=[[int(x) for x in match] for match in re.findall(r'^\|\s*第(\d+)次\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|',text,re.M)]
        for index,eye in zip(indexes,['left','right']):
            actual=[[e['ordinal'],*[s['original_video_frame_index'] for s in e['states']]] for e in index['events']]
            if not table or actual!=table:raise ValueError('Markdown/image frame mismatch: '+eye)
            if max(f for e in table for f in e[1:])>=metadata['topics'][eye]['written_frames']:
                raise ValueError('Triplet exceeds capture frame bounds')
        target=item['target']
        if not isinstance(target,str) or not target.strip():raise ValueError('Explicit target required')
        fields={k:None for k in ['采集人员','场景','要找的物体','是否在容器内','是否有曝光','物体原始位置','物体最后所在位置']}
        for k in fields:
            found=re.search(r'(?:^|\n)\s*(?:[-*]\s*)?'+re.escape(k)+r'\s*[:：]\s*(.+)',text)
            if found:fields[k]=found[1].strip()
        if fields['要找的物体']!=target:raise ValueError('Target differs from Markdown')
        row={'episode_id':eid,'source':str(raw),'scene_group':fields['场景'] or 'unknown',
             'target':target,'fps':fps,'target_markdown':str(md),'target_markdown_sha256':sha(md),
             'event_images':str(left),'event_count':indexes[0]['event_count'],
             'event_index':indexes[0],'right_event_index':indexes[1],
             'source_frame_count':metadata['topics']['left']['written_frames'],
             'metadata_sha256':sha(raw/'meta.json'),'source_annotation_metadata':fields,'problems':[]}
        if item.get('location_facts'):
            facts_path=resolve(item['location_facts'],base)
            disjoint(output,[facts_path])
            row['dataset_location_facts']=read(facts_path)
            from pipeline115.location_facts import verify
            verify(row)
        rows.append(row)
        cfg=deepcopy(m['preprocessing']);cfg.update(source=str(raw),output=str(output/'run/preprocessed'/eid))
        cfg['openvins_binary']=str(resolve(cfg['openvins_binary'],base))
        for key in ['repo','checkpoint'] :cfg['depth'][key]=str(resolve(cfg['depth'][key],base))
        # A venv Python is commonly a symlink; resolving it loses pyvenv.cfg.
        cfg['depth']['python']=str(resolve(cfg['depth']['python'],base,follow_symlinks=False)) if cfg['depth'].get('python') else sys.executable
        cfg['ros_output']={'enabled':False,'mode':'none'}
        path=output/'preprocessing_configs'/(eid+'.json');configs.append((path,cfg))
        records.append({'id':eid,'source':str(raw),'config':str(path)})
    if not rows:raise ValueError('No episodes')
    selector=deepcopy(m['selector']);endpoints=deepcopy(m['endpoints'])
    for p in [selector['model'],*endpoints.values()]:
        if any(k.lower() in {'api_key','token','authorization','password'} for k in p):raise ValueError('Use api_key_env; never store credentials in manifests')
    gpu={'device':0,'lock_path':os.environ.get('EGOMEMORY_GPU_LOCK',str(Path.home()/'.cache/egomemory/gpu0.lock')),
         'depth_min_free_mib':10000,'local_min_free_mib':20000,'local_server_config':str(output/'local_server.json')}
    local=m.get('local_server',{})
    if any(p['placement']=='local_gpu' for p in endpoints.values()) and not local:raise ValueError('Local endpoints require explicit local_server configuration')
    cfg={'version':'11-5','output_root':str(output/'run'),'annotation_inventory':str(output/'annotation_inventory.json'),
         'preprocessing_inventory':str(output/'preprocessing_inventory.json'),'facts':str(resolve(m['facts'],base)) if m.get('facts') else None,
         'astra_config':str(output/'selector.json'),'routes':m['routes'],'endpoints':endpoints,
         'limits':m.get('limits',{'cpu_workers':1,'queue_episodes':2,'queue_bytes':42949672960,
             'min_available_ram_mib':8192,'min_free_disk_mib':20480,'depth_group_size':1,'local_residency_budget_s':900}), 'gpu':gpu}
    # Validate config before publishing any inventory. Model endpoints are not contacted.
    from pipeline115.engine import load_config
    for path,value in configs:write(path,value)
    write(output/'annotation_inventory.json',{'episodes':rows})
    write(output/'preprocessing_inventory.json',{'records':records})
    write(output/'selector.json',selector);write(output/'local_server.json',local);write(output/'pipeline.json',cfg)
    load_config(output/'pipeline.json')
    print(json.dumps({'status':'prepared','episodes':len(rows),'config':str(output/'pipeline.json'),'model_calls':0},ensure_ascii=False))

def prepare_anchors(sources_path,models_path,output):
    from PIL import Image
    sources_path=Path(sources_path).resolve();models_path=Path(models_path).resolve();output=Path(output).resolve()
    disjoint(output,[sources_path,models_path,ROOT/'components',ROOT/'configs'])
    records=[];ids=set();groups=[]
    for group in read(sources_path)['groups']:
        name=group['name']
        for part in name.split('/'):safe_id(part)
        folder=resolve(group['directory'],sources_path.parent);disjoint(output,[folder])
        files=sorted(folder.glob('*.json'))
        if not files:raise ValueError('No source JSON files: '+str(folder))
        for path in files:
            rid=safe_id(path.stem);key=name+'/'+rid
            if key in ids:raise ValueError('Duplicate grouped record: '+key)
            ids.add(key);d=read(path);frames=[];state_ids=set()
            events=d['search_event_states']['events']
            if not events:raise ValueError('No events: '+key)
            for e in events:
                if [s['phase'] for s in e['states']]!=['start','middle','end']:raise ValueError('Invalid triplet order')
                for s in e['states']:
                    image=Path(s['image_path'])
                    if not image.is_absolute():raise ValueError('Source image paths must be absolute; relocate input explicitly')
                    disjoint(output,[image])
                    if s['state_id'] in state_ids:raise ValueError('Duplicate state ID')
                    state_ids.add(s['state_id'])
                    h=sha(image)
                    if h!=s['image_sha256']:raise ValueError('Image hash mismatch: '+str(image))
                    with Image.open(image) as im:im.load();size=list(im.size)
                    frames.append({'event_id':e['event_id'],'state_id':s['state_id'],'phase':s['phase'],
                        'frame_index':s['original_video_frame_index'],'path':str(image),
                        'expected_sha256':h,'actual_sha256':h,'size':size})
            records.append({'group':name,'record_id':rid,'episode_id':d['metadata']['episode_id'],
                'source_json':str(path),'source_sha256':sha(path),'event_count':len(events),'frame_count':len(frames),'frames':frames,'errors':[]})
        groups.append({'group':name,'source_directory':str(folder),'expected':len(files)})
    if not records:raise ValueError('No records')
    models=read(models_path)
    for key in ['grounding_dino_path','fastsam_weights','gpu_lock']:models[key]=str(resolve(models[key],models_path.parent))
    write(output/'input_inventory.json',{'expected_records':len(records),'records':records,'groups':groups,'errors':[]})
    write(output/'model_config.json',models)
    call(ROOT/'components/anchors/scripts/prepare_resume50.py',['--root',str(output),'--ready'])
    report=read(output/'selection_preparation_report.json')
    if report['errors'] or report['empty_events']:raise ValueError('Candidate maps need review; inspect selection_preparation_report.json before inference')

def call(script,args):
    subprocess.run([sys.executable,str(script),*map(str,args)],env=environment(),check=True)

def doctor():
    modules=['numpy','scipy','PIL','cv2','jsonschema','requests','psutil','yaml']
    report={'python':sys.version.split()[0],'platform':sys.platform,
            'dependencies':{m:importlib.util.find_spec(m) is not None for m in modules},
            'inference_modules':{m:importlib.util.find_spec(m) is not None for m in ['torch','transformers','ultralytics','rclpy']},
            'model_calls':0,'note':'Does not certify GPU, model weights, ROS overlay or model endpoints.'}
    activate()
    from pipeline115.common import verify_dependencies
    verify_dependencies();report['source_integrity']='passed'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not all(report['dependencies'].values()):raise SystemExit(1)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    sub.add_parser('doctor')
    p=sub.add_parser('prepare');p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('annotate');p.add_argument('operation',choices=['check','run','status','validate']);p.add_argument('--config',required=True);p.add_argument('--only',action='append');p.add_argument('--retry-failed',action='store_true');p.add_argument('--probe',action='store_true')
    p=sub.add_parser('preprocess');p.add_argument('--config',required=True);p.add_argument('--through',choices=['audit','vio','select','rectify','depth'],default='depth');p.add_argument('--resume',action='store_true')
    p=sub.add_parser('anchors-prepare');p.add_argument('--sources',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('anchors-run');p.add_argument('--root',type=Path,required=True)
    p=sub.add_parser('anchors-validate');p.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='doctor':doctor()
    elif args.action=='prepare':prepare(args.manifest,args.output)
    elif args.action=='anchors-prepare':prepare_anchors(args.sources,args.models,args.output)
    elif args.action=='annotate':
        argv=[args.operation,'--config',args.config]
        for eid in args.only or []:argv+=['--only',eid]
        if args.retry_failed:argv+=['--retry-failed']
        if args.probe:argv+=['--probe']
        call(ROOT/'components/annotation/pipeline_11_5.py',argv)
    elif args.action=='preprocess':
        argv=[sys.executable,'-m','daaam.preprocessing.runner','--config',args.config,'--through',args.through,'--ros-mode','none']
        if args.resume:argv+=['--resume']
        subprocess.run(argv,env=environment(),check=True)
    elif args.action=='anchors-run':
        call(ROOT/'components/anchors/scripts/run_noqwen_batches.py',['--root',args.root.resolve()])
        if read(args.root/'final_validation.json')['status']!='complete':raise SystemExit(1)
    elif args.action=='anchors-validate':call(ROOT/'components/anchors/scripts/validate_resumed_batch.py',['--root',args.root.resolve()])

if __name__=='__main__':main()
