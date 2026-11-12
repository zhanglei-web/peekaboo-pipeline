"""Run prepared source-bound maps through Grounding DINO + FastSAM only.

There is no text model, HTTP client, selector server, or fallback extractor in
this entry point. Missing candidate maps remain pending. Previously completed
records may be retained with their historical selection provenance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from frame_anchors.common import atomic,digest,locked,read,sha
from frame_anchors.cli import DEFAULTS,run
from frame_anchors.contract import plan
from frame_anchors.models import Models,gpu_lease


def identity(pid=None):
    pid=os.getpid() if pid is None else pid
    proc=Path('/proc')/str(pid)
    stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':int(stat[19]),
            'command':(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()}


def model_config(root):
    supplied=read(Path(root)/'model_config.json')
    allowed=set(DEFAULTS)|{'grounding_dino_path','fastsam_weights'}
    if set(supplied)-allowed:raise ValueError('Unknown model configuration fields')
    for key in ('grounding_dino_path','fastsam_weights','gpu_lock'):
        if key not in supplied or not Path(supplied[key]).is_absolute():
            raise ValueError('Model paths and gpu_lock must be absolute: '+key)
    return {**DEFAULTS,**supplied,'scope':'event_description','output_dir':str(Path(root)/'runtime')}


def compact_check(source,output):
    doc=read(output);original=read(source)
    allowed={'name','ref','anchor_xy_normalized','size_wh_normalized','status'}
    frames=0
    for event in doc['search_event_states']['events']:
        for state in event['states']:
            entries=state.pop('object_anchors')
            if not isinstance(entries,list):raise ValueError('Compact anchors must be an array')
            for item in entries:
                if set(item)!=allowed:raise ValueError('Unexpected compact anchor fields')
                if item['status']=='not_detected':
                    if item['anchor_xy_normalized'] is not None or item['size_wh_normalized'] is not None:
                        raise ValueError('Undetected object has coordinates')
                else:
                    if item['status'] not in {'mask_unreviewed','mask_needs_review','bbox_center_needs_review'}:
                        raise ValueError('Unknown status')
                    for key in ['anchor_xy_normalized','size_wh_normalized']:
                        if len(item[key])!=2 or not all(isinstance(v,(int,float)) and 0<=v<=1 for v in item[key]):
                            raise ValueError('Invalid normalized values')
            frames+=1
    if doc!=original:raise ValueError('Original JSON values changed')
    return frames


def load_exporter():
    spec=importlib.util.spec_from_file_location('compact_export_noqwen',PROJECT/'scripts/export_compact.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.export


def run_batch(root,wait_for_maps=False,max_records=None):
    root=Path(root).resolve();inventory=read(root/'input_inventory.json')
    if inventory.get('errors'):raise ValueError('Input inventory contains errors')
    records=inventory['records'];keys={r['group']+'/'+r['record_id'] for r in records}
    if len(keys)!=len(records):raise ValueError('Duplicate inventory record keys')
    cfg=model_config(root);exporter=load_exporter()
    binding={'inventory':sha(root/'input_inventory.json'),'runner':sha(__file__),
        'code':{p.name:sha(p) for p in sorted((PROJECT/'frame_anchors').glob('*.py'))},
        'exporter':sha(PROJECT/'scripts/export_compact.py'),
        'models':{k:cfg[k] for k in ['grounding_dino_path','fastsam_weights']},
        'defaults':DEFAULTS,'text_model':None,'method':'prepared source-bound candidates + Grounding DINO + FastSAM'}
    binding_id=digest(binding)
    state_path=root/'batch_state.json'
    with locked(root/'batch.lock'):
        manifest_path=root/'batch_manifest.json'
        if manifest_path.exists() and read(manifest_path)['identity']!=binding_id:
            raise ValueError('Bound inputs/code changed; use a new batch revision')
        if not manifest_path.exists():atomic(manifest_path,{'identity':binding_id,'binding':binding,'created_at':time.time()})
        old=read(state_path) if state_path.exists() else {}
        state={'status':'running','total':len(records),'completed':old.get('completed',{}),
            'failures':{},'stage':'preflight','process':identity(),'started_at':time.time(),
            'text_model':None,'semantic_human_reviewed':False}
        def save(stage,**values):
            state.update(stage=stage,updated_at=time.time(),**values)
            state['completed_count']=len(state['completed'])
            state['pending_count']=len(records)-len(state['completed'])
            atomic(state_path,state)
        save('preflight')
        processed=0
        try:
            while True:
                pending=[r for r in records if r['group']+'/'+r['record_id'] not in state['completed']]
                ready=[]
                for row in pending:
                    key=row['group']+'/'+row['record_id']
                    mapping_path=root/'selections'/row['group']/(row['record_id']+'.json')
                    if not mapping_path.exists() or key in state['failures']:continue
                    try:
                        mapping=read(mapping_path)
                        policy=mapping.get('policy',{})
                        if 'qwen' in policy.get('selection_review_status','').lower():
                            raise ValueError('Qwen-derived map is not permitted for new inference: '+key)
                        if not policy.get('ready_for_grounding',False):continue
                        if sha(row['source_json'])!=row['source_sha256']:raise ValueError('Source changed: '+key)
                        frames=plan(read(row['source_json']),mapping,'event_description')
                        if any(not f['candidates'] for f in frames):raise ValueError('Unresolved empty event candidates: '+key)
                        ready.append((row,mapping_path,sha(mapping_path)))
                    except Exception as exc:
                        state['failures'][key]={'stage':'preflight_conflict','type':type(exc).__name__,'error':str(exc)}
                save('grounding_frames' if ready else 'waiting_for_candidate_maps',ready_records=len(ready),current_record=None)
                if ready:
                    with gpu_lease(cfg):
                        models=Models(cfg)
                        for row,mapping_path,mapping_hash in ready:
                            key=row['group']+'/'+row['record_id'];save('grounding_frames',current_record=key)
                            try:
                                if sha(mapping_path)!=mapping_hash:raise ValueError('Candidate map changed before run')
                                detailed=root/'detailed'/row['group']/row['record_id']
                                run_cfg={**cfg,'input_json':row['source_json'],'prompt_map':str(mapping_path),'output_dir':str(detailed)}
                                run(run_cfg,shared_models=models,gpu_is_leased=True)
                                compact=root/'compact_records'/row['group']/row['record_id']
                                if not compact.exists():report=exporter(detailed,compact)
                                else:
                                    report=read(compact/'validation.json')
                                    if report['status']!='passed' or report['source_sha256']!=row['source_sha256']:
                                        raise ValueError('Existing compact report mismatch')
                                    if sha(report['compact_json'])!=report['compact_sha256']:raise ValueError('Existing compact changed')
                                if sha(mapping_path)!=mapping_hash:raise ValueError('Candidate map changed during run')
                                compact_check(row['source_json'],report['compact_json'])
                                dest=root/'json'/row['group']/(row['record_id']+'.json')
                                dest.parent.mkdir(parents=True,exist_ok=True)
                                content=Path(report['compact_json']).read_bytes()
                                if dest.exists() and dest.read_bytes()!=content:raise ValueError('Conflicting published result')
                                if not dest.exists():
                                    temp=dest.with_suffix('.json.tmp');temp.write_bytes(content);temp.replace(dest)
                                state['completed'][key]={'json':str(dest),'sha256':sha(dest),'events':row['event_count'],
                                    'frames':row['frame_count'],'mapping_sha256':mapping_hash,'completed_at':time.time(),
                                    'selection_review_status':read(mapping_path)['policy']['selection_review_status']}
                                processed+=1
                                print(json.dumps({'record':key,'completed':len(state['completed']),'total':len(records)},ensure_ascii=False),flush=True)
                            except Exception as exc:
                                state['failures'][key]={'type':type(exc).__name__,'error':str(exc)}
                                print(json.dumps({'record':key,'error':str(exc)},ensure_ascii=False),flush=True)
                            save('grounding_frames',current_record=key)
                            if max_records is not None and processed>=max_records:break
                        del models
                if len(state['completed'])==len(records) or (max_records is not None and processed>=max_records):break
                if not wait_for_maps or state['failures']:break
                time.sleep(10)
            save('final_validation',current_record=None)
            issues={}
            for row in records:
                key=row['group']+'/'+row['record_id']
                if key not in state['completed']:continue
                try:
                    output=state['completed'][key]
                    if sha(row['source_json'])!=row['source_sha256'] or sha(output['json'])!=output['sha256']:
                        raise ValueError('Final source/output hash mismatch')
                    frames=compact_check(row['source_json'],output['json'])
                    if frames!=row['frame_count']:raise ValueError('Frame count mismatch')
                except Exception as exc:issues[key]=str(exc)
            state['status']='complete' if len(state['completed'])==len(records) and not state['failures'] and not issues else 'incomplete'
            save('finished',validation_issues=issues)
            atomic(root/'final_validation.json',{'status':state['status'],'expected':len(records),'completed':len(state['completed']),
                'failures':state['failures'],'validation_issues':issues,'semantic_human_reviewed':False,'text_model':None,
                'unprocessed':[r['group']+'/'+r['record_id'] for r in records if r['group']+'/'+r['record_id'] not in state['completed']],
                'json_directory':str(root/'json')})
        except BaseException as exc:
            state['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
            save(state['stage'],fatal_error=str(exc));raise


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--wait-for-maps',action='store_true');ap.add_argument('--max-records',type=int)
    args=ap.parse_args()
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    run_batch(args.root,args.wait_for_maps,args.max_records)


if __name__=='__main__':main()
