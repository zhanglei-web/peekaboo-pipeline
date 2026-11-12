#!/usr/bin/env python3
"""Terminal entrypoint for the separately versioned direct-frame pipeline."""
from pathlib import Path
import argparse
import json
import os
import sys
from pipeline115.common import ROOT, alive, lock, read, verify_dependencies
from pipeline115.engine import Engine, load_config
from pipeline115.dispatch import endpoint_keys, episode_mode

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['check','run','status','validate'])
    p.add_argument('--config',type=Path,default=ROOT/'pipeline115.example.json')
    p.add_argument('--only',action='append',help='exact episode ID; repeat to select more')
    p.add_argument('--reuse-preprocessed-state',type=Path,help='read-only reuse after full stage verification')
    p.add_argument('--retry-failed',action='store_true',help='retry repaired stage; retains completed inference')
    p.add_argument('--probe',action='store_true',help='query remote model identity; does not start local server')
    args=p.parse_args(); cfg=load_config(args.config); root=Path(cfg['output_root'])
    if args.action=='status':
        state=read(root/'state.json') if (root/'state.json').exists() else {'status':'not_started'}
        print(json.dumps({'status':state.get('status'),'live':alive(state),'counts':state.get('counts',{}),'resource_error':state.get('resource_error')},ensure_ascii=False)); return
    verify_dependencies()
    if args.action=='check':
        result={'status':'passed','version':'11-5','routes':cfg['routes'],'output_root':str(root)}
        envs={p['api_key_env'] for k,p in cfg['endpoints'].items() if k in endpoint_keys(cfg) and p.get('api_key_env')}
        envs.add(read(cfg['astra_config'])['model']['api_key_env'])
        result['credentials_available']={key:bool(os.environ.get(key)) for key in sorted(envs)}
        if args.probe:
            from pipeline115.client import Client
            result['remote_probes']={k:Client(v).probe(require_context=episode_mode(cfg)) for k,v in cfg['endpoints'].items()
                                     if k in endpoint_keys(cfg) and v['placement']=='remote'}
        print(json.dumps(result,ensure_ascii=False)); return
    if args.action=='validate':
        from pipeline115.validate import validate_run
        result=validate_run(root); print(json.dumps(result,ensure_ascii=False));
        if result['status']!='passed': raise SystemExit(1)
        return
    envs={p['api_key_env'] for k,p in cfg['endpoints'].items() if k in endpoint_keys(cfg) and p.get('api_key_env')}
    envs.add(read(cfg['astra_config'])['model']['api_key_env'])
    missing=[key for key in sorted(envs) if not os.environ.get(key)]
    if missing:raise ValueError('Missing credential environment; no jobs started: '+', '.join(missing))
    if episode_mode(cfg):
        from pipeline115.client import Client
        for key in endpoint_keys(cfg):
            if cfg['endpoints'][key]['placement']=='remote':
                Client(cfg['endpoints'][key]).probe(require_context=True)
    with lock(root/'pipeline.lock'):
        engine=Engine(cfg,args.only,args.reuse_preprocessed_state,args.retry_failed)
        result=engine.run(); print(json.dumps({'status':result['status'],'counts':result['counts'],'output_root':str(root)},ensure_ascii=False))
        if result['status']!='complete': raise SystemExit(1)
        from pipeline115.validate import validate_run
        validation=validate_run(root); print(json.dumps(validation,ensure_ascii=False))
        engine.state['final_validation']=validation
        if validation['status']!='passed': engine.state['status']='needs_repair'
        engine.save()
        if validation['status']!='passed': raise SystemExit(1)

if __name__=='__main__': main()
