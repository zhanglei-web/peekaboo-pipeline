from pathlib import Path
REPOSITORY = next(p for p in Path(__file__).resolve().parents if (p / 'repository.marker').is_file())
from pathlib import Path
from contextlib import contextmanager
from collections import Counter
import argparse
import importlib.metadata
import json
import os
import time
from PIL import Image
from . import SCHEMA, VERSION
from .common import read,sha,digest,atomic,locked,inside
from .contract import plan,augment

DEFAULTS={"scope":"scene_and_focus","device":"cuda:0","gpu_lock":os.environ.get("EGOMEMORY_GPU_LOCK",str(Path.home()/".cache/egomemory/gpu0.lock")),
    "min_free_gpu_mib":6000,"detection_threshold":.28,"nms_iou":.5,"prompts_per_pass":6,
    "max_instances_per_prompt":5,"fastsam_imgsz":1024,"fastsam_conf":.25,"fastsam_iou":.9,
    "max_masks":200,"mask_box_iou":.5,"mask_containment":.85,"mask_score_margin":.06,
    "min_mask_pixels":16,"identity_conflict_iou":.8}
PATHS=['input_json','prompt_map','output_dir','grounding_dino_path','fastsam_weights','gpu_lock']


def config(path):
    supplied=read(path)
    allowed=set(DEFAULTS)|set(PATHS)
    if set(supplied)-allowed:raise ValueError('Unknown config keys: '+repr(set(supplied)-allowed))
    cfg={**DEFAULTS,**supplied}
    for key in PATHS:
        cfg[key]=str(Path(cfg[key]).expanduser().resolve())
    for key in ['detection_threshold','nms_iou','fastsam_conf','fastsam_iou','mask_box_iou','mask_containment','mask_score_margin','identity_conflict_iou']:
        if not isinstance(cfg[key],(int,float)) or isinstance(cfg[key],bool) or not 0<cfg[key]<1:raise ValueError('Invalid threshold: '+key)
    for key in ['min_free_gpu_mib','prompts_per_pass','max_instances_per_prompt','fastsam_imgsz','max_masks','min_mask_pixels']:
        if not isinstance(cfg[key],int) or isinstance(cfg[key],bool) or cfg[key]<1:raise ValueError('Invalid integer: '+key)
    if cfg['prompts_per_pass']>12:raise ValueError('Limit caption size; use <=12 prompts per pass')
    return cfg


def safe_output(cfg, frames):
    out=Path(cfg['output_dir'])
    for p in [Path(cfg['input_json']),Path(cfg['prompt_map']),Path(cfg['fastsam_weights']),*[Path(f['image_path']) for f in frames]]:
        parent=p.parent.resolve()
        if out==parent or out in p.resolve().parents or parent in out.parents:
            raise ValueError('Output overlaps protected input directory: '+str(parent))
    model=Path(cfg['grounding_dino_path'])
    if out==model or out in model.parents or model in out.parents:raise ValueError('Output overlaps model directory')


def provenance(cfg,frames):
    model=Path(cfg['grounding_dino_path'])
    required=['config.json','model.safetensors','preprocessor_config.json','tokenizer_config.json']
    if any(not (model/f).is_file() for f in required):
        raise FileNotFoundError('Grounding DINO model directory incomplete; run scripts/download_model.py')
    if not (model/'tokenizer.json').exists() and not (model/'vocab.txt').exists():raise FileNotFoundError('Tokenizer absent')
    code={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','transformers','ultralytics','numpy','Pillow','scipy']}
    value={'schema':SCHEMA,'code_version':VERSION,'config':cfg,'input_sha256':sha(cfg['input_json']),
           'prompt_map_sha256':sha(cfg['prompt_map']),'code_sha256':code,'runtime_versions':versions,
           'grounding_dino_files':{p.name:sha(p) for p in sorted(model.iterdir()) if p.is_file() and p.suffix in ('.json','.txt','.safetensors')},
           'fastsam_sha256':sha(cfg['fastsam_weights']),'frames':[{k:f[k] for k in ['state_id','image_sha256','width','height']} for f in frames]}
    value['manifest_key']=digest(value)
    return value


@contextmanager
def report_failures(out):
    try:
        yield
    except Exception as exc:
        previous=read(out/'status.json') if (out/'status.json').exists() else {}
        atomic(out/'status.json',{**previous,'status':'failed','pid':os.getpid(),
               'error_type':type(exc).__name__,'error':str(exc)})
        raise


def run(cfg, shared_models=None, gpu_is_leased=False):
    from .models import Models,gpu_lease
    from .geometry import ground_frame
    from .artifacts import save_masks,overlay,make_review,validate
    if shared_models is not None and not gpu_is_leased:
        raise ValueError('Shared models require an externally held GPU lease')
    original=read(cfg['input_json']);frames=plan(original,read(cfg['prompt_map']),cfg['scope']);safe_output(cfg,frames)
    manifest=provenance(cfg,frames);out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with locked(out/'run.lock'):
        mp=out/'run_manifest.json'
        if mp.exists():
            if read(mp)!=manifest:raise ValueError('Existing run inputs/code/models/config differ; use a new output directory')
        else:
            if any(p.name!='run.lock' for p in out.iterdir()):raise ValueError('Output directory is not empty')
            atomic(mp,manifest)
        with report_failures(out):
            if (out/'validation.json').exists() and (out/'enriched.json').exists():
                result=validate(out);print(json.dumps(result,ensure_ascii=False));return
            results={};pending=[];timings=[]
            for i,frame in enumerate(frames):
                folder=out/'frames'/f'{i:03d}';rp=folder/'receipt.json'
                if rp.exists():
                    receipt=read(rp)
                    if receipt['manifest_key']!=manifest['manifest_key'] or receipt['state_id']!=frame['state_id']:raise ValueError('Wrong cached frame identity')
                    for p,h in receipt['files'].items():
                        if sha(inside(out,p))!=h:raise ValueError('Cached frame changed: '+p)
                    results[frame['state_id']]=read(folder/'anchors.json');timings.append(read(folder/'timing.json'))
                else:pending.append((i,frame,folder))
            started=time.perf_counter()
            atomic(out/'status.json',{'status':'running','pid':os.getpid(),'completed_frames':len(results),'total_frames':len(frames)})
            if pending:
                from contextlib import nullcontext
                with (nullcontext() if gpu_is_leased else gpu_lease(cfg)):
                    load_tick=time.perf_counter();models=shared_models if shared_models is not None else Models(cfg);load_seconds=time.perf_counter()-load_tick
                    for i,frame,folder in pending:
                        tick=time.perf_counter()
                        if sha(frame['image_path'])!=frame['image_sha256']:raise ValueError('Input image changed before inference')
                        folder.mkdir(parents=True,exist_ok=True)
                        with Image.open(frame['image_path']) as image:
                            image=image.convert('RGB');prompts=sorted({c['prompt'] for c in frame['candidates']})
                            detections,masks,timing=models.infer(image,prompts)
                            result=ground_frame(frame,detections,masks,cfg)
                            atomic(folder/'detections.json',detections);save_masks(folder/'masks.npz',masks)
                            result['mask_artifact']=str((folder/'masks.npz').relative_to(out))
                            result['mask_artifact_sha256']=sha(folder/'masks.npz')
                            result['proposal_limit_reached']=timing['mask_limit_reached']
                            atomic(folder/'anchors.json',result)
                            overlay(image,result,masks,folder/'overlay.jpg');overlay(image,result,masks,folder/'focus.jpg',True)
                        timing.update(total_frame_seconds=time.perf_counter()-tick,state_id=frame['state_id']);atomic(folder/'timing.json',timing);timings.append(timing)
                        if sha(frame['image_path'])!=frame['image_sha256']:raise ValueError('Input image changed during inference')
                        atomic(folder/'receipt.json',{'state_id':frame['state_id'],'manifest_key':manifest['manifest_key'],
                            'files':{str(p.relative_to(out)):sha(p) for p in sorted(folder.iterdir()) if p.is_file() and p.name!='receipt.json'}})
                        results[frame['state_id']]=result
                        atomic(out/'status.json',{'status':'running','pid':os.getpid(),'completed_frames':len(results),'total_frames':len(frames),'last_state_id':frame['state_id']})
                        print(json.dumps({'frame':i+1,'total':len(frames),'objects':len(result['objects']),'seconds':round(timing['total_frame_seconds'],2)},ensure_ascii=False),flush=True)
                    if shared_models is None:
                        del models
                        import torch
                        torch.cuda.empty_cache()
            else:load_seconds=0
            if sha(cfg['input_json'])!=manifest['input_sha256']:raise ValueError('Source JSON changed during run')
            meta={'schema':SCHEMA,'source_json':cfg['input_json'],'source_sha256':manifest['input_sha256'],
                  'manifest_key':manifest['manifest_key'],'source_schema':original['schema'],'semantic_human_reviewed':False,
                  'compatibility':'Derived augmented document; original schema consumers must use source JSON plus anchors sidecar, or explicitly support this extension.',
                  'scope':cfg['scope'],'physical_dimensions_available':False,'cross_frame_identity_verified':False}
            enriched=augment(original,results,meta)
            atomic(out/'enriched.json',enriched);atomic(out/'anchors.sidecar.json',{'metadata':meta,'frames':results})
            make_review(out,frames,results)
            atomic(out/'timing.json',{'model_load_seconds_this_invocation':load_seconds,'wall_seconds_this_invocation':time.perf_counter()-started,'frame_timings':timings,'total_inference_seconds':sum(t['detection_seconds']+t['segmentation_seconds'] for t in timings)})
            validation=validate(out);atomic(out/'validation.json',validation)
            atomic(out/'status.json',{'status':'complete','pid':os.getpid(),'completed_frames':len(frames),'total_frames':len(frames),'semantic_human_reviewed':False})
            print(json.dumps(validation,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description='Add per-frame anchors without changing original annotation values')
    subs=parser.add_subparsers(dest='action',required=True)
    for action in ['plan','run']:
        subs.add_parser(action).add_argument('--config',required=True,type=Path)
    subs.add_parser('validate').add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.action=='validate':
        from .artifacts import validate
        print(json.dumps(validate(args.output),ensure_ascii=False));return
    cfg=config(args.config)
    if args.action=='plan':
        frames=plan(read(cfg['input_json']),read(cfg['prompt_map']),cfg['scope']);safe_output(cfg,frames)
        print(json.dumps({'frames':len(frames),'candidate_observations':sum(len(f['candidates']) for f in frames),
                          'input_images_verified':True,'output_dir':cfg['output_dir']},ensure_ascii=False));return
    try:run(cfg)
    except Exception as exc:
        # Never mutate another run's status on a manifest mismatch or lock failure.
        print(json.dumps({'status':'failed','error_type':type(exc).__name__,'error':str(exc)},ensure_ascii=False),flush=True)
        raise


if __name__=='__main__':main()
