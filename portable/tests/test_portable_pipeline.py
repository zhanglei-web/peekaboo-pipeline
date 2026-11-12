from pathlib import Path
from contextlib import nullcontext
import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
import pipeline
from make_demo import generate

ROOT=pipeline.ROOT

def load_script(path):
    spec=importlib.util.spec_from_file_location('portable_'+path.stem,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_synthetic_assembly_to_compact_and_tamper_detection(tmp_path,monkeypatch):
    demo=tmp_path/'input';doc=generate(demo)
    assert doc['history'] is None
    assert doc['scene_memory']['spatial_structure']['nodes'][0]['children'][0]['parent_relations']
    models=tmp_path/'models';models.mkdir()
    for name in ['config.json','model.safetensors','preprocessor_config.json','tokenizer_config.json','tokenizer.json','fastsam.pt']:
        (models/name).write_text('mock-only')
    cfg=tmp_path/'model_config.json';pipeline.write(cfg,{'grounding_dino_path':str(models),'fastsam_weights':str(models/'fastsam.pt'),'gpu_lock':str(tmp_path/'gpu.lock')})
    out=tmp_path/'anchors';pipeline.prepare_anchors(demo/'anchor_sources.json',cfg,out)
    runner=load_script(ROOT/'components/anchors/scripts/run_noqwen_batches.py')
    class FakeModels:
        def __init__(self,cfg):pass
        def infer(self,image,prompts):
            masks=np.zeros((1,64,96),dtype=bool);masks[0,20:42,20:40]=True
            return {p:[{'box':[20,20,40,42],'score':.9}] for p in prompts},masks,{'detection_seconds':0.,'segmentation_seconds':0.,'mask_limit_reached':False}
    monkeypatch.setattr(runner,'Models',FakeModels);monkeypatch.setattr(runner,'gpu_lease',lambda cfg:nullcontext())
    monkeypatch.setattr('frame_anchors.cli.importlib.metadata.version',lambda name:'mock-test-only')
    runner.run_batch(out)
    assert pipeline.read(out/'final_validation.json')['status']=='complete'
    validator=load_script(ROOT/'components/anchors/scripts/validate_resumed_batch.py')
    assert validator.audit(out)['status']=='passed'
    result=pipeline.read(next((out/'json').rglob('*.json')))
    for event in result['search_event_states']['events']:
        for state in event['states']:
            assert state.pop('object_anchors')
    assert result==doc
    (demo/'evidence/start.png').write_bytes(b'changed')
    assert validator.audit(out)['status']=='incomplete'

def test_prepare_validates_original_triplets_and_rejects_mismatch(tmp_path):
    eid='episode_001';raw=tmp_path/'capture';raw.mkdir();labels=tmp_path/'labels';labels.mkdir()
    pipeline.write(raw/'meta.json',{'fps':30,'topics':{'left':{'written_frames':10},'right':{'written_frames':10}}})
    for eye in ['left','right']:
        (labels/eye).mkdir()
        for name,frame in [('开始',1),('进行中',4),('结束',7)]:
            Image.new('RGB',(96,64),'blue').save(labels/eye/f'{eid}_第1次_{name}_f{frame}.jpg')
    md=labels/'event.md';md.write_text('帧率 30 帧/秒\n要找的物体：blue cup\n| 第1次 | 1 | 4 | 7 |\n')
    m=pipeline.read(ROOT/'configs/pipeline.example.json');m['episodes']=[{'episode_id':eid,'capture':str(raw),'markdown':str(md),'left_triplets':str(labels/'left'),'right_triplets':str(labels/'right'),'target':'blue cup','fps':30}]
    venv_python=tmp_path/'venv/bin/python';venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(sys.executable)
    m['preprocessing']['openvins_binary']=str(tmp_path/'openvins');m['preprocessing']['depth'].update(repo=str(tmp_path/'stereo'),checkpoint=str(tmp_path/'weights.pth'),python=str(venv_python))
    manifest=tmp_path/'manifest.json';pipeline.write(manifest,m)
    pipeline.prepare(manifest,tmp_path/'prepared')
    row=pipeline.read(tmp_path/'prepared/annotation_inventory.json')['episodes'][0]
    assert [s['original_video_frame_index'] for s in row['event_index']['events'][0]['states']]==[1,4,7]
    assert pipeline.read(tmp_path/'prepared/pipeline.json')['facts'] is None
    assert pipeline.read(tmp_path/'prepared/preprocessing_configs/episode_001.json')['depth']['python']==str(venv_python)
    md.write_text(md.read_text().replace('| 1 | 4 | 7 |','| 1 | 5 | 7 |'))
    with pytest.raises(ValueError,match='mismatch'):pipeline.prepare(manifest,tmp_path/'bad')

def test_duplicate_group_and_source_hash_rejected(tmp_path):
    demo=tmp_path/'demo';generate(demo)
    cfg=tmp_path/'models.json';pipeline.write(cfg,{'grounding_dino_path':'dino','fastsam_weights':'sam.pt','gpu_lock':'gpu.lock'})
    sources=pipeline.read(demo/'anchor_sources.json');sources['groups']*=2
    (demo/'anchor_sources.json').write_text(json.dumps(sources))
    with pytest.raises(ValueError,match='Duplicate'):pipeline.prepare_anchors(demo/'anchor_sources.json',cfg,tmp_path/'out')
    sources['groups']=sources['groups'][:1];(demo/'anchor_sources.json').write_text(json.dumps(sources))
    (demo/'evidence/start.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):pipeline.prepare_anchors(demo/'anchor_sources.json',cfg,tmp_path/'out2')

def test_relocated_checkout_has_no_source_workspace_dependency(tmp_path):
    target=tmp_path/'moved checkout'
    shutil.copytree(ROOT,target,ignore=shutil.ignore_patterns('.git','runs','__pycache__','.pytest_cache'))
    for args in [['scripts/check_repository.py'],['scripts/pipeline.py','doctor'],['scripts/make_demo.py','--output',str(tmp_path/'relocated_demo')]]:
        result=subprocess.run([sys.executable,*args],cwd=target,text=True,capture_output=True)
        assert result.returncode==0,result.stdout+result.stderr

def test_gpu_lock_and_process_identity(tmp_path):
    from pipeline115.common import lock,identity,alive
    with lock(tmp_path/'lock'):
        with pytest.raises(BlockingIOError):
            with lock(tmp_path/'lock'):pass
    record=identity();assert alive(record)
    assert 'proc_start_ticks' in record
    bad={**record,'proc_start_ticks':record['proc_start_ticks']+1};assert not alive(bad)

def test_preprocessing_runtime_imports_are_packaged():
    """The real runner and its GPU worker must not import omitted DAAAM APIs."""
    code = """
import daaam.preprocessing.runner
import importlib.util
from daaam.preprocessing.common import legacy
for name in ['prepare_ego_stereo_dataset', 'run_ego_openvins',
             'select_ego_keyframes']:
    legacy(name)
if importlib.util.find_spec('torch'):
    legacy('run_fast_foundation_stereo_depth')
import daaam.depth.confidence
import daaam.realtime.gpu
import daaam.memory.legacy_gate
"""
    result=subprocess.run([sys.executable,'-c',code],env=pipeline.environment(),
                          text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr

def test_depth_worker_exit_preserves_redirected_error(tmp_path):
    from daaam.preprocessing.depth import _read_worker_message
    log_path=tmp_path/'worker.log'
    with log_path.open('w') as log:
        process=subprocess.Popen([sys.executable,'-c',
            "import sys; sys.stderr.write('backend initialization failed\\n'); sys.exit(2)"],
            stdout=subprocess.PIPE,stderr=log,text=True)
        process.wait(timeout=10)
        with pytest.raises(RuntimeError,match='backend initialization failed'):
            _read_worker_message(process,1,log_path)

def test_assembly_exports_use_result_names_not_template_names(tmp_path,monkeypatch):
    import time
    import annotate_batch as annotation
    from temporal_spatial_v2 import initialize_temporal_graph
    demo=tmp_path/'demo';doc=generate(demo)
    source=pipeline.read(demo/'intermediate/source.json')
    temporal=pipeline.read(demo/'intermediate/temporal.json')
    eid=doc['metadata']['episode_id'];md=demo/'evidence/event.md'
    row={'episode_id':eid,'target':'blue cup','target_markdown':str(md),
         'target_markdown_sha256':pipeline.sha(md),'event_count':1,
         'source_annotation_metadata':doc['metadata']['source_annotation_metadata'],
         'event_index':{'events':source['search_event_states']['events']}}
    out=tmp_path/'annotations';folder=out/'episodes'/eid;folder.mkdir(parents=True)
    pipeline.write(folder/'temporal.json',temporal)
    pipeline.write(out/'target.schema.json',annotation.build_contract())
    monkeypatch.setattr(annotation,'OUT',out)
    annotation.finish(row,folder,source,[{'potential_search_regions':[]}],temporal,
                      initialize_temporal_graph(source['scene_memory']),{},time.perf_counter())
    exports=out/'exports'/eid
    assert {p.name for p in exports.iterdir()}=={eid+'.json','experiment_records.json'}
    annotation.assert_target_fields(pipeline.read(ROOT/'components/additive/experiment_export.shape.json'),
                                    pipeline.read(exports/'experiment_records.json'),'experiment')
