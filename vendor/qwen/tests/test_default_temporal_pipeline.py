import json
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_scene_event_batch as entry
import run_scene_temporal_batch as pipeline


def test_default_entry_routes_to_temporal(tmp_path,monkeypatch):
    run=Mock();monkeypatch.setattr(pipeline,'run',run)
    monkeypatch.setattr(sys,'argv',['run_scene_event_batch.py','--output',str(tmp_path),'--inventory-only'])
    entry.main()
    run.assert_called_once()
    assert run.call_args.args[2] is True


def test_default_pipeline_refuses_legacy_directory(tmp_path):
    (tmp_path/'target.schema.json').write_text(json.dumps({'properties':{'schema':{'const':'legacy'}}}))
    with pytest.raises(ValueError,match='legacy'):
        pipeline.run(tmp_path,tmp_path)


def test_default_pipeline_emits_nested_tree_one_call_per_event(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    batch=root/'outputs/todo_astra_tf_qwen';eid='cp_0007_session_20261026_122149'
    source=batch/'json'/(eid+'.json')
    if not source.exists():pytest.skip('Real fixture unavailable')
    doc=json.loads(source.read_text())
    row=next(r for r in json.loads((batch/'inventory.json').read_text())['episodes'] if r['episode_id']==eid)
    trial=root/'outputs/temporal_v2_trial_122149_r2/events'
    monkeypatch.setattr(pipeline,'inventory',lambda todo,out:[row])
    def prepare(row,out):
        p=out/'episodes'/eid;p.mkdir(parents=True);return p
    monkeypatch.setattr(pipeline,'prepare_episode',prepare)
    monkeypatch.setattr(pipeline,'scene_job',lambda folder:doc['scene_memory'])
    calls=[]
    def annotate(event,folder,graph,target,correction=None):
        calls.append(event['ordinal'])
        return json.loads((trial/f"event_{event['ordinal']:02d}"/'result.json').read_text())
    monkeypatch.setattr(pipeline,'annotate_event',annotate)
    status=pipeline.run(tmp_path,tmp_path/'out')
    assert status['status']=='complete',status
    result=json.loads((tmp_path/'out/json'/source.name).read_text())
    assert isinstance(result['scene_memory']['spatial_structure']['nodes'],list)
    assert result['scene_memory']['spatial_structure']['schema']=='ego.scene.temporal_parent_tree.v1'
    assert len(calls)==row['event_count']
    # Resume must not trigger model calls or silently revert to a static result.
    assert pipeline.run(tmp_path,tmp_path/'out')['status']=='complete'
    assert len(calls)==row['event_count']
