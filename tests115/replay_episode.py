"""Real saved-inference regression; explicitly NOT fresh model generation."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import json
import shutil
import tempfile
import jsonschema
from pipeline115.common import ROOT,atomic,read,sha
from pipeline115.engine import load_config
from pipeline115.worker import run

def main():
    cfg=load_config(ROOT/'pipeline115.example.json')
    row=read(cfg['annotation_inventory'])['episodes'][0];eid=row['episode_id']
    old=ROOT.parent/'search_episode_outputs/episodes'/eid
    direct=Path('/home/baai/zl/todo/11.5/validation/direct_equivalence')/eid
    assert (direct/'ready.json').exists(),'Run the direct equivalence test first'
    calls=[]
    class Replay:
        def __init__(self,profile,*args):pass
        def __call__(self,folder,prompt,images,schema,max_tokens=10000):
            relative=Path(folder).relative_to(work)
            saved=old/relative;descriptor=read(saved/'inputs.json')
            assert descriptor['prompt']==prompt,'Replay prompt differs'
            assert descriptor['schema']==schema,'Replay schema differs'
            assert descriptor['max_tokens']==max_tokens,'Replay budget differs'
            assert [sha(p) for p in images]==[i['sha256'] for i in descriptor['images']],'Replay image bytes differ'
            result=read(saved/'result.json');jsonschema.validate(result,schema)
            Path(folder).mkdir(parents=True,exist_ok=True)
            shutil.copyfile(saved/'response.json',Path(folder)/'response.json')
            atomic(Path(folder)/'result.json',result)
            calls.append(str(relative));return result
    with tempfile.TemporaryDirectory(prefix='pipeline115-replay-') as tmp:
        cfg['output_root']=tmp;work=Path(tmp)/'annotations/episodes'/eid
        work.mkdir(parents=True)
        shutil.copytree(old/'selection',work/'selection')
        # Remove old scene cache: the new scene adapter must call our strict
        # saved-input comparator instead of accepting an old opaque success.
        shutil.rmtree(work/'selection/qwen_scene_v2')
        atomic(Path(tmp)/'annotations/target.schema.json',read(ROOT/'contracts/final_annotation.schema.json'))
        with patch('pipeline115.worker.Client',Replay):
            for stage in ('scene','events','temporal','assemble'):run(stage,cfg,row)
        final=read(Path(tmp)/'annotations/json'/(eid+'.json'))
        baseline=read(ROOT.parent/'search_episode_outputs/json'/(eid+'.json'))
        for key in ('scene_memory','search_units','inspections','episode_outcome','derived_statistics','hidden','priors'):
            assert final[key]==baseline[key],key
        events=final['search_event_states']['events']
        assert events==baseline['search_event_states']['events'],'Event content changed'
        confirmed=read(Path(tmp)/'annotations/confirmed_json'/(eid+'.json'))
        assert confirmed['episode_outcome']['result']=='found'
        assert [e['result'] for e in confirmed['search_event_states']['events']]==['not_found']*(len(events)-1)+['found']
        assert final['episode_outcome']['result']=='unknown'
        assert confirmed['metadata']['no_revisits_confirmed'] is None
        report={'status':'passed','episode_id':eid,'events':len(events),'saved_requests_compared':len(calls),
                'scope':'Saved 11-4 responses replayed through new scene/event/temporal adapters and final assembly; no new model inference',
                'event_content_preserved':True,'spatial_temporal_content_preserved':True,
                'both_target_exports_generated':True,'unknown_and_scoped_human_facts_verified':True}
        atomic(Path('/home/baai/zl/todo/11.5/validation/replay_report.json'),report);print(json.dumps(report))

if __name__=='__main__':main()
