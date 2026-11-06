from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import tempfile
import threading
import time
import unittest

from pipeline115.common import ROOT,atomic,read
from pipeline115 import dispatch
from pipeline115.engine import Engine,STEPS,load_config


class Tests(unittest.TestCase):
    def make_engine(self,folder,count=4):
        cfg=load_config(ROOT/'pipeline115.example.json');cfg['output_root']=str(folder)
        cfg['dispatch']={'mode':'episode','endpoints':['local','lan']}
        cfg['endpoints']['lan']['context_tokens']=16384
        cfg['limits'].update(queue_episodes=8,depth_group_size=2,local_residency_budget_s=.03)
        ids=[r['episode_id'] for r in read(cfg['annotation_inventory'])['episodes'][:count]]
        return Engine(cfg,only=ids)

    def ready(self,engine):
        for state in engine.state['episodes'].values():state.update(next='scene',status='queued')

    def test_whole_episode_context_budget_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config(ROOT/'pipeline115.example.json')
            cfg['dispatch']={'mode':'episode','endpoints':['local','lan']}
            path=Path(tmp)/'config.json';atomic(path,cfg)
            with self.assertRaisesRegex(ValueError,'unchanged scene budget'):load_config(path)
            cfg['endpoints']['lan']['context_tokens']=16384;atomic(path,cfg)
            self.assertTrue(dispatch.episode_mode(load_config(path)))

    def test_concurrent_claims_and_sticky_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp);self.ready(engine);jobs={};barrier=threading.Barrier(2)
            def claim(kind):barrier.wait();jobs[kind]=engine.claim(kind)
            threads=[threading.Thread(target=claim,args=(kind,)) for kind in ('local','remote:lan')]
            for t in threads:t.start()
            for t in threads:t.join()
            self.assertNotEqual(jobs['local'][0],jobs['remote:lan'][0])
            local=jobs['local'][0];remote=jobs['remote:lan'][0]
            for eid,key in [(local,'local'),(remote,'lan')]:
                self.assertEqual(read(dispatch.config_path(engine.root,eid))['routes'],
                                 {s:key for s in dispatch.MODEL_STAGES})
            engine.state['episodes'][local].update(status='queued',next='events')
            engine.state['episodes'][remote].update(status='queued',next='events')
            self.assertEqual(engine.claim('remote:lan'),(remote,'events'))
            self.assertEqual(engine.claim('local'),(local,'events'))
            self.assertIsNone(engine.claim('remote:lan'))
            self.assertIsNone(engine.claim('local'))
            with self.assertRaisesRegex(ValueError,'transfer'):dispatch.assign(engine.root,engine.cfg,local,'lan')

    def test_assignment_recovers_after_publication_before_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp);eid=next(iter(engine.rows))
            dispatch.assign(engine.root,engine.cfg,eid,'lan')
            engine.state['episodes'][eid].update(status='queued',next='scene');engine.save()
            resumed=Engine(engine.cfg,only=list(engine.rows))
            self.assertEqual(resumed.state['episodes'][eid]['owner'],'lan')
            self.assertIsNone(resumed.claim('local'))
            self.assertEqual(resumed.claim('remote:lan'),(eid,'scene'))

    def test_orphan_bound_config_recovers_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp);eid=next(iter(engine.rows))
            atomic(dispatch.config_path(engine.root,eid),dispatch.owned_config(engine.cfg,'lan'))
            owner=dispatch.recover_assignment(engine.root,engine.cfg,eid)
            self.assertEqual(owner['endpoint'],'lan')
            cfg=read(dispatch.config_path(engine.root,eid));cfg['routes']['events']='local'
            atomic(dispatch.config_path(engine.root,eid),cfg)
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                dispatch.read_assignment(engine.root,engine.cfg,eid)

    def test_partial_legacy_work_not_silently_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp);eid=next(iter(engine.rows))
            engine.state['episodes'][eid].update(status='queued',next='temporal');engine.save()
            with self.assertRaisesRegex(ValueError,'partial model work'):Engine(engine.cfg,only=list(engine.rows))

    def test_selector_migration_preserves_owned_configs_but_rejects_other_changes(self):
        from pipeline115.common import sha
        from copy import deepcopy
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp);eid=next(iter(engine.rows))
            dispatch.assign(engine.root,engine.cfg,eid,'lan')
            cfg=deepcopy(engine.cfg);cfg['astra_config']='new_selector.json'
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                dispatch.read_assignment(engine.root,cfg,eid)
            registry=engine.root/'legacy_episode_configs.json'
            atomic(registry,{'episodes':{eid:{'assignment_sha256':sha(dispatch.assignment_path(engine.root,eid)),
                'config_sha256':sha(dispatch.config_path(engine.root,eid))}}})
            manifest=read(engine.root/'run_manifest.json');manifest['legacy_episode_configs_sha256']=sha(registry)
            atomic(engine.root/'run_manifest.json',manifest)
            self.assertEqual(dispatch.read_assignment(engine.root,cfg,eid)['endpoint'],'lan')
            cfg['limits']['queue_episodes']+=1
            with self.assertRaisesRegex(ValueError,'configuration changed'):
                dispatch.read_assignment(engine.root,cfg,eid)

    def test_pooled_episodes_overlap_without_cross_endpoint_stages(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp);events=[];guard=threading.Lock();gpu_active=[]
            @contextmanager
            def resource(kind):
                with guard:
                    self.assertFalse(gpu_active);gpu_active.append(kind)
                try:yield
                finally:
                    with guard:gpu_active.pop()
            class FakeGPU:
                def lease(self,*args):return resource('depth')
                def local_server(self,*args):return resource('local')
            engine.gpu=FakeGPU()
            def execute(job,lane=0):
                eid,stage=job;start=time.monotonic();owner=engine.state['episodes'][eid].get('owner')
                time.sleep(.035 if stage in dispatch.MODEL_STAGES else .008)
                if stage=='cpu':atomic(engine.root/'prepared'/(eid+'.json'),{'read_only_reuse':False})
                with engine.mutex:
                    index=STEPS.index(stage)+1
                    engine.state['episodes'][eid].update(status='complete' if index==len(STEPS) else 'queued',
                        next='complete' if index==len(STEPS) else STEPS[index]);engine.save()
                with guard:events.append((eid,stage,owner,start,time.monotonic()))
            with patch.object(engine,'execute',side_effect=execute),patch.object(engine,'pressure',return_value=None):
                result=engine.run()
            self.assertEqual(result['status'],'complete')
            for eid in engine.rows:
                rows=sorted([r for r in events if r[0]==eid],key=lambda r:r[3])
                self.assertEqual([r[1] for r in rows],list(STEPS))
                owners={r[2] for r in rows if r[1] in dispatch.MODEL_STAGES}
                self.assertEqual(len(owners),1);self.assertNotIn(None,owners)
            local=[r for r in events if r[1] in dispatch.MODEL_STAGES and r[2]=='local']
            remote=[r for r in events if r[1] in dispatch.MODEL_STAGES and r[2]=='lan']
            self.assertTrue(local and remote)
            self.assertTrue(any(max(a[3],b[3])<min(a[4],b[4]) for a in local for b in remote))
            self.assertFalse(gpu_active)


if __name__=='__main__':unittest.main()
