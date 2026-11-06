from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import tempfile
import threading
import time
import unittest
from pipeline115.common import ROOT,atomic,lock
from pipeline115.engine import Engine,STEPS,load_config
from pipeline115.resources import GPU,ExternalResourceBusy,check_bind

class Tests(unittest.TestCase):
    def make_engine(self,folder):
        cfg=load_config(ROOT/'pipeline115.example.json');cfg['output_root']=str(folder)
        cfg['limits'].update(queue_episodes=2,depth_group_size=2,local_residency_budget_s=.03)
        from pipeline115.common import read
        ids=[r['episode_id'] for r in read(cfg['annotation_inventory'])['episodes'][:2]]
        return Engine(cfg,only=ids)
    def test_os_exclusive_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with lock(Path(tmp)/'lock'):
                with self.assertRaises(BlockingIOError):
                    with lock(Path(tmp)/'lock'):pass
    def test_gpu_foreign_process_never_signalled(self):
        with tempfile.TemporaryDirectory() as tmp:
            gpu=GPU({'lock_path':str(Path(tmp)/'gpu'),'device':0},Path(tmp))
            with patch('pipeline115.resources.gpu_snapshot',return_value=({0:24000},[12345])):
                with self.assertRaises(ExternalResourceBusy):
                    with gpu.lease('depth',10000):self.fail('Foreign GPU lease accepted')
    def test_state_machine_overlap_and_exclusive_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Use real inventory IDs without loading datasets or calling models.
            cfg=load_config(ROOT/'pipeline115.example.json');cfg['output_root']=tmp
            from pipeline115.common import read
            ids=[r['episode_id'] for r in read(cfg['annotation_inventory'])['episodes'][:2]]
            cfg['limits'].update(queue_episodes=2,depth_group_size=2,local_residency_budget_s=.03)
            engine=Engine(cfg,only=ids);events=[];gpu_active=[];guard=threading.Lock()
            @contextmanager
            def resource(kind):
                with guard:
                    self.assertFalse(gpu_active);gpu_active.append(kind);events.append(('acquire',kind,time.monotonic()))
                try:yield
                finally:
                    with guard:events.append(('release',kind,time.monotonic()));gpu_active.pop()
            class FakeGPU:
                def lease(self,*args):return resource('depth')
                def local_server(self,*args):return resource('local')
            engine.gpu=FakeGPU()
            def execute(job,lane=0):
                eid,stage=job;events.append(('start',eid,stage,time.monotonic()));time.sleep(.02)
                if stage=='cpu':atomic(engine.root/'prepared'/(eid+'.json'),{'read_only_reuse':False})
                with engine.mutex:
                    s=engine.state['episodes'][eid];index=STEPS.index(stage)+1
                    s.update(status='complete' if index==len(STEPS) else 'queued',next=STEPS[index] if index<len(STEPS) else 'complete');engine.save()
                events.append(('end',eid,stage,time.monotonic()))
            with patch.object(engine,'execute',side_effect=execute),patch.object(engine,'pressure',return_value=None):
                result=engine.run()
            self.assertEqual(result['status'],'complete')
            self.assertEqual(sum(x[0]=='start' for x in events),len(STEPS)*2)
            for eid in ids:
                self.assertEqual([x[2] for x in events if x[0]=='start' and x[1]==eid],list(STEPS))
            self.assertTrue(any(x[:2]==('acquire','local') for x in events))
            self.assertTrue(any(x[:2]==('acquire','depth') for x in events))
            self.assertFalse(gpu_active)
    def test_historical_output_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=load_config(ROOT/'pipeline115.example.json');cfg['output_root']=str(ROOT.parent/'old_output')
            atomic(Path(tmp)/'cfg.json',cfg)
            with self.assertRaisesRegex(ValueError,'historical'):load_config(Path(tmp)/'cfg.json')
    def test_resource_block_requires_explicit_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp)
            engine.state.update(status='waiting_external_resource',resource_error='GPU occupied')
            engine.save()
            resumed=self.make_engine(tmp)
            with patch.object(resumed,'gpu_loop') as gpu:
                self.assertEqual(resumed.run()['resource_error'],'GPU occupied')
                gpu.assert_not_called()
            retried=Engine(engine.cfg,only=list(engine.rows),retry_failed=True)
            self.assertNotIn('resource_error',retried.state)
    def test_operator_stop_drains_and_remains_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=self.make_engine(tmp)
            def execute(job,lane=0):
                (engine.root/'stop_requested').touch();engine.stop.set()
                with engine.mutex:
                    engine.state['episodes'][job[0]].update(status='queued',next='depth')
                    engine.save()
            with patch.object(engine,'execute',side_effect=execute),patch.object(engine,'pressure',return_value=None):
                result=engine.run()
            self.assertEqual(result['status'],'stopped')
            self.assertIsNone(engine.claim('depth'))
            self.assertTrue(all(s['status']=='queued' for s in result['episodes'].values()))
    def test_port_preflight_rejects_listener_allows_time_wait(self):
        import socket
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            listener.bind(('127.0.0.1',0));listener.listen()
            host,port=listener.getsockname()
            with self.assertRaises(OSError):check_bind(host,port)
            with socket.create_connection((host,port)) as client:
                connection,_=listener.accept()
                connection.close()
                self.assertEqual(client.recv(1),b'')
        check_bind(host,port)

if __name__=='__main__':unittest.main()
