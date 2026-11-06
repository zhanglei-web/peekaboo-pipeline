"""Persisted sequential batch runner: strict current-batch gate, scoped skip policy."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time
from .common import ROOT,read,atomic,alive,identity,lock,now,sha,bind


def predecessor_complete(state,root,expected):
    episodes=state.get('episodes',{})
    validation=Path(root)/'final_validation.json'
    return (len(episodes)==expected and all(e['status']=='complete' for e in episodes.values())
            and validation.exists() and read(validation).get('status')=='passed'
            and read(validation).get('validated')==expected)


def summary(root,enrollment=None):
    root=Path(root);s=read(root/'state.json');complete=[eid for eid,e in s['episodes'].items() if e['status']=='complete']
    failures=[{'episode_id':eid,'status':'skipped_runtime_question','stage':e['next'],'reason':e.get('error',e['status'])}
              for eid,e in s['episodes'].items() if e['status']!='complete']
    before=read(enrollment)['skipped'] if enrollment else []
    return {'created_at':now(),'complete_count':len(complete),'complete_episode_ids':complete,
            'input_skipped':before,'runtime_skipped':failures,'skipped_count':len(before)+len(failures),
            'input_total':len(complete)+len(failures)+len(before),
            'status':'processed_with_skips' if before or failures else 'complete',
            'output_directory':str(root/'annotations/confirmed_json')}


class Sequence:
    def __init__(self,plan_path):
        self.plan_path=Path(plan_path);self.plan=read(plan_path);self.folder=self.plan_path.parent
        bind(self.folder/'plan.lock.json',{'sha256':sha(plan_path)})
        self.path=self.folder/'sequence_state.json'
        self.state=read(self.path) if self.path.exists() else {'batch_index':0,'passes':{},'batch_results':[]}
        self.state.update(identity(),status='running',updated_at=now());self.save()

    def save(self):
        self.state['updated_at']=now();atomic(self.path,self.state)

    def block(self,reason):
        self.state.update(status='blocked',reason=reason);self.save()
        print('Sequence blocked: '+reason,flush=True)

    def run(self):
        from .validate import validate_run
        for index,batch in enumerate(self.plan['batches']):
            if index<self.state['batch_index']:continue
            cfg=read(batch['config']);root=Path(cfg['output_root']);state_path=root/'state.json'
            while True:
                s=read(state_path) if state_path.exists() else None
                if s and alive(s):
                    self.state.update(active_batch=batch['name'],active_pipeline={k:s[k] for k in ('pid','create_time','command')});self.save()
                    time.sleep(10);continue
                # Probe the real OS lock as well as process identity before launch.
                try:
                    with lock(root/'pipeline.lock'):pass
                except BlockingIOError:
                    self.block('Unidentified owner holds pipeline lock: '+str(root));return
                if s and s.get('resource_error'):
                    self.block('Resource error; no next batch started: '+str(s['resource_error']));return
                if s and s.get('status')=='stopped':
                    self.block('Operator stopped pipeline; no automatic restart: '+batch['name']);return
                if s and all(e['status']=='complete' for e in s['episodes'].values()):
                    if not (root/'final_validation.json').exists() or read(root/'final_validation.json').get('status')!='passed':
                        result=validate_run(root)
                        if result['status']!='passed':self.block('Completed outputs failed final validation: '+batch['name']);return
                    if not predecessor_complete(s,root,len(read(cfg['annotation_inventory'])['episodes'])):
                        self.block('Final completion gate did not pass: '+batch['name']);return
                    break
                if s and batch['policy']=='skip_questions':
                    if any(e['status'] in ('queued','running','waiting_external_resource') for e in s['episodes'].values()):
                        self.block('Pipeline exited with unfinished queue or resource blocker: '+batch['name']);return
                    complete=[eid for eid,e in s['episodes'].items() if e['status']=='complete']
                    result=validate_run(root,only=complete)
                    if result['status']!='passed':self.block('Successful subset failed validation: '+batch['name']);return
                    break
                used=self.state['passes'].get(batch['name'],0)
                if used>=batch.get('max_passes',1):
                    self.block('Retry allowance exhausted; predecessor incomplete: '+batch['name']);return
                if s:
                    errors=[e for e in s['episodes'].values() if e['status']=='needs_repair']
                    # Retry only selector transport failures, never malformed inputs,
                    # unknown Qwen completions, or schema/evidence/identity errors.
                    if not errors or any(e['next']!='astra' or not self.transport_failure(root,eid)
                            for eid,e in s['episodes'].items() if e['status']=='needs_repair'):
                        self.block('Unresolved non-transport error in predecessor: '+batch['name']);return
                    time.sleep(20)
                self.state['passes'][batch['name']]=used+1;self.save()
                env=os.environ.copy()
                key=self.plan.get('qwen_key_file')
                if key:env['QWEN_API_KEY']=Path(key).read_text().strip()
                cmd=['bash',str(ROOT/'pipeline115.sh'),'run','--config',batch['config']]
                if s:cmd.append('--retry-failed')
                log_path=Path(batch['log']);log_path.parent.mkdir(parents=True,exist_ok=True)
                with log_path.open('a') as log:
                    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.state.update(active_batch=batch['name'],launch_pid=p.pid);self.save()
                # Waiting is done in this detached coordinator, not the chat turn.
                code=p.wait();self.state['last_exit_code']=code;self.save()
                # An early launch failure must never reuse a preceding terminal
                # state as if the new process had actually processed this batch.
                if not state_path.exists():self.block('Pipeline did not initialize state: '+batch['name']);return
            if batch['policy']=='strict_complete' and not predecessor_complete(read(state_path),root,batch['expected']):
                self.block('Strict predecessor count/validation gate failed: '+batch['name']);return
            result=summary(root,batch.get('enrollment_report'))
            atomic(root/'batch_summary.json',result)
            self.state['batch_results'].append({'name':batch['name'],**result})
            self.state['batch_index']=index+1;self.save()
        self.state.update(status='complete',finished_at=now());self.save()

    @staticmethod
    def transport_failure(root,eid):
        logs=sorted((root/'logs'/eid).glob('astra-*.log'),key=lambda p:p.stat().st_mtime)
        return bool(logs and 'TransportError:' in logs[-1].read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
    args.plan.parent.mkdir(parents=True,exist_ok=True)
    with lock(args.plan.parent/'sequence.lock'):
        runner=Sequence(args.plan)
        try:runner.run()
        except Exception as exc:
            runner.block(type(exc).__name__+': '+str(exc));raise

if __name__=='__main__':main()
