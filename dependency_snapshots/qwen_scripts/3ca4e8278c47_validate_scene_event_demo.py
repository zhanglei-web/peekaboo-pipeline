"""End-to-end demo validation before the todo batch."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import argparse
from pathlib import Path
import signal
import time

import psutil
from run_scene_event_batch import ROOT,save,sha,scene_job,event_job,export_episode,full_contract,verify_processes,triplets


def main():
    parser=argparse.ArgumentParser(description='Historical static demo reproduction only')
    parser.add_argument('--legacy-static',action='store_true')
    args=parser.parse_args()
    if not args.legacy_static:
        parser.error('This is a frozen legacy demo. For new annotations use run_scene_event_batch.py (nested temporal trees by default); historical reproduction requires --legacy-static.')
    out=ROOT/'outputs/astra_tf_selected_scene_demo5'
    reference=json.loads(Path('/home/baai/zl/demo/search_episode_5.json').read_text())
    index=triplets(Path('/home/baai/zl/demo/5三帧图像'),'5',30)
    row={'episode_id':'5','event_index':index,'event_images':index['directory'],'event_count':index['event_count'],
         'target':reference['metadata']['target']['description'],'fps':30}
    record=json.loads((ROOT/'logs/server.pid.json').read_text());server,workers=verify_processes(record)
    stopped=server.status()==psutil.STATUS_STOPPED
    report={'status':'running','event_count':index['event_count']};tick=time.perf_counter()
    if stopped:server.send_signal(signal.SIGCONT)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures={pool.submit(scene_job,out):('scene',0)}
            for e in index['events']:
                futures[pool.submit(event_job,e,out/'events_v2'/f"event_{e['ordinal']:02d}")]=('event',e['ordinal'])
            events={};scene=None
            for f in as_completed(futures):
                kind,n=futures[f];data=f.result()
                if kind=='scene':scene=data
                else:events[n]=data
                print(json.dumps({'completed':kind,'ordinal':n,'events_done':len(events)},ensure_ascii=False),flush=True)
        result=export_episode(row,scene,[events[e['ordinal']] for e in index['events']])
        save(out/'search_episode_5.v2.json',result);save(out/'target.schema.json',full_contract())
        report.update(status='validated',objects=len(scene['objects']),events=len(events),output_sha256=sha(out/'search_episode_5.v2.json'))
    finally:
        if stopped:
            server,workers=verify_processes(record);server.send_signal(signal.SIGSTOP)
            save(out/'pause_restored.json',{'server_pid':server.pid,'workers':workers})
        report['seconds']=time.perf_counter()-tick;save(out/'demo_report.v2.json',report)
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
