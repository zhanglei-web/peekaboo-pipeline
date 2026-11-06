"""Actual batch execution time, excluding downtime between runs; never sum overlapping lanes."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'search_episode_outputs'

def read(p):return json.loads(p.read_text())
def seconds(a,b):return (datetime.fromisoformat(b)-datetime.fromisoformat(a)).total_seconds()
def fmt(v):
 n=round(v);return f'{n//3600}小时{n%3600//60:02d}分{n%60:02d}秒'

def main():
 pre=read(ROOT/'state.json');inv=read(ROOT/'annotation_inventory.json');facts=read(ROOT/'dataset_facts.json')
 original=sorted((ROOT/'repair_evidence').glob('*/batch_status.json'))[0];initial=read(original)
 last=OUT/'episodes/cs_0001_session_20261030_184455/complete.json'
 if not last.exists():raise RuntimeError('Last episode still running; final actual total cannot be reported yet')
 orch=read(ROOT/'orchestrator_state.json');last_receipt=read(last)
 startup=orch['started_at'];end=last_receipt['completed_at']
 pre_s=seconds(pre['started_at'],pre['finished_at']);first_s=seconds(initial['started_at'],initial['finished_at']);retry_s=seconds(startup,end)
 # Stage intervals in HTTP receipts are a workload breakdown, not additional wall time.
 local=[]
 for ep in inv['episodes']:
  folder=OUT/'episodes'/ep['episode_id'];receipt=read(folder/'complete.json');assert receipt['status']=='passed'
  records=[]
  for timing in folder.rglob('timing.json'):
   response=timing.parent/'response.json'
   if not response.exists():continue
   d=read(response)
   if d.get('model')=='local-vlm' and d.get('choices',[{}])[0].get('finish_reason')=='stop':records.append(read(timing)['seconds'])
  local.append({'episode_id':ep['episode_id'],'events':ep['event_count'],'successful_local_request_seconds':sum(records),'local_request_count':len(records)})
 result={'version':'11-4','episodes':len(local),'events':sum(e['events'] for e in local),'preprocessing_seconds':pre_s,'first_annotation_execution_seconds':first_s,'resumed_annotation_execution_seconds':retry_s,'active_batch_wall_seconds':pre_s+first_s+retry_s,'successful_local_request_seconds':sum(e['successful_local_request_seconds'] for e in local),'rows':local,'definition':'Sum of actual batch execution windows; excludes stopped time between runs; includes transport timeouts/retries during execution. Preparation/local pipeline overlap is counted once. CPU-only publication repair was not separately wall-clock-instrumented and is not invented. HTTP request durations are not pure GPU kernel time.','evidence':{'preprocessing':str(ROOT/'state.json'),'first_annotation':str(original),'resume':str(ROOT/'orchestrator_state.json'),'resume_final_receipt':str(last)}}
 (ROOT/'docs/processing_time.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 lines=['# 29条实际处理耗时（11-4）','',f'**实际运行总时长：{fmt(result["active_batch_wall_seconds"])}。**','', '| 实际处理阶段 | 用时 |','|---|---:|',f'| 29条预处理、深度、rosbag及读回 | {fmt(pre_s)} |',f'| 首次流水线标注运行 | {fmt(first_s)} |',f'| 剩余一条断点补跑（含服务启动） | {fmt(retry_s)} |',f'| 实际运行合计 | **{fmt(result["active_batch_wall_seconds"])}** |','', '口径：只累计进程实际处理的运行时间段；扣除两次运行之间停机等待修复的时间。流水线准备与本地标注重叠部分只计一次。实际运行中发生的网络请求、超时和重试仍消耗运行时间，已包含；这不是纯GPU内核计时。CPU组装修复未单独记录完整墙钟时间，不补造秒数。','',f'模型计算量另可核对：全部29条成功本地请求累计{fmt(result["successful_local_request_seconds"])}（作为阶段明细，已经包含在上述运行时间，不能再加一次）。不将Astra与Qwen并行请求时间相加充当批次耗时。','', '逐条请求时间及审计文件路径见 [processing_time.json](processing_time.json)。所有数值来自实际运行记录，没有使用5条样本估算整批时间。']
 (ROOT/'docs/PROCESSING_TIME.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['rows','evidence']},ensure_ascii=False))
if __name__=='__main__':main()
