"""Read local demo progress without inference or mutation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT/"integrations/astra_pipeline/config/demo_5.local.json").read_text())
output = Path(config["output"])
result = {"output": str(output), "model": config["model"]["name"], "endpoint": config["model"]["base_url"],
          "final_json_exists": (output/"search_episode_5.json").is_file()}
for key, relative in [("s2_prefetch", "s2_prefetch.json"), ("selection", "input/selection.json")]:
    path=output/relative
    if path.exists():
        d=json.loads(path.read_text())
        result[key]={k:v for k,v in d.items() if k not in {"frames", "selected_original_video_frames"}}
for key, filename in [("pipeline", "pipeline_manifest.json"), ("observations", "run_manifest.json")]:
    path = output/filename
    if path.exists():
        doc = json.loads(path.read_text())
        result[key] = {k: doc[k] for k in ["status", "stages", "scope", "processed_frame_count", "failure", "final_episode_generated"] if k in doc}
for stage in ["s2", "s3", "final"]:
    path = output/stage/"last_attempt.json"
    if path.exists():
        doc = json.loads(path.read_text())
        result[stage] = {k: doc[k] for k in ["status", "failure", "error", "phase", "processed_count"] if k in doc}
launch = output/"launch.json"
if launch.exists():
    doc = json.loads(launch.read_text())
    import psutil
    try:
        proc = psutil.Process(doc["pid"])
        result["worker_alive"] = abs(proc.create_time()-doc["create_time"]) < .01 and proc.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        result["worker_alive"] = False
    result["log"] = doc["log"]
print(json.dumps(result, ensure_ascii=False, indent=2))
