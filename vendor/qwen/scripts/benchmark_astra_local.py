"""Fixed real-image workload for comparing local GPU server configurations."""
import argparse
import base64
import io
import json
from pathlib import Path
import threading
import time
import subprocess

from PIL import Image
import requests
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stress-lines", type=int, default=0)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    content = [{"type": "text", "text": "按顺序观察四张搜索事件图，简要描述可见工作台、主要工具和手部动作；只描述可见事实。用中文输出JSON，summary一两句话，visible_objects最多8项，changes最多3项，uncertainties最多2项。"}]
    folder = Path("/home/baai/zl/demo/5三帧图像")
    names = ["5_第1次_开始_f150.jpg", "5_第1次_进行中_f240.jpg", "5_第1次_结束_f330.jpg", "5_第2次_开始_f450.jpg"]
    for index, name in enumerate(names):
        with Image.open(folder/name) as im:
            im = im.convert("RGB")
            im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
        content.extend([{"type": "text", "text": name}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()}}])
    schema = {"type": "object", "properties": {"summary": {"type": "string"},
        **{k: {"type": "array", "items": {"type": "string"}} for k in ["visible_objects", "changes", "uncertainties"]}},
        "required": ["summary", "visible_objects", "changes", "uncertainties"], "additionalProperties": False}
    body = {"model": cfg["server"]["alias"], "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024, "temperature": 0, "seed": 42, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False}, "reasoning_format": "deepseek",
        "response_format": {"type": "json_schema", "json_schema": {"name": "benchmark", "strict": True, "schema": schema}}}
    if args.stress_lines:
        content.insert(0, {"type": "text", "text": "以下仅是容量压力测试的填充数据，不是场景事实，请忽略其语义。\n"+
            "\n".join(f"capacity_test_entry_{i:05d}: context capacity test, not visual evidence." for i in range(args.stress_lines))})
    samples = []
    stop = threading.Event()
    def sample():
        while not stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            if r.returncode == 0:
                samples.append({"time": time.time(), "values": r.stdout.strip()})
            stop.wait(.5)
    thread = threading.Thread(target=sample);thread.start()
    start = time.monotonic()
    report = {"config": cfg, "input_images": names, "image_max_edge": 1024, "schema": schema,
              "stress_lines": args.stress_lines, "workload": "synthetic_context_stress_with_real_images" if args.stress_lines else "real_images_short_context"}
    try:
        session = requests.Session();session.trust_env = False
        response = session.post("http://127.0.0.1:8080/v1/chat/completions",json=body,timeout=600)
        (out/"response.json").write_text(response.text)
        response.raise_for_status()
        raw = response.json()
        report.update(elapsed_s=time.monotonic()-start,usage=raw.get("usage"),timings=raw.get("timings"),finish_reason=raw["choices"][0]["finish_reason"])
        assert report["finish_reason"] == "stop"
        from jsonschema import Draft202012Validator
        result=json.loads(raw["choices"][0]["message"]["content"]);Draft202012Validator(schema).validate(result)
        (out/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
        report["status"]="passed"
    except Exception as exc:
        report.update(status="failed",error=str(exc),elapsed_s=time.monotonic()-start)
        raise
    finally:
        stop.set();thread.join()
        report["samples"] = samples
        report["peak_vram_mib"] = max((int(s["values"].split(",")[0]) for s in samples),default=None)
        (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k not in {"samples","schema","config","input_images"}},ensure_ascii=False))


if __name__ == "__main__":
    main()
