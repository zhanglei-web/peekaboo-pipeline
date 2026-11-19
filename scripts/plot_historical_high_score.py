"""Plot the archived 14-event high-score cohort with its actual model attribution."""
import argparse
import csv
from pathlib import Path
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

from compare_event_segments import (read, jsonl, digest, write_json, temporal_iou,
                                    match_intervals, metrics, top_scores, style, timeline, heatmap)


MODEL = "gpt-6.1-sol"
THRESHOLDS = (.3, .5, .7)


def build_records(root):
    base = root / "runs/pnr"
    pilot_path = base / "triplet_accuracy_pilot_v1/summary.json"
    sol_path = base / "vlm_review_v1/summary.json"
    pilot, sol = read(pilot_path), read(sol_path)
    assert sol["model"] == MODEL and sol["endpoints_fixed"] is True and sol["upstream_events_modified"] is False
    by_sol = {(r["recording"], r["event_id"]): r for r in sol["results"]}
    hashes = {str(p): digest(p) for p in [pilot_path, sol_path, base / "vlm_review_v1/verification.json"]}
    diagnostics = base / "tsm_events_v2/diagnostics"
    action_sources = {digest(p): p for p in diagnostics.glob("*/actions.jsonl")}
    records, receipts = [], []
    for detail in pilot["recording_details"]:
        for name, value in detail["input_sha256"].items():
            assert digest(name) == value, f"Historical source changed: {name}"
            hashes[name] = value
        recording = detail["recording_id"]
        annotation = Path(detail["source_annotation"])
        text = annotation.read_text()
        fps = float(re.search(r"帧率\s*([\d.]+)\s*帧", text).group(1))
        scene = re.search(r"^场景：\s*(.+)$", text, re.M).group(1).strip()
        target = re.search(r"^要找的物体：\s*(.+)$", text, re.M).group(1).strip()
        truth = [{"id": f"G{i}", "start": int(s), "middle": int(m), "end": int(e)}
                 for i, s, m, e in re.findall(r"^\| 第(\d+)次 \| (\d+) \| (\d+) \| (\d+) \|", text, re.M)]
        pred_path = next(Path(p) for p in detail["input_sha256"] if p.endswith("events/predictions.json"))
        predictions = read(pred_path)["predictions"]
        event_manifest_path = pred_path.with_name("manifest.json")
        manifest = read(event_manifest_path)
        action_path = action_sources[manifest["actions_sha256"]]
        labels_path = action_path.with_name("labels.json")
        actions, labels = jsonl(action_path), read(labels_path)
        assert digest(action_path.with_name("manifest.json")) == manifest["actions_manifest_sha256"]
        for p in [annotation, pred_path, event_manifest_path, action_path, labels_path, action_path.with_name("manifest.json")]:
            hashes[str(p)] = digest(p)
        cap = cv2.VideoCapture(detail["source_video"])
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); video_fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
        assert frame_count > 0 and abs(video_fps - fps) < .01
        rows = []
        for i, pred in enumerate(predictions):
            selected = by_sol[recording, pred["event_id"]]
            assert selected["model"] == MODEL and selected["endpoints_fixed"] is True
            assert selected["event_range"] == [pred["start_source_frame_id"], pred["end_source_frame_id"]]
            case = Path(selected["case"])
            baseline = read(case / "selection/result.json")
            assert [selected["p_frame_id"], selected["r_frame_id"]] == [baseline["p_frame_id"], baseline["r_frame_id"]]
            receipt_path = case / "vlm_review/receipts.json"
            for receipt in read(receipt_path):
                assert receipt["model"] == receipt["returned_model"] == MODEL
                receipts.append({"recording":recording,"event":pred["event_id"],"requested_model":receipt["model"],
                                 "returned_model":receipt["returned_model"],"image_count":receipt["image_count"]})
            hashes[str(receipt_path)] = digest(receipt_path)
            frames = dict(zip(("pre","pnr","post"), (selected[k] for k in ("p_frame_id","n_frame_id","r_frame_id"))))
            assert frames["pre"] <= frames["pnr"] <= frames["post"]
            windows = [a for a in actions if pred["start_source_frame_id"] <= a["source_frame_id"] <= pred["end_source_frame_id"]]
            assert windows
            mean = np.mean([a["verb_probabilities"] for a in windows], axis=0)
            rows.append({"id":f"P{i+1}","event_id":pred["event_id"],"start":pred["start_source_frame_id"],
                "end":pred["end_source_frame_id"],"frames":frames,"baseline_n":selected["baseline_n_frame_id"],
                "model":MODEL,"status":selected["status"],"review_required":selected["review_required"],
                "model_action":selected["action"],"tsm_top_verbs":top_scores(mean,labels["verb"]),
                "boundary_reason":pred["boundary_reason"]})
        assert len(rows) == detail["prediction_count"] and len(truth) == detail["reference_count"]
        matrix = np.array([[temporal_iou((p["start"],p["end"]),(g["start"],g["end"])) for g in truth] for p in rows])
        historical_metrics = {}
        for threshold in THRESHOLDS:
            old = detail["alignments"][str(threshold)]
            pairs = []
            for pair in old["matched_pairs"]:
                i = next(i for i,p in enumerate(rows) if p["event_id"] == pair["prediction_id"])
                j = int(pair["reference_id"].split("_")[-1])-1
                assert abs(matrix[i,j]-pair["temporal_iou"]) < 1e-12
                pairs.append({"prediction":rows[i]["id"],"truth":truth[j]["id"],"tIoU":float(matrix[i,j])})
            assert len(match_intervals(matrix,threshold)) == len(pairs)
            historical_metrics[str(threshold)] = {**metrics(len(pairs),len(rows),len(truth)),"pairs":pairs}
        probabilities = np.array([a["verb_probabilities"] for a in actions])
        display = ["take","put","open","close"]
        display += [v for v in (labels["verb"][i] for i in np.argsort(probabilities.mean(0))[::-1]) if v not in display][:2]
        records.append({"recording":recording,"scene":scene,"target":target,"fps":fps,"frame_count":frame_count,
            "truth":truth,"predictions":rows,"metrics":historical_metrics,"tIoU_matrix":matrix.tolist(),
            "action_source":str(action_path),"display_verbs":display,
            "action_heatmap":[[float(a["verb_probabilities"][labels["verb"].index(v)]) for a in actions] for v in display],
            "action_windows":[{"source_frame_id":a["source_frame_id"]} for a in actions]})
    assert len(receipts) == read(base / "vlm_review_v1/verification.json")["actual_calls"] == 19
    micro = {}
    for threshold in THRESHOLDS:
        count = sum(r["metrics"][str(threshold)]["TP"] for r in records)
        p = sum(len(r["predictions"]) for r in records); g = sum(len(r["truth"]) for r in records)
        micro[str(threshold)] = metrics(count,p,g)
        assert count == pilot["event_metrics"][str(threshold)]["matched"]
        assert abs(micro[str(threshold)]["F1"]-pilot["event_metrics"][str(threshold)]["f1"]) < 1e-12
    return {"plot_date":"2026-11-19","experiment_date":"2026-11-17","model":MODEL,
        "scope":pilot["scope"],"endpoints_fixed":True,"model_selects_only":"N",
        "event_source":"TSM + cycles rules; event F1 is not frame selection accuracy",
        "new_model_calls":0,"archived_model_receipts":receipts,"micro_metrics":micro,
        "source_hashes":hashes,"recordings":records}


def render(report, output):
    style()
    fig,axes=plt.subplots(3,2,figsize=(14,12),gridspec_kw={"width_ratios":[1.12,1]},layout="constrained")
    for i,record in enumerate(report["recordings"]):
        timeline(axes[i,0],record)
        axes[i,0].set_yticks([4.1,3.1,2.,1.2,.4],labels=["人工区间 G","TSM 预测 P","P（固定）","N（Sol）","R（固定）"])
        axes[i,0].set_xlim(0,record["frame_count"]/record["fps"]+.3)
        mesh=heatmap(axes[i,1],record)
    cbar=fig.colorbar(mesh,ax=axes[:,1].tolist(),shrink=.5,pad=.025);cbar.set_label("TSM 动作 softmax")
    fig.suptitle("历史高分批次：事件区间、P/N/R 与动作\nTSM＋规则切分｜gpt-6.1-sol 只选 N，P/R 固定｜历史开发/回归数据\n事件 F1：0.929（tIoU≥0.3）／0.857（≥0.5）／0.571（≥0.7）",fontsize=12)
    for suffix in ("png","pdf"):fig.savefig(output/f"event-action-overview.{suffix}",dpi=190,bbox_inches="tight")
    plt.close(fig)


def write_report(report, output):
    write_json(output/"comparison.json",report)
    table=[]
    for rec in report["recordings"]:
        pairs={a["prediction"]:a for a in rec["metrics"]["0.3"]["pairs"]}
        for p in rec["predictions"]:
            match=pairs.get(p["id"],{})
            table.append({"recording":rec["recording"],"target":rec["target"],"prediction":p["id"],
                "start":p["start"],"end":p["end"],"matched_truth_0.3":match.get("truth"),"tIoU":match.get("tIoU"),
                **p["frames"],"baseline_n":p["baseline_n"],"status":p["status"],"review_required":p["review_required"],
                "model":MODEL,"model_action":p["model_action"],
                "tsm_top_verbs":"; ".join(f"{a['label']}:{a['score']:.6f}" for a in p["tsm_top_verbs"])})
    with (output/"events.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator="\n");writer.writeheader();writer.writerows(table)
    lines=["# 历史高分批次 event-action-overview", "", "绘图日期：2026-11-19。原始实验：2026-11-17。", "",
        "该批实际模型为 **gpt-6.1-sol**，19 份历史请求回执的请求/返回型号均一致。TSM＋规则先切分事件，P/R 固定为有效接触簇首尾，Sol 只从候选图中选择 N。没有将该批标为 Qwen 27B。", "",
        "## 原始指标复核", "", "| 事件 tIoU 门槛 | 匹配/预测/人工 | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- |"]
    for threshold,m in report["micro_metrics"].items():
        lines.append(f"| {threshold} | {m['TP']}/14/14 | {m['precision']:.3f} | {m['recall']:.3f} | {m['F1']:.3f} |")
    lines += ["", "0.7 是区间匹配门槛；该门槛下 F1=0.571。0.929/0.857 是上游事件匹配成绩，不是 Sol 或三帧的语义准确率。", "",
        "![历史事件、三帧与动作总览](event-action-overview.png)", "",
        "蓝色条为人工事件，橙色条为自动事件，白色短线为人工进行帧；三个点行分别为固定 P、Sol 选择 N、固定 R。右侧仅使用该批事件生成时实际引用的 TSM 动作概率，已按原始事件 manifest 的动作文件/manifest 哈希匹配。动作分数未校准为分类准确率。", "",
        "## 逐录制与逐事件", "", "| 录制 | 人工事件 | 预测事件 | F1@0.3 | F1@0.5 | F1@0.7 |", "| --- | --- | --- | --- | --- | --- |"]
    for rec in report["recordings"]:
        lines.append(f"| {rec['scene']} {rec['target']}／{rec['recording'][-6:]} | {len(rec['truth'])} | {len(rec['predictions'])} | {rec['metrics']['0.3']['F1']:.3f} | {rec['metrics']['0.5']['F1']:.3f} | {rec['metrics']['0.7']['F1']:.3f} |")
    for rec in report["recordings"]:
        lines += ["",f"### {rec['target']}／{rec['recording'][-6:]}","", "人工帧（开始/进行/结束）："+"；".join(f"{g['id']}={g['start']}/{g['middle']}/{g['end']}" for g in rec["truth"]),"",
            "| ID | 自动区间 | P / Sol N / R | 原规则 N | Sol 状态/需复核 | TSM 动词 Top-3 区间均分 |", "| --- | --- | --- | --- | --- | --- |"]
        for p in rec["predictions"]:
            frames="/".join(str(p["frames"][role]) for role in ("pre","pnr","post"))
            action="；".join(f"{a['label']} {a['score']:.3f}" for a in p["tsm_top_verbs"])
            lines.append(f"| {p['id']} | {p['start']}–{p['end']} | {frames} | {p['baseline_n']} | {p['status']} / {'是' if p['review_required'] else '否'} | {action} |")
    lines += ["", "## 解释范围与来源", "",
        "- 这三条均为历史开发或已知回归录制，不是独立盲测。不能将该批高分直接解释为跨数据集泛化能力。",
        "- 它与当前 8B 完整管线的三条录制不是同一组数据；模型、选帧角色约束也不同，不能用两批 F1 差异直接比较模型优劣。",
        "- 人工进行帧仅为过程采样，不是经过复核的唯一 N 真值；没有新补动作类别或三帧语义真值。",
        "- 本次只读取历史数据绘图，未重新运行检测、TSM、事件解码或任何 VLM。43 条原历史快照输入哈希全部核对一致，19 份历史模型回执型号全部核对一致。",
        "- 同时复算闭区间 tIoU 并一对一匹配，三个门槛的匹配数量均与原报告一致，绘图后再次核对源文件未变。", "",
        "原始报告目录：`/home/user/zl/pipeline/runs/pnr/triplet_accuracy_pilot_v1/`。原模型结果目录：`/home/user/zl/pipeline/runs/pnr/vlm_review_v1/`。", "",
        "[逐事件 CSV](events.csv) · [完整数据及哈希](comparison.json) · [矢量 PDF](event-action-overview.pdf)", "",
        "```bash", "python3 scripts/plot_historical_high_score.py \\","  --legacy-root /home/user/zl/pipeline \\","  --output docs/historical-high-score", "```", ""]
    (output/"README.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--legacy-root",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=build_records(Path(args.legacy_root).resolve());report["plot_script_sha256"]=digest(__file__)
    render(report,output)
    for name,value in report["source_hashes"].items():assert digest(name)==value, "Source changed while plotting"
    write_report(report,output)
    print({"model":MODEL,"recordings":len(report["recordings"]),"historical_model_calls":len(report["archived_model_receipts"]),
           "source_files_verified":len(report["source_hashes"]),"metrics":report["micro_metrics"]})


if __name__=="__main__":
    main()
