"""Plot all six scenes / twenty held-out recordings using action probability curves."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from compare_event_segments import read, jsonl, digest, write_json, temporal_iou, metrics, style


VERBS = [("open", "打开", "#2563eb"), ("close", "关闭", "#dc2626"),
         ("take", "拿取", "#7c3aed"), ("put", "放置", "#0d9488")]


def collect(root):
    manifest_path, score_path = root / "manifest.json", root / "holdout_score.json"
    manifest, score = read(manifest_path), read(score_path)
    assert digest(manifest_path) == score["summary"]["manifest_sha256"]
    assert len(manifest["recordings"]) == score["summary"]["recordings"] == 20
    assert score["summary"]["sol_review_used"] is False
    scored = {r["recording_id"]: r for r in score["recordings"]}
    hashes = {str(p): digest(p) for p in (manifest_path, score_path, root / "HOLDOUT_RESULTS.md")}
    records = []
    for source in manifest["recordings"]:
        name = source["recording_id"]; folder = root / name; original = scored[name]
        annotation = Path(source["annotation"])
        assert digest(annotation) == source["annotation_sha256"]
        text = annotation.read_text()
        truth = [{"id": f"G{i}", "start": int(s), "middle": int(m), "end": int(e)}
                 for i,s,m,e in re.findall(r"^\| 第(\d+)次 \| (\d+) \| (\d+) \| (\d+) \|", text, re.M)]
        pred_path = folder / "events/predictions.json"
        triplet_path = folder / "triplets/summary.json"
        assert digest(pred_path) == original["predictions_sha256"]
        assert digest(triplet_path) == original["triplets_sha256"]
        predictions = read(pred_path)["predictions"]
        visual = read(folder / "events_visual/predictions.json")["predictions"]
        triplets = read(triplet_path)
        assert triplets["vlm_used"] is False
        by_event = {t["event_id"]: t for t in triplets["results"]}
        event_manifest = read(folder / "events/manifest.json")
        assert event_manifest["vlm_used"] is False
        action_path = folder / "actions/actions.jsonl"
        assert digest(action_path) == event_manifest["actions_sha256"]
        assert digest(folder / "actions/manifest.json") == event_manifest["actions_manifest_sha256"]
        actions, labels = jsonl(action_path), read(folder / "actions/labels.json")
        detector = read(folder / "detection/detection_report.json")
        detections = jsonl(folder / "detection/detections.jsonl")
        assert len(detections) == detector["processed_frames"] == detector["expected_frames"]
        hand_threshold = event_manifest["visual_configuration"]["hand_score_min"]
        observed_hands, observed_contact = [], []
        for row in detections:
            if row["status"] != "ok":
                observed_hands.append(None); observed_contact.append(None); continue
            observed_hands.append(int(any(h["score"] >= hand_threshold for h in row["hands"])))
            observed_contact.append(int(any(h["score"] >= hand_threshold and h["contact_state_id"] in (3,4)
                                           and h.get("matched_object_id") is not None for h in row["hands"])))
        rows = []
        for i,p in enumerate(predictions):
            t = by_event[p["event_id"]]
            assert t["status"] == "complete" and t["vlm_used"] is False and t["model_calls"] == 0
            assert t["event_source_range"] == [p["start_source_frame_id"],p["end_source_frame_id"]]
            frames = dict(zip(("pre","pnr","post"),(t[k] for k in ("p_frame_id","n_frame_id","r_frame_id"))))
            assert p["start_source_frame_id"] <= frames["pre"] <= frames["pnr"] <= frames["post"] <= p["end_source_frame_id"]
            rows.append({"id":f"P{i+1}","event_id":p["event_id"],"start":p["start_source_frame_id"],
                "end":p["end_source_frame_id"],"frames":frames,"boundary_reason":p["boundary_reason"],
                "phase_labels_unordered":p["action_phases"],"selection_method":t["selection_method"],
                "maximum_feature_change":t["maximum_feature_change"],"contact_object":t["object_category"]})
        assert len(rows) == original["prediction_count"] and len(truth) == original["reference_count"]
        assert len(visual) == original["visual_prediction_count"]
        by_id = {p["event_id"]:p for p in rows}
        values = {}
        for threshold in ("0.3","0.5","0.7"):
            aligned = original["alignments"][threshold]
            pairs = []
            for pair in aligned["matched_pairs"]:
                p = by_id[pair["prediction_id"]]; g = truth[int(pair["reference_id"].split("_")[-1])-1]
                value = temporal_iou((p["start"],p["end"]),(g["start"],g["end"]))
                assert abs(value - pair["temporal_iou"]) < 1e-12
                assert value >= float(threshold)
                pairs.append({"prediction":p["id"],"truth":g["id"],"tIoU":value})
            assert len({p["prediction"] for p in pairs}) == len({p["truth"] for p in pairs}) == len(pairs)
            values[threshold] = {**metrics(len(pairs),len(rows),len(truth)),"pairs":pairs}
        inputs = [annotation,pred_path,triplet_path,action_path,folder/"actions/labels.json",
                  folder/"actions/manifest.json",folder/"events/manifest.json",
                  folder/"events_visual/predictions.json",folder/"detection/detection_report.json",
                  folder/"detection/detections.jsonl"]
        for path in inputs: hashes[str(path)] = digest(path)
        records.append({"recording":name,"scene":source["scene"],"target":source["category"],
            "frame_count":detector["expected_frames"],"truth":truth,"predictions":rows,
            "visual_predictions":[{"id":f"V{i+1}","start":p["start_source_frame_id"],"end":p["end_source_frame_id"]} for i,p in enumerate(visual)],
            "metrics":values,"action_frames":[a["source_frame_id"] for a in actions],
            "action_curves":{v:[float(a["verb_probabilities"][labels["verb"].index(v)]) for a in actions] for v,_,_ in VERBS},
            "evidence_frames":[a["source_frame_id"] for a in detections],
            "observed_hands":observed_hands,"observed_contact":observed_contact})
        print(f"Read {source['scene']} {name}: GT={len(truth)}, predicted={len(rows)}",flush=True)
    for threshold, stored in score["summary"]["event_metrics"].items():
        computed = metrics(sum(r["metrics"][threshold]["TP"] for r in records),
                           sum(len(r["predictions"]) for r in records),sum(len(r["truth"]) for r in records))
        assert computed["TP"] == stored["matched"]
        assert abs(computed["F1"] - stored["f1"]) < 1e-12
    return {"plot_date":"2026-11-19","source_root":str(root),"scope":score["summary"]["scope"],
        "primary_threshold":manifest["primary_threshold"],"selection":manifest["selection"],"seed":manifest["seed"],
        "vlm_used_for_triplets":False,"n_selection":score["summary"]["n_selection"],
        "new_model_calls":0,"event_metrics":score["summary"]["event_metrics"],
        "visual_only_event_metrics":score["summary"]["visual_only_event_metrics"],
        "scene_metrics":score["summary"]["scene_metrics"],"source_hashes":hashes,"recordings":records}


def intervals(ax, rows, y, color, mark_middle=False):
    for row in rows:
        ax.broken_barh([(row["start"],row["end"]-row["start"]+1)],(y-.27,.54),facecolors=color,edgecolors="white",linewidth=.6)
        ax.text((row["start"]+row["end"])/2,y+.03,row["id"],ha="center",va="center",fontsize=8,color="white")
        if mark_middle: ax.plot([row["middle"]]*2,[y-.24,y-.11],color="white",lw=1.2)


def event_axis(ax, rec):
    intervals(ax,rec["truth"],5.,"#475569",True)
    intervals(ax,rec["visual_predictions"],4.,"#b7791f")
    intervals(ax,rec["predictions"],3.,"#087f5b")
    for y,role,marker,color in [(2.,"pre","o","#475569"),(1.,"pnr","D","#7c3aed"),(0.,"post","^","#0d9488")]:
        for row in rec["predictions"]:
            ax.scatter(row["frames"][role],y,s=27,marker=marker,color=color,zorder=3)
            ax.annotate(row["id"],(row["frames"][role],y),xytext=(0,5),textcoords="offset points",ha="center",fontsize=7)
    ax.set_yticks([5,4,3,2,1,0],labels=["人工 G","仅视觉 V","TSM P","P（簇首）","N（规则）","R（簇末）"])
    ax.set_ylim(-.4,5.5);ax.set_xlim(0,rec["frame_count"]+15)
    ax.grid(axis="x",color="#e2e8f0",lw=.6);ax.set_axisbelow(True)
    ax.set_xlabel("原视频源帧号（从 0 开始）")


def action_axis(ax, rec):
    for verb,cn,color in VERBS:
        ax.plot(rec["action_frames"],rec["action_curves"][verb],color=color,lw=1.25,label=f"{verb} {cn}")
    ax.set_xlim(0,rec["frame_count"]+15);ax.set_ylim(-.025,1.06)
    ax.set_ylabel("TSM softmax（非准确率）");ax.set_xlabel("原视频源帧号（从 0 开始）")
    ax.grid(alpha=.18);ax.set_axisbelow(True)


def evidence_axis(ax, rec):
    ax.step(rec["evidence_frames"],np.array(rec["observed_hands"],dtype=float),where="post",color="#64748b",ls="--",lw=1.2,label="检测到手")
    ax.step(rec["evidence_frames"],np.array(rec["observed_contact"],dtype=float),where="post",color="#0891b2",lw=1,label="检测到物体接触")
    ax.set_ylim(-.1,1.2);ax.set_yticks([0,1]);ax.set_ylabel("观测证据")
    ax.set_xlabel("原视频源帧号（从 0 开始）");ax.legend(ncol=2,loc="upper right",fontsize=8)
    ax.grid(alpha=.18);ax.set_xlim(0,rec["frame_count"]+15)


def render(report, output):
    style();scene_root=output/"scenes";detail_root=output/"recordings"
    scene_root.mkdir(exist_ok=True);detail_root.mkdir(exist_ok=True)
    groups=defaultdict(list)
    for rec in report["recordings"]:groups[rec["scene"]].append(rec)
    with PdfPages(output/"all-scenes-overview.pdf") as book:
        for scene,records in groups.items():
            fig,axes=plt.subplots(len(records),2,figsize=(14,3.65*len(records)+.5),layout="constrained",squeeze=False)
            for i,rec in enumerate(records):
                event_axis(axes[i,0],rec);action_axis(axes[i,1],rec)
                axes[i,0].set_title(f"{rec['target']}／{rec['recording'][-6:]}｜人工 {len(rec['truth'])} / TSM {len(rec['predictions'])}",loc="left",fontsize=11)
                axes[i,1].set_title(f"动作概率曲线｜F1@0.5={rec['metrics']['0.5']['F1']:.3f}",loc="left",fontsize=11)
                axes[i,1].legend(ncol=4,loc="upper center",fontsize=8)
            fig.suptitle(f"{scene}｜六场景20录制固定测试中的 {len(records)} 条\nTSM＋接触融合事件；N=空间变化最大帧（无 VLM 选帧）",fontsize=13)
            for suffix in ("png","pdf"):fig.savefig(scene_root/f"{scene}-event-action-overview.{suffix}",dpi=165,bbox_inches="tight")
            book.savefig(fig,bbox_inches="tight");plt.close(fig)
            print(f"Rendered scene {scene}",flush=True)
    with PdfPages(output/"all-recordings-detail.pdf") as book:
        for rec in report["recordings"]:
            fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,gridspec_kw={"height_ratios":[2,1.45,.65]},layout="constrained")
            event_axis(axes[0],rec);action_axis(axes[1],rec);evidence_axis(axes[2],rec)
            axes[1].legend(ncol=4,loc="upper right",fontsize=9)
            for ax in axes[:2]:ax.set_xlabel("")
            m=rec['metrics']['0.5']
            fig.suptitle(f"{rec['scene']} {rec['target']}／{rec['recording'][-6:]}｜原归档动作概率折线\n人工 {len(rec['truth'])} / TSM {len(rec['predictions'])}｜主门槛 tIoU≥0.5：P={m['precision']:.3f}，R={m['recall']:.3f}，F1={m['F1']:.3f}",fontsize=13)
            stem=f"{rec['scene']}-{rec['recording'][-6:]}"
            for suffix in ("png","pdf"):fig.savefig(detail_root/f"{stem}.{suffix}",dpi=160,bbox_inches="tight")
            book.savefig(fig,bbox_inches="tight");plt.close(fig)
            print(f"Rendered recording {stem}",flush=True)


def write_outputs(report, output):
    write_json(output/"comparison.json",report)
    rows=[]
    for rec in report["recordings"]:
        row={"scene":rec["scene"],"target":rec["target"],"recording":rec["recording"],
             "human_events":len(rec["truth"]),"tsm_events":len(rec["predictions"]),"visual_events":len(rec["visual_predictions"])}
        for t,m in rec["metrics"].items():
            for key in ["TP","FP","FN","precision","recall","F1"]:row[f"{key}@{t}"]=m[key]
        rows.append(row)
    with (output/"recording-metrics.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    triplets=[{"scene":r["scene"],"target":r["target"],"recording":r["recording"],"event":p["id"],"start":p["start"],"end":p["end"],
               **p["frames"],"contact_object":p["contact_object"],"selection_method":p["selection_method"]} for r in report["recordings"] for p in r["predictions"]]
    with (output/"event-triplets.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(triplets[0]),lineterminator="\n");writer.writeheader();writer.writerows(triplets)
    lines=["# 六场景20录制固定测试：动作概率曲线版", "", "绘图日期：2026-11-19。原始批次：`holdout_twenty_six_scene_v1`，分层抽样 seed=20261117。", "",
        "这份图对应多场景固定测试，而非三条历史开发/回归录制。全部 20 条来自 S01-1、S03、S04、S07、S08、S09；共 136 个人工事件、114 个 TSM＋接触融合预测事件。", "",
        "实际归档流程为 **TSM＋接触融合/规则切分 → 接触簇端点 P/R → 空间变化最大帧 N**。该批 `sol_review_used=false`，全部三帧选择器 `vlm_used=false`、`model_calls=0`，不能标作 Qwen 27B 或 Sol 选帧结果。", "",
        "## 原始指标", "", "主门槛在原 manifest 中预先固定为 tIoU≥0.5，0.3/0.7 为辅助门槛。", "",
        "| tIoU 门槛 | 匹配/预测/人工 | Precision | Recall | F1 |", "| --- | --- | --- | --- | --- |"]
    for t,m in report["event_metrics"].items():lines.append(f"| {t} | {m['matched']}/{m['predictions']}/{m['references']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    lines += ["", "其中约 0.71 是宽松门槛下的 Precision，不是总体准确率，也不是 F1=0.9。先前 0.929 的三条开发/回归批次见 [历史高分图](../historical-high-score/README.md)，两批数据与评价范围不同。", "",
        "## 六个场景总览", "", "每个场景覆盖该场景的全部录制。左侧为人工区间、仅视觉基线、TSM 预测及规则 P/N/R；右侧为实际 TSM 的 open/close/take/put 概率折线，未平滑或重跑模型。", ""]
    scenes=list(dict.fromkeys(r['scene'] for r in report['recordings']))
    for scene in scenes:
        records=[r for r in report['recordings'] if r['scene']==scene]
        lines += [f"### {scene}（{len(records)} 条）", "",f"![{scene} 事件与动作曲线](scenes/{scene}-event-action-overview.png)", "",f"[该场景 PDF](scenes/{scene}-event-action-overview.pdf)", ""]
    lines += ["## 全部20条明细", "", "| 场景 | 目标 | 录制 | 人工/预测 | F1@0.3 | F1@0.5 | F1@0.7 | 曲线详情 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rec in report["recordings"]:
        stem=f"{rec['scene']}-{rec['recording'][-6:]}"
        lines.append(f"| {rec['scene']} | {rec['target']} | {rec['recording'][-6:]} | {len(rec['truth'])}/{len(rec['predictions'])} | {rec['metrics']['0.3']['F1']:.3f} | {rec['metrics']['0.5']['F1']:.3f} | {rec['metrics']['0.7']['F1']:.3f} | [PNG](recordings/{stem}.png) / [PDF](recordings/{stem}.pdf) |")
    lines += ["", "## 曲线与证据的含义", "",
        "- 四条彩色曲线是分类 softmax，不是动作类别准确率，也不是事件正确概率；没有新增人工动作类别真值。未显示的其余动词类别仍可能占有概率。",
        "- 单条详情的底部为检测到手/物体接触的二值输出，不是人工接触真值。检测处理失败画为空缺；检测不到手不直接等于确认松手。",
        "- 三帧依旧按原规则，N 允许与 P/R 重合；完整三帧输出率不等于三张不同图或语义充分性准确率。",
        "- 原批次按六场景分层随机、每场景物体不重复，预测先冻结后评估；没有用本次绘图调整参数或替换表现不佳的样本。",
        "- 图中仅视觉基线与 TSM 融合来自相同录制的两份原预测；未将仅视觉基线标作 TSM v1。",
        "- 本次复核原抽样 manifest、人工标注、事件/三帧摘要以及动作 manifest 哈希；重新计算全部原匹配对的闭区间 tIoU，汇总 81/56/28 对与原报告一致。", "",
        "## 下载与复现", "", "- [六场景总览 PDF（6页）](all-scenes-overview.pdf)", "- [20条完整曲线详情 PDF（20页）](all-recordings-detail.pdf)",
        "- [逐录制指标 CSV](recording-metrics.csv)", "- [114组三帧 CSV](event-triplets.csv)", "- [完整绘图数据和输入哈希](comparison.json)", "",
        f"原结果目录：`{report['source_root']}`。", "", "```bash", "python3 scripts/plot_multiscene_holdout.py \\",
        "  --source /home/user/zl/pipeline/runs/pnr/holdout_twenty_six_scene_v1 \\", "  --output docs/multiscene-holdout-curves", "```", ""]
    (output/"README.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--source",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    report=collect(Path(args.source).resolve());report["plot_script_sha256"]=digest(__file__)
    render(report,output)
    for path,value in report["source_hashes"].items():assert digest(path)==value, "Source changed while plotting"
    write_outputs(report,output)
    print({"recordings":len(report["recordings"]),"source_files_verified":len(report["source_hashes"]),"event_metrics":report["event_metrics"]},flush=True)


if __name__=="__main__":main()
