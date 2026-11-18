"""Compare saved predictions with original human event intervals; no model inference.

python scripts/compare_event_segments.py --run-root RUN --output docs/event-comparison
Requires matplotlib, fonttools and scipy in addition to the runtime package.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment


THRESHOLDS = (.3, .5, .7)
VERB_CN = {"take": "拿取", "put": "放置", "open": "打开", "close": "关闭",
           "move": "移动", "remove": "移除", "check": "检查", "adjust": "调整",
           "wash": "清洗", "insert": "放入", "cut": "切割", "turn-on": "开启电器"}
PHASE_CN = {"acquire": "拿取/移开", "place": "放置/归位", "open": "打开", "close": "关闭"}
METHOD_CN = {"validated_geometric_union": "严格几何", "geometric_union_with_quality_warnings": "近似几何",
             "available_registered_views_lower_bound": "可用视图下界", "qwen_continuous_visual_estimate": "视觉估计",
             "no_evidence_operational_default": "默认回退"}
REASON_CN = {"contact_island_start": "接触段起点",
             "tsm_completed_phase_then_new_operation": "动作完成后出现新操作",
             "visual_change_with_observed_release": "视觉变化并观察到释放"}


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def temporal_iou(a, b):
    intersection = overlap(a, b)
    return intersection / ((a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection)


def match_intervals(matrix, threshold):
    """Maximize eligible pair count first, then summed IoU; never reuse a GT."""
    eligible = matrix >= threshold
    reward = eligible * (min(matrix.shape) + 1) + np.where(eligible, matrix, 0.)
    rows, cols = linear_sum_assignment(-reward)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if eligible[i, j]]


def metrics(tp, predicted, reference):
    precision = tp / predicted if predicted else 0.
    recall = tp / reference if reference else 0.
    return {"TP": tp, "FP": predicted - tp, "FN": reference - tp,
            "precision": precision, "recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.}


def top_scores(probabilities, labels, count=3):
    order = np.argsort(probabilities)[::-1][:count]
    return [{"label": labels[i], "score": float(probabilities[i])} for i in order]


def build_record(session, run):
    folder = run / session.name
    annotation = next(session.glob("*三帧.md"))
    text = annotation.read_text()
    fps = float(re.search(r"帧率\s*([\d.]+)\s*帧", text).group(1))
    scene = re.search(r"^场景：\s*(.+)$", text, re.M).group(1).strip()
    target = re.search(r"^要找的物体：\s*(.+)$", text, re.M).group(1).strip()
    truth = [{"id": f"G{i}", "start": int(s), "middle": int(m), "end": int(e)}
             for i, s, m, e in re.findall(r"^\| 第(\d+)次 \| (\d+) \| (\d+) \| (\d+) \|", text, re.M)]
    assert truth and [g["id"] for g in truth] == [f"G{i+1}" for i in range(len(truth))]
    actions = jsonl(folder / "actions/actions.jsonl")
    labels = read(folder / "actions/labels.json")
    diagnostics = read(folder / "events/action_diagnostics.json")
    predictions = read(folder / "events/predictions.json")["predictions"]
    previous_end = -1
    rows = []
    event_sources = []
    for i, prediction in enumerate(predictions):
        s, e = prediction["start_source_frame_id"], prediction["end_source_frame_id"]
        event_root = folder / "event_runs" / prediction["event_id"]
        selection = read(event_root / "final_selection.json")
        record = read(event_root / "scoring/committed_event.json")
        event_sources += [event_root / "final_selection.json", event_root / "scoring/committed_event.json",
                          event_root / "clusters/clusters.json"]
        windows = [a for a in actions if s <= a["source_frame_id"] <= e]
        assert windows, prediction["event_id"]
        verb_mean = np.mean([a["verb_probabilities"] for a in windows], axis=0)
        noun_mean = np.mean([a["noun_probabilities"] for a in windows], axis=0)
        phases = sorted([p for p in diagnostics["phases"] if p["end_source_frame_id"] >= s
                         and p["start_source_frame_id"] <= e], key=lambda p: p["start_source_frame_id"])
        sequence = []
        for phase in phases:
            if not sequence or sequence[-1] != phase["phase"]:
                sequence.append(phase["phase"])
        clusters = read(event_root / "clusters/clusters.json")
        selected_cluster = next((c for c in clusters if c["selected"]), None)
        boundaries = [b for b in diagnostics["selected_boundaries"]
                      if previous_end < b["source_frame_id"] <= s]
        rows.append({"id": f"P{i+1}", "event_id": prediction["event_id"], "start": s, "end": e,
            "start_seconds": s / fps, "end_seconds": e / fps,
            "boundary_reason": prediction["boundary_reason"], "boundary_evidence": boundaries,
            "event_confidence": None, "event_confidence_note": "No calibrated event probability in saved outputs",
            "phase_labels_unordered": prediction["action_phases"], "phase_sequence": sequence, "phases": phases,
            "tsm_top_verbs": top_scores(verb_mean, labels["verb"]),
            "tsm_top_nouns": top_scores(noun_mean, labels["noun"]),
            "tsm_window_count": len(windows),
            "tsm_candidate_fraction": sum(a["action_status"] == "candidate" for a in windows) / len(windows),
            "selected_contact_object": selected_cluster["object_category"] if selected_cluster else None,
            "contact_cluster": selected_cluster,
            "frames": {role: selection[role] for role in ("pre", "pnr", "post")},
            "selection_status": selection["status"], "selection_reason": selection["reason"],
            "q_event": record["q_event_estimate"],
            "scores": [{k: a.get(k) for k in ("search_unit_id", "q", "status", "score_type", "estimation_method")}
                       for a in record["assessments"]]})
        previous_end = e
    matrix = np.array([[temporal_iou((p["start"], p["end"]), (g["start"], g["end"])) for g in truth] for p in rows])
    threshold_metrics = {}
    for threshold in THRESHOLDS:
        pairs = match_intervals(matrix, threshold)
        threshold_metrics[str(threshold)] = {**metrics(len(pairs), len(rows), len(truth)),
            "pairs": [{"prediction": rows[i]["id"], "truth": truth[j]["id"], "tIoU": float(matrix[i,j]),
                       "start_error_seconds": (rows[i]["start"] - truth[j]["start"]) / fps,
                       "end_error_seconds": (rows[i]["end"] - truth[j]["end"]) / fps} for i, j in pairs],
            "unmatched_predictions": [rows[i]["id"] for i in range(len(rows)) if i not in {x for x,y in pairs}],
            "unmatched_truth": [truth[j]["id"] for j in range(len(truth)) if j not in {y for x,y in pairs}]}
    assigned = {p["prediction"]: p for p in threshold_metrics["0.3"]["pairs"]}
    for i, p in enumerate(rows):
        j = int(matrix[i].argmax())
        p["best_truth"] = truth[j]["id"] if matrix[i,j] > 0 else None
        p["best_tIoU"] = float(matrix[i,j])
        p["match_at_0.3"] = assigned.get(p["id"])
        p["frames_truth_membership"] = {role: [g["id"] for g in truth if g["start"] <= fid <= g["end"]]
                                        for role, fid in p["frames"].items()}
        p["truth_overlaps"] = [{"truth": g["id"], "overlap_frames": overlap((p["start"],p["end"]),(g["start"],g["end"])),
                                "truth_fraction": overlap((p["start"],p["end"]),(g["start"],g["end"]))/(g["end"]-g["start"]+1),
                                "tIoU": float(matrix[i,j])} for j,g in enumerate(truth) if matrix[i,j]>0]
    prob = np.array([a["verb_probabilities"] for a in actions])
    display_verbs = ["take", "put", "open", "close"]
    display_verbs += [v for v in (labels["verb"][i] for i in np.argsort(prob.mean(0))[::-1]) if v not in display_verbs][:2]
    sources = [annotation, folder / "events/predictions.json", folder / "events/action_diagnostics.json",
               folder / "actions/actions.jsonl", folder / "actions/labels.json",
               folder / "detection/detection_report.json"] + event_sources
    return {"recording": session.name, "scene": scene, "target": target, "fps": fps,
            "frame_count": read(folder / "detection/detection_report.json")["expected_frames"],
            "truth": truth, "predictions": rows, "tIoU_matrix": matrix.tolist(), "metrics": threshold_metrics,
            "selected_boundaries": diagnostics["selected_boundaries"],
            "display_verbs": display_verbs,
            "action_heatmap": [[float(a["verb_probabilities"][labels["verb"].index(v)]) for a in actions] for v in display_verbs],
            "action_windows": [{k: a[k] for k in ("window_id", "source_frame_id", "start_source_frame_id",
                                "end_source_frame_id", "relative_seconds", "top_verbs", "top_nouns", "action_status", "action_label")}
                               for a in actions],
            "input_hashes": {str(p): digest(p) for p in sources}}


def style():
    fonts = [Path("/usr/share/fonts/truetype/arphic/uming.ttc"),
             Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")]
    font = next((p for p in fonts if p.exists()), fonts[0])
    pdf_fonttype = 42
    if font.exists():
        # Older matplotlib PDF backends cannot subset a TTC directly.
        from fontTools.ttLib import TTFont
        cached = Path(__file__).resolve().parents[1] / f".cache/plot-fonts/PeekabooComparison-{font.stem}.ttf"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            extracted = TTFont(str(font), fontNumber=0)
            # A unique family prevents findfont from choosing the installed TTC.
            for name in extracted["name"].names:
                replacement = {1: "Peekaboo Comparison CJK", 16: "Peekaboo Comparison CJK",
                               4: "Peekaboo Comparison CJK Regular", 6: "PeekabooComparisonCJK-Regular"}.get(name.nameID)
                if replacement:
                    name.string = replacement.encode(name.getEncoding())
            extracted.save(cached)
        if "glyf" not in TTFont(str(cached)):
            pdf_fonttype = 3
        font = cached
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.unicode_minus": False, "pdf.fonttype": pdf_fonttype})


def timeline(ax, record, triplets=True):
    fps = record["fps"]
    for g in record["truth"]:
        ax.broken_barh([(g["start"] / fps, (g["end"]-g["start"]+1)/fps)], (3.8,.6), facecolors="#2b6cb0")
        ax.text((g["start"]+g["end"])/2/fps,4.14,g["id"],ha="center",va="center",color="white",fontsize=9)
        ax.plot([g["middle"]/fps]*2,[3.83,3.98],color="white",lw=1.4)
    for p in record["predictions"]:
        ax.broken_barh([(p["start"]/fps,(p["end"]-p["start"]+1)/fps)],(2.8,.6),facecolors="#d97706",edgecolors="white",linewidth=.9)
        ax.text((p["start"]+p["end"])/2/fps,3.1,p["id"],ha="center",va="center",color="white",fontsize=9)
    if triplets:
        for y, role, marker, color in [(2.,"pre","o","#475569"),(1.2,"pnr","D","#7e22ce"),(.4,"post","^","#047857")]:
            previous, tier = None, 0
            for p in record["predictions"]:
                f = p["frames"][role]
                tier = 1-tier if previous is not None and abs(f-previous)/fps < .8 else 0
                yy = y + (.06 if tier == 0 else -.06)
                inside = p["start"] <= f <= p["end"]
                ax.scatter(f/fps,yy,s=35,marker=marker,facecolor=color if inside else "white",edgecolor=color,zorder=3)
                ax.annotate(p["id"],(f/fps,yy),xytext=(0,5 if tier == 0 else -12),textcoords="offset points",ha="center",fontsize=7)
                previous = f
    ax.set_yticks([4.1,3.1,2.,1.2,.4],labels=["人工区间 G","预测区间 P","Pre","PNR","Post"])
    ax.set_ylim(-.05,4.65)
    ax.set_xlim(0,record["frame_count"]/fps)
    ax.set_xlabel("视频时间（秒）")
    ax.grid(axis="x",color="#e2e8f0",linewidth=.6)
    ax.set_axisbelow(True)
    ax.set_title(f"{record['scene']} {record['target']}｜人工 {len(record['truth'])} / 预测 {len(record['predictions'])}",loc="left",pad=12)


def heatmap(ax, record):
    times = np.array([a["source_frame_id"] / record["fps"] for a in record["action_windows"]])
    edges = np.r_[0,(times[:-1]+times[1:])/2,record["frame_count"]/record["fps"]]
    mesh = ax.pcolormesh(edges,np.arange(len(record["display_verbs"])+1),np.array(record["action_heatmap"]),
                         cmap="YlGnBu",vmin=0,vmax=1,shading="flat")
    ax.set_yticks(np.arange(len(record["display_verbs"]))+.5,
                  labels=[f"{v} {VERB_CN.get(v,'')}" for v in record["display_verbs"]])
    ax.invert_yaxis()
    ax.set_xlim(0,record["frame_count"]/record["fps"])
    ax.set_xlabel("视频时间（秒）")
    ax.set_title("TSM 动作 softmax（未校准；非分类准确率）",loc="left",pad=12)
    for p in record["predictions"]:
        ax.axvline(p["start"]/record["fps"],color="#64748b",lw=.65,ls="--",alpha=.7)
    return mesh


def render_figures(records, output):
    style()
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(len(records),2,figsize=(14,11.5),gridspec_kw={"width_ratios":[1.12,1]},layout="constrained")
    for i,r in enumerate(records):
        timeline(axes[i,0],r)
        mesh=heatmap(axes[i,1],r)
    cbar=fig.colorbar(mesh,ax=axes[:,1].tolist(),shrink=.55,pad=.025)
    cbar.set_label("动作 softmax 分数")
    fig.suptitle("三段视频：人工事件、自动切分、三帧选择与 TSM 动作\n空心三帧标记＝位于对应预测区间外的上下文；人工条内白线＝人工进行帧",fontsize=13)
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"event-action-overview.{suffix}",dpi=190,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,len(records),figsize=(13.5,4.8),layout="constrained")
    for ax,r in zip(axes,records):
        a=np.array(r["tIoU_matrix"])
        image=ax.imshow(a,aspect="auto",vmin=0,vmax=1,cmap="Blues")
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                value = format(a[i,j], ".4f" if abs(a[i,j]-.3)<.001 else ".3f")
                ax.text(j,i,value,ha="center",va="center",color="white" if a[i,j]>.6 else "#1e293b",fontsize=9)
        for pair in r["metrics"]["0.3"]["pairs"]:
            i=int(pair["prediction"][1:])-1;j=int(pair["truth"][1:])-1
            ax.add_patch(Rectangle((j-.48,i-.48),.96,.96,fill=False,edgecolor="#d97706",lw=2))
        ax.set_xticks(range(len(r["truth"])),[g["id"] for g in r["truth"]])
        ax.set_yticks(range(len(r["predictions"])),[p["id"] for p in r["predictions"]])
        ax.set_xlabel("人工事件")
        ax.set_ylabel("预测事件")
        m=r["metrics"]["0.3"]
        ax.set_title(f"{r['scene']} {r['target']}｜TP={m['TP']}，F1={m['F1']:.3f}")
    fig.colorbar(image,ax=axes.tolist(),shrink=.7,label="区间 tIoU")
    fig.suptitle("事件重叠矩阵｜橙框：tIoU≥0.3 的一对一匹配",fontsize=13)
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"event-iou-matrices.{suffix}",dpi=190,bbox_inches="tight")
    plt.close(fig)


def write_csv(output, records):
    rows=[]
    for r in records:
        for p in r["predictions"]:
            match=p["match_at_0.3"] or {}
            boundary=p["boundary_evidence"]
            rows.append({"scene":r["scene"],"target":r["target"],"prediction":p["id"],
                "start_frame":p["start"],"end_frame":p["end"],"start_seconds":p["start_seconds"],"end_seconds":p["end_seconds"],
                "best_truth":p["best_truth"],"best_tIoU":p["best_tIoU"],"matched_truth_0.3":match.get("truth"),
                "start_error_seconds":match.get("start_error_seconds"),"end_error_seconds":match.get("end_error_seconds"),
                "boundary_reason":p["boundary_reason"],"boundary_decision_frame":";".join(str(b["source_frame_id"]) for b in boundary),
                "boundary_raw_score":";".join(str(b["score"]) for b in boundary),
                "phase_sequence":" > ".join(p["phase_sequence"]),
                "tsm_top_verbs":"; ".join(f"{a['label']}:{a['score']:.8f}" for a in p["tsm_top_verbs"]),
                "tsm_top_nouns":"; ".join(f"{a['label']}:{a['score']:.8f}" for a in p["tsm_top_nouns"]),
                "tsm_candidate_fraction":p["tsm_candidate_fraction"],"contact_object":p["selected_contact_object"],
                **p["frames"],"q_event":p["q_event"],
                "pnr_in_human_event":";".join(p["frames_truth_membership"]["pnr"]),
                "score_methods":";".join(a["estimation_method"] for a in p["scores"])})
    with (output/"predicted-events.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    gt=[{"scene":r["scene"],"target":r["target"],**g} for r in records for g in r["truth"]]
    with (output/"human-events.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(gt[0]));writer.writeheader();writer.writerows(gt)
    windows=[]
    for r in records:
        for a in r["action_windows"]:
            windows.append({"scene":r["scene"],"target":r["target"],"window":a["window_id"],
                "center_frame":a["source_frame_id"],"start_frame":a["start_source_frame_id"],"end_frame":a["end_source_frame_id"],
                "top_verb":a["top_verbs"][0]["label"],"top_verb_score":a["top_verbs"][0]["probability"],
                "top_noun":a["top_nouns"][0]["label"],"top_noun_score":a["top_nouns"][0]["probability"],
                "accepted_action":a["action_label"],"status":a["action_status"]})
    with (output/"action-windows.csv").open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(windows[0]));writer.writeheader();writer.writerows(windows)


def write_report(output, report):
    records=report["recordings"]
    lines=["# 三段事件切分、动作分类与三帧对照（2026-11-18）", "",
           "人工真值来自原始录制目录中的 `*_三帧.md`，各 6 个，共 18 个；预测来自完整运行 v4，共 5/5/8 个。模型与阈值未重跑、未按本次结果调参。", "",
           "## 事件匹配结果", "",
           "采用源帧闭区间 tIoU=交集帧数/并集帧数。在每段视频内一对一匹配：先最大化达到阈值的匹配数量，再最大化总 tIoU。主表用先前讨论的 0.3，同时报告 0.5/0.7。", "",
           "| 录制 | 人工/预测 | TP/FP/FN @0.3 | Precision | Recall | F1 | 未匹配人工 | 未匹配预测 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in records:
        m=r["metrics"]["0.3"]
        lines.append(f"| {r['scene']} {r['target']} | {len(r['truth'])}/{len(r['predictions'])} | {m['TP']}/{m['FP']}/{m['FN']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['F1']:.3f} | {', '.join(m['unmatched_truth'])} | {', '.join(m['unmatched_predictions'])} |")
    lines += ["", "| tIoU 阈值 | 总 TP/FP/FN | Micro-P | Micro-R | Micro-F1 |", "| --- | --- | --- | --- | --- |"]
    for t,m in report["micro_metrics"].items():
        lines.append(f"| {t} | {m['TP']}/{m['FP']}/{m['FN']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['F1']:.3f} |")
    lines += ["", "FP/FN 是在给定 tIoU 门槛下未匹配的预测/人工区间；一个人工事件可能已被预测部分覆盖，却因区间太短或延伸太长未匹配。不能把所有 FN 都直接解释为完全没有检测到动作。", "",
              "![事件、三帧与动作](figures/event-action-overview.png)", "",
              "![事件重叠矩阵](figures/event-iou-matrices.png)", "",
              "## 主要偏差", "",
              "- S03：P1=132–338 覆盖 G1=135–216 和 G2=241–316；P2 又同时覆盖 G3 和 G4 的开头。P4=581–683 覆盖 G5 及 G6 的前段。P5=725–802 位于全部人工事件之后。数量少 1 并不能概括这些合并、碎片和额外区间。",
              "- S07：P1=105–262 合并 G1/G2；P2=269–391 包含 G3 和 G4 开头，P3 仅覆盖 G4 后段。P5 包含完整 G6，但结束延至视频末尾 650，tIoU 只有 0.270，低于 0.3。",
              "- S09：G3 被 P3/P4 分拆，G5 被 P6/P7 分拆；P2 同时覆盖 G1 后段与 G2，P6 还跨越 G4/G5。P1 与 G1 的 tIoU 为 0.297，接近但低于 0.3。", "",
              "- 切分和三帧还会发生错位：S03 P2 的最佳区间匹配为 G3，但 PNR=432 位于 G4；S03 P4 匹配 G5，但 PNR=676 位于 G6。S07 P5 的 PNR=530 早于预测起点 540，属于 G5；S09 P2 匹配 G2，但 PNR=113 位于 G1。这些是按人工区间得到的时间归属，不是已完成人工视觉验收。", "",
              "## 数值口径", "",
              "- 动作表的分数是在该预测区间内、以动作窗口中心帧为准，对全部 TSM softmax 向量取均值后的 Top-3；包括未通过动作门槛的窗口。候选比例另列。它不是事件正确概率。",
              "- 阶段序列来自已保存的动作阶段诊断，按阶段开始时间排列；原始 `action_phases` 是无序集合，未将其字母排序误当时序。热图固定展示 take/put/open/close，另加全片平均分最高的两个其余动词。",
              "- 原始人工标注没有动作类别、操作对象或经过复核的 PNR 真值，因此不报告动作分类准确率。人工“进行帧”仅作时间参考，不能默认等于模型 PNR。TSM noun 与 YOLO 接触对象是两个独立输出。",
              "- 三帧可取已登记的前后文，所以有些帧落在预测区间之外，图中用空心标记。部分三帧跨度还会跨人工事件；这是一项需复核的现象，不据此直接认定语义正确。",
              "- 当前输出没有经过校准的事件置信度。边界诊断 `score` 可能来自动作阶段 softmax 或视觉差异，不同来源不能视为同一种概率；q 是充分性估计，与 tIoU、动作分数也不同。", ""]
    for r in records:
        lines += [f"## {r['scene']} {r['target']}／{r['recording'][-6:]}", "",
                  "### 原始人工事件", "", "| ID | 开始帧 | 进行帧 | 结束帧 | 时间范围（秒） |", "| --- | --- | --- | --- | --- |"]
        for g in r["truth"]:
            lines.append(f"| {g['id']} | {g['start']} | {g['middle']} | {g['end']} | {g['start']/r['fps']:.2f}–{g['end']/r['fps']:.2f} |")
        lines += ["", "### 预测切分与三帧", "", "| ID | 预测起止帧（秒） | 最佳人工/tIoU | 一对一匹配 @0.3 | 起/止偏移秒 | Pre/PNR/Post | PNR时间归属 | 事件 q（来源） |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in r["predictions"]:
            m=p["match_at_0.3"]
            error=f"{m['start_error_seconds']:+.2f}/{m['end_error_seconds']:+.2f}" if m else "—"
            frame="/".join(str(p["frames"][role]) for role in ("pre","pnr","post"))
            methods="、".join(dict.fromkeys(METHOD_CN[a["estimation_method"]] for a in p["scores"]))
            pnr_membership=", ".join(p["frames_truth_membership"]["pnr"]) or "人工区间外"
            lines.append(f"| {p['id']} | {p['start']}–{p['end']} ({p['start_seconds']:.2f}–{p['end_seconds']:.2f}) | {p['best_truth'] or '无'}/{p['best_tIoU']:.4f} | {m['truth'] if m else '未匹配'} | {error} | {frame} | {pnr_membership} | {p['q_event']:.3f}（{methods}） |")
        lines += ["", "正偏移表示预测更晚，负偏移表示预测更早；偏移仅对成功匹配的区间计算。", "", "### 动作分类与接触对象", "",
                  "| ID | TSM 动词 Top-3 均分 | 阶段时序（预测） | 候选窗口比例 | TSM 名词 Top-1 | YOLO 接触目标 |", "| --- | --- | --- | --- | --- | --- |"]
        for p in r["predictions"]:
            verbs="；".join(f"{a['label']} {a['score']:.3f}" for a in p["tsm_top_verbs"])
            phases=" → ".join(PHASE_CN.get(v,v) for v in p["phase_sequence"]) or "无可靠阶段"
            noun=p["tsm_top_nouns"][0]
            lines.append(f"| {p['id']} | {verbs} | {phases} | {p['tsm_candidate_fraction']:.1%} | {noun['label']} {noun['score']:.3f} | {p['selected_contact_object'] or 'unknown'} |")
        lines += ["", "### 边界依据与原始分数", "", "| 预测 ID | 切分依据 | 规则决策帧 | 原始分数（未校准） |", "| --- | --- | --- | --- |"]
        for p in r["predictions"]:
            evidence=p["boundary_evidence"]
            decisions=", ".join(str(b["source_frame_id"]) for b in evidence) or "无分数：接触段起点"
            raw_scores=", ".join(format(b["score"], ".6f") for b in evidence) or "—"
            lines.append(f"| {p['id']} | {REASON_CN.get(p['boundary_reason'],p['boundary_reason'])} | {decisions} | {raw_scores} |")
        lines += ["", "规则决策帧与实际区间起点可能不同：例如决定在间隙切分后，下一段从首个实际接触帧开始。", ""]
    lines += ["## 文件与复现", "", "- [完整对照数据与输入哈希](comparison.json)",
              "- [预测事件、动作、边界、三帧、q 的 CSV](predicted-events.csv)",
              "- [人工事件 CSV](human-events.csv)", "- [逐窗口动作分类 CSV](action-windows.csv)",
              "- [时间轴与动作热图 PDF](figures/event-action-overview.pdf)", "- [tIoU 矩阵 PDF](figures/event-iou-matrices.pdf)", "",
              "```bash", "pip install matplotlib fonttools", "python3 scripts/compare_event_segments.py \\",
              "  --run-root /data/storage/peekaboo-runs/full_numeric_20261118_v4 \\",
              "  --output docs/event-comparison", "```", "", "这是对已保存完整运行的只读分析，没有修改模型输出，也没有使用人工区间重新生成预测。"]
    (output/"README.md").write_text("\n".join(lines)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    run=Path(args.run_root).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    records=[build_record(Path(s),run) for s in read(run/"run_manifest.json")["sessions"]]
    total_pred=sum(len(r["predictions"]) for r in records);total_gt=sum(len(r["truth"]) for r in records)
    micro={str(t): metrics(sum(r["metrics"][str(t)]["TP"] for r in records),total_pred,total_gt) for t in THRESHOLDS}
    report={"run_root":str(run),"intervals":"inclusive zero-based source frame IDs", "thresholds":THRESHOLDS,
            "analysis_script_sha256":digest(__file__),
            "matching":"maximize eligible pair count, then summed tIoU; one-to-one within each recording",
            "human_middle_is_not_validated_pnr":True,"action_classification_ground_truth_available":False,
            "micro_metrics":micro,"recordings":records}
    write_json(output/"comparison.json",report);write_csv(output,records);render_figures(records,output);write_report(output,report)
    print(json.dumps({"output":str(output),"human_events":total_gt,"predicted_events":total_pred,"micro_metrics":micro},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
