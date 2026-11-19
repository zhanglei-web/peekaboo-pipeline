"""Select explicitly curated strengths and limitations from the frozen 20-recording test."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np

from compare_event_segments import (digest, match_intervals, metrics, read, style,
                                    temporal_iou, write_json)
from plot_multiscene_holdout import VERBS


CASES = [
    dict(key="184735", label="A", kind="优势", title="区间定位较准，保留多次操作",
         focus=[834, 969], note="G8 / P8：tIoU = 0.934；全段 8 个真值中匹配 6 个。",
         pair=["P8", "G8"]),
    dict(key="191840", label="B", kind="优势", title="减少碎片化，前三次操作逐一对应",
         focus=[88, 347], note="P1–P3 对应 G1–G3：tIoU = 0.691 / 0.675 / 0.724。",
         pairs=[["P1", "G1"], ["P2", "G2"], ["P3", "G3"]]),
    dict(key="185031", label="C", kind="优势", title="连续操作可分离，多个边界与人工接近",
         focus=[156, 607], note="P3–P7 连续匹配 G2–G6；G5 / P6 的 tIoU = 0.817。",
         pairs=[[f"P{i}", f"G{i-1}"] for i in range(3, 8)]),
    dict(key="184050", label="D", kind="局限", title="欠切分：多个真实事件被合并",
         focus=[512, 877], note="P4 = 522–877，跨越 G5–G9；全段 9 个真值仅匹配 2 个。",
         merge_prediction="P4", merge_truth=[f"G{i}" for i in range(5, 10)]),
    dict(key="185159", label="E", kind="局限", title="过切分：一个真实事件被切碎",
         focus=[330, 509], note="G3 = 341–494，被 P3 / P4 分割；P4 在主门槛下未匹配。",
         split_truth="G3", split_predictions=["P3", "P4"]),
]


def evaluated(rows, truth):
    matrix = np.array([[temporal_iou((p["start"], p["end"]), (g["start"], g["end"]))
                       for g in truth] for p in rows])
    pairs = match_intervals(matrix, .5)
    return {**metrics(len(pairs), len(rows), len(truth)),
            "pairs": [{"prediction": rows[i]["id"], "truth": truth[j]["id"],
                       "tIoU": float(matrix[i, j])} for i, j in pairs]}


def collect(source):
    report = read(source)
    # Verify full-cohort results before selecting examples; no metric is computed on crops.
    for rec in report["recordings"]:
        checked = evaluated(rec["predictions"], rec["truth"])
        for key in ("TP", "FP", "FN", "precision", "recall", "F1"):
            assert abs(checked[key] - rec["metrics"]["0.5"][key]) < 1e-12
    selected = []
    for case in CASES:
        found = [r for r in report["recordings"] if r["recording"].endswith(case["key"])]
        assert len(found) == 1
        rec = found[0]
        actual = evaluated(rec["predictions"], rec["truth"])
        visual = evaluated(rec["visual_predictions"], rec["truth"])
        for p, g in case.get("pairs", [case["pair"]] if "pair" in case else []):
            assert any(x["prediction"] == p and x["truth"] == g for x in actual["pairs"])
        if "merge_prediction" in case:
            pred = next(x for x in rec["predictions"] if x["id"] == case["merge_prediction"])
            assert all(temporal_iou((pred["start"], pred["end"]), (g["start"], g["end"])) > 0
                       for g in rec["truth"] if g["id"] in case["merge_truth"])
        if "split_truth" in case:
            gt = next(x for x in rec["truth"] if x["id"] == case["split_truth"])
            assert all(temporal_iou((pred["start"], pred["end"]), (gt["start"], gt["end"])) > 0
                       for pred in rec["predictions"] if pred["id"] in case["split_predictions"])
            assert not any(x["prediction"] == "P4" for x in actual["pairs"])
        selected.append({"case": case, "record": rec, "fusion": actual, "visual": visual})
    return report, selected


def draw_case(left, right, item):
    case, rec = item["case"], item["record"]
    accent = "#047857" if case["kind"] == "优势" else "#be123c"
    for ax in (left, right):
        ax.axvspan(*case["focus"], color=accent, alpha=.07, zorder=0)
        for x in case["focus"]: ax.axvline(x, color=accent, lw=.7, ls=":", alpha=.65)
        ax.set_xlim(0, rec["frame_count"] - 1)
        ax.grid(axis="x", color="#e2e8f0", lw=.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.set_xlabel("源帧号（完整录制）", fontsize=9)
    matched = {x["prediction"] for x in item["fusion"]["pairs"]}
    for rows, y, category in [(rec["truth"], 2., "truth"),
                              (rec["visual_predictions"], 1., "visual"),
                              (rec["predictions"], 0., "fusion")]:
        for row in rows:
            color = {"truth": "#475569", "visual": "#94a3b8"}.get(category)
            if category == "fusion": color = "#059669" if row["id"] in matched else "#d97706"
            left.broken_barh([(row["start"], row["end"]-row["start"]+1)], (y-.24, .48),
                             facecolors=color, edgecolors="white", linewidth=.9, zorder=3)
            left.text((row["start"]+row["end"])/2, y, row["id"], color="white",
                      ha="center", va="center", fontsize=8, zorder=4)
    left.set_yticks([2, 1, 0], labels=["人工真值 G", "仅视觉 V", "TSM＋接触 P"])
    left.set_ylim(-.48, 2.55)
    left.set_title(f"{case['label']}  {case['kind']}｜{rec['scene']} {rec['target']} / {case['key']}\n"
                   f"{case['title']}", loc="left", fontsize=11, color=accent, pad=12)
    left.text(0, -.33, case["note"], transform=left.transAxes, fontsize=9, color=accent)
    for verb, cn, color in VERBS:
        right.plot(rec["action_frames"], rec["action_curves"][verb], lw=1.25,
                   color=color, label=f"{verb} {cn}")
    right.set_ylim(-.025, 1.03); right.set_yticks([0, .5, 1.])
    right.set_ylabel("动作概率", fontsize=9)
    right.set_title(f"全段 F1@0.5：仅视觉 {item['visual']['F1']:.3f} → 融合 {item['fusion']['F1']:.3f}\n"
                    f"融合 TP / 预测 / 真值 = {item['fusion']['TP']} / {len(rec['predictions'])} / {len(rec['truth'])}",
                    loc="left", fontsize=11, pad=12)
    right.text(0, -.33, "四条原始 softmax 曲线；未平滑，未新增人工动作类别标签。",
               transform=right.transAxes, fontsize=8.5, color="#64748b")


def figure(items, title):
    height = 3.05*len(items)+2.8
    fig, axes = plt.subplots(len(items), 2, figsize=(17, height), squeeze=False)
    fig.subplots_adjust(left=.09, right=.985, top=1-1.7/height,
                        bottom=1.3/height, hspace=1.35, wspace=.22)
    for row, item in zip(axes, items): draw_case(*row, item)
    fig.suptitle(title, fontsize=17, y=.982, fontweight="bold")
    handles = [Patch(facecolor="#475569", label="人工区间"),
               Patch(facecolor="#94a3b8", label="仅视觉基线"),
               Patch(facecolor="#059669", label="融合预测：tIoU≥0.5 匹配"),
               Patch(facecolor="#d97706", label="融合预测：主门槛下未匹配")]
    handles += [plt.Line2D([], [], color=color, lw=1.5, label=f"{v}") for v, _, color in VERBS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53,1-.65/height), ncol=8,
               frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.3)
    fig.text(.09, .017, "有目的挑选的定性案例；阴影标出讨论位置，所有时间轴与指标均覆盖完整录制。\n"
             "20段总体主指标 F1@0.5 = 0.448；图中案例不代表总体均值，动作概率峰值不等于分类准确率。",
             fontsize=10, color="#475569")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="docs/multiscene-holdout-curves/comparison.json")
    parser.add_argument("--output", default="docs/representative-cases")
    args = parser.parse_args(); source = Path(args.source).resolve(); output = Path(args.output).resolve()
    source_hash = digest(source); report, items = collect(source)
    output.mkdir(parents=True, exist_ok=True); style()
    with PdfPages(output/"representative-cases.pdf") as book:
        for stem, subset, title in [
            ("representative-cases", items, "多场景事件预测｜三个优势案例与两个局限案例"),
            ("strengths", items[:3], "优势案例｜区间定位、减少碎片化与连续操作切分"),
            ("limitations", items[3:], "局限案例｜欠切分合并与过切分碎片")]:
            fig = figure(subset, title)
            fig.savefig(output/f"{stem}.png", dpi=180)
            if stem != "representative-cases": fig.savefig(output/f"{stem}.pdf")
            book.savefig(fig); plt.close(fig)
            print(f"Rendered {stem}", flush=True)
    assert digest(source) == source_hash
    write_json(output/"selection.json", {"source":str(source), "source_sha256":source_hash,
               "script_sha256":digest(__file__), "selection":"curated qualitative strengths and limitations after evaluation",
               "primary_threshold":.5, "cohort_event_metrics":report["event_metrics"],
               "cases":[{"case":x["case"], "recording":x["record"]["recording"],
                         "scene":x["record"]["scene"], "target":x["record"]["target"],
                         "fusion":x["fusion"], "visual":x["visual"]} for x in items]})
    lines = ["# 多场景代表案例：优势与局限", "",
        "从固定的六场景20段测试中，有目的挑选三个优势案例及两个局限案例用于定性说明。"
        "没有重新选择测试集、调整门槛、重跑模型或更换预测。主门槛统一为 tIoU≥0.5，所有指标均按完整录制计算。", "",
        "## 推荐展示方式", "", "正文用前三个案例展示方法能力，并保留后两个案例解释边界；全量结果链接保留供核查。"
        "图中人工区间、仅视觉基线、融合预测逐行对齐，右侧展示原始动作概率折线，阴影圈出讨论区间。", "",
        "![三个优势与两个局限](representative-cases.png)", "",
        "| 案例 | 场景 / 目标 | 可说明的能力或局限 | 全段 TP/预测/真值 | 仅视觉 F1@0.5 | 融合 F1@0.5 |", "| --- | --- | --- | --- | --- | --- |"]
    for x in items:
        c,r,m=x["case"],x["record"],x["fusion"]
        lines.append(f"| {c['label']} | {r['scene']} / {r['target']}（{c['key']}） | {c['title']} | {m['TP']}/{len(r['predictions'])}/{len(r['truth'])} | {x['visual']['F1']:.3f} | {m['F1']:.3f} |")
    lines += ["", "## 图中应强调的位置", ""]
    for x in items: lines.append(f"- **{x['case']['label']}：**{x['case']['note']}")
    lines += ["", "## 解释范围", "",
        "这些图支持事件分段与时间定位的定性分析。前三个案例的融合 F1 均高于相同录制的仅视觉基线；"
        "后两个案例显示提升后仍存在的合并或切碎问题。仅凭这些曲线不能认定动作分类、PNR 语义或充分性计算已准确。", "",
        "绿色/橙色仅表示统一一对一事件匹配下是否达到 tIoU≥0.5，橙色不等于没有发生动作。"
        "右侧 softmax 为模型输出，没有人工动作类别真值，不报告动作准确率。"
        "该历史20段批次采用规则选三帧，没有 VLM 选帧；本图省略三帧标记以集中展示事件与动作线索。", "",
        "20段总体 F1@0.5=0.448（辅助 F1@0.3=0.648），不可用精选案例代替总体性能。"
        "[全部20段结果](../multiscene-holdout-curves/README.md)。", "",
        "## 下载", "", "- [三页 PDF：合图、优势、局限](representative-cases.pdf)",
        "- [优势图 PNG](strengths.png) / [PDF](strengths.pdf)",
        "- [局限图 PNG](limitations.png) / [PDF](limitations.pdf)",
        "- [案例选择、输入哈希和原始匹配指标](selection.json)", "",
        "```bash", "python3 scripts/plot_representative_cases.py", "```", ""]
    (output/"README.md").write_text("\n".join(lines))
    print(f"Validated all {len(report['recordings'])} recordings; selected {len(items)} cases.", flush=True)


if __name__ == "__main__": main()
