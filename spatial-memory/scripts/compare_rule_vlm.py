"""Analyze an actual legacy-rule rerun against the saved Qwen triplets.

Raw video contact sheets stay in the local selection run; public outputs contain
only timing plots, numeric tables and provenance hashes.
"""
import argparse
import csv
from html import escape
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_event_segments import read, jsonl, digest, write_json, style


def inside(frame, interval):
    return isinstance(frame, int) and interval[0] <= frame <= interval[1]


def diagnose(records, method, threshold):
    matched = pre = n = post = all_three = 0
    for rec in records:
        truth = {g["id"]: g for g in rec["truth"]}
        pairs = {p["prediction"]: p["truth"] for p in rec["event_metrics"][threshold]["pairs"]}
        for row in rec["events"]:
            if row["id"] not in pairs:
                continue
            g = truth[pairs[row["id"]]]; interval = (g["start"], g["end"])
            values = [inside(row[method][role], interval) for role in ("pre", "pnr", "post")]
            matched += 1; pre += values[0]; n += values[1]; post += values[2]; all_three += all(values)
    return {"matched_events": matched, "pre_in_matched_truth": pre, "pnr_in_matched_truth": n,
            "post_in_matched_truth": post, "all_three_in_matched_truth": all_three}


def collect(reference, selections):
    run = Path(reference["run_root"])
    records = []
    for rec in reference["recordings"]:
        events = []
        for pred in rec["predictions"]:
            folder = run / rec["recording"] / "event_runs" / pred["event_id"]
            receipt_path = selections / rec["recording"] / pred["event_id"] / "receipt.json"
            receipt = read(receipt_path); rule = receipt["result"]
            for path, value in receipt["input_hashes"].items():
                assert digest(path) == value, "Shared input changed after the actual rule run"
            assert receipt["exit_code"] == 0 and rule["vlm_used"] is False and rule["model_calls"] == 0
            frames = dict(zip(("pre", "pnr", "post"), (rule[k] for k in ("p_frame_id", "n_frame_id", "r_frame_id"))))
            packets = jsonl(folder / "rank/packets.jsonl")
            scores = jsonl(folder / "rank/feature_scores.jsonl")
            assert len(packets) == 1 and rule["status"] == "complete"
            target = packets[0]["target"]
            assert [frames["pre"], frames["post"]] == [target["start_source_frame_id"], target["end_source_frame_id"]]
            assert {a["source_frame_id"] for a in scores} == set(range(frames["pre"], frames["post"] + 1))
            valid = [a for a in scores if a.get("score_valid", True)]
            maximum = max(valid, key=lambda a: (a["feature_change"], -a["source_frame_id"]))
            assert frames["pnr"] == maximum["source_frame_id"]
            assert abs(rule["maximum_feature_change"] - maximum["feature_change"]) < 1e-12
            selected = read(folder / "final_selection.json")
            assert pred["frames"] == {role: selected[role] for role in ("pre", "pnr", "post")}
            displayed = {f["source_frame_id"] for f in read(folder / "final_candidates.json")["candidates"]}
            frame_owners = {method: {role: [g["id"] for g in rec["truth"] if inside(f, (g["start"],g["end"]))]
                                      for role,f in choice.items()} for method,choice in [("rule",frames),("vlm",pred["frames"]) ]}
            events.append({"id":pred["id"], "event_id":pred["event_id"], "start":pred["start"], "end":pred["end"],
                "rule":frames, "vlm":pred["frames"], "owners":frame_owners,
                "match_at_0.3":pred["match_at_0.3"], "best_truth":pred["best_truth"], "best_tIoU":pred["best_tIoU"],
                "maximum_feature_change":maximum["feature_change"], "cluster":target["cluster_id"],
                "contact_object":target["object_category"], "rule_status":rule["status"],
                "rule_equals_endpoint":frames["pnr"] in (frames["pre"],frames["post"]),
                "rule_triplet_shown_to_vlm":set(frames.values()).issubset(displayed),
                "pnr_shift_vlm_minus_rule_frames":pred["frames"]["pnr"]-frames["pnr"],
                "input_hashes":{str(p):digest(p) for p in [receipt_path, folder / "rank/feature_scores.jsonl",
                    folder / "rank/packets.jsonl", folder / "final_selection.json", folder / "final_candidates.json"]}})
        records.append({"recording":rec["recording"], "scene":rec["scene"], "target":rec["target"],
            "fps":rec["fps"], "frame_count":rec["frame_count"], "truth":rec["truth"], "events":events,
            "event_metrics":rec["metrics"], "video":read(run / rec["recording"] / "detection/manifest.json")["source"]["video"]})
    return records


def all_event_stats(records, method):
    result = {"events":0, "complete_triplets":0, "three_distinct_frames":0, "all_three_in_prediction":0,
              "pnr_in_prediction":0, "pnr_in_any_human_interval":0, "n_equals_endpoint":0}
    for rec in records:
        for row in rec["events"]:
            f = row[method]; values = list(f.values()); interval = (row["start"],row["end"])
            result["events"] += 1
            result["complete_triplets"] += all(isinstance(v,int) for v in values)
            result["three_distinct_frames"] += len(set(values)) == 3
            result["all_three_in_prediction"] += all(inside(v,interval) for v in values)
            result["pnr_in_prediction"] += inside(f["pnr"],interval)
            result["pnr_in_any_human_interval"] += bool(row["owners"][method]["pnr"])
            result["n_equals_endpoint"] += f["pnr"] in (f["pre"],f["post"])
    return result


def make_contact_sheets(records, destination):
    sheets = destination / "contact_sheets"; sheets.mkdir(exist_ok=True)
    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font = ImageFont.truetype(font_path,22); small = ImageFont.truetype(font_path,18)
    gallery = ["<!doctype html><html lang='zh'><meta charset='utf-8'><title>规则与 VLM 三帧原图对照</title>",
               "<style>body{max-width:1400px;margin:24px auto;font-family:sans-serif;background:#f5f5f5;color:#202020}img{width:100%;height:auto}section{margin-bottom:28px}h1{font-size:24px}</style>",
               "<h1>18 组原视频三帧对照：上排规则，下排 Qwen3-VL-8B-Instruct</h1>",
               "<p>全部事件均列出，不按观感筛选。G 为已有人工时间区间；图上归属不是人工语义裁决。</p>"]
    for rec in records:
        wanted = sorted({v for row in rec["events"] for method in ("rule","vlm") for v in row[method].values()})
        cap = cv2.VideoCapture(rec["video"]); images={}
        try:
            for fid in wanted:
                cap.set(cv2.CAP_PROP_POS_FRAMES,fid);ok,bgr=cap.read()
                if not ok: raise ValueError(f"Cannot decode actual frame {fid}")
                images[fid]=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        for row in rec["events"]:
            canvas=Image.new("RGB",(1380,810),"#f4f4f4");draw=ImageDraw.Draw(canvas)
            match=row["match_at_0.3"]
            gt=next((g for g in rec["truth"] if match and g["id"]==match["truth"]),None)
            gt_label=f"匹配 {gt['id']}={gt['start']}–{gt['end']}" if gt else "未在 tIoU≥0.3 下匹配"
            draw.text((16,6),f"{rec['scene']} {rec['target']} | {row['id']} 预测 {row['start']}–{row['end']} | {gt_label}",font=font,fill="#111111")
            for index,method in enumerate(("rule","vlm")):
                y=52+index*375
                name="原规则：接触簇首尾 + 簇内最大空间变化" if method=="rule" else "当前 Qwen3-VL-8B-Instruct：三帧自由选择"
                draw.text((16,y),name,font=font,fill="#9a4a00" if method=="rule" else "#6721a3")
                for col,role in enumerate(("pre","pnr","post")):
                    fid=row[method][role];x=12+col*460
                    owner=",".join(row["owners"][method][role]) or "人工区间外"
                    draw.text((x,y+35),f"{role.upper()}  F{fid} | {owner}",font=small,fill="#111111")
                    thumb=images[fid].copy();thumb.thumbnail((448,300))
                    canvas.paste(thumb,(x+(448-thumb.width)//2,y+65+(300-thumb.height)//2))
            path=sheets/f"{rec['scene']}_{row['id']}.jpg";canvas.save(path,quality=94)
            row["local_contact_sheet"]=str(path)
            gallery.append(f"<section><h2>{escape(rec['scene']+' '+row['id'])}</h2><img loading='lazy' src='contact_sheets/{path.name}' alt='规则和 VLM 三帧'></section>")
    gallery.append("</html>");(destination/"gallery.html").write_text("\n".join(gallery))


def plot(records, output):
    style();fig,axes=plt.subplots(len(records),1,figsize=(13,13),layout="constrained")
    colors={"rule":"#c56b08","vlm":"#7e22ce"}
    for ax,rec in zip(axes,records):
        truth={g["id"]:g for g in rec["truth"]};fps=rec["fps"]
        for i,row in enumerate(rec["events"]):
            match=row["match_at_0.3"]
            if match:
                g=truth[match["truth"]]
                ax.broken_barh([(g["start"]/fps,(g["end"]-g["start"]+1)/fps)],(i-.42,.84),facecolors="#e2edf7")
            for method,dy in (("rule",-.15),("vlm",.15)):
                frames=row[method];y=i+dy
                ax.plot([frames["pre"]/fps,frames["post"]/fps],[y,y],color=colors[method],alpha=.75,lw=1.5)
                for role,marker in (("pre","o"),("pnr","D"),("post","^")):
                    # Offset only vertically when old N=Post; never move time.
                    yy=y+(.055 if role=="pnr" and frames["pnr"]==frames["post"] else 0)
                    ax.scatter(frames[role]/fps,yy,s=40 if role=="pnr" else 28,marker=marker,
                               edgecolor=colors[method],facecolor=colors[method] if role=="pnr" else "white",zorder=4)
        ax.set_yticks(range(len(rec["events"])),[f"{r['id']} → {r['match_at_0.3']['truth'] if r['match_at_0.3'] else '未匹配'}" for r in rec["events"]])
        duration=rec["frame_count"]/fps
        ax.set_xlim(0,duration+max(.4,duration*.02));ax.set_ylim(len(rec["events"])-.5,-.5)
        ax.set_xlabel("原视频时间（秒）");ax.grid(axis="x",color="#e5e7eb",lw=.6);ax.set_axisbelow(True)
        ax.set_title(f"{rec['scene']} {rec['target']}｜固定相同预测区间与特征；仅替换选帧",loc="left",pad=12)
    handles=[Line2D([],[],color=colors[m],label=l,lw=2) for m,l in [("rule","原规则"),("vlm","当前 VLM")]]
    handles += [Line2D([],[],color="#444444",marker=marker,markerfacecolor="white" if role!="PNR" else "#444444",lw=0,label=role)
                for role,marker in [("Pre","o"),("PNR","D"),("Post","^")]]
    handles += [Patch(facecolor="#e2edf7",label="一对一匹配的人工区间（tIoU≥0.3）")]
    fig.legend(handles=handles,loc="outside upper center" if tuple(int(s) for s in plt.matplotlib.__version__.split('.')[:2]) >= (3,7) else "upper center",ncol=3,bbox_to_anchor=(.5,1.045),frameon=False)
    for suffix in ("png","pdf"):fig.savefig(output/f"rule-vlm-triplets.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def write_outputs(report, output):
    write_json(output/"comparison.json",report)
    rows=[]
    for rec in report["recordings"]:
        for row in rec["events"]:
            rows.append({"scene":rec["scene"],"event":row["id"],"predicted_start":row["start"],"predicted_end":row["end"],
                "matched_truth_0.3":(row["match_at_0.3"] or {}).get("truth"),
                **{f"{method}_{role}":v for method in ("rule","vlm") for role,v in row[method].items()},
                "rule_pnr_human_owner":",".join(row["owners"]["rule"]["pnr"]),
                "vlm_pnr_human_owner":",".join(row["owners"]["vlm"]["pnr"]),
                "rule_maximum_feature_change":row["maximum_feature_change"],"local_contact_sheet":row["local_contact_sheet"]})
            rows[-1]["pnr_shift_vlm_minus_rule_frames"]=row["pnr_shift_vlm_minus_rule_frames"]
    with (output/"triplets.csv").open("w",newline="",encoding="utf-8-sig") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    stats=report["all_events"];diagnostics=report["matched_diagnostics"]
    lines=["# 原规则重新选帧 vs 当前 Qwen 三帧（2026-11-18）","",
           "实际调用原 `pnr select` 处理本次 v4 的全部 18 个事件，不是只读取旧历史选帧表。共享当前检测、接触簇和所有逐帧空间变化分数，不重新切分事件、不重新调用 VLM。18 个命令均成功，规则选择器模型调用为 0。","",
           "## 控制条件与角色定义","",
           "- 原规则：Pre=有效接触簇首帧，Post=簇末帧，PNR=全部有效簇帧中空间变化分数最大者；同分取较早帧，允许 PNR 与端点重合。严格照旧实现，没有强行改成三张不同帧。",
           "- 当前 VLM：复用 Qwen3-VL-8B-Instruct 的原始最终结果，从事件与前后文候选中选择语义 Pre/PNR/Post，要求严格递增。",
           "- 两者共享事件、检测、簇与特征；搜索范围与角色约束存在上述差别，所以这是实际方法对照，不是完全等搜索空间的模型消融。",
           f"- 原规则的三个源帧在当前 VLM 实际展示集合中均可找到的事件数为 {report['rule_triplet_shown_to_vlm_count']}/18；VLM 并非没有看到这些规则候选。",
           "- 事件切分在选帧之前完成，因此两种选帧的切分 F1 都仍为 0.611@0.3、0.444@0.5，不能用这些 F1 判断选帧谁更好。","",
           "## 时间诊断（不是语义准确率）","",
           "| 全部 18 个预测事件 | 原规则 | 当前 VLM |","| --- | --- | --- |"]
    for key,label in [("complete_triplets","产出三个角色引用"),("three_distinct_frames","三个不同源帧"),
                      ("all_three_in_prediction","三个帧均在预测事件内"),("pnr_in_prediction","PNR 在预测事件内"),
                      ("n_equals_endpoint","PNR 与某个端点重合")]:
        lines.append(f"| {label} | {stats['rule'][key]}/18 | {stats['vlm'][key]}/18 |")
    lines += ["","规则在预测区间内是搜索范围约束的结果；VLM 允许使用前后文。该表不能把越出预测边界自动判为错误，也不能把区间内自动判为语义正确。","",
              "| 固定事件匹配门槛 | 诊断项 | 原规则 | 当前 VLM |","| --- | --- | --- | --- |"]
    for threshold,data in diagnostics.items():
        for key,label in [("pre_in_matched_truth","Pre 在匹配人工区间内"),("pnr_in_matched_truth","PNR 在匹配人工区间内"),
                          ("post_in_matched_truth","Post 在匹配人工区间内"),("all_three_in_matched_truth","三帧均在匹配人工区间内")]:
            n=data['rule']['matched_events'];lines.append(f"| {threshold} | {label} | {data['rule'][key]}/{n} | {data['vlm'][key]}/{n} |")
    lines += ["","人工开始/结束是搜索事件范围，人工进行帧只是过程采样，缺少可接受 PNR 区间及三帧语义裁决。上述包含关系不能当作 PNR 最优率或证据充分率。","",
              "![规则与 VLM 时间轴](rule-vlm-triplets.png)","",
              "## 逐事件结果","",
              f"本批 VLM 的 PNR 在 {report['pnr_shift']['earlier_count']}/18 组中比规则更早，平均提前 {-report['pnr_shift']['mean_seconds']:.2f} 秒，中位提前 {-report['pnr_shift']['median_seconds']:.2f} 秒。规则偏向累计变化最大时刻，VLM 的时刻更早；这项差异本身不能判定哪一帧语义更优。", "",
              "G 为原人工事件编号；“N归属”仅按时间判断。匹配使用上次同一份一对一 tIoU≥0.3 结果。","",
              "| 录制/事件 | 预测区间 | 匹配人工 | 原规则 Pre/N/Post | 原规则 N归属 | VLM Pre/N/Post | VLM N归属 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for rec in report["recordings"]:
        for row in rec["events"]:
            r="/".join(str(row['rule'][role]) for role in ('pre','pnr','post'));v="/".join(str(row['vlm'][role]) for role in ('pre','pnr','post'))
            ro=','.join(row['owners']['rule']['pnr']) or '人工区间外';vo=','.join(row['owners']['vlm']['pnr']) or '人工区间外'
            match=(row['match_at_0.3'] or {}).get('truth','未匹配')
            lines.append(f"| {rec['scene']} {row['id']} | {row['start']}–{row['end']} | {match} | {r} | {ro} | {v} | {vo} |")
    lines += ["","## 如何解读","",
              "- 规则把三帧收在同一接触簇内，时间上更集中；当前 VLM 的自由前后文会扩展跨度，可能跨相邻人工事件。",
              "- 结果混合：tIoU≥0.3 时，PNR 在对应人工区间内为规则 6/11、VLM 8/11；tIoU≥0.5 时均为 5/8。因此不能判定规则全面优于 VLM，也没有证据表明当前 VLM 在三帧语义上具有明确整体优势。",
              "- S09 P2：规则为 198/245/305，N=245 属于匹配的 G2；VLM 为 72/113/236，N=113 属于前一个 G1，是当前 VLM 更差的时间归属例。",
              "- S03 P1：规则为 247/310/310，三帧指向 G2；VLM 为 102/178/216，N=178 指向匹配的 G1。上游 P1 原本合并 G1/G2，不能靠选帧修复这个合并。",
              "- 原图对照可直接看到：S09 P2 的规则三帧集中在抽屉内部物品操作，VLM 将较早的开抽屉过程与后面的内部操作放在一组；S03 P1 的规则主要展示笔记本电脑附近操作，而 VLM 展示枕头移开过程。这些是案例观察，没有作为盲评标签或语义准确率。",
              "- 原规则允许 N=Post，得到的可能只有两张不同图；“更稳定”不等于三种语义角色都充分。",
              "- 本次只重跑选帧，未为规则三帧重算 q 或累计记忆，未把 VLM 原 q 当作规则结果。未自动切换生产管线的默认选帧。","",
              "## 文件与复现","",
              "- [18 组帧号 CSV](triplets.csv)","- [完整指标与来源哈希](comparison.json)","- [时间轴 PDF](rule-vlm-triplets.pdf)",
              f"- 全部 18 组实际左目原图对照留在本机：`{report['selection_root']}/gallery.html`；逐事件图片在 `contact_sheets/`，未发布原视频截图。","",
              "```bash","python3 scripts/run_legacy_rule.py \\",
              "  --run-root /data/storage/peekaboo-runs/full_numeric_20261118_v4 \\",
              "  --legacy-root /home/user/zl/pipeline \\","  --legacy-python /home/user/zl/pipeline/.venv-pnr/bin/python \\",
              "  --output /path/to/new_rule_run","python3 scripts/compare_rule_vlm.py \\",
              "  --reference docs/event-comparison/comparison.json \\","  --selections /path/to/new_rule_run \\",
              "  --output docs/rule-vs-vlm","```", ""]
    (output/"README.md").write_text("\n".join(lines))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",required=True);parser.add_argument("--selections",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();selections=Path(args.selections).resolve();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    summary=read(selections/"selection_summary.json")
    assert summary["expected_events"]==summary["executed_events"]==summary["successful_commands"]==18
    reference=read(args.reference);records=collect(reference,selections)
    for rec in reference["recordings"]:
        for path,value in rec["input_hashes"].items():
            assert digest(path)==value, "Reference comparison source changed"
    make_contact_sheets(records,selections)
    shifts=[row["pnr_shift_vlm_minus_rule_frames"]/rec["fps"] for rec in records for row in rec["events"]]
    report={"selection_root":str(selections),"reference_sha256":digest(args.reference),
            "runner_summary_sha256":digest(selections/"selection_summary.json"),
            "legacy_implementation_hashes":summary["legacy_source_hashes"],
            "analysis_source_sha256":digest(__file__),
            "rule_triplet_shown_to_vlm_count":sum(row["rule_triplet_shown_to_vlm"] for rec in records for row in rec["events"]),
            "pnr_shift":{"earlier_count":sum(v<0 for v in shifts),"same_count":sum(v==0 for v in shifts),
                         "later_count":sum(v>0 for v in shifts),"mean_seconds":float(np.mean(shifts)),
                         "median_seconds":float(np.median(shifts))},
            "all_events":{method:all_event_stats(records,method) for method in ('rule','vlm')},
            "matched_diagnostics":{threshold:{method:diagnose(records,method,threshold) for method in ('rule','vlm')} for threshold in ('0.3','0.5')},
            "recordings":records}
    plot(records,output);write_outputs(report,output)
    print(json.dumps({k:report[k] for k in ('all_events','matched_diagnostics')},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
