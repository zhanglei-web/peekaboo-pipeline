"""Inspectable candidate overlays and a self-contained local HTML report."""
import html
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont
from .frames import FONT_PATH


COLORS = ["#2563eb", "#dc5b32", "#079177", "#8b52c7", "#b17a07", "#b54383"]


def overview(catalog, selected, output):
    output = Path(output)
    frames = {row["frame_index"]: row for row in selected}
    index = max(frames, key=lambda key: sum(
        any(item["frame_index"] == key for item in region["evidence"])
        for region in catalog["regions"]))
    source = Image.open(frames[index]["image"]).convert("RGB")
    source.thumbnail((1120, 760))
    width, height = source.size
    row_height = 54
    canvas = Image.new("RGB", (width + 570, max(height + 110, len(catalog["regions"]) * row_height + 130)), "#f6f8fc")
    draw = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(FONT_PATH, 26)
    font = ImageFont.truetype(FONT_PATH, 20)
    small = ImageFont.truetype(FONT_PATH, 17)
    canvas.paste(source, (0, 80))
    draw.text((20, 16), f"初始候选区域 · 本地 Qwen · F{index:06d}", font=title, fill="#16304b")
    for i, region in enumerate(catalog["regions"]):
        color = COLORS[i % len(COLORS)]
        evidences = [row for row in region["evidence"] if row["frame_index"] == index]
        for ev in evidences:
            x1, y1, x2, y2 = ev["bbox"]
            box = (int(x1 * width), 80 + int(y1 * height),
                   int(x2 * width), 80 + int(y2 * height))
            draw.rectangle(box, outline=color, width=3)
            left, top = box[:2]
            draw.rectangle((left, top, left + 44, top + 28), fill=color)
            draw.text((left + 4, top), f"{i+1:02d}", font=font, fill="white")
        y = 85 + i * row_height
        draw.rounded_rectangle((width + 16, y, width + 59, y + 29), radius=5, fill=color)
        draw.text((width + 21, y - 1), f"{i+1:02d}", font=font, fill="white")
        name = region["name"]
        if len(name) > 23:
            name = name[:22] + "…"
        draw.text((width + 71, y - 1), name, font=font, fill="#243a54")
        kinds = {"open_support_surface": "开放支持面", "occluder_footprint": "覆盖物原占地区",
                 "container_interior": "容器内部"}
        draw.text((width + 72, y + 27), kinds[region["region_type"]], font=small, fill="#66798e")
    draw.text((20, canvas.height - 25), "框表示可见锚定结构；候选内部几何与充分性尚未计算。", font=small, fill="#60738b")
    canvas.save(output / "regions_overview.jpg", quality=94)


def write_report(output, catalog, audit, keyframes, receipts):
    output = Path(output)
    overview(catalog, keyframes, output)
    esc = html.escape
    def table_row(i, region):
        return "<tr>" + "".join("<td>" + esc(str(value)) + "</td>" for value in [
            f"R{i:03d}", region["name"], region["region_type"], region["boundary_description"],
            region["existence_basis"], region["interior_visibility"], region["confidence"]]) + "</tr>"
    rows = "\n".join(table_row(i, row) for i, row in enumerate(catalog["regions"], 1))
    warnings = "".join("<li>" + esc(s) + "</li>" for s in catalog["uncertainties"])
    frame_ids = ", ".join(str(row["frame_index"]) for row in keyframes)
    single_request = catalog.get("model_review", {}).get("review_mode") == "single_multiscale_request"
    review_description = ("一次完整区域请求，同时读取整体图与局部放大图；未执行单独的第二轮复核。"
                          if single_request else "模型执行区域初稿与同图完整性复核；复核不等于独立真值验证。")
    report = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>初始候选区域 · 单条原始数据试跑</title>
<style>
body{{font:16px/1.75 system-ui,sans-serif;color:#263b55;background:#f1f5fa;margin:0}}
main{{max-width:1480px;margin:36px auto;background:white;padding:36px;border-radius:18px}}
h1{{font-size:30px;margin:0}}h2{{margin-top:32px;font-size:22px}}p{{color:#536a84}}
.tags{{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}}.tag{{background:#edf3fb;padding:12px 20px;border-radius:10px}}
img{{max-width:100%;border-radius:12px;border:1px solid #e0e8f2}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:12px;border-bottom:1px solid #e2e8f1}}
th{{background:#edf3fb}}code{{background:#edf3fb;padding:2px 6px}}a{{color:#2563eb}}
</style><main>
<h1>场景初始化与候选区域 · 真实试跑</h1>
<p>{esc(catalog['episode_id'])} · 本地模型 {esc(receipts['regions']['model'])}</p>
<div class="tags"><div class="tag">观察窗口：前 {audit['observation_seconds']} 秒</div>
<div class="tag">采样 {audit['sampled_frames']} 帧 → 语义关键帧 {len(keyframes)} 帧</div>
<div class="tag">候选 {len(catalog['regions'])} 个 · 父级 {len(catalog['parents'])} 个</div></div>
<p>模型只接收初始化窗口中的图像及通用规则。目标文本、数据集路径、事件三帧和最终位置均未发送。
本次生成二维语义候选，几何身份未独立确认，尚未计算充分性或运行策略预测。</p>
<h2>实际选中的关键帧</h2><p>原始零基帧号：{esc(frame_ids)}。{esc(receipts['keyframe_reason'])}</p>
<img src="selected_keyframes.jpg" alt="关键帧">
<h2>候选区域与可见参考结构</h2><p>{esc(catalog['scene_description'])}</p>
<img src="regions_overview.jpg" alt="候选区域参考框">
<h2>逐区域结果</h2><table><tr><th>ID</th><th>名称</th><th>类型</th><th>边界描述</th>
<th>存在依据</th><th>内部可见性</th><th>模型置信</th></tr>{rows}</table>
<h2>保留的不确定性</h2><ul>{warnings}</ul>
<h2>运行记录</h2>
<p>{review_description}</p>
<p>选帧模型耗时 {receipts['keyframes']['seconds']:.1f} 秒；区域模型耗时 {receipts['regions']['seconds']:.1f} 秒。
区域复核耗时 {receipts.get('region_review', {}).get('seconds', 0):.1f} 秒。
模型请求、图像哈希、原始响应和输出 token 用量均保存在 model/ 下，密钥未写入报告。</p>
<p><a href="catalog.v0.json">候选目录 JSON</a> · <a href="input_audit.json">输入审计</a> ·
<a href="sampled_frames.json">采样与质量指标</a> · <a href="model/regions/result.json">模型原始结构化结果</a></p>
</main></html>"""
    (output / "report.html").write_text(report)
    md = [
        "# 单条原始录制：候选区域初始化实测",
        "", f"- 录制：{catalog['episode_id']}", f"- 模型：{receipts['regions']['model']}",
        f"- 观察窗口：前 {audit['observation_seconds']} 秒，右端点不包含。",
        f"- 采样：{audit['sampled_frames']} 帧；最终关键帧：{frame_ids}。",
        f"- 输出：{len(catalog['regions'])} 个候选，{len(catalog['parents'])} 个组织父级。",
        "- " + review_description,
        "", "![实际关键帧](selected_keyframes.jpg)", "", "![候选参考框](regions_overview.jpg)", "",
        "| ID | 名称 | 类型 | 内部可见性 |", "|---|---|---|---|",
    ]
    md.extend(f"| R{i:03d} | {r['name']} | {r['region_type']} | {r['interior_visibility']} |"
              for i, r in enumerate(catalog["regions"], 1))
    md.extend(["", "模型只读取初始化图像，不读取目标或未来交互。输出为语义候选提议；"
               "参考框不代表隐藏搜索空间几何，充分性保持未知。", "",
               "[HTML 详细报告](report.html) · [目录](catalog.v0.json) · [输入审计](input_audit.json)", ""])
    (output / "REPORT.md").write_text("\n".join(md))
