> 发布版说明：`<DATA_ROOT>`、`<OUTPUT_ROOT>`、`<MODEL_ROOT>` 是本地资源占位符；历史运行数字保留。新仓库的可执行入口以 [运行指南](RUNNING.md) 为准，历史命令用于解释来源。

# 精简锚点 JSON 生成规范

当前交付规则：保留原 JSON 的所有字段和值，仅在每帧 `search_event_states.events[].states[]` 增加 `object_anchors` 数组。候选只来自该事件描述明确提及的物体，包括背景物体及明确部件，在三帧分别检测。详细运行信息单独保存。

每个数组元素只有五个字段：

| 字段 | 定义 |
| --- | --- |
| name | 物体名称，优先沿用原引用标签 |
| ref | scene / entity / mention 命名空间下的对象候选引用 |
| anchor_xy_normalized | `[x/W, y/H]`；原图左上角为原点；未检出为 null |
| size_wh_normalized | `[检测框宽/W, 检测框高/H]`；未检出为 null |
| status | mask_unreviewed、mask_needs_review、bbox_center_needs_review 或 not_detected |

size 是画面中的检测框宽高比例，不是面积占比或实际厘米数。例如 960×600 图像中的 `[0.2,0.3]` 表示框约 192×180 像素。坐标与比例保留六位小数。

掩码锚点取最大连通域的内部最大距离点。无法可靠匹配掩码时用框中心并标记 bbox_center_needs_review；存在冲突/多实例等情况标记 mask_needs_review。mask_unreviewed 同样不代表语义正确。未检出不等于不存在；不得编造坐标。多个候选保留多条同 ref 记录，不强制选一个，也不宣称跨帧实体身份已确认。

导出脚本：`components/anchors/scripts/export_compact.py`。

已有完整运行目录可按以下命令导出。输出目录必须尚不存在，避免覆盖：

```bash
<WORKSPACE>/.venv/bin/python components/anchors/scripts/export_compact.py \
  --run-dir <OUTPUT_ROOT>/frame_anchor_pilot_20261109/190346_event_description_run001 \
  --output-dir <OUTPUT_ROOT>/frame_anchor_pilot_20261109/190346_compact_v2
```

已有示例位于 `190346_compact_v1`，不需要重跑。脚本先校验详细运行的输入、图像、证据和几何，然后由原 JSON 深拷贝生成精简结果；去掉新增数组后必须与原值一致。所有检测及未检出项均需保留，生成后重新解析校验。

精简目录含主 JSON、audit_index.json（按 state_id 和数组下标追溯详细 anchor_id）、validation.json。检测分数、原文出处、掩码和冲突详情保留在详细运行目录。主文件不嵌入这些审计信息。

旧 schema 若禁止附加字段，不能直接将精简文件当作旧 schema 合法输入；需支持 object_anchors 扩展。几何校验和文件完整性通过不代表人工语义验收。已有模型误检不会因精简格式而自动修正。

用户已于 2026-11-09 重新授权按此格式批量处理既有数据；批量交付仍需核对来源、三帧图像、完成数量和失败项，不能将跳过项算成功。
