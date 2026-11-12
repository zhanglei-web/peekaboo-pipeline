> 发布版说明：`<DATA_ROOT>`、`<OUTPUT_ROOT>`、`<MODEL_ROOT>` 是本地资源占位符；历史运行数字保留。新仓库的可执行入口以 [运行指南](RUNNING.md) 为准，历史命令用于解释来源。

# 逐帧物体锚点标注 Pipeline（下游详细说明）

完整上游流程见 [从原始采集到最终 JSON 的 Pipeline](PIPELINE.md)。

本文记录本次将既有 episode JSON 与三帧图像扩展为精简锚点 JSON 的实际流程，覆盖数据清点、候选物体确定、逐帧检测与分割、坐标计算、JSON 导出和最终核验。它描述的是已完成的 320 条交付，不代表对上游原始视频重新执行了完整的事件生成流程。

## 1. 目标与产物

输入是已经存在的 episode JSON。每个搜索事件有 `start`、`middle`、`end` 三个状态，每个状态引用一张现成图像。处理后保留原 JSON 的所有原字段和值，只在每个状态中增加 `object_anchors` 数组。

最终精简交付目录：

`<OUTPUT_ROOT>/frame_anchor_batch_20261110_resume50/json/`

按原来源分组保存，目录下共有 320 个 JSON 文件。压缩交付包为：

`<OUTPUT_ROOT>/frame_anchor_batch_20261110_resume50/frame_anchors_compact_320_20261110.zip`

精简数组每项仅含五个字段：

```json
{
  "name": "blue coffee cup",
  "ref": "scene:o01",
  "anchor_xy_normalized": [0.52, 0.43],
  "size_wh_normalized": [0.13, 0.22],
  "status": "mask_unreviewed"
}
```

实际输出中的 `ref` 也可能使用事件实体 `entity:` 或事件局部描述对象 `mention:`。上例仅用于解释字段结构，不是某条输出的逐字样例。

## 2. 数据来源和三帧对应关系

处理范围由既有 JSON 和其中记录的 `image_path`、`image_sha256` 决定。没有根据 episode 名称猜测图像，也没有在锚点任务中重新抽帧。每个事件的三个状态必须按 `start → middle → end` 对齐；每帧使用其自己的图像路径和哈希。

| 来源批次 | JSON 数 | 事件数 | 帧数 |
|---|---:|---:|---:|
| `initial81/base68` | 68 | 401 | 1,203 |
| `initial81/nightstand13` | 13 | 133 | 399 |
| `followup29` | 29 | 196 | 588 |
| `remaining72` | 72 | 458 | 1,374 |
| `followup138/11.2` | 60 | 379 | 1,137 |
| `followup138/11.3` | 71 | 626 | 1,878 |
| `followup138/input7_11.2` | 3 | 16 | 48 |
| `followup138/input7_11.3` | 4 | 28 | 84 |
| **合计** | **320** | **2,237** | **6,711** |

来源 JSON 目录如下：

| 来源 | 输入目录 |
|---|---|
| `initial81/base68` | `<SEMANTIC_WORKSPACE>/outputs/todo_astra_tf_qwen/research_json/` |
| `initial81/nightstand13` | `<DATA_ROOT>/11.3/search_episode_outputs/json/` |
| `followup29` | `<DATA_ROOT>/11.4/search_episode_outputs/confirmed_json/` |
| `remaining72` | `<DATA_ROOT>/11.5/remaining72_20261105/run001/annotations/confirmed_json/` |
| `followup138/11.2` | `<DATA_ROOT>/11.5/followup138_20261105/11.2/run001/annotations/confirmed_json/` |
| `followup138/11.3` | `<DATA_ROOT>/11.5/followup138_20261105/11.3/run001/annotations/confirmed_json/` |
| `followup138/input7_11.2` | `<DATA_ROOT>/11.5/followup138_20261105/input7_11.2/run001/annotations/confirmed_json/` |
| `followup138/input7_11.3` | `<DATA_ROOT>/11.5/followup138_20261105/input7_11.3/run001/annotations/confirmed_json/` |

图片实际根目录跨多个历史任务目录。应以清单逐帧的绝对路径为准，而不是按目录名推断：

`<OUTPUT_ROOT>/frame_anchor_batch_20261110_resume50/input_inventory.json`

该清单记录源 JSON 路径和 SHA-256、事件/帧数、每帧 `state_id`、阶段、原视频帧号、图像路径、JSON 中预期图像哈希、实测哈希及图像尺寸。建立清单时校验了 320 个源 JSON、2,237 个事件、6,711 张图像；图像路径均存在，且实测 SHA-256 与源 JSON 中记录相符。部分 episode ID 在不同来源批次重复，因此输出按来源目录隔离，不按文件名合并或覆盖。

这些输入 JSON 的早期语义内容来自各自上游标注工作流；例如早期 68 条 JSON 内记载了 Astra/TF/Qwen 相关来源信息。本锚点阶段复用现成语义、事件和帧引用，没有重新调用 Astra、重新生成事件描述或重新抽取视频帧。

## 3. 事件范围与候选物体

本次遵循的范围是：**仅尝试定位当前搜索事件描述明确提到的物体**。包括事件中明确说到的背景物体和具体部件。一个事件内选出的对象候选并集，会在该事件的三帧中分别运行检测；不将一帧坐标复制到另一帧。

描述证据取自：

- `event_summary`
- `observed_changes`
- `search_region`
- 三帧状态中的 `state_description`
- 三帧状态中的 `hand_action`
- 三帧状态中的 `visibility_and_occlusion`

排除或谨慎处理：

- 手、手指、人体等身体部位，以及动作和抽象区域。
- 没有具体指明成员的 `items`、`objects`、杂物集合，不作为单个确定物体。
- 仅作为搜索目标、否定观察、假设对象或无法确认对象出现的文字，不作为已确认候选。
- 对同名或同外观物体不擅自确认跨帧物理身份。
- 不为了补齐数量而从图像中发掘原描述没有提到的物体。

候选引用按优先级复用输入中的事件实体或场景对象引用。若描述明确提到一个对象，但输入 JSON 没有可安全复用的 ID，则建立事件局部 `mention:` 引用；不伪造 `scene_memory` ID。原始名称与描述证据留在审计数据中，精简输出保留 `name` 和 `ref`。

## 4. 候选提取方式和历史来源差异

320 条交付并非全部使用同一种候选整理来源，必须保留来源差异：

| 记录范围 | 候选来源 | 边界 |
|---|---|---|
| 原有 50 条 + 另 1 条试跑，共 51 条 | 最终来源统计为 49 条本机 Qwen 提取、2 条助手整理 | 复用已完成结果，均未完成人工语义验收 |
| 后续批量 269 条 | 固定中英文名词词表 + 原文精确提及匹配 + 既有引用匹配 | 不调用文本模型；词表辅助不等于人工逐事件语义确认 |

注意：表中 49+2 是最终交付校验按候选映射 `selection_review_status` 统计的记录数；“保留原来已完成的 50 条”描述的是续跑接手时的历史完成基数，另有一条无 Qwen 试跑结果并入续跑起点。最终 provenance 计数以 `independent_final_validation.json` 为准：49 条本机 Qwen 来源、2 条助手整理来源、269 条词表来源。

早期批量提取阶段曾使用本机 Qwen3.8-27B 解析既有事件文本，结果需要通过原文引用和 ID 约束；远端 Qwen 没有被本次锚点批处理调用；上游场景、事件和时序标注使用过远端 Qwen。用户随后明确要求后续续跑不再使用 Qwen。因此，最终 269 条由 `prepare_resume50.py` 的固定中英文物体词表处理：从源事件字段匹配具体名词及颜色、形状、方位等描述，过滤括号中的搜索目标、否定/不确定提及和身体部位，优先关联当前事件实体/场景对象，无法唯一关联时生成事件局部 `mention:` 引用。

这套词表与规则是可追溯的文本候选辅助，不保证覆盖所有同义词、指代或复杂语义，也不等同于人工审核。候选清单保存原文路径和引用片段，并绑定源 JSON 摘要；定位前检查候选范围和引用有效性。

## 5. 视觉模型与几何计算

### Grounding DINO Tiny

本地模型目录：

`<MODEL_ROOT>/frame_anchors/grounding-dino-tiny/`

根据每个候选的英文短语，在当前帧图像中产生零样本检测框及检测分数。候选短语按批输入；默认参数为 detection threshold `0.28`、每轮最多 6 个短语、同一候选最多保留 5 个实例、同候选检测框 NMS IoU `0.5`。分数是模型启发式输出，不是校准概率，也不是已测量准确率。

### FastSAM-s

本地权重：

`<MODEL_ROOT>/search_pipeline/FastSAM-s.pt`

每帧运行 FastSAM-s 产生分割掩码候选，默认 `imgsz=1024`、`conf=0.25`、`iou=0.9`、最多 200 个掩码。掩码必须与原始图像像素尺寸一致，程序不会静默缩放错误尺寸的掩码。

### 检测框与掩码匹配

对于每个 Grounding DINO 检测框，程序按以下条件匹配 FastSAM 掩码：掩码至少 16 像素；掩码框与检测框 IoU 至少 `0.5`；掩码落在检测框内的比例至少 `0.85`。综合匹配排序为 `0.7 × IoU + 0.3 × containment`。若前两名差值小于 `0.06`，视为歧义，不强行选掩码。不同来源引用之间不静默合并；共享掩码或高重叠框会记录竞争引用冲突。

### 锚点与 size

- 有可靠掩码时，锚点取掩码最大连通域内距边界最远的像素中心，避免质心落入掩码空洞。
- 没有可靠掩码但有检测框时，使用检测框中心并标记 `bbox_center_needs_review`。
- 没有检测框时，锚点与大小为 `null`，状态 `not_detected`。这仅表示模型未检出，不证明物体不存在。
- `anchor_xy_normalized = [x / W, y / H]`，原点为原图左上角。
- `size_wh_normalized = [bbox_width / W, bbox_height / H]`，来源是检测框宽高，不是面积比例、物理宽高或厘米数。
- 坐标与比例写入精简 JSON 时保留六位小数。

状态定义：

| 状态 | 意义 |
|---|---|
| `mask_unreviewed` | 掩码与框几何匹配，但尚无人工语义确认 |
| `mask_needs_review` | 已有匹配掩码，但存在多实例、引用冲突等复核原因；匹配歧义会退回框中心 |
| `bbox_center_needs_review` | 没有可靠匹配掩码，锚点退回检测框中心 |
| `not_detected` | 该候选没有产生检测框；不代表确认不存在 |

上述结果是二维图像投影。透视、遮挡和裁切会影响框大小；不推断物体实际尺寸、完整三维形状或遮挡后的面积。多个提示词可能指向同一画面区域，部件与整体也可能重叠，因此保留冲突信息供复核。

## 6. 从源 JSON 到精简 JSON

每个源记录的处理顺序如下：

1. 从 `input_inventory.json` 读取唯一来源分组、JSON 路径、源文件哈希、事件数及每帧图像信息。
2. 校验源 JSON 哈希；候选映射绑定源文档摘要、完整事件集合及原文证据。
3. 对每个事件生成候选并集，把候选应用到该事件的 start/middle/end 三个状态。
4. 再核对图像哈希，逐帧调用 Grounding DINO 和 FastSAM，生成检测、掩码和几何结果。
5. 将完整增强结果及证据写入审计运行目录，包括 `enriched.json`、逐帧 `detections.json`、`masks.npz`、`anchors.json`、叠加图、时间信息和文件 receipt。
6. `scripts/export_compact.py` 先验证详细运行结果，再深拷贝原 JSON，仅在各状态增加 `object_anchors`。它逐条映射检测和未检出候选，确保无候选被丢弃。
7. 导出后删除新增字段进行逐值比较，要求恢复结果与源 JSON 完全相同；检查源 JSON 哈希、写出 JSON 可重新解析及精简字段合法。
8. 只有单条记录导出通过后，批处理器才将结果原子发布到 `json/<来源批次>/<记录名>.json`，并记录文件 SHA-256。

详细检测分数、证据、分割掩码、叠加图和冲突原因不塞进精简主 JSON；这些内容保留在审计目录，并由 `audit_index.json` 中的 state ID 和数组下标关联。精简格式详见 [COMPACT_JSON.md](COMPACT_JSON.md)。

## 7. 批处理、断点与错误处理

最终续跑运行目录：

`<OUTPUT_ROOT>/frame_anchor_batch_20261110_resume50/`

最终 `run_noqwen_batches.py` 保留 51 条已完成结果，收集候选映射就绪的待处理记录，在获得 GPU 租约并加载模型后，依次处理 ready 列表，共新增处理 269 条；这一轮共用加载的模型。历史 `run_all_batches.py` 的“每批最多 4 条、独立进程”调度不能套用到本续跑版本。

GPU 通过共享排他锁 `<GPU_LOCK>` 管理，并检查其他计算进程和显存；不终止其他任务。输出目录有运行锁，manifest 绑定输入清单、代码、模型配置及参数；候选映射和源文件另行核验哈希，变化不能静默复用。单帧 receipt 校验通过后才可复用，原图在处理前后校验。记录失败写入 `failures`，未通过导出检查的记录不发布为有效 JSON；本续跑入口没有记录级自动重试循环。

恢复前需核实状态文件、PID、实际命令、Linux 启动 tick 和排他锁，不能仅凭旧 `status` 判断仍在运行。

主要状态及审计文件：

| 路径 | 用途 |
|---|---|
| `input_inventory.json` | 输入记录、逐帧图像路径和源哈希 |
| `selections/<来源>/<记录>.json` | 事件候选、原文证据和候选来源状态 |
| `selection_preparation_report.json` | 候选整理覆盖、事件数、候选数及异常 |
| `batch_manifest.json` | 本次运行输入/代码/模型/方法绑定 |
| `batch_state.json` | 已完成、失败、当前阶段和进程身份 |
| `detailed/<来源>/<记录>/` | 详细增强证据和逐帧文件 |
| `compact_records/<来源>/<记录>/` | 单条导出、audit index 和验证报告 |
| `json/<来源>/<记录>.json` | 最终精简 JSON 发布目录 |
| `independent_final_validation.json` | 独立的全量文件、图像、候选覆盖及字段检查 |
| `delivery_manifest.json` | 交付文件清单、归档哈希及最终统计 |

## 8. 复现和检查命令

以下命令适用于检查已经完成的续跑结果，不会启动模型推理：

```bash
cd <HISTORICAL_PATH>
.venv/bin/python src/frame-anchor-enrichment/scripts/validate_resumed_batch.py \
  --root outputs/frame_anchor_batch_20261110_resume50 \
  --output outputs/frame_anchor_batch_20261110_resume50/independent_final_validation.recheck.json
```

该独立校验会遍历全部记录，核对源和交付哈希、候选来源/事件范围、每帧候选覆盖、精简字段及归一化值，并可核验全部图像哈希。它是结构和来源完整性校验，不是人工语义验收。

历史批处理入口记录如下：

```bash
.venv/bin/python src/frame-anchor-enrichment/scripts/prepare_resume50.py \
  --root outputs/frame_anchor_batch_20261110_resume50 --ready

.venv/bin/python -u src/frame-anchor-enrichment/scripts/run_noqwen_batches.py \
  --root outputs/frame_anchor_batch_20261110_resume50
```

第一条是本次续跑的候选准备阶段，第二条是无 Qwen 的检测/分割批处理阶段。这些是已完成运行的记录；续跑目录已有完成状态和交付文件，不要再次执行以覆盖或改变本次交付。重新处理应建立新的空输出目录，先准备与该目录绑定的 inventory、候选映射和 manifest，并确认没有其他实例持有 GPU/运行锁。`prepare_noqwen_selections.py` 是更早的词表候选准备实现；本次最终 269 条以 `prepare_resume50.py` 生成的映射和报告为准。

历史早期候选模型和图像模型曾按对应时期的运行入口执行；当前代码树中仍保留相关脚本。应以具体批次的 `batch_manifest.json`、`selection_preparation_report.json`、`batch_state.json` 和日志为准，不能把新旧批次混用成同一模型配置。

## 9. 最终校验与验收范围

本次最终记录见：

- `outputs/frame_anchor_batch_20261110_resume50/final_validation.json`
- `outputs/frame_anchor_batch_20261110_resume50/independent_final_validation.json`
- `outputs/frame_anchor_batch_20261110_resume50/delivery_manifest.json`

记录结果：320/320 条，2,237 个事件，6,711 帧和 6,711 张唯一图像；记录失败 0、跳过 0、未处理项 0。独立校验确认源图像哈希、源 JSON 原值保留、事件/逐帧候选覆盖、精简字段和归一化范围。归档压缩包 SHA-256 记录在 `delivery_manifest.json`。

校验**没有**测量识别准确率，也没有完成逐条人工语义验收。独立报告中 `semantic_human_reviewed=false`、`accuracy_measured=false`。抽查已发现模型误检/错配的可能，包括部件与相似物体混淆；误检、漏检、遮挡、描述本身不准确及词表漏词都可能存在。若用于高要求任务，应通过逐帧叠加图或复核页面人工检查：物体类别/引用是否正确、锚点是否落在对应可见物体内、检测框是否可用、哪些候选应该出现但未检出。`not_detected` 不能标成“物体不存在”。

## 10. 可追溯文件

工程代码：`components/anchors/`

关键模块：

- `scripts/inventory_320.py`：历史输入与图像清单生成脚本。
- `scripts/prepare_resume50.py`：本次续跑的词表候选准备。
- `scripts/run_noqwen_batches.py`：无文本模型的检测、分割、导出和发布。
- `scripts/validate_resumed_batch.py`：独立批次校验。
- `scripts/export_compact.py`：详细增强结果到精简 JSON 的导出器。
- `frame_anchors/contract.py`：源 JSON 适配、帧和引用契约、原值保留检查。
- `frame_anchors/description_scope.py`：事件描述候选及原文证据校验。
- `frame_anchors/models.py`：Grounding DINO/FastSAM 本地权重适配和 GPU 锁。
- `frame_anchors/geometry.py`：框、掩码匹配、冲突检测、锚点和尺寸计算。
- `frame_anchors/artifacts.py`：掩码、可视化、审计页和几何复算验证。

更细字段规范见 [COMPACT_JSON.md](COMPACT_JSON.md)，范围规则见 [EVENT_DESCRIPTION_SCOPE.md](EVENT_DESCRIPTION_SCOPE.md)，无 Qwen 续跑来源与边界见 [NOQWEN_RESUME_20261110.md](NOQWEN_RESUME_20261110.md)，实现约束见 [IMPLEMENTATION.md](IMPLEMENTATION.md)。
