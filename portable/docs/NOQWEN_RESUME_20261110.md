> 发布版说明：`<DATA_ROOT>`、`<OUTPUT_ROOT>`、`<MODEL_ROOT>` 是本地资源占位符；历史运行数字保留。新仓库的可执行入口以 [运行指南](RUNNING.md) 为准，历史命令用于解释来源。

# 2026-11-10：保留 50 条，从未完成项续跑

用户要求：从现在起不再使用本机或远端 Qwen；沿用第一条的 Grounding DINO + FastSAM 定位、掩码和精简 JSON 格式。保留原来已经完成的 50 条，问题自行处理，冲突记录后继续下一条。

当前目录：`<OUTPUT_ROOT>/frame_anchor_batch_20261110_resume50`。

- `json/<原批次>/<记录名>.json`：交付的精简 JSON。最初 81 条、后续 29 条、72 条、138 条保留各自子目录。同名记录以批次路径区分。
- `batch_state.json`：真实已发布数、当前记录、异常，以及 PID、Linux 启动 tick、命令。必须同时核实 `/proc` 和 `batch.lock`，不能仅凭状态文件称进程在运行。
- `resume_receipt.json`：保留原 50 条及其哈希，另外已有的 1 条无 Qwen 试跑结果；续跑起点合计 51 条，余下 269 条。
- `selections/`：新候选清单，每个候选绑定原事件描述字段及精确引用。原先 50 条的选择来源保留在旧目录；不会将其改称无 Qwen 生成。
- `detailed/`、`compact_records/`：新推理证据、逐帧叠图和导出校验，均放在交付 JSON 之外。
- `selection_preparation_report.json`：本次余下 269 条、1934 个事件的候选整理覆盖和异常。
- `final_validation.json`：运行器结束后生成；只有全部 320 条成功且最终校验无异常才会标为 `complete`。

## 新增处理与第一条的关系

检测和分割直接复用第一条的 `frame_anchors.cli.run`、Grounding DINO tiny、FastSAM-s、阈值、几何算法、逐帧独立推理与精简导出器。不调用 Qwen，不启动文本模型服务，不使用网络文本模型，也没有 Qwen 回退路径。

第一条候选是助手逐事件显式整理的。为批量续跑新增的 `prepare_resume50.py` 使用助手编写的中英文物体词表，按最长名词匹配原事件描述，过滤搜索目标括号、否定/不确定观察、身体部位和不明确杂物，对背景物体和明确部件按原描述纳入。根据名称、颜色、位置等保守匹配原引用；无唯一合适引用则使用事件局部 `mention:`，不伪造 scene ID。

这一步是**有原文证据的词表辅助候选整理**，不等同于第一条逐事件显式语义整理，更不是人工语义验收。其状态明确记为 `source_bound_lexicon_not_human_semantic_reviewed`。词表可能漏掉同义词、混淆指代，检测模型仍可能误定位；不能把候选覆盖或结构校验说成识别准确率。

## 字段、冲突和跳过

原 JSON、原图、模型权重不修改。交付仅向原始事件每帧增加 `object_anchors` 数组，每项只有 `name`、`ref`、`anchor_xy_normalized`、`size_wh_normalized`、`status`。`size` 是投影包围框宽/图宽、高/图高，不是实物尺寸。

未检出时锚点与大小为 `null`，状态为 `not_detected`，不表示物体不存在。掩码歧义、不同来源竞争、框中心回退等保留需复核状态，不伪装成确认定位。

记录级输入哈希、候选映射、导出冲突记入 `failures` 并继续下一条，不纳入已完成。最终列表保留所有未处理/失败项。没有异常且全部 320 条完成时，最终校验仍只证明原值保留、输入哈希、帧覆盖、几何与文件完整性；`semantic_human_reviewed` 始终为 `false`。

## 运行入口

```bash
cd <HISTORICAL_PATH>
.venv/bin/python src/frame-anchor-enrichment/scripts/prepare_resume50.py --root outputs/frame_anchor_batch_20261110_resume50 --ready
.venv/bin/python -u src/frame-anchor-enrichment/scripts/run_noqwen_batches.py --root outputs/frame_anchor_batch_20261110_resume50
```

不要在运行期间改动已发布候选映射。新运行器只加载已准备的清单；缺清单的记录不会静默调用模型补全。GPU 与批次均使用排他锁，其他计算进程不会被终止。
