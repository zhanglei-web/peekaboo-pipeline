> 发布版说明：`<DATA_ROOT>`、`<OUTPUT_ROOT>`、`<MODEL_ROOT>` 是本地资源占位符；历史运行数字保留。新仓库的可执行入口以 [运行指南](RUNNING.md) 为准，历史命令用于解释来源。

# 当前交付：按搜索事件描述筛选锚点对象

根据用户确认：**描述明确提到的物体都纳入，包括背景物体，并在该事件的三帧分别尝试定位。**不再把 scene_memory 中所有物体自动加入每帧，也不只读取 focus_object_states。

当前配置：`components/anchors/examples/event_description_config.json`

候选与原文出处：`components/anchors/examples/event_description_candidates.json`

新输出：`<OUTPUT_ROOT>/frame_anchor_pilot_20261109/190346_event_description_run001`

## 物体范围

从 event_summary、observed_changes、search_region，以及三帧的 state_description、hand_action、visibility_and_occlusion 中整理具体物体及明确部件，取**本事件内部并集**。该并集在 start/middle/end 各尝试定位一次。某帧未检出只记 not_detected，不能把另一帧坐标复制过来。

| 事件 | 候选数 | 纳入对象 |
| --- | ---: | --- |
| 1 | 12 | 银色笔记本、白桌、显示器、独立键盘、鼠标、蓝色咖啡杯、相框、笔记本盖、屏幕、内置键盘、触控板、显示器支架 |
| 2 | 6 | 木色相框、白桌、显示器、键盘、鼠标、描述中露出的小白色物体 |
| 3 | 6 | 蓝色包装、咖啡杯、白桌、键盘、显示器、显示器支架 |
| 4 | 12 | 键盘、鼠标、黑色长方形物体、橙色小物、小型深色设备、白桌、显示器、笔记本、台历、咖啡杯、相框、显示器支架 |

36 项事件内候选 × 3 帧 = 108 次候选观察。它们不是 108 个不同的物理物体。同一事件优先复用原事件引用，避免再加入同名场景引用。无合适原 ID 的具体描述使用 `mention:<event_id>:<name>` 扩展命名空间，例如小型深色设备；不伪造原 scene_memory ID。

手、人体动作、抽象搜索区域不作为物体。"various small items" 等不明确集合不作为独立对象，只纳入其明确列出的实例。黄色门禁卡仅以搜索目标或否定/不确定的视觉描述出现，因此记录为 excluded_mentions，不根据任务结果强行创建已确认目标的锚点。橙色物体保留原描述身份，不改称门禁卡。

笔记本内置键盘与桌面独立键盘分别列出，整机与部件允许几何重叠；这并不保证检测器能够正确区分。相似/嵌套掩码仍需复核。当前原文有开合状态与图像可能不一致的问题，本步骤保留原文，不替其修订状态。

## 可追溯与校验

本条数据的候选清单由助手阅读原 JSON 后显式整理，**不是**通用自动名词抽取器，也不是人工审核完成的清单。为另一条数据运行时，需要先整理对应清单。

每个候选包含原文路径和精确引用片段。运行时核对源文档摘要、事件覆盖、原文引用、原 ID 是否有效、事件实体是否属于当前事件，以及是否有重复候选。候选选择文件受 manifest 哈希约束，修改后必须使用新输出目录。

输出每个锚点和 candidate_outcomes 均保留 description_evidence、description_event_id、description_role、selection_scope。界面提供“本事件描述涉及的候选及原文出处”列表。scene_entity_candidates 仍只是候选对应，不是已验证的真实身份。

原文存在该词仅证明选择有文字依据，不能证明其场景 ID 对应或图像定位正确。selection_review_status 为 assistant_curated_not_human_semantic_reviewed，人工准确率仍需另外评估。

## 执行

```bash
cd <HISTORICAL_PATH>
PYTHONPATH=<WORKSPACE>/src/frame-anchor-enrichment .venv/bin/python -m frame_anchors.cli plan --config components/anchors/examples/event_description_config.json
PYTHONPATH=<WORKSPACE>/src/frame-anchor-enrichment .venv/bin/python -m frame_anchors.cli run --config components/anchors/examples/event_description_config.json
PYTHONPATH=<WORKSPACE>/src/frame-anchor-enrichment .venv/bin/python -m frame_anchors.cli validate --output <OUTPUT_ROOT>/frame_anchor_pilot_20261109/190346_event_description_run001
```

`pilot_config.json` 与 run002 保留为先前全场景试验记录，**不作为当前筛选口径的交付**。当前请使用上面的 event_description 配置与输出。

原图哈希、原值保留、每帧独立推理、掩码匹配、锚点回退、锁和原子写入机制保持沿用。新增测试验证：描述清单在三帧覆盖、未提及场景物体不会混入、只在最后一帧出现于 focus 列表的对象仍进入本事件三帧候选、原文引用或跨事件引用不合法时拒绝。完整测试现为 21 项。


## 本次运行结果

4 个事件、12 帧均已处理。108 次候选观察中，107 次至少有一个检测，1 次未检出（事件 1 中间帧的笔记本内置键盘）。输出 121 个候选锚点：85 个有掩码匹配，36 个框中心回退（22 个无匹配掩码、14 个掩码歧义）。由于多实例和部件重叠，锚点数可能大于候选观察数；107/108 不能称为准确率。

平均每帧处理约 0.77 秒；运行主体约 12.44 秒，不含下载、进程启动、前置文件哈希扫描和最后独立校验。21 项测试通过，真实输出的原值保留、原图哈希、原文证据、事件范围和几何复算检查通过。

已检查 12 帧叠加拼图，发现部分明确误检：笔记本屏幕被定位到独立显示器；事件 3 结束帧的一个显示器支架候选落在台灯上；黑色长方形物体也可能误命中显示器。因此此次完成的是描述限定范围的自动候选标注，尚未达到人工语义验收。发现记录保存在输出目录 `assistant_visual_review.json`，未将自动检测分数或几何校验包装成识别准确率。

`description_contactsheet.jpg` 可快速查看全部 12 帧。`review.html` 可逐帧查看原文出处、未检出条目与冲突；`enriched.json` 是此次新结果，原输入保持不变。
