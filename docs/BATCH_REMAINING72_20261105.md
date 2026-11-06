# 2026-11-05：101 条中的剩余 72 条

调度更新：用户随后要求 Astra 后由本机与远端 Qwen 分别领取整条数据，完成各自数据的场景、事件和时序。旧协调器已在阶段收尾后正常退出，迁移审计位于 `run001/migrations/001_scheduler_utilization/`；具体策略见 [整条调度说明](EPISODE_DISPATCH_11_5.md)。原文以下登记信息与来源规则继续有效，实际启用状态以配置、运行状态和迁移记录为准。

输入：`/home/baai/zl/todo/11.4标注已完成`。共 101 个原始采集和 101 份事件 MD。与昨日清单比较，29 个重合采集的 data_id、9 项必需原始文件哈希及既有完成 JSON 哈希均相符，排除这 29 条。

本次登记 72 条、458 次事件，左目事件图 1374 张；S03 15 条、S04 34 条、S07 23 条。原始数据及旧结果保持原位。每条启动 CPU 阶段时仍必须完成原视频与三帧内容匹配，登记成功不代表该内容检查已执行完毕。

## 配置与运行

批次根目录：`/home/baai/zl/todo/11.5/remaining72_20261105`

- `config.json`：11-5 配置，实际输出 `run001/`。
- `input/enrollment_report.json`：排除的 29 条、原始哈希、72 条数量及缺项记录。
- `input/annotation_inventory.json`：72 条 MD、左右图、初始位置及实际放置位置来源。
- `input/preprocessing_inventory.json`、`input/configs/`：保持既有算法参数，关闭 rosbag 输出。
- `input/dataset_facts.json`：用户本次再次明确确认，72 条均为前面未找到、最后一次找到。只作用于列出的 72 个 ID。
- `input/right_frame_exception.json`：用户明确同意的一条右目缺帧例外。
- `run001/state.json`、`run001/logs/`：进度、错误与子进程身份。
- `run001/annotations/json/`：视觉标注及位置来源记录，搜索结果保持基础未知值。
- `run001/annotations/confirmed_json/`：投影了本批人工确认搜索结果的最终 JSON。
- `run001/annotations/confirmed_exports/<id>/`：两种目标格式导出。
- `run001/final_validation.json`：整批所有所需产物完成后的独立校验。

终端命令：

```bash
cd /home/baai/zl/todo/11.4/processing
# Astra 的 ZYAPI_API_KEY 沿用现有环境；Qwen 密钥只放在环境中。
export QWEN_API_KEY="$(cat /home/baai/Downloads/api-key.txt)"
bash pipeline115.sh check --config /home/baai/zl/todo/11.5/remaining72_20261105/config.json --probe
bash pipeline115.sh run --config /home/baai/zl/todo/11.5/remaining72_20261105/config.json
bash pipeline115.sh status --config /home/baai/zl/todo/11.5/remaining72_20261105/config.json
```

如果已运行，不再次启动；操作系统排他锁保护同一个运行目录。检查日志、修复实际错误后，才用原命令加 `--retry-failed` 继续。代码、配置和输入身份变化不能无审计地复用旧运行目录。

并发保持 CPU 2、Astra 1、远端 Qwen 1、本机 Qwen 1；深度与本机 Qwen 在同一 GPU 上分时。不同采集可处于不同阶段，无整批预处理屏障。尚无这 72 条的实际批处理耗时结论。

## 位置字段的来源与边界

最终主 JSON 新增可选字段 `metadata.dataset_location_facts`，不改变原空间树、时序更新、事件及图像证据结构。此信息在确定性组装阶段附加，不把隐藏真实位置提示给视觉推理调用。

- `initial_location`：从同场景、同目标初始照片文件名截取位置描述，并保存照片路径、文件名和 SHA256。67 条有记录；S04 马克笔的 5 条 MD 明确写“无”，记录为 `not_provided`。
- S03 的“标签”与“便签纸”已由用户明确确认是同一物品；“记号笔”匹配同场景“马克笔”名称，映射保存在 `aliases.json`，不更改 MD 的原目标名。
- `actual_location`：原样保留 MD 的“物体最后所在位置”，按用户本次说明作为实际放置位置来源，保存 MD 路径、哈希及该采集内全部事件 ID。
- MD 为每条采集提供一个实际放置位置，并没有为其中每次三帧检查提供不同的搬移时刻。记录因此限定为 episode 级；搬移时间、模型空间节点/搜索区域 ID 不自动补猜。
- 原始 `source_annotation_metadata`（包括原始位置的“1”或“无”）保持不变，解析后的描述单独存储。
- 两种目标导出的 `hidden.location_descriptions` 携带相同来源记录；已有 `true_unit_id`、`original_unit_id` 未完成可靠区域身份映射时仍为空。

## 右目缺帧的用户批准例外

`cs_0001_session_20261030_192457`（S04，橡皮）的最后一组左目结束图是 f700，完整存在。原视频顺序解码确认：左目 701 帧（0–700），右目 700 帧（0–699），与原始 meta.json 一致；MD 指定的右目 f700 并不存在。不能用 f699 代替，也没有生成“恢复图”。

用户明确回复“使用左目继续，保留缺帧记录”。本次仅对此精确采集、精确右目文件启用例外：所有左目三帧仍完整，已有右图仍需校验，右目索引显式标记为部分证据。最终 JSON 以及两种导出的 metadata 中保存 `input_evidence_exception`，包含批准文件、原视频哈希、解码帧数、缺失帧和 `substitute_frame_used=false`。该例外不扩展到其他数据。

启动前验证：24 项自动测试、旧版组装回归和严格历史请求回放通过；72 条位置记录和单条缺帧例外的来源校验通过；远端模型身份及鉴权通过。以上不等于新 72 条模型结果已经生成完成。
