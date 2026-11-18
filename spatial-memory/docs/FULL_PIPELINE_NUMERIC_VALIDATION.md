# 三段完整管线与数值充分性实测（2026-11-18）

优化后的两阶段空间记忆管线已从三段原始录制处理到全部事件的数值评分及最终记忆。实测使用 **Qwen3-VL-8B-Instruct，Q8_0 权重和 F16 视觉投影**，模型清单与校验值见 [权重清单](../configs/qwen8b_manifest.json)。本次不使用人工事件边界或人工三帧，不复用历史预测缓存。

**人工事件对照已补充**：三段人工标注各 6 个，预测为 5/5/8 个；总数相同不代表一一对应。按 tIoU≥0.3 匹配成功 11 对，Micro-F1=0.611。逐事件切分、动作分数、三帧错位和对照图见 [详细对比](event-comparison/README.md)。下表的“有数值事件”仍以预测事件为分母。

## 完成范围

| 录制 | 原视频帧 | 初始候选区域 | 人工事件 | 预测事件 | 有数值的事件 | 累计记忆重放 |
| --- | --- | --- | --- | --- | --- | --- |
| S03 耳机／184551 | 803 | 9 | 6 | 5 | 5/5 | 通过 |
| S07 数据线／192339 | 651 | 8 | 6 | 5 | 5/5 | 通过 |
| S09 铅笔／185159 | 1044 | 9 | 6 | 8 | 8/8 | 通过 |

共 **2498 帧、18 个事件**。检测阶段全部帧重新推理、状态均为 `ok`，`reused_frames=0`。每个预测事件均处理对象识别、接触簇、候选排序、8B 三帧选择、充分性与记忆提交，没有只取首个事件。三帧全部由模型从实际候选图像中选择；事件和接触簇边界仅为参考。

完整链为：第一阶段前缀图像与区域目录 → 全视频 100DOH → TSM 动作 → CLIP 视觉信号 → cycles 事件 → 逐事件 YOLO 对象／接触簇／变化候选 → 8B 选择 Pre／PNR／Post → 几何优先的数值评分 → 累计空间记忆。几何前提不足时按 [数值策略](NUMERIC_SUFFICIENCY.md) 继续估计，原始失败原因另存。

## 数值来源

以下数量按区域／搜索单元评分统计，一个事件可以有多项评分：

| 方法 | 数量 |
| --- | --- |
| `qwen_continuous_visual_estimate` | 16 |
| `validated_geometric_union` | 2 |
| `geometric_union_with_quality_warnings` | 1 |

运行默认值回退共 **0 项**。三帧选择状态统计：`{'ok': 18}`。事件层 `q_event_estimate` 是本次单元 q 的等权平均，用于核对事件数值可用性，不解释为面积加权的几何覆盖真值。

主 `Q=max(previous_Q,q)` 保存带来源的运行充分性估计；只有严格几何通过的评分进入 `Q_geometric`，近似评分进入 `Q_estimated`，主值来源由 `Q_source` 保存。未检查单元的初值 0 通过 `searched/Q_source` 区别于检查评分。无区域关联时使用待关联的事件局部单元，避免写入任意候选区域。

## 逐事件结果

| 录制 | 事件 | Pre／PNR／Post 源帧号 | 事件汇总值 | 各单元 q 与方法 |
| --- | --- | --- | --- | --- |
| S03 耳机 | event_001 | 102／178／216 | 0.300000 | 0.300000 (qwen_continuous_visual_estimate) |
| S03 耳机 | event_002 | 309／432／447 | 0.500000 | 0.300000 (qwen_continuous_visual_estimate), 0.700000 (qwen_continuous_visual_estimate) |
| S03 耳机 | event_003 | 449／485／498 | 0.300000 | 0.300000 (qwen_continuous_visual_estimate) |
| S03 耳机 | event_004 | 551／676／683 | 0.600000 | 0.600000 (qwen_continuous_visual_estimate) |
| S03 耳机 | event_005 | 695／725／784 | 0.300000 | 0.300000 (qwen_continuous_visual_estimate) |
| S07 数据线 | event_001 | 105／137／232 | 0.143098 | 0.143098 (validated_geometric_union) |
| S07 数据线 | event_002 | 265／286／343 | 0.300000 | 0.300000 (qwen_continuous_visual_estimate) |
| S07 数据线 | event_003 | 399／433／440 | 0.300000 | 0.300000 (qwen_continuous_visual_estimate) |
| S07 数据线 | event_004 | 475／498／528 | 1.000000 | 1.000000 (validated_geometric_union) |
| S07 数据线 | event_005 | 510／530／570 | 0.700000 | 0.700000 (qwen_continuous_visual_estimate) |
| S09 铅笔 | event_001 | 58／73／116 | 0.000000 | 0.000000 (geometric_union_with_quality_warnings) |
| S09 铅笔 | event_002 | 72／113／236 | 0.300000 | 0.300000 (qwen_continuous_visual_estimate) |
| S09 铅笔 | event_003 | 300／382／467 | 0.600000 | 0.600000 (qwen_continuous_visual_estimate) |
| S09 铅笔 | event_004 | 408／488／539 | 0.600000 | 0.600000 (qwen_continuous_visual_estimate) |
| S09 铅笔 | event_005 | 480／514／539 | 0.600000 | 0.600000 (qwen_continuous_visual_estimate) |
| S09 铅笔 | event_006 | 525／696／780 | 0.600000 | 0.600000 (qwen_continuous_visual_estimate) |
| S09 铅笔 | event_007 | 787／817／866 | 0.600000 | 0.600000 (qwen_continuous_visual_estimate) |
| S09 铅笔 | event_008 | 912／942／1043 | 0.300000 | 0.300000 (qwen_continuous_visual_estimate) |

严格几何失败后继续估计的原因包括：

- `ValueError: No FastSAM mask matches the operated-object grounding`：7 项。
- `ValueError: No adjacent connected support plane`：3 项。
- `ValueError: Initial operated object cannot be grounded`：5 项。
- `geometry_quality_audit_failed_or_missing`：1 项。
- `ValueError: ValueError: Selected source frame 1043 has no synchronized right image`：1 项。

这些原因被保留，并没有作为有效几何测量发布。

## 复核与复现

最终运行目录为 `/data/storage/peekaboo-runs/full_numeric_20261118_v4`。入口：

```bash
python3 scripts/full_pipeline.py \
  --config /data/storage/peekaboo-runs/full_numeric_20261118_config.json \
  --output /data/storage/peekaboo-runs/full_numeric_20261118_v4 \
  --frontend-workers 2
```

该输出目录已经使用，复现时必须改为新目录。配置路径按本机环境填写，模板见 [完整运行配置](../configs/full_pipeline.example.json)。正式运行的总墙钟时间约 **15.23 分钟**，包括第一阶段及全帧前端，不包括此前故障排查。

可用 `python3 scripts/verify_full_run.py /path/to/completed_run` 复核。该脚本在推理结束后执行，不属于运行起始时记录的 65 份推理文件。独立读取全部落盘事件后，核对预测 ID 与提交 ID、实际候选帧引用及严格时间顺序、每项 q 有限且位于 [0,1]、事件平均值、每个单元的最大值及独立几何累计值；从目录重放得到的 `units`、物体／大区汇总及全部事件快照与增量记忆完全一致。运行时 **65 份源码／配置／提示词哈希**与最终代码一致。25 项单元测试通过；wheel 构建并确认包含接触簇默认配置。

早期尝试保留在 `full_numeric_20261118_v1`、`full_numeric_20261118_v2`：前者发现 YOLO-World 在新路径重复下载已有 CLIP 权重，后者发现独立仓库缺少 `pipeline_config.json`。补齐权重路径与配置后，在新目录重新运行；v3 在第三段第 5 个事件遇到模型将同一区域按三帧重复列举，严格 ID 检查拒绝提交。最终选择器增加同 ID 说明合并、保留全部依据并采用最低置信等级，原始响应和合并审计仍保存；最终 v4 再次从原始视频重跑三段。中断尝试不计入上表。修复后复用旧事件目录的诊断尝试被输入哈希保护拒绝，也未计入正式结果。

机读结果见 [完整验证摘要](full_pipeline_numeric_summary.json)，本地还保留 `run_manifest.json`、`full_summary.json`、`independent_verification.json`、逐阶段日志、模型请求／响应、严格几何诊断和最终记忆。原始媒体、模型权重和凭据不发布。

本次证明三段全流程能完成并为每次检查提供数值；尚未提供独立人工真值对照的区域语义、三帧角色或几何准确率。视觉估计不等于实测几何覆盖率。当前是离线运行，提交时间是选择／几何环节顺序重放的时间估计，不能作为完整在线时延基准。跨事件 max 仍有互补覆盖与状态变化的适用限制。
