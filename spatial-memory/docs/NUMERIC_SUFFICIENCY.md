# 每次检查均输出数值的充分性策略

按用户最新要求，自动管线默认使用 `score_policy=best_effort`：每个预测事件、每个实际关联的检查单元均输出有限的 `[0,1]` 数值。系统采用当前能获得的相对可信证据，不以几何组件失败或质量警告中断事件记忆更新。

## 输出次序

| 优先级 | 输出方法 | score_type / status | 数值含义 |
| --- | --- | --- | --- |
| 1 | `validated_geometric_union` | geometric_coverage / valid | 通过几何质量检查的固定参考三帧覆盖比例 |
| 2 | `geometric_union_with_quality_warnings` | geometric_coverage / estimated | 有参考和覆盖诊断，但有支撑面等质量警告的近似几何值 |
| 2 | `available_registered_views_lower_bound` | geometric_coverage / estimated | 在固定分母上，只合并可用配准帧的覆盖，作为不完整观测的下界估计 |
| 3 | `qwen_continuous_visual_estimate` | visual_sufficiency_estimate / estimated | Qwen3-VL-8B-Instruct 根据实际三帧和声明范围给出的连续视觉估计 |
| 4 | `no_evidence_operational_default` | default_sufficiency / fallback | 模型估计也无法取得时的运行默认值 0，明确不属于几何测量 |

视觉估计不把“有效／部分／无效”固定编码为 1／0.5／0。模型须给出一个具体比例、观察依据、范围描述和高／中／低证据强度；证据强度不是校准概率。原始严格结果、降级原因、全部请求／响应和最终数值分开保存。默认值为 0 时必须注明无可用估计，不能解释成测得“几何覆盖率为 0”。

## 累计记忆

主状态 `Q` 表示同一稳定范围的**当前运行所采用的充分性估计**，继续按 `max(previous_Q,q)` 更新。未检查目录单元的运行初值为 0；“尚未检查”与“已经算得 0”由 `searched`、`Q_source` 区分。

每个单元另外保存：

- `Q_geometric`：只累计优先级 1 的几何评分，其余来源不覆盖该字段。
- `Q_estimated`：累计近似几何和视觉估计。
- `Q_source`：当前主 Q 的事件、类型、状态和方法；相同数值优先保留更直接的证据来源。
- `history`：逐次 q、计算类型、方法、证据强度和可用时间。

因此主 Q 不能全部称为实测几何覆盖率。几何准确性实验应读取 `Q_geometric` 或按 `score_type/status` 分组；运行策略可以使用带来源的主 Q。论文或评测报告需要说明使用了哪种口径。

事件层 `q_event_estimate` 为本次去重后的检查单元数值的等权平均，用于报告事件是否产生了数值；它不是面积加权的几何覆盖真值。

## 无关联与不完整三帧

三帧仍由 8B 从实际提交的候选图像中选择。证据不完整时，它也选择最接近角色的三张图并标记 `uncertain`，程序继续估计并保留该状态，不把固定边界帧直接当作角色。无目录关联时建立明确标为 `event_local` 的待关联上下文单元，保存其数值，不把它传给任意物体或候选区域的几何覆盖；该单元不参与已确认物体／大区汇总。

若模型将同一 `candidate_id` 按不同帧重复列举，选择器合并该 ID 的全部说明并采用其中最低置信等级，一个搜索单元只更新一次。原始响应与 `association_merge_audit.json` 保留全部条目；合并不改变任何三帧引用，不补造候选 ID，目录外 ID 仍拒绝。

## 全量运行

配置模板见 [full_pipeline.example.json](../configs/full_pipeline.example.json)。模型及前端环境配置完成后：

YOLO-World 的文本编码器也需要同一份 CLIP 权重。默认本地 Ultralytics 设置使用仓库下的 `weights/clip/ViT-B-32.pt`，可将已验证的 CLIP 文件链接到这里，避免新目录再次下载。该目录由 Git 排除；使用的 CLIP 文件须通过前端内置 SHA-256 校验。

```bash
python scripts/full_pipeline.py --config /path/to/full_pipeline.json \
  --output /path/to/new_empty_run --frontend-workers 2
```

这个入口要求新的输出目录，并依次执行第一阶段目录、全部原视频帧的 100DOH、TSM 动作、CLIP 视觉信号、cycles 事件划分；随后处理全部预测事件的 YOLO 对象、接触簇、变化候选、8B 三帧、数值充分性和累计记忆。不同事件读取的是**本次新生成**的共享检测结果，不读取历史运行的预测缓存。

最终检查检测 `reused_frames=0`、预测事件与提交事件 ID 一致、全部事件数值非空、全部主 Q 有限，以及增量记忆和从头重放一致。目录、帧引用、运行阶段失败依然如实记录；数值输出完成率不等于视觉准确率或几何真值完成率。完整数值入口强制采用 best_effort；若只复算严格几何，可在单独的 `compile-geometry` 配置中设置 `score_policy=strict`，保留严格组件诊断行为。
