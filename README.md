# Peekaboo Pipeline

从原始第一人称双目视频、IMU、标定及人工事件三帧，生成场景空间关系、事件描述、时序关系和二维物体锚点的精简 JSON。

## 两阶段空间记忆

新增 **[spatial-memory/](spatial-memory/README.md)**：第一阶段生成候选搜索区域目录，第二阶段处理事件候选帧，由 **Qwen3-VL-8B-Instruct** 同时选择 Pre／PNR／Post，并自动接入双目深度、物体分割、固定局部支撑占地及累计记忆。安装、命令行、输入格式和代码位于该目录，使用 `cd spatial-memory && pip install -e .` 安装。

该模块默认每次检查输出数值：优先几何计算，条件不足时输出明确标记的近似几何或 8B 视觉估计，并分别保存严格几何累计值与主 Q 的来源。25 项单元测试通过；三段原始录制共 2,498 帧，18/18 个预测事件全部完成三帧选择、评分和累计记忆，历史检测缓存复用为 0，独立记忆重放一致。19 项单元评分包括 2 项严格几何、1 项近似几何和 16 项视觉估计，默认值回退为 0 项。详见 [三段完整实测报告](spatial-memory/docs/FULL_PIPELINE_NUMERIC_VALIDATION.md)；本次运行通过不代表已测得视觉语义准确率。下面的 `portable/` 文档描述视频标注入口。

三段的原始人工事件均为 6 个，预测数量为 5/5/8；按 tIoU≥0.3 一对一匹配，11 对成功，Micro-P/R/F1=0.611，tIoU≥0.5 时 F1=0.444。[逐事件对照与动作图](spatial-memory/docs/event-comparison/README.md) 列出了人工/预测区间、TSM 动作分数、三帧跨事件现象和充分性数值，并提供 CSV 与 PDF。

原规则选帧也已在相同 18 个事件、接触簇与特征上实际重跑，模型调用为 0。[规则 vs 当前 VLM](spatial-memory/docs/rule-vs-vlm/README.md) 给出全部帧号与对照图：PNR 落在匹配人工区间内为规则 6/11、VLM 8/11（匹配门槛 0.3），门槛 0.5 时均为 5/8；规则有 8 组 N 与端点重合。这些是时间诊断，尚非三帧语义准确率。

## 视频标注入口

当前维护入口位于 **[portable/](portable/README.md)**。2026-11-12 已完成一条真实数据的全部阶段：4 个事件、12 张事件图，最终 JSON 与历史结果的核心字段一致，并通过同一 Schema 和独立证据校验。66 项自动测试通过。

```text
双目视频 + IMU + 标定 → OpenVINS → 运动关键帧 → 双目校正与深度
→ 语义筛帧 → Qwen 场景 / 事件 / 时序关系 → 基础 JSON
→ 描述中的物体候选 → Grounding DINO + FastSAM → 精简增强 JSON
```

事件三帧仍由已有人工标注提供；不是仅凭视频自动发现全部搜索事件。锚点只针对描述明确提及的物体，`size_wh_normalized` 为二维检测框的相对宽高。

## 开始使用

```bash
cd portable
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_repository.py
python scripts/pipeline.py doctor
pytest -q
```

上面验证 CPU 契约和源码完整性。真实处理还需要 ROS/OpenVINS、模型权重与视觉语言模型服务，按 [安装说明](portable/docs/INSTALL.md) 和 [运行指南](portable/docs/RUNNING.md) 配置后运行。

## 完整文档

- [历史高分批次事件／动作图：TSM＋规则＋Sol 选 N](spatial-memory/docs/historical-high-score/README.md)

- [原始数据到最终 JSON 的完整 Pipeline](portable/docs/PIPELINE.md)
- [单条真实验证与修复记录](portable/docs/REAL_SAMPLE_VALIDATION.md)
- [输入输出与坐标格式](portable/docs/DATA_FORMAT.md)
- [精简锚点 JSON 字段](portable/docs/COMPACT_JSON.md)
- [描述物体检测与分割流程](portable/docs/ANCHOR_PIPELINE.md)
- [来源、迁移和验证范围](portable/docs/PORTABILITY.md)
- [机器可读验证摘要](portable/provenance/real_sample_validation.json)

真实样本的 JSON 结构通过，但 OpenVINS 位姿覆盖门限未通过，退出码为 `-11`；下游只使用具有有效位姿的帧。不能将结构通过当作轨迹质量或人工语义验收，也没有测量物体识别准确率。具体结果和限制已记录在验证文档中。

## 目录与历史版本

`portable/` 包含可迁移入口、组件、配置模板、文档与测试。根目录原有的 `pipeline115/`、`vendor/`、`contracts/` 等保留为 11-5 历史快照，其部署边界见 [原版说明](docs/README_11_5_SNAPSHOT.md)。新运行请按 `portable/` 文档配置，避免混用两个版本的入口和依赖。

历史交付共 320 条；本次只重新测试一条，并未批量重跑全部数据。原始视频、真实标注 JSON、模型权重、服务密钥、虚拟环境和运行日志不随源码发布。来源及许可说明见 [新组件说明](portable/THIRD_PARTY_NOTICES.md) 与 [原第三方说明](THIRD_PARTY.md)。
