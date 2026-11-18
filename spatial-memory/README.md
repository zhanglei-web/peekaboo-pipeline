# Peekaboo 两阶段空间记忆

本仓库把当前代码整理为一条可检查的数据链：

```text
录制前缀 → 第一阶段候选区域目录 → 空间记忆初态
完整视频 → 事件区间、动作和接触证据 → 带时间戳的候选图像
候选图像 + 目录 → Qwen3.8-8B 同时选择 Pre / PNR / Post 及候选区域关联
最终三帧 + 固定参考几何 → 每个区域的几何 q → 按事件时间重放累计 Q
```

`online_annotation/` 是当前第一阶段 B 版代码。`event_frontend/` 是当前 100DOH、TSM、视觉信号与 cycles 事件解码，以及对象、接触簇和图像变化候选代码。`spatial_memory/` 实现两阶段衔接、三帧选择、固定参考的几何覆盖与累计记忆。方法定义见 [第一阶段](docs/stage1_method.md)、[第二阶段](docs/stage2_method.md)、[几何评分](docs/geometry_definition.md)。

## 安装与外部输入

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
python -m unittest discover -s tests -v
```

原始视频、100DOH／TSM／CLIP／YOLO 权重、本地 Qwen 服务配置与密钥、双目深度和分割掩码不进入仓库。运行视频前端还需对应模型依赖及 GPU 环境。几何代码接收**已校正的双目深度**、有效掩码、物体掩码、支撑面掩码、前景掩码和配准位姿；它不会把未建立可信几何的候选区域强制评为 0。

## 第一阶段：建立目录和初态

```bash
python -m online_annotation initialize \
  --session /path/to/session --output runs/episode/stage1 \
  --candidate-mode brief --model-root /path/to/local-qwen-service
python -m spatial_memory initialize \
  --catalog runs/episode/stage1/catalog.v0.json \
  --output runs/episode/initial_memory.json
```

模型目录须包含现有本地 Qwen 服务使用的 `config.json` 和由其引用的凭据文件。可以用 `--base-url` 指向已启动服务。初态中的候选区域有身份与边界语义，`Q=null`；模型提出的框不是几何评分的分母。

## 第二阶段：事件到三帧

以下命令使用仓库中的事件前端源码；路径和权重由运行者提供。`export-specs` 从事件预测、源帧索引和动作结果生成逐事件输入，不读取人工真值。

```bash
python -m event_frontend detect --session /path/to/session \
  --repository /path/to/100DOH --checkpoint /path/to/100DOH.pth \
  --output runs/episode/detection
python -m event_frontend recognize-actions --run runs/episode/detection \
  --checkpoint /path/to/tsm.ckpt --labels-dir /path/to/tsm-labels \
  --output runs/episode/actions
python -m event_frontend extract-signals --run runs/episode/detection \
  --clip-checkpoint /path/to/clip.pt --output runs/episode/signals
python -m event_frontend segment-events --run runs/episode/detection \
  --actions runs/episode/actions --signals runs/episode/signals \
  --output runs/episode/events
python -m event_frontend export-specs --run runs/episode/detection \
  --actions runs/episode/actions --events runs/episode/events \
  --output runs/episode/event_specs
```

对一个 `event_001.json`：

```bash
python -m spatial_memory prepare-event \
  --spec runs/episode/event_specs/event_001.json \
  --output runs/episode/event_001/candidates
export QWEN_API_KEY='your-local-service-key'
python -m spatial_memory select \
  --candidates runs/episode/event_001/candidates/event_candidates.json \
  --catalog runs/episode/stage1/catalog.v0.json \
  --base-url http://127.0.0.1:8089/v1 \
  --output runs/episode/event_001/selection
```

`prepare-event` 总是包含事件起止帧及前后上下文。若对象接触簇已确定，可将 `contact_cluster_start_source_frame_id`、`contact_cluster_end_source_frame_id` 写入事件 spec；它们同样被保留为重要参考帧。`important_frames` 可加入动作转折、局部变化峰值或观察停顿帧，格式为 `{ "source_frame_id": 123, "tag": "change_peak" }`。如需生成对象／接触簇与变化候选，`event_frontend clip-event`、`recognize-objects`、`prepare-contact-cluster`、`rank-candidates` 提供当前前端实现。所有重要帧都只是候选，三种角色完全由 Qwen3.8-8B 从实际提交的图像中选择。模型若返回 `uncertain`，不提交几何评分。

选择请求默认将每张图的最大边缩为 768，以适配多图上下文预算，保留全部候选帧。可用 `--image-max-edge` 修改，该参数写入请求身份。HTTP 失败和原始模型响应会落盘；不能把上下文超限时遗漏图像的请求当成完整三帧选择。

## 几何 q 与累计 Q

对**每个**实际关联的搜索单元提供一个 job JSON：

```json
{
  "search_unit_id": "episode:R001",
  "scope": "original_support_footprint",
  "reference_npz": "reference.npz",
  "frames": {"pre": "pre.npz", "pnr": "pnr.npz", "post": "post.npz"},
  "quality_audit": {"accepted": true, "reasons": [], "source": "path/to/geometry_audit.json"}
}
```

`reference.npz` 包含 `depth`（米）、`valid`、`object_mask`、`support_mask`、`K`（3×3）。它描述操作初态的固定局部外部支撑占地。三份帧 NPZ 各包含 `source_frame_id`、`depth`、`valid`、`foreground`、`K`、`T`（4×4，**目标相机坐标相对参考坐标**）和 `pose_valid`。`prepare-geometry` 读取逐帧左右目生成的深度文件和掩码，用静态背景 RGB-D 特征估计 `T`，核对帧号，输出上述 NPZ 与 job。输入 JSON 的 `reference` 需要 `source_frame_id`、`left_image`、`depth_npz`、`object_mask`、`support_mask`、`dynamic_mask`；`frames` 下的 `pre`／`pnr`／`post` 各需要 `source_frame_id`、`left_image`、`depth_npz`、`foreground_mask`、`dynamic_mask`；顶层需要 `search_unit_id`、`scope` 和 3×3 `K`。这里的 `depth_npz` 至少含 `depth` 和 `valid`。位姿质量、深度有效性与固定分母必须先满足要求。容器内部或整片候选范围需要各自的几何定义；当前外部占地代码不替代它们。

```bash
python -m spatial_memory prepare-geometry \
  --sources /path/to/event_001/unit_R001_sources.json \
  --selection runs/episode/event_001/selection/selection.json \
  --output runs/episode/event_001/unit_R001_inputs
python -m spatial_memory score \
  --job runs/episode/event_001/unit_R001_inputs/job.json \
  --selection runs/episode/event_001/selection/selection.json \
  --output runs/episode/event_001/unit_R001
python -m spatial_memory assemble-event \
  --candidates runs/episode/event_001/candidates/event_candidates.json \
  --selection runs/episode/event_001/selection/selection.json \
  --assessments runs/episode/event_001/unit_R001/assessment.json \
  --available-after-s 15.2 --output runs/episode/event_001/committed_event.json
python -m spatial_memory replay \
  --catalog runs/episode/stage1/catalog.v0.json \
  --event-records runs/episode/event_001/committed_event.json \
  --output runs/episode/memory.json
```

每个关联区域都要提供一条评估记录，无法计算时 `q=null` 且注明原因。`q` 是本次三帧固定参考点的**几何可见并集比例**；`Q=max(Q_previous,q)` 是同一稳定单元跨事件的历史最佳单次分数。等级标注与几何覆盖不能进入同一数值通道。未检查单元保持原状态；首次检查不可计算时仍未知。物体层等权平均已确认单元，大区层等权平均物体；未知项使主均值为 `null`，同时给出上下界。`--cutoff-s` 只读取当时已经提交的事件。修订事件须从原目录按顺序重放。

几何任务还须提供来自上游支撑面及跨帧审计的 `quality_audit`，并由 `prepare-geometry` 原样带入 job。只有审计明确通过且无失败原因时，诊断覆盖率才可写成有效 `q`；缺少审计、支撑面歧义、深度容差敏感或静态表面跨帧不一致时保留 `q=null`。示例中的 `accepted=true` 必须来自实际审计结果。该接口消费审计结果，本仓库尚未自动生成完整的支撑面歧义审计。

## 验证范围

`python -m unittest discover -s tests -v` 覆盖第一阶段引用与未知状态、三帧候选引用、合成几何并集、`max` 更新、未知值和时间截止。此处没有实际运行 Qwen3.8-8B、100DOH 或双目模型，也没有由这些单元测试得出视觉准确率。实际视频评测仍应独立人工复核三帧角色、区域关联和几何参考范围。

2026-11-18 的真实输入接口测试、暂停与恢复后的结果见 [实测记录](docs/SMOKE_TEST_STATUS.md)。可复用脚本为 `scripts/smoke_real.py` 和 `scripts/smoke_geometry.py`；运行视频实测会调用模型服务。`smoke_real.py --stage1-root` 可复用已有有效目录，来源单独记录，避免选帧重试时重新调用第一阶段模型。
