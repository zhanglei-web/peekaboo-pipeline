# Peekaboo 两阶段空间记忆

本仓库把当前代码整理为一条可检查的数据链：

```text
录制前缀 → 第一阶段候选区域目录 → 空间记忆初态
完整视频 → 事件区间、动作和接触证据 → 带时间戳的候选图像
候选图像 + 目录 → Qwen3-VL-8B-Instruct 同时选择 Pre / PNR / Post 及候选区域关联
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

原始视频、100DOH／TSM／CLIP／YOLO 权重、本地 Qwen 服务配置与密钥、双目深度和分割掩码不进入仓库。运行视频前端还需对应模型依赖及 GPU 环境。自动几何入口从最终三帧、原始双目录制、标定与源帧手部检测生成深度、掩码、固定参考和配准；NPZ 输入入口也保留用于组件复算。

## 指定的 8B 模型

本次确认的模型是 **Qwen/Qwen3-VL-8B-Instruct**。使用官方 GGUF 仓库的 Q8_0 权重与 F16 视觉投影，文件版本和 SHA-256 见 [权重清单](configs/qwen8b_manifest.json)。将两份同名权重放入模型目录，并将清单复制为该目录的 `manifest.json`。可从 [官方 GGUF 仓库](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/f982a07559d4a2f6c8744d840bf6fccab30eea96) 下载。

```bash
python scripts/qwen8b_service.py start \
  --model-root /path/to/Qwen3-VL-8B-Instruct \
  --llama-server /path/to/llama-server
python scripts/qwen8b_service.py status --model-root /path/to/Qwen3-VL-8B-Instruct
```

服务绑定本机 8090 端口，两个 32768 token 槽，每张图使用 1024 图像 tokens。启动前验证完整权重哈希；配置和凭据保存在外部模型目录。运行记录保留实际模型名与权重版本，不以修改别名替代模型。停止命令将 `start` 换成 `stop`。

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
  --base-url http://127.0.0.1:8090/v1 \
  --output runs/episode/event_001/selection
```

`prepare-event` 总是包含事件起止帧及前后上下文。若对象接触簇已确定，可将 `contact_cluster_start_source_frame_id`、`contact_cluster_end_source_frame_id` 写入事件 spec；它们同样被保留为重要参考帧。`important_frames` 可加入动作转折、局部变化峰值或观察停顿帧，格式为 `{ "source_frame_id": 123, "tag": "change_peak" }`。如需生成对象／接触簇与变化候选，`event_frontend clip-event`、`recognize-objects`、`prepare-contact-cluster`、`rank-candidates` 提供当前前端实现。所有重要帧都只是候选，三种角色完全由 Qwen3-VL-8B-Instruct 从实际提交的图像中选择。模型若返回 `uncertain`，不提交几何评分。

选择请求默认将每张图的最大边缩为 768，以适配多图上下文预算，保留全部候选帧。可用 `--image-max-edge` 修改，该参数写入请求身份。HTTP 失败和原始模型响应会落盘；不能把上下文超限时遗漏图像的请求当成完整三帧选择。

## 几何 q 与累计 Q

### 自动生成几何证据并更新记忆

在几何运行环境安装 `requirements-inference.txt` 及本包，并配置 Fast-FoundationStereo 官方源码／序列化权重、FastSAM-x 权重。将 [配置模板](configs/geometry.example.json) 的路径替换为实际路径。录制目录需要 `videos/left.mkv`、`videos/right.mkv`、左右目采集时间戳、双目 Double-Sphere 标定；初态稳定加速度与相机—IMU 标定用于约束局部水平承托面方向。

```bash
python -m spatial_memory compile-geometry \
  --session /path/to/session --config /path/to/geometry.json \
  --event runs/episode/event_001/candidates/event_candidates.json \
  --selection runs/episode/event_001/selection/selection.json \
  --catalog runs/episode/stage1/catalog.v0.json \
  --memory runs/episode/initial_memory.json \
  --detections runs/episode/detection/detections.jsonl \
  --model-root /path/to/Qwen3-VL-8B-Instruct \
  --base-url http://127.0.0.1:8090/v1 --output runs/episode/event_001/geometry
```

计算链为：最终源帧号 → 采集时间同步双目 → 标定校正 → 1280×960 Fast-FoundationStereo、4 次迭代、左右一致性 → 640×480 深度 → 8B 定位操作物体、FastSAM 实例掩码、校正后的源帧手部检测 → 仅初态的邻近支撑面拟合与固定占地采样 → 静态背景 RGB-D 配准 → 2 cm 深度一致性与前景否决 → 三帧逐点可见并集 → `q` 和累计 `Q`。原始模型响应、手部检测哈希、IMU 方向依据、逐点状态和质量诊断均落盘。

局部物体占地作为单独的搜索单元和事件发现的物体分组保存，不把它的 q 填入整片台面或容器候选。来自多个文字区域的同一几何占地在事件内去重，并保留区域关联歧义。后续事件可通过 `--registry` 与当前记忆传入保存的参考；只有静态配准、空间重合及平面距离支持同一范围时才复用原分母。范围或内容变化时需新的累积区间，当前没有自动内容变化检测。

有效参考已建立后，局部深度未知的点保留在分母、不进入分子；未知比例只作诊断，不单独丢弃连续评分。参考物体、支撑平面或配准不可建立时结果仍为不可计算。容器内部需要专门的参考几何，目前自动入口只实现外部局部支撑占地。

### 已有 NPZ 的组件入口

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

`python -m unittest discover -s tests -v` 覆盖第一阶段引用、具名坐标修订、三帧候选、双目分辨率换算、静态深度检查、标定 IMU 方向、固定分母、`max` 更新与时间截止。2026-11-18 已实际运行 Qwen3-VL-8B-Instruct、Fast-FoundationStereo 和 FastSAM；手部检测、动作与事件区间使用已有真实缓存。该实测未给出独立视觉语义准确率，结果和失败均见 [8B 实测记录](docs/QWEN8B_VALIDATION.md)。

2026-11-18 的真实输入接口测试、暂停与恢复后的结果见 [实测记录](docs/SMOKE_TEST_STATUS.md)。可复用脚本为 `scripts/smoke_real.py` 和 `scripts/smoke_geometry.py`；运行视频实测会调用模型服务。`smoke_real.py --stage1-root` 可复用已有有效目录，来源单独记录，避免选帧重试时重新调用第一阶段模型。
