# 运行指南

本页命令从仓库根目录运行。历史 Pipeline 文档中的 `<DATA_ROOT>` 等仅用于解释来源，新运行使用这里的显式配置。

## 1. 准备原始输入

每个 episode 包含双目 MKV、左右时间戳、IMU CSV、标定 YAML 和 meta.json；目录见 [数据格式](DATA_FORMAT.md)。事件三帧已经人工选好，分 left/right 目录保存，并在 Markdown 中列出原视频帧号。

复制 `configs/pipeline.example.json` 为 `configs/pipeline.local.json`，填写：

- `episodes`：每条录制路径、Markdown、左右三帧目录、episode ID、目标和 FPS。
- `episodes[].location_facts`（可选）：已有人工位置事实 JSON 的路径。入口校验其 episode、场景、目标、Markdown、原始位置照片和哈希后导入 `metadata.dataset_location_facts`；不能只凭输出结构需要自行编造位置事实。
- `preprocessing`：OpenVINS 可执行文件、Fast-FoundationStereo 源码和权重路径。
- `selector.model`：实际可用的 Responses 兼容服务、精确模型名和密钥环境变量名。
- `endpoints` 与 `routes`：Qwen 接口和场景、事件、时序阶段分工。

配置中的相对文件路径以配置文件所在目录为基准；支持 `${ENV_VARIABLE}`，缺失环境变量会直接报错。模型和服务名字沿用历史配置示例，必须改成自己的实际服务返回的模型名称，不保证示例服务可访问。

示例使用一个 `placement: remote` 的预先启动 Qwen 服务；“remote”表示调度器不管理该服务生命周期，它也可以位于本机。若该服务占用了深度处理同一张 GPU，资源检查会阻塞，应将服务放到其他 GPU / 主机，或显式配置 `local_gpu` 及完整 `local_server`。本仓库不会终止无关进程。

Qwen 场景阶段输出预算较大，示例上下文为 16,384；不能为了适配较小上下文静默删除证据。未配置 `facts` 时不会应用“最后一次找到”等历史实验规则。若需人工确认事实投影，提供适用于本批次且明确限定 episode IDs 的 facts 文件。

## 2. 准备和检查

```bash
export OPENVINS_BINARY=/absolute/ros_ws/build/ov_msckf/run_subscribe_msckf
export STEREO_REPO=/absolute/Fast-FoundationStereo
export STEREO_CHECKPOINT=/absolute/models/model_best_bp2_serialize.pth
# 通过自己的秘密管理方式设置 SELECTOR_API_KEY 和 QWEN_API_KEY。
python scripts/pipeline.py prepare --manifest configs/pipeline.local.json --output runs/annotation
python scripts/pipeline.py annotate check --config runs/annotation/pipeline.json
```

prepare 检查 FPS、目标、MD 三帧表、左右图片文件名/哈希、帧号范围、ID 和输出路径隔离，生成可审计 inventory。它不会请求模型。完整原始视频内容匹配在实际运行的 audit 阶段执行；`check` 是配置/来源检查，不是硬件、GPU 和数据全流程成功证明。

独立深度环境的 `preprocessing.depth.python` 必须保留虚拟环境入口路径。入口会将它转换为绝对路径，但不会追踪 Python 符号链接到系统解释器，以免丢失该虚拟环境中的 PyTorch 等依赖。

可加 `--probe` 检查远端 Qwen 的身份；这会发只读服务请求，不启动本地模型。`doctor` 仅检查 Python 模块和源代码完整性，也不联网。

## 3. 运行基础标注

先激活包含 ROS 依赖的环境，再运行：

```bash
python scripts/pipeline.py annotate run --config runs/annotation/pipeline.json
python scripts/pipeline.py annotate status --config runs/annotation/pipeline.json
python scripts/pipeline.py annotate validate --config runs/annotation/pipeline.json
```

阶段顺序为 `cpu → depth → direct → astra → scene → events → temporal → assemble`。其中 `astra` 是保留的阶段名，实际模型读 selector profile。CPU 阶段内部含 audit/vio/select/rectify。

产物：

```text
runs/annotation/
  pipeline.json, annotation_inventory.json, preprocessing_inventory.json
  run/
    state.json, run_manifest.json, receipts/, logs/
    preprocessed/<episode>/audit,vio,select,rectify,depth
    annotations/episodes/<episode>/input,selection,events,temporal
    annotations/json/<episode>.json
    annotations/exports/<episode>/...
    final_validation.json
```

提供适用的 facts 时才另外生成 confirmed_json / confirmed_exports。直接帧包引用上游文件；不能仅复制包索引后删除原 RGB、深度和校正产物。

## 4. 只执行预处理

可使用生成的单条 preprocessing_configs，或 `components/preprocessing/config/ego_preprocessing/demo.json` 模板（需填齐环境变量）：

```bash
python scripts/pipeline.py preprocess --config runs/annotation/preprocessing_configs/episode_001.json --through rectify
python scripts/pipeline.py preprocess --config runs/annotation/preprocessing_configs/episode_001.json --through depth --resume
```

该低层命令不承担完整调度器的 GPU 资源协调，应在独立空闲环境中使用。常规完整运行优先使用 `annotate run`。

## 5. 从已有 JSON 增加锚点

复制 `configs/anchor_sources.example.json` 与 `configs/anchor_models.example.json` 为 `.local.json`，填入源 JSON 分组、模型路径与 GPU 锁路径。按来源分组可避免不同批次同名 episode 覆盖。

```bash
export GROUNDING_DINO_DIR=/absolute/models/grounding-dino-tiny
export FASTSAM_WEIGHTS=/absolute/models/FastSAM-s.pt
export EGOMEMORY_GPU_LOCK="$HOME/.cache/egomemory/gpu0.lock"
python scripts/pipeline.py anchors-prepare --sources configs/anchor_sources.local.json --models configs/anchor_models.local.json --output runs/anchors
python scripts/pipeline.py anchors-run --root runs/anchors
python scripts/pipeline.py anchors-validate --root runs/anchors
```

新仓库默认用历史续跑的双语词表规则整理候选，不需要额外文本模型。每个事件候选并集分别作用于三帧，只匹配描述中的具体对象。规则可能漏掉同义词或指代；不把候选生成称作人工语义审核。空候选事件会阻止直接推理，需要查看 `selection_preparation_report.json`，按候选契约补充映射后再运行。

源 JSON 的图像路径要求为有效绝对路径且 SHA-256 匹配；迁移数据时应明确重定位这些路径并建立新运行版本，不能偷偷修改旧交付。

精简结果在 `runs/anchors/json/<group>/<episode>.json`；详细检测、掩码和叠加图在 detailed，导出审计在 compact_records，最终独立校验在 independent_final_validation.json。

## 6. 恢复与失败

同一输出根目录绑定输入、配置、模型和代码；修改后需要新运行目录或显式审计迁移。保留已通过哈希和回执检查的阶段，避免重复付费推理。

- 基础标注正常断点：再次使用同一 `annotate run` 命令。
- 修复后重试：使用 `--retry-failed`；先确认原请求/进程已经结束。
- 人工停止基础调度：创建 `run/stop_requested`，调度器排空自己的工作；继续前移除该标记。
- 锚点：同一配置再次 `anchors-run` 复用已完成条目与有效单帧回执。
- 核对进程时检查 PID、命令、启动 tick、boot ID 和锁；不要只读 status 字符串。
- 传输超时不证明远端推理已结束，不能据此立即重复发送请求。

历史特定批次的冻结协方差恢复 runner 不随仓库复制。遇到对应异常默认报错；如确需复用，须通过 `EGOMEMORY_RECOVERY_RUNNER` 提供有完整 policy 和代码哈希的专用 runner，不会自动降低阈值。

## 7. 验证范围

本仓库移植验证使用 CPU、合成证据与 mock 模型，包含配置登记、完整 additive schema 装配、候选提取、逐帧几何、精简导出、原值保持、哈希篡改检查及换路径运行。没有因打包而重跑 320 条，也没有在新机器验证真实 ROS/GPU/API 全链路。

2026-11-12 在原有硬件上另外完成一条真实样本的所有阶段，结果和限制见 [单条真实验证](REAL_SAMPLE_VALIDATION.md)。输入包含原始视频/IMU/标定及既有人工事件三帧；模型结果重新生成，导出路径修复后仅重跑装配并保留迁移审计。

每次真实运行仍应分别看 VIO 质量、模型返回和结构完整性报告。不能把完整处理数或 schema 通过转换成物体识别准确率。
