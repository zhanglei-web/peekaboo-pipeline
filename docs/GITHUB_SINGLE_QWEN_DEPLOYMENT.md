# 单 Qwen 搜索标注管线：模型下载、环境部署与启动说明

整理日期：2026-11-06。适用于计划发布到 GitHub 的项目说明。

本文以本项目已经使用的模型、后端、提示词和输出契约为依据。代码基线为 `11-5` 分支提交 `03364c7`；历史单 Qwen、含 rosbag 的批处理基线为标签 `11-4`。本文中的单 Qwen 指**一个本地 Qwen 服务完成场景、事件、时序三个阶段**，筛帧仍调用独立 Astra/Sol API。它不是完全离线的整条管线。

## 1. 先明确复现范围

```text
原始双目视频 + IMU + 标定 + 三帧图片/MD
    ↓ 输入核对
OpenVINS → 运动筛帧 → 双目校正 → Fast-FoundationStereo 深度
    ↓ 11-5 直接帧包，不生成 rosbag
Astra/Sol API：场景候选帧粗筛、精筛
    ↓
同一个本地 Qwen 服务
    ├─ 场景结构与物体信息
    ├─ 每次搜索的事件描述及潜在搜索区域
    └─ 按事件顺序更新空间关系和时序树
    ↓ CPU 组装、schema/证据/哈希/时序校验
主 JSON + 两种 end_version 目标 JSON
```

三个层次的验收不能混用：

| 层次 | 当前依据 | 能说明什么 |
|---|---|---|
| Qwen 服务 | 已有模型下载校验、CUDA 构建、服务和真实图像推理记录 | 本地模型可加载、可接收图像、可返回结果 |
| 历史完整单 Qwen 管线 | `11-4` 的 29 条批次及最终校验记录 | 该历史管线实际跑通过；其预处理有 rosbag 中转 |
| 11-5 直接帧管线 | 已有真实阶段联调、直接帧验证及当前双 Qwen 批处理运行 | 直接帧调度已实现；本文全本机单端点配置另做配置加载验证，尚未按本文在全新机器从零完成业务验收 |

模型单图返回成功不代表整条标注通过，JSON 结构通过也不代表模型语义已经人工审核。

## 2. GitHub 仓库需要包含什么

本次公开仓库使用下列布局，调度入口位于仓库根目录：

```text
<repository>/
├── README.md / VERSION / SOURCE_MANIFEST.json / THIRD_PARTY.md
├── pipeline_11_5.py / pipeline115.sh / pipeline115/
├── contracts/ / dependency_snapshots/ / dependencies.lock.json
├── docs/
└── vendor/
    ├── qwen/                     本地Qwen工具与冻结业务依赖
    ├── egomemory-core/            预处理源码子集
    ├── covariance_filter/         恢复代码与策略
    ├── astra-scene-memory/        筛帧相关依赖
    └── target_format/             目标格式辅助代码
```

`vendor/` 提供源码与溯源，不自动改变既有绝对路径导入。后文 `$PIPELINE_DIR` 对应仓库根目录，`$QWEN_DIR` 可对应 `vendor/qwen` 或你适配后的独立 Qwen 工程位置。OpenVINS、深度权重等外部环境仍需另行安装。

仅上传 `pipeline115/` 不够。运行还导入 Qwen 工程脚本、`astra_memory`、预处理工程和 `end_version` 的辅助模块。`dependency_snapshots/` 用于追溯冻结来源，当前入口不会自动从该目录恢复完整运行树。

权重、原始视频、私人图片、运行结果、密钥、虚拟环境和编译目录不需要作为代码上传。模型通过固定 revision 下载。各组件的许可证及第三方声明随对应源码保留；不要给整个聚合工程直接套用一个未经核对的许可证。

### 当前代码的路径迁移边界

当前版本运行于既有工作空间，仍有写在源码中的绝对路径。新仓库尚未成为 `git clone` 后即可执行的独立安装包。发布迁移至少需完成：

| 位置 | 需要适配的内容 |
|---|---|
| `pipeline115/common.py` | `CORE`、`QWEN`、`RECOVERY` |
| `annotate_batch.py`、`audit_inputs.py` | Qwen scripts 和 target_format 的导入路径；目标参考文件路径 |
| `annotation_contract.py`、`export_confirmed.py`、Qwen 业务脚本 | 实际依赖的契约、辅助模块、提示词和默认配置路径 |
| `pipeline115/engine.py` | 历史输出保护路径，迁移时保留等效的原始数据保护 |
| `dependencies.lock.json` | 新路径下逐项对应的冻结源码及内容哈希 |
| 管线配置、输入清单、逐条预处理配置 | 所有数据、输出、模型服务、OpenVINS、深度环境和权重路径 |

迁移时先逐项验证冻结内容，再记录新的路径和版本；不能通过删除哈希校验、批量信任未知源码来“解决”依赖错误。不要在当前正在运行的批次目录中改这些路径或切换分支。

后续命令以已经放置好的两个工程为前提。先按自己的磁盘位置设置变量；变量只是文档命令使用，当前程序不会自动读取它们来替换源码硬编码：

```bash
export QWEN_DIR="/absolute/path/to/qwen"
export PIPELINE_DIR="/absolute/path/to/processing"
export EGO_WS="/absolute/path/to/egomemory_ws"
export RUN_BASE="/absolute/path/to/new_run"
```

## 3. 硬件和环境基线

以下是本项目本地记录，不是最低硬件保证：

| 项目 | 已使用环境 |
|---|---|
| 操作系统 | Ubuntu 24.04，x86_64 |
| Python | 3.12.3；完整链路使用 ROS 2 Jazzy 对应的 Python 3.12 |
| GPU | RTX 5090 Laptop，约 24 GB 显存 |
| 内存 | 约 64 GB |
| Qwen CUDA Toolkit | 项目内 CUDA 12.8.1，nvcc 12.8.93 |
| 编译工具 | GCC/G++ 13.3、CMake 3.28.3、Git 2.43 |
| Qwen 推理后端 | llama.cpp CUDA，固定提交，见第 5 节 |
| 预处理 | ROS 2 Jazzy、OpenVINS、Fast-FoundationStereo |

主模型和视觉投影合计约 19.60 GB（十进制文件大小），还需为下载缓存、编译、深度权重和数据输出留空间。调度器的剩余磁盘门槛 20 GiB 不等于完整安装只需要 20 GiB。

同一张 24 GB GPU 上，深度与本地 Qwen 按阶段交替占用。Qwen 即使没有请求，权重仍占显存，因此完整管线由调度器启动和卸载它。CPU 预处理和远端筛帧在依赖允许时可以重叠。

## 4. 安装并下载 Qwen

### 4.1 系统工具和独立 Python 环境

在一台新 Ubuntu 主机上准备以下工具；已有环境不必重复安装或修改驱动：

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake git python3-venv curl xz-utils libssl-dev fonts-dejavu-core

cd "$QWEN_DIR"
python3 -m venv .venv
.venv/bin/python -I -m pip install -r requirements.lock.txt
.venv/bin/python -I -m pip install --no-deps -e .
bash scripts/check_env.sh
```

Qwen GGUF 推理环境不需要 PyTorch。不要把深度工程的 PyTorch/ROS 依赖混装进这个环境。当前 Qwen 工程已有 `requirements.lock.txt`，应随仓库提供；不要用无版本的最新依赖替代后声称是同一已验收环境。

### 4.2 模型身份与下载位置

本项目保存的下载锁如下，模型名称按原锁保留，不根据接口别名推断模型：

| 项目 | 固定值 |
|---|---|
| Hugging Face repo_id | `ggml-org/Qwen3.8-27B-GGUF` |
| revision | `a316ea527fde32c12487be7574e2db5ac2ad2a7e` |
| 主模型 | `Qwen3.8-27B-Q4_K_M.gguf` |
| 视觉投影 | `mmproj-Qwen3.8-27B-Q8_0.gguf` |
| 目标目录 | `$QWEN_DIR/models/qwen3.8-27b/` |

这里的 **Q8_0 是视觉投影的量化**，本机语言主模型是 **Q4_K_M**。这与远端使用 Q8_0 主模型是两回事。

优先运行已有的下载脚本：

```bash
cd "$QWEN_DIR"
.venv/bin/python -I scripts/download_model.py
```

脚本读取 `configs/model.lock.json`，按指定 revision 下载两个文件，核对文件大小、GGUF 文件头和完整 SHA256，写出 `models/qwen3.8-27b/manifest.json`。保留锁文件及下载缓存；不要删除锁后重新解析仓库最新 revision 来替代本次版本。

若需要在下载机单独取得文件，可以使用以下等价的固定下载命令，然后仍运行上述校验脚本：

```bash
cd "$QWEN_DIR"
.venv/bin/python -I - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

dest = Path.cwd() / "models/qwen3.8-27b"
for filename in ["Qwen3.8-27B-Q4_K_M.gguf", "mmproj-Qwen3.8-27B-Q8_0.gguf"]:
    hf_hub_download(
        repo_id="ggml-org/Qwen3.8-27B-GGUF",
        revision="a316ea527fde32c12487be7574e2db5ac2ad2a7e",
        filename=filename,
        local_dir=dest,
    )
PY
```

离线部署时拷贝这两个完整文件及原锁文件到相同目录，再运行下载脚本；文件完整时脚本只做本地校验。模型来源、revision 和哈希以仓库随附锁为验收依据；本文没有重新下载约 19.60 GB 权重。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 18973870528 | `c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747` |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 629247008 | `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb` |

## 5. 编译 llama.cpp CUDA 后端

```bash
cd "$QWEN_DIR"
nvidia-smi

# 仅在没有可用 CUDA Toolkit 时执行；下载到项目目录，不安装/升级 NVIDIA 驱动
python3 -I scripts/install_cuda_local.py

# 读取 configs/backend.commit，固定版本构建 llama-server
bash scripts/build_backend.sh
```

后端固定提交为 `29c720eaf72fcf8e4e61a56a88b994d92668261d`，构建结果为 `third_party/llama.cpp/build/bin/llama-server`。脚本按真实 GPU compute capability 编译，启用 `GGML_CUDA=ON`。已有后端提交与锁不符时会拒绝隐式更新。

```bash
"$QWEN_DIR/third_party/llama.cpp/build/bin/llama-server" --version
"$QWEN_DIR/third_party/llama.cpp/build/bin/llama-server" --list-devices
```

需要能看到 CUDA GPU。`nvidia-smi` 显示的 CUDA 版本是驱动能力，不等于已经安装相应 nvcc。

## 6. 单独验收 Qwen 服务

此步骤在空闲 GPU 上执行。**若完整管线正在运行，不要另起服务或停止它的模型。**

使用 processing 中已有的 `local_qwen.server.json`，避免误用 Qwen 通用示例的 8192 上下文。完整标注需要的配置是：

| 参数 | 值 |
|---|---|
| host / port / alias | `127.0.0.1` / `8080` / `local-vlm` |
| ctx_size / parallel | `16384` / `1` |
| n_gpu_layers | `99` |
| batch_size / ubatch_size | `512` / `128` |
| flash_attn / image_max_tokens | `auto` / `1024` |
| reasoning / reasoning_preserve | `off` / `false` |
| mmproj_offload | `true` |

```bash
bash "$QWEN_DIR/scripts/start_server.sh" --config "$PIPELINE_DIR/local_qwen.server.json"
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models

# 替换为一张自己有权使用的真实图片
export TEST_IMAGE="/absolute/path/to/test.jpg"
"$QWEN_DIR/.venv/bin/python" -I "$QWEN_DIR/scripts/run_image.py" \
  --config "$PIPELINE_DIR/local_qwen.server.json" \
  --image "$TEST_IMAGE" \
  --prompt 'Describe only the visible objects and their spatial relations. State uncertainty explicitly.'
```

健康接口应返回 `status=ok`，模型列表包含 `local-vlm`。该名称是服务别名；实际权重身份看下载锁、启动配置和加载日志。业务结果保存在 Qwen 工程的 `outputs/` 中，检查 `result.json`、原始响应和 metadata。

关闭思考同时依赖服务的 `--reasoning off --no-reasoning-preserve`，以及业务请求的 `chat_template_kwargs.enable_thinking=false`。本项目的专用客户端已经配置这些参数。

服务单独验收完成、请求结束且确认空闲后，使用该独立启动方式对应的停止脚本：

```bash
curl --fail http://127.0.0.1:8080/slots
bash "$QWEN_DIR/scripts/stop_server.sh"
```

完整管线不会接管此方式预先启动的服务；进入第 9 节前应让它正常退出。完整管线自己创建的服务由调度器管理，不使用这里的脚本跨任务停止。

## 7. 完整链路还需要的预处理环境

只运行第 6 节不需要 ROS、OpenVINS 和深度模型；从原始采集跑最终 JSON 时需要它们。

### 7.1 ROS 与 OpenVINS

先按 ROS 2 Jazzy 的 Ubuntu 安装方式准备 ROS 软件源和 Jazzy，再安装构建工具及项目依赖。已经安装 ROS 的环境可从下面开始：

```bash
source /opt/ros/jazzy/setup.bash
sudo apt-get install -y python3-colcon-common-extensions python3-rosdep \
  libeigen3-dev libopencv-dev libopencv-contrib-dev libboost-all-dev libceres-dev
```

OpenVINS 使用随发布版本保存/固定的源码放到 `$EGO_WS/src/open_vins/`。首次使用 rosdep 时需先完成 `rosdep init` 和 `rosdep update`，然后构建：

```bash
cd "$EGO_WS"
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src/open_vins --ignore-src -r -y --rosdistro jazzy
colcon build --packages-up-to ov_msckf --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
test -x "$EGO_WS/build/ov_msckf/run_subscribe_msckf"
```

本项目依赖已有的 OpenVINS 启动参数、回放和坐标单位处理，不能用上游最新源码替代后跳过验收。发布时还应保存该源码的提交及本地补丁。

### 7.2 深度环境与模型

深度 worker 使用自己的 Python，而管线包装脚本固定使用 `/usr/bin/python3`。激活某个 venv 不会自动改变包装脚本的解释器。需要分别确认系统 Python 的 ROS/调度依赖和深度 venv 的 PyTorch 依赖。

本机系统 Python 已使用 `numpy 1.26.4`、`Pillow 10.2.0`、`PyYAML 6.0.1`、`jsonschema 4.10.3`、`requests 2.31.0`、`psutil 5.9.8`、`scipy 1.11.4`，OpenCV 由已有系统环境提供。Ubuntu 对应的基础包可这样安装：

```bash
sudo apt-get install -y python3-numpy python3-opencv python3-pil python3-yaml \
  python3-jsonschema python3-requests python3-psutil python3-scipy
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -c 'import rclpy, cv2, numpy, PIL, yaml, jsonschema, requests, psutil, scipy; print("pipeline imports OK")'
```

安装结果版本需记录；上述 apt 命令本身不是锁定这些版本的可重现系统镜像。

深度已有环境使用 `torch 2.8.0+cu128`、`torchvision 0.23.0+cu128`、`numpy 1.26.4`、`triton 3.4.0`，以及模型源码的其他依赖。新环境的准备顺序：

```bash
cd "$EGO_WS"
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install 'numpy==1.26.4'
mkdir -p third_party models/fast_foundation_stereo
git clone https://github.com/NVlabs/Fast-FoundationStereo.git third_party/Fast-FoundationStereo
git -C third_party/Fast-FoundationStereo checkout --detach a290ba04c1b3ad1ec41a33974a157b2917b624d4
```

按该固定源码的 requirements 补齐深度依赖，并核对 resolver 没有替换上述 torch/CUDA/NumPy 基线；不能把整套 egomemory 研究工程的通用 requirements 当作这条链路的最小依赖锁。

预处理源码使用发布时固定的、包含 `daaam.preprocessing` 的本项目版本；它不是只克隆原始 DAAAM 上游就能替代的代码：

```bash
"$EGO_WS/.venv/bin/python" -m pip install --no-deps -e "$EGO_WS/src/egomemory-core"
"$EGO_WS/.venv/bin/python" -m pip install gdown imageio
cd "$EGO_WS"
.venv/bin/python -m gdown --folder \
  https://drive.google.com/drive/folders/1WWT0I8VB1k7cA2LqnjhUHOux8Sz-BZMw \
  -O models/fast_foundation_stereo/20-30-48
```

保留同目录的 `cfg.yaml` 和 `model_best_bp2_serialize.pth`。固定 SHA256 分别为：

```text
cfg.yaml
d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc

model_best_bp2_serialize.pth
98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692
```

```bash
sha256sum "$EGO_WS/models/fast_foundation_stereo/20-30-48/cfg.yaml" \
  "$EGO_WS/models/fast_foundation_stereo/20-30-48/model_best_bp2_serialize.pth"
source /opt/ros/jazzy/setup.bash
"$EGO_WS/.venv/bin/python" -c 'import rclpy, torch; print(torch.__version__, torch.cuda.is_available())'
```

逐条预处理配置的 `openvins_binary`、`depth.repo`、`depth.checkpoint`、`depth.python` 必须指向实际文件。11-5 强制 `ros_output.enabled=false`、`mode=none`，不会走旧文档的 rosbag 导出命令。

## 8. 输入、Astra/Sol 和单 Qwen 配置

### 8.1 输入目录

```text
dataset/
├── baai_ego_task/<episode_id>/
│   ├── meta.json
│   ├── videos/left.mkv、right.mkv
│   ├── timestamps/left_timestamps.txt、right_timestamps.txt
│   ├── imu/imu0.csv
│   └── calibrations/              相机、相机/IMU 和 IMU 标定
├── 标注数据/<scene>/<target>/<episode_id>/
│   ├── <episode_id>_三帧.md
│   ├── left/                     每次搜索的开始/进行/结束图片
│   └── right/                    现有输入校验仍核对右目证据
└── 原始位置照片/<scene>/          本批规则下由文件名记录初始位置
```

场景/事件图像标注使用左目；双目深度仍需要同步的原视频。右目缺图或末尾缺帧应保留独立诊断并按明确作用域处理，不能改原始视频、填造右目或借用别的会话。当前登记器还会因为右目不完整跳过条目；本文不声称已经修复这一登记限制。

准备 `annotation_inventory.json`、`preprocessing_inventory.json` 和逐条预处理 JSON。前者包含唯一会话身份、目标、MD/hash、三帧索引及图片 hash；后者通过 `records[].id` 指向同一采集及其配置。保持场景及原始采集唯一对应，不能只凭同名会话跨子集匹配。

已有清单可直接复用并用 `--only` 选择一条。当前 `pipeline115.enroll_followup` 是本批布局登记器，仍读取 `processing/inventory.json` 的历史配置作为模板；新仓库还需把模板入口独立整理后才能当作通用数据导入命令。不要仅把 `--dataset` 换成任意目录就认为登记一定完整。

### 8.2 筛帧 API

提供独立筛帧配置文件，并在管线 JSON 的 `astra_config` 指向它。当前实现只读取其中 `model`，沿用严格的 Responses SSE 传输；所需字段示意：

```json
{
  "model": {
    "name": "gpt-6.1-sol",
    "base_url": "https://YOUR_SELECTOR_GATEWAY/v1",
    "api_key_env": "ZYAPI_API_KEY",
    "transport": "responses",
    "reasoning_effort": "low",
    "max_output_tokens": 5000,
    "timeout_s": 300
  }
}
```

`base_url` 应是网关的 API 根，不包含 `/responses` 后缀。网关需要支持图像输入、JSON Schema 和 Responses SSE，并在完成响应中返回真实模型身份。当前 SSE 实际连接/read/总超时策略分别为 30/1200/1800 秒，以 `pipeline115/selector_transport.py` 为准；`selector.py` 将筛帧输出预算固定为 5000。

模型名是当前项目网关使用的配置值，不能据此假定其他提供方同名可用。模型或网关变更后，应作为新配置重新验收。密钥通过环境提供：

```bash
read -rsp 'Selector API key: ' ZYAPI_API_KEY
export ZYAPI_API_KEY
```

仅使用本机 Qwen 时不需要远端 `QWEN_API_KEY`。

### 8.3 生成单 Qwen 配置

在已经完成第 2 节路径适配、清单准备和依赖验证的工程中，复制样例并把所有 Qwen 阶段指向 `local`。以下命令只生成新配置，不运行模型：

```bash
export ANNOTATION_INVENTORY="/absolute/path/to/annotation_inventory.json"
export PREPROCESSING_INVENTORY="/absolute/path/to/preprocessing_inventory.json"
export SELECTOR_CONFIG="/absolute/path/to/selector.json"
export RUN_CONFIG="$RUN_BASE/single-qwen.json"
mkdir -p "$RUN_BASE"

python3 - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ['PIPELINE_DIR'])
cfg = json.loads((root / 'pipeline115.example.json').read_text())
cfg['output_root'] = str(Path(os.environ['RUN_BASE']) / 'run001')
cfg['annotation_inventory'] = os.environ['ANNOTATION_INVENTORY']
cfg['preprocessing_inventory'] = os.environ['PREPROCESSING_INVENTORY']
cfg['astra_config'] = os.environ['SELECTOR_CONFIG']
cfg['facts'] = None
cfg['routes'] = {stage: 'local' for stage in ['scene', 'events', 'temporal']}
cfg['dispatch'] = {'mode': 'episode', 'endpoints': ['local']}
cfg['endpoints'] = {'local': cfg['endpoints']['local']}
cfg['gpu']['local_server_config'] = str(root / 'local_qwen.server.json')
cfg['gpu']['lock_path'] = str(Path(os.environ['EGO_WS']) / '.pipeline115-gpu0.lock')
Path(os.environ['RUN_CONFIG']).write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
PY
```

`RUN_BASE` 必须是新的独立目录，不能在受保护的历史数据目录或 processing 仓库内创建输出。所有使用同一 GPU 的批次必须共用同一个锁文件路径。示例显式关闭人工事实投影，防止把历史批次规律套给新数据。

Qwen 请求保持原预算：场景 `10000`、每事件描述 `6144`、每事件时序 `3072`，temperature `0`、seed `42`，关闭思考。16K 是完整单槽上下文，不是 16K 输出；图片、提示词、schema 和输出都需要容纳。容量不足进入失败诊断，不自动删图、缩预算或漏字段。

## 9. 从终端跑通一条完整数据

先确认独立 Qwen 服务已退出，GPU 未被其他计算任务占用。完整管线会按需启动自己的本机服务。

```bash
cd "$PIPELINE_DIR"
bash pipeline115.sh check --config "$RUN_CONFIG"
```

`check` 核对冻结依赖、配置结构和密钥可用性；它不做业务推理，也不能替代输入内容审计、模型加载和完整验收。单本地配置的 `--probe` 也不会替你启动本机模型。

```bash
# 必须替换为输入清单中真实存在的 ID
export EPISODE_ID="YOUR_EPISODE_ID"
bash pipeline115.sh run --config "$RUN_CONFIG" --only "$EPISODE_ID"
```

在另一终端中可用绝对路径查看同一配置的状态：

```bash
bash "$PIPELINE_DIR/pipeline115.sh" status --config "$RUN_CONFIG"
```

完成后执行独立校验：

```bash
bash "$PIPELINE_DIR/pipeline115.sh" validate --config "$RUN_CONFIG"
```

单条运行以 `--only` 选择范围生成独立 run manifest。该输出目录以后不能直接改成全批范围；要批处理，换新输出根和配置，再去掉 `--only`。不要对当前已在运行的 72 条目录启动这个新配置。

## 10. 输出与通过标准

```text
run001/
├── state.json、run_manifest.json、config.json
├── logs/<episode_id>/                    每阶段实际日志
├── receipts/<episode_id>/                阶段回执和哈希
├── resources/                            本机服务身份、启动和停止记录
├── preprocessed/                         VIO、校正、深度等上游结果
├── annotations/
│   ├── target.schema.json
│   ├── json/<episode_id>.json            主 JSON
│   ├── exports/<episode_id>/
│   │   ├── <episode_id>.json             搜索过程目标格式
│   │   └── experiment_records.json       实验记录目标格式
│   └── episodes/<episode_id>/            输入包、筛帧、场景、事件、时序证据
└── final_validation.json
```

单条通过要求：该条所有阶段真实完成；主 JSON 和两份导出存在且通过 schema/目标字段校验；目标名、事件数、三帧原标签、图片和来源哈希对应；空间树及时间关系审计通过；最终 `validate` 成功。不能只看 PID 消失、健康检查或某个输出文件存在。

输出继续保留物体/事件信息、`nodes → children` 嵌套树、未定位节点、逐事件空间变化、原视频帧的左闭右开有效区间、潜在搜索区域及实验字段接口。没有证据的参与者/搬移时间/区域测量等保持空缺，不能靠 schema 有字段就声称已采集到真值。

基础视觉结果未知时保持 `unknown`。只有用户明确确认且写入本次 facts 作用域的批次，才投影“前面未找到、最后一次找到”，生成额外 `confirmed_json/`、`confirmed_exports/`。这不是任意新数据的硬编码规则。

直接帧包引用上游图像和深度文件，不是随便搬走 `input/` 就独立可用的归档；需要保留其引用的上游产物及哈希。

## 11. 续跑与常见问题

正常停止后，以完全相同的配置、范围和输出目录继续：

```bash
bash "$PIPELINE_DIR/pipeline115.sh" run --config "$RUN_CONFIG" --only "$EPISODE_ID"
```

失败时先查 `state.json`、`logs/<episode_id>/` 以及模型阶段的原始响应/transport_attempts；原因修复后再恢复失败阶段：

```bash
bash "$PIPELINE_DIR/pipeline115.sh" run --config "$RUN_CONFIG" \
  --only "$EPISODE_ID" --retry-failed
```

| 现象 | 核对方式 |
|---|---|
| 缺少 Qwen 模型/投影 | 检查两份 GGUF、固定锁、大小与 SHA256，不只检查文件名 |
| CUDA 构建失败 | 确认 GPU 架构受 Toolkit 支持，后端提交与锁一致 |
| 8080 占用或已有 llama-server | 核对已有任务和进程身份；调度器不会接管外部服务 |
| GPU occupied | 确认其他计算任务和常驻模型；保持深度/Qwen 互斥 |
| Context 超限/输出截断 | 核对单槽 16384、图片输入和输出预算；不把截断记为成功 |
| 筛帧超时、无 model、空 JSON | 检查完成事件、真实模型身份与原响应；未通过验证不可发布结果 |
| Frozen dependencies changed | 比对路径及冻结源码；建立可追溯迁移，不关闭校验 |
| 右目末尾缺帧 | 分开看左目事件证据与双目预处理；保留缺项，不用相邻帧冒充 |
| VIO 协方差失败 | 只按冻结 recovery_policy 排除坏样本，原阈值不变，禁止跨坏样本插值 |

HTTP 超时不证明模型请求已结束；先确认服务侧状态，避免盲目重复提交。不要用强杀其他任务的方法让本任务继续。

## 12. 发布前记录与本次文档验证

建议在首个 GitHub Release 固定代码提交、Qwen model lock、llama.cpp commit、预处理源码/补丁、深度权重哈希、Python 依赖及一条脱敏的成功验收记录。README 应明确新机器路径适配和依赖打包是否完成。

本次文档依据本机保存的下载锁、环境记录、服务启动代码、11-4 历史验收、11-5 调度/校验代码整理；未下载模型、安装依赖、重启模型或另外发起业务推理。随文档检查命令接口、JSON 配置和 shell/Python 示例语法。**不能将本文编写或配置检查说成新机器整条业务验收。**

相关说明：[输出规范](OUTPUT_SPEC.md)、[字段字典](FIELD_DICTIONARY.md)、[11-5 现有工程说明](PIPELINE_11_5.md)、[整条 Qwen 调度](EPISODE_DISPATCH_11_5.md)。
