# 安装与外部资源

参考环境是 Linux、Python 3.12、ROS 2 Jazzy 和 NVIDIA CUDA GPU。完整推理依赖外部 OpenVINS 可执行文件、Fast-FoundationStereo 源码与权重、Grounding DINO/FastSAM 权重，以及可用的视觉语言模型服务。仓库包含适配代码，不包含这些大型资源。

## CPU 检查环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/pipeline.py doctor
pytest -q
```

Linux 还需要 `curl`、`ffmpeg`、`fonts-dejavu-core`。筛帧传输使用支持 HTTP/2 的 curl；Pillow 拼图默认使用 `/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf`。

这些命令能检查契约和合成流程。doctor 会单独报告 torch、transformers、ultralytics、rclpy 是否存在，不因为它们缺失就把 CPU 模式说成完整推理就绪。

## ROS 和 OpenVINS

用已有 ROS 工作区构建 OpenVINS，保留相机、IMU 消息依赖与 `cv_bridge`。ROS 必须对执行预处理的 Python 可见；基于系统 ROS 的环境通常应使用 `python3 -m venv --system-site-packages .venv`，再安装 Python 依赖。运行前 source ROS 及工作区 overlay。

本机历史源码记录：

```text
repository: https://github.com/rpng/open_vins.git
commit: 69488123ed9362dd44b6f28e7f4680abbff1442b
binary: <ros_workspace>/build/ov_msckf/run_subscribe_msckf
```

`patches/openvins_jazzy_compat.patch` 保留本地兼容修改。请在对应源码版本上先执行 `git apply --check`，再构建；未对任意新版本保证适用。外部版本记录见 `provenance/external_dependencies.json`。

配置 `OPENVINS_BINARY` 指向构建完成的二进制。预处理会从每条录制的标定生成 estimator 配置，并读取仓库内的 dynamic_initializer_override.yaml。相机/IMU 标定、时间偏移和 VIO 质量报告必须保留。

## Fast-FoundationStereo

```text
repository: https://github.com/NVlabs/Fast-FoundationStereo.git
commit: a290ba04c1b3ad1ec41a33974a157b2917b624d4
checkpoint id: 20-30-48
checkpoint file: model_best_bp2_serialize.pth
historical SHA-256: 98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692
```

安装对应仓库所需的 CUDA/PyTorch/Triton 依赖，准备权重及同目录 `cfg.yaml`。通过 `STEREO_REPO` 和 `STEREO_CHECKPOINT` 配置绝对路径；如深度模型需要独立环境，在 manifest 的 `preprocessing.depth.python` 指定 Python。版本记录来自已有运行证据，不是“最新版本”推荐。

## Grounding DINO 与 FastSAM

按照自己的 CUDA 环境安装匹配的 torch / torchvision，再安装：

```bash
pip install -r requirements-inference.txt
python components/anchors/scripts/download_model.py --help
```

Grounding DINO Tiny 使用本地目录，至少包含 config.json、model.safetensors、preprocessor_config.json、tokenizer_config.json 和 tokenizer.json 或 vocab.txt；FastSAM-s 使用 `FastSAM-s.pt`。下载适配器内置固定 revision `2e311b914779978895827276d44f59285fc49f95` 并检查仓库提供的哈希；使用 `--output` 指定目录，参数以 `--help` 为准。模型使用和分发遵守各自来源声明，权重保持在 Git 忽略目录。

```bash
export GROUNDING_DINO_DIR=/absolute/models/grounding-dino-tiny
export FASTSAM_WEIGHTS=/absolute/models/FastSAM-s.pt
export EGOMEMORY_GPU_LOCK="$HOME/.cache/egomemory/gpu0.lock"
```

同一张 GPU 上的各工作区应使用相同锁文件，且保留真实显存与外部进程检查。不要为绕过冲突而换锁名。

## 视觉语言模型服务

筛帧端要求 Responses 兼容服务及精确模型身份校验；配置示例保留历史 `gpt-6.1-sol` 名称。Qwen 端要求 OpenAI-compatible `/v1` Chat Completions、图像输入及结构化 JSON 输出能力；模型名按服务实际公布的 ID 填写。

这里提到的服务协议只用于描述现有适配器；仓库不会创建在线账户或购买服务。凭据通过 `api_key_env` 对应的环境变量传入，不写进 JSON。无可用模型服务时仍可运行合成测试；不能用合成输出冒充真实标注。

默认示例把 Qwen 视为已有远端服务。若由调度器管理本地 llama-server，需要在 manifest 增加 `local_server`，满足 `components/semantic/src/local_vlm/config.py` 和 `server.py` 的配置字段；设 `placement: local_gpu` 后与深度阶段共享 GPU 生命周期锁。此配置依赖具体二进制及 GGUF/mmproj 组合，默认模板没有代填一套未验证的路径。
