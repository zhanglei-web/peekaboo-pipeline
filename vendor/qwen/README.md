# 本地 Qwen3.8-27B 视觉语言模块

历史静态树形 pipeline 已冻结为 [`11-3`](pipeline_versions/11-3/README.md)，对应原始 81 条 JSON。
代码、提示词、schema、运行配置和来源哈希由独立的 [pipeline 版本仓库](pipeline_versions/README.md) 管理。
当前开发入口继续使用下述时序树格式。

## 默认标注输出（2026-11-04 起）

后续完整 episode 标注默认采用 **`nodes → children` 嵌套树＋原视频帧时序关系**。
稳定物体 ID 可在不同历史分支重复引用，每次出现都有左闭右开的有效帧区间；
父节点未知的时段放入 `unlocated_nodes`。新事件与关系标注使用英文。

新数据入口：

```bash
python3 scripts/run_scene_event_batch.py --todo /path/to/dataset --output /path/to/new_output
```

该默认入口使用 `run_scene_temporal_batch.py`，每个事件一次模型请求同时生成描述和关系变化，
由 CPU 维护区间和父侧子树。已有静态结果目录不会被静默覆盖。
`--legacy-static` 仅用于明确请求的历史复现。底层单图/片段 API 的原始 JSON 是中间证据，
不是完整 episode 的交付格式。详细说明见 `docs/temporal-parent-relations.md`。

已有 JSON 的视觉补标入口：

```bash
python3 scripts/batch_temporal_trees.py --source /path/to/source_json --output /path/to/tree_json --evidence /path/to/evidence
python3 scripts/audit_temporal_tree_batch.py --source /path/to/source_json --output /path/to/tree_json --evidence /path/to/evidence
```

缺少时序标注时必须读取原事件图片，不能靠静态树编造历史。旧元数据、物体属性、搜索结果和研究标签保留。

使用指定的 Qwen3.8-27B Q4_K_M 和 Q8_0 视觉投影，通过项目内 llama.cpp CUDA 后端提供本地 OpenAI 兼容接口。默认只监听 `127.0.0.1:8080`。模型、编译工具链和 Python 依赖准备好后，启动及推理不需要联网。

环境、部署及验收依据分别见 [环境记录](docs/environment.md)、[部署记录](docs/deployment.md)、[验证记录](docs/validation.md)。合成测试不能证明真实工作台标注质量。

## Astra-pipeline 严格格式标注

已增加独立接入版本，所有 VLM 角色调用本地模型，最终格式严格沿用
`/home/baai/zl/demo/search_episode_5.json` 的冻结契约。5090 Laptop 24 GB / 64 GB
优化配置、真实图像容量测试、运行与进度命令见 [本地 Astra 接入记录](docs/astra-pipeline.md)。
该配置使用 `configs/astra-pipeline.yaml`，与下方通用示例的默认 8192 上下文配置分开。

## 本机启动与使用

从任意目录执行以下命令；相对输入路径按项目根目录解析，manifest 内的图片路径按 manifest 所在目录解析。使用项目虚拟环境的 Python，`-I` 隔离本机 shell 的 ROS/PYTHONPATH。

```bash
# 启动后台服务；等待健康检查通过才返回
/home/baai/zl/Qwen3.8-27B/scripts/start_server.sh

# 单图（可换成自己的图片绝对路径）
/home/baai/zl/Qwen3.8-27B/.venv/bin/python -I /home/baai/zl/Qwen3.8-27B/scripts/run_image.py --image data/synthetic/frame-0.png --prompt '请描述可见物体及其相对位置。'

# 有序多帧及结构化 JSON
/home/baai/zl/Qwen3.8-27B/.venv/bin/python -I /home/baai/zl/Qwen3.8-27B/scripts/run_frames.py --manifest data/synthetic/frames-4.json --task event_annotation

# 本地视频：CPU 抽帧后多图理解
/home/baai/zl/Qwen3.8-27B/.venv/bin/python -I /home/baai/zl/Qwen3.8-27B/scripts/run_video.py --video data/synthetic/synthetic-demo.mp4 --start 0 --end 2 --num-frames 4 --task event_annotation

# 只停止本项目创建、PID/启动时间/可执行文件/命令行核验通过的服务
/home/baai/zl/Qwen3.8-27B/scripts/stop_server.sh
```

前台运行：`scripts/start_server.sh --foreground`，Ctrl+C 停止；日志仍保存在 `logs/server-日期时间.log`。健康检查和模型列表：

```bash
/home/baai/zl/Qwen3.8-27B/.venv/bin/python -I -m local_vlm.server status
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8080/v1/models
```

所有服务参数集中在 `configs/default.yaml`。默认 context 8192、parallel 1、GPU layers 99、batch 512 / micro-batch 128、Flash Attention auto、输出最多 1024 tokens、最长边 768、最多 4 帧、超时 600 秒。显式设定每图最多 1024 视觉 tokens，用于保守上下文预算。关闭 backend 自动 fit 和 context shift，避免静默降低卸载层数或截断证据。若修改 server 配置，先停止并重新启动。CLI 可用 `--max-tokens 1536` 调整单次输出预算；仍会检查总上下文预算。

## 新建环境与复现

本项目已经下载和构建的目录可直接复用。需要重建时：

```bash
cd /home/baai/zl/Qwen3.8-27B
python3 -m venv .venv
.venv/bin/python -I -m pip install -r requirements.lock.txt
.venv/bin/python -I -m pip install --no-deps -e .
bash scripts/check_env.sh
python3 -I scripts/install_cuda_local.py
.venv/bin/python -I scripts/download_model.py
bash scripts/build_backend.sh
.venv/bin/python -I scripts/create_test_media.py
.venv/bin/python -I -m pytest -q
bash scripts/start_server.sh
.venv/bin/python -I scripts/smoke_test.py
```

最少系统依赖：Python 3.11+、venv、Git、CMake、GCC/G++、make、curl、xz、OpenSSL 开发库、可用 NVIDIA 驱动。若另一台 Ubuntu 24.04 缺依赖，管理员可执行 `sudo apt-get install build-essential cmake git python3-venv curl xz-utils libssl-dev`；本次没有执行系统升级或修改驱动。不需要 PyTorch。

CUDA 安装脚本仅取 NVIDIA 官方 nvcc/cudart/CCCL/cuBLAS 组件到 `.toolchains/cuda-12.8/`，校验 SHA256，支持续传。后端固定在 `configs/backend.commit`，重复构建不会拉取新版本。Python 依赖版本固定在 `requirements.lock.txt`。

权重下载使用 Hugging Face 官方 `huggingface_hub`，仅下载 `configs/model.lock.json` 所列两份文件。首次查询并固定 revision，重复运行复用本地完整文件并重新校验 SHA256；失败时保留 `.cache/huggingface/` 中续传内容。不要在下载未完成时删除缓存。离线手动入口：把指定两份 GGUF 放到 `models/qwen3.8-27b/`，运行同一下载脚本即可只进行本地验证（lock 已随项目保存，完整文件存在时不发起网络请求）。

## 输入与输出约定

- 单图：读取 EXIF 方向，透明背景合成为白色，转换 JPEG，不修改原图；最长边缩至 768，不放大小图。`metadata.json` 记录原始、方向校正后、实际送入的尺寸。
- 多帧：见 `examples/frames_manifest.example.json`。严格按 JSON 数组顺序使用图片，每张图片前插入 frame_id 和 timestamp_s，重复 ID、缺文件、非递增时间戳和超过最大帧数均报错。
- 视频：CPU/OpenCV 解码，区间 `[start,end)` 按帧索引均匀采样，记录解码器返回的帧索引与 PTS；不支持 PTS 时明确记录 index/fps 回退。短视频重复索引只抽一次，manifest 明确记录请求帧数、实际帧数及去重数。推荐恒定帧率输入；可变帧率视频的严格等时间采样尚未验证。原视频不修改。
- 这是**视频抽帧后多图理解**，不是连续视频事件检测。稀疏采样会漏短动作。未实现全视频跨窗口关联、空间建图、分割或全局追踪。

每次调用在 `outputs/时间-随机ID/` 新建目录：成功时有 `result.json`、`response.json`、`metadata.json`、`resources.json`、`input_manifest.json`、处理后 JPEG；视频另有抽出的原尺寸帧及 `frames_manifest.json`。错误时保存 `error.json` 与 metadata，已有原响应保留，不写成功 result，命令退出码非零。

`event_annotation` 同时使用服务端 JSON Schema 约束和 Python 本地校验，包括 clip_id、对象引用、证据帧及首尾状态对应关系。无关系或事件时允许空数组。对象 ID 仅在片段内有效；没有度量三维坐标或校准准确率。schema 合法不等于视觉语义正确，仍需人工核查。

服务使用当前后端的 `--reasoning off --no-reasoning-preserve`，请求传入 `chat_template_kwargs={"enable_thinking":false}`，backend 使用 `reasoning-format=deepseek` 分离 reasoning。仅解析最终 content，拒绝混入 think 标签的内容及任何非 stop 的 finish_reason。解析/校验失败最多修复一次，两次响应分别保留；长度截断不会被记为成功。

客户端先调用本地 `/apply-template`、`/tokenize`，按真实模板文本、每图视觉 token 上界、输出预算及安全余量保守检查上下文。超限直接报错，不静默删帧、截断提示词或 schema。metadata 中记录该估算，实际 token 数见 response.usage。

显存记录为每 0.5 秒调用 `nvidia-smi` 得到的**采样观测峰值**，包含桌面进程，不是精确瞬时峰值。另记录主机内存和服务进程 RSS。温机/重复请求可能复用 prompt cache，性能比较应看原始 timings。API 密钥若需要通过 `LOCAL_VLM_API_KEY` 环境变量传入客户端，不写入日志；默认本地服务不需要 key。

## 验证与排错

```bash
cd /home/baai/zl/Qwen3.8-27B
.venv/bin/python -I -m pytest -q
.venv/bin/python -I scripts/smoke_test.py
```

smoke_test 会真实调用已启动的 CUDA 服务，按单图首轮、单图重复、2 帧、4 帧、视频、4 帧重复执行。报告在 `outputs/smoke-report.json`。原始日志在 `logs/`，验收结论在 `docs/validation.md`。

若启动失败，查看打印的具体 server 日志；端口已占用或已有项目进程时拒绝覆盖。OOM 时先检查显存及日志，再一次只改一个因素：图片尺寸/帧数、batch/ubatch、context，最后再考虑视觉 CPU 或减少 GPU 层数。任何降级应写入配置的 `client.degradations` 并重测。代码不会自动减少输入或换模型。输出超限可精简任务或显式增加 `--max-tokens`，总预算仍必须通过。

权重、虚拟环境、toolchain、后端及编译目录、个人输入、日志和推理输出均在 `.gitignore` 内。原始任务书保留未修改。
