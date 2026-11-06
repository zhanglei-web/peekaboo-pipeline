# 部署记录

日期：2026-11-03（Asia/Shanghai）。原始任务书保持不变。

## 固定来源

- 原始模型：`Qwen/Qwen3.8-27B`，revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`。
- GGUF 仓库：`ggml-org/Qwen3.8-27B-GGUF`，revision `a316ea527fde32c12487be7574e2db5ac2ad2a7e`。
- llama.cpp 官方仓库 `ggml-org/llama.cpp`，commit `29c720eaf72fcf8e4e61a56a88b994d92668261d`，提交时间 2026-11-02T20:52:39+02:00。
- 主模型 GGUF 架构 `qwen35`，配套视觉投影 `qwen3vl_merger`；官方原始 config 为 `qwen3_5`。这属于目标模型本身的架构名称，没有替换为另一个模型。

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `Qwen3.8-27B-Q4_K_M.gguf` | 18973870528 | `c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747` |
| `mmproj-Qwen3.8-27B-Q8_0.gguf` | 629247008 | `2e968a6af97ce35d8971890b257b9b7edabf20ad91450501fa53162a19ee33eb` |

官方客户端下载及完整校验成功的输出在 `logs/model-download.log`。实际文件清单在 `models/qwen3.8-27b/manifest.json`。没有下载 BF16 主权重、MTP、DFLASH 或其他大模型。下载文件大小不等于显存。

## 工具链与构建

本机最初缺少 nvcc；没有系统安装权限需求，因为通过 NVIDIA 官方 redistrib 在项目中部署编译器、运行时、cuBLAS 和 CCCL。没有安装 PyTorch，没有变更驱动。版本及 SHA256 见 `docs/cuda-source.json`。

```bash
python3 -I scripts/install_cuda_local.py
bash scripts/build_backend.sh
```

构建脚本查询本机 `nvidia-smi --query-gpu=compute_cap` 得到 12.0，并验证 `nvcc --list-gpu-code` 有 sm_120。CMake 按上游 Blackwell 规则将 120 转为 120a，相关代码位于 `ggml/src/ggml-cuda/CMakeLists.txt`。具体参数为 Release、GGML_CUDA=ON、CMAKE_CUDA_ARCHITECTURES=120、LLAMA_BUILD_TESTS=OFF、LLAMA_BUILD_EXAMPLES=OFF，8 并发，仅构建 llama-server 目标。CPU backend 与 CUDA backend 一起编译是上游正常行为，不表示 CPU-only。

构建日志 `logs/backend-build.log`，帮助 `logs/backend-help.txt`。工具链使用 GCC/G++ 13.3、CMake 3.28.3、nvcc 12.8.93。NCCL 缺失仅影响多 GPU，本机为单 GPU；ccache 缺失只影响重复构建速度。未为这些提示安装额外组件。

构建时上游还会取得其固定 b1 web UI 资源；该资源在构建中嵌入，推理不需要在线下载。源代码和两份 GGUF 都已本地保存。

## 配置与接口

配置只在 `configs/default.yaml`；启动参数由 `local_vlm.server` 生成并逐项对照该二进制 `--help`。服务进程记录保存 PID、create_time、完整命令行和可执行文件路径，停止前逐项核验。默认端口 8080，模型别名 local-vlm，context 8192，parallel 1，GPU layers 99，batch 512、ubatch 128，Flash Attention auto，视觉投影在 GPU，image max tokens 1024。关闭自动 fit、context shift。

请求使用 `/v1/chat/completions` + base64 image_url。结构化请求格式为：

```json
{"response_format":{"type":"json_schema","json_schema":{"name":"event_annotation","strict":true,"schema":{}}},"chat_template_kwargs":{"enable_thinking":false},"reasoning_format":"deepseek"}
```

上面的空 schema 仅展示包装格式；实际请求传完整 `schemas/event_annotation.schema.json`。该包装格式已对照固定 commit 的 `tools/server/server-common.cpp`。只解析 content，reasoning_content 单独保存在原响应/metadata。模板备份为 `docs/embedded-chat-template.jinja`。

## 实现中发现并修复的问题

1. 缺 CUDA Toolkit：项目内安装必要 NVIDIA redistrib 组件，驱动保持不变。
2. 默认 pytest 会误收集 third_party/llama.cpp 的 Appium、Torch 等上游测试：设置 testpaths=tests，不安装无关依赖。
3. shell PYTHONPATH 含 ROS：推荐命令和服务脚本使用 Python -I；依赖锁仅来自项目 venv。

4. 前台启动阶段中断打印异常栈：统一启动/运行阶段清理，SIGINT/SIGTERM 明确退出 130；真实测试确认子进程和 PID 文件都已清理。
5. 后端 CLI thinking 参数弃用提示：改用当前 --help 支持的 --reasoning off --no-reasoning-preserve；请求级 enable_thinking=false 保持有效。

6. 立即重启时端口预检查遇到 TIME_WAIT：预检查 socket 增加 SO_REUSEADDR，保留真实监听冲突检查；已实际立即重启验证。

后续 GPU 加载与接口实测结果、运行时错误及处理记录见 `docs/validation.md`。

## 官方依据（实施时下载和核对）

- https://huggingface.co/Qwen/Qwen3.8-27B
- https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF
- https://github.com/ggml-org/llama.cpp
- https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.8.1.json

原始 API 快照：`docs/model-source.json`、`docs/gguf-source.json`；原始模型 config/card：`docs/original-model-config.json`、`docs/original-model-card.md`。后端文档位于固定源码的 `docs/build.md`、`docs/multimodal.md`、`tools/server/README.md`。
