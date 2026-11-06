# 本机环境记录

检查日期：2026-11-03（Asia/Shanghai）。初始检查保留在 `logs/environment-initial.log`，含完整设备及 GPU 占用进程列表；仅报告，没有结束其他任务。

| 项目 | 实测 |
|---|---|
| 系统 | Ubuntu 24.04.5 LTS，Linux 7.0.0-38-generic，x86_64 |
| CPU | Intel Core Ultra 9 275HX，24 逻辑 CPU |
| 内存 / swap | 62 GiB / 2 GiB；初始可用内存约 54 GiB |
| 项目磁盘 | 初始可用约 1016 GiB |
| GPU 完整名称 | NVIDIA GeForce RTX 5090 Laptop GPU |
| 显存 | 24463 MiB；首次查询可用 23470 MiB（桌面占用会变化） |
| 驱动 | 595.99.02 |
| 驱动 CUDA 能力 | nvidia-smi 显示 13.2；不表示已安装 Toolkit |
| 实际 Toolkit | 最初没有 nvcc；项目内安装 NVIDIA CUDA 12.8.1 组件，nvcc 12.8.93 |
| GPU 架构 | nvidia-smi 实测 compute capability 12.0；nvcc 列表确认支持 sm_120；llama.cpp CMake 按自身规则转换为 120a |
| Python / Git / CMake | 3.12.3 / 2.43.0 / 3.28.3 |
| C / C++ | GCC / G++ 13.3.0 |
| 网络 | Hugging Face API、文件下载及 GitHub clone 实际成功；NVIDIA redistrib 下载成功 |

GPU 原有占用包括 Xorg、gnome-shell、桌面 portal、浏览器和 Nautilus。没有终止这些进程。

没有安装 PyTorch，没有替换或升级 NVIDIA 驱动，没有使用 sudo。CUDA Toolkit 仅在 `.toolchains/cuda-12.8/`；取 NVIDIA 官方 redistrib 的 nvcc、cudart、CCCL、cuBLAS 四个组件，逐一校验 SHA256。来源及版本保存在 `docs/cuda-source.json`，下载日志在 `logs/cuda-install.log`。脚本 `scripts/install_cuda_local.py` 可重现。

本机 shell 的 PYTHONPATH 会带入系统 ROS 包。为隔离它，启动脚本和推荐命令使用 `.venv/bin/python -I`；依赖锁只采集本项目 venv 的 site-packages。没有把 ROS 或 PyTorch 添加为项目依赖。

原始模型 config 的 architectures 为 Qwen3_5ForConditionalGeneration，GGUF 的 general.architecture 为 qwen35，视觉投影类型为 qwen3vl_merger。模型名称没有更换。来源 JSON、模型卡、GGUF 元数据及内嵌模板已保存到 docs。
