# 验收记录

验收日期：2026-11-03（Asia/Shanghai）。测试使用本机 RTX 5090 Laptop GPU；所有图像/视频均由 `scripts/create_test_media.py` 生成，明确标记 SYNTHETIC。没有用户真实工作台图片或视频。

## 验收结论

| 检查 | 结果与证据 |
|---|---|
| 后端 | **通过**：llama.cpp 0.8.0-dev，commit `29c720eaf72fcf8e4e61a56a88b994d92668261d`；CUDA 编译和 --list-devices 成功；`logs/backend-build.log` |
| 权重 | **通过**：指定 Q4_K_M 和 Q8_0 mmproj 完整 SHA256 校验，并成功加载；`models/qwen3.8-27b/manifest.json` |
| GPU 执行 | **通过**：日志显示 offloaded 65/65 layers to GPU，CLIP using CUDA0 backend，Flash Attention enabled；`logs/server-20261103-072944.log` |
| API | **通过**：/health=ok，/v1/models 列出 local-vlm，/props 标明 vision=true、ctx=8192；`logs/api-status.jsonl`、`logs/server-props-first.json` |
| 单图 | **通过合成链路**：真实 base64 图像输入，回答指出红色方块、蓝色圆形、绿色三角形及位置，并读出测试文字；不是纯文本请求 |
| 2 帧 / 4 帧 | **通过合成链路**：顺序文字标签送入，首尾帧/对象/证据引用校验通过；2 帧输出描述红方块从蓝圆左侧变至右侧 |
| 视频 | **通过合成 CFR 链路**：CPU 解码，0/6/13/19 帧，对应 0/0.6/1.3/1.9 秒，manifest、JPEG、推理和 JSON 均成功 |
| JSON | **通过**：服务端 schema 约束 + 本地结构/语义引用校验；6 项正向请求均首试成功、finish_reason=stop、没有修复请求；thinking 内容为空 |
| 错误处理 | **通过**：真实请求 max_tokens=1，2 次均 length（初次 + 唯一修复），退出 1，无成功 result；保留两份原响应及 error/metadata |
| 资源 | **通过**：每 0.5 秒 nvidia-smi 采样 + psutil 内存；每个输出目录有 resources.json 和 metadata.json |
| 可复现 | **通过本机 CLI 复现**：从 /tmp、新 Python 进程、无激活 venv，重新启动并逐条调用单图、多帧、视频和停止命令；`outputs/repro-report.json` |
| 前台中断 | **通过**：健康后 SIGINT、加载阶段 SIGTERM 均退出 130、无 traceback、子进程已结束、PID 文件清除；`outputs/foreground-report.json` |
| 自动检查 | **20 项通过**：`logs/pytest.log`；覆盖排序/重复/缺文件/NaN/坏图/EXIF/透明图/JSON/引用/截断/超时/500/短视频/预算/一次修复上限 |
| 真实工作台标注质量 | **尚未验证**：没有用户真实图片/视频。不能从简单合成图推出实景准确率 |

验收后已停止本项目服务，CLI 复现结束时 nvidia-smi 观测整卡占用 445 MiB（随后桌面占用仍会波动）；用户可以直接运行 README 的启动命令。没有结束其他 GPU 任务。

## 稳定配置与实际分配

采用 `configs/default.yaml`：context 8192、parallel 1、n_gpu_layers 99、batch 512 / ubatch 128、Flash Attention auto（实测 enabled）、视觉投影 GPU、图像最长边 768、最多 4 帧、每图 token 上限 1024、输出上限 1024、超时 600 秒。temperature=0、seed=42，thinking off，reasoning preserve off。没有触发 OOM，没有采用 CPU 视觉卸载、减少层数、缩小上下文或删帧等降级，metadata.degradations=[]。

加载日志记录 CUDA0 主权重 buffer 17402.38 MiB、KV 512 MiB、recurrent state 149.62 MiB、视觉 compute 120.06 MiB。另有 CPU_Mapped 权重 buffer 682.03 MiB 和正常主机输出/计算 buffer。因此这里“65/65 层 GPU”指后端层卸载记录，不宣称整个输入/解码/文件处理流程都没有 CPU 工作。

未加载服务时整卡使用 435 MiB；最终配置服务加载完成、首请求前 19571 MiB。最终配置请求采样观测峰值 **19667 MiB（约 19.21 GiB）**。首轮旧日志级别配置最大 19669 MiB；桌面占用有小幅波动。均为整卡数值，含桌面进程；不是精确瞬时峰值或模型独占显存。

## 最终配置性能样本

原图及视频 960×640，实际发送 JPEG 均为 **768×512**。context 8192，生成上限 1024，按单图 → 单图重复 → 2 帧 → 4 帧 → 视频 → 4 帧重复顺序执行。首请求是服务加载及后端 warmup 后的第一个应用请求，不包括模型加载时间。耗时为 Client.run 全程（含图片处理、预算查询和 API；视频解码抽帧另计）。

| 请求 | 帧数 | 耗时 s | 输入/输出 tokens | 已缓存 tokens | 后端生成 tokens/s | 显存采样峰值 MiB | 主机 used 峰值 GiB | 服务 RSS 峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 单图首请求 | 1 | 3.80 | 488 / 78 | 0 | 26.28 | 19660 | 8.28 | 1.69 |
| 单图重复 | 1 | 3.13 | 488 / 78 | 484 | 26.17 | 19660 | 8.32 | 1.69 |
| 2 帧 | 2 | 28.62 | 1550 / 702 | 0 | 26.31 | 19666 | 8.85 | 2.22 |
| 4 帧 | 4 | 28.42 | 2348 / 682 | 726 | 26.15 | 19667 | 9.46 | 2.82 |
| 视频抽 4 帧 | 4 | 31.20 | 2376 / 753 | 726 | 26.12 | 19667 | 10.12 | 3.45 |
| 4 帧重复 | 4 | 26.59 | 2348 / 682 | 2344 | 26.16 | 19667 | 10.24 | 3.46 |

生成速度直接来自 llama.cpp response.timings.predicted_per_second（有效生成时间定义以该后端为准），没有用总墙钟时间冒充生成速度。重复请求复用了 prompt cache；单图与 4 帧重复请求的 result.json 均与首试一致。主机 used 为 psutil virtual_memory.used，不含可回收文件缓存；最小 available 约 51.99 GiB，不能把显存和主机 RSS 混算。这里只是少量样本，不是压力测试或延迟分位统计。

从 /tmp 的独立 CLI 复现：服务健康启动约 3.16 秒，单图进程约 4.49 秒，4 帧约 28.96 秒，视频约 31.40 秒；这些时间包含 Python 进程开销，不应与上表的 Client.run 时间直接混比。详见 `outputs/repro-report.json`。

## 错误与日志警告的定位

- 初次缺少 CUDA Toolkit、pytest 误收集上游测试、ROS PYTHONPATH 污染，处理见 deployment.md；最终构建和测试成功。
- 首次启动提示 CLI chat-template-kwargs 设置 enable_thinking 已弃用：已改为当前 `--help` 支持的 `--reasoning off --no-reasoning-preserve`，保留请求级 enable_thinking=false，并重跑完整正向测试。弃用提示已消失。
- 多模态 prefill 的 `non-consecutive token position` 警告仍保留。已定位到上游 `src/llama-memory-recurrent.cpp:669` 的 `last_pos != cell.pos + n_seq_tokens` 检查；图像 M-RoPE 在 `tools/mtmd/mtmd.cpp` 的 MTMD_POS_TYPE_MROPE 分支设置 `pos.t = pos_0`，多个图像 token 的时间位置相同，因而不满足逐 token +1 的检查。没有修改或隐藏上游警告；本次单图/2 帧/4 帧/视频与重复请求均成功。该定位及合成样例不构成对任意实景质量的保证。
- 上游提示 grounding 建议至少 1024 image tokens；本项目输入 768×512 的自然图像 token 数约 384，遵循“不放大小图”的初始策略。本次没有像素坐标 grounding 任务，没有把简单图形识别结果当作定位精度验证。
- 补充前台中断测试发现启动阶段 Ctrl+C 会打印 KeyboardInterrupt 栈（子进程已经清理）。已将健康等待与前台运行纳入统一清理路径，并在顶层输出明确的中断状态；随后实测健康后 SIGINT 与加载期 SIGTERM，均退出 130、没有异常栈且无残留子进程。
- 立即重启的补充测试发现端口预检查把 TIME_WAIT 当成端口占用（ss 显示没有 LISTEN 进程）。已在预检查 socket 上设置 SO_REUSEADDR，仍拒绝真正的监听冲突；随即重测前台中断、立即后台启动、健康检查和停止均通过。启动日志名同时增加纳秒后缀，避免同秒重启覆盖记录。
- 真实长度截断负向测试按设计失败，保留原始响应，不记作部署失败或成功标注。错误是 `Incomplete response: finish_reason=length`。

## 样例与原始依据

完整成功输出（原响应、usage、输入、配置、资源采样均保留）：
- 单图首请求：`outputs/20261102T233017-7d2e858b/`
- 单图重复：`outputs/20261102T233021-cc6501b2/`
- 2 帧：`outputs/20261102T233024-184d5d65/`
- 4 帧：`outputs/20261102T233053-d1927637/`
- 视频抽 4 帧：`outputs/20261102T233121-a6110f59/`
- 4 帧重复：`outputs/20261102T233153-2455dba4/`

独立 CLI 视频输出：`outputs/20261102T233351-58d11418/`。实际抽帧 manifest 另存为 `examples/synthetic-video.extracted-manifest.json`（图片在该输出目录）。真实截断失败证据：`outputs/20261102T233422-4edde1f8/`。

便于查看的成功结果已复制到 `examples/synthetic-image.result.json`、`examples/synthetic-2frames.result.json`、`examples/synthetic-4frames.result.json`、`examples/synthetic-video.result.json`，均明确来自合成链路。输出目录名称使用 UTC，metadata 也标明 UTC；本记录的验收日期使用 Asia/Shanghai。

机器可读汇总：`outputs/smoke-report.json`、`outputs/metrics-summary.json`、`outputs/repro-report.json`。初轮报告保留为 `outputs/smoke-report-initial.json`。

## 尚未验证与范围限制

1. **真实工作台/第一视角标注质量尚未验证**：缺少真实数据。没有精度、召回率或置信度百分比结论。简单图形的位置描述正确，不等于复杂物体/遮挡/动作语义已经可靠。
2. **8 帧及更高分辨率、长时间负载、多请求并发没有测试**：本次稳定结论限于最多 4 帧、768×512、8192 context、单并发。
3. **可变帧率视频的严格等时间采样尚未验证**：当前按均匀帧索引抽样，记录解码 PTS；已验证的是 CFR 视频。未承诺全视频跨窗口实体关联。
4. **没有进行物理断网/网络命名空间隔离实验**：启动参数明确使用本地 GGUF、动态库及模板，推理代码仅访问 127.0.0.1，没有启动时下载步骤；安装、下载与构建依赖联网。不能把这些事实写成“已做物理断网实测”。
5. BF16 视觉投影、CPU 视觉卸载和其他降级配置没有测试：当前 Q8_0 视觉文件与初始 GPU 配置已成功，不另下载第二套大权重。
