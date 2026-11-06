# 源码来源与第三方说明

本仓库发布自本地 11-5 管线工作空间的源码快照。`SOURCE_MANIFEST.json` 记录原始提交、逐项冻结依赖 SHA256，以及未发布的数据文件。`dependencies.lock.json` 和 `dependency_snapshots/` 保留原运行环境的依赖溯源信息；其中的绝对路径是历史记录，迁移需按部署文档核对。

| 目录 | 来源与用途 | 许可证记录 |
|---|---|---|
| 根目录、`pipeline115/` | 11-5 调度、组装、校验及保留的 11-4 代码 | 原来源未提供聚合项目许可证；本次不新增授权声明 |
| `vendor/egomemory-core/` | 冻结的预处理相关源码与打包元数据 | 保留原 `LICENSE`（BSD 3-Clause） |
| `vendor/covariance_filter/` | 独立协方差恢复策略及冻结预处理源码 | 来源同预处理工程；随附原许可证 |
| `vendor/qwen/` | 本地模型服务工具、场景/事件/时序脚本和筛帧集成 | 原工程未提供独立 LICENSE；不据此推断第三方权重的授权 |
| `vendor/astra-scene-memory/` | 原 Astra 场景记忆工程的冻结代码 | 原来源未提供独立 LICENSE |
| `vendor/target_format/` | 主 JSON 组装、schema、目标格式导出和校验辅助代码 | 原来源未提供独立 LICENSE |

`vendor/` 是当前项目实际依赖的源码子集，不声称是各上游完整发行包。模型权重、llama.cpp、OpenVINS、Fast-FoundationStereo 等外部组件并未作为完整源码或二进制打包；下载和使用时需保留其各自许可证。本仓库未给这些独立组件重新授权。

真实数据集、实验记录及最终标注样例未随本次源码发布。文档中的历史验收记录不代表这些数据已公开，也不代表全新机器部署已经验收。
