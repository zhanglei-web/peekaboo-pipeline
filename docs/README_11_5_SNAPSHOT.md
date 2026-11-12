# Peekaboo Pipeline · 11-5

从双目视频、IMU、标定和搜索事件三帧标注生成结构化搜索过程 JSON。11-5 使用直接帧包衔接预处理与模型阶段，无需生成 rosbag 中转。

```text
输入核对 → VIO / 运动筛帧 / 双目校正 → 深度 → 直接帧包校验
→ Astra/Sol 筛帧 → Qwen 场景 → 搜索事件 → 顺序时序更新
→ 组装与证据校验 → 主 JSON / 两种目标 JSON
```

输出包括场景与物体、树形空间结构、事件时序、每次搜索的潜在区域、证据帧与来源，以及实验记录的预留字段。未知信息保留为 unknown 或 null；搜索结果事实必须有明确的数据集作用域，不能把某批数据的规律推广到新数据。

## 部署和使用

首先阅读 [单 Qwen 部署说明](GITHUB_SINGLE_QWEN_DEPLOYMENT.md)，其中包含固定模型下载、SHA256 校验、llama.cpp CUDA 构建、服务启动及单条链路命令。单 Qwen 配置仍需独立的 Astra/Sol 筛帧 API。

**当前发布是既有工作空间的源码快照。代码仍有原机器绝对路径，需按部署文档完成路径、依赖和输入清单适配，尚不是克隆后即可运行的独立安装包。** `vendor/` 保存依赖源码，入口不会自动改用这些路径。发布前完成了源码语法与依赖哈希检查；未在全新机器完成从零部署的端到端验收。

适配环境后，在仓库目录使用：

```bash
cp pipeline115.example.json /absolute/path/to/my_run.json
# 编辑配置中的输入清单、输出目录、筛帧服务、GPU锁与模型服务配置。
bash pipeline115.sh check --config /absolute/path/to/my_run.json
bash pipeline115.sh run --config /absolute/path/to/my_run.json
bash pipeline115.sh status --config /absolute/path/to/my_run.json
bash pipeline115.sh validate --config /absolute/path/to/my_run.json
```

示例默认由一个本地 Qwen 负责场景、事件和时序。双 Qwen 整条领取与并发边界见 [整条调度文档](EPISODE_DISPATCH_11_5.md)。不同 Qwen 可领取不同数据；同一条数据的时序更新必须顺序执行。

## 文档

- [11-5 管线、恢复与验收](PIPELINE_11_5.md)
- [最终 JSON 输出规范](OUTPUT_SPEC.md)
- [字段字典](FIELD_DICTIONARY.md) 和 [JSON Schema](../contracts/final_annotation.schema.json)
- [终端操作说明](TERMINAL_GUIDE.md)
- [流水线设计](PIPELINE_EVOLUTION_PLAN.md)
- [版本变更](CHANGELOG.md)

历史批次文档记录原工作空间的路径、接口和验收情况；新部署以部署说明和实际服务配置为准。历史文档提到的真实样例和数据集事实文件没有随仓库发布，依赖这些私有资料的回归测试需要另行准备获授权的夹具。

## 源码布局与版本

- `pipeline_11_5.py`、`pipeline115/`、`pipeline115.sh`：11-5 入口和调度。
- `contracts/`：输出契约；`dependency_snapshots/`：冻结依赖来源。
- `vendor/qwen/`：Qwen 服务、部署脚本、提示词和模型阶段。
- `vendor/egomemory-core/`、`vendor/covariance_filter/`：预处理及恢复相关源码。
- `vendor/astra-scene-memory/`、`vendor/target_format/`：筛帧与目标格式相关依赖。
- `pipeline.sh` 等根目录旧入口：保留的 11-4 基线。

原始版本来自本地 `11-5` 分支提交 `03364c74365d6727207e8668af122a8327ce13e5`；本公开仓库使用新的提交历史。详见 [来源清单](../SOURCE_MANIFEST.json) 和 [第三方说明](../THIRD_PARTY.md)。模型权重通过固定版本下载；原始数据、真实标注结果、密钥、运行日志和虚拟环境不随源码发布。
