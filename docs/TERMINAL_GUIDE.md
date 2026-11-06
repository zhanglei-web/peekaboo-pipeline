# 11-4 管线终端使用文档

## 1. 版本与范围

代码仓库：`/home/baai/zl/todo/11.4/processing`；Git标签：`11-4`。这是一份**当前机器、当前29条清单的可运行版本**，不是任意目录/任意条数的通用CLI。本版本保留现有绝对路径以保证断点缓存、输入与审计一致。需要处理新批次时，复制为独立任务、生成新的清单和输出目录，并重新确认该批人工结果事实；不要把本版29条确认自动套到未来数据。

受版本管理的内容：管线脚本、输出契约、真实例子、人工事实清单、服务器配置、预处理参数模板、外部代码快照/哈希及文档。Git不包含视频、图像、rosbag、深度输出、运行日志、模型权重、密钥；运行状态仍在原目录。

## 2. 一次性环境条件

当前机器已有以下依赖，终端命令不会自动安装或更新它们：

- Linux、Bash、`/usr/bin/python3`；jsonschema、requests、psutil、Pillow、OpenCV、NumPy等。
- ROS 2 Jazzy：`/opt/ros/jazzy/setup.bash`。包装脚本自动source；ROS环境脚本不兼容未初始化变量的nounset，因此不使用`set -u`。
- 预处理Python：`/home/baai/zl/egomemory_ws/.venv/bin/python`。
- OpenVINS编译产物；Fast-FoundationStereo仓库与权重；具体路径见 `contracts/preprocessing.template.json`。
- 本地Qwen工程：`/home/baai/zl/Qwen3.8-27B`；llama-server、Q4_K_M权重和视觉投影权重。
- Astra调用所需 `ZYAPI_API_KEY` 环境变量以及可用网络。密钥只从环境读取，不写进代码、配置、日志或Git。
- 足够磁盘空间。深度阶段检查GPU空闲显存至少10000MiB；标注模型使用约20GB级显存，需避免其他GPU作业争用。

终端交互式设置密钥（输入不回显，不把密钥字面值写进shell历史）：

```bash
read -rsp 'Astra API key: ' ZYAPI_API_KEY
export ZYAPI_API_KEY
```

如终端已正确继承该变量，无须重复设置。检查命令只显示变量是否存在，不显示内容。

## 3. 常用命令

```bash
cd /home/baai/zl/todo/11.4/processing
bash pipeline.sh status
bash pipeline.sh check
bash pipeline.sh run
```

`run` 创建本机后台进程，输出PID及日志路径，终端断开不影响本次执行。重复调用不会重复创建终端工作进程。若旧调度器还在运行，包装工作进程等待它完成，不中断当前请求。

```bash
# 与run相同的断点继续入口
bash pipeline.sh resume

# 校验当前已发布的基础JSON及存在的人工确认版本，不调用模型
bash pipeline.sh validate

# 仅合并人工确认结果并导出最终版本，不调用模型、不使用GPU
bash pipeline.sh export-confirmed

# 重建机器可读状态：只在没有标注/调度写进程时允许
bash pipeline.sh validate --reconcile
```

`validate` 不完整返回退出码2；错误返回非零。`check` 核对150个相关代码/契约/提示词快照及哈希、5个外部大资产的存在/大小/mtime、原输入和保护文件。大资产没有全文件哈希验证，不能把大小/mtime检查说成密码学内容验证。`check` 报依赖变化时应定位变化并建立新版本，不能绕过检查继续假称11-4。

## 4. 查看日志

```bash
cd /home/baai/zl/todo/11.4/processing
bash pipeline.sh status

tail -f terminal.log
tail -f batch.log
tail -f orchestrator.log
tail -f annotation.log
```

可分别在不同终端查看；Ctrl+C只结束tail查看，不会停止管线。单条失败诊断在：

```text
../search_episode_outputs/episodes/<episode_id>/failure_traceback.txt
../search_episode_outputs/episodes/<episode_id>/selection.log
../search_episode_outputs/episodes/<episode_id>/selection/astra_*/transport_attempts/
../search_episode_outputs/episodes/<episode_id>/temporal/event_*/attempt_*/failure.json
```

`status` 区分进程身份核验、基础标注完成数和人工确认最终完成数。旧错误文字或上次finished_at可能在恢复初期尚未清理，判断当前运行还需结合PID/create_time/command和新日志；`validate --reconcile` 只用真实通过的完成回执重建状态。

## 5. 各阶段做什么

1. **输入清单**：14+15条原始数据与三帧MD按唯一episode及所属目录匹配；核对目标、帧率、表格帧号、左右目三帧、原文件身份。已有29条清单无需重建。
2. **内容审计**：左目三帧与该条原视频比较，原阈值相关系数≥0.99、缩略图平均绝对误差≤4；搜索标签附近±90解码帧。保留帧号偏差，不改用户标签。右目检查文件哈希与可解码性；本版本没有把右目全部做同样的像素匹配。
3. **预处理**：audit → OpenVINS → select → rectify → depth → ROS输出。CPU两路，深度GPU一路；独立ROS_DOMAIN_ID。深度与本地Qwen不同时占用GPU。
4. **rosbag验证**：29条全部完成9个topic逐条反序列化读回、数量/时间戳/格式核验，才进入标注。VIO几何质量门槛另记，不把bag导出完成当成全部几何质量通过。
5. **图像与筛帧**：从已验证bag提取RGB/TF；Astra粗筛、精筛。TF仅为未经独立核验的相机视角先验，不能当物体三维真值。
6. **Qwen标注**：场景一次；每事件一次描述（同时生成潜在区域）；每事件一次时序关系。单个事件后的时序图成为下一事件的上下文，不能随意并行这些依赖步骤。
7. **流水线**：一个准备线程预取下一条的提取/筛帧，与当前条单槽本地推理重叠。没有多开Qwen并发槽。此优化减少整批等待，不保证每条端到端延迟减半。
8. **基础组装**：保留旧式物体/事件结构，增加参考树的parent_relations、事件spatial_changes、实验接口和潜在区域字段；严格检验、发布基础JSON及目标导出。
9. **人工结果投影**：依据明确列出的当前29条用户确认，生成confirmed_json和两份confirmed_exports；原基础JSON保留。
10. **最终校验与GPU释放**：核验结果与哈希，只有本次创建、身份匹配且请求空闲的服务才能停止。原暂停demo及其他任务保持原状态。

## 6. 参数与模型预算

| 项目 | 11-4配置 |
|---|---|
| 图像几何 | 960×600，水平FOV 100° |
| 运动选帧 | soft/hard平移60/150mm；旋转5/12°；max_gap 0.25s |
| 深度 | 8 iterations；max_disp416；triton；left-right置信度；warmup2 |
| 本地模型 | Qwen3.8-27B-Q4_K_M + Q8_0 mmproj |
| llama-server | ctx_size16384、parallel1、image_max_tokens1024、GPU layers99 |
| 采样 | temperature0、seed42、cache_prompt=false |
| 场景请求 | max_tokens10000 |
| 事件请求 | max_tokens6144，包含潜在搜索区域 |
| 时序请求 | max_tokens3072；一致性失败最多3次有证据保留的尝试 |
| Astra | coarse/refine；max_output_tokens5000；timeout300s，保留传输尝试 |

`local_qwen.server.json` 中client.max_tokens1024是通用客户端默认值；本管线的场景/事件/时序预算由上表的专用调用覆盖。不能据此误认为本管线只生成1024 token。

固定代码与seed仍不能保证外部服务和GPU运行逐字节重现。版本管理保证可追溯配置/代码/证据，不把模型语义正确性或绝对确定性作为承诺。

## 7. 断点恢复与失败处理

- 已完成的同输入场景、事件、时序结果可复用，但提示词、schema、图像哈希必须一致。修改提示词/schema后不要直接复用旧缓存，也不要覆盖证据。
- 预处理明确失败时保留状态/日志；CLI拒绝把失败项直接当queued再跑。诊断协方差问题后仅允许冻结恢复策略，阈值不变、排除坏样本、禁止跨坏样本插值，另写恢复输出。
- 不允许只凭PID消失判断任务完成。完成以真实回执和内容校验为准。
- Astra超时/HTTP失败保留transport_attempts，原请求没有通过模型身份/完成性验证不得作为成功。不要为继续运行改成其他模型或取消校验。
- final assembly错误可以用已有成功模型输出在CPU修复；不必重新进行推理。不能为了通过而删除“原字段保持”检查。
- `validate --reconcile` 解决已存在合法完成回执但状态未更新的问题，不能把不存在或未通过校验的输出补成complete。
- `export-confirmed` 幂等：内容一致可重复运行；如现有最终文件与应生成内容不同则拒绝覆盖，需先保留并审查人工改动。

## 8. 文件和状态入口

| 文件 | 内容 |
|---|---|
| inventory.json / configs/ | 原始数据、预处理参数和输出位置 |
| annotation_inventory.json | 29条、196事件及三帧哈希映射 |
| source_manifests/ | 原始文件大小/mtime、meta/MD哈希 |
| input_audit.json | 内容审计与帧号偏差 |
| state.json | 预处理各条状态及bag校验 |
| orchestrator_state.json | 两阶段调度及模型服务释放 |
| ../search_episode_outputs/batch_status.json | 基础标注状态 |
| dataset_facts.json | 当前批人工结果事实及29条作用域 |
| ../search_episode_outputs/confirmed_status.json | 最终人工确认输出状态 |
| terminal_validation.json / terminal_run_result.json | CLI验证与整次执行结果 |
| dependencies.lock.json / dependency_snapshots/ | 外部源码快照、哈希与大资产信息 |

## 9. Git版本操作

```bash
cd /home/baai/zl/todo/11.4/processing
git status --short
git log --oneline --decorate -5
git show --stat 11-4

# 导出该版本代码及文档到独立文件，不含输入数据/权重/运行状态
git archive --format=tar --output=/tmp/pipeline-11-4.tar 11-4
```

不要在有运行进程时切换代码版本。需要修改时，先建立新分支/版本，保留11-4标签；修改后的契约、提示词和依赖清单一起更新并回归验证。当前仓库仅保存在本机，未向远端推送。

源码快照用于审计和恢复，运行时仍导入本机原绝对路径；CLI在启动前检查哈希。一份git archive不能单独在另一台机器执行，迁移还需安装ROS、模型、编译产物、Python依赖、输入清单并适配路径。

## 10. 本地无模型测试

```bash
cd /home/baai/zl/todo/11.4/processing
python3 test_annotation_contract.py
python3 test_release.py
bash pipeline.sh check
bash pipeline.sh validate
```

测试检查新增字段必须存在、证据帧不能跨事件、实际组装不能改原字段、完成回调正常、人工结果按清单一致、没有编造位置/成功帧/重访。测试只在临时目录使用旧样本作为回归fixture，不向本批发布合成标注，不启动Qwen。
