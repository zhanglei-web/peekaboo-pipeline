# 单条真实全流程验证

验证日期：2026-11-12。结论：一条真实样本已从原始录制及已有人工事件三帧生成精简增强 JSON，并通过结构与证据一致性校验。不是仅将历史 JSON 再导出，也没有用 mock 替代本次模型推理。

## 输入与执行

样本目标为橡皮，录制约 15 秒，4 个搜索事件、12 张左目事件图。输入包括双目 MKV、时间戳、IMU CSV、标定、meta.json、人工 Markdown、左右三帧及经过原始照片和 Markdown 哈希核验的位置事实。

通过 `scripts/pipeline.py prepare`、`annotate run/validate`、`anchors-prepare/run` 执行，独立输出目录使用 `runs/real_e2e_20261112/`。原始数据及历史交付保持不变。真实数据、权重、凭据、具体路径和运行日志不包含在 GitHub 源码包中。

| 阶段 | 实际使用 | 结果 |
|---|---|---|
| 输入与 VIO | 原视频内容核验、OpenVINS | 得到 350/448 个相机帧位姿；质量限制见下文 |
| 运动关键帧、校正、深度 | 几何规则、Fast-FoundationStereo | 55 帧深度结果 |
| 语义筛帧 | 配置服务的 gpt-6.1-sol | 粗筛 6 帧，精筛 3 帧 |
| 场景、事件、时序 | 本机 Qwen3.8-27B Q4_K_M，16,384 上下文 | 1 次场景、4 次事件、4 次时序请求完成 |
| 基础装配 | Schema 与时序关系校验 | 生成基础 JSON 和两种导出 |
| 物体锚点 | Grounding DINO Tiny + FastSAM-s | 全部 12 张事件图处理完成 |
| 精简导出 | 保留原字段、添加 object_anchors | 基础原值保持与独立校验通过 |

本次使用词表提取描述候选；没有额外 Qwen 候选提取。远端 Qwen 服务连接超时，因此使用本机真实 Qwen 模型。筛帧服务仍为远端真实请求。11 次成功语义请求不将缓存副本重复计数。

## 结构验证

- 历史 JSON 去除锚点后、新基础 JSON、新精简 JSON 去除锚点后，都通过同一基础 Schema。
- 对照顶层、metadata、README、scene_memory、对象、事件、逐帧状态、锚点、搜索单元、inspections、episode_outcome、derived_statistics、hidden 和 priors 的字段集合，无差异。
- 4 个事件、12 张图的 phase、原视频帧号、图像 SHA-256 与历史样本一致。
- 增强结果删除 object_anchors 后与本次基础 JSON 完全相等。
- 原始录制、阶段回执、图像哈希、候选覆盖、锚点五字段与归一化范围均通过独立校验。

单条最终 JSON SHA-256：`30b3648eada2ee6b10b8664dbb0df7bfa9167a2b6eed85d053cca34bb87e4f7e`。本地完整报告为运行目录中的 `real_e2e_validation.json`、`anchor_independent_validation.json` 与基础运行的 `final_validation.json`。源码包提供不含个人路径的 [验证摘要](../provenance/real_sample_validation.json)。

## 实跑发现并修复的问题

1. 预处理子包初始化仍导入未打包的旧 DAAAM 接口。将子包导出限制到已包含的预处理功能，新增真实入口导入回归测试。
2. 配置路径解析追踪 Python 符号链接，使虚拟环境解释器变成系统 Python。解释器路径现保留虚拟环境入口，新增符号链接回归检查。
3. 深度 worker 的 stderr 已重定向，但错误处理仍读取 `process.stderr`。现在从保留的日志读取错误，新增异常测试。
4. 补充显式 `episodes[].location_facts` 导入，并验证原始 Markdown/照片与事实的对应关系。
5. 模板名 `experiment_export.shape.json` 被误用作实际导出文件名。统一实际文件为 `experiment_records.json`，模板只用于结构检查，新增装配导出测试。

前两个失败尝试保存在独立目录。第 5 项在模型阶段完成后修复；原装配产物、回执和旧代码身份归档至 `migrations/export_filename_fix/`，核验并保留本次七个上游阶段回执，只重新运行装配，没有重新发送模型请求或静默修改历史交付。最终结果由修复后的导出器和校验器验收。

## 质量边界

OpenVINS 位姿覆盖率约 78.1%，覆盖门限失败，进程退出码为 `-11`；其余所报告位姿门限通过。下游只使用具有可插值位姿的帧，未对初始化缺失位姿外推。不能将本次 JSON 结构通过表述为轨迹质量通过。

锚点数组共有 118 个条目：`mask_unreviewed` 19、`mask_needs_review` 58、`bbox_center_needs_review` 34、`not_detected` 7。条目包括同一对象跨帧和多实例，不等于独立物体数。未检出项保留 null 坐标与尺寸；没有人工语义验收，也未测量识别准确率。

本次证明这一输入和模型配置可以生成符合现有字段契约的 JSON，不承诺逐字复现历史描述、其他输入全部成功或仅凭视频自动发现事件三帧。
