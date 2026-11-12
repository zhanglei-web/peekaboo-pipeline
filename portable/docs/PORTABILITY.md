# 来源、迁移与验证

本仓库将已运行的代码整理为可迁移源码包。它不是用文档重新猜测出来的算法，也没有重新标注历史 320 条。

## 代码来源

- `components/preprocessing` 取自历史处理依赖锁指定的预处理快照，避免把后来实验分支混入既有流程。
- `components/semantic` 保留筛帧、场景、事件、时序的实现及其 Python 依赖、提示词和 schema。
- `components/annotation` 保留 pipeline115 的队列、共享 GPU 锁、逐阶段回执、直接包与装配。
- `components/additive` 保留附加字段及时序关系链接，删除依赖真实样本内容才能构建 schema 的要求。
- `components/anchors` 保留最终续跑的候选词表、检测分割、逐帧几何、精简导出及独立检查。

源文件的原始 SHA-256 与逻辑来源位于 `provenance/source_manifest.json`。它们是迁移前的来源记录。实际仓库文件哈希位于 `components/annotation/dependencies.lock.json`，用于运行时验证和防止静默复用不同代码。

## 便携修改

1. 用 repository.marker 定位仓库根目录，统一入口设置组件导入路径。
2. 用显式 manifest 登记录制和三帧；不扫描作者的个人目录。
3. 用相对依赖锁和环境变量提供模型、二进制与密钥。
4. 研究字段 schema 使用类型模板；导出字段检查使用仅含键结构的模板，不发布真实实验 JSON。
5. 新锚点批次不要求先有 50 条完成记录；模型路径来自该批次 model_config.json。
6. 不复制某批次专用的协方差恢复 runner，异常默认阻止继续。
7. 删除复核说明里写死的特定样例事件结论，保留通用不确定性说明。
8. 新增完整合成装配、候选、mock 检测/分割、精简导出、篡改检查与异路径运行测试。

`legacy/` 路径只留给未作为公开入口的历史脚本参数；它不包含历史数据。公开运行入口是 scripts/pipeline.py。历史脚本中的 standalone main 可能仍需旧运行产物，不能把它们全部当作已迁移 CLI。

## 修改源码后的依赖锁

输入和代码变化应使用新运行目录。确认自己的源码修改后，维护者可执行：

```bash
python scripts/check_repository.py --write-lock
pytest -q
```

刷新锁只表示接受当前源码身份，不代表模型语义正确，也不保证此前运行可继续复用。不要为了绕过失败而直接刷新旧运行的回执。

## 验证边界

验证报告位于 `provenance/validation_report.json`。自动测试中的视觉模型使用显式 mock，合成示例不含真实人或采集图像。2026-11-12 另外使用一条真实录制运行 OpenVINS、深度模型、在线筛帧、本机 Qwen 与 DINO/FastSAM，并验证最终精简 JSON 的字段结构。详见 [单条真实验证](REAL_SAMPLE_VALIDATION.md)。该样本的 VIO 覆盖门限未通过，结构验证不能替代轨迹质量或语义准确率验收。

历史完整交付数量与方法仍可见 [完整 Pipeline](PIPELINE.md)。当前代码包的可迁移测试与历史数据的处理结果是两类证据。
