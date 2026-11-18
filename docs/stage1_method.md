# 第一阶段：初始化搜索空间记忆

入口是 `python -m online_annotation initialize --session SESSION --output RUN --candidate-mode brief --model-root MODEL_ROOT`。输入为同一录制的前缀帧和左右目时间索引；具体采样窗口、帧率、短名单和模型端点由 `online_annotation/config.json` 控制并随运行保存。

处理顺序：前缀截断与时间对齐 → 图像质量和外观去冗余 → 本地视觉模型选语义关键帧 → 候选物体分组和区域提议 → 同图完整性复核 → 引用、边界类型及框坐标校验 → `catalog.v0.json`。`--candidate-mode brief` 是当前选择的 B 版输出方式，保留完整图像复核。模型权重、服务及凭据在运行环境中，不随仓库提供。

每个候选区域含稳定的 `candidate_id`、`parent_id`、`region_type`、`identity_anchor` 和文字边界。外部占地固定在原位置；容器内部跟随容器身份。可见框只锚定语义身份，**不是**隐藏区域的真实几何边界。阶段一将 `geometry_status` 与 `cumulative_sufficiency` 留为未知，不把尚未检查写成 0。

运行 `python -m spatial_memory initialize --catalog RUN/catalog.v0.json --output RUN/initial_memory.json` 可将目录转为阶段二可用的记忆初态。候选为 provisional；后续事件的区域关联会确认实际检查单元。物体均值只统计已确认单元，未知项保持未知并附上下界。
