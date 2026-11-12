# 数据与输出格式

## 原始录制

```text
episode_001/
  videos/{left,right}.mkv
  timestamps/{left,right}_timestamps.txt
  imu/imu0.csv
  calibrations/camera-camchain.yaml
  calibrations/camera_imu-camchain-imucam.yaml
  calibrations/camera_imu-imu.yaml
  meta.json
```

meta.json 至少包含与录制一致的 fps 和 topics.left/right.written_frames；完整原始审计由预处理代码执行。timestamps 是传感器时间依据，不能用帧号 / FPS 替代绝对采集时间。

## 人工事件三帧

左右图分别放在独立目录，文件名遵循：

```text
episode_001_第1次_开始_f1.jpg
episode_001_第1次_进行中_f4.jpg
episode_001_第1次_结束_f7.jpg
```

Markdown 示例：

```text
采集人员：SYNTHETIC
场景：demo
要找的物体：blue cup
帧率 30 帧/秒

| 搜索 | 开始 | 过程 | 结束 |
|---|---:|---:|---:|
| 第1次 | 1 | 4 | 7 |
```

统一入口要求事件从第 1 次连续编号，每次恰有 start/middle/end 三帧，帧号严格递增且不超出原视频范围。目标和 FPS 必须与 manifest/录制一致。原始批次的额外事实与输入例外不能自动套用到新数据；新的不完整输入应明确登记和修复。

## 空间与时间

- VIO：相机位姿、协方差和局部世界坐标，非物体坐标。
- 参考空间树：`scene_memory.spatial_structure.nodes` 中递归 `children`。
- 时间父关系：各节点 `parent_relations`，采用原视频帧号半开区间 `[from,to)`。
- 事件变化：`search_event_states.events[].spatial_changes`。
- 顶层 `history` 可为 null；不应据此认定没有节点级时序关系。
- 无真值、未见或未知信息保持 null/unknown；不会从模型分数推导已测准确率。

## 精简锚点

在每个 `search_event_states.events[].states[]` 中新增：

```json
{
  "object_anchors": [
    {
      "name": "blue cup",
      "ref": "scene:o01",
      "anchor_xy_normalized": [0.31, 0.48],
      "size_wh_normalized": [0.21, 0.34],
      "status": "mask_unreviewed"
    }
  ]
}
```

这是格式示意。锚点基于该张原始事件图的像素坐标归一化；size 基于检测框宽高归一化。未检出时两项均为 null，状态为 not_detected。详细检测分数、掩码与复核原因位于审计目录。

基础主 JSON 仍保留原先 scene、event、研究字段和 null。增强导出会移除新增锚点再与源 JSON 逐值比较，保证原内容没有被重新解释或改写。完整契约见 [精简 JSON](COMPACT_JSON.md) 与 [完整流程](PIPELINE.md)。
