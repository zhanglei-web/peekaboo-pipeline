> 发布版说明：`<DATA_ROOT>`、`<OUTPUT_ROOT>`、`<MODEL_ROOT>` 是本地资源占位符；历史运行数字保留。新仓库的可执行入口以 [运行指南](RUNNING.md) 为准，历史命令用于解释来源。

# 在现有 JSON 中增加逐帧物体锚点：单条数据试点

**当前范围已按用户反馈调整：仅选择每个搜索事件描述明确涉及的物体，包括提到的背景物体，在该事件三帧分别定位。请优先阅读 [事件描述范围说明](EVENT_DESCRIPTION_SCOPE.md)，使用 `examples/event_description_config.json`。下文中的 scene_and_focus 配置及 run002 结果是早期全场景基线，保留作实现和历史对照。**

## 目标与交付

对原 JSON 的每个搜索事件按 start / middle / end 三帧分别定位物体。Grounding DINO 根据既有对象的英文短语生成检测框；FastSAM 对每张图运行一次，生成分割候选；将框与掩码匹配后计算物体内部锚点。输出原 JSON 的派生副本，不改写原文件或原字段。

本次试点输入：

`<DATA_ROOT>/11.5/followup138_20261105/repaired25_20261107/11.2/cs_0001_session_20261030_190346.json`

包含 4 个事件、12 帧，原图索引为 112、142、191、207、232、257、299、317、372、396、419、523。现有 scene_memory 有 21 个对象候选；所有帧共 252 次场景候选查询，加 15 次当帧关注对象查询，合计 267 次候选观察。观察数不是独立物体数。

工程：`<WORKSPACE>/src/frame-anchor-enrichment`

默认输出：`<OUTPUT_ROOT>/frame_anchor_pilot_20261109/190346_run002`

最终实际运行结果记录在该输出目录的 `validation.json`、`timing.json` 与 `status.json`；另见本文同目录的 `PILOT_RESULT.md`。有测试通过记录不等于真实模型已经完成试点。

## 下载及运行

使用工作区现有 `.venv`，无需修改现有标注环境。已安装的包版本为 torch 2.8.0+cu128、torchvision 0.23.0+cu128、transformers 4.55.4、ultralytics 8.4.191、numpy 1.26.4。实际运行时还会将依赖版本写入 manifest。

Grounding DINO 下载命令（约 689 MB 权重，另有少量配置）：

```bash
<WORKSPACE>/.venv/bin/python components/anchors/scripts/download_model.py
```

脚本使用 ModelScope 的 `IDEA-Research/grounding-dino-tiny`，固定 revision `2e311b914779978895827276d44f59285fc49f95`，逐文件核对仓库声明的 SHA-256 和大小，生成 `download_receipt.json`。仅完整校验的 `.part` 文件会更名为正式文件。重试会跳过已校验的完整文件；未完成的大文件通过 HTTP Range 续传，传输错误最多尝试 6 次；服务器不支持 Range 时从头下载。最终仍必须通过完整文件哈希校验。下载锁防止并发写入。已有文件哈希不符时拒绝覆盖。

保存到：`<MODEL_ROOT>/frame_anchors/grounding-dino-tiny/`。必须包含 model.safetensors、config.json、preprocessor_config.json、tokenizer_config.json，以及 tokenizer.json 或 vocab.txt。不要只下载单个 .pth 文件：本工程使用 Transformers 格式。

FastSAM 复用：`<MODEL_ROOT>/search_pipeline/FastSAM-s.pt`。不需要再次下载。

在其他网络可通的机器下载时，也可使用 Hugging Face 后将完整目录复制到上述位置。两种来源不要混合下载到同一非空目录，模型变更需新建运行输出目录。

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='IDEA-Research/grounding-dino-tiny', local_dir='grounding-dino-tiny', allow_patterns=['*.json','*.txt','*.safetensors'])"
```

计划检查、运行、独立校验命令：

```bash
cd <HISTORICAL_PATH>
PYTHONPATH=<WORKSPACE>/src/frame-anchor-enrichment .venv/bin/python -m frame_anchors.cli plan --config components/anchors/examples/pilot_config.json
PYTHONPATH=<WORKSPACE>/src/frame-anchor-enrichment .venv/bin/python -m frame_anchors.cli run --config components/anchors/examples/pilot_config.json
PYTHONPATH=<WORKSPACE>/src/frame-anchor-enrichment .venv/bin/python -m frame_anchors.cli validate --output <OUTPUT_ROOT>/frame_anchor_pilot_20261109/190346_run002
```

`examples/pilot_config.json` 全部采用绝对路径。自定义配置中的相对路径按执行命令的工作目录解释。将模型路径、输入 JSON、提示词映射和输出目录显式指定，不依赖自动下载或隐式模型替换。

## 输入适配和范围

`scene_memory.objects[].id` 是场景对象引用。`search_event_states.entity_reference_catalog` 是事件实体引用表。每帧只补充该帧 `focus_object_states` 所列的实体，不能把其他帧的关注实体直接当成本帧可见对象。

默认 `scene_and_focus`：每帧检索全部场景对象候选，以及该帧原有关注对象。可切换 `focus` 来减少查询，但必须相应提供无多余键的提示词文件，并新建输出目录。

中文对象名通过 `examples/pilot_prompts.json` 显式映射为英文名词短语，例如 `scene:o14 → wooden picture frame`。事件对象优先使用原来的英文 label，可通过 `entity:<完整原引用>` 覆盖。映射是检测描述，不改变原标签。该映射已经按本条数据编写；扩展到其他数据需核对，不应机械复制。

这是**既有候选对象的定位补充**，不能宣称发现了图中全部物体。原 JSON 未提及的对象不在本版本范围内。未检出候选仍保留 `not_detected`，不能直接认定对象不在画面。默认不调用 Qwen；在独立复核步骤中可对未检出、歧义候选调用视觉模型，但不应让语言模型直接编造像素坐标。

## 新增字段与坐标

每个 `events[].states[]` 新增 `object_anchors`；顶层增加 `frame_anchor_enrichment`。原始字段的值均保留，原 JSON 文件哈希在运行前后检查。

`object_anchors.objects[]` 的主要内容：

| 字段 | 含义 |
| --- | --- |
| source_candidate_id / source_ref | 原场景或事件引用，二者命名空间独立 |
| bbox_xyxy_pixels | 原始图像上的 x0、y0、x1、y1，半开区间 |
| bbox_xyxy_normalized | 上述坐标分别除以 W、H |
| anchor.pixel_xy | 物体内部锚点；匹配失败时为检测框中心 |
| anchor.normalized_xy | x/W、y/H，左上角为原点 |
| anchor.method | 最大内距点或明确的 bbox_center_fallback |
| mask_index | 当帧 masks.npz 的索引；无可靠匹配时为 null |
| apparent_size | 框宽高像素数及画面占比；匹配掩码的可见面积占比 |
| review_flags / conflicting_anchor_ids | 多实例、候选冲突、分割歧义等复核原因 |
| identity_status / track_id | 仅候选身份、track_id 为 null |

锚点采用掩码**最大连通域中距边界最远的像素中心**，不是可能落在空洞里的质心。像素中心为 `(x+0.5,y+0.5)`；归一化值按原图 W、H 计算。图像边界外补背景后计算距离，避免满图掩码的边界偏差。

没有可靠掩码时返回框中心，并标记 `inside_segmented_region:null`、`geometry_status:needs_review`，不宣称点一定在真实物体上。它适合人工复核，不应当作高质量掩码锚点直接验收。

大小仅代表**图像中的投影大小和可见面积**。不估计厘米、完整三维体积或遮挡后的真实面积，`metric_dimensions:null`。透视缩放、遮挡和镜头变化可能改变比例，即使是同一个真实物体。

## 模型及冲突规则

每帧相同提示词去重；默认每批 6 个短语，防止过长文本超过模型 256 token 限制。逐短语 token 的 sigmoid 分数取均值，默认阈值 0.28；这是本工程的启发式分数，不是校准概率，也不能称为准确率。形容词和长短语会影响分数，需通过人工复核决定是否缩短提示词或调整阈值。

同一候选按框 IoU 0.5 做 NMS，最多保留 5 个实例。不同候选之间不直接 NMS，以免丢掉原有引用冲突证据。FastSAM-s 默认 imgsz=1024、conf=0.25、iou=0.9、max_det=200、retina_masks=True。要求掩码与原图分辨率一致，不静默按错误尺寸缩放。

匹配条件：掩码至少 16 像素；掩码外接框与检测框 IoU ≥ 0.5；掩码位于检测框中的面积比例 ≥ 0.85。排序分数为 `0.7*IoU + 0.3*面积包含比例`，前两名差值 < 0.06 时判定歧义。小物体、细长物体、透明容器可能难以得到可靠分割，必须允许回退和未检出。

| 边界或冲突 | 处理 |
| --- | --- |
| 场景 ID 与事件 ID 指向同一掩码，或框 IoU ≥ 0.8 | 保留双方，标注 competing_source_references；不强行合并 |
| 一个原引用匹配多个物体 | 保留多实例并标注复核；不任意选择唯一身份 |
| 达到实例或掩码上限 | 显式标记达到上限，不能宣称覆盖完整 |
| 多块不连通掩码 | 锚点取最大连通域，面积仍统计整个掩码，并标注 disconnected_mask |
| 目标接近图像边界 | 标记可能截断；不补画外形状 |
| 空检测或分割失败 | 空检测保留 not_detected；空掩码可框中心回退；推理异常则停止并记录 failed |
| 原标签/映射错误 | 不重写原标签；通过复核表记录，需要单独修订映射后新建运行 |
| 跨帧名称相同 | 仅保留原引用，不认定已验证跨帧真实身份 |
| 物体在篮内、被遮挡等关系 | 不从二维框重叠推断物理 inside/on 或完整遮挡状态 |
| 重复 ID、悬空引用、已有新增字段 | 拒绝输入，避免静默覆盖 |
| 原图 SHA 不符、EXIF 旋转不为 1 | 拒绝；必须明确修复输入来源或坐标迁移 |

原 schema 如设 additionalProperties=false，派生 `enriched.json` 不能直接送给只接受旧 schema 的客户端。旧客户端继续读原 JSON；新增信息从 `anchors.sidecar.json` 按 state_id 关联。支持扩展的客户端才读 enriched.json。不要修改原 schema 名称来伪装旧格式兼容。

## 输出、安全写入及复用

- `enriched.json`：保留原字段的增强副本。
- `anchors.sidecar.json`：独立附加信息及来源元数据。
- `frames/000…011/`：原始检测框、压缩掩码、锚点、时间、叠加图、仅关注对象图、校验 receipt。
- `review.html`：本地浏览器可查看的 12 帧复核页面，可切换全部候选/关注对象。
- `human_review_template.json`：逐锚点语义、位置和逐帧漏检复核表；默认均未审核。
- `run_manifest.json`：输入/提示词/图像/模型/代码 SHA、配置、依赖版本；严格绑定缓存。
- `validation.json`：结构和几何复算校验结果；不代表人工语义验收。
- `timing.json`：模型载入、每帧检测及分割、当次运行总耗时。
- `status.json`：running / failed / complete；进程异常退出时可能残留 running，应结合进程核实。

输出目录不能覆盖或嵌套到源数据、图像、配置映射或模型目录。使用独立运行锁、原子 JSON 写入与每帧校验收据。仅相同 manifest 且完整文件哈希匹配的帧可断点复用；模型、代码、参数、输入或映射改变都必须换输出目录。

GPU 使用 `<GPU_LOCK>` 非阻塞锁，检查 compute PID 和至少 6000 MiB 空闲显存，不抢占、不终止既有计算进程。锁只能约束遵守同一协议的进程，其他程序仍可能在检查后启动；显存分配失败会中止，不假装成功。

## 验证与验收

执行测试：

```bash
cd <HISTORICAL_PATH>
PYTHONPATH=<WORKSPACE>/src/frame-anchor-enrichment .venv/bin/python -m pytest -q components/anchors/tests
```

测试覆盖环状/多连通/满图掩码、分割歧义和回退、身份冲突、多实例、坐标裁剪、未检出语义、原值保持、无效输入、掩码归档，以及用模拟模型进行的完整输出/恢复/篡改检测。模拟测试不评价模型识别能力。

`validate` 检查源哈希、原值完整保留、12 帧覆盖、附加文件一致、掩码引用和所有锚点/大小/冲突几何复算。它读取保存的模型证据，不重新进行推理，也不证明框中的物体就是提示词所指对象。

人工验收需分两类统计：当帧关注对象、全部既有候选。每个检出条目标记物体类别/原引用是否正确、锚点是否落在正确可见物体上、框是否可用；每帧再核对应该出现却未检出的目标。重复场景/事件引用要记录重复，不能将它们当成两个独立真实物体增加成功数量。计算有效锚点比例和可见目标召回率需先明确人工分母；12 帧只能反映本条数据，不能宣称全量准确率。

速度以实测载入时间、检测时间、分割时间和每帧总时间报告。首次 GPU 初始化和模型载入单列；不能用模拟测试耗时或仅 FastSAM 时间当完整流水线速度。试点通过后再讨论扩展到批量任务。
