# 11-4 最终 JSON 输出规范

版本：`11-4`。整理日期：2026-11-05。适用数据：`/home/baai/zl/todo/11.4` 清单中的29条、196个搜索事件。基础设计沿用已确认的增量样板，保留旧81条中的字段类别、物体档案、事件三帧、空间参考树与时序关联；旧81条文件本身不在本次修改范围。

机器契约：[final_annotation.schema.json](../contracts/final_annotation.schema.json)。完整字段类型表：[FIELD_DICTIONARY.md](FIELD_DICTIONARY.md)。真实完整样例：[final_annotation.example.json](../contracts/final_annotation.example.json)。样例是本批某一条的真实导出，不是所有场景的固定内容。

## 1. 最终交付文件

| 文件 | 作用 |
|---|---|
| `search_episode_outputs/confirmed_json/<episode_id>.json` | 最终主JSON：视觉、空间、时序、潜在搜索区域，以及用户确认的事件结果 |
| `search_episode_outputs/confirmed_exports/<episode_id>/<episode_id>.json` | 对应 end_version 搜索过程标注格式：`peekaboo.episode.annotation.v1` |
| `search_episode_outputs/confirmed_exports/<episode_id>/experiment_records.json` | 对应 end_version 实验补充记录格式；未提供的实测量保持null |
| `search_episode_outputs/confirmed_receipts/<episode_id>.json` | 原输出、人工事实文件、契约和三份交付文件的SHA-256与校验回执 |
| `search_episode_outputs/confirmed_status.json` | 全批最终交付计数；只有29条全部完成才为complete |
| `search_episode_outputs/json/<episode_id>.json` | 人工结果投影前的基础标注，保留原模型及未知结果，作为审计输入 |
| `search_episode_outputs/episodes/<episode_id>/` | 图像、请求、响应、筛帧、时序图、错误诊断及完成回执 |

`confirmed_json` 是本版本纳入用户结果确认后的交付目录。不得把仍为unknown的基础目录与最终目录混用。最终两个目标JSON由主JSON确定性导出，不再次调用模型。实验字段存在不代表实验值已测得。

## 2. 用户确认的本批结果事实

用户于2026-11-05明确说明：**当前所有数据中，每条数据除最后一次搜索外，前面的每次搜索都没有找到目标；最后一次搜索找到目标。**

这是数据拥有者确认的当前数据事实，不是“看到事件排在最后就猜它成功”的模型规则，更不是对所有未来采集数据的通用约束。实施时以 [dataset_facts.json](../dataset_facts.json) 中列出的29个episode ID限定写入范围。未来批次只有获得相应事实确认、形成自己的清单后才可使用；不自动继承。

对含N个按 `ordinal` 排序的事件：

| 字段 | 前N−1个事件 | 第N个事件 |
|---|---|---|
| `events[].result` | `not_found` | `found` |
| `events[].outcome.target_found` | `false` | `true` |
| 对应 `inspections[].result` | `not_found` | `found` |
| 两处 `result_source` | `user_confirmed_current_dataset_outcomes_2026_11_05` | 同左 |

同时写入：

- `metadata.only_final_event_found_target_confirmed=true`。
- `episode_outcome.result=found`，`success_event_id` 指向最后事件，`success_event_order=N`。
- `derived_statistics.found_event_count=1`，`events_before_success=N-1`，`events_through_success=N`。
- 原模型结果保存在 `events[].visual_outcome_before_protocol`；名称沿用既有格式，虽然其中含protocol，来源含义仍是“人工结果合并前的视觉记录”。
- `README.annotation_record.label_source` 明确视觉与人工结果两类来源，`manifest` 指向事实文件，回执保存事实文件哈希。

**不得从此事实推导**：无重访、已穷尽搜索、每次检查有效、确切找到帧、目标精确实例ID、目标精确隐藏位置、参与者收到过何种搜索前信息。最后事件可有多个潜在区域，不能任选一个作为真实位置。`target_identity_confirmed` 保留原视觉身份确认值；人工确认“找到目标”不等于已把画面中某物体与目录实例匹配。

事件描述保留模型对采样帧可见性的原始描述。出现“采样帧未清楚看到目标”，同时人工结果为found，并非必须删除前者；这反映观察范围与人工结果来源不同。

## 3. 顶层布局

`schema` 为 `spatial_memory.search_episode.additive_research_temporal.potential_regions.v1`。其余顶层字段如下：

| 字段 | 信息内容 | 主要来源 |
|---|---|---|
| `README` | 用途、范围、解释规则、版本扩展、生成记录、缺失项 | 程序装配、审计 |
| `metadata` | episode/场景/参与者/目标、帧率、原MD元数据、实验背景字段 | 原数据及MD；缺值null |
| `scene_memory` | 物体档案、参考空间树、时间化直接父关系 | Astra筛帧后Qwen视觉标注 |
| `search_event_states` | 三帧事件、状态、操作区域、潜在搜索区域、空间变化 | 原三帧及Qwen；结果另用人工确认 |
| `search_units` | 事件局部检查区域、区域属性、待补测量 | 事件描述及程序ID |
| `inspections` | 按事件排列的检查记录和结果 | 主事件映射、人工结果 |
| `episode_outcome` | 全过程结果和成功事件引用 | 用户确认；精确位置/帧未确认则null |
| `derived_statistics` | 事件数、已找到次数、成功前事件数、重访待核信息 | 从已核验字段计算 |
| `history` | 历史/曝光记录接口 | 当前未结构化完整记录时null |
| `hidden` | 真实位置及搬移记录接口 | 当前只保留有来源的文字，未确认ID则null |
| `priors` | 习惯、同住者、语义先验 | 未采集值null |

缺少“键”与值为null不同。严格schema要求应有的键存在。禁止用0、false、空串代替未知。[]表示列表没有已记录项，不能自动解释为“现实中没有发生”。

## 4. metadata：身份、目标、采样和原始说明

- `episode_id`：唯一数据ID，连接所有输出、事件和目录。
- `scene_id`：从本条MD的场景读取；不能只按相同视频名跨场景拼接数据。
- `zone_id`：本条局部工作区ID。`search_scope`、`search_scope_source`、`full_search_coverage_confirmed` 说明覆盖范围，不把局部证据称为完整房间。
- `fps`、`time_axis`：原视频帧率和时间轴。事件时间来自文件名帧号/fps，非rosbag抽样序号。
- `target.description`：MD“要找的物体”；`category`、`confirmed_instance_id` 未确认则null；`identity_source` 记录来源。
- `participant_id` 来自“采集人员”；`home_id`、`encoding_condition`、`target_instance_id`、`target_category`、`target_description_to_participant`、`mobility_tier` 有字段但不臆填。
- `video_duration_s`、`search_start_unit`、`search_start_s`、`search_end_s`、`boundary_sampling_interval_s` 是完整接口。当前未做相应边界标注的值保持null，不能把第一张/最后一张采样图当完整动作边界。
- `source_annotation_metadata` 保留MD的7项原文：采集人员、场景、要找的物体、是否在容器内、是否有曝光、物体原始位置、物体最后所在位置。
- `source_annotation_markdown` 与 `source_annotation_markdown_sha256` 绑定原说明文档；`provenance` 保留视觉标注来源。
- `pre_search_information_received` 为null：MD“有曝光”未被自动等同于“参与者收到搜索前信息”；原曝光标记仍完整保留。
- `no_revisits_confirmed` 为null：最后找到不表示此前没有重复搜索同一区域。

## 5. scene_memory：物体档案、参考树与时序

### 5.1 物体档案

`objects[]` 以稳定 `id` 建档，`spatial_node_id` 关联空间节点；`identity` 包括名称/类别等身份描述；`appearance` 记录外观；`commonsense` 是类别常识，不是该实例已发生事实；`affordances` 描述容纳、支承、遮挡等可能能力，不代表容器实际装有这些物品，也不凭视觉编造升数。

`object_schema` 声明物体档案契约及解释规则。原物体历史/末次观测字段并非本版新增；此处继续使用已确定的无历史物体档案契约，事件时间信息由下面的关系与事件承担。

### 5.2 树结构

- `spatial_structure.nodes[]`：已能安排到参考层级中的节点列表；每个节点递归包含 `children[]`。
- `context_nodes[]`：相关上下文节点，单独保留，不强行放入确定工作区层级。
- `unlocated_nodes[]`：已识别、但缺少可靠直接父级/归属依据的节点。它们不是无用数据，也不是“物体不存在”；禁止为让树看起来完整而猜父节点。
- `root_id` 为局部工作区根，`reference_frame` 说明参考坐标/视角，`coverage` 说明几何、房间位置和导航连接的覆盖限制。
- 节点 `id` 稳定；非区域节点需要 `object_ref`；`placement` 保留参考布局，`relations`/`associations`/`cross_level_relations` 保留有证据的其他关系；`identity_candidates` 表示可能同一实体但尚未确认的候选。

最终主JSON保留原样板的参考 `children`，不会把它重写成按事件重复排列的另一种整体格式。**参考树不等于整段时间始终不变的树。**

### 5.3 每个节点的 parent_relations

| 字段 | 含义 |
|---|---|
| `relation_id` | 一段直接父关系的唯一ID |
| `parent_id` | 该时间段内直接父节点；null为未知/无已支持父级，不自动表示在手中 |
| `predicate` | `on / inside / mounted_on / hanging_on / within_region / unknown` |
| `relative_position` | 相对直接父级的位置，未知为null |
| `valid_from_frame` | 有效区间起点，原视频帧号 |
| `valid_to_frame_exclusive` | 有效区间排他终点；null为后续未观测到结束，不宣称永久有效 |
| `started_by_event_id`、`ended_by_event_id` | 由哪个事件建立/终止；参考观察可没有事件ID |
| `start_evidence_state_id`、`end_evidence_state_id` | 有事件边界时必须能找到该事件的对应采样状态 |
| `boundary_status` | 参考场景观察或首次事件观察，均注明未经独立人工审核 |

区间约定是 `[start,end)`，即 start ≤ f < end。查询帧f的树时选择该帧有效的直接父关系，按parent_id重建；不能只读静态children。不能从多时刻场景拼图捏造从第0帧就成立的关系。同一节点关系不应重叠，不能成环，所有非空父ID须可解析。

`parent_relations=null` 表示未做相应时序标注；空数组在生成约定中用于没有父级的根节点，不应把漏标物体伪装成根节点。

搬动容器只改变容器的直接父级；其内部物体如果仍在容器中，直接父级不变。接触不等于拿起；视角变动不等于物体移动；目录中缺少的物体要记录不确定性，不能生成不存在的node_id。

## 6. search_event_states：每次事件

该对象包含 `event_count`、`image_count`、`image_directory`、`frame_numbering`、`entity_reference_catalog`、`provenance` 及 `events[]`。

### 6.1 事件身份和过程

- `event_id` 全局唯一，`ordinal` 从1连续编号。
- `event_summary`：三帧内的事件描述；`observed_changes[]`：可见变化。不要从三张静态图补写完整连续动作或确切意图。
- `search_region`：原有的局部实际接触/操作区域描述；`search_region_status` 解释其确认程度。
- `source_human_interval_id`、`previous_json_interval_seconds`、`interval_alignment` 保留与旧区间的关系；未比对时明示not_compared。
- `phase_frame_indices` 记录start/middle/end原帧号，`supplied_image_interval_seconds` 是采样窗口，不是确认的操作起止。
- `result`、`result_source`、`outcome` 按第2节处理。`visual_outcome_before_protocol` 不随人工结果重写。
- `search_region_id` 是事件局部区域ID；`region_visit_index`、`region_total_visits`、`is_revisit` 未跨事件核对则null。
- `validity`、`validity_status`、`inspect_cost_s` 预留检查有效性与耗时测量，不能由not_found反推有效排除。

### 6.2 states：恰好三帧

每事件恰好三个状态，顺序 `start → middle → end`。字段包括：

| 字段 | 含义 |
|---|---|
| `state_id` | 由事件ID与phase生成；供关系证据引用 |
| `phase` / `phase_label` | 阶段枚举与显示名 |
| `original_video_frame_index` / `frame_label` | 原视频帧标签；后者为f加帧号 |
| `original_video_seconds` | 文件名帧号/fps |
| `image_filename` / `image_path` / `image_sha256` | 原图身份、真实路径、内容哈希 |
| `state_description` | 这一帧独立的可见描述 |
| `hand_action` | 这一帧可见手部动作 |
| `visibility_and_occlusion` | 可见性与遮挡 |
| `focus_object_states[]` | 事件局部实体引用和该帧状态 |
| `evidence_status` | 模型视觉标注及审核状态 |

`entity_reference_catalog` 将事件局部实体ID映射到label、场景候选ID和链接状态；同名只能是候选，不能当已验证同一实例。

输入审计发现文件名标签与顺序解码帧号偏差时，同时保存两种编号及内容匹配审计。交付保持用户提供的文件名标签；不得暗改帧号或冒充已校准采集时间。

### 6.3 昨天新增字段：potential_search_regions

每个事件必须有 `potential_search_regions` 数组。**这是2026-11-04在既有样板基础上新增的模型生成信息**，与事件描述在同一次Qwen调用中生成，不增加每事件调用次数；输出token量会增加。

| 子字段 | 类型 | 生成方式与要求 |
|---|---|---|
| `description` | string | 模型描述“本次可能检查哪些空间区域”，例如抬起枕头下面、打开抽屉内部 |
| `visual_basis` | string | 模型给出三帧中可见的支持依据 |
| `evidence_frames` | integer[] | 模型引用当前事件提供的帧，至少1个、无重复、不能跨事件 |
| `uncertainty` | string | 模型说明遮挡、身份或范围不确定性 |
| `region_id` | string | 程序生成的事件内候选ID，如event_id:potential_region_01 |
| `evidence_state_ids` | string[] | 程序按证据帧绑定本事件state_id，并逐一核验 |
| `annotation_status` | string | 固定 `potential_search_region_inferred_not_confirmed_intention` |

它既不是“全场景所有可能藏目标的地方”，也不证明候选区域都实际检查过；不因目标可能放得下就泛列其他背景区域。没有足够支持可为[]，原事件不确定性中应说明。该列表不能直接写入隐藏真值、无重访结论或成功位置。

三个区域概念要分清：`search_region` 是本事件实际操作范围描述；`potential_search_regions` 是有图像支持的潜在检查区域列表；`search_units` 是当前导出器使用的事件局部检查区域目录。三者不是天然一一对应的物理区域ID。

### 6.4 spatial_changes

每个变化含 `node_id`、`change_type`、`change_frame`、`parent_id`、`predicate`、`relative_position`、`evidence_state_id`。变化帧须属于当前事件采样帧，node_id/parent_id须来自场景目录。`spatial_change_annotation` 包含模型标注状态、来源、语言和 `uncertainties[]`。空变化列表只表示没有可支持的目录关系更新，不表示没有动作。

## 7. search_units、inspections 与重访

`search_units` 带scope/catalog_status/coverage_note/id_policy及list。每项包含unit_id、zone_id、label、type、is_container、boundary、source_entity_ids、entity_link_basis、visit_count/visit_count_source，以及新增实验接口room、parent_id、inspect_cost_s、inspect_cost_trials_s、volume_l、surface_vs_enclosed。

当前每事件一个局部inspection_area，不保证跨事件独立，也不保证全场景覆盖。`source_entity_ids=[]` 不能被理解为区域不存在。

`inspections[]` 按事件顺序记录event_id、order、unit_ids、granularity、action_summary、sample_window_s、source_region_text、result/result_source、unit_link_basis、区域/重访字段、validity/start_s/end_s/obstruction/revisit_kind等。当前目标导出器要求每次检查恰好一个unit_id；潜在区域数组有多项时不得自动拆成多次检查或无根据合并为一个已确认区域。

用户对重访的定义继续有效：**不同事件搜索同一实际区域，就是重访。** 但本版本尚未完成稳定跨事件区域身份合并；相关统计保留null。`potential_search_regions` 的事件局部ID不同不代表没有重访。结果确认与区域身份合并是两个独立问题。

## 8. 实验记录、先验、全过程结果

- `history`：保留历史/曝光记录接口。当前null意味着未完整结构化，不强行使用空数组表达“确认没有曝光”。
- `hidden.target_location`、`original_unit_id`：未确认真实空间unit映射时null。原始/最后位置文字见metadata.source_annotation_metadata。`relocation_source`、`relocation_evidence_state`、`evidence_coverage`、`evidence_confidence`、`intervention_time_h` 预留搬移与证据字段。
- `priors.habit_prior.counts`、`cohabitant_counts.counts`、`semantic_scores.scores` 的键与unit目录一致。未知数量/分数为null，不捏造为0或概率；semantic_scores.source记录来源。
- `episode_outcome.final_location`、`target_instance_id`、`found_evidence_frames`、`stop_reason`、`duration_s` 没有相应确认时保持null。用户确认最后事件找到，仅补充其对应结果和事件引用。
- `derived_statistics`：事件数与目录项数分别统计；两者相同也不表示独立物理区域数相同。重访计数和每区访问次数须待身份核对。

## 9. 两份目标 JSON 的字段映射

| 主JSON来源 | 搜索过程目标JSON | experiment_records目标JSON |
|---|---|---|
| README.annotation_record | annotation_profile / label_source / label_status / source / scene_ref | 不复制全部生成记录 |
| metadata规定字段 | metadata | participant_id、home_id、encoding_condition、target_description_to_participant |
| search_units.list | units[]：区域身份、边界、测量接口 | units以unit_id为键：inspect_cost_trials_s、volume_l |
| inspections + 对应事件三帧 | inspections[]：采样窗口、结果、证据、有效性、遮挡和重访类型 | 不输出事件序列 |
| episode_outcome | result、success_order、found_unit_id、found_evidence_frames、stop_reason、duration_s | 不输出搜索过程结果 |
| history | history | history |
| hidden | target_location.unit_id映射true_unit_id，其余指定字段 | 同左 |
| priors | habit/cohabitant/semantic三类 | habit/cohabitant两类 |
| README.completeness | completeness | 通过保留null体现缺失值 |

**编号转换**：主事件ordinal及主inspection.order从1开始；目标inspections.order从0开始，目标success_order=N−1。不要混用。目标证据帧显示为 `left_f%06d`。scene_ref以相对路径指回主JSON里的scene_memory；空间树和潜在区域保留在主JSON，不丢弃，也不擅自向两个既定目标格式塞入未约定字段。

两个导出可完整提供约定字段结构；它们不能凭空提供尚未采集的实验数值。`completeness.spec_complete=false`、缺项列表与null应如实保留，不能将“结构完整”声称为“信息真值全部已知”。

## 10. 校验与可追溯性

发布前核对：严格schema；原字段保存；目标名与MD哈希；事件数量与顺序；三帧phase、帧号、路径、哈希；潜在区域证据帧与状态ID；时序边界与父节点；两个目标导出的字段；读回；人工结果限定清单与一致性。

人工结果投影只改结果相关字段与来源，保持空间树、物体、三帧、事件文字、潜在区域、时序更新、隐藏位置、先验及重访未知值。回执记录源输出与事实文件哈希。模型响应、基础JSON、既有81条及原始数据继续保留。

视觉/空间描述是模型生成且未经独立人工逐条验收；成功结果是用户确认。通过schema不等于模型的每一句视觉描述都正确。
