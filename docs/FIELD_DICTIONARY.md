# 11-4 完整字段字典

由版本内的JSON Schema生成；语义、来源与结果确认约定见 [OUTPUT_SPEC.md](OUTPUT_SPEC.md)。`必有`指父对象存在时该键必须存在；null是允许的值类型，不等于可以省略。数组行表示成员类型。`#/$defs/node`在末尾单列并递归用于children，避免无限展开。动态键主要用于实体/先验目录。

## README

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `README` | 必有 | `object` | additionalProperties=False |
| `README.purpose` | 必有 | `string` | minLength=1 |
| `README.rules` | 必有 | `array` | — |
| `README.rules[]` | 必有 | `string` | minLength=1 |
| `README.scope` | 必有 | `array` | — |
| `README.scope[]` | 必有 | `string` | minLength=1 |
| `README.format_extension` | 必有 | `object` | additionalProperties=False |
| `README.format_extension.version` | 必有 | `string` | — |
| `README.format_extension.policy` | 必有 | `string` | — |
| `README.format_extension.legacy_revisit_policy` | 必有 | `string` | — |
| `README.format_extension.temporal_policy` | 必有 | `string` | — |
| `README.format_extension.null_policy` | 必有 | `string` | — |
| `README.format_extension.identity_policy` | 必有 | `string` | — |
| `README.annotation_record` | 必有 | `object` | additionalProperties=False |
| `README.annotation_record.annotation_profile` | 必有 | `string` | — |
| `README.annotation_record.label_source` | 必有 | `string` | — |
| `README.annotation_record.label_status` | 必有 | `string` | — |
| `README.annotation_record.source_annotation_json` | 必有 | `string` | — |
| `README.annotation_record.source_annotation_sha256` | 必有 | `string` | — |
| `README.annotation_record.temporal_annotation_json` | 必有 | `string/null` | — |
| `README.annotation_record.manifest` | 必有 | `string/null` | — |
| `README.annotation_record.completion` | 必有 | `string/null` | — |
| `README.annotation_record.home_id_basis` | 必有 | `string/null` | — |
| `README.annotation_record.scene_ref` | 必有 | `string` | — |
| `README.annotation_record.model_evidence_available_after_s` | 必有 | `number/null` | — |
| `README.annotation_record.new_model_calls` | 必有 | `integer` | — |
| `README.annotation_record.temporal_annotation_sha256` | 必有 | `string` | — |
| `README.completeness` | 必有 | `object` | additionalProperties=False |
| `README.completeness.video_labels_complete` | 必有 | `boolean` | — |
| `README.completeness.spec_complete` | 必有 | `boolean` | — |
| `README.completeness.required_fields_present` | 必有 | `boolean` | — |
| `README.completeness.experiment_values_required_for_template` | 必有 | `boolean` | — |
| `README.completeness.temporal_links_verified` | 必有 | `boolean` | — |
| `README.completeness.missing` | 必有 | `object` | additionalProperties=False |
| `README.completeness.missing.experiment_records` | 必有 | `array` | — |
| `README.completeness.missing.experiment_records[]` | 必有 | `string` | — |
| `README.completeness.missing.unit_measurements` | 必有 | `array` | — |
| `README.completeness.missing.unit_measurements[]` | 必有 | `string` | — |
| `README.completeness.missing.pending_annotations` | 必有 | `array` | — |
| `README.completeness.missing.pending_annotations[]` | 必有 | `string` | — |
| `README.completeness.missing.spatial_temporal_review` | 必有 | `array` | — |
| `README.completeness.missing.spatial_temporal_review[]` | 必有 | `string` | — |

## derived_statistics

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `derived_statistics` | 必有 | `object` | additionalProperties=False |
| `derived_statistics.counting_unit` | 必有 | `const "original_search_event"` | — |
| `derived_statistics.events_before_success` | 必有 | `integer/null` | minimum=0 |
| `derived_statistics.events_through_success` | 必有 | `integer/null` | minimum=0 |
| `derived_statistics.found_event_count` | 必有 | `integer` | minimum=0 |
| `derived_statistics.note` | 必有 | `string` | minLength=1 |
| `derived_statistics.search_event_count` | 必有 | `integer` | minimum=0 |
| `derived_statistics.search_unit_catalog_count` | 必有 | `integer` | minimum=0 |
| `derived_statistics.revisit_count` | 必有 | `integer/null` | minimum=0 |
| `derived_statistics.revisit_count_source` | 必有 | `string/null` | — |
| `derived_statistics.region_visit_counts` | 必有 | `array/null` | — |
| `derived_statistics.region_visit_counts[]` | 必有 | `object` | additionalProperties=False |
| `derived_statistics.region_visit_counts[].unit_id` | 必有 | `string` | minLength=1 |
| `derived_statistics.region_visit_counts[].description` | 必有 | `string` | minLength=1 |
| `derived_statistics.region_visit_counts[].count` | 必有 | `integer/null` | minimum=1 |

## episode_outcome

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `episode_outcome` | 必有 | `object` | additionalProperties=False |
| `episode_outcome.final_location` | 必有 | `object 或 null` | — |
| `episode_outcome.final_location.description` | 必有 | `string` | minLength=1 |
| `episode_outcome.final_location.evidence_state_id` | 必有 | `string 或 null` | — |
| `episode_outcome.final_location.location_granularity` | 必有 | `string` | minLength=1 |
| `episode_outcome.final_location.note` | 必有 | `string` | minLength=1 |
| `episode_outcome.final_location.sample_time_s` | 必有 | `number 或 null` | — |
| `episode_outcome.final_location.source` | 必有 | `string` | minLength=1 |
| `episode_outcome.final_location.unit_id` | 必有 | `string` | minLength=1 |
| `episode_outcome.result` | 必有 | `enum ["found", "not_found", "unknown"]` | — |
| `episode_outcome.success_event_id` | 必有 | `string/null` | — |
| `episode_outcome.success_event_order` | 必有 | `integer/null` | minimum=1 |
| `episode_outcome.target_instance_id` | 必有 | `string/null` | — |
| `episode_outcome.result_source` | 必有 | `string/null` | — |
| `episode_outcome.found_evidence_frames` | 必有 | `null 或 array` | — |
| `episode_outcome.found_evidence_frames[]` | 必有 | `string` | — |
| `episode_outcome.stop_reason` | 必有 | `string/null` | — |
| `episode_outcome.duration_s` | 必有 | `number/null` | — |

## inspections

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `inspections` | 必有 | `array` | — |
| `inspections[]` | 必有 | `object` | additionalProperties=False |
| `inspections[].action_summary` | 必有 | `string` | minLength=1 |
| `inspections[].event_id` | 必有 | `string` | minLength=1 |
| `inspections[].granularity` | 必有 | `enum ["single_unit_event", "multi_unit_event"]` | — |
| `inspections[].order` | 必有 | `integer` | — |
| `inspections[].result` | 必有 | `enum ["unknown", "found", "not_found"]` | — |
| `inspections[].result_source` | 必有 | `string/null` | — |
| `inspections[].sample_window_s` | 必有 | `array` | — |
| `inspections[].sample_window_s[]` | 必有 | `number` | — |
| `inspections[].source_region_text` | 必有 | `string` | minLength=1 |
| `inspections[].unit_ids` | 必有 | `array` | minItems=1; uniqueItems=True |
| `inspections[].unit_ids[]` | 必有 | `string` | minLength=1 |
| `inspections[].unit_link_basis` | 必有 | `string` | minLength=1 |
| `inspections[].search_region_id` | 必有 | `string` | minLength=1 |
| `inspections[].region_visit_index` | 必有 | `integer/null` | minimum=1 |
| `inspections[].region_total_visits` | 必有 | `integer/null` | minimum=1 |
| `inspections[].is_revisit` | 必有 | `boolean/null` | — |
| `inspections[].region_identity_source` | 必有 | `string/null` | — |
| `inspections[].validity` | 必有 | `string/number/null` | — |
| `inspections[].validity_status` | 必有 | `enum ["pending_definition", "annotated"]` | — |
| `inspections[].inspect_cost_s` | 必有 | `number/null` | minimum=0 |
| `inspections[].search_region` | 必有 | `string` | minLength=1 |
| `inspections[].start_s` | 必有 | `number/null` | — |
| `inspections[].end_s` | 必有 | `number/null` | — |
| `inspections[].obstruction` | 必有 | `string/null` | — |
| `inspections[].revisit_kind` | 必有 | `string/null` | — |

## metadata

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `metadata` | 必有 | `object` | additionalProperties=False |
| `metadata.episode_id` | 必有 | `string` | minLength=1 |
| `metadata.fps` | 必有 | `number` | — |
| `metadata.full_search_coverage_confirmed` | 必有 | `boolean` | — |
| `metadata.only_final_event_found_target_confirmed` | 必有 | `boolean` | — |
| `metadata.provenance` | 必有 | `string` | minLength=1 |
| `metadata.scene_id` | 必有 | `string` | minLength=1 |
| `metadata.search_scope` | 必有 | `string` | minLength=1 |
| `metadata.search_scope_source` | 必有 | `string` | minLength=1 |
| `metadata.target` | 必有 | `object` | additionalProperties=False |
| `metadata.target.category` | 必有 | `string/null` | — |
| `metadata.target.confirmed_instance_id` | 必有 | `string/null` | — |
| `metadata.target.description` | 必有 | `string` | minLength=1 |
| `metadata.target.identity_source` | 必有 | `string` | minLength=1 |
| `metadata.time_axis` | 必有 | `const "original_video_seconds"` | — |
| `metadata.zone_id` | 必有 | `string` | minLength=1 |
| `metadata.participant_id` | 必有 | `string/null` | — |
| `metadata.home_id` | 必有 | `string/null` | — |
| `metadata.encoding_condition` | 必有 | `string/null` | — |
| `metadata.target_instance_id` | 必有 | `string/null` | — |
| `metadata.target_category` | 必有 | `string/null` | — |
| `metadata.pre_search_information_received` | 必有 | `boolean/null` | — |
| `metadata.pre_search_information_source` | 必有 | `string/null` | — |
| `metadata.research_fields_source` | 必有 | `string/null` | — |
| `metadata.pending_metadata_fields` | 必有 | `array` | — |
| `metadata.pending_metadata_fields[]` | 必有 | `string` | minLength=1 |
| `metadata.no_revisits_confirmed` | 必有 | `boolean/null` | — |
| `metadata.target_description_to_participant` | 必有 | `string/null` | — |
| `metadata.mobility_tier` | 必有 | `string/null` | — |
| `metadata.video_duration_s` | 必有 | `number/null` | — |
| `metadata.search_start_unit` | 必有 | `string/null` | — |
| `metadata.search_start_s` | 必有 | `number/null` | — |
| `metadata.search_end_s` | 必有 | `number/null` | — |
| `metadata.boundary_sampling_interval_s` | 必有 | `number/null` | — |
| `metadata.source_annotation_metadata` | 必有 | `object` | additionalProperties=False |
| `metadata.source_annotation_metadata.采集人员` | 必有 | `string/null` | — |
| `metadata.source_annotation_metadata.场景` | 必有 | `string/null` | — |
| `metadata.source_annotation_metadata.要找的物体` | 必有 | `string/null` | — |
| `metadata.source_annotation_metadata.是否在容器内` | 必有 | `string/null` | — |
| `metadata.source_annotation_metadata.是否有曝光` | 必有 | `string/null` | — |
| `metadata.source_annotation_metadata.物体原始位置` | 必有 | `string/null` | — |
| `metadata.source_annotation_metadata.物体最后所在位置` | 必有 | `string/null` | — |
| `metadata.source_annotation_markdown` | 必有 | `string` | minLength=1 |
| `metadata.source_annotation_markdown_sha256` | 必有 | `string` | minLength=1 |

## scene_memory

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `scene_memory` | 必有 | `object` | additionalProperties=False |
| `scene_memory.object_schema` | 必有 | `object` | additionalProperties=False |
| `scene_memory.object_schema.rules` | 必有 | `object` | additionalProperties=False |
| `scene_memory.object_schema.rules.affordances` | 必有 | `string` | minLength=1 |
| `scene_memory.object_schema.rules.appearance` | 必有 | `string` | minLength=1 |
| `scene_memory.object_schema.rules.commonsense` | 必有 | `string` | minLength=1 |
| `scene_memory.object_schema.rules.identity` | 必有 | `string` | minLength=1 |
| `scene_memory.object_schema.rules.provenance` | 必有 | `string` | minLength=1 |
| `scene_memory.object_schema.rules.spatial_node_id` | 必有 | `string` | minLength=1 |
| `scene_memory.object_schema.version` | 必有 | `const "ego.scene.self_contained_object_profiles.v8.no_history"` | — |
| `scene_memory.objects` | 必有 | `array` | — |
| `scene_memory.objects[]` | 必有 | `object` | additionalProperties=False |
| `scene_memory.objects[].affordances` | 可选 | `object` | additionalProperties=False |
| `scene_memory.objects[].affordances.containment` | 可选 | `string` | minLength=1 |
| `scene_memory.objects[].affordances.support` | 可选 | `string` | minLength=1 |
| `scene_memory.objects[].affordances.occlusion` | 可选 | `string` | minLength=1 |
| `scene_memory.objects[].appearance` | 必有 | `object` | additionalProperties=False |
| `scene_memory.objects[].appearance.attributes` | 必有 | `object` | additionalProperties=动态键值类型 array |
| `scene_memory.objects[].appearance.attributes.<动态键>` | 可选 | `array` | uniqueItems=True |
| `scene_memory.objects[].appearance.attributes.<动态键>[]` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].appearance.shape` | 可选 | `string` | minLength=1 |
| `scene_memory.objects[].appearance.shape_status` | 可选 | `string` | minLength=1 |
| `scene_memory.objects[].commonsense` | 必有 | `array` | minItems=1; maxItems=3; uniqueItems=True |
| `scene_memory.objects[].commonsense[]` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].id` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].identity` | 必有 | `object` | additionalProperties=False |
| `scene_memory.objects[].identity.category` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].identity.entity_type` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].identity.identity_status` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].identity.name` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].identity.source_categories` | 必有 | `array` | — |
| `scene_memory.objects[].identity.source_categories[]` | 必有 | `string` | minLength=1 |
| `scene_memory.objects[].spatial_node_id` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure` | 必有 | `object` | additionalProperties=False |
| `scene_memory.spatial_structure.context_nodes` | 必有 | `array` | — |
| `scene_memory.spatial_structure.context_nodes[]` | 必有 | `#/$defs/node` | — |
| `scene_memory.spatial_structure.coverage` | 必有 | `object` | additionalProperties=False |
| `scene_memory.spatial_structure.coverage.metric_geometry` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.coverage.navigation_connectivity` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.coverage.parent_room` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.coverage.position_in_room` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.description` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.identity_candidates` | 必有 | `array` | — |
| `scene_memory.spatial_structure.identity_candidates[]` | 必有 | `object` | additionalProperties=False |
| `scene_memory.spatial_structure.identity_candidates[].entity_ids` | 必有 | `array` | — |
| `scene_memory.spatial_structure.identity_candidates[].entity_ids[]` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.identity_candidates[].id` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.identity_candidates[].name` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.identity_candidates[].status` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.nodes` | 必有 | `array` | minItems=1 |
| `scene_memory.spatial_structure.nodes[]` | 必有 | `#/$defs/node` | — |
| `scene_memory.spatial_structure.reference_frame` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.root_id` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.schema` | 必有 | `const "ego.scene.nested_local_spatial_graph.v4"` | — |
| `scene_memory.spatial_structure.semantics` | 必有 | `object` | additionalProperties=False |
| `scene_memory.spatial_structure.semantics.cross_level` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.semantics.evidence` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.semantics.hierarchy` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.semantics.identity` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.semantics.missing` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.semantics.position` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.semantics.query` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.semantics.time` | 必有 | `string` | minLength=1 |
| `scene_memory.spatial_structure.unlocated_nodes` | 必有 | `array` | — |
| `scene_memory.spatial_structure.unlocated_nodes[]` | 必有 | `#/$defs/node` | — |
| `scene_memory.spatial_structure.timebase` | 必有 | `object` | additionalProperties=False |
| `scene_memory.spatial_structure.timebase.unit` | 必有 | `string` | — |
| `scene_memory.spatial_structure.timebase.index_base` | 必有 | `integer` | — |
| `scene_memory.spatial_structure.timebase.fps` | 必有 | `number` | — |
| `scene_memory.spatial_structure.timebase.interval_convention` | 必有 | `string` | — |
| `scene_memory.spatial_structure.timebase.seconds_formula` | 必有 | `string` | — |
| `scene_memory.spatial_structure.temporal_status` | 必有 | `string` | — |

## schema

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `schema` | 必有 | `const "spatial_memory.search_episode.additive_research_temporal.potential_regions.v1"` | — |

## search_event_states

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `search_event_states` | 必有 | `object` | additionalProperties=False |
| `search_event_states.entity_reference_catalog` | 必有 | `object` | additionalProperties=动态键值类型 object |
| `search_event_states.entity_reference_catalog.<动态键>` | 可选 | `object` | additionalProperties=False |
| `search_event_states.entity_reference_catalog.<动态键>.label` | 必有 | `string` | minLength=1 |
| `search_event_states.entity_reference_catalog.<动态键>.link_status` | 必有 | `string` | minLength=1 |
| `search_event_states.entity_reference_catalog.<动态键>.memory_entity_candidates` | 必有 | `array` | — |
| `search_event_states.entity_reference_catalog.<动态键>.memory_entity_candidates[]` | 必有 | `string` | minLength=1 |
| `search_event_states.event_count` | 必有 | `integer` | — |
| `search_event_states.events` | 必有 | `array` | — |
| `search_event_states.events[]` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].event_id` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].event_summary` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].interval_alignment` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].interval_alignment.basis_for_this_record` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].interval_alignment.note` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].interval_alignment.status` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].observed_changes` | 必有 | `array` | — |
| `search_event_states.events[].observed_changes[]` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].ordinal` | 必有 | `integer` | minimum=1 |
| `search_event_states.events[].outcome` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].outcome.reason` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].outcome.target_found` | 必有 | `boolean/null` | — |
| `search_event_states.events[].outcome.target_identity_confirmed` | 必有 | `boolean` | — |
| `search_event_states.events[].phase_frame_indices` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].phase_frame_indices.end` | 必有 | `integer` | — |
| `search_event_states.events[].phase_frame_indices.middle` | 必有 | `integer` | — |
| `search_event_states.events[].phase_frame_indices.start` | 必有 | `integer` | — |
| `search_event_states.events[].previous_json_interval_seconds` | 必有 | `array` | — |
| `search_event_states.events[].previous_json_interval_seconds[]` | 必有 | `number` | — |
| `search_event_states.events[].search_region` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].search_region_status` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].source_human_interval_id` | 必有 | `string/null` | — |
| `search_event_states.events[].states` | 必有 | `array` | minItems=3; maxItems=3 |
| `search_event_states.events[].states[]` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].states[].evidence_status` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].focus_object_states` | 必有 | `array` | — |
| `search_event_states.events[].states[].focus_object_states[]` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].states[].focus_object_states[].entity_ref` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].focus_object_states[].observed_state` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].frame_label` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].frame_namespace` | 必有 | `const "original_video_frame_index_from_filename"` | — |
| `search_event_states.events[].states[].hand_action` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].image_filename` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].image_path` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].image_sha256` | 必有 | `string` | — |
| `search_event_states.events[].states[].original_video_frame_index` | 必有 | `integer` | minimum=0 |
| `search_event_states.events[].states[].original_video_seconds` | 必有 | `number` | — |
| `search_event_states.events[].states[].phase` | 必有 | `enum ["start", "middle", "end"]` | — |
| `search_event_states.events[].states[].phase_label` | 必有 | `enum ["开始", "进行中", "结束"]` | — |
| `search_event_states.events[].states[].state_description` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].state_id` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].states[].visibility_and_occlusion` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].supplied_image_interval_seconds` | 必有 | `array` | — |
| `search_event_states.events[].supplied_image_interval_seconds[]` | 必有 | `number` | — |
| `search_event_states.events[].search_region_id` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].region_visit_index` | 必有 | `integer/null` | minimum=1 |
| `search_event_states.events[].region_total_visits` | 必有 | `integer/null` | minimum=1 |
| `search_event_states.events[].is_revisit` | 必有 | `boolean/null` | — |
| `search_event_states.events[].region_identity_source` | 必有 | `string/null` | — |
| `search_event_states.events[].validity` | 必有 | `string/number/null` | — |
| `search_event_states.events[].validity_status` | 必有 | `enum ["pending_definition", "annotated"]` | — |
| `search_event_states.events[].inspect_cost_s` | 必有 | `number/null` | minimum=0 |
| `search_event_states.events[].result` | 必有 | `enum ["unknown", "found", "not_found"]` | — |
| `search_event_states.events[].result_source` | 必有 | `string/null` | — |
| `search_event_states.events[].visual_outcome_before_protocol` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].visual_outcome_before_protocol.reason` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].visual_outcome_before_protocol.target_found` | 必有 | `boolean/null` | — |
| `search_event_states.events[].visual_outcome_before_protocol.target_identity_confirmed` | 必有 | `boolean` | — |
| `search_event_states.events[].spatial_changes` | 必有 | `null 或 array` | — |
| `search_event_states.events[].spatial_changes[]` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].spatial_changes[].node_id` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].spatial_changes[].change_type` | 必有 | `enum ["establish", "reparent", "unparent", "position_change"]` | — |
| `search_event_states.events[].spatial_changes[].change_frame` | 必有 | `integer` | minimum=0 |
| `search_event_states.events[].spatial_changes[].parent_id` | 必有 | `string/null` | minLength=1 |
| `search_event_states.events[].spatial_changes[].predicate` | 必有 | `enum ["on", "inside", "mounted_on", "hanging_on", "within_region", "unknown"]` | — |
| `search_event_states.events[].spatial_changes[].relative_position` | 必有 | `string/null` | minLength=1 |
| `search_event_states.events[].spatial_changes[].evidence_state_id` | 必有 | `string` | — |
| `search_event_states.events[].spatial_change_annotation` | 必有 | `object 或 object` | — |
| `search_event_states.events[].spatial_change_annotation.status` | 必有 | `const "local_qwen_visual_inference_unreviewed"` | — |
| `search_event_states.events[].spatial_change_annotation.uncertainties` | 必有 | `array` | — |
| `search_event_states.events[].spatial_change_annotation.uncertainties[]` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].spatial_change_annotation.source` | 必有 | `const "original_event_triplet_images"` | — |
| `search_event_states.events[].spatial_change_annotation.description_language` | 必有 | `const "en"` | — |
| `search_event_states.events[].spatial_change_annotation.status` | 必有 | `const "not_annotated"` | — |
| `search_event_states.events[].potential_search_regions` | 必有 | `array` | — |
| `search_event_states.events[].potential_search_regions[]` | 必有 | `object` | additionalProperties=False |
| `search_event_states.events[].potential_search_regions[].description` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].potential_search_regions[].visual_basis` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].potential_search_regions[].evidence_frames` | 必有 | `array` | minItems=1; uniqueItems=True |
| `search_event_states.events[].potential_search_regions[].evidence_frames[]` | 必有 | `integer` | minimum=0 |
| `search_event_states.events[].potential_search_regions[].uncertainty` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].potential_search_regions[].region_id` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].potential_search_regions[].evidence_state_ids` | 必有 | `array` | minItems=1; uniqueItems=True |
| `search_event_states.events[].potential_search_regions[].evidence_state_ids[]` | 必有 | `string` | minLength=1 |
| `search_event_states.events[].potential_search_regions[].annotation_status` | 必有 | `const "potential_search_region_inferred_not_confirmed_intention"` | — |
| `search_event_states.frame_numbering` | 必有 | `object` | additionalProperties=False |
| `search_event_states.frame_numbering.fps_source` | 必有 | `string` | minLength=1 |
| `search_event_states.frame_numbering.namespace_note` | 必有 | `string` | minLength=1 |
| `search_event_states.frame_numbering.original_video_fps` | 必有 | `integer` | — |
| `search_event_states.frame_numbering.original_video_frame_index_base` | 必有 | `integer` | — |
| `search_event_states.frame_numbering.source` | 必有 | `string` | minLength=1 |
| `search_event_states.frame_numbering.time_formula` | 必有 | `string` | minLength=1 |
| `search_event_states.image_count` | 必有 | `integer` | — |
| `search_event_states.image_directory` | 必有 | `string` | minLength=1 |
| `search_event_states.provenance` | 必有 | `object` | additionalProperties=False |
| `search_event_states.provenance.coverage` | 必有 | `string` | minLength=1 |
| `search_event_states.provenance.description_source` | 必有 | `string` | minLength=1 |
| `search_event_states.provenance.knowledge_scope` | 必有 | `string` | minLength=1 |
| `search_event_states.schema` | 必有 | `const "ego.search_event_triplet_states.v1"` | — |

## search_units

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `search_units` | 必有 | `object` | additionalProperties=False |
| `search_units.catalog_status` | 必有 | `string` | minLength=1 |
| `search_units.coverage_note` | 必有 | `string` | minLength=1 |
| `search_units.id_policy` | 必有 | `string` | minLength=1 |
| `search_units.list` | 必有 | `array` | — |
| `search_units.list[]` | 必有 | `object` | additionalProperties=False |
| `search_units.list[].boundary` | 必有 | `string` | minLength=1 |
| `search_units.list[].entity_link_basis` | 必有 | `string` | minLength=1 |
| `search_units.list[].is_container` | 必有 | `boolean/null` | — |
| `search_units.list[].label` | 必有 | `string` | minLength=1 |
| `search_units.list[].source_entity_ids` | 必有 | `array` | uniqueItems=True |
| `search_units.list[].source_entity_ids[]` | 必有 | `string` | minLength=1 |
| `search_units.list[].type` | 必有 | `enum ["inspection_area", "container_interior", "covered_surface", "exposed_surface"]` | — |
| `search_units.list[].unit_id` | 必有 | `string` | minLength=1 |
| `search_units.list[].zone_id` | 必有 | `string` | minLength=1 |
| `search_units.list[].visit_count` | 必有 | `integer/null` | minimum=1 |
| `search_units.list[].visit_count_source` | 必有 | `string/null` | — |
| `search_units.list[].room` | 必有 | `string/null` | — |
| `search_units.list[].parent_id` | 必有 | `string/null` | — |
| `search_units.list[].inspect_cost_s` | 必有 | `number/null` | — |
| `search_units.list[].inspect_cost_trials_s` | 必有 | `null 或 array` | — |
| `search_units.list[].inspect_cost_trials_s[]` | 必有 | `number` | minimum=0 |
| `search_units.list[].volume_l` | 必有 | `number/null` | — |
| `search_units.list[].surface_vs_enclosed` | 必有 | `string/null` | — |
| `search_units.scope` | 必有 | `string` | minLength=1 |

## hidden

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `hidden` | 必有 | `object` | additionalProperties=False |
| `hidden.target_location` | 必有 | `null 或 object` | — |
| `hidden.target_location.unit_id` | 必有 | `string` | minLength=1 |
| `hidden.target_location.description` | 必有 | `string` | minLength=1 |
| `hidden.target_location.source_event_id` | 必有 | `string` | minLength=1 |
| `hidden.target_location.location_granularity` | 必有 | `const "search_region"` | — |
| `hidden.target_location.description_source` | 必有 | `const "qwen_search_region_unreviewed"` | — |
| `hidden.target_instance_id` | 必有 | `string/null` | — |
| `hidden.source` | 必有 | `string/null` | — |
| `hidden.rule` | 必有 | `string` | minLength=1 |
| `hidden.original_unit_id` | 必有 | `string/null` | — |
| `hidden.relocation_source` | 必有 | `string/null` | — |
| `hidden.relocation_evidence_state` | 必有 | `string/null` | — |
| `hidden.evidence_coverage` | 必有 | `number/null` | — |
| `hidden.evidence_confidence` | 必有 | `number/null` | — |
| `hidden.intervention_time_h` | 必有 | `number/null` | — |

## history

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `history` | 必有 | `null 或 array` | — |

## priors

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `priors` | 必有 | `object` | additionalProperties=False |
| `priors.habit_prior` | 必有 | `object` | additionalProperties=False |
| `priors.habit_prior.counts` | 必有 | `object` | additionalProperties=动态键值类型 number/null |
| `priors.habit_prior.counts.<动态键>` | 可选 | `number/null` | — |
| `priors.cohabitant_counts` | 必有 | `object` | additionalProperties=False |
| `priors.cohabitant_counts.counts` | 必有 | `object` | additionalProperties=动态键值类型 number/null |
| `priors.cohabitant_counts.counts.<动态键>` | 可选 | `number/null` | — |
| `priors.semantic_scores` | 必有 | `object` | additionalProperties=False |
| `priors.semantic_scores.source` | 必有 | `string/null` | — |
| `priors.semantic_scores.scores` | 必有 | `object` | additionalProperties=动态键值类型 number/null |
| `priors.semantic_scores.scores.<动态键>` | 可选 | `number/null` | — |

## $defs.node

| 路径 | 必选性 | 类型/枚举 | 约束 |
|---|---|---|---|
| `$defs.node` | 可选 | `object` | additionalProperties=False; 含条件约束，见机器schema（非region节点须有object_ref） |
| `$defs.node.additional_parent_relations` | 可选 | `array` | — |
| `$defs.node.additional_parent_relations[]` | 必有 | `object` | additionalProperties=False |
| `$defs.node.additional_parent_relations[].id` | 必有 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].predicate` | 必有 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].kind` | 必有 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].status` | 必有 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].description` | 可选 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].observation_summary` | 可选 | `object` | additionalProperties=False |
| `$defs.node.additional_parent_relations[].observation_summary.count` | 必有 | `integer` | minimum=1 |
| `$defs.node.additional_parent_relations[].observation_summary.first_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.additional_parent_relations[].observation_summary.last_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.additional_parent_relations[].observation_summary.first_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].observation_summary.last_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].evidence_frames` | 可选 | `array` | uniqueItems=True |
| `$defs.node.additional_parent_relations[].evidence_frames[]` | 必有 | `string` | minLength=1 |
| `$defs.node.additional_parent_relations[].position_in_region` | 可选 | `null` | — |
| `$defs.node.associations` | 可选 | `array` | — |
| `$defs.node.associations[]` | 必有 | `object` | additionalProperties=False |
| `$defs.node.associations[].id` | 必有 | `string` | minLength=1 |
| `$defs.node.associations[].predicate` | 必有 | `string` | minLength=1 |
| `$defs.node.associations[].kind` | 必有 | `string` | minLength=1 |
| `$defs.node.associations[].status` | 必有 | `string` | minLength=1 |
| `$defs.node.associations[].description` | 可选 | `string` | minLength=1 |
| `$defs.node.associations[].observation_summary` | 可选 | `object` | additionalProperties=False |
| `$defs.node.associations[].observation_summary.count` | 必有 | `integer` | minimum=1 |
| `$defs.node.associations[].observation_summary.first_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.associations[].observation_summary.last_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.associations[].observation_summary.first_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.associations[].observation_summary.last_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.associations[].evidence_frames` | 可选 | `array` | uniqueItems=True |
| `$defs.node.associations[].evidence_frames[]` | 必有 | `string` | minLength=1 |
| `$defs.node.associations[].position_in_region` | 可选 | `null` | — |
| `$defs.node.associations[].target_id` | 必有 | `string` | minLength=1 |
| `$defs.node.basis` | 可选 | `string` | minLength=1 |
| `$defs.node.cross_level_relations` | 可选 | `array` | — |
| `$defs.node.cross_level_relations[]` | 必有 | `object` | additionalProperties=False |
| `$defs.node.cross_level_relations[].id` | 必有 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].predicate` | 必有 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].kind` | 必有 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].status` | 必有 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].description` | 可选 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].observation_summary` | 可选 | `object` | additionalProperties=False |
| `$defs.node.cross_level_relations[].observation_summary.count` | 必有 | `integer` | minimum=1 |
| `$defs.node.cross_level_relations[].observation_summary.first_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.cross_level_relations[].observation_summary.last_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.cross_level_relations[].observation_summary.first_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].observation_summary.last_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].evidence_frames` | 可选 | `array` | uniqueItems=True |
| `$defs.node.cross_level_relations[].evidence_frames[]` | 必有 | `string` | minLength=1 |
| `$defs.node.cross_level_relations[].position_in_region` | 可选 | `null` | — |
| `$defs.node.cross_level_relations[].target_id` | 必有 | `string` | minLength=1 |
| `$defs.node.entity_name` | 可选 | `string` | minLength=1 |
| `$defs.node.id` | 必有 | `string` | minLength=1 |
| `$defs.node.name` | 必有 | `string` | minLength=1 |
| `$defs.node.object_ref` | 可选 | `string` | minLength=1 |
| `$defs.node.parent_room_id` | 可选 | `null` | — |
| `$defs.node.placement` | 可选 | `object` | additionalProperties=False |
| `$defs.node.placement.id` | 可选 | `string` | minLength=1 |
| `$defs.node.placement.predicate` | 必有 | `string` | minLength=1 |
| `$defs.node.placement.kind` | 可选 | `string` | minLength=1 |
| `$defs.node.placement.status` | 必有 | `string` | minLength=1 |
| `$defs.node.placement.description` | 可选 | `string` | minLength=1 |
| `$defs.node.placement.observation_summary` | 可选 | `object` | additionalProperties=False |
| `$defs.node.placement.observation_summary.count` | 必有 | `integer` | minimum=1 |
| `$defs.node.placement.observation_summary.first_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.placement.observation_summary.last_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.placement.observation_summary.first_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.placement.observation_summary.last_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.placement.evidence_frames` | 可选 | `array` | uniqueItems=True |
| `$defs.node.placement.evidence_frames[]` | 必有 | `string` | minLength=1 |
| `$defs.node.placement.position_in_region` | 可选 | `null` | — |
| `$defs.node.placement_status` | 可选 | `string` | minLength=1 |
| `$defs.node.position_in_room` | 可选 | `null` | — |
| `$defs.node.relations` | 可选 | `array` | — |
| `$defs.node.relations[]` | 必有 | `object` | additionalProperties=False |
| `$defs.node.relations[].id` | 必有 | `string` | minLength=1 |
| `$defs.node.relations[].predicate` | 必有 | `string` | minLength=1 |
| `$defs.node.relations[].kind` | 必有 | `string` | minLength=1 |
| `$defs.node.relations[].status` | 必有 | `string` | minLength=1 |
| `$defs.node.relations[].description` | 可选 | `string` | minLength=1 |
| `$defs.node.relations[].observation_summary` | 可选 | `object` | additionalProperties=False |
| `$defs.node.relations[].observation_summary.count` | 必有 | `integer` | minimum=1 |
| `$defs.node.relations[].observation_summary.first_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.relations[].observation_summary.last_seen_s` | 必有 | `number` | minimum=0 |
| `$defs.node.relations[].observation_summary.first_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.relations[].observation_summary.last_frame` | 必有 | `string` | minLength=1 |
| `$defs.node.relations[].evidence_frames` | 可选 | `array` | uniqueItems=True |
| `$defs.node.relations[].evidence_frames[]` | 必有 | `string` | minLength=1 |
| `$defs.node.relations[].position_in_region` | 可选 | `null` | — |
| `$defs.node.relations[].target_id` | 必有 | `string` | minLength=1 |
| `$defs.node.type` | 必有 | `string` | minLength=1 |
| `$defs.node.children` | 必有 | `array` | — |
| `$defs.node.children[]` | 必有 | `#/$defs/node` | — |
| `$defs.node.parent_relations` | 必有 | `null 或 array` | — |
| `$defs.node.parent_relations[]` | 必有 | `object` | additionalProperties=False |
| `$defs.node.parent_relations[].relation_id` | 必有 | `string` | minLength=1 |
| `$defs.node.parent_relations[].parent_id` | 必有 | `string/null` | minLength=1 |
| `$defs.node.parent_relations[].predicate` | 必有 | `enum ["on", "inside", "mounted_on", "hanging_on", "within_region", "unknown"]` | — |
| `$defs.node.parent_relations[].relative_position` | 必有 | `string/null` | minLength=1 |
| `$defs.node.parent_relations[].valid_from_frame` | 必有 | `integer` | minimum=0 |
| `$defs.node.parent_relations[].valid_to_frame_exclusive` | 必有 | `integer/null` | minimum=0 |
| `$defs.node.parent_relations[].started_by_event_id` | 必有 | `string/null` | minLength=1 |
| `$defs.node.parent_relations[].ended_by_event_id` | 必有 | `string/null` | minLength=1 |
| `$defs.node.parent_relations[].boundary_status` | 必有 | `enum ["scene_reference_observation_unreviewed", "first_observed_in_event_unreviewed"]` | — |
| `$defs.node.parent_relations[].start_evidence_state_id` | 可选 | `string` | — |
| `$defs.node.parent_relations[].end_evidence_state_id` | 可选 | `string` | — |

