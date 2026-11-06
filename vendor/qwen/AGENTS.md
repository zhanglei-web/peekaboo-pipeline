# Annotation output requirements

The user explicitly requires all future annotation deliverables from this project to use the nested temporal tree format (confirmed 2026-11-04).

- Keep `scene_memory.spatial_structure.nodes` as a list of nested `children`, not a dictionary in final episode JSON.
- Keep stable object IDs and profiles. A repeated ID in historical branches denotes the same object.
- Direct-parent `placement` relations and node occurrences carry original-video-frame half-open intervals. Parent-side trees are derived on CPU. Never fabricate an initial frame-0 relation from a mixed-time reference scene.
- Track all manipulated catalog objects, not only the search target. Container movement alone does not change the direct parents of its contents.
- New event/relation annotations are English. Preserve existing metadata, research labels and nulls unless the user authorizes changes. Do not apply one dataset's research protocol to another.
- Use actual event image evidence to fill missing temporal changes. A schema-only conversion is not completed temporal annotation. Unknowns stay explicit. Structural validation is not semantic acceptance.
- `scripts/run_scene_event_batch.py` defaults to the temporal-tree pipeline. `--legacy-static` is only for explicitly requested historical reproduction, never a silent fallback. Preserve old output directories and raw evidence.
- Existing static datasets can be processed with `scripts/batch_temporal_trees.py`; validate all outputs with `scripts/audit_temporal_tree_batch.py`.
- Do not interrupt or reconfigure unrelated workers or model requests.
