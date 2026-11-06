"""Deterministic projection into the frozen final scene_memory/search_units fields."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

from .io import load, require
from .s2_state import LOCATION, PLACEMENT


RULES = {
    "identity": "保留S1候选实例ID及未确认关联，不合并疑似重复实体；机器连续性关联不等于人工确认。",
    "appearance": "appearance保存最近非空属性；详细历史取值与证据在独立对象账本，不将颜色变化直接解释成物体变化。",
    "spatial_node_id": "引用同ID的拓扑节点；树表示最新已知支承/容纳的参考归属，不是末帧全景快照。",
    "commonsense": "类别知识C按类别缓存，与实例实际位置分离；通常位置、用途及功能伙伴至多合并为三条。",
    "affordances": "能力是有条件的类别潜力与已见形状的组合，未认证实例内腔、可达性、尺寸或装有目标。",
    "observations": "最后位置采用最近有直接关系证据的采样；未再看见保留最后已知位置；明确移动使旧位置失效。",
    "history": "仅将A明确报告且非unknown的变化写入state_changes；空数组不证明物体静止；完整位置历史另存。",
    "provenance": "观测来自经契约校验的Astra A输出，常识来自独立C调用；证据/请求/帧号可追溯，未认证视觉准确率。",
    "search_event_states": "S2不生成或改写事件状态；事件比较与人工确认由S3负责。",
}


def knowledge_text(entry):
    doc = entry["document"]
    result = []
    for key, label in [("usual_locations", "通常放置或安装位置（类别常识）"),
                       ("functions", "通常用途（类别常识）"), ("functional_partners", "功能关联（类别常识）")]:
        if doc[key]:
            result.append(label+"："+"；".join(doc[key]))
    return result or ["该类别的通常位置、用途及功能关联尚不明确；不据此推断实例状态。"]


def project_scene(memory, knowledge):
    parents, parent_relations = memory.topology_parents()
    root_id = memory.episode_id+":region:tool_workspace"
    root = {"id": root_id, "type": "region", "name": "已观测局部工作区", "children": [],
            "parent_room_id": None, "position_in_room": None, "basis": "输入图像中可见实体与经校验的空间关系"}
    nodes = {oid: {"id": oid, "type": "object", "name": obj["name"], "entity_name": obj["name"],
                   "object_ref": oid, "children": []} for oid, obj in memory.objects.items()}
    anchors = set(parents.values())-parents.keys()
    layout_parents = {oid: parents.get(oid, root_id if oid in anchors else None) for oid in nodes}
    for oid, node in nodes.items():
        if oid in parents:
            nodes[parents[oid]]["children"].append(node)
        elif oid in anchors:
            node["placement"] = {"predicate": "part_of", "status": "semantic_member_of_observed_workspace_not_physical_placement"}
            root["children"].append(node)
        else:
            node["placement_status"] = "无当前有效的直接支承或容纳关系；最后位置与变化见对象档案及账本。"
    for rid, relation in sorted(memory.relations.items()):
        subject, target = relation["subject_id"], relation["target_id"]
        edge = {"id": rid, "predicate": relation["predicate"],
            "kind": "interaction" if relation["predicate"] in {"held_by", "near", "touching", "uncertain"} else "spatial_observation",
            "description": relation["description"],
            "status": "machine_observations_not_continuous_or_human_confirmed:"+",".join(relation["epistemic_statuses"]),
            "observation_summary": deepcopy(relation["observation_summary"])}
        if parent_relations.get(subject) == rid:
            nodes[subject]["placement"] = edge
            continue
        edge["target_id"] = target
        if edge["kind"] == "interaction":
            field = "associations"
        elif layout_parents[subject] is not None and layout_parents[subject] == layout_parents[target]:
            field = "relations"
        else:
            field = "cross_level_relations"
        nodes[subject].setdefault(field, []).append(edge)
    candidates = []
    emitted = set()
    for oid, obj in memory.objects.items():
        for candidate in obj["identity_candidates"]:
            pair = tuple(sorted([oid, candidate]))
            if pair in emitted:
                continue
            emitted.add(pair)
            candidates.append({"id": "identity:"+"|".join(pair), "entity_ids": list(pair),
                "name": obj["name"]+"的候选同一性", "status": "unresolved_not_merged"})
    graph = {"schema": "ego.scene.nested_local_spatial_graph.v4", "root_id": root_id,
        "description": "children表达支承/容纳的参考层级，placement为相对父节点的位置，relations仅连接同级节点。",
        "reference_frame": "每条图像方位关系以其证据采样的相机图像为准；没有已标定的统一世界方向。",
        "coverage": {k: "unknown" for k in ["parent_room", "position_in_room", "metric_geometry", "navigation_connectivity"]},
        "semantics": {
            "hierarchy": "children无环；选取最新有效直接观测作参考父关系，其余证据保留为关系，不把握持当作固定支承。",
            "position": "placement描述相对父物体位置；没有证据不补充方位、距离或坐标。",
            "query": "按节点ID/关系ID在证据账本检索；按原视频帧号的历史查询只重放截至该帧的观测。",
            "time": "整棵树是最后已知参考归属，非所有对象同时可见的当前快照；可见性缺失不表示移动。",
            "identity": "使用S1候选实例ID，未核验物理唯一性；identity_candidates只记录候选，不自动合并。",
            "evidence": "首次/末次/次数仅代表离散观测范围，不证明中间连续成立；方向随证据图变化，冲突单独保留。",
            "missing": "未知房间、世界几何和导航通路保持unknown；unknown不等于不存在。",
            "cross_level": "跨层空间关系与交互分别保存；仅父归属要求无环，图像相邻等关系不用于导航或从属推理。"},
        "nodes": [root], "context_nodes": [],
        "unlocated_nodes": [nodes[oid] for oid in nodes if layout_parents[oid] is None],
        "identity_candidates": candidates}
    profiles = []
    for oid, obj in memory.objects.items():
        entry = knowledge[obj["category"]]
        latest = obj["appearance_latest"]
        appearance = {"attributes": {k: ([v] if isinstance(v, str) else v) for k, v in latest.items()
                                     if k != "shape" and v}}
        if latest.get("shape"):
            appearance.update(shape=latest["shape"], shape_status="qualitative_Astra_visual_observation_not_measured_dimensions")
        profile = {"id": oid, "identity": {"name": obj["name"], "category": obj["category"],
            "entity_type": "provisional_visual_instance", "identity_status": obj["identity_status"],
            "source_categories": [obj["category"]]}, "spatial_node_id": oid, "appearance": appearance,
            "commonsense": knowledge_text(entry),
            "historical_observation": {"state_changes": [{k: change[k] for k in ["frame_id", "time_s", "description"]}
                                                         for change in obj["state_changes"]]},
            "last_observation": {"location": None, "status": "没有有效的直接位置观测；不能以通常位置替代。"}}
        if obj["last_location"]:
            last = obj["last_location"]
            profile["last_observation"] = {"frame_id": last["image_ref"], "time_s": last["time_s"],
                "location": [{"relation": memory.relations[rid]["predicate"],
                    "reference_id": memory.relations[rid]["target_id"],
                    "reference_name": memory.objects[memory.relations[rid]["target_id"]]["name"],
                    "description": memory.relations[rid]["description"]} for rid in last["relation_ids"]]}
        elif obj["placement_invalidated_at"]:
            profile["last_observation"]["status"] = "明确位置变化使此前位置失效；尚无新位置证据。历史位置保留在独立账本。"
        if latest.get("shape"):
            potential = {kind: "仅属类别的条件性潜力，尚未验证该实例的内腔、开口、真实尺寸或可达性："+statement
                         for kind, statement in entry["document"]["conditional_affordances"].items() if statement}
            if potential:
                profile["affordances"] = potential
        profiles.append(profile)
    return {"spatial_structure": graph, "objects": profiles,
            "object_schema": {"version": "ego.scene.self_contained_object_profiles.v7", "rules": RULES}}


def validate_projection(scene, units, episode_id):
    schema = load(Path(__file__).parent/"resources/search_episode.schema.json")
    for name, value in [("scene_memory", scene), ("search_units", units)]:
        section = {"$schema": schema["$schema"], "$defs": schema["$defs"], **schema["properties"][name]}
        Draft202012Validator(section).validate(value)
    objects = {o["id"]: o for o in scene["objects"]}
    require(len(objects) == len(scene["objects"]), "duplicate profile ID")
    graph, nodes, parents, edges = scene["spatial_structure"], {}, {}, set()

    def visit(node, parent):
        require(node["id"] not in nodes, "duplicate/cyclic topology node")
        nodes[node["id"]], parents[node["id"]] = node, parent
        for child in node["children"]:
            visit(child, node["id"])

    for group in ["nodes", "context_nodes", "unlocated_nodes"]:
        for node in graph[group]:
            visit(node, None)
    require(graph["root_id"] in nodes and parents[graph["root_id"]] is None, "invalid topology root")
    for oid, obj in objects.items():
        require(oid in nodes and obj["spatial_node_id"] == oid and nodes[oid].get("object_ref") == oid, "broken object/node mapping")
        for loc in obj["last_observation"]["location"] or []:
            require(loc["reference_id"] in objects, "unknown location reference")
    for nid, node in nodes.items():
        if nid != graph["root_id"]:
            require(node["object_ref"] in objects, "unresolved node object_ref")
        for field in ["placement", "relations", "cross_level_relations", "associations", "additional_parent_relations"]:
            values = ([node[field]] if field in node else []) if field == "placement" else node.get(field, [])
            for edge in values:
                if "id" in edge:
                    require(edge["id"] not in edges, "relation projected twice")
                    edges.add(edge["id"])
                if "target_id" in edge:
                    require(edge["target_id"] in nodes, "unresolved graph target")
                    if field == "relations":
                        require(parents[nid] is not None and parents[nid] == parents[edge["target_id"]], "non-sibling relation")
    unit_ids = set()
    for unit in units["list"]:
        require(unit["unit_id"] not in unit_ids and unit["zone_id"] == episode_id+":zone:tool_workspace", "invalid unit ID/zone")
        unit_ids.add(unit["unit_id"])
        require(set(unit["source_entity_ids"]) <= objects.keys(), "unit has unknown entity")
    return {"schema": "astra_memory.s2_validation.v1", "status": "passed", "objects": len(objects),
            "topology_nodes": len(nodes), "relations": len(edges), "search_units": len(unit_ids),
            "visual_accuracy": "not_certified_by_structural_checks"}
