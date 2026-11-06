"""Persistent semantic search places with explicit anchors and temporal revisions."""
from __future__ import annotations

from copy import deepcopy

from .io import digest


class UnitRegistry:
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.units, self.conflicts = {}, []

    @staticmethod
    def key(proposal):
        # An entire container interior/surface must not fragment by view or wording.
        region = "entire" if proposal["type"] in {"container_interior", "exposed_surface"} else proposal["boundary_key"]
        return (proposal["anchor_entity_id"], proposal["anchor_kind"], proposal["type"], region)

    def candidates(self, visible_entities):
        return {uid: deepcopy(unit) for uid, unit in self.units.items()
                if unit["anchor_entity_id"] in visible_entities}

    def ingest(self, document, frame, request_ref):
        for proposal in document["proposals"]:
            key = self.key(proposal)
            proposed = proposal["existing_unit_candidates"]
            exact = [uid for uid, unit in self.units.items() if tuple(unit["semantic_key"]) == key]
            if len(proposed) > 1 or (proposed and any(tuple(self.units[uid]["semantic_key"]) != key for uid in proposed)):
                self.conflicts.append({"kind": "ambiguous_or_changed_unit_anchor", "image_ref": frame["image_ref"],
                    "available_at_s": frame["time_s"], "proposal": deepcopy(proposal), "resolution": "unresolved_no_merge_or_reanchor"})
                continue
            uid = proposed[0] if proposed else (exact[0] if exact else None)
            if uid is None:
                uid = self.episode_id+":unit:"+digest(key)[:16]
                self.units[uid] = {"unit_id": uid, "semantic_key": list(key),
                    "anchor_entity_id": key[0], "anchor_kind": key[1], "boundary_key": key[3],
                    "identity_status": "anchor_provisional_not_cross_instance_merged", "revisions": []}
            unit = self.units[uid]
            payload = {k: deepcopy(proposal[k]) for k in ["label", "type", "is_container", "boundary",
                "anchor_rule", "source_entity_candidates", "container_visual_basis", "uncertainties"]}
            payload["source_entity_candidates"] = sorted(payload["source_entity_candidates"])
            revisions = unit["revisions"]
            if revisions and revisions[-1]["definition"] == payload:
                revision = revisions[-1]
                for image_ref in proposal["support_image_refs"]:
                    if image_ref not in revision["support_image_refs"]:
                        revision["support_image_refs"].append(image_ref)
                revision["last_supported_at"] = frame
                revision["support_history"].append({"available_at": frame, "image_refs": list(proposal["support_image_refs"])})
            else:
                revisions.append({"version": len(revisions)+1, "available_at": frame,
                    "last_supported_at": frame, "definition": payload,
                    "support_image_refs": list(proposal["support_image_refs"]), "request_ref": request_ref,
                    "support_history": [{"available_at": frame, "image_refs": list(proposal["support_image_refs"])}]})

    def snapshot(self, as_of_s=None):
        result = []
        for unit in self.units.values():
            revisions = [r for r in unit["revisions"] if as_of_s is None or r["available_at"]["time_s"] <= as_of_s]
            if not revisions:
                continue
            item = {k: deepcopy(v) for k, v in unit.items() if k != "revisions"}
            item["revisions"] = deepcopy(revisions)
            if as_of_s is not None:
                for revision in item["revisions"]:
                    history = [h for h in revision["support_history"] if h["available_at"]["time_s"] <= as_of_s]
                    revision["support_history"] = history
                    revision["support_image_refs"] = list(dict.fromkeys(ref for h in history for ref in h["image_refs"]))
                    revision["last_supported_at"] = history[-1]["available_at"]
            result.append(item)
        return {"schema": "astra_memory.unit_registry.v1", "episode_id": self.episode_id,
                "units": result, "unresolved_proposals": deepcopy([c for c in self.conflicts
                    if as_of_s is None or c["available_at_s"] <= as_of_s])}

    def export(self, as_of_s=None):
        result = []
        for unit in self.snapshot(as_of_s)["units"]:
            definition = unit["revisions"][-1]["definition"]
            result.append({"unit_id": unit["unit_id"], "zone_id": self.episode_id+":zone:tool_workspace",
                "label": definition["label"], "type": definition["type"], "is_container": definition["is_container"],
                "source_entity_ids": definition["source_entity_candidates"], "boundary": definition["boundary"],
                "entity_link_basis": f'validated_Astra_D; anchor={unit["anchor_entity_id"]}; version={unit["revisions"][-1]["version"]}; provisional_identity'})
        return {"scope": "已观测局部工作区", "catalog_status": "evidence_based_proposals_with_provisional_anchors",
            "coverage_note": "仅覆盖输入观测支持的区域；未提议不表示不存在；未确定的地点链接保留在冲突账本。",
            "id_policy": "由锚定实体、锚定类型及语义边界确定ID；重访与容器平移不变；边界细化保留版本，不跨候选身份强合并。",
            "list": result}
