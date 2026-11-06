"""Evidence-only temporal memory. General knowledge never mutates this state."""
from __future__ import annotations

from copy import deepcopy

from .io import digest, require

PLACEMENT = {"on", "inside", "mounted_on", "part_of"}
LOCATION = PLACEMENT | {"held_by"}
INVALIDATES = {"moved", "lifted", "placed"}
INVERSE = {"right_of_image": "left_of_image", "below_image": "above_image", "behind": "in_front_of"}
SYMMETRIC = {"adjacent_in_image", "overlaps_in_image", "near", "touching"}
WORD = {"on": "位于其上", "inside": "位于其内部", "mounted_on": "安装或悬挂于其上",
        "part_of": "是其组成部分", "left_of_image": "位于其图像左侧", "above_image": "位于其图像上方",
        "in_front_of": "位于其前方", "occluding": "遮挡它", "adjacent_in_image": "与其在图像中相邻",
        "overlaps_in_image": "与其在图像中重叠", "near": "靠近它", "touching": "与其接触",
        "held_by": "被其握持", "uncertain": "与其交互不明确"}


def canonical_relation(subject, predicate, target):
    if predicate in INVERSE:
        subject, target, predicate = target, subject, INVERSE[predicate]
    if predicate in SYMMETRIC and subject > target:
        subject, target = target, subject
    return subject, predicate, target


def cycle_edges(edges):
    """Return all edges within nontrivial strongly connected components."""
    graph = {}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set())
    counter, stack, indices, low, active, components = [0], [], {}, {}, set(), []

    def visit(node):
        indices[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        active.add(node)
        for target in sorted(graph[node]):
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], indices[target])
        if low[node] == indices[node]:
            component = set()
            while True:
                target = stack.pop()
                active.remove(target)
                component.add(target)
                if target == node:
                    break
            components.append(component)

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return {(a, b) for component in components if len(component) > 1
            for a, b in edges if a in component and b in component}


class TemporalMemory:
    def __init__(self, episode_id):
        self.episode_id = episode_id
        self.objects, self.relations = {}, {}
        self.evidence, self.conflicts, self.frames = [], [], []
        self.previous_order = -1

    def describe(self, key):
        source, predicate, target = key
        return f'{self.objects[source]["name"]}{WORD[predicate]}（参照：{self.objects[target]["name"]}）。'

    def ingest(self, record):
        doc, association = record["observation"], record["association"]
        image_ref, time_s = record["image_ref"], record["original_video_seconds"]
        order = len(self.frames)
        require(not self.frames or time_s >= self.frames[-1]["time_s"], "non-increasing S2 record time")
        require(image_ref not in {f["image_ref"] for f in self.frames}, "duplicate observation image_ref")
        self.frames.append({"image_ref": image_ref, "time_s": time_s,
            "original_video_frame_index": record["original_video_frame_index"]})
        mapping = association["local_to_instance"]
        new_ids = []
        for observed in doc["objects"]:
            oid = mapping[observed["local_id"]]
            if oid not in self.objects:
                new_ids.append(oid)
                self.objects[oid] = {"id": oid, "name": observed["name"], "category": observed["category"],
                    "identity_status": "provisional_not_human_confirmed", "first_seen": self.frames[-1],
                    "observation_count": 0, "appearance_values": {}, "appearance_latest": {},
                    "appearance_evidence": {}, "location_observations": [], "last_location": None,
                    "last_placement": [], "placement_invalidated_at": None, "state_changes": [],
                    "identity_candidates": []}
            obj = self.objects[oid]
            obj.update(last_seen=self.frames[-1], latest_state_description=observed["state_description"],
                latest_visibility=observed["visibility"], latest_epistemic_status=observed["epistemic_status"],
                last_bbox=observed["bbox"])
            obj["observation_count"] += 1
            for attr, value in observed["appearance"].items():
                values = value if isinstance(value, list) else ([value] if value is not None else [])
                for text in values:
                    if text not in obj["appearance_values"].setdefault(attr, []):
                        obj["appearance_values"][attr].append(text)
                    evidence_key = digest([attr, text])[:16]
                    obj["appearance_evidence"].setdefault(evidence_key, {"attribute": attr, "value": text, "image_refs": []})["image_refs"].append(image_ref)
                if values:
                    obj["appearance_latest"][attr] = value
            for candidate in observed["identity"]["candidate_ids"]:
                if candidate != oid and candidate not in obj["identity_candidates"]:
                    obj["identity_candidates"].append(candidate)

        raw = []
        for i, relation in enumerate(doc["relations"]):
            key = canonical_relation(mapping[relation["subject_local_id"]], relation["predicate"],
                                     mapping[relation["object_local_id"]])
            raw.append((key, relation, f"relations/{i}"))
        for i, interaction in enumerate(doc["interactions"]):
            actor, obj = mapping[interaction["actor_local_id"]], mapping[interaction["object_local_id"]]
            key = (obj, "held_by", actor) if interaction["action"] == "holding" else canonical_relation(actor, interaction["action"], obj)
            raw.append((key, interaction, f"interactions/{i}"))

        rejected = set()
        # Conflicts are evaluated within the current frame, never across camera views.
        for predicates in [PLACEMENT, {"left_of_image"}, {"above_image"}, {"in_front_of"}]:
            directed = {(k[0], k[2]) for k, r, _ in raw if k[1] in predicates and r["epistemic_status"] == "observed"}
            cyclic = cycle_edges(directed)
            rejected.update(k for k, r, _ in raw if k[1] in predicates and (k[0], k[2]) in cyclic)
        locations = {}
        for key, proposal, pointer in raw:
            rid = "relation:"+digest(key)[:20]
            accepted = key not in rejected
            status = proposal["epistemic_status"]
            self.evidence.append({"relation_id": rid, "image_ref": image_ref, "time_s": time_s,
                "subject_id": key[0], "predicate": key[1], "target_id": key[2], "source_pointer": pointer,
                "source_description": proposal["description"], "epistemic_status": status,
                "support_image_refs": proposal["support_image_refs"], "accepted": accepted})
            if not accepted:
                self.conflicts.append({"kind": "current_frame_cycle_or_direction_conflict", "image_ref": image_ref,
                    "relation_id": rid, "subject_id": key[0], "predicate": key[1], "target_id": key[2],
                    "resolution": "excluded_from_layout_and_location; raw_evidence_retained"})
                continue
            relation = self.relations.setdefault(rid, {"id": rid, "subject_id": key[0], "predicate": key[1],
                "target_id": key[2], "description": self.describe(key), "observations": [],
                "observation_summary": {}, "epistemic_statuses": []})
            if status not in relation["epistemic_statuses"]:
                relation["epistemic_statuses"].append(status)
            existing = next((r for r in relation["observations"] if r["image_ref"] == image_ref), None)
            if existing is None:
                relation["observations"].append({"image_ref": image_ref, "time_s": time_s, "epistemic_status": status})
            elif status == "observed":
                existing["epistemic_status"] = "observed"
            obs = relation["observations"]
            relation["observation_summary"] = {"count": len(obs), "first_seen_s": obs[0]["time_s"],
                "last_seen_s": obs[-1]["time_s"], "first_frame": obs[0]["image_ref"], "last_frame": obs[-1]["image_ref"]}
            if status == "observed" and key[1] in LOCATION:
                locations.setdefault(key[0], {})[rid] = key

        invalidated = set()
        for change in doc["changes"]:
            oid = mapping[change["object_local_id"]]
            # Unknown change hypotheses never erase a known location.
            if change["epistemic_status"] == "unknown":
                continue
            self.objects[oid]["state_changes"].append({"frame_id": image_ref, "time_s": time_s,
                "description": change["description"], "kind": change["kind"],
                "before_image_ref": change["before_image_ref"], "epistemic_status": change["epistemic_status"]})
            if change["kind"] in INVALIDATES and change["epistemic_status"] == "observed":
                invalidated.add(oid)
                obj = self.objects[oid]
                obj["last_placement"] = []
                obj["placement_invalidated_at"] = image_ref
                obj["last_location"] = None
                if obj["location_observations"]:
                    obj["location_observations"][-1]["invalidated_at"] = self.frames[-1]
        for oid, current in locations.items():
            obj = self.objects[oid]
            signature = sorted(current)
            history = obj["location_observations"]
            sample = {"image_ref": image_ref, "time_s": time_s, "relation_ids": signature}
            if not history or history[-1]["relation_ids"] != signature or "invalidated_at" in history[-1]:
                history.append({"relation_ids": signature, "first_seen": self.frames[-1],
                    "last_seen": self.frames[-1], "support_image_refs": [image_ref], "count": 1})
            else:
                history[-1]["last_seen"] = self.frames[-1]
                history[-1]["support_image_refs"].append(image_ref)
                history[-1]["count"] += 1
            obj["last_location"] = sample
            placement = [rid for rid, key in current.items() if key[1] in PLACEMENT]
            if placement:
                obj["last_placement"] = placement
                obj["placement_invalidated_at"] = None
        self.previous_order = order
        return new_ids

    def topology_parents(self):
        parents, edge_by_child = {}, {}
        candidates = []
        for oid, obj in self.objects.items():
            for rid in obj["last_placement"]:
                relation = self.relations[rid]
                direct = [o for o in relation["observations"] if o["epistemic_status"] == "observed"]
                candidates.append((-direct[-1]["time_s"], {"inside": 0, "mounted_on": 1, "on": 2, "part_of": 3}[relation["predicate"]], rid, oid, relation["target_id"]))
        for _, _, rid, oid, target in sorted(candidates):
            if oid in parents:
                continue
            cursor, visited = target, {oid}
            while cursor in parents and cursor not in visited:
                visited.add(cursor)
                cursor = parents[cursor]
            if cursor in visited:
                self._conflict_once({"kind": "historical_layout_cycle", "relation_id": rid,
                    "resolution": "retained_as_cross_relation_not_parent"})
                continue
            parents[oid], edge_by_child[oid] = target, rid
        return parents, edge_by_child

    def _conflict_once(self, item):
        if item not in self.conflicts:
            self.conflicts.append(item)

    def object_summary(self):
        return {"schema": "astra_memory.object_history.v1", "episode_id": self.episode_id,
            "objects": deepcopy(list(self.objects.values())),
            "semantics": "observed_ranges_not_continuous_certainty; unseen_is_not_moved; identities_provisional"}
