"""Conservative provisional identity association, with an explicit decision trail."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy


def iou(a, b):
    width = max(0, min(a[2], b[2])-max(a[0], b[0]))
    height = max(0, min(a[3], b[3])-max(a[1], b[1]))
    area = width*height
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union else 0


class Registry:
    def __init__(self, episode_id, threshold):
        self.episode_id = episode_id
        self.threshold = threshold
        self.instances = {}
        self.previous_image_ref = None

    def candidates(self, maximum):
        recent = sorted(self.instances.values(), key=lambda o: (o["last_seen_order"], o["id"]), reverse=True)
        return [{k: deepcopy(o[k]) for k in ["id", "name", "category", "appearance", "last_image_ref",
            "last_bbox", "last_state_description", "identity_status"]} for o in recent[:maximum]]

    def ingest(self, doc, frame, order, geometry):
        proposals = Counter(o["identity"]["candidate_ids"][0] for o in doc["objects"]
            if o["identity"]["decision"] == "match_candidate")
        decisions, mapping = [], {}
        for obj in doc["objects"]:
            ident = obj["identity"]
            candidates = ident["candidate_ids"]
            matched, reason = None, "no_unambiguous_proposal"
            if ident["decision"] == "match_candidate":
                old = self.instances[candidates[0]]
                if proposals[old["id"]] > 1:
                    reason = "co_visible_objects_proposed_same_instance"
                elif old["last_image_ref"] != self.previous_image_ref:
                    reason = "not_observed_in_immediately_previous_frame_requires_review"
                elif old["category"] != obj["category"]:
                    reason = "category_changed_requires_review"
                elif old["appearance"]["color"] != obj["appearance"]["color"]:
                    reason = "color_description_changed_requires_review"
                elif old["appearance"]["color"] is None:
                    reason = "missing_color_requires_review"
                elif iou(old["last_bbox"], obj["bbox"]) < self.threshold:
                    reason = "image_overlap_below_conservative_gate"
                elif obj["epistemic_status"] != "observed":
                    reason = "current_object_not_directly_observed"
                else:
                    matched, reason = old["id"], "model_proposal_and_adjacent_frame_continuity_gate"
            if matched is None:
                matched = f"{self.episode_id}:instance:{len(self.instances)+1:05d}"
                self.instances[matched] = {"id": matched, "name": obj["name"], "category": obj["category"],
                    "appearance": deepcopy(obj["appearance"]), "identity_status": "provisional_not_human_confirmed",
                    "first_image_ref": doc["image_ref"], "first_video_frame": frame["original_video_frame_index"],
                    "first_seen_order": order, "observation_count": 0}
            entry = self.instances[matched]
            entry.update(last_image_ref=doc["image_ref"], last_bbox=obj["bbox"],
                last_state_description=obj["state_description"], last_seen_order=order,
                last_video_frame=frame["original_video_frame_index"], latest_visibility=obj["visibility"],
                last_geometry=geometry[obj["local_id"]])
            entry["observation_count"] += 1
            mapping[obj["local_id"]] = matched
            decisions.append({"local_id": obj["local_id"], "instance_id": matched, "reason": reason,
                "proposed_candidates": candidates, "model_basis": ident["basis"],
                "identity_status": "provisional_not_human_confirmed"})
        self.previous_image_ref = doc["image_ref"]
        # Unseen entries are retained as last known observations, never marked moved/absent.
        return {"schema": "astra_memory.association.v1", "image_ref": doc["image_ref"],
            "local_to_instance": mapping, "decisions": decisions}

    def snapshot(self):
        return {"schema": "astra_memory.provisional_registry.v1", "episode_id": self.episode_id,
            "instances": list(self.instances.values()), "last_image_ref": self.previous_image_ref,
            "note": "相邻帧关联仍为候选身份；未出现不等于已移动；疑难复核由下游S3保存独立审阅记录。"}
