import base64
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from spatial_memory.catalog import initialize
from spatial_memory.geometry import score_job
from spatial_memory.memory import replay
from spatial_memory.selection import select, validate_selection


def catalog():
    return {"schema_version": "online.candidate_catalog.v0.1", "episode_id": "ep",
            "catalog_version": 0, "mode": "initial_semantic_proposals",
            "parents": [{"parent_id": "p", "name": "box"}],
            "regions": [{"candidate_id": "u", "parent_id": "p", "name": "footprint",
                         "region_type": "occluder_footprint", "identity_anchor": "original_footprint",
                         "boundary_description": "box original support area"}]}


def selected():
    return {"status": "ok", "pre": 1, "pnr": 2, "post": 3, "reason": "states",
            "event_id": "e1", "episode_id": "ep", "catalog_version": 0,
            "associations": [{"candidate_id": "u", "reason": "exposed footprint", "confidence": "high"}]}


def assessment(eid, q):
    return {"episode_id": "ep", "event_id": eid, "search_unit_id": "u",
            "score_type": "geometric_coverage", "scope": "original_support_footprint",
            "status": "valid" if q is not None else "uncomputable", "q": q}


def event(eid, q, end, available):
    sel = selected()
    sel["event_id"] = eid
    return {"episode_id": "ep", "event_id": eid, "catalog_version": 0,
            "event_end_time_s": end, "available_after_s": available,
            "selection": sel, "assessments": [assessment(eid, q)]}


class SpatialMemoryTests(unittest.TestCase):
    def test_stage_one_is_provisional_and_unknown(self):
        state = initialize(catalog())
        self.assertIsNone(state["units"]["u"]["Q"])
        self.assertFalse(state["units"]["u"]["confirmed"])
        self.assertEqual(state["object_summary"], {})

    def test_max_update_and_unknown_preserves_history(self):
        events = [event("e1", 0.6, 2, 3), event("e2", 0.3, 4, 5),
                  event("e3", None, 6, 7)]
        state = replay(catalog(), events)
        u = state["units"]["u"]
        self.assertEqual(u["Q"], 0.6)
        self.assertIsNone(u["last_q"])
        self.assertEqual(u["visit_count"], 3)
        self.assertEqual(state["object_summary"]["p"]["Q"], 0.6)
        self.assertEqual(replay(catalog(), events, cutoff_s=4)["units"]["u"]["Q"], 0.6)
        self.assertIsNone(replay(catalog(), events, cutoff_s=2)["units"]["u"]["Q"])

    def test_rejects_numeric_semantic_grade_and_duplicate_event(self):
        bad = event("e1", 0.5, 2, 3)
        bad["assessments"][0]["score_type"] = "ordinal_grade"
        with self.assertRaisesRegex(ValueError, "semantics"):
            replay(catalog(), [bad])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            replay(catalog(), [event("e1", 0.5, 2, 3), event("e1", 0.4, 4, 5)])

    def test_unsupported_scope_stays_unknown_in_memory(self):
        with TemporaryDirectory() as tmp:
            result = score_job({"search_unit_id": "u", "scope": "container_interior"},
                               selected(), Path(tmp))
            self.assertIsNone(result["q"])
            self.assertEqual(result["reason"], "unsupported_geometry_scope")
            record = event("e1", None, 2, 3)
            record["assessments"] = [result]
            state = replay(catalog(), [record])
            self.assertIsNone(state["units"]["u"]["Q"])
            self.assertEqual(state["object_summary"]["p"]["unknown_count"], 1)

    def test_local_footprint_does_not_fill_parent_region_and_keeps_max_on_revisit(self):
        first = event("e1", .4, 2, 3)
        uid = "u:local_footprint"
        first["selection"]["associations"][0]["candidate_id"] = uid
        first["assessments"][0]["search_unit_id"] = uid
        first["new_parents"] = [{"parent_id": "operated_object", "name": "box"}]
        first["new_units"] = [{"search_unit_id": uid, "parent_id": "operated_object",
            "source_region_id": "u", "name": "local box footprint", "region_type": "occluder_footprint",
            "identity_anchor": "original_footprint", "boundary_description": "frozen local footprint",
            "confirmation_basis": "geometric reference"}]
        second = event("e2", .2, 4, 5)
        second["selection"]["associations"][0]["candidate_id"] = uid
        second["assessments"][0]["search_unit_id"] = uid
        state = replay(catalog(), [first, second])
        self.assertEqual(state["units"][uid]["Q"], .4)
        self.assertIsNone(state["units"]["u"]["Q"])
        self.assertEqual(state["object_summary"]["operated_object"]["Q"], .4)
        self.assertNotIn("p", state["object_summary"])

    def test_selection_must_use_displayed_ordered_frames(self):
        with TemporaryDirectory() as tmp:
            image = Path(tmp) / "f.jpg"
            Image.new("RGB", (1600, 1000), "white").save(image)
            frames = [{"source_frame_id": i, "timestamp_s": float(i), "image": str(image),
                       "action": "unknown", "object_state": "unknown", "tags": []}
                      for i in (1, 2, 3, 4)]
            manifest = {"episode_id": "ep", "event_id": "e1", "candidates": frames,
                        "required_reference_frame_ids": [1, 4]}
            chosen = selected()
            chosen.update({"pre": 2, "pnr": 3, "post": 4})
            raw = {"model": "Qwen3-VL-8B-Instruct", "choices": [{"finish_reason": "stop",
                    "message": {"content": json.dumps({k: chosen[k] for k in
                    ("status", "pre", "pnr", "post", "reason", "associations")})}}]}
            def request(payload):
                shown = [block["image_url"]["url"] for block in payload["messages"][0]["content"]
                         if block["type"] == "image_url"]
                self.assertEqual(len(shown), len(frames))
                for value in shown:
                    submitted = Image.open(BytesIO(base64.b64decode(value.split(",", 1)[1])))
                    self.assertLessEqual(max(submitted.size), 768)
                return raw
            result = select(manifest, catalog(), Path(tmp) / "out", "http://unused", request=request)
            self.assertEqual((result["pre"], result["pnr"], result["post"]), (2, 3, 4))
            chosen["pnr"] = 99
            with self.assertRaisesRegex(ValueError, "displayed"):
                validate_selection(manifest, catalog(), chosen)

    def test_geometry_uses_union_of_three_selected_frames(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            h = w = 100
            K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
            depth = np.ones((h, w), np.float32)
            obj = np.zeros((h, w), np.uint8)
            obj[42:58, 42:58] = 1
            depth[obj > 0] = 0.9
            support = (1 - obj).astype(np.uint8)
            ref = root / "reference.npz"
            np.savez(ref, depth=depth, valid=np.ones_like(obj), object_mask=obj,
                     support_mask=support, K=K)
            paths = {}
            for role, fid in (("pre", 1), ("pnr", 2), ("post", 3)):
                foreground = np.zeros((h, w), np.uint8)
                if role == "pre":
                    foreground[:, 50:] = 1
                elif role == "pnr":
                    foreground[:, :50] = 1
                else:
                    foreground[:] = 1
                path = root / f"{role}.npz"
                np.savez(path, depth=np.ones((h, w), np.float32), valid=np.ones((h, w), np.uint8),
                         foreground=foreground, K=K, T=np.eye(4), pose_valid=True,
                         source_frame_id=fid)
                paths[role] = str(path)
            job = {"search_unit_id": "u", "scope": "original_support_footprint",
                   "reference_npz": str(ref), "frames": paths,
                   "quality_audit": {"accepted": True, "reasons": [], "source": "synthetic_fixture"}}
            result = score_job(job, selected(), root / "score")
            self.assertEqual(result["status"], "valid")
            self.assertAlmostEqual(result["q"], 1.0)
            self.assertLess(result["per_frame_visible_fraction"]["pre"], 1.0)
            self.assertLess(result["per_frame_visible_fraction"]["pnr"], 1.0)
            job["quality_audit"] = {"accepted": False, "reasons": ["competing_support_planes"]}
            rejected = score_job(job, selected(), root / "rejected")
            self.assertIsNone(rejected["q"])
            self.assertEqual(rejected["raw_q"], 1.0)
            # 3.md: unknown point observations stay in the frozen denominator.
            for path in paths.values():
                with np.load(path, allow_pickle=False) as data:
                    arrays = {key: data[key] for key in data.files}
                arrays["valid"] = np.zeros((h, w), np.uint8)
                np.savez(path, **arrays)
            job["quality_audit"] = {"accepted": True, "reasons": []}
            unknown = score_job(job, selected(), root / "point_unknown")
            self.assertEqual(unknown["q"], 0.0)
            self.assertEqual(unknown["unknown_fraction"], 1.0)
            self.assertIn("many_reference_points_without_valid_depth", unknown["warnings"])


if __name__ == "__main__":
    unittest.main()
