"""Meaningful boundaries: causality, synchronization, references and unknown evidence."""
import unittest

from online_annotation.contracts import build_catalog, validate_proposals
from online_annotation.frames import prefix_rows, stereo_match


def proposal():
    return {
        "scene_description": "一个带抽屉的柜子",
        "parents": [{"ref": "p1", "name": "柜子", "description": "可见柜体",
                     "evidence": [{"frame_index": 0, "bbox_1000": [200, 200, 800, 900]}]}],
        "regions": [{
            "ref": "r1", "name": "柜子上层抽屉内部", "region_type": "container_interior",
            "parent_ref": "p1", "boundary_description": "抽屉内部，尚不可见",
            "existence_basis": "可见抽屉面板和把手", "identity_anchor": "fixed_support",
            "interior_visibility": "not_visible", "confidence": "medium",
            "evidence": [{"frame_index": 0, "bbox_1000": [250, 300, 700, 500]}],
        }], "uncertainties": ["内部几何未知"],
    }


class InitializationBoundaries(unittest.TestCase):
    def test_observation_cutoff_excludes_future_and_endpoint(self):
        rows = [(0, 10_000_000_000), (1, 11_999_999_999),
                (2, 12_000_000_000), (3, 13_000_000_000)]
        prefix, cutoff = prefix_rows(rows, 2)
        self.assertEqual([row[0] for row in prefix], [0, 1])
        self.assertEqual(cutoff, 12_000_000_000)

    def test_stereo_matches_timestamp_instead_of_index(self):
        right = [(4, 100), (5, 200), (6, 300)]
        self.assertEqual(stereo_match(201, right, 400, 2)["frame_index"], 5)
        self.assertIsNone(stereo_match(205, right, 400, 2))
        self.assertIsNone(stereo_match(300, right, 300, 2))

    def test_unseen_frame_and_invalid_parent_are_rejected(self):
        value = proposal()
        value["regions"][0]["evidence"][0]["frame_index"] = 90
        with self.assertRaisesRegex(ValueError, "unseen"):
            validate_proposals(value, [0])
        value = proposal()
        value["regions"][0]["parent_ref"] = "invented"
        with self.assertRaisesRegex(ValueError, "unknown parent"):
            validate_proposals(value, [0])

    def test_invisible_interior_never_becomes_geometry_or_zero_score(self):
        value = build_catalog(proposal(), "episode", [{"frame_index": 0}],
                              {"observation_end_exclusive_ns": 1}, "local")
        region = value["regions"][0]
        self.assertIsNone(region["geometry_support"])
        self.assertIsNone(region["cumulative_sufficiency"])
        self.assertFalse(region["searched"])
        self.assertFalse(value["parents"][0]["selectable"])
        self.assertEqual(region["interior_visibility"], "not_visible")

    def test_model_coordinates_are_explicitly_normalized_without_changing_raw(self):
        source = proposal()
        value = build_catalog(source, "episode", [{"frame_index": 0}],
                              {"observation_end_exclusive_ns": 1}, "local")
        evidence = value["regions"][0]["evidence"][0]
        self.assertEqual(evidence["bbox"], [0.25, 0.3, 0.7, 0.5])
        self.assertEqual(evidence["bbox_1000"], [250, 300, 700, 500])
        self.assertNotIn("bbox", source["regions"][0]["evidence"][0])


if __name__ == "__main__":
    unittest.main()
