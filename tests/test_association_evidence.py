"""Repeated frame explanations must not duplicate a spatial memory update."""
import unittest

from spatial_memory.selection import merge_association_evidence


class AssociationEvidenceTests(unittest.TestCase):
    def test_three_frame_explanations_preserve_one_id_and_all_evidence(self):
        original = {"pre": 480, "pnr": 514, "post": 539, "associations": [
            {"candidate_id": "drawer", "reason": "closed before", "confidence": "high"},
            {"candidate_id": "drawer", "reason": "open during", "confidence": "medium"},
            {"candidate_id": "drawer", "reason": "visible after", "confidence": "high"}]}
        result, audit = merge_association_evidence(original)
        self.assertEqual(len(result["associations"]), 1)
        item = result["associations"][0]
        self.assertEqual(item["candidate_id"], "drawer")
        self.assertEqual(item["confidence"], "medium")
        for entry in original["associations"]:
            self.assertIn(entry["reason"], item["reason"])
        self.assertEqual(audit[0]["original_entries"], original["associations"])
        self.assertEqual([result[r] for r in ("pre", "pnr", "post")], [480, 514, 539])
        self.assertEqual(len(original["associations"]), 3)

    def test_invalid_duplicate_evidence_is_not_hidden_by_merging(self):
        with self.assertRaises(ValueError):
            merge_association_evidence({"associations": [
                {"candidate_id": "drawer", "reason": "open", "confidence": "high"},
                {"candidate_id": "drawer", "reason": "", "confidence": "high"}]})


if __name__ == "__main__":
    unittest.main()
