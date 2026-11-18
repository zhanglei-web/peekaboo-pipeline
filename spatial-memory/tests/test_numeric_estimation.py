from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spatial_memory.estimation import ensure_numeric
from spatial_memory.memory import replay
from test_spatial_memory import catalog, event


class EstimatorModel:
    model = "Qwen3-VL-8B-Instruct"

    def __init__(self, value=None, fail=False):
        self.value, self.fail, self.calls = value, fail, 0

    def infer(self, *args):
        self.calls += 1
        if self.fail:
            raise RuntimeError("test transport unavailable")
        return self.value, {"model": self.model, "seconds": 0}


def failed_assessment():
    return {"episode_id": "ep", "event_id": "e1", "search_unit_id": "u",
            "score_type": "geometric_coverage", "scope": "original_support_footprint",
            "status": "uncomputable", "q": None, "reason": "object mask unavailable"}


class NumericEstimationTests(unittest.TestCase):
    def test_visual_estimate_is_continuous_and_explicitly_not_geometric(self):
        model = EstimatorModel({"q": .37, "reason": "partial view", "scope_description": "drawer interior",
                                "evidence_strength": "low"})
        with TemporaryDirectory() as tmp:
            result = ensure_numeric(failed_assessment(), None, model,
                {r: {"left": "synthetic-placeholder-unused-by-stub"} for r in ("pre", "pnr", "post")},
                {"name": "drawer", "boundary_description": "interior"}, tmp)
            self.assertEqual(result["q"], .37)
            self.assertEqual(result["score_type"], "visual_sufficiency_estimate")
            self.assertFalse(result["strict_geometry_accepted"])
            self.assertTrue((Path(tmp) / "strict_assessment.json").exists())

    def test_quality_flagged_geometric_value_is_retained_as_estimate(self):
        original = {**failed_assessment(), "raw_q": .23}
        model = EstimatorModel(fail=True)
        job = {"scope": "original_support_footprint", "quality_audit": {"accepted": False,
               "reasons": ["competing_support_plane_hypotheses"]}}
        with TemporaryDirectory() as tmp:
            result = ensure_numeric(original, job, model, {}, {}, tmp)
        self.assertEqual(result["q"], .23)
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(result["score_type"], "geometric_coverage")
        self.assertEqual(model.calls, 0)

    def test_no_evidence_default_is_numeric_and_never_labeled_measurement(self):
        model = EstimatorModel(fail=True)
        with TemporaryDirectory() as tmp:
            result = ensure_numeric(failed_assessment(), None, model, {}, {}, tmp)
        self.assertEqual(result["q"], 0.)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["score_type"], "default_sufficiency")
        self.assertFalse(result["default_is_geometric_measurement"])

    def test_operational_max_keeps_strict_geometry_in_its_own_field(self):
        first = event("e1", .4, 2, 3)
        second = event("e2", .81, 4, 5)
        second["score_policy"] = "best_effort"
        second["assessments"][0].update(status="estimated", score_type="visual_sufficiency_estimate",
                                         scope="candidate_visual_scope", estimation_method="qwen_continuous_visual_estimate")
        state = replay(catalog(), [first, second])
        self.assertEqual(state["units"]["u"]["Q"], .81)
        self.assertEqual(state["units"]["u"]["Q_geometric"], .4)
        self.assertEqual(state["units"]["u"]["Q_estimated"], .81)
        self.assertEqual(state["units"]["u"]["Q_source"]["score_type"], "visual_sufficiency_estimate")

    def test_best_effort_rejects_missing_number(self):
        record = event("e1", None, 2, 3)
        record["score_policy"] = "best_effort"
        with self.assertRaisesRegex(ValueError, "numeric value"):
            replay(catalog(), [record])


if __name__ == "__main__":
    unittest.main()
