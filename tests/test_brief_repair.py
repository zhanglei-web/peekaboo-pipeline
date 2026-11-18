"""A model must resolve contradictory visual output; invalid evidence is never repaired in code."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from online_annotation.brief import propose


def valid_output():
    return {"s": "table", "p": [["table", [[42, 10, 10, 900, 900]]]],
            "r": [[0, "top", "s", "bounded tabletop", "visible support", "a", "h",
                   [[42, 20, 20, 800, 800]]]], "u": []}


class StubModel:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []

    def infer(self, folder, prompt, images, schema, max_tokens):
        self.requests.append((Path(folder).name, prompt, images))
        return deepcopy(next(self.values)), {"seconds": 1, "model": "test-only"}


class BriefRepairTests(unittest.TestCase):
    def test_reasks_for_each_error_and_retains_valid_model_evidence(self):
        bad_kind = valid_output()
        bad_kind["r"][0][5] = "v"
        bad_box = valid_output()
        bad_box["r"][0][-1] = [[42, 20, 30, 800, 30]]
        model = StubModel([bad_kind, bad_box, valid_output(), valid_output()])
        with TemporaryDirectory() as tmp:
            draft, final, receipt, _ = propose(model, tmp, ["shown-image.jpg"],
                                               [{"frame_index": 42}], "prompt", "review")
        self.assertEqual([x[0] for x in model.requests],
                         ["regions", "regions_repair_1", "regions_repair_2", "region_review"])
        self.assertEqual(draft["regions"][0]["evidence"][0]["bbox_1000"], [20, 20, 800, 800])
        self.assertEqual(final, draft)
        self.assertEqual(receipt["seconds"], 3)
        self.assertTrue(all(request[2] == ["shown-image.jpg"] for request in model.requests))

    def test_stops_after_three_failed_model_revisions(self):
        invalid = valid_output()
        invalid["r"][0][5] = "v"
        model = StubModel([invalid] * 4)
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "visibility"):
                propose(model, tmp, ["shown-image.jpg"], [{"frame_index": 42}], "prompt", "review")
        self.assertEqual(len(model.requests), 4)


if __name__ == "__main__":
    unittest.main()
