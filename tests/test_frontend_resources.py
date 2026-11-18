"""The standalone event frontend must include its runtime configuration."""
import unittest

from event_frontend.clusters import pipeline_config


class FrontendResourcesTests(unittest.TestCase):
    def test_standalone_contact_config_loads_and_validates(self):
        config = pipeline_config()
        self.assertEqual(config["target_selection_strategy"], "contact_quality")
        self.assertEqual(config["hand_score_min"], .3)
        self.assertGreaterEqual(config["max_frames_per_packet"], 4)


if __name__ == "__main__":
    unittest.main()
