import json
import os
import tempfile
import unittest

from src.processing import batch_processor as batch


class TestBatchProcessorIntegrationPayloads(unittest.TestCase):

    def test_build_summary_empty(self):
        summary = batch.build_summary({"events": []})
        self.assertEqual(summary["total_events"], 0)
        self.assertEqual(summary["total_offside_players"], 0)
        self.assertEqual(summary["average_confidence"], 0.0)

    def test_build_summary_counts_events(self):
        payload = {
            "events": [
                {"type": "ruck", "detection_confidence": 0.8, "offside_count": 2},
                {"type": "lineout", "detection_confidence": 0.6, "offside_count": 1},
            ]
        }
        summary = batch.build_summary(payload)
        self.assertEqual(summary["total_events"], 2)
        self.assertEqual(summary["ruck_count"], 1)
        self.assertEqual(summary["lineout_count"], 1)
        self.assertEqual(summary["total_offside_players"], 3)
        self.assertAlmostEqual(summary["average_confidence"], 0.7)

    def test_frontend_payload_defaults(self):
        payload = batch.build_frontend_payload("clip.mp4", {"events": []})
        self.assertEqual(payload["video_name"], "clip")
        self.assertIn("summary", payload)
        self.assertIn("logs", payload)
        self.assertIn("output_files", payload)

    def test_write_json_summary(self):
        data = {"video_name": "clip", "events": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "clip_analysis.json")
            written = batch.write_json_summary(path, data)
            self.assertTrue(os.path.exists(written))
            with open(written, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded["video_name"], "clip")

    def test_make_log_entry_contains_required_fields(self):
        entry = batch.make_log_entry(
            "testing",
            "message",
            level="WARNING",
            event_type="ruck",
            frame_number=10,
            confidence=0.9,
            player_ids=[1, 2],
            error=ValueError("bad frame"),
        )
        self.assertEqual(entry["stage"], "testing")
        self.assertEqual(entry["level"], "WARNING")
        self.assertEqual(entry["event_type"], "ruck")
        self.assertIn("bad frame", entry["error"])


if __name__ == "__main__":
    unittest.main()
