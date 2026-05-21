import unittest

from src.logging.detection_event import DetectionEvent
from src.logging.session_stats import DetectionSessionStats


class TestStructuredLogging(unittest.TestCase):

    def test_detection_event_to_dict(self):
        event = DetectionEvent(
            event_type="ruck",
            confidence=0.91,
            frame_number=42,
            timestamp=1.68,
            offside_count=2,
            team_counts={"team_0": 5},
        )
        data = event.to_dict()
        self.assertEqual(data["type"], "ruck")
        self.assertEqual(data["detection_confidence"], 0.91)
        self.assertEqual(data["team_counts"]["team_0"], 5)

    def test_detection_event_from_dict(self):
        event = DetectionEvent.from_dict({
            "type": "lineout",
            "detection_confidence": 0.75,
            "frame_number": 100,
            "timestamp": 4.0,
            "offside_count": 1,
        })
        self.assertEqual(event.event_type, "lineout")
        self.assertEqual(event.confidence, 0.75)

    def test_session_stats_add_event_object(self):
        stats = DetectionSessionStats()
        stats.add_event(DetectionEvent("ruck", 0.8, 10, 0.4, 2))
        self.assertEqual(stats.total_detections, 1)
        self.assertEqual(stats.ruck_count, 1)
        self.assertEqual(stats.total_offside_players, 2)
        self.assertAlmostEqual(stats.average_confidence, 0.8)

    def test_session_stats_add_event_dict(self):
        stats = DetectionSessionStats()
        stats.add_event({"type": "lineout", "detection_confidence": 0.6, "offside_count": 3})
        data = stats.to_dict()
        self.assertEqual(data["lineout_count"], 1)
        self.assertEqual(data["total_offside_players"], 3)


if __name__ == "__main__":
    unittest.main()
