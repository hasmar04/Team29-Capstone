import unittest
import numpy as np

from src.player_detection import (
    filter_player_boxes,
    get_player_bottom_centre,
    get_jersey_crop,
    extract_jersey_colour,
    assign_teams_by_colour,
    count_teams_and_refs
)


class MockBox:
    def __init__(self, xyxy, conf, cls=None):
        self.xyxy = [np.array(xyxy)]
        self.conf = [conf]
        self.cls = [cls] if cls is not None else None


class MockResult:
    def __init__(self, boxes, names=None):
        self.boxes = boxes
        self.names = names or {0: "player", 1: "ref"}


class TestPlayerDetection(unittest.TestCase):

    def test_filter_player_boxes_confidence(self):
        boxes = [
            MockBox([0, 0, 10, 10], 0.7, 0),
            MockBox([0, 0, 10, 10], 0.5, 0),
        ]
        result = MockResult(boxes)

        filtered = filter_player_boxes(result)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["box"], (0, 0, 10, 10))

    def test_bottom_centre(self):
        filtered_boxes = [
            {"box": (0, 0, 10, 20), "class_id": 0, "class_name": "player"}
        ]

        players = get_player_bottom_centre(filtered_boxes)

        self.assertEqual(players[0]["bottom_centre"], (5, 20))

    def test_jersey_crop_shape(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8)

        players = [{"box": (10, 10, 50, 90)}]

        players = get_jersey_crop(frame, players)

        crop = players[0]["jersey_crop"]

        self.assertIsNotNone(crop)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_extract_colour(self):
        crop = np.ones((10, 10, 3), dtype=np.uint8) * [10, 20, 30]

        players = [{"jersey_crop": crop}]

        players = extract_jersey_colour(players)

        self.assertEqual(players[0]["jersey_colour"], (30, 20, 10))

    def test_assign_teams(self):
        players = [
            {"jersey_colour": (255, 0, 0)},
            {"jersey_colour": (250, 0, 0)},
            {"jersey_colour": (0, 0, 255)},
            {"jersey_colour": (0, 0, 240)},
        ]

        players = assign_teams_by_colour(players)

        teams = [p["team"] for p in players]

        self.assertEqual(set(teams), {0, 1})

    def test_counting(self):
        players = [
            {"team": 0, "class_name": "player"},
            {"team": 1, "class_name": "player"},
            {"team": None, "class_name": "player"},
            {"class_name": "ref"},
        ]

        counts = count_teams_and_refs(players)

        self.assertEqual(counts["team_0"], 1)
        self.assertEqual(counts["team_1"], 1)
        self.assertEqual(counts["unknown_team"], 1)
        self.assertEqual(counts["refs"], 1)


if __name__ == "__main__":
    unittest.main()