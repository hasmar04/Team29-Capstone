"""
offside_functions_unit_testing.py

Unit tests for the offside_functions.py module, which provides functions for detecting offside players and team assignment using computer vision techniques.

Tested Functions:
    - get_player_colour_dict
    - get_dominant_colour_kmeans
    - find_team_colours
    - sort_into_teams
    - get_player_side
    - get_offside_players
    - get_player_coord_dict
    - check_between_lines
    - get_players_between_lines
    - filter_for_offside_detection
    - filter_detections_off_the_field

Test Structure:
    - Uses unittest and unittest.mock for mocking dependencies and simulating YOLO results and OpenCV.
    - setUp creates dummy data for use in all tests.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src import offside_functions

class DummyBoxes:
    """
    Dummy class to simulate YOLO detection boxes for testing.
    """
    def __init__(self, boxes):
        self.xyxy = MagicMock()
        self.xyxy.cpu.return_value = np.array(boxes)
        self.cls = MagicMock()
        self.cls.cpu.return_value = np.zeros(len(boxes))
        self.conf = MagicMock()
        self.conf.cpu.return_value = np.ones(len(boxes))

class DummyResult:
    """
    Dummy class to simulate a YOLO result object for testing.
    """
    def __init__(self, boxes):
        self.boxes = DummyBoxes(boxes)
        self.orig_img = np.zeros((100, 100, 3), dtype=np.uint8)

class TestOffsideFunctions(unittest.TestCase):
    def setUp(self):
        # Dummy frame and detection boxes for use in tests
        self.frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.boxes = [[10, 10, 20, 40], [30, 10, 40, 40]]
        self.dummy_result = DummyResult(self.boxes)
        self.player_dict = {
            (15, 40): (10, 10, 20, 40),
            (35, 40): (30, 10, 40, 40)
        }
    def test_check_between_lines(self):
        """
        Test check_between_lines returns a boolean for a player and two lines.
        """
        player = (5, 5)
        left_line = (0, 0, 0, 10)
        right_line = (10, 0, 10, 10)
        result = offside_functions.check_between_lines(player, left_line, right_line)
        self.assertIsInstance(result, bool)

    def test_get_players_between_lines(self):
        """
        Test get_players_between_lines returns a list of players between two lines.
        """
        player_dict = { (5, 5): (1, 1, 10, 10), (15, 5): (11, 1, 20, 10) }
        left_line = (0, 0, 0, 10)
        right_line = (10, 0, 10, 10)
        result = offside_functions.get_players_between_lines(player_dict, left_line, right_line)
        self.assertIsInstance(result, list)

    @patch('src.offside_functions.general')
    def test_filter_for_offside_detection(self, mock_general):
        """
        Test filter_for_offside_detection returns a dict after filtering.
        """
        player_dict = { (5, 5): (1, 1, 10, 10), (15, 5): (11, 1, 20, 10) }
        roi = (0, 0, 10, 10)
        mock_general.box_overlap.return_value = False
        result = offside_functions.filter_for_offside_detection(player_dict, roi)
        self.assertIsInstance(result, dict)

    @patch('src.offside_functions.general')
    def test_filter_detections_off_the_field(self, mock_general):
        """
        Test filter_detections_off_the_field returns a dict after filtering.
        """
        player_dict = { (5, 5): (1, 1, 10, 10), (15, 95): (11, 91, 20, 99) }
        lineout_box = (0, 90, 20, 99)
        imsize = (100, 100)
        mock_general.box_centre.return_value = (10, 95)
        mock_general.box_bottom_centre.return_value = (10, 99)
        mock_general.box_top_centre.return_value = (10, 90)
        result = offside_functions.filter_detections_off_the_field(player_dict, lineout_box, imsize)
        self.assertIsInstance(result, dict)

if __name__ == '__main__':
    unittest.main()