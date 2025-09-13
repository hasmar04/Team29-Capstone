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
import offside_functions

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

    @patch('offside_functions.general')
    def test_get_player_colour_dict_empty(self, mock_general):
        """
        Test get_player_colour_dict returns empty team dicts if no players detected.
        """
        # Simulate empty generator
        def gen():
            if False:
                yield
        players_result = gen()
        result = offside_functions.get_player_colour_dict(players_result, self.frame)
        self.assertEqual(result, {'Team 1': {}, 'Team 2': {}, 'Other': {}})

    @patch('offside_functions.general')
    @patch('offside_functions.get_dominant_colour_kmeans')
    @patch('offside_functions.find_team_colours')
    @patch('offside_functions.sort_into_teams')
    def test_get_player_colour_dict(self, mock_sort, mock_find, mock_dom, mock_general):
        """
        Test get_player_colour_dict returns a dict with 'Team 1' key when players are present.
        """
        # Simulate generator with one result
        def gen():
            yield self.dummy_result
        players_result = gen()
        mock_general.round_list_values.side_effect = lambda x: x
        mock_dom.return_value = (100, 100, 100)
        mock_find.return_value = [(100, 100, 100), (200, 200, 200)]
        mock_sort.return_value = {'Team 1': { (15, 40): (10, 10, 20, 40) }, 'Team 2': {}, 'Other': {}}
        result = offside_functions.get_player_colour_dict(players_result, self.frame)
        self.assertIn('Team 1', result)

    def test_get_dominant_colour_kmeans_empty_roi(self):
        """
        Test get_dominant_colour_kmeans returns (0,0,0) for empty ROI.
        """
        roi = np.zeros((0, 0, 3), dtype=np.uint8)
        colour = offside_functions.get_dominant_colour_kmeans(roi)
        self.assertEqual(colour, (0, 0, 0))

    def test_get_dominant_colour_kmeans_valid(self):
        """
        Test get_dominant_colour_kmeans returns correct colour for uniform ROI.
        """
        roi = np.ones((10, 10, 3), dtype=np.uint8) * 100
        colour = offside_functions.get_dominant_colour_kmeans(roi, k=1)
        self.assertEqual(colour, (100, 100, 100))

    def test_find_team_colours(self):
        """
        Test find_team_colours returns two tuples for two input colours.
        """
        colours = [(100, 100, 100), (200, 200, 200)]
        result = offside_functions.find_team_colours(colours)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(c, tuple) for c in result))

    @patch('offside_functions.general')
    def test_sort_into_teams(self, mock_general):
        """
        Test sort_into_teams returns dict with 'Team 1', 'Team 2', and 'Other' keys.
        """
        player_dict = { (10, 10, 20, 40): (100, 100, 100), (30, 10, 40, 40): (200, 200, 200) }
        team_colours = [(100, 100, 100), (200, 200, 200)]
        mock_general.box_bottom_centre.side_effect = lambda box: (int((box[0]+box[2])/2), box[3])
        result = offside_functions.sort_into_teams(player_dict, team_colours, threshold=50)
        self.assertIn('Team 1', result)
        self.assertIn('Team 2', result)
        self.assertIn('Other', result)

    def test_get_player_side(self):
        """
        Test get_player_side returns 'left' or 'right' for a player and line.
        """
        player = (5, 5)
        line = (0, 0, 10, 0)
        side = offside_functions.get_player_side(player, line)
        self.assertIn(side, ['left', 'right'])

    def test_get_offside_players(self):
        """
        Test get_offside_players returns a list for sorted player dict and lines.
        """
        sorted_player_dict = {
            'Team 1': { (5, 5): (1, 1, 10, 10) },
            'Team 2': { (15, 5): (11, 1, 20, 10) },
            'Other': {}
        }
        centre_line = (0, 0, 10, 0)
        left_line = (0, 0, 0, 10)
        right_line = (20, 0, 20, 10)
        result = offside_functions.get_offside_players(centre_line, sorted_player_dict, left_line, right_line)
        self.assertIsInstance(result, list)

    @patch('offside_functions.general')
    def test_get_player_coord_dict(self, mock_general):
        """
        Test get_player_coord_dict returns a dict for a valid generator.
        """
        def gen():
            yield self.dummy_result
        mock_general.round_list_values.side_effect = lambda x: x
        mock_general.box_bottom_centre.side_effect = lambda box: (int((box[0]+box[2])/2), box[3])
        result = offside_functions.get_player_coord_dict(gen())
        self.assertTrue(isinstance(result, dict))

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

    @patch('offside_functions.general')
    def test_filter_for_offside_detection(self, mock_general):
        """
        Test filter_for_offside_detection returns a dict after filtering.
        """
        player_dict = { (5, 5): (1, 1, 10, 10), (15, 5): (11, 1, 20, 10) }
        roi = (0, 0, 10, 10)
        mock_general.box_overlap.return_value = False
        result = offside_functions.filter_for_offside_detection(player_dict, roi)
        self.assertIsInstance(result, dict)

    @patch('offside_functions.general')
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