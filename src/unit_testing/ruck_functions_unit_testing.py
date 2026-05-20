"""
ruck_functions_unit_testing.py

Unit tests for the ruck_functions.py module, which provides utilities for ruck detection, ball release, and bounding box processing.

Tested Functions:
    - get_last_feet: Tests extraction of bottom-left and bottom-right coordinates from a ruck bounding box.
    - get_ball_release: Tests early exit behavior when paused_state['exit'] is True and generator exhaustion.

Test Structure:
    - Uses unittest and unittest.mock for mocking dependencies and simulating YOLO results and OpenCV.
    - setUp creates dummy frame and result objects for use in all tests.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src import ruck_functions

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

class TestRuckFunctions(unittest.TestCase):
    def setUp(self):
        # Dummy frame and detection boxes for use in tests
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.boxes = [[10, 10, 50, 50], [60, 10, 100, 50]]
        self.dummy_result = DummyResult(self.boxes)

    def test_get_last_feet(self):
        """
        Test get_last_feet returns correct bottom-left and bottom-right coordinates from a ruck bounding box.
        """
        ruck_box = (10, 20, 30, 40)
        bottom_left, bottom_right = ruck_functions.get_last_feet(ruck_box)
        self.assertEqual(bottom_left, (10, 40))
        self.assertEqual(bottom_right, (30, 40))

    def test_get_ball_release_generator_exhausted(self):
        """
        Test get_ball_release raises SystemExit if any generator is None (exhausted).
        """
        paused_state = {'exit': False}
        fps = 25
        with self.assertRaises(SystemExit):
            ruck_functions.get_ball_release(None, None, None, paused_state, fps)

    @patch('ruck_functions.general')
    @patch('ruck_functions.draw')
    @patch('ruck_functions.ball')
    @patch('ruck_functions.cv2')
    def test_get_ball_release_early_exit(self, mock_cv2, mock_ball, mock_draw, mock_general):
        """
        Test get_ball_release returns the correct tuple when paused_state['exit'] is True.
        Mocks all dependencies and simulates a single frame for each generator.
        """
        # Setup minimal mocks for dependencies
        mock_general.get_class_detections.return_value = (
            [np.array([10, 10, 50, 50])], [1], [0.9]
        )
        mock_general.box_centre.return_value = (30, 30)
        mock_general.round_list_values.side_effect = lambda x: x
        mock_general.box_overlap.return_value = True
        mock_general.display_frame.return_value = None

        mock_draw.draw_boxes.side_effect = lambda img, boxes, **kwargs: img

        mock_ball.update_tracker.return_value = ((30, 30), False, 0)
        mock_ball.detect_ball_release.return_value = (False, False)

        mock_cv2.TrackerCSRT_create.return_value = MagicMock(init=MagicMock())

        # Create generators for ruck, ball, and lineout results
        def gen():
            yield self.dummy_result
        ruck_results = gen()
        ball_results = gen()
        lineout_results = gen()

        paused_state = {'exit': True}
        fps = 25

        # Should return the early exit tuple
        result = ruck_functions.get_ball_release(
            ruck_results, ball_results, lineout_results, paused_state, fps
        )
        self.assertEqual(result, (None, [], (100, 100)))

if __name__ == '__main__':
    unittest.main()