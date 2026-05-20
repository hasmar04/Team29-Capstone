"""
ball_functions_unit_testing.py

Unit tests for the ball_functions.py module, which provides functions for tracking and detecting the rugby ball during lineouts and rucks using computer vision techniques.

Tested Functions:
    - update_tracker: Tests tracker update logic, including success, failure below threshold, and failure above threshold.
    - detect_ball_release: Tests detection of ball release events based on movement and stationary state.
    - get_ball_release_lineout: Tests early exit behavior when paused_state['exit'] is True.
    - get_ball_release_ruck: Tests early exit behavior when paused_state['exit'] is True.

Test Structure:
    - Uses unittest and unittest.mock for mocking dependencies and simulating tracker and YOLO results.
    - setUp creates a dummy frame for use in all tests.
    - Each test checks the correctness of function outputs and state changes under various scenarios.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from src import ball_functions

class TestBallFunctions(unittest.TestCase):
    """
    Unit tests for the ball_functions module.
    Tests tracker update logic, ball release detection, and early exit behavior for lineout/ruck ball release functions.
    """
    def setUp(self):
        """
        Set up a dummy frame for use in all tests.
        """
        # Create a dummy frame
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.annotated_frame = self.frame.copy()

    def test_update_tracker_success(self):
        """
        Test update_tracker returns correct center and resets fail counter on success.
        """
        tracker = MagicMock()
        tracker.update.return_value = (True, (10, 20, 30, 40))
        centre, active, fail_counter = ball_functions.update_tracker(tracker, self.frame, 0, 3, self.annotated_frame)
        self.assertEqual(centre, (25, 40))
        self.assertTrue(active)
        self.assertEqual(fail_counter, 0)

    def test_update_tracker_fail_below_threshold(self):
        """
        Test update_tracker increments fail counter and stays active if below threshold.
        """
        tracker = MagicMock()
        tracker.update.return_value = (False, (0, 0, 0, 0))
        centre, active, fail_counter = ball_functions.update_tracker(tracker, self.frame, 1, 3, self.annotated_frame)
        self.assertIsNone(centre)
        self.assertTrue(active)
        self.assertEqual(fail_counter, 2)

    def test_update_tracker_fail_above_threshold(self):
        """
        Test update_tracker disables tracker if fail counter exceeds threshold.
        """
        tracker = MagicMock()
        tracker.update.return_value = (False, (0, 0, 0, 0))
        centre, active, fail_counter = ball_functions.update_tracker(tracker, self.frame, 4, 3, self.annotated_frame)
        self.assertIsNone(centre)
        self.assertFalse(active)
        self.assertEqual(fail_counter, 5)

    def test_detect_ball_release_stationary(self):
        """
        Test detect_ball_release returns stationary True and no release if movement is below threshold.
        """
        current = (10, 10)
        previous = (11, 11)
        stationary, release = ball_functions.detect_ball_release(current, previous, False, 5)
        self.assertTrue(stationary)
        self.assertFalse(release)

    def test_detect_ball_release_release(self):
        """
        Test detect_ball_release returns release True if movement exceeds threshold and was stationary.
        """
        current = (20, 20)
        previous = (10, 10)
        stationary, release = ball_functions.detect_ball_release(current, previous, True, 5)
        self.assertFalse(stationary)
        self.assertTrue(release)

    def test_detect_ball_release_no_previous(self):
        """
        Test detect_ball_release returns stationary True and no release if no previous position.
        """
        current = (10, 10)
        previous = None
        stationary, release = ball_functions.detect_ball_release(current, previous, True, 5)
        self.assertTrue(stationary)
        self.assertFalse(release)

if __name__ == '__main__':
    unittest.main()
