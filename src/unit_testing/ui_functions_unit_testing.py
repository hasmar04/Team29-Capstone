"""
ui_functions_unit_testing.py

Unit tests for the ui_functions.py module, which provides user interface utilities for file selection, input validation, and user-guided parameter selection.

Tested Functions:
    - check_is_video
    - get_model_type
    - get_boolean
    - get_frame

Test Structure:
    - Uses unittest and unittest.mock for mocking dependencies and simulating user input and file dialogs.
    - GUI and interactive functions (select_file, threshold_slider, get_point_locations, get_coordinates) are not tested here due to their interactive nature.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src import ui_functions

class TestUIFunctions(unittest.TestCase):
    def test_check_is_video_image(self):
        """
        Test check_is_video returns False for image file extensions.
        """
        self.assertFalse(ui_functions.check_is_video("test.jpg"))
        self.assertFalse(ui_functions.check_is_video("test.jpeg"))
        self.assertFalse(ui_functions.check_is_video("test.png"))

    def test_check_is_video_video(self):
        """
        Test check_is_video returns True for video file extensions.
        """
        self.assertTrue(ui_functions.check_is_video("test.mp4"))
        self.assertTrue(ui_functions.check_is_video("test.avi"))
        self.assertTrue(ui_functions.check_is_video("test.mov"))
        self.assertTrue(ui_functions.check_is_video("test.gif"))

    def test_check_is_video_unknown(self):
        """
        Test check_is_video raises SystemExit for unknown file extensions.
        """
        with self.assertRaises(SystemExit):
            ui_functions.check_is_video("test.txt")

    @patch('src.builtins.input', return_value='ball')
    def test_get_model_type_ball(self, mock_input):
        """
        Test get_model_type returns 'ball' when user inputs 'ball'.
        """
        self.assertEqual(ui_functions.get_model_type(), 'ball')

    @patch('src.builtins.input', return_value='all')
    def test_get_model_type_all(self, mock_input):
        """
        Test get_model_type returns 'rugby' when user inputs 'all'.
        """
        self.assertEqual(ui_functions.get_model_type(), 'rugby')

    @patch('src.builtins.input', return_value='invalid')
    def test_get_model_type_invalid(self, mock_input):
        """
        Test get_model_type raises SystemExit for invalid input.
        """
        with self.assertRaises(SystemExit):
            ui_functions.get_model_type()

    @patch('src.builtins.input', return_value='yes')
    def test_get_boolean_true(self, mock_input):
        """
        Test get_boolean returns True for 'yes' input.
        """
        self.assertTrue(ui_functions.get_boolean("Prompt: "))

    @patch('src.builtins.input', return_value='no')
    def test_get_boolean_false(self, mock_input):
        """
        Test get_boolean returns False for 'no' input.
        """
        self.assertFalse(ui_functions.get_boolean("Prompt: "))

    @patch('src.builtins.input', return_value='maybe')
    def test_get_boolean_invalid(self, mock_input):
        """
        Test get_boolean raises SystemExit for invalid input.
        """
        with self.assertRaises(SystemExit):
            ui_functions.get_boolean("Prompt: ")

    @patch('src.ui_functions.cv2')
    def test_get_frame_success(self, mock_cv2):
        """
        Test get_frame returns a valid frame when video can be opened and read.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: 10 if x == mock_cv2.CAP_PROP_FRAME_COUNT else 0
        mock_cap.read.return_value = (True, np.ones((100, 100, 3), dtype=np.uint8))
        mock_cv2.VideoCapture.return_value = mock_cap
        frame = ui_functions.get_frame("dummy.mp4")
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (100, 100, 3))
        mock_cap.release.assert_called_once()

    @patch('src.ui_functions.cv2')
    def test_get_frame_not_opened(self, mock_cv2):
        """
        Test get_frame returns None if the video cannot be opened.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        frame = ui_functions.get_frame("dummy.mp4")
        self.assertIsNone(frame)

    @patch('src.ui_functions.cv2')
    def test_get_frame_no_frames(self, mock_cv2):
        """
        Test get_frame returns None if the video has zero frames.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 0
        mock_cv2.VideoCapture.return_value = mock_cap
        frame = ui_functions.get_frame("dummy.mp4")
        self.assertIsNone(frame)

    @patch('src.ui_functions.cv2')
    def test_get_frame_read_fail(self, mock_cv2):
        """
        Test get_frame returns None if reading the frame fails.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 10
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap
        frame = ui_functions.get_frame("dummy.mp4")
        self.assertIsNone(frame)

if __name__ == '__main__':
    unittest.main()