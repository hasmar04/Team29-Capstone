"""
general_functions_unit_testing.py

Unit tests for the general_functions.py module, which provides general utility functions for image processing, coordinate conversion, detection result processing, and frame display.

Tested Functions:
    - load_and_resize_image
    - get_video_fps
    - get_class_detections
    - box_centre
    - box_bottom_centre
    - box_top_centre
    - box_overlap
    - convert_coordinates
    - round_list_values

Test Structure:
    - Uses unittest and unittest.mock for mocking dependencies and simulating file I/O and OpenCV.
    - setUp creates dummy data for use in all tests.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src import general_functions

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

class TestGeneralFunctions(unittest.TestCase):
    def setUp(self):
        # Dummy image and bounding boxes for use in tests
        self.dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.box = (10, 20, 30, 40)
        self.roi = (15, 25, 35, 45)
        self.boxes = [[10, 10, 20, 40], [30, 10, 40, 40]]
        self.dummy_result = DummyResult(self.boxes)

    @patch('general_functions.cv2')
    def test_load_and_resize_image_color(self, mock_cv2):
        # Test loading and resizing a color image
        mock_cv2.imread.return_value = np.ones((200, 200, 3), dtype=np.uint8) * 255
        mock_cv2.resize.side_effect = lambda img, size: np.ones((size[1], size[0], 3), dtype=np.uint8)
        img = general_functions.load_and_resize_image('dummy_path.jpg', width=50, height=50, gray=False)
        self.assertEqual(img.shape, (50, 50, 3))

    @patch('general_functions.cv2')
    def test_load_and_resize_image_gray(self, mock_cv2):
        # Test loading and resizing a grayscale image
        mock_cv2.imread.return_value = np.ones((200, 200), dtype=np.uint8) * 255
        mock_cv2.resize.side_effect = lambda img, size: np.ones((size[1], size[0]), dtype=np.uint8)
        img = general_functions.load_and_resize_image('dummy_path.jpg', width=60, height=40, gray=True)
        self.assertEqual(img.shape, (40, 60))

    @patch('general_functions.cv2')
    def test_load_and_resize_image_file_not_found(self, mock_cv2):
        # Test FileNotFoundError when image cannot be loaded
        mock_cv2.imread.return_value = None
        with self.assertRaises(FileNotFoundError):
            general_functions.load_and_resize_image('not_found.jpg')

    @patch('general_functions.cv2')
    def test_get_video_fps(self, mock_cv2):
        # Test getting video FPS from a valid video file
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 29.97
        mock_cv2.VideoCapture.return_value = mock_cap
        fps = general_functions.get_video_fps('dummy_video.mp4')
        self.assertEqual(fps, 30)
        mock_cap.release.assert_called_once()

    @patch('general_functions.cv2')
    def test_get_video_fps_file_not_found(self, mock_cv2):
        # Test FileNotFoundError when video file cannot be opened
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap
        with self.assertRaises(FileNotFoundError):
            general_functions.get_video_fps('not_found.mp4')

    def test_get_class_detections(self):
        # Test extracting class detections from a dummy YOLO result
        boxes, classes, confidences = general_functions.get_class_detections(self.dummy_result)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(len(classes), 2)
        self.assertEqual(len(confidences), 2)

    def test_get_class_detections_empty(self):
        # Test extracting class detections from an empty result
        class EmptyResult:
            boxes = None
        boxes, classes, confidences = general_functions.get_class_detections(EmptyResult())
        self.assertEqual(len(boxes), 0)
        self.assertEqual(len(classes), 0)
        self.assertEqual(len(confidences), 0)

    def test_box_centre(self):
        # Test calculation of bounding box centre
        centre = general_functions.box_centre(self.box)
        self.assertEqual(centre, (20, 30))

    def test_box_bottom_centre(self):
        # Test calculation of bounding box bottom centre
        bottom = general_functions.box_bottom_centre(self.box)
        self.assertEqual(bottom, (20, 40))

    def test_box_top_centre(self):
        # Test calculation of bounding box top centre
        top = general_functions.box_top_centre(self.box)
        self.assertEqual(top, (20, 20))

    def test_box_overlap_true(self):
        # Test overlap detection for overlapping boxes
        overlap = general_functions.box_overlap(self.box, self.roi, threshold=0.1)
        self.assertTrue(overlap)

    def test_box_overlap_false(self):
        # Test overlap detection for non-overlapping boxes
        box = (0, 0, 5, 5)
        roi = (10, 10, 20, 20)
        overlap = general_functions.box_overlap(box, roi, threshold=0.1)
        self.assertFalse(overlap)

    def test_convert_coordinates_xy(self):
        # Test coordinate conversion for a single (x, y) point
        coords = (100, 50)
        original_size = (200, 100)
        resized_size = (800, 400)
        result = general_functions.convert_coordinates(coords, original_size, resized_size)
        self.assertEqual(result, (400, 200))

    def test_convert_coordinates_xyxy(self):
        # Test coordinate conversion for a bounding box (x1, y1, x2, y2)
        coords = (0, 0, 200, 100)
        original_size = (200, 100)
        resized_size = (800, 400)
        result = general_functions.convert_coordinates(coords, original_size, resized_size)
        self.assertEqual(result, (0, 0, 800, 400))

    def test_convert_coordinates_list_xy(self):
        # Test coordinate conversion for a list of (x, y) points
        coords = [(0, 0), (100, 50)]
        original_size = (200, 100)
        resized_size = (800, 400)
        result = general_functions.convert_coordinates(coords, original_size, resized_size)
        self.assertEqual(result, [(0, 0), (400, 200)])

    def test_convert_coordinates_list_xyxy(self):
        # Test coordinate conversion for a list of bounding boxes
        coords = [(0, 0, 200, 100), (50, 25, 150, 75)]
        original_size = (200, 100)
        resized_size = (800, 400)
        result = general_functions.convert_coordinates(coords, original_size, resized_size)
        self.assertEqual(result, [(0, 0, 800, 400), (200, 100, 600, 300)])

    def test_convert_coordinates_invalid(self):
        # Test ValueError for invalid coordinate input
        with self.assertRaises(ValueError):
            general_functions.convert_coordinates("invalid", (100, 100))

    def test_round_list_values_1d(self):
        # Test rounding of 1D list values
        arr = [1.2, 2.7, 3.5]
        rounded = general_functions.round_list_values(arr)
        self.assertEqual(rounded, [1, 3, 4])

    def test_round_list_values_2d(self):
        # Test rounding of 2D list values
        arr = [[1.2, 2.7], [3.5, 4.1]]
        rounded = general_functions.round_list_values(arr)
        self.assertEqual(rounded, [[1, 3], [4, 4]])

    def test_round_list_values_tuple(self):
        # Test rounding of tuple values
        arr = [(1.2, 2.7), (3.5, 4.1)]
        rounded = general_functions.round_list_values(arr)
        self.assertEqual(rounded, [(1, 3), (4, 4)])

    def test_round_list_values_numpy(self):
        # Test rounding of numpy array values
        arr = np.array([[1.2, 2.7], [3.5, 4.1]])
        rounded = general_functions.round_list_values(arr)
        self.assertEqual(rounded, [[1, 3], [4, 4]])

    def test_round_list_values_invalid(self):
        # Test ValueError for invalid input to round_list_values
        with self.assertRaises(ValueError):
            general_functions.round_list_values("invalid")

if __name__ == '__main__':
    unittest.main()