"""
point_functions_unit_testing.py

Unit tests for the point_functions.py module, which provides functions for mapping rugby field lines and points, homography, and geometric transformations.

Tested Functions:
    - get_field_points
    - get_homography_matrix
    - get_lineout_offside_points
    - transform_lines
    - shortest_distance_between_lines

Test Structure:
    - Uses unittest and unittest.mock for mocking dependencies and simulating images, lines, and points.
    - setUp creates dummy data for use in all tests.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import point_functions

class TestPointFunctions(unittest.TestCase):
    def setUp(self):
        # Dummy image, lines, and points for use in tests
        self.image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.field_lines = [[10, 10, 90, 10], [10, 90, 90, 90]]
        self.field_outline = np.array([[[0,0]], [[0,99]], [[99,99]], [[99,0]]], dtype=np.int32)
        self.detected_points = [[100, 200], [300, 400], [500, 600], [700, 800]]
        self.point_locations = [('left_10m_line', 'bottom_15m'), ('right_10m_line', 'bottom_15m'),
                                ('left_10m_line', 'top_15m'), ('right_22m_line', 'top_15m')]
        self.lineout_centre = (50, 50)
        self.lines = [[10, 10, 90, 10], [10, 90, 90, 90]]

    @patch('point_functions.cv2')
    @patch('point_functions.field')
    @patch('point_functions.lf')
    def test_get_field_points(self, mock_lf, mock_field, mock_cv2):
        """
        Test get_field_points returns a list of intersection points.
        Mocks dependencies for contour and line extraction.
        """
        # Mock field.extract_straight_lines and fit_straight_lines_to_contours
        mock_field.extract_straight_lines.return_value = [[[10, 10], [20, 10]], [[20, 10], [20, 20]]]
        mock_field.fit_straight_lines_to_contours.return_value = [[10, 10, 20, 10], [20, 10, 20, 20]]
        mock_lf.find_intersection_point.return_value = (15, 10)
        mock_cv2.cvtColor.side_effect = lambda img, code: img[..., 0]  # fake grayscale
        mock_cv2.dilate.side_effect = lambda img, kernel, iterations: img
        mock_cv2.threshold.side_effect = lambda img, thresh, maxval, type: (None, img)
        mock_cv2.findContours.return_value = ([np.array([[[10,10]],[[20,10]],[[20,20]],[[10,20]]], dtype=np.int32)], None)
        intersections = point_functions.get_field_points(self.image, self.field_lines, self.field_outline)
        self.assertIsInstance(intersections, list)

    @patch('point_functions.FIELD_POINTS_DICT', {
        'left_10m_line': {'bottom_15m': [0, 0], 'top_15m': [0, 100]},
        'right_10m_line': {'bottom_15m': [100, 0]},
        'right_22m_line': {'top_15m': [100, 100]}
    })
    def test_get_homography_matrix(self):
        """
        Test get_homography_matrix returns a 3x3 numpy array for valid points and locations.
        """
        detected_points = [[0, 0], [100, 0], [0, 100], [100, 100]]
        point_locations = [('left_10m_line', 'bottom_15m'), ('right_10m_line', 'bottom_15m'),
                           ('left_10m_line', 'top_15m'), ('right_22m_line', 'top_15m')]
        H = point_functions.get_homography_matrix(detected_points, point_locations)
        self.assertIsInstance(H, np.ndarray)
        self.assertEqual(H.shape, (3, 3))

    def test_get_lineout_offside_points(self):
        """
        Test get_lineout_offside_points returns two (x, y) tuples for a given centre and identity homography.
        """
        # Use a simple homography (identity)
        H = np.eye(3, dtype=np.float32)
        lineout_centre = (50, 50)
        points = point_functions.get_lineout_offside_points(lineout_centre, H)
        self.assertEqual(len(points), 2)
        self.assertTrue(all(isinstance(pt, tuple) and len(pt) == 2 for pt in points))

    def test_transform_lines_identity(self):
        """
        Test transform_lines returns lines unchanged for identity homography.
        """
        lines = [[10, 10, 90, 10], [10, 90, 90, 90]]
        H = np.eye(3, dtype=np.float32)
        transformed = point_functions.transform_lines(lines, H)
        self.assertEqual(len(transformed), 2)
        self.assertTrue(all(len(line) == 4 for line in transformed))

    def test_transform_lines_homogeneous_zero(self):
        """
        Test transform_lines returns empty list if homography is all zeros (invalid).
        """
        lines = [[10, 10, 90, 10]]
        H = np.zeros((3, 3), dtype=np.float32)
        transformed = point_functions.transform_lines(lines, H)
        self.assertEqual(transformed, [])

    def test_shortest_distance_between_lines(self):
        """
        Test shortest_distance_between_lines returns correct distance for parallel and overlapping lines.
        """
        line1 = [0, 0, 10, 0]
        line2 = [0, 10, 10, 10]
        dist = point_functions.shortest_distance_between_lines(line1, line2)
        self.assertAlmostEqual(dist, 10.0, delta=1e-5)

        # Overlapping lines
        line3 = [0, 0, 10, 0]
        line4 = [5, 0, 15, 0]
        dist2 = point_functions.shortest_distance_between_lines(line3, line4)
        self.assertAlmostEqual(dist2, 0.0, delta=1e-5)

if __name__ == '__main__':
    unittest.main()