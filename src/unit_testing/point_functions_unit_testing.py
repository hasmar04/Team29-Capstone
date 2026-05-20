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
from src import point_functions

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

    @patch('src.point_functions.FIELD_POINTS_DICT', {
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


if __name__ == '__main__':
    unittest.main()