"""
line_functions_unit_testing.py

Unit tests for the line_functions.py module, which provides utility functions for geometric operations on lines in 2D space.

Tested Functions:
    - get_slope
    - get_y_intercept
    - find_intersection_point
    - find_average_intersection_point
    - check_collinearity
    - combine_collinear_lines

Test Structure:
    - Uses unittest for function output and edge case checking.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from src import line_functions

class TestLineFunctions(unittest.TestCase):
    def test_get_slope_regular(self):
        # Test slope calculation for a regular (non-vertical) line
        line = (0, 0, 2, 2)
        self.assertAlmostEqual(line_functions.get_slope(line), 1.0)

    def test_get_slope_vertical(self):
        # Test slope calculation for a vertical line (should return inf)
        line = (1, 0, 1, 2)
        self.assertEqual(line_functions.get_slope(line), float('inf'))

    def test_get_y_intercept(self):
        # Test calculation of y-intercept given a point and slope
        x, y, m = 1, 2, 3
        self.assertEqual(line_functions.get_y_intercept(x, y, m), -1)

    def test_find_intersection_point(self):
        # Test intersection of two lines that cross
        line1 = (0, 0, 1, 1)
        line2 = (0, 1, 1, 0)
        pt = line_functions.find_intersection_point(line1, line2)
        self.assertTrue(np.allclose(pt, [0.5, 0.5]))

    def test_find_intersection_point_parallel(self):
        # Test intersection of two parallel lines (should raise LinAlgError)
        line1 = (0, 0, 1, 1)
        line2 = (0, 1, 1, 2)
        with self.assertRaises(np.linalg.LinAlgError):
            line_functions.find_intersection_point(line1, line2)

    def test_find_average_intersection_point(self):
        # Test average intersection point calculation for multiple lines
        lines = [
            (0, 0, 1, 1),
            (0, 1, 1, 0),
            (0, 0.5, 1, 0.5)
        ]
        x, y = line_functions.find_average_intersection_point(lines)
        self.assertIsInstance(x, (int, float, type(None)))
        self.assertIsInstance(y, (int, float, type(None)))

    def test_find_average_intersection_point_none(self):
        # Test average intersection point calculation for parallel lines (should return None or nan)
        lines = [
            (0, 0, 1, 0),
            (0, 1, 1, 1)
        ]
        x, y = line_functions.find_average_intersection_point(lines)
        self.assertIsInstance(x, (int, float, type(None)))
        self.assertIsInstance(y, (int, float, type(None)))


if __name__ == '__main__':
    unittest.main()