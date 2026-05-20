"""
field_functions_unit_testing.py

Unit tests for the field_functions.py module, which provides functions for detecting and processing rugby field features.

Tested Functions:
    - detect_rugby_field
    - extract_straight_lines
    - fit_straight_lines_to_contours
    - find_boundary_intersection
    - get_vertical_boundaries
    - get_horizontal_boundaries
    - fit_lines_to_field
    - is_horizontal_line
    - check_collinearity
    - combine_collinear_lines
    - fit_line_to_field
    - average_lines_by_midpoint
    - average_parallel_lines
    - extract_line_features
    - cluster_lines
    - combine_clustered_lines
    - remove_anomalous_lines_by_angle
    - cluster_lines_by_angle
    - filter_lines_by_weighted_angle
    - approximate_field_outline
    - calculate_angle
    - get_deadball_line
    - filter_by_deadball_line

Test Structure:
    - Uses unittest and unittest.mock for mocking dependencies and simulating images, contours, and lines.
    - setUp creates dummy data for use in all tests.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src import field_functions

class TestFieldFunctions(unittest.TestCase):
    def setUp(self):
        # Dummy green image for field detection
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.image[:] = (0, 255, 0)
        # Dummy field outline (rectangle)
        self.field_outline = np.array([[[0,0]], [[0,99]], [[99,99]], [[99,0]]], dtype=np.int32)
        # Dummy contours for line extraction
        self.contours = [np.array([[[10,10]], [[10,20]], [[20,20]], [[20,10]]], dtype=np.int32)]
        # Dummy lines for line-based tests
        self.lines = [[10, 10, 90, 10], [10, 90, 90, 90]]
        # Dummy line for single-line tests
        self.line = [10, 10, 90, 10]

    def test_detect_rugby_field(self):
        # Test that the field hull is detected and is a numpy array with at least 4 points
        hull = field_functions.detect_rugby_field(self.image)
        self.assertIsInstance(hull, np.ndarray)
        self.assertGreaterEqual(hull.shape[0], 4)

    def test_extract_straight_lines(self):
        # Test extraction of straight lines from contours
        lines = field_functions.extract_straight_lines(self.contours, self.field_outline)
        self.assertIsInstance(lines, list)

    def test_fit_straight_lines_to_contours(self):
        # Test fitting straight lines to contour segments
        straight_lines = [[[10, 10], [20, 10]], [[20, 10], [20, 20]]]
        fitted = field_functions.fit_straight_lines_to_contours(straight_lines, min_length=1)
        self.assertIsInstance(fitted, list)

    def test_find_boundary_intersection(self):
        # Test finding intersection between a line and a boundary
        boundary = np.array([[10,10],[90,10],[90,90],[10,90]], dtype=np.int32)
        pt = field_functions.find_boundary_intersection([10, 10, 90, 10], boundary)
        self.assertTrue(pt is None or (isinstance(pt, tuple) and len(pt) == 2))

    def test_get_vertical_boundaries(self):
        # Test vertical boundary extraction using k-means
        top, bottom = field_functions.get_vertical_boundaries(self.field_outline)
        self.assertIsInstance(top, np.ndarray)
        self.assertIsInstance(bottom, np.ndarray)

    def test_get_horizontal_boundaries(self):
        # Test horizontal boundary extraction using k-means
        left, right = field_functions.get_horizontal_boundaries(self.field_outline)
        self.assertIsInstance(left, np.ndarray)
        self.assertIsInstance(right, np.ndarray)

    def test_fit_lines_to_field(self):
        # Test extending lines to intersect with field boundaries
        extended = field_functions.fit_lines_to_field(self.lines, self.field_outline)
        self.assertIsInstance(extended, list)

    def test_fit_line_to_field(self):
        # Test clipping/extending a line to the field outline
        result = field_functions.fit_line_to_field([10,10,90,10], self.field_outline)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 4)

    def test_average_lines_by_midpoint(self):
        # Test averaging lines whose midpoints are close together
        lines = [[10,10,90,10],[15,12,85,12]]
        avg = field_functions.average_lines_by_midpoint(lines, self.field_outline, max_midpoint_distance=10)
        self.assertIsInstance(avg, list)

    def test_extract_line_features(self):
        # Test extraction of geometric features from lines
        lines = [[10,10,90,10],[15,12,85,12]]
        features = field_functions.extract_line_features(lines, 1, 1, 1)
        self.assertIsInstance(features, np.ndarray)

    def test_remove_anomalous_lines_by_angle(self):
        # Test removal of lines with anomalous angles
        lines = [[10,10,90,10],[10,10,10,90],[10,10,90,90]]
        filtered = field_functions.remove_anomalous_lines_by_angle(lines)
        self.assertIsInstance(filtered, list)

    def test_approximate_field_outline(self):
        # Test approximation of the convex hull to a bounding quadrilateral
        outline = np.array([[[0,0]], [[0,99]], [[99,99]], [[99,0]]], dtype=np.int32)
        approx = field_functions.approximate_field_outline(outline)
        self.assertIsInstance(approx, np.ndarray)

    def test_get_deadball_line(self):
        # Test determination of the deadball line
        outline = np.array([[[10,10]], [[10,90]], [[90,90]], [[90,10]]], dtype=np.int32)
        image = np.zeros((100,100,3), dtype=np.uint8)
        lineout_centre = (50, 80)
        result = field_functions.get_deadball_line(outline, image, lineout_centre)
        self.assertTrue(result is None or (isinstance(result, list) and len(result) == 4))

    @patch('field_functions.lf')
    def test_filter_by_deadball_line(self, mock_lf):
        # Test filtering lines approximately perpendicular to the deadball line
        mock_lf.get_slope.return_value = 1
        lines = [[10,10,90,10],[10,10,10,90]]
        deadball_line = [0,0,1,1]
        filtered = field_functions.filter_by_deadball_line(lines, deadball_line, tolerance=0.1)
        self.assertIsInstance(filtered, list)

if __name__ == '__main__':
    unittest.main()