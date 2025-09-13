"""
Unit tests for drawing_functions.py.

This module contains a suite of unit tests for the drawing utility functions. It tests the following functions:
    - draw_line: Draws a single line on an image.
    - draw_lines: Draws multiple lines on an image.
    - draw_points: Draws multiple points on an image.
    - draw_boxes: Draws bounding boxes (with optional annotations) on an image.

The tests cover correct output shape, non-trivial drawing, handling of single and multiple annotations, and error handling for invalid input.

Classes:
    TestDrawingFunctions (unittest.TestCase):
        setUp(): Initializes a blank test image.
        test_draw_line(): Tests drawing a single line.
        test_draw_lines(): Tests drawing multiple lines.
        test_draw_points(): Tests drawing multiple points.
        test_draw_boxes_single_annotation(): Tests drawing boxes with a single annotation.
        test_draw_boxes_multiple_annotations(): Tests drawing boxes with multiple annotations.
        test_draw_boxes_invalid_boxes(): Tests error handling for invalid box input.
        test_draw_points_invalid_points(): Tests error handling for invalid point input.
        test_draw_lines_invalid_lines(): Tests error handling for invalid line input.
        test_draw_line_invalid_image(): Tests error handling for invalid image input.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
import drawing_functions

class TestDrawingFunctions(unittest.TestCase):
    def setUp(self):
        """
        Set up a blank 100x100 black image for use in all drawing tests.
        """
        self.img = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_draw_line(self):
        """
        Test that draw_line draws a line on the image and returns an image of the same shape with nonzero pixels.
        """
        img_out = drawing_functions.draw_line(self.img, (10, 10), (90, 90), line_colour=(255, 0, 0), thickness=2, show_image=False)
        self.assertEqual(img_out.shape, self.img.shape)
        # Check that at least one pixel is not black
        self.assertTrue(np.any(img_out != 0))

    def test_draw_lines(self):
        """
        Test that draw_lines draws multiple lines and returns an image of the same shape with nonzero pixels.
        """
        lines = [(10, 10, 90, 10), (10, 20, 90, 20)]
        img_out = drawing_functions.draw_lines(self.img, lines, line_colour=(0, 255, 0), thickness=2, show_image=False)
        self.assertEqual(img_out.shape, self.img.shape)
        self.assertTrue(np.any(img_out != 0))

    def test_draw_points(self):
        """
        Test that draw_points draws points at specified locations and returns an image of the same shape with nonzero pixels.
        """
        points = [(20, 20), (80, 80)]
        img_out = drawing_functions.draw_points(self.img, points, point_colour=(0, 0, 255), radius=5, show_image=False)
        self.assertEqual(img_out.shape, self.img.shape)
        self.assertTrue(np.any(img_out != 0))

    def test_draw_boxes_single_annotation(self):
        """
        Test that draw_boxes draws multiple boxes with a single annotation and returns an image of the same shape with nonzero pixels.
        """
        boxes = [(10, 10, 30, 30), (40, 40, 60, 60)]
        img_out = drawing_functions.draw_boxes(self.img, boxes, outline_colour=(255, 0, 0), box_annotation='Test', show_image=False)
        self.assertEqual(img_out.shape, self.img.shape)
        self.assertTrue(np.any(img_out != 0))

    def test_draw_boxes_multiple_annotations(self):
        """
        Test that draw_boxes draws multiple boxes with different annotations and returns an image of the same shape with nonzero pixels.
        """
        boxes = [(10, 10, 30, 30), (40, 40, 60, 60)]
        annotations = ['A', 'B']
        img_out = drawing_functions.draw_boxes(self.img, boxes, outline_colour=(0, 255, 0), box_annotation=annotations, show_image=False)
        self.assertEqual(img_out.shape, self.img.shape)
        self.assertTrue(np.any(img_out != 0))

    def test_draw_boxes_invalid_boxes(self):
        """
        Test that draw_boxes raises a ValueError when given an invalid box format.
        """
        with self.assertRaises(ValueError):
            drawing_functions.draw_boxes(self.img, [(10, 10, 30)], show_image=False)

    def test_draw_points_invalid_points(self):
        """
        Test that draw_points raises a ValueError when given an invalid point format.
        """
        with self.assertRaises(ValueError):
            drawing_functions.draw_points(self.img, [(10,)], show_image=False)

    def test_draw_lines_invalid_lines(self):
        """
        Test that draw_lines raises a ValueError when given an invalid line format.
        """
        with self.assertRaises(ValueError):
            drawing_functions.draw_lines(self.img, [(10, 10, 30)], show_image=False)

    def test_draw_line_invalid_image(self):
        """
        Test that draw_line raises a ValueError when given an invalid image input (None).
        """
        with self.assertRaises(ValueError):
            drawing_functions.draw_line(None, (0, 0), (1, 1), show_image=False)

if __name__ == '__main__':
    unittest.main()
