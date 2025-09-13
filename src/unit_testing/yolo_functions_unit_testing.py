"""
Unit tests for yolo_functions.py.

This module contains a suite of unit tests for the YOLO utility functions used in rugby analytics and computer vision tasks. It tests the following functions:
    - load_model: Loads a YOLO model from a file path.
    - perform_inference: Runs inference on an image or video using a YOLO model.
    - show_annotated_file: Displays annotated frames or images from YOLO inference results.

The tests use unittest and unittest.mock to patch dependencies and simulate different scenarios, including successful and failed model loading, inference, and display operations.

Classes:
    TestYOLOFunctions (unittest.TestCase):
        test_load_model_success: Tests successful model loading.
        test_load_model_fail: Tests model loading failure and exit handling.
        test_perform_inference_success: Tests successful inference and result streaming.
        test_perform_inference_fail: Tests inference failure and error raising.
        test_show_annotated_file: Tests display of annotated frames and window cleanup.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import yolo_functions

class TestYOLOFunctions(unittest.TestCase):
    @patch('yolo_functions.YOLO')
    def test_load_model_success(self, mock_yolo):
        """
        Test that load_model successfully loads and returns a YOLO model when no exception is raised.
        """
        mock_yolo.return_value = 'mock_model'
        model = yolo_functions.load_model('some_path.pt')
        self.assertEqual(model, 'mock_model')

    @patch('yolo_functions.YOLO', side_effect=Exception('fail'))
    @patch('sys.exit')
    def test_load_model_fail(self, mock_exit, mock_yolo):
        """
        Test that load_model calls sys.exit when model loading fails (raises an exception).
        """
        yolo_functions.load_model('bad_path.pt')
        mock_exit.assert_called_once()

    @patch('yolo_functions.YOLO')
    def test_perform_inference_success(self, mock_yolo):
        """
        Test that perform_inference returns a generator yielding the expected results when inference succeeds.
        """
        mock_model = MagicMock()
        mock_model.predict.return_value = iter(['result1', 'result2'])
        results = yolo_functions.perform_inference('file', mock_model, show_output=False, conf=0.5)
        self.assertEqual(list(results), ['result1', 'result2'])

    @patch('yolo_functions.YOLO')
    def test_perform_inference_fail(self, mock_yolo):
        """
        Test that perform_inference raises SystemError when the model's predict method raises an exception.
        """
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception('fail')
        with self.assertRaises(SystemError):
            list(yolo_functions.perform_inference('file', mock_model))

    @patch('cv2.imshow')
    @patch('cv2.waitKey', side_effect=lambda *args, **kwargs: ord('q'))
    @patch('cv2.destroyAllWindows')
    def test_show_annotated_file(self, mock_destroy, mock_waitkey, mock_imshow):
        """
        Test that show_annotated_file displays annotated frames and calls window cleanup functions.
        """
        dummy_result = MagicMock()
        dummy_result.plot.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        results = iter([dummy_result, dummy_result])
        yolo_functions.show_annotated_file(results)
        self.assertTrue(mock_imshow.called)
        self.assertTrue(mock_destroy.called)

if __name__ == '__main__':
    unittest.main()