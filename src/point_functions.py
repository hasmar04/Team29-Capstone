"""
point_functions.py
-----------------
This module provides functions and data structures for mapping rugby field lines and points in image and top-down coordinates. It includes a dictionary of standard field points, utilities for extracting and matching field features from images, computing homography matrices for perspective transformation, and transforming lines and points between image and field coordinates. The module is designed to support computer vision tasks such as field registration, offside line calculation, and visualization in rugby analytics.

Key Functions:
---------------
- `get_field_points`: Detects and returns intersection points between field lines and detected contours in the image.
- `get_homography_matrix`: Computes a homography matrix mapping detected image points to reference field points.
- `get_lineout_offside_points`: Calculates the offside points for a lineout using the homography matrix.
- `transform_lines`: Applies a homography transformation to a list of lines.

Dependencies:
---------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- matplotlib (`matplotlib.pyplot`)
- field_functions, general_functions, drawing_functions, line_functions, constants
"""

import cv2
import numpy as np

from constants import FIELD_POINTS_DICT

def get_homography_matrix(detected_points, point_locations):
    """
    Computes the homography matrix from detected image points to reference field points.

    Parameters
        detected_points (list of list of int): Detected points in the image, each as [x, y].
        point_locations (list of tuple): Keys to retrieve reference points from field_points_dict.

    Returns:
        numpy.ndarray: Homography matrix mapping detected points to reference points.
    """
    reference_points = []
    print(point_locations)

    for x, y in point_locations:
        reference_points.append(FIELD_POINTS_DICT[x][y])
    
    detected_points = np.array(detected_points, dtype=np.float32)
    reference_points = np.array(reference_points, dtype=np.float32)

    H, _ = cv2.findHomography(detected_points, reference_points, method=cv2.RANSAC)

    return H

def get_lineout_offside_points(lineout_centre, H):
    """
    Calculates the offside points for a lineout using the homography matrix.
    Maps the lineout centre to top-down coordinates, offsets by ±100 units, and maps back to image coordinates.

    Parameters
        lineout_centre (tuple): (x, y) coordinates of the lineout centre in the image.
        H (numpy.ndarray): Homography matrix from image to top-down field coordinates.

    Returns:
        list: Two (x, y) tuples for the left and right offside points in image coordinates.
    """
    # Convert lineout centre to homogeneous coordinates
    lineout_centre_homogeneous = np.array([lineout_centre[0], lineout_centre[1], 1], dtype=np.float32)
    # Apply the homography
    lineout_centre = np.dot(H, lineout_centre_homogeneous)
    # Normalize to get (x, y) coordinates
    lineout_centre /= lineout_centre[2]

    # Get the lineout offside points 10m from the lineout centre
    left_offside_point_top_down = np.array([lineout_centre[0] - 100, lineout_centre[1], 1], dtype=np.float32)
    right_offside_point_top_down = np.array([lineout_centre[0] + 100, lineout_centre[1], 1], dtype=np.float32)

    # Map the offside points back to the original image
    H_inv = np.linalg.inv(H)

    left_offside_point_homogeneous = np.dot(H_inv, left_offside_point_top_down)
    right_offside_point_homogeneous = np.dot(H_inv, right_offside_point_top_down)

    # Normalize to get (x, y) coordinates
    left_offside_point = left_offside_point_homogeneous[:2] / left_offside_point_homogeneous[2]
    right_offside_point = right_offside_point_homogeneous[:2] / right_offside_point_homogeneous[2]

    # Round the points to the nearest integer
    left_offside_point = np.round(left_offside_point).astype(int)
    right_offside_point = np.round(right_offside_point).astype(int)

    return [tuple(left_offside_point), tuple(right_offside_point)]


def transform_lines(lines, H):
    """
    Applies a homography transformation to a list of lines.

    Parameters
        lines (list of list of int): List of lines, each as [x1, y1, x2, y2].
        H (numpy.ndarray): Homography matrix to apply.

    Returns:
        list: List of transformed lines, each as [x1, y1, x2, y2].
    """
    transformed_lines = []

    for i, line in enumerate(lines):
        # Convert line endpoints to homogeneous coordinates
        p1 = np.array([line[0], line[1], 1], dtype=np.float32)
        p2 = np.array([line[2], line[3], 1], dtype=np.float32)

        # Apply the homography
        p1_trans = np.dot(H, p1)
        p2_trans = np.dot(H, p2)

        # Check for valid normalization
        if p1_trans[2] == 0 or p2_trans[2] == 0:
            print(f"Warning: Homogeneous coordinate is zero for line {i + 1}. Skipping normalization.")
            continue

        # Normalize to get (x, y) coordinates
        p1_trans /= p1_trans[2]
        p2_trans /= p2_trans[2]

        transformed_lines.append([round(p1_trans[0]), round(p1_trans[1]), round(p2_trans[0]), round(p2_trans[1])])

    return transformed_lines