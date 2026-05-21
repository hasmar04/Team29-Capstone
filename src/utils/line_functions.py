"""
line_functions.py
-----------------
This module provides utility functions for geometric operations on lines in 2D space, including slope and intercept calculation, intersection finding, collinearity checks, and combining collinear lines. These functions are useful for computer vision tasks such as field line detection and analysis in sports analytics.

Functions:
---------------
- `get_slope`: Calculates the slope of a line given its endpoints.
- `get_y_intercept`: Calculates the y-intercept of a line given a point and its slope.
- `find_intersection_point`: Finds the intersection point of two lines in 2D space.
- `find_average_intersection_point`: Finds the average intersection point of all pairs of lines, excluding outliers.

Dependencies:
---------------
- NumPy (`numpy`)
- SciPy (`scipy.stats`)
"""

import numpy as np
from scipy.stats import zscore

def get_slope(line):
    """
    Calculates the slope of a line given its endpoints.

    Parameters:
        line (tuple): Four numerical values (x1, y1, x2, y2) representing two points on the line.

    Returns:
        float: The slope of the line, or float('inf') if the line is vertical.
    """
    x1, y1, x2, y2 = line
    try:
        return (y2 - y1) / (x2 - x1)
    except:
        return float('inf')


def get_y_intercept(x, y, m):
    """
    Calculates the y-intercept of a line given a point and its slope.

    Parameters:
        x (float): The x-coordinate of a point on the line.
        y (float): The y-coordinate of a point on the line.
        m (float): The slope of the line.

    Returns:
        float: The y-intercept of the line.
    """
    return (y - m * x)


def find_intersection_point(line_1, line_2):
    """
    Finds the intersection point of two lines in 2D space.

    Parameters:
        line_1 (tuple): Four numerical values (x1, y1, x2, y2) for the first line.
        line_2 (tuple): Four numerical values (x1, y1, x2, y2) for the second line.

    Returns:
        numpy.ndarray: 1D array with the x and y coordinates of the intersection point.

    Raises:
        numpy.linalg.LinAlgError: If the lines are parallel or coincident.
    """

    m_1 = get_slope(line_1)
    m_2 = get_slope(line_2)

    c_1 = get_y_intercept(line_1[0], line_1[1], m_1)
    c_2 = get_y_intercept(line_2[0], line_2[1], m_2)

    A = np.array([[-m_1, 1],
                  [-m_2, 1]])
    
    B = np.array([c_1, c_2])
    
    return np.linalg.solve(A, B)


def find_average_intersection_point(lines, z_threshold=1.0):
    """
    Finds the average intersection point of all pairs of lines, excluding outliers using Z-score filtering.

    Parameters:
        lines (list): List of tuples, each with four numerical values (x1, y1, x2, y2).
        z_threshold (float, optional): Z-score threshold for outlier removal. Defaults to 1.0.

    Returns:
        tuple: (x, y) coordinates of the average intersection point, or (None, None) if not found.
    """
    points = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            try:
                point = find_intersection_point(lines[i], lines[j])
                if not np.any(np.isnan(point)) and not np.any(np.isinf(point)):
                    points.append(point)
            except np.linalg.LinAlgError:
                # Skip if lines are parallel or coincident
                continue

    if not points:
        print("Warning: No intersection points found.")
        return None, None

    points = np.array(points)

    # Calculate Z-scores for x and y coordinates
    z_scores = np.abs(zscore(points, axis=0))

    # Identify non-outlier points (Z-scores below the threshold)
    non_outliers = (z_scores < z_threshold).all(axis=1)
    filtered_points = points[non_outliers]

    if len(filtered_points) == 0:
        filtered_points = points

    # Calculate the mean of the filtered points
    if len(filtered_points) > 0:
        x, y = np.mean(filtered_points, axis=0)
        if not np.isnan(x) and not np.isnan(y):
            return round(x), round(y)

    print("Warning: No valid intersection points after filtering.")
    return None, None
