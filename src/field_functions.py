"""
field_functions.py
-----------------
This module provides a suite of functions for detecting, extracting, and processing rugby field features 
from images. It includes algorithms for field detection, line extraction, boundary finding, line fitting, 
collinearity and parallelism checks, and clustering of field lines. The functions are designed to support 
rugby analytics workflows such as homography, offside detection, and field mapping.

Functions:
---------------
- `detect_rugby_field`: Detects the largest green area in an image, assumed to be the rugby field, and returns its convex hull.
- `extract_straight_lines`: Extracts straight line segments from contours that are mostly inside the field outline.
- `fit_straight_lines_to_contours`: Fits straight lines to given contour segments and returns line endpoints.
- `find_boundary_intersection`: Finds the intersection point between a line and a boundary.
- `get_vertical_boundaries`: Uses k-means clustering to find the top and bottom boundaries of the field.
- `get_horizontal_boundaries`: Uses k-means clustering to find the left and right boundaries of the field.
- `fit_lines_to_field`: Extends given lines to intersect with the field boundaries.
- `fit_line_to_field`: Clips or extends a line to the field outline.
- `average_lines_by_midpoint`: Averages lines whose midpoints are close together.
- `extract_line_features`: Extracts geometric features from lines for clustering.
- `remove_anomalous_lines_by_angle`: Removes lines with anomalous angles.
- `approximate_field_outline`: Approximates the convex hull of the field outline to a minimum area bounding quadrilateral.
- `get_deadball_line`: Determines the deadball line based on field outline and lineout centre.
- `filter_by_deadball_line`: Filters lines that are approximately perpendicular to the deadball line.
- `visualise_step`: Helper function to visualise each step.
- `get_field_lines`: Extracts and processes the field lines from an image of a rugby field.

Dependencies:
---------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- scikit-learn (`KMeans`, `DBSCAN`)
- matplotlib (for optional visualization)
- general_functions, line_functions
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from src import general_functions as general
from src import line_functions as lf

FIELD_DEBUG = False


def _field_debug(message):
    """Print field extraction diagnostics only when FIELD_DEBUG is enabled."""
    if FIELD_DEBUG:
        print(message)


def detect_rugby_field(image):
    """
    Detects the rugby field in an image by identifying the largest green area. Processes the input image to isolate green regions (using HSV color space), 
    applies morphological operations, and finds the largest contour. Returns the convex hull of the largest contour, clipped to image boundaries.

    Parameters:
        image (np.ndarray): The input image (BGR format).
    Returns:
        np.ndarray: 2D array of points representing the convex hull of the largest green area (the field), clipped to image boundaries.
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define green color range for the field
    lower_green = np.array([35, 55, 55])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green areas
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _field_debug(f"[FIELD] contours found: {len(contours)}")

    # Draw the largest contour (assumed to be the field)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        hull = cv2.convexHull(largest_contour)

        # Get image dimensions
        height, width = mask.shape

        # Clip the convex hull points to the image boundaries
        return np.clip(hull, [0, 0], [width - 1, height - 1])

def extract_straight_lines(contours, field_outline, exclusion_box=None, threshold=0.35):
    """
    Extracts straight line segments from contours that are mostly inside the field outline.

    This function iterates through each contour, checks what fraction of its points are inside the provided field outline, and only processes contours that meet the threshold. It then approximates each qualifying contour to reduce complexity and splits it into line segments. If an exclusion box is provided, segments that intersect this box are skipped. The result is a list of straight line segments, each represented by two endpoints.

    Parameters:
        contours (list): List of contours detected in the image (each contour is a numpy array of points).
        field_outline (np.ndarray): Convex hull of the field outline.
        exclusion_box (tuple or list or None): Optional. A tuple/list of four values (x1, y1, x2, y2) defining a rectangular region to exclude segments from. If None, no exclusion is applied.
        threshold (float, optional): Minimum fraction (0-1) of contour points that must be inside the field outline for the contour to be processed. Defaults to 0.8.

    Returns:
        list: List of straight line segments, each as a list of two points [[x1, y1], [x2, y2]].
    """
    straight_lines = []

    for cnt in contours:
        # Reject tiny noisy contours early.
        contour_area = cv2.contourArea(cnt)

        if contour_area < 40:
            continue

        """# Check if the contour is inside the field outline
        inside_count = 0
        for point in cnt:
            x, y = map(int, point[0])  # Extract the x, y coordinates

            # Check if the point is inside the field outline
            if cv2.pointPolygonTest(field_outline, (x, y), False) >= 0:
                inside_count += 1
        inside_ratio = inside_count / len(cnt)

        print(
            "[FIELD] contour area:",
            round(contour_area, 1),
            "inside ratio:",
            round(inside_ratio, 2),
            "points:",
            len(cnt)
        )

        # Skip this contour if less than the threshold of its points are inside the field outline
        if inside_ratio < threshold:
            continue"""
        
        _field_debug(
            f"[FIELD] contour area: {round(contour_area, 1)} points: {len(cnt)}"
        )

        # Approximate the contour to simplify it
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        for i in range(len(approx) - 1):
            x1, y1 = map(int, approx[i][0])
            x2, y2 = map(int, approx[i + 1][0])
            
            if exclusion_box is not None and len(exclusion_box) == 4:
                box_x1, box_y1, box_x2, box_y2 = exclusion_box

                if (box_x1 <= x1 <= box_x2 and box_y1 <= y1 <= box_y2) or \
                    (box_x1 <= x2 <= box_x2 and box_y1 <= y2 <= box_y2):
                    continue

            straight_lines.append([[x1, y1], [x2, y2]])
        
    return straight_lines

def fit_straight_lines_to_contours(straight_lines, min_length=5):
    """
    Fits straight lines to the given contour segments and returns the line endpoints.

    Parameters:
        straight_lines (list): List of straight line segments (each as a list of two points).
        min_length (float): Minimum length of the line segment to be included.
    Returns:
        list: List of fitted straight line segments as tuples (x1, y1, x2, y2).
    """
    contour_lines = []

    for line in straight_lines:
        x_coords = [point[0] for point in line]
        y_coords = [point[1] for point in line]

        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        line_np = np.array(line, dtype=np.float32)

        # Fit a straight line to the contour points
        [vx, vy, x0, y0] = cv2.fitLine(line_np, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the slope and intercept
        if vx == 0:  # Handle vertical lines
            slope = float('inf')
        else:
            slope = vy / vx

        intercept = y0 - slope * x0

        # Calculate the endpoints of the line within the bounding box
        x_min, x_max = min(x_coords), max(x_coords)
        y1 = round((slope * x_min + intercept).item())
        y2 = round((slope * x_max + intercept).item())

        # Ensure the endpoints are within the bounding box
        y1 = max(y_min, min(y_max, y1))
        y2 = max(y_min, min(y_max, y2))

        # Calculate the length of the line
        line_length = np.sqrt((x_max - x_min) ** 2 + (y2 - y1) ** 2)

        # Reject tiny lines.
        if line_length <= min_length:
            continue

        # Calculate line angle.
        angle = abs(np.degrees(np.arctan2(y2 - y1, x_max - x_min)))

        # Reject near-vertical junk lines.
        if angle > 80:
            continue

        contour_lines.append((x_min, y1, x_max, y2))

    return contour_lines

def find_boundary_intersection(line, boundary):
    """
    Finds the intersection point between a line and a boundary polyline.

    Parameters:
        line (tuple): Line segment as (x1, y1, x2, y2).
        boundary (np.ndarray): Points representing the boundary polyline.
    Returns:
        (tuple or None): Intersection point as (x, y), or None if no intersection exists.
    """
    x1, y1, x2, y2 = line
    for i in range(len(boundary) - 1):
        x3, y3 = boundary[i]
        x4, y4 = boundary[i + 1]

        # Solve for intersection between the two lines
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            continue  # Lines are parallel

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        # Check if the intersection point is within the boundary segment
        if min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4):
            return int(px), int(py)

    return None

def get_vertical_boundaries(field_outline):
    """
    Uses k-means clustering to find the average top and bottom y-coordinates and returns the points for the top and bottom boundaries.
    
    Parameters:
        field_outline (np.ndarray): Convex hull of the field outline.
    Returns:
        tuple: Two np.ndarrays for the top and bottom boundaries.
    """
    # Extract y-coordinates from the field outline
    points = field_outline[:, 0, :]
    y_coords = points[:, 1].reshape(-1, 1)

    # Apply k-means clustering with k=2
    kmeans = KMeans(n_clusters=2, random_state=0).fit(y_coords)
    labels = kmeans.labels_
    cluster_centres = kmeans.cluster_centers_

    # Identify the cluster corresponding to the top boundary (smallest y-coordinate)
    top_cluster_label = np.argmin(cluster_centres)
    top_boundary = points[labels == top_cluster_label]

    bottom_boundary = points[labels != top_cluster_label]

    return np.array(top_boundary, dtype=np.int32), np.array(bottom_boundary, dtype=np.int32)

def get_horizontal_boundaries(field_outline):
    """
    Uses k-means clustering to find the average left and right x-coordinates and returns the points for the left and right boundaries.

    Parameters:
        field_outline (np.ndarray): Convex hull of the field outline.
    Returns:
        tuple: Two np.ndarrays for the left and right boundaries.
    """
    # Extract x-coordinates from the field outline
    points = field_outline[:, 0, :]
    x_coords = points[:, 0].reshape(-1, 1)

    # Apply k-means clustering with k=2
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x_coords)
    labels = kmeans.labels_
    cluster_centres = kmeans.cluster_centers_

    # Identify the cluster corresponding to the left boundary (smallest x-coordinate)
    left_cluster_label = np.argmin(cluster_centres)
    left_boundary = points[labels == left_cluster_label]

    right_boundary = points[labels != left_cluster_label]

    return np.array(left_boundary, dtype=np.int32), np.array(right_boundary, dtype=np.int32)

def extend_line_to_image_bounds(line, image_shape):
    """
    Extends a line to the image boundaries.
    """
    x1, y1, x2, y2 = line

    # Force NumPy scalars into Python floats.
    x1 = float(np.asarray(x1).item())
    y1 = float(np.asarray(y1).item())
    x2 = float(np.asarray(x2).item())
    y2 = float(np.asarray(y2).item())

    height, width = image_shape[:2]

    # Handle vertical lines.
    if abs(x2 - x1) < 1e-6:
        return [int(x1), 0, int(x1), height - 1]

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    left_y = int(intercept)
    _field_debug(f"[FIELD] line slope={slope:.4f} intercept={intercept:.2f}")
    right_y = int((slope * (width - 1)) + intercept)

    return [0, left_y, width - 1, right_y]

def fit_lines_to_field(contour_lines, field_outline):
    """
    Extends the given lines to intersect with the field boundaries (top/bottom or left/right).

    Parameters:
        contour_lines (list): List of line segments as (x1, y1, x2, y2).
        field_outline (np.ndarray): Convex hull of the field outline.
    Returns:
        list: List of extended line segments as (x1, y1, x2, y2).
    """
    top_boundary, bottom_boundary = get_vertical_boundaries(field_outline)

    extended_lines = []

    for x1, y1, x2, y2 in contour_lines:
        # Find intersections with the top and bottom boundaries
        top_intersection = find_boundary_intersection((x1, y1, x2, y2), top_boundary)
        bottom_intersection = find_boundary_intersection((x1, y1, x2, y2), bottom_boundary)

        # If both intersections are found, add the extended line
        if top_intersection and bottom_intersection:
            extended_lines.append((*top_intersection, *bottom_intersection))
        
        elif top_intersection or bottom_intersection:
            left_boundary, right_boundary = get_horizontal_boundaries(field_outline)

            left_intersection = find_boundary_intersection((x1, y1, x2, y2), left_boundary)
            right_intersection = find_boundary_intersection((x1, y1, x2, y2), right_boundary)

            if top_intersection and left_intersection:
                extended_lines.append((*top_intersection, *left_intersection))
            elif top_intersection and right_intersection:
                extended_lines.append((*top_intersection, *right_intersection))
            elif bottom_intersection and left_intersection:
                extended_lines.append((*bottom_intersection, *left_intersection))
            elif bottom_intersection and right_intersection:
                extended_lines.append((*bottom_intersection, *right_intersection))
    
    return extended_lines

def fit_line_to_field(line, field_outline):
    """
    Fits a line to the field outline by extending it to intersect with the field boundaries in both directions.

    Parameters:
        line (list): Line as [x1, y1, x2, y2].
        field_outline (np.ndarray): Convex hull of the field outline.
    Returns:
        list: Clipped/extended line as [x1, y1, x2, y2].
    """
    x1, y1, x2, y2 = line
    intersections = []

    for i in range(len(field_outline)):
        x3, y3 = field_outline[i][0]
        x4, y4 = field_outline[(i + 1) % len(field_outline)][0]

        x1, y1, x2, y2 = map(np.float64, [x1, y1, x2, y2])
        x3, y3, x4, y4 = map(np.float64, [x3, y3, x4, y4])

        # Solve for intersection between the line and the field edge
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            continue  # Lines are parallel

        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom

        # Check if the intersection point is within both segments
        if min(x3, x4) <= px <= max(x3, x4) and min(y3, y4) <= py <= max(y3, y4):
            intersections.append((px, py))

    if len(intersections) >= 2:
        # Sort intersections by distance from the original line's first point
        intersections = sorted(intersections, key=lambda p: np.sqrt((p[0] - x1)**2 + (p[1] - y1)**2))
        return [int(intersections[0][0]), int(intersections[0][1]), int(intersections[-1][0]), int(intersections[-1][1])]

    return line

def average_lines_by_midpoint(lines, field_outline, max_midpoint_distance=30):
    """
    Averages lines if their midpoints are close enough and ensures the lines stop at the field outline.

    Parameters:
        lines (list): List of lines as [x1, y1, x2, y2].
        field_outline (np.ndarray): Convex hull of the field outline.
        max_midpoint_distance (float): Max distance between midpoints to consider for averaging.
    Returns:
        list: List of averaged line segments as [x1, y1, x2, y2].
    """
    averaged_lines = []
    used_indices = set()

    for i, line1 in enumerate(lines):
        if i in used_indices:
            continue

        x1, y1, x2, y2 = line1
        midpoint1 = ((x1 + x2) / 2, (y1 + y2) / 2)

        group = [line1]
        for j, line2 in enumerate(lines):
            if j in used_indices or i == j:
                continue

            x3, y3, x4, y4 = line2
            midpoint2 = ((x3 + x4) / 2, (y3 + y4) / 2)

            # Check if the midpoints are close
            distance = np.sqrt((midpoint1[0] - midpoint2[0])**2 + (midpoint1[1] - midpoint2[1])**2)
            if distance <= max_midpoint_distance:
                group.append(line2)
                used_indices.add(j)

        # Average the group of lines
        all_points = []
        for line in group:
            all_points.append((line[0], line[1]))
            all_points.append((line[2], line[3]))

        # Fit a line to all points in the group
        points_np = np.array(all_points, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(points_np, cv2.DIST_L2, 0, 0.01, 0.01)

        vx = vx.item()
        vy = vy.item()
        x0 = x0.item()
        y0 = y0.item()

        # Extend the line to its maximum extent
        t_min = float('inf')
        t_max = float('-inf')
        for x, y in all_points:
            t = (x - x0) / vx if vx != 0 else (y - y0) / vy
            t_min = min(t_min, t)
            t_max = max(t_max, t)

        x1 = int(x0 + t_min * vx)
        y1 = int(y0 + t_min * vy)
        x2 = int(x0 + t_max * vx)
        y2 = int(y0 + t_max * vy)

        # Clip the line to the field outline
        clipped_line = fit_line_to_field([x1, y1, x2, y2], field_outline)
        averaged_lines.append(clipped_line)
        used_indices.add(i)

    return averaged_lines

def extract_line_features(lines, angle_weight, length_weight, midpoint_weight):
    """
    Extracts features for each line and applies a weighting factor to the angle.

    Parameters:
        lines (list): List of lines as [x1, y1, x2, y2].
        angle_weight (float): Weight for the angle feature.
        length_weight (float): Weight for the length feature.
        midpoint_weight (float): Weight for the midpoint feature.
    Returns:
        np.ndarray: Array of features for each line.
    """
    features = []
    max_length = max(np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2) for line in lines)

    for x1, y1, x2, y2 in lines:
        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) * angle_weight
        # Calculate midpoint
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2) * midpoint_weight
        # Calculate length
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * length_weight / max_length
        # Combine features (weighted angle, midpoint x, midpoint y, length)
        features.append([angle, midpoint[0], midpoint[1], length])
    return np.array(features)

def remove_anomalous_lines_by_angle(lines, angle_tolerance=12):
    """
    Keeps the dominant cluster of similarly-oriented lines.

    This is more robust than mean/stddev filtering because sports footage
    often contains multiple competing angle groups from stands, signage,
    shadows, and field markings.
    """

    if not lines:
        return []

    angles = []

    # Calculate angles.
    for line in lines:
        x1, y1, x2, y2 = line

        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle = angle % 180

        angles.append(angle)

    angles = np.array(angles)

    best_count = 0
    best_angle = None

    # Find the dominant angle cluster.
    for angle in angles:
        count = np.sum(np.abs(angles - angle) < angle_tolerance)

        if count > best_count:
            best_count = count
            best_angle = angle

    filtered_lines = []

    # Keep only lines near the dominant angle.
    for line, angle in zip(lines, angles):
        if abs(angle - best_angle) < angle_tolerance:
            filtered_lines.append(line)

    _field_debug(f"[FIELD] dominant angle cluster: {round(best_angle, 2)}")
    _field_debug(f"[FIELD] kept lines: {len(filtered_lines)} / {len(lines)}")

    return filtered_lines

def approximate_field_outline(field_outline):
    """
    Approximates the convex hull of the field outline to a minimum area bounding quadrilateral.

    Parameters:
        field_outline (np.ndarray): Convex hull of the field outline as a 2D array of points.

    Returns:
        np.ndarray: A 2D array of 4 points representing the minimum area bounding quadrilateral.
    """

    hull = np.squeeze(field_outline) # Remove unnecessary dimensions
    approx = cv2.approxPolyDP(hull, 8, True)
    return np.squeeze(approx)


def get_deadball_line(field_outline, image, lineout_centre):
    """
    Determines the deadball line on a rugby field based on the field outline, 
    image dimensions, and the lineout center's position. The deadball line is 
    typically located at the top or bottom of the field.

    The function approximates the field outline, identifies the topmost point, 
    and evaluates its connected points. Based on the lineout center's position, 
    it determines whether the deadball line is on the top or bottom side and 
    selects the appropriate connected points.

    Parameters:
        field_outline (list of tuples): Coordinates representing the field outline.
        image (numpy.ndarray): Image of the field, used to determine dimensions.
        lineout_centre (tuple): Center point of the lineout (x, y).

    Returns:
        list: Coordinates [x1, y1, x2, y2] of the deadball line, or None if it 
        cannot be determined.
    """

    imsize = (image.shape[1], image.shape[0])
    
    field_outline = approximate_field_outline(field_outline)

    top_point_index = min(range(len(field_outline)), key=lambda i: field_outline[i][1])
    top_point = field_outline[top_point_index]

    connected_points = [field_outline[top_point_index - 1], field_outline[top_point_index + 1] if len(field_outline) > top_point_index + 1 else field_outline[0]]

    if top_point[0] == 0 or top_point[0] == imsize[0] - 1 or top_point[1] == 0 or top_point[1] == imsize[1] - 1:
        # If the top point is at the edge of the image, cannot determine a deadball line
        return None
    
    line_1 = [top_point[0], top_point[1], connected_points[0][0], connected_points[0][1]]
    line_2 = [top_point[0], top_point[1], connected_points[1][0], connected_points[1][1]]

    if general.calculate_angle(line_1, line_2) > 170:
        return None
    
    if lineout_centre[1] > imsize[1] // 2:
        # Bottom side of the field
        if lineout_centre[0] < imsize[0] // 2:
            # Left side of the field
            if connected_points[0][0] < connected_points[1][0]:
                deadball_line = [connected_points[0][0], connected_points[0][1], top_point[0], top_point[1]]
            else:
                deadball_line = [connected_points[1][0], connected_points[1][1], top_point[0], top_point[1]]
        
        else:
            # Right side of the field
            if connected_points[0][0] > connected_points[1][0]:
                deadball_line = [connected_points[0][0], connected_points[0][1], top_point[0], top_point[1]]
            else:
                deadball_line = [connected_points[1][0], connected_points[1][1], top_point[0], top_point[1]]

        return deadball_line
    else:
        return None  # If the lineout center is not on the bottom half of the field, return None

def filter_by_deadball_line(lines, deadball_line, tolerance=0.1):
    """
    Filters a list of lines by removing those that are approximately perpendicular 
    to a given deadball line within a specified tolerance.
    Parameters:
        lines (list): A list of lines, where each line is represented in a format 
                      compatible with the `lf.get_slope` function.
        deadball_line: The reference line used to determine perpendicularity, 
                       represented in a format compatible with the `lf.get_slope` function.
        tolerance (float, optional): The tolerance value for determining approximate 
                                     perpendicularity. Defaults to 0.1.
    Returns:
        list: A filtered list of lines that are not approximately perpendicular 
              to the deadball line.
    """
    db_slope = lf.get_slope(deadball_line)

    filtered_lines = []
    for line in lines:
        slope = lf.get_slope(line)
        if slope is not None:

            # Skip any lines that are approximately perpendicular to the deadball line
            if abs((slope * db_slope) + 1) < tolerance:
                continue
           
            filtered_lines.append(line)

    return filtered_lines

def visualise_step(image, title, lines=None):
    """
    Visualises an image with an optional overlay of lines and a title.

    Parameters:
    image : array-like
        The image data to be displayed. Typically a 2D or 3D array.
    title : str
        The title to be displayed above the image.
    lines : list of tuples, optional
        A list of line coordinates to overlay on the image. Each line is 
        represented as a tuple (x1, y1, x2, y2), where (x1, y1) and (x2, y2) 
        are the start and end points of the line. Default is None.

    Returns:
    None
        Displays the image with the specified title and optional lines.
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(title)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            plt.plot([x1, x2], [y1, y2], 'g-', linewidth=2)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def create_field_line_mask(image):
    """
    Creates a mask for likely white rugby field markings.

    This is more reliable than simple grayscale thresholding because it looks
    for low-saturation, high-value pixels inside the field image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    # Field markings are usually bright and low saturation.
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 135]),
        np.array([180, 90, 255])
    )

    # Remove small noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connect broken painted line segments.
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return white_mask

def get_field_lines(im_path, lineout_centre=None, exclusion_box=None, thresh=165, visualise_steps=False, is_path=True):
    """
    Extracts and processes the field lines from an image of a rugby field.

    Parameters:
        im_path (str or numpy.ndarray): The path to the image file or the image itself.
            If `is_path` is True, this should be a string representing the file path.
            Otherwise, it should be a numpy array representing the image.
        lineout_centre (tuple, optional): A tuple (x, y) representing the center point of the lineout.
            Used to determine the deadball line. Defaults to None.
        exclusion_box (tuple or list or None, optional): A tuple/list of four values (x1, y1, x2, y2) defining
            a rectangular region. Line segments whose endpoints fall within this box are excluded. If None, no exclusion is applied. Defaults to None.
        thresh (int, optional): Threshold value for binary thresholding. Defaults to 165.
        visualise_steps (bool, optional): If True, visualizes intermediate processing steps. Defaults to False.
        is_path (bool, optional): Indicates whether `im_path` is a file path (True) or an image array (False). Defaults to True.

    Returns:
        tuple:
            - final_lines (list of tuples): A list of tuples representing the final processed field lines.
              Each tuple contains four integers (x1, y1, x2, y2) representing the endpoints of a line.
            - field_outline (np.ndarray): An array of points representing the detected field outline.

    Notes:
        - The function performs several image processing steps tailored for rugby field images, including resizing, thresholding, contour detection,
          line extraction from contours, deadball line filtering, line extension to field boundaries, anomaly removal, and line averaging.
        - If `lineout_centre` is provided, the deadball line is determined and used to filter lines.
        - If `visualise_steps` is True, intermediate results are displayed for debugging or visualisation purposes.
    """

    if is_path:
        # Step 1: Load and resize the image    
        image_gray = general.load_and_resize_image(im_path, gray=True)
        image = general.load_and_resize_image(im_path)
    else:
        image_gray = cv2.resize(cv2.cvtColor(im_path, cv2.COLOR_BGR2GRAY), (800, 450))
        image = cv2.resize(im_path, (800, 450))

    if visualise_steps:
        # Visualise resized image
        visualise_step(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), "Resized Image")

    # Step 2: Detect field outline using green area segmentation
    field_outline = detect_rugby_field(image)

    if visualise_steps:
        # Visualise detected field outline
        field_outline_img = image.copy()
        cv2.drawContours(field_outline_img, [np.array(field_outline, dtype=np.int32)], -1, (255, 0, 0), 2)
        visualise_step(cv2.cvtColor(field_outline_img, cv2.COLOR_BGR2RGB), "Field Outline Detection")

    # Step 3: Create a field-line mask.
    threshold = create_field_line_mask(image)

    if visualise_steps:
        visualise_step(threshold, "White Field Line Mask")

    # Step 4: Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if visualise_steps:
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        visualise_step(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), "All Contours")

    # Step 5: Extract straight line segments from contours (within field outline, excluding box if provided)
    straight_lines = extract_straight_lines(contours, field_outline, exclusion_box)
    _field_debug(f"[FIELD] straight lines: {len(straight_lines)}")
    
    if visualise_steps:
        straight_contours_img = image.copy()
        line_contours = [np.array(segment, dtype=np.int32) for segment in straight_lines]
        cv2.drawContours(straight_contours_img, line_contours, -1, (0, 255, 0), 2)
        visualise_step(cv2.cvtColor(straight_contours_img, cv2.COLOR_BGR2RGB), "Extracted Straight Contours")

    # Step 6: Fit straight lines to the extracted segments
    contour_lines = fit_straight_lines_to_contours(straight_lines, min_length=20)
    _field_debug(f"[FIELD] fitted contour lines: {len(contour_lines)}")
    
    if visualise_steps:
        fitted_lines_img = image.copy()
        for x1, y1, x2, y2 in contour_lines:
            cv2.line(fitted_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        visualise_step(cv2.cvtColor(fitted_lines_img, cv2.COLOR_BGR2RGB), "Fitted Straight Lines")

    deadball_line = None
    if lineout_centre is not None:
        deadball_line = get_deadball_line(field_outline, image, lineout_centre)

    if deadball_line is not None:
        if visualise_steps:
            deadball_line_img = image.copy()
            cv2.line(deadball_line_img, (deadball_line[0], deadball_line[1]), (deadball_line[2], deadball_line[3]), (255, 0, 0), 2)
            visualise_step(cv2.cvtColor(deadball_line_img, cv2.COLOR_BGR2RGB), "Deadball Line")
            
        # Step 6b: Filter out lines approximately perpendicular to the deadball line
        contour_lines = filter_by_deadball_line(contour_lines, deadball_line, tolerance=1)

        if visualise_steps:
            db_filtered_img = image.copy()
            for x1, y1, x2, y2 in contour_lines:
                cv2.line(db_filtered_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            visualise_step(cv2.cvtColor(db_filtered_img, cv2.COLOR_BGR2RGB), "Lines Filtered by Deadball Line")

    # Step 7: Extend lines to field boundaries (top/bottom or left/right)
    extended_lines = [
    extend_line_to_image_bounds(line, image.shape)
    for line in contour_lines
]
    _field_debug(f"[FIELD] extended lines: {len(extended_lines)}")
    
    if visualise_steps:
        extended_lines_img = image.copy()
        for x1, y1, x2, y2 in extended_lines:
            cv2.line(extended_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        visualise_step(cv2.cvtColor(extended_lines_img, cv2.COLOR_BGR2RGB), "Extended Lines to Boundaries")

    # Step 8: Remove initial anomalous lines based on angle outliers
    removed_initial_anomalous_lines = remove_anomalous_lines_by_angle(extended_lines)
    
    if visualise_steps:
        initial_filtered_img = image.copy()
        for x1, y1, x2, y2 in removed_initial_anomalous_lines:
            cv2.line(initial_filtered_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        visualise_step(cv2.cvtColor(initial_filtered_img, cv2.COLOR_BGR2RGB), "Initial Anomalous Lines Removed")

    # Step 9: Average/merge lines whose midpoints are close together
    average_lines = average_lines_by_midpoint(removed_initial_anomalous_lines, field_outline, 100)
    
    
    if visualise_steps:
        averaged_img = image.copy()
        for x1, y1, x2, y2 in average_lines:
            cv2.line(averaged_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        visualise_step(cv2.cvtColor(averaged_img, cv2.COLOR_BGR2RGB), "Lines Averaged by Midpoint")

    # Step 10: Final anomalous line removal (second pass after averaging)
    final_lines = remove_anomalous_lines_by_angle(average_lines)
    
    if visualise_steps:
        final_img = image.copy()
        for x1, y1, x2, y2 in final_lines:
            cv2.line(final_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        visualise_step(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB), "Final Lines After Anomaly Removal")
    _field_debug(f"[FIELD] final lines: {len(final_lines)}")
    return final_lines, field_outline
