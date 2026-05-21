"""
general_functions.py
--------------------
This module provides general utility functions for image processing, coordinate conversion, detection result processing, and frame display. These utilities are used throughout the rugby analytics pipeline for tasks such as field mapping, player detection, and offside analysis.

Functions:
---------------
- **Image and Video Processing**
    - `load_and_resize_image`: Loads an image from disk and resizes it to a standard size (optionally grayscale).
    - `get_video_fps`: Retrieves the frames per second (FPS) of a video file.

- **Detection and Bounding Box Utilities**
    - `get_class_detections`: Extracts bounding boxes, class IDs, and confidence scores from YOLO detection results.
    - `box_centre`: Calculates the center point of a bounding box.
    - `box_bottom_centre`: Calculates the bottom center point of a bounding box.
    - `box_top_centre`: Calculates the top center point of a bounding box.
    - `box_overlap`: Checks if two bounding boxes overlap.

- **Coordinate Conversion**
    - `convert_coordinates`: Converts coordinates from the original image size to a resized image size.

- **Frame Display**
    - `display_frame`: Displays a frame with interactive pause/resume/exit controls for auto mode.
    - `display_frame_manual`: Displays a frame with interactive controls for manual mode (pause, play, lineout, ruck, exit).

- **Utility**
    - `round_list_values`: Rounds all values in a list, tuple, or numpy array to the nearest integer, preserving structure.
    - `calculate_angle`: Calculates the angle (in degrees) between two lines defined by their endpoints.

Dependencies:
---------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
"""

import cv2
import numpy as np

def load_and_resize_image(image_path, width=800, height=450, gray=False):
    """
    Load an image from the specified path and resize it to the given dimensions.

    Parameters:
        image_path (str): Path to the image file.
        width (int, optional): Desired width of the resized image. Defaults to 800.
        height (int, optional): Desired height of the resized image. Defaults to 450.
        gray (bool, optional): Whether to load the image in grayscale. Defaults to False.

    Returns:
        numpy.ndarray: The resized image.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the specified path.
    """
    if gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load the image. Check the file path: {image_path}")
    return cv2.resize(image, (width, height))
    
def get_video_fps(video_path):
    """
    Get the frames per second (FPS) of a video file.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        float: FPS of the video.

    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Could not open video {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return round(fps)


def get_class_detections(result):
    """
    Extract bounding boxes, class IDs, and confidence scores from the detection result.

    Parameters:
        result: An object containing detection results, which includes bounding boxes, 
                class IDs, and confidence scores. It is expected to have `boxes.xyxy`, 
                `boxes.cls`, and `boxes.conf` attributes.

    Returns:
        tuple: A tuple containing three elements:
            - boxes (numpy.ndarray): An array of bounding box coordinates [x_min, y_min, x_max, y_max].
            - classes (numpy.ndarray): An array of class IDs corresponding to the detected objects.
            - confidences (numpy.ndarray): An array of confidence scores for each detection.
    """

    boxes, classes, confidences = [], [], []
    
    if result and result.boxes:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        confidences = result.boxes.conf

        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu()
        if hasattr(boxes, "numpy"):
            boxes = boxes.numpy()

        if hasattr(classes, "cpu"):
            classes = classes.cpu()
        if hasattr(classes, "numpy"):
            classes = classes.numpy()

        if hasattr(confidences, "cpu"):
            confidences = confidences.cpu()
        if hasattr(confidences, "numpy"):
            confidences = confidences.numpy()

    return boxes, classes, confidences

def box_centre(box):
    """
    Calculates the center point of a bounding box.

    Parameters:
        box (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: (x_centre, y_centre) of the box.
    """
    x1, y1, x2, y2 = box
    return (round((x1 + x2) / 2), round((y1 + y2) / 2))


def box_bottom_centre(box):
    """
    Calculates the bottom center point of a bounding box.

    Parameters:
        box (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: (x, y) coordinates of the bottom center of the box.
    """

    x1, y1, x2, y2 = box
    return (round((x1 + x2) / 2), round(max(y1, y2)))

def box_top_centre(box):
    """
    Calculates the top center point of a bounding box.

    Parameters:
        box (tuple): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        tuple: (x, y) coordinates of the top center of the box.
    """

    x1, y1, x2, y2 = box
    return (round((x1 + x2) / 2), round(min(y1, y2)))

def box_overlap(box, roi, threshold=0.5):
    """
    Checks the overlap ratio between two bounding boxes.

    Parameters:
        first_box (tuple): Coordinates of the first box (x1, y1, x2, y2).
        second_box (tuple): Coordinates of the second box (x1, y1, x2, y2).

    Returns:
        float: The overlap ratio between the two boxes (intersection area / union area).
               Returns 0 if there is no overlap.
    """
    box_x1, box_y1, box_x2, box_y2 = box
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    # Calculate the intersection coordinates
    inter_x1 = max(box_x1, roi_x1)
    inter_y1 = max(box_y1, roi_y1)
    inter_x2 = min(box_x2, roi_x2)
    inter_y2 = min(box_y2, roi_y2)

    # Check if there is no overlap
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return False

    # Calculate intersection area
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

    # Calculate areas of both boxes
    box_area = (box_x2 - box_x1) * (box_y2 - box_y1)

    return intersection_area / box_area > threshold

def convert_coordinates(original_coords, original_size, resized_size=(800, 450)):
    """
    Converts coordinates from the original image size to their equivalent on a resized image.

    Parameters:
        original_coords (tuple or list of tuples): Coordinates (x, y) or list of (x, y) points.
        original_size (tuple): (width, height) of the original image.
        resized_size (tuple, optional): (width, height) of the resized image. Defaults to (800, 450).

    Returns:
        tuple or list of tuples: Resized coordinates.

    Raises:
        ValueError: If original_coords is not a tuple or list of tuples.
    """
    original_width, original_height = original_size
    resized_width, resized_height = resized_size

    # Calculate scaling factors
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height  

    # Convert coordinates
    if isinstance(original_coords, tuple) or (isinstance(original_coords, list) and all(isinstance(coord, (float, int)) for coord in original_coords)):
        if len(original_coords) == 2:
            x, y = original_coords
            return (round(x * scale_x), round(y * scale_y))
        elif len(original_coords) == 4:
            x1, y1, x2, y2 = original_coords
            return (round(x1 * scale_x), round(y1 * scale_y), round(x2 * scale_x), round(y2 * scale_y))
        else:
            raise ValueError("Can only convert (x, y) or (x1, y1, x2, y2) coordinates.")
        
    elif isinstance(original_coords, list) and all(isinstance(coord, (list, tuple)) for coord in original_coords):
        if all(len(coord) == 2 for coord in original_coords):
            # List of (x, y) tuples
            return [(round(x * scale_x), round(y * scale_y)) for x, y in original_coords]
        elif all(len(coord) == 4 for coord in original_coords):
            return [(round(x1 * scale_x), round(y1 * scale_y), round(x2 * scale_x), round(y2 * scale_y)) for x1, y1, x2, y2 in original_coords]
        else:
            raise ValueError("Can only convert lists of (x, y) or (x1, y1, x2, y2).")
    else:
        raise ValueError("original_coords must be a tuple, list or a list of tuples or lists.")

def display_frame(frame, paused_state, fps, window_id='Frame', window_title='Frame', display_size=(800, 450)):
    """
    Displays the current frame and handles pausing/resuming and exit functionality for auto mode.

    
        frame (numpy.ndarray): The frame to display.
        paused_state (dict): Dictionary containing the paused state (e.g., {'paused': False, 'exit': False}).
        window_name (str, optional): Name of the OpenCV window. Defaults to 'Frame'.
        display_size (tuple, optional): Size to display the frame. Defaults to (800, 450).
    """
    imsize = (frame.shape[1], frame.shape[0])
    
    if imsize != display_size:
        frame = cv2.resize(frame, display_size)

    cv2.imshow(window_id, frame)
    
    if window_id != window_title:
        cv2.setWindowTitle(window_id, window_title)

    if paused_state['paused']:  # If the video is paused, wait for user input to resume or exit
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key & 0xFF == ord('p'):  # Play when 'p' key is pressed
            paused_state['paused'] = False
        elif key & 0xFF == ord('q'):  # Exit when 'q' key is pressed
            paused_state['exit'] = True
    else:
        # Wait for key press to pause the video
        key = cv2.waitKey(int(1000 / fps))  # Process key press for pausing
        if key & 0xFF == ord('p'):  # Pause when 'p' key is pressed
            paused_state['paused'] = True
        elif key & 0xFF == ord('q'):  # Exit when 'q' key is pressed
            paused_state['exit'] = True

def display_frame_manual(frame, manual_state, fps, window_id='Frame', window_title='Frame', display_size=(800, 450)):
    """
    Displays the current frame and handles pausing/resuming, lineout, ruck, and exit controls for manual mode.

    
        frame (numpy.ndarray): The frame to display.
        manual_state (dict): Dictionary containing manual state (e.g., {'paused': False, 'exit': False, 'lineout': False, 'ruck': False}).
        window_id (str, optional): OpenCV window ID. Defaults to 'Frame'.
        window_title (str, optional): Window title. Defaults to 'Frame'.
        display_size (tuple, optional): Size to display the frame. Defaults to (800, 450).
    """
    imsize = (frame.shape[1], frame.shape[0])
    
    if imsize != display_size:
        frame = cv2.resize(frame, display_size)

    
    cv2.imshow(window_id, frame)

    if window_id != window_title:
        cv2.setWindowTitle(window_id, window_title)

    if manual_state['paused']:  # If the video is paused, wait for user input to resume or exit
        key = cv2.waitKey(0)  # Wait indefinitely for a key press
        if key & 0xFF == ord('p'):  # Play when 'p' key is pressed
            manual_state['paused'] = False
        elif key & 0xFF == ord('q'):  # Exit when 'q' key is pressed
            manual_state['exit'] = True
        elif key & 0xFF == ord('l'):
            manual_state['lineout'] = True
        elif key & 0xFF == ord('r'):
            manual_state['ruck'] = True
    else:
        # Wait for key press to pause the video
        key = cv2.waitKey(int(1000 / fps))  # Process key press for pausing
        if key & 0xFF == ord('p'):  # Pause when 'p' key is pressed
            manual_state['paused'] = True
        elif key & 0xFF == ord('q'):  # Exit when 'q' key is pressed
            manual_state['exit'] = True
        elif key & 0xFF == ord('l'):
            manual_state['lineout'] = True
        elif key & 0xFF == ord('r'):
            manual_state['ruck'] = True


def round_list_values(array):
    """
    Rounds all values in a list, tuple, or NumPy array to the nearest integer, preserving structure.
    Supports 1D and 2D lists, tuples, or NumPy arrays.

    
        array (list, tuple, or numpy.ndarray): Input data structure containing numeric values.

    Returns:
        list: New list with rounded values, preserving the input structure.

    Raises:
        ValueError: If input is not a list, tuple, or NumPy array, or if array has more than 2 dimensions.
    """
    if not isinstance(array, (list, tuple, np.ndarray)):
        raise ValueError("Input must be a list, tuple, or NumPy array.")
    
    if isinstance(array, np.ndarray) and array.ndim > 2:
        raise ValueError("Input array must be 1D or 2D. Higher dimensions are not supported.")
    
    new_list = []

    for value in array:
        if isinstance(value, tuple):
            new_list.append(tuple(round(float(coord)) for coord in value))
        elif isinstance(value, list):
            new_list.append([round(float(coord)) for coord in value])
        elif isinstance(value, np.ndarray):
            new_list.append(np.round(value).astype(int).tolist())
        elif isinstance(value, (int, float, np.integer, np.floating)):
            new_list.append(round(float(value)))
    
    return new_list

def calculate_angle(line1, line2):
    """
    Calculate the angle (in degrees) between two lines defined by their endpoints.
    Parameters:
        line1 (tuple): A tuple of four floats (x1, y1, x2, y2) representing the 
                       coordinates of the first line's endpoints.
        line2 (tuple): A tuple of four floats (x3, y3, x4, y4) representing the 
                       coordinates of the second line's endpoints.
    Returns:
        float: The angle between the two lines in degrees.
    Notes:
        - The function uses the dot product and arccosine to compute the angle.
        - The angle is always returned as a positive value between 0 and 180 degrees.
    """
    # Extract the coordinates of the lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate the direction vectors of the lines
    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    # Normalize the vectors
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    # Calculate the dot product and angle
    dot_product = np.dot(v1_norm, v2_norm)
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

    return angle
