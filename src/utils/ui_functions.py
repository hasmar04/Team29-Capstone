"""
ui_functions.py
---------------
This module provides user interface utilities for file selection, input validation, and user-guided parameter selection in rugby analytics workflows. It includes functions for selecting files, checking file types, prompting for model types and booleans, extracting frames from videos, interactively selecting threshold values, and robustly collecting field point locations from user input. The module is designed to ensure robust and user-friendly input handling for downstream computer vision tasks.

Functions:
---------------
- `select_file`: Opens a file dialog for the user to select a file and returns the selected file's path.
- `check_is_video`: Determines whether a given file is a video based on its file extension.
- `get_model_type`: Prompts the user for the model type and validates it.
- `get_boolean`: Prompts the user for a boolean input and converts it to a boolean value.
- `get_frame`: Extracts a frame from the middle of a video file.
- `threshold_slider`: Displays a slider to adjust the threshold value for field line detection.
- `get_point_locations`: Prompts the user to input field point locations for 4 points, with robust input validation.
- `get_coordinates`: Allows the user to click on a point in an image to retrieve its coordinates.

Dependencies:
---------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- tkinter
- constants
"""

import os, sys, cv2
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
from src.constants import POINT_SELECTION_STRING

def select_file():
    """
    Opens a file dialog for the user to select a file and returns the selected file's path.
    Allows the user to choose video or image files with specific extensions.

    Returns:
        str: The full path of the selected file.
    Raises:
        SystemExit: If no file is selected.
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Ensure the dialog appears on top of other windows
    filename = askopenfilename(
        title='Select a File',
        filetypes=[('Video Files', '*.gif *.mov *.mp4'), ('Image Files', '*.jpg *.jpeg *.png')]
    )
    root.destroy()
    
    if filename:
        return filename
    else:
        print(f'Error: No file selected')
        sys.exit(1)


def select_directory():
    """
    Opens a directory dialog for the user to select a folder and returns the selected folder's path.
    
    Returns:
        str: The full path of the selected directory, or None if no directory is selected.
    """
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Ensure the dialog appears on top of other windows
    directory = askdirectory(
        title='Select a Directory'
    )
    root.destroy()
    
    if directory:
        return directory
    else:
        return None


def check_is_video(filename):
    """
    Determines whether a given file is a video based on its file extension.

    Parameters
        filename (str): The name of the file to check, including its extension.

    Returns:
        bool: True if the file is a video, False if it is an image.

    Raises:
        SystemExit: If the file has an unknown or unsupported file extension.
    """
    _, file_extension = os.path.splitext(filename)
    if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
        return False
    elif file_extension.lower() in ['.mp4', '.avi', '.mov', '.gif']:
        return True
    else:
        print(f"Unknown or unsupported file type: {file_extension}")
        sys.exit(1)


def get_model_type():
    """
    Prompts the user for the model type and validates it.

    Returns:
        str: The chosen model type (e.g., 'ball', 'lineout', 'ruck', or 'rugby').

    Raises:
        SystemExit: If the user enters an invalid model type.
    """
    model_types = ['ball', 'lineout', 'ruck', 'all']
    model_type = input("Please type ball, lineout, ruck or all to choose the model: ").lower()

    if model_type not in model_types:
        print(f"Error: '{model_type}' is not a valid model type.")
        sys.exit(1)
    elif model_type == 'all':
        model_type = 'rugby'

    return model_type


def get_boolean(prompt):
    """
    Prompts the user for a boolean input and converts it to a boolean value.

    Parameters
        prompt (str): The prompt message to display to the user.

    Returns:
        bool: The converted boolean value.

    Raises:
        SystemExit: If the input cannot be converted to a boolean.
    """
    user_input = str(input(prompt)).strip().lower()

    if user_input in ['true', 'yes', '1']:
        return True
    elif user_input in ['false', 'no', '0']:
        return False

    print(f"Error: Cannot convert '{user_input}' into True or False.")
    sys.exit(1)

def get_frame(video_path):
    """
    Extracts a frame from the middle of a video file.

    Parameters
        video_path (str): Path to the video file.

    Returns:
        numpy.ndarray or None: The extracted frame as a NumPy array, or None if an error occurs.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        return None

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
      return None
    
    frame_number = total_frames // 2
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = video_capture.read()
    video_capture.release()
    return frame if ret else None

def threshold_slider(example_frame):
    """
    Displays a slider to adjust the threshold value for field line detection.
    The slider dynamically updates the thresholded image in real time.

    Parameters
        example_frame (numpy.ndarray): The example frame to apply thresholding on.

    Returns:
        int: The final threshold value selected by the user.
    """
    def update_threshold(val):
        """
        Callback function to update the thresholded image when the slider value changes.
        """
        _, thresholded = cv2.threshold(resized_frame, val, 255, cv2.THRESH_BINARY)
        cv2.imshow("Select the highest value where field markings are still visible. Select any key to confirm", thresholded)

    # Convert the example frame to grayscale
    gray_frame = cv2.cvtColor(example_frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (800, 450))

    # Create a window for the slider
    window_name = "Select the highest value where field markings are still visible. Select any key to confirm"
    cv2.namedWindow(window_name)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Set the window to be on top

    # Create a trackbar (slider) for threshold adjustment
    cv2.createTrackbar("Threshold", window_name, 0, 255, update_threshold)

    # Set an initial threshold value
    initial_thresh = 165
    cv2.setTrackbarPos("Threshold", window_name, initial_thresh)

    _, thresholded = cv2.threshold(resized_frame, initial_thresh, 255, cv2.THRESH_BINARY)
    cv2.imshow(window_name, thresholded)

    # Wait for the user to close the window
    cv2.waitKey(0)
    # Get the final threshold value from the slider
    try:
        final_thresh = cv2.getTrackbarPos("Threshold", window_name)
    except:
        raise SystemExit("No key selected - exiting...")

    # Destroy the window
    cv2.destroyWindow(window_name)

    return final_thresh


def get_point_locations(frame, field_points):
    """
    Prompts the user to input field point locations for 4 points, each as a set of 4 answers (numbers separated by commas).
    For each point, a dot is shown on the frame, and the user is asked to provide answers for:
        1. Is the dot on the left (1), right (2) or middle of the 50m line (3)?
        2. Is it the try (1), 5m (2), 22m (3), 10m (4) or halfway (5) line?
        3. Is it on the top (1) or bottom (2) half of the field?
        4. Is it on the side (1), 5m (2), 15m (3), centre (4), or halfway (5) line?
    Robustly validates user input for each question.

    Parameters
        frame (numpy.ndarray): The frame on which to display the points.
        field_points (list): List of (x, y) tuples for the points to be labeled.

    Returns:
        list: A list of tuples containing line type and point type for each selected point.
    """
    point_locations = []
    frame_copy = np.array([])

    print(f"\nAnswer the following questions with 4 numbers separated by commas:")
    for idx, q in enumerate(POINT_SELECTION_STRING, 1):
            print(f"{idx}. {q}")

    for current_point in range(4):
        if frame_copy.size == 0:
            frame_copy = frame.copy()
            cv2.circle(frame_copy, field_points[current_point], 5, (0, 0, 255), -1)
            cv2.imshow('Point frame', frame_copy)
            cv2.setWindowProperty('Point frame', cv2.WND_PROP_TOPMOST, 1)  # Set the window to be on top
            cv2.waitKey(1)
        
        while True:
            ans = input(f"Point {current_point+1}: ")
            parts = [x.strip() for x in ans.split(',') if x.strip()]
            if len(parts) != 4:
                print("Please enter exactly 4 numbers separated by commas for this point.")
                continue
            line_side, line_type, point_side, point_type = parts
            # Validate user input for each question
            if line_side not in ['1', '2', '3']:
                print("For Q1, please enter 1 (left), 2 (right), or 3 (middle of 50m line).")
                continue
            if line_type not in ['1', '2', '3', '4', '5']:
                print("For Q2, please enter 1 (try), 2 (5m), 3 (22m), 4 (10m), or 5 (halfway).")
                continue
            if point_side not in ['1', '2']:
                print("For Q3, please enter 1 (top) or 2 (bottom).")
                continue
            if point_type not in ['1', '2', '3', '4']:
                print("For Q4, please enter 1 (side), 2 (5m), 3 (15m), 4 (halfway).")
                continue
            break
        # Define mapping dictionaries inside the loop so they're always in scope
        line_side_map = {'1': 'left', '2': 'right', '3': 'centre'}
        line_type_map = {'1': 'try', '2': '5', '3': '22', '4': '10', '5': 'halfway'}
        point_side_map = {'1': 'top', '2': 'bottom'}
        point_type_map = {'1': 'side', '2': '5', '3': '15', '4': 'centre', '5': 'halfway'}
        # If user selects centre (50m line) or halfway (5) for line_type, always use halfway_line
        if line_side == '3' or line_type == '5':
            line_str = 'halfway_line'
            # If user selects halfway for point_type (option 4 or 5 in Q4), use 'centre' for point_str (to match field_points_dict keys)
            if point_type in ['4', '5']:
                point_str = 'centre'
            else:
                point_str = point_type_map[point_type]
        else:
            line_side_str = line_side_map[line_side]
            line_type_str = line_type_map.get(line_type, line_type)
            if line_type_str in ['try', 'tryline']:
                line_str = f'{line_side_str}_tryline'
            elif line_type_str == 'halfway':
                line_str = 'halfway_line'
            else:
                line_str = f'{line_side_str}_{line_type_str}m_line'
            point_side_str = point_side_map[point_side]
            if point_type in ['1', '4', '5']:
                # For 'centre' or 'halfway', use 'centre' to match field_points_dict keys
                if point_type in ['4', '5']:
                    point_str = 'centre'
                else:
                    point_str = point_type_map[point_type]
            else:
                point_str = f'{point_side_str}_{point_type_map[point_type]}m'
        point_locations.append((line_str, point_str))
        cv2.destroyWindow('Point frame')
        frame_copy = np.array([])
    return point_locations


def click_event(event, x, y, flags, param):
    """
    Handles mouse click events on an OpenCV window.
    This function is triggered when a mouse event occurs. Specifically, it 
    listens for left mouse button clicks (cv2.EVENT_LBUTTONDOWN) and stores 
    the clicked coordinates in a list passed via the `param` argument.
    Parameters
        event (int): The type of mouse event (e.g., cv2.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        flags (int): Any relevant flags passed by OpenCV (not used in this function).
        param (list): A list to store the clicked coordinates.
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        param.append((x, y))  # Store the coordinates

def get_coordinates(image, window_id='Frame', window_title='Frame'):
    """
    Displays an image in a window and allows the user to click on a point to retrieve its coordinates.
    Parameters
        image (numpy.ndarray): The image to be displayed in the window.
        window_id (str, optional): The identifier for the OpenCV window. Defaults to 'Frame'.
        window_title (str, optional): The title of the OpenCV window. Defaults to 'Frame'.
    Returns:
        tuple: A tuple containing the (x, y) coordinates of the clicked point.
    Notes:
        - The function waits until the user clicks on the image to return the coordinates.
        - The `click_event` function is used to handle mouse click events.
    """
    clicked_coordinates = []  # Local list to store clicked coordinates

    # Display the image in a window
    cv2.imshow(window_id, image)
    cv2.setWindowTitle(window_id, window_title)

    # Set the mouse callback function, passing the clicked_coordinates list as param
    cv2.setMouseCallback(window_id, click_event, clicked_coordinates)

    # Wait until a click is registered
    while not clicked_coordinates:
        cv2.waitKey(1)

    # Close the OpenCV window
    cv2.destroyWindow(window_id)

    # Return the clicked coordinates
    return clicked_coordinates[0]