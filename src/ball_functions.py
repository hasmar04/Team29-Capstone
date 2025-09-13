"""
ball_functions.py
----------------
This module provides functions for tracking and detecting the rugby ball during lineouts and rucks using 
computer vision techniques. It includes utilities for updating a tracker, detecting ball release events, 
and orchestrating the logic for ball release detection in both lineout and ruck scenarios.

Functions:
---------------
- `update_tracker`: Updates the tracker for the ball, draws the tracked box, and manages tracker failure logic.
- `detect_ball_release`: Determines if the ball has been released based on its movement between frames.
    
Dependencies:
---------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
"""

import cv2
import numpy as np

def update_tracker(tracker, frame, tracker_fail_counter, tracker_fail_threshold, annotated_frame):
    """
    Update the OpenCV tracker for the ball, draw the tracked bounding box, and manage tracker failure logic.

    Parameters:
        tracker: OpenCV tracker object.
        frame (np.ndarray): Current video frame.
        tracker_fail_counter (int): Current count of consecutive tracker failures.
        tracker_fail_threshold (int): Maximum allowed consecutive tracker failures before deactivating the tracker.
        annotated_frame (np.ndarray): Frame to draw the tracked box and label on.

    Returns:
        tuple: (centre, tracker_active, tracker_fail_counter)
            - centre (tuple or None): (x, y) centre of the tracked box if successful, else None.
            - tracker_active (bool): Whether the tracker is still active.
            - tracker_fail_counter (int): Updated tracker failure counter.
    """
    success, tracked_box = tracker.update(frame)
    if success:
        x, y, w, h = map(int, tracked_box)
        centre = (x + w // 2, y + h // 2)
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(annotated_frame, "Ball (tracked)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return centre, True, 0
    else:
        tracker_fail_counter += 1
        if tracker_fail_counter > tracker_fail_threshold:
            return None, False, tracker_fail_counter
        return None, True, tracker_fail_counter


def detect_ball_release(current, previous, stationary, threshold):
    """
    Determine if the ball has been released based on its movement between frames.

    Parameters:
        current (tuple or None): Current (x, y) centre of the ball.
        previous (tuple or None): Previous (x, y) centre of the ball.
        stationary (bool): Whether the ball was previously stationary.
        threshold (float): Movement threshold to consider the ball as released.

    Returns:
        tuple: (stationary, release_detected)
            - stationary (bool): Whether the ball is now stationary.
            - release_detected (bool): True if a release event is detected, else False.
    """
    release_detected = False

    if current and previous:
        dx, dy = current[0] - previous[0], current[1] - previous[1]
        distance = np.hypot(dx, dy)

        if distance < threshold:
            stationary = True
            last_stationary_centre = current
        elif distance >= threshold and stationary:
            release_detected = True
            stationary = False

    return stationary, release_detected
