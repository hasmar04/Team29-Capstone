"""
ruck_functions.py
-----------------
This module provides utilities for rugby video analytics, focusing on ruck detection, ball release, and bounding box processing using YOLO model outputs and OpenCV. It includes functions for extracting ruck feet positions, tracking the ball, and identifying the frame of ball release during a ruck.

Functions:
---------------
- `get_last_feet`: Returns the bottom-left and bottom-right coordinates from a ruck bounding box.
- `get_ball_release`: Processes detection results to identify the frame where the ball is released during a ruck, handling tracking, cooldowns, and state management.

Dependencies:
---------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- general_functions, drawing_functions, ball_functions, constants
"""

import cv2
import numpy as np
import general_functions as general
import drawing_functions as draw
import ball_functions as ball
from constants import RUCK_MODEL_CLASS_NUMBERS, BALL_MODEL_CLASS_NUMBERS

def get_last_feet(ruck_box):
    """
    Extracts the bottom-left and bottom-right coordinates from a ruck bounding box.

    Parameters
        ruck_box (list or tuple): Coordinates (x1, y1, x2, y2) of the bounding box.

    Returns:
        tuple: (bottom_left, bottom_right) as (x, y) tuples.
    """
    x1, y1, x2, y2 = ruck_box

    # Bottom left is the smallest x and largest y
    bottom_left = (round(min(x1, x2)), round(max(y1, y2)))

    # Bottom right is the largest x and the largest y
    bottom_right = (round(max(x1, x2)), round(max(y1, y2)))

    return bottom_left, bottom_right


def get_ball_release(ruck_results, ball_results, lineout_results, paused_state, fps, movement_threshold=10, tracker_fail_threshold=10):
    """
    Processes ruck and lineout detection results to identify the frame where the ball is released during a ruck.
    Handles tracking, cooldowns, and manages state for robust detection.

    Parameters:
        ruck_results (generator): YOLO inference results for ruck detection.
        lineout_results (generator): YOLO inference results for lineout detection.
        paused_state (dict): Dictionary with pause/exit state for frame display.
        movement_threshold (float, optional): Threshold for ball movement to detect release. Default is 50.
        tracker_fail_threshold (int, optional): Max allowed tracker failures before deactivation. Default is 10.

    Returns:
        tuple: (ruck_results, ruck_result, ruck_box, imsize, lineout_results)
            - ruck_results: The (possibly updated) generator for ruck results.
            - ruck_result: The last processed ruck result.
            - ruck_box (list): The bounding box of the ruck.
            - imsize (tuple): The image size as (width, height).
            - lineout_results: The (possibly updated) generator for lineout results.
    """
    if not ruck_results or not ball_results or not lineout_results:
        raise SystemExit("Video results generator exhausted - no more frames to process")

    previous_ball_centre = None
    ball_stationary = True
    release_frame_detected = False
    tracker = None
    tracker_active = False
    tracker_fail_counter = 0
    no_ruck_frames = 0
    stored_results = []
    imsize = tuple()
    ruck_box = []
    ball_box = []

    for ruck_result, ball_result, _ in zip(ruck_results, ball_results, lineout_results):

        frame = ruck_result.orig_img
        annotated_frame = frame.copy()
        ball_box = []

        if not imsize:
            imsize = frame.shape[:2]
            imsize = (imsize[1], imsize[0])

        if paused_state['exit']:
            return None, [], imsize

        ruck_model_boxes, ruck_model_classes, ruck_model_confidences = general.get_class_detections(ruck_result)
        ball_model_boxes, ball_model_classes, ball_model_confidences = general.get_class_detections(ball_result)

        ruck_boxes = [box for box, cls in zip(ruck_model_boxes, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]
        ball_boxes = [box for box, cls in zip(ball_model_boxes, ball_model_classes) if cls == BALL_MODEL_CLASS_NUMBERS['Ball']]
        ruck_confidences = [conf for conf, cls in zip(ruck_model_confidences, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]
        ball_confidences = [conf for conf, cls in zip(ball_model_confidences, ball_model_classes) if cls == BALL_MODEL_CLASS_NUMBERS['Ball']]         

        if ruck_boxes:
            no_ruck_frames = 0

            ruck_box = ruck_boxes[np.argmax(ruck_confidences)]
            ruck_box = general.round_list_values(ruck_box)
            annotated_frame = draw.draw_boxes(annotated_frame, [ruck_box], outline_colour=(255, 0, 0), line_thickness=2, box_annotation='Ruck', show_image=False) if ruck_box else annotated_frame

        if ball_boxes:
            ruck_x1, ruck_y1, ruck_x2, ruck_y2 = ruck_box

            ruck_width = ruck_x2 - ruck_x1
            ruck_height = ruck_y2 - ruck_y1

            ball_roi = [ruck_x1 - ruck_width * 2, ruck_y1 - ruck_height * 2, ruck_x2 + ruck_width * 2, ruck_y2 + ruck_height * 2]

            ball_boxes_overlap = [box for box in ball_boxes if general.box_overlap(box, ball_roi, 0)]

            if ball_boxes_overlap:
                ball_box = ball_boxes_overlap[np.argmax(ball_confidences)]

                ball_box = general.round_list_values(ball_box)
                annotated_frame = draw.draw_boxes(annotated_frame, [ball_box], outline_colour=(0, 165, 255), line_thickness=2, box_annotation='Ball', show_image=False) if ball_box else annotated_frame

        current_ball_centre = None

        if ball_box:
            x1, y1, x2, y2 = map(int, ball_box)
            current_ball_centre = general.box_centre(ball_box)
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            tracker_active = True
            tracker_fail_counter = 0

        elif tracker_active:
            current_ball_centre, tracker_active, tracker_fail_counter = ball.update_tracker(
                tracker, frame, tracker_fail_counter, tracker_fail_threshold, annotated_frame)

        if current_ball_centre:
            ball_stationary, release_frame_detected = ball.detect_ball_release(
                current_ball_centre, previous_ball_centre, ball_stationary, movement_threshold)

            previous_ball_centre = current_ball_centre

            stored_results.append(ruck_result)

            if not ruck_boxes:
                no_ruck_frames += 1
        
        elif not ruck_boxes:
            stored_results.append(ruck_result)
            no_ruck_frames += 1

        elif not current_ball_centre and ruck_boxes and stored_results:
            stored_results.clear()
        
        general.display_frame(annotated_frame, paused_state, 1000, window_title="Waiting for ruck to end...")

        if release_frame_detected:
            print('Ball released!')
            return stored_results, ruck_box, imsize

        # No ruck detected for a second
        elif no_ruck_frames >= fps:
            print('Ruck no longer detected')
            # return stored_ruck_results, stored_ruck_result, ruck_box, imsize, stored_lineout_results, stored_ball_results
            # return ruck_results, stored_ruck_result, ruck_box, imsize, lineout_results, ball_results
            return stored_results, ruck_box, imsize

        
    print('Generator exhausted - no more frames to process')
    return None, [], imsize




