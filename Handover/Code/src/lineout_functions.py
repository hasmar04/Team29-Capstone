"""
lineout_functions.py
--------------------
This module provides functions for detecting and processing lineouts in rugby video analytics. It includes logic for tracking the ball, identifying the frame of ball release during a lineout, and managing detection state and cooldowns. The functions are designed to work with YOLO model outputs and OpenCV for video analysis and visualization.

Functions:
---------------
- `get_ball_release`: Processes detection results to identify the frame where the ball is released during a lineout, handling tracking, cooldowns, and state management.

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
from constants import LINEOUT_MODEL_CLASS_NUMBERS, BALL_MODEL_CLASS_NUMBERS

def get_ball_release(lineout_results, ball_results, ruck_results, paused_state, fps, movement_threshold=50, cooldown_duration=30, tracker_fail_threshold=10):
    """
    Process lineout, ball, and ruck detection results to identify the frame where the ball is released during a lineout.
    Handles ball tracking, release detection, cooldowns, and manages state for robust detection.

    Parameters:
        lineout_results (generator): YOLO inference results for lineout detection.
        ball_results (generator): YOLO inference results for ball detection.
        ruck_results (generator): YOLO inference results for ruck detection.
        paused_state (dict): Dictionary with pause/exit state for frame display.
        fps (int): Frames per second of the video.
        movement_threshold (float, optional): Threshold for ball movement to detect release. Default is 50.
        cooldown_duration (int, optional): Number of frames to wait after a release before detecting another. Default is 30.
        tracker_fail_threshold (int, optional): Max allowed tracker failures before deactivation. Default is 10.

    Returns:
        tuple: (lineout_result, lineout_box, hooker_centre/lineout_centre, imsize)
            - lineout_result: The last processed lineout result or stored results if lineout is lost.
            - lineout_box: The bounding box of the lineout.
            - hooker_centre/lineout_centre (tuple): The (x, y) centre of the hooker or lineout.
            - imsize (tuple): The image size as (width, height).
    """
    if not lineout_results or not ball_results or not ruck_results:
        raise SystemExit("Video results generator exhausted - no more frames to process")

    second_linout_release = False
    previous_ball_centre = None
    ball_stationary = True
    cooldown_frames = 0
    release_frame_detected = False
    tracker = None
    tracker_active = False
    tracker_fail_counter = 0
    no_lineout_frames = 0
    stored_results = []
    lineout_box = []
    ball_box = []
    hooker_centre = tuple()
    lineout_centre = tuple()
    imsize = tuple()

    for lineout_result, ball_result, _ in zip(lineout_results, ball_results, ruck_results):
        frame = lineout_result.orig_img
        resized_frame = cv2.resize(frame, (850, 450))
        annotated_frame = resized_frame.copy()
        
        if not imsize:
            imsize = frame.shape[:2]
            imsize = (imsize[1], imsize[0])

        if paused_state['exit']:
            return None, None, [], imsize

        lineout_model_boxes, lineout_model_classes, lineout_model_confidences = general.get_class_detections(lineout_result)
        ball_model_boxes, ball_model_classes, ball_model_confidences = general.get_class_detections(ball_result)

        lineout_boxes = [box for box, cls in zip(lineout_model_boxes, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Lineout']]
        ball_boxes = [box for box, cls in zip(ball_model_boxes, ball_model_classes) if cls == BALL_MODEL_CLASS_NUMBERS['Ball']]
        hooker_boxes = [box for box, cls in zip(lineout_model_boxes, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Hooker']]
        lineout_confidences = [conf for conf, cls in zip(lineout_model_confidences, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Lineout']]
        ball_confidences = [conf for conf, cls in zip(ball_model_confidences, ball_model_classes) if cls == BALL_MODEL_CLASS_NUMBERS['Ball']]
        hooker_confidences = [conf for conf, cls in zip(lineout_model_confidences, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Hooker']]

        if not lineout_boxes:
            # Store the lineout state if no lineout is detected
            stored_results.append(lineout_result)
            no_lineout_frames += 1

        else:
            stored_results.clear()
            no_lineout_frames = 0

        if hooker_boxes:
            hooker_box = hooker_boxes[np.argmax(hooker_confidences)]
            hooker_centre = general.box_bottom_centre(hooker_box)
            hooker_box = general.round_list_values(hooker_box)
            hooker_box = general.convert_coordinates(hooker_box, imsize)
            annotated_frame = draw.draw_boxes(annotated_frame, [hooker_box], outline_colour=(0, 255, 0), line_thickness=2, box_annotation='Hooker', show_image=False) if hooker_boxes else annotated_frame
        
        if lineout_boxes:
            lineout_box = lineout_boxes[np.argmax(lineout_confidences)]
            lineout_centre = general.box_bottom_centre(lineout_box)

            lineout_box = general.round_list_values(lineout_box)
            lineout_box = general.convert_coordinates(lineout_box, imsize)
            annotated_frame = draw.draw_boxes(annotated_frame, [lineout_box], outline_colour=(255, 0, 0), line_thickness=2, box_annotation='Lineout', show_image=False) if lineout_boxes else annotated_frame

        current_ball_centre = None

        if ball_boxes:
            lineout_x1, lineout_y1, lineout_x2, lineout_y2 = lineout_box

            lineout_width = lineout_x2 - lineout_x1
            lineout_height = lineout_y2 - lineout_y1

            ball_roi = [lineout_x1 - lineout_width // 4, lineout_y1 - lineout_height // 4, lineout_x2 + lineout_width // 4, lineout_y2 + lineout_height // 4]

            ball_boxes_overlap = [box for box in ball_boxes if general.box_overlap(box, ball_roi)]

            if ball_boxes_overlap:
                ball_box = ball_boxes_overlap[np.argmax(ball_confidences)]

                ball_box = general.round_list_values(ball_box)
                ball_box = general.convert_coordinates(ball_box, imsize)
                annotated_frame = draw.draw_boxes(annotated_frame, [ball_box], outline_colour=(0, 165, 255), line_thickness=2, box_annotation='Ball', show_image=False) if ball_boxes else annotated_frame
        
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

        if current_ball_centre and cooldown_frames == 0:
            ball_stationary, release_frame_detected = ball.detect_ball_release(
                current_ball_centre, previous_ball_centre, ball_stationary, movement_threshold)

            previous_ball_centre = current_ball_centre
        
        general.display_frame(annotated_frame, paused_state, 1000, window_title="Waiting for lineout to end...")

        if release_frame_detected and second_linout_release:
            print('Ball released twice!')
            if hooker_centre:
                return lineout_result, lineout_box, hooker_centre, imsize
            else:
                return lineout_result, lineout_box, lineout_centre, imsize
            
        elif release_frame_detected and not second_linout_release:
            print('Ball released!')
            release_frame_detected = False
            second_linout_release = True
            cooldown_frames = cooldown_duration

        # No lineout detected for a second
        elif no_lineout_frames >= fps:
            print('Lineout no longer detected')
            if hooker_centre:
                return stored_results, lineout_box, hooker_centre, imsize

            else:
                return stored_results, lineout_box, hooker_centre, imsize

        if cooldown_frames > 0:
            cooldown_frames -= 1

    print('Generator exhausted - no more frames to process')
    return None, None, tuple(), imsize
