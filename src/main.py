from unittest import result

import cv2
import numpy as np
import os
from src import ui_functions as UI
from src import yolo_functions as yolo_utils
from src import field_functions as field
from src import ruck_functions as ruck
from src import lineout_functions as lineout
from src import drawing_functions as draw
from src import line_functions as line
from src import general_functions as general
from src import offside_functions as offside
from src import point_functions as points
from src import batch_processor as batch
from src import player_detection as player
from src import player_tracking as tracker
from src.events.detection_event import DetectionEvent
from src.events.session_stats import DetectionSessionStats
from src.constants import (
    RUCK_MODEL_CLASS_NUMBERS,
    LINEOUT_MODEL_CLASS_NUMBERS
)

def main():
    """
    Main function to execute the offside detection program.
    This function allows the user to select a video file or directory for processing, validates the file type,
    loads the necessary YOLO models for detection, and provides options to run the program
    in manual, automatic, or batch mode.
    Steps  :
    1. Prompts the user to select a processing mode (manual, auto, or batch).
    2. For single video modes: prompts the user to select a video file through a graphical user interface.
    3. For batch mode: prompts the user to select a directory containing videos.
    4. Validates that the selected file/directory is valid.
    5. Loads YOLO models for ruck, lineout, ball, and player detection.
    6. Executes the selected mode with the provided video(s) and models.
    Raises:
        SystemExit: If the selected file is not a valid video file or directory doesn't exist.
    Inputs:
        - Video file path or directory path selected by the user.
        - User input to choose between manual, automatic, or batch mode.
    Outputs:
        - Processes the video file(s) using the selected mode and detection models.
    """
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (Team29-Capstone)
    project_root = os.path.dirname(script_dir)
    # Models directory path
    models_dir = os.path.join(project_root, 'models')

    print("=" * 80)
    print("QUEENSLAND REDS - RUGBY OFFSIDE DETECTION SYSTEM")
    print("=" * 80)
    print("\nProcessing Modes:  ")
    print("  1. MANUAL - Manually select when to detect rucks/lineouts (single video)")
    print("  2. AUTO   - Automatic detection of rucks/lineouts (single video)")
    print("  3. BATCH  - Process multiple videos automatically (directory input)")
    print()

    # Get mode selection
    mode_selection = ''
    while True:
        mode_selection = input("Select mode (manual/auto/batch): ").lower().strip()
        
        # Map number inputs to mode names
        if mode_selection == '1':
            mode_selection = 'manual'
        elif mode_selection == '2':
            mode_selection = 'auto'
        elif mode_selection == '3':
            mode_selection = 'batch'
        
        if mode_selection in ['manual', 'auto', 'automatic', 'batch']:
            break
        print("Please type 'manual', 'auto', 'batch' or enter 1, 2, or 3 to select a mode.")

    # Loading ruck, lineout, ball and player detection models
    print("\nLoading YOLO models...")
    ruck_model = yolo_utils.load_model(os.path.join(models_dir, 'ruck.pt'))
    lineout_model = yolo_utils.load_model(os.path.join(models_dir, 'lineout.pt'))
    ball_model = yolo_utils.load_model(os.path.join(models_dir, 'ball.pt'))
    player_model = yolo_utils.load_model(os.path.join(models_dir, 'player-id.pt'))
    print("✓ YOLO models loaded successfully")

    # BATCH MODE
    if mode_selection == 'batch':
        print("\n" + "=" * 80)
        print("BATCH PROCESSING MODE")
        print("=" * 80)
        print("\nSelect a directory containing video files to process.")
        print("Outputs will be saved in the 'batch_output' folder.\n")
        
        # Get input directory
        input_path = UI.select_directory()
        
        if not input_path:
            raise SystemExit("No directory selected. Exiting...")
        
        print(f"\nInput directory: {input_path}")
        
        # Get output directory (create default if needed)
        output_dir = os.path.join(project_root, 'batch_output')
        custom_output = input(f"\nUse default output directory '{output_dir}'? (y/n): ").lower().strip()
        
        if custom_output == 'n':
            output_dir = UI.select_directory()
            if not output_dir:
                print("No output directory selected. Using default.")
                output_dir = './batch_output'
        
        print(f"Output directory: {output_dir}")
        print("\nStarting batch processing...\n")
        
        # Process batch
        batch.process_video_batch(input_path, output_dir, ruck_model, lineout_model, ball_model, player_model)
        
        print("\n✓ Batch processing complete!")
        print(f"Check '{output_dir}' for annotated videos and analysis reports.")
        return

    # SINGLE VIDEO MODE (Manual or Auto)
    # Get the video path from a graphical user input
    video_path = UI.select_file()
    is_video = UI.check_is_video(video_path)
    
    # Ensure the selected file was a video
    if not is_video:
        raise SystemExit("Selected file is not a video. Please select a video file (.mp4, .avi, .mov, .gif) for offside detection. Exiting...")

    fps = general.get_video_fps(video_path)
    
    print(f"\nSelected video: {video_path}")
    print(f"Video FPS: {fps}")

    if mode_selection == 'manual':
        print("\n" + "=" * 80)
        print("MANUAL MODE - Use keyboard controls:")
        print("  L - Detect lineout")
        print("  R - Detect ruck")
        print("  P - Pause/Play")
        print("  Q - Quit")
        print("=" * 80 + "\n")
        manual_mode(video_path, fps, ruck_model, lineout_model, player_model)
    else:
        print("\n" + "=" * 80)
        print("AUTOMATIC MODE - Automatic ruck/lineout detection")
        print("  P - Pause")
        print("  Q - Quit")
        print("=" * 80 + "\n")
        auto_mode(video_path, fps, ruck_model, lineout_model, ball_model, player_model)


def manual_mode(video_path, fps, ruck_model, lineout_model, player_model):
    """
    Processes a video in manual mode to detect and annotate ruck and lineout offside lines 
    and players using YOLO models and user input.

    Parameters:
        video_path (str): Path to the video file to be processed.
        ruck_model (YOLO): Pre-trained YOLO model for detecting ruck-related objects.
        lineout_model (YOLO): Pre-trained YOLO model for detecting lineout-related objects.
        player_model (YOLO): Pre-trained YOLO model for detecting players.
    
    Functionality:
        - Allows the user to manually pause and interact with the video using keyboard inputs:
            - 'L': Detect and annotate lineout offside lines and players.
            - 'R': Detect and annotate ruck offside lines and players.
            - 'P': Pause or resume video playback.
            - 'Q': Quit the video processing.
        - Detects ruck and lineout objects using YOLO models and calculates offside lines.
        - Allows the user to manually select points on the video frame when detections are not available.
        - Annotates the video frame with offside lines and highlights offside players.
        - Displays the annotated frames for user review.
    
    Notes:
        - The function uses OpenCV for video processing and user interaction.
        - Requires pre-trained YOLO models for ruck, lineout, and player detection.
        - User input is required for certain operations, such as selecting points on the frame.

    Raises:
        SystemExit: If the video file cannot be loaded.
    """
    
    manual_pause_state = {
        'paused': False,
        'lineout': False,
        'ruck': False,
        'exit': False
    }

    video = cv2.VideoCapture(video_path)
    
    if not video.isOpened():
        raise SystemExit("Video could not be loaded. Exiting...")
    
    player_tracker = tracker.PlayerTracker()
    


    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        if manual_pause_state['exit']:
            print("Exiting video processing")
            break

        frame = cv2.resize(frame, (800, 450))
        imsize = (frame.shape[1], frame.shape[0])

        general.display_frame_manual(frame, manual_pause_state, fps, window_id='Frame', window_title="Select 'L' for lineout, 'R' for ruck, 'P' to pause/play and 'Q' to quit")

        if manual_pause_state['ruck']:
            ruck_result = yolo_utils.perform_inference(frame, ruck_model, False)

            try:
                ruck_result = next(ruck_result)
            except:
                raise SystemExit("YOLO inference failed. Exiting...")
            
            ruck_model_boxes, ruck_model_classes, ruck_model_confidences = general.get_class_detections(ruck_result)

            ruck_boxes = [box for box, cls in zip(ruck_model_boxes, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]
            ruck_confidences = [conf for conf, cls in zip(ruck_model_confidences, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]

            if ruck_boxes:
                ruck_box = ruck_boxes[np.argmax(ruck_confidences)]
                ruck_left, ruck_right = ruck.get_last_feet(ruck_box)
                ruck_centre = general.box_bottom_centre(ruck_box)

            else:
                ruck_left = UI.get_coordinates(frame, window_id='Frame', window_title='Select the left last feet of the ruck')
                ruck_right = UI.get_coordinates(frame, window_id='Frame', window_title='Select the right last feet of the ruck')
                ruck_centre = ((ruck_left[0] + ruck_right[0]) // 2, (ruck_left[1] + ruck_right[1]) // 2)

            ui_thresh = UI.threshold_slider(frame)

            field_lines, field_outline = field.get_field_lines(frame, thresh=ui_thresh, is_path=False, visualise_steps=False)

            intersection_point = line.find_average_intersection_point(field_lines)

            # Create lines from the last feet of the ruck to the intersection point of the lines on the field
            left_ruck_line = (intersection_point[0], intersection_point[1], ruck_left[0], ruck_left[1])
            right_ruck_line = (intersection_point[0], intersection_point[1], ruck_right[0], ruck_right[1])
            centre_ruck_line = (intersection_point[0], intersection_point[1], ruck_centre[0], ruck_centre[1])

            # Fit the offside lines to the field outline
            left_ruck_line = field.fit_line_to_field(left_ruck_line, field_outline)
            right_ruck_line = field.fit_line_to_field(right_ruck_line, field_outline)
            centre_ruck_line = field.fit_line_to_field(centre_ruck_line, field_outline)

            ruck_frame = draw.draw_lines(frame, [left_ruck_line, right_ruck_line], (0, 0, 255), show_image=False)

            player_dict = build_offside_player_dict(frame, player_model, player_tracker)

            offside_player_boxes = offside.get_players_between_lines(player_dict, left_ruck_line, right_ruck_line)

            # Draw the player box in red for offside players
            ruck_frame = draw.draw_boxes(ruck_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False)

            manual_pause_state['paused'] = True
            manual_pause_state['ruck'] = False

            # Display annotated ruck frame
            general.display_frame_manual(ruck_frame, manual_pause_state, fps, window_id='Frame', window_title="Displaying ruck offside detections. Press 'P' to continue playing.")


        if manual_pause_state['lineout']:
            # CHECK FOR HOOKER AND/OR LINEOUT IN IMAGE, OTHERWISE
            lineout_result = yolo_utils.perform_inference(frame, lineout_model, False)
            lineout_box = None
            lineout_centre = None

            try:
                lineout_result = next(lineout_result)
            except:
                raise SystemExit("YOLO inference failed. Exiting...")
            
            lineout_model_boxes, lineout_model_classes, lineout_model_confidences = general.get_class_detections(lineout_result)

            lineout_boxes = [box for box, cls in zip(lineout_model_boxes, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Lineout']]
            lineout_confidences = [conf for conf, cls in zip(lineout_model_confidences, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Lineout']]
            hooker_boxes = [box for box, cls in zip(lineout_model_boxes, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Hooker']]
            hooker_confidences = [conf for conf, cls in zip(lineout_model_confidences, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Hooker']]

            if hooker_boxes:
                hooker_box = hooker_boxes[np.argmax(hooker_confidences)]
                lineout_centre = general.box_bottom_centre(hooker_box)

            elif lineout_boxes:
                lineout_box = lineout_boxes[np.argmax(lineout_confidences)]
                lineout_centre = general.box_centre(lineout_box)

            else:
                lineout_centre = UI.get_coordinates(frame, window_id='Frame', window_title='Select the centre point of the lineout on the sideline.')

            thresh = UI.threshold_slider(frame)

            field_lines, field_outline = field.get_field_lines(frame, lineout_centre=lineout_centre, thresh=thresh, is_path=False, visualise_steps=False)

            intersection_point = line.find_average_intersection_point(field_lines)

            # field_points = points.get_field_points(frame, field_lines, field_outline, thresh)

            field_points = []
            point_frame = frame.copy()

            for _ in range(4):
                field_points.append(UI.get_coordinates(point_frame, window_id='Frame', window_title='Select four crosspoints of the field lines'))
            
            point_locations = UI.get_point_locations(point_frame, field_points)
            
            H = points.get_homography_matrix(field_points, point_locations)

            left_offside_point, right_offside_point = points.get_lineout_offside_points(lineout_centre, H)

            lineout_frame = draw.draw_points(frame, [lineout_centre], (0, 255, 0), show_image=False)
            lineout_frame = draw.draw_points(lineout_frame, [left_offside_point, right_offside_point], show_image=False)
            
            player_result_gen = yolo_utils.perform_inference(frame, player_model, False)

            try:
                player_result = next(player_result_gen)
            except StopIteration:
                player_result = None

            players = player.build_player_data(frame, player_result)
            players = player.assign_teams_by_colour(players)
            players = player_tracker.update(players)

            for p in players:
                x1, y1, x2, y2 = p["box"]
                cv2.putText(
                    lineout_frame,
                    str(p["track_id"]),
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            left_lineout_line = (intersection_point[0], intersection_point[1], left_offside_point[0], left_offside_point[1])
            right_lineout_line = (intersection_point[0], intersection_point[1], right_offside_point[0], right_offside_point[1])
        
            left_lineout_line = field.fit_line_to_field(left_lineout_line, field_outline)
            right_lineout_line = field.fit_line_to_field(right_lineout_line, field_outline)

            lineout_frame = draw.draw_lines(lineout_frame, [left_lineout_line, right_lineout_line], (255, 0, 0), show_image=False)

            player_dict = build_offside_player_dict(frame, player_model, player_tracker)

            if lineout_box is not None: 
                player_dict = offside.filter_for_offside_detection(player_dict, lineout_box)
                player_dict = offside.filter_detections_off_the_field(player_dict, lineout_box, imsize)


            offside_player_boxes = offside.get_players_between_lines(player_dict, left_lineout_line, right_lineout_line)

            lineout_frame = draw.draw_boxes(lineout_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False)

            manual_pause_state['paused'] = True
            manual_pause_state['lineout'] = False

            # Display annotated lineout frame
            general.display_frame_manual(lineout_frame, manual_pause_state, fps, window_id='Frame', window_title="Displaying lineout offside detections. Press 'P' to continue playing")


def auto_mode(video_path, fps, ruck_model, lineout_model, ball_model, player_model):
    """
    Processes a video to detect and analyse rucks and lineouts in rugby matches using YOLO models.
    This function performs inference on a video file using pre-trained YOLO models for ruck, lineout, 
    ball, and player detection. It identifies rucks and lineouts, detects offside players, and visualises 
    the results with annotated frames.

    Parameters:
        video_path (str): Path to the input video file.
        fps (int): Frames per second of the video.
        ruck_model (YOLO): Pre-trained YOLO model for ruck detection.
        lineout_model (YOLO): Pre-trained YOLO model for lineout detection.
        ball_model (YOLO): Pre-trained YOLO model for ball detection.
        player_model (YOLO): Pre-trained YOLO model for player detection.

    Raises:
        AssertionError: If the ruck and lineout result generators are out of sync.
        SystemExit: If no ruck or lineout could be detected during fallback scenarios.

    Notes:
        - The function uses a threshold value for field line detection, which is obtained via user input.
        - Rucks and lineouts are detected based on consecutive frame counts.
        - Offside players are identified based on their positions relative to the offside lines.
        - The function supports pausing and exiting during video processing.

    Workflow:
        1. Perform inference on the video using ruck, lineout, and ball models.
        2. Detect rucks and lineouts based on consecutive frame counts.
        3. For detected rucks:
            - Detect the ball release and calculate offside lines.
            - Identify offside players and annotate the frame.
        4. For detected lineouts:
            - Detect the ball release and calculate offside lines using homography.
            - Identify offside players and annotate the frame.
        5. Display annotated frames and handle user interactions (pause/exit).

    Returns:
        None
    """

    # Initialise paused state
    paused_state = {'paused': False, 'exit': False}

    player_tracker = tracker.PlayerTracker()
    
    # Store overall session statistics
    session_stats = DetectionSessionStats()

    # Save video FPS
    session_stats.fps = fps

    frame_threshold = fps // 10

    #Initialize team counts
    final_counts = {
    "team_0": 0,
    "team_1": 0,
    "unknown_team": 0,
    "refs": 0
    }

    # Initialise ruck and lineout frame counts
    ruck_frame_count = 0
    lineout_frame_count = 0
    
    # Automatically set threshold value to 120 for field line detection
    ui_thresh = 120
    print(f"Using automatic threshold value: {ui_thresh}")
    
    # Setup for output video and logging
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_video_path = f"{video_name}_offside_detection.mp4"
    output_log_path = f"{video_name}_detection_log.txt"
    frame_size = None
    detection_log = []  # Store detection events with frame ranges for overlays
    detection_annotations = {}  # Map frame_index -> annotated_frame
    total_frame_count = 0  # Track total frames processed in Pass 1

    # Create generators for all inference predictions from the ruck and lineout models
    ruck_results = yolo_utils.perform_inference(video_path, ruck_model, False, 0.1)
    lineout_results = yolo_utils.perform_inference(video_path, lineout_model, False, 0.1) # Also includes hooker detection
    ball_results = yolo_utils.perform_inference(video_path, ball_model, False, 0.4) # Ball detection

    print("Video inference complete")
    print("Commencing analysis...")
    print()
    print("="*80)
    print("PASS 1: Detection Pass - Finding rucks and lineouts")
    print("="*80)
    print()

    # Iterate through the ruck and lineout result generators (each generator will yield the same number of frames)
    for ruck_result, lineout_result, _ in zip(ruck_results, lineout_results, ball_results):

        # Ensure the generators are in sync by checking the original images
        if not np.array_equal(ruck_result.orig_img, lineout_result.orig_img):
            raise AssertionError("Generators are out of sync. Please check the video file and models. Exiting...")

        # Check if the user has requested to exit the video processing
        if paused_state['exit']:
            print("Exiting video processing.")
            break
        
        # Get current frame and store it
        frame = ruck_result.orig_img
        
        # Initialise frame size for video writer
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])
        
        # Resize frame for processing and display (800x450)
        display_frame = cv2.resize(frame, (800, 450))
        
        # Store this frame (will be used if no detection)
        current_display_frame = display_frame.copy()
        
        # Track total frames in pass 1
        total_frame_count += 1

        # Update live session statistics
        session_stats.total_frames = total_frame_count

        # Calculate current video duration
        session_stats.video_duration = total_frame_count / fps
        
        # Show frame to user
        general.display_frame(display_frame, paused_state, 1000, window_title="Searching for a ruck or lineout. Press 'P' to pause or 'Q' to quit")

        # Extract ruck and lineout boxes and confidences
        ruck_model_boxes, ruck_model_classes, ruck_model_confidences = general.get_class_detections(ruck_result)
        lineout_model_boxes, lineout_model_classes, lineout_model_confidences = general.get_class_detections(lineout_result)

        ruck_boxes = [box for box, cls in zip(ruck_model_boxes, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]
        lineout_boxes = [box for box, cls in zip(lineout_model_boxes, lineout_model_classes) if cls == (LINEOUT_MODEL_CLASS_NUMBERS['Lineout'])]
        ruck_confidences = [conf for conf, cls in zip(ruck_model_confidences, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]
        lineout_confidences = [conf for conf, cls in zip(lineout_model_confidences, lineout_model_classes) if cls == (LINEOUT_MODEL_CLASS_NUMBERS['Lineout'])]

        if ruck_frame_count < 0:
            # If ruck detection is in the cooldown period, increment each frame
            ruck_frame_count += 1
        
        if lineout_frame_count < 0:
            # If lineout detection is in the cooldown period, increment each frame
            lineout_frame_count += 1

        # Reset the ruck and lineout frame counts if no boxes are detected
        if not ruck_boxes and not lineout_boxes:
            ruck_frame_count = 0 if ruck_frame_count > 0 else ruck_frame_count
            lineout_frame_count = 0 if lineout_frame_count > 0 else lineout_frame_count
            # Frame already stored above
            continue
        
        # If both ruck and lineout boxes are detected, check the confidence levels
        elif ruck_boxes and lineout_boxes:
            if np.max(ruck_confidences) >= np.max(lineout_confidences):
                lineout_frame_count = 0 if lineout_frame_count > 0 else lineout_frame_count
                ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
                # Frame already stored above
                continue
            
            else:
                ruck_frame_count = 0 if ruck_frame_count > 0 else ruck_frame_count
                lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count
                # Frame already stored above
                continue

        # If only ruck boxes are detected, increment the ruck frame count
        elif ruck_boxes:
            ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
            # Frame already stored above
        
        # If only lineout boxes are detected, increment the lineout frame count
        elif lineout_boxes:
            lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count
            # Frame already stored above

        # If a ruck is detected for three consecutive frames, process the ruck results
        if ruck_frame_count >= frame_threshold:
            # Set a three second cooldown after a ruck is detected
            ruck_frame_count = -(fps * 3) + frame_threshold
            
            # Store the frame index where detection occurs (before ball release processing)
            detection_frame_index = total_frame_count
            detection_timestamp = total_frame_count / fps  # Calculate timestamp in seconds

            print('Ruck detected!')
            print('Starting ball detection...')

            # Continues iterating through ruck generator until the ball is released or the ruck is no longer detected
            # WARNING: This consumes frames from the generators
            ruck_result, ruck_box, imsize = ruck.get_ball_release(ruck_results, ball_results, lineout_results, paused_state, fps)

            # If the user has requested to exit the video processing, break the loop
            if paused_state['exit']:
                break

            # Check if ruck_box is empty (could be list or numpy array)
            if not isinstance(ruck_box, np.ndarray) or ruck_box.size == 0:
                ruck_box = ruck_boxes[np.argmax(ruck_confidences)] if ruck_boxes else None

                # This is a fallback but should never be reached
                if ruck_box is None or (isinstance(ruck_box, np.ndarray) and ruck_box.size == 0):
                    raise SystemExit('Error, no ruck could not be detected. Exiting ...')

            ruck_box = ruck_box.tolist()
            ruck_box = general.convert_coordinates(ruck_box, imsize)
            
            print('Ruck and ball detection finished')

            # Ensures the generator was not exhausted
            if ruck_result is not None:
                if isinstance(ruck_result, list):
                    current_result = ruck_result[0]
                else:
                    current_result = ruck_result
                
                paused_state['paused'] = True

                # Gets the current frame and resizes it for computer vision
                current_frame = current_result.orig_img  
                resized_frame = cv2.resize(current_frame, (800, 450))

                # Get the last feet of the ruck and convert the coordinates to the correct image size
                ruck_left, ruck_right = ruck.get_last_feet(ruck_box)

                ruck_centre = general.box_centre(ruck_box)

                ruck_frame = draw.draw_points(resized_frame, [ruck_left, ruck_right], (0, 255, 0), show_image=False)

                # Get the lines on the field and the field outline
                field_lines, field_outline = field.get_field_lines(current_frame, thresh=ui_thresh, is_path=False, visualise_steps=False)
                
                # Find the average intersection point between all lines on the field
                intersection_point = line.find_average_intersection_point(field_lines)
                
                # Ensure intersection point is valid
                if all(coord is not None for coord in intersection_point):

                    # Create lines from the last feet of the ruck to the intersection point of the lines on the field
                    left_ruck_line = (intersection_point[0], intersection_point[1], ruck_left[0], ruck_left[1])
                    right_ruck_line = (intersection_point[0], intersection_point[1], ruck_right[0], ruck_right[1])
                    centre_ruck_line = (intersection_point[0], intersection_point[1], ruck_centre[0], ruck_centre[1])

                    # Fit the offside lines to the field outline
                    left_ruck_line = field.fit_line_to_field(left_ruck_line, field_outline)
                    right_ruck_line = field.fit_line_to_field(right_ruck_line, field_outline)

                    ruck_frame = draw.draw_lines(ruck_frame, [left_ruck_line, right_ruck_line], (0, 0, 255), show_image=False)

                    # Run player detection on the current frame
                    player_result_gen = yolo_utils.perform_inference(resized_frame, player_model, False)

                    try:
                        player_result = next(player_result_gen)
                    except StopIteration:
                        player_result = None

                    players = player.build_player_data(resized_frame, player_result)
                    players = player.assign_teams_by_colour(players)
                    players = player_tracker.update(players)
                    ruck_frame = draw.draw_player_debug(ruck_frame, players)
                    counts = player.count_teams_and_refs(players)
                    for p in players:
                        x1, y1, x2, y2 = p["box"]

                        team = p.get("team")
                        track_id = p.get("track_id")

                        if team == 0:
                            box_colour = (255, 0, 0)
                        elif team == 1:
                            box_colour = (0, 0, 255)
                        else:
                            box_colour = (0, 255, 255)

                        # Main player box
                        cv2.rectangle(ruck_frame, (x1, y1), (x2, y2), box_colour, 2)

                        label = f"ID:{track_id} T:{team}"

                        cv2.putText(
                            ruck_frame,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            box_colour,
                            2
                        )

                        # ---- JERSEY CROP BOX (GREEN) ----
                        crop_box = p.get("jersey_crop_box")
                        if crop_box is not None:
                            jx1, jy1, jx2, jy2 = crop_box

                            cv2.rectangle(
                                ruck_frame,
                                (jx1, jy1),
                                (jx2, jy2),
                                (0, 255, 0),
                                2
                            )

                    player_dict = player.build_player_coord_dict(players)

                    if ruck_box is not None:
                        player_dict = offside.filter_for_offside_detection(player_dict, ruck_box, overlap_threshold=0.4, width_expansion_factor=0, height_expansion_factor=0)

                    offside_player_boxes = offside.get_players_between_lines(player_dict, left_ruck_line, right_ruck_line)

                    # Draw the player box in red for offside players
                    ruck_frame = draw.draw_boxes(ruck_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False, font_scale=0.5, font_thickness=1)

                    # Add text overlay showing detection info
                    cv2.putText(ruck_frame, f"RUCK DETECTED - {len(offside_player_boxes)} Offside Players", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(ruck_frame, f"Time: {int(detection_timestamp//60)}:{int(detection_timestamp%60):02d}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Store annotated frame for pass 2
                    detection_annotations[detection_frame_index] = ruck_frame.copy()
                    
                    # Log this detection with overlay range
                    overlay_duration_frames = int(fps * 5)  # 5 seconds
                    detection_log.append({
                        'type': 'RUCK',
                        'timestamp': detection_timestamp,
                        'frame': detection_frame_index,
                        'offside_count': len(offside_player_boxes),
                        'confidence': float(max(ruck_confidences)) if ruck_confidences else 0.0,
                        'overlay_start': detection_frame_index,
                        'overlay_end': detection_frame_index + overlay_duration_frames,
                        'annotated_frame': ruck_frame.copy()
                    })

                    # Create structured detection event
                    event = DetectionEvent(

                        # Detection type
                        event_type="ruck",

                        # Highest confidence ruck box
                        confidence=float(max(ruck_confidences)),

                        # Frame number where detection occurred
                        frame_number=detection_frame_index,

                        # Timestamp in seconds
                        timestamp=detection_timestamp,

                        # Number of offside players
                        offside_count=len(offside_player_boxes),

                        # Store tracked player information
                        tracked_players=players,

                        # Store team counts
                        team_counts=counts
                    )

                    # Add event to session statistics
                    session_stats.add_event(event)
                    
                    # Display annotated ruck frame
                    general.display_frame(ruck_frame, paused_state, fps, window_title="Displaying ruck offside detections. Press 'P' to continue playing")
                    
                    print(f"  Ruck analysis complete: {len(offside_player_boxes)} offside players detected")

                else:
                    print("Offside lines not drawn as no valid intersection was detected.")
            else:
                print("  WARNING: Generator exhausted - cannot create annotated frame for this detection")
                print("  (Detection occuautorred at the end of the video with insufficient remaining frames)")

        # If a lineout is detected for three consecutive frames, process the lineout results
        if lineout_frame_count >= frame_threshold:
            # Set a 10 second cooldown after a lineout is detected
            lineout_frame_count = -(fps * 10) + frame_threshold
            
            # Store the frame index where detection occurs (before ball release processing)
            detection_frame_index = total_frame_count
            detection_timestamp = total_frame_count / fps  # Calculate timestamp in seconds
                
            print('Lineout detected!')
            print('Starting ball detection...')

            # Continues iterating through lineout generator until the ball is released or the lineout is no longer detected
            # WARNING: This consumes frames from the generators
            lineout_result, lineout_box, lineout_centre, imsize = lineout.get_ball_release(lineout_results, ruck_results, ball_results, paused_state, fps)

            # If the user has requested to exit the video processing, break the loop
            if paused_state['exit']:
                break

            # If the centre of the lineout could not be found during the ball release processing, use the initially detected frame instead
            if not lineout_centre:
                if lineout_box is not None and (not isinstance(lineout_box, np.ndarray) or lineout_box.size > 0):
                    lineout_centre = general.box_bottom_centre(lineout_box)
                else:
                    lineout_box = lineout_boxes[np.argmax(lineout_confidences)] if lineout_boxes else None

                    if lineout_box is not None and (not isinstance(lineout_box, np.ndarray) or lineout_box.size > 0):
                        lineout_centre = general.box_bottom_centre(lineout_box)
                    
                    # This is a fallback but should never be reached
                    else:
                        raise SystemExit('No hooker or lineout could be detected')
            
            # Convert the coordinates to the correct image size
            lineout_centre = general.convert_coordinates(lineout_centre, imsize)

            print('Lineout and ball detection finished')

            # Ensures the generator was not exhausted
            if lineout_result is not None:
                if isinstance(lineout_result, list):
                    current_result = lineout_result[0]
                else:
                    current_result = lineout_result
                
                paused_state['paused'] = True

                # Gets the current frame and resizes it for computer vision
                current_frame = current_result.orig_img
                resized_frame = cv2.resize(current_frame, (800, 450))

                # Get the lines on the field and the field outline
                field_lines, field_outline = field.get_field_lines(current_frame, lineout_centre=lineout_centre, exclusion_box=lineout_box, thresh=ui_thresh, is_path=False, visualise_steps=False)

                # Find the average intersection point between all lines on the field
                intersection_point = line.find_average_intersection_point(field_lines)

                # Automatically estimate field points (no user input required)
                frame_height, frame_width = resized_frame.shape[:2]
                
                # Estimate field intersection points based on standard rugby field layout
                # These are heuristic estimates that work well for typical camera angles
                field_points = [
                    (int(frame_width * 0.25), int(frame_height * 0.35)),  # Top-left
                    (int(frame_width * 0.75), int(frame_height * 0.35)),  # Top-right
                    (int(frame_width * 0.75), int(frame_height * 0.70)),  # Bottom-right
                    (int(frame_width * 0.25), int(frame_height * 0.70))   # Bottom-left
                ]
                
                point_locations = [
                ('left_tryline', 'top_5m'),
                ('right_tryline', 'top_5m'),
                ('right_tryline', 'bottom_5m'),
                ('left_tryline', 'bottom_5m')
                ]           
                
                print(f"  Using automatic field point estimation for lineout")

                # Get the homography matrix from current frame to a top-down view
                H = points.get_homography_matrix(field_points, point_locations)

                # Use the homography matrix to get the offside points +-10m from the centre of the lineout
                left_offside_point, right_offside_point = points.get_lineout_offside_points(lineout_centre, H)

                lineout_frame = draw.draw_points(resized_frame, [lineout_centre], (0, 255, 0), show_image=False)
                lineout_frame = draw.draw_points(lineout_frame, [left_offside_point, right_offside_point], show_image=False)

                # Create lines from the two offside points to the intersection point of the lines on the field
                left_offside_line = (intersection_point[0], intersection_point[1], left_offside_point[0], left_offside_point[1])
                right_offside_line = (intersection_point[0], intersection_point[1], right_offside_point[0], right_offside_point[1])
                centre_offside_line = (intersection_point[0], intersection_point[1], lineout_centre[0], lineout_centre[1])

                # Fit the offside lines to the field outline
                left_offside_line = field.fit_line_to_field(left_offside_line, field_outline)
                right_offside_line = field.fit_line_to_field(right_offside_line, field_outline)

                lineout_frame = draw.draw_lines(lineout_frame, [left_offside_line, right_offside_line], (255, 0, 0), show_image=False)
                
                # Detect players
                player_result_gen = yolo_utils.perform_inference(resized_frame, player_model, False)

                try:
                    player_result = next(player_result_gen)
                except StopIteration:
                    player_result = None

                players = player.build_player_data(resized_frame, player_result)
                players = player.assign_teams_by_colour(players)
                players = player_tracker.update(players)
               
                counts = player.count_teams_and_refs(players)

                lineout_frame = draw.draw_player_debug(lineout_frame, players)
                for p in players:
                    x1, y1, x2, y2 = p["box"]

                    team = p.get("team")
                    track_id = p.get("track_id")

                    if team == 0:
                        box_colour = (255, 0, 0)
                    elif team == 1:
                        box_colour = (0, 0, 255)
                    else:
                        box_colour = (0, 255, 255)

                    # Main player box
                    cv2.rectangle(lineout_frame, (x1, y1), (x2, y2), box_colour, 2)

                    label = f"ID:{track_id} T:{team}"

                    cv2.putText(
                        lineout_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_colour,
                        2
                    )

                    # Jersey crop box
                    crop_box = p.get("jersey_crop_box")
                    if crop_box is not None:
                        jx1, jy1, jx2, jy2 = crop_box

                        cv2.rectangle(
                            lineout_frame,
                            (jx1, jy1),
                            (jx2, jy2),
                            (0, 255, 0),
                            2
                        )
                player_dict = player.build_player_coord_dict(players)
                # Detect players on resized frame


                if lineout_box is not None:
                    player_dict = offside.filter_for_offside_detection(player_dict, lineout_box, 0)
                    player_dict = offside.filter_detections_off_the_field(player_dict, lineout_box, imsize)

                if (left_offside_line is None or right_offside_line is None or
                    any(v is None for v in left_offside_line) or
                    any(v is None for v in right_offside_line)):
                    print("Invalid offside lines for lineout; skipping offside player check")
                    offside_player_boxes = []
                else:
                    offside_player_boxes = offside.get_players_between_lines(
                        player_dict,
                        left_offside_line,
                        right_offside_line
    )

                # Draw the player box in red for offside players
                lineout_frame = draw.draw_boxes(lineout_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False)

                # Store annotated frame for pass 2
                detection_annotations[detection_frame_index] = lineout_frame.copy()
                
                # Log this detection with overlay range
                overlay_duration_frames = int(fps * 5)  # 5 seconds
                detection_log.append({
                    'type': 'LINEOUT',
                    'timestamp': detection_timestamp,
                    'frame': detection_frame_index,
                    'offside_count': len(offside_player_boxes),
                    'confidence': float(max(lineout_confidences)) if lineout_confidences else 0.0,
                    'overlay_start': detection_frame_index,
                    'overlay_end': detection_frame_index + overlay_duration_frames,
                    'annotated_frame': lineout_frame.copy()
                })

                # Create structured detection event
                event = DetectionEvent(

                    # Detection type
                    event_type="lineout",

                    # Highest confidence lineout box
                    confidence=float(max(lineout_confidences)),

                    # Frame number where detection occurred
                    frame_number=detection_frame_index,

                    # Timestamp in seconds
                    timestamp=detection_timestamp,

                    # Number of offside players
                    offside_count=len(offside_player_boxes),

                    # Store tracked player information
                    tracked_players=players,

                    # Store team counts
                    team_counts=counts
                )

                # Add event to session statistics
                session_stats.add_event(event)
                
                # Display annotated lineout frame
                general.display_frame(lineout_frame, paused_state, fps, window_title="Displaying lineout offside detections. Press 'P' to continue playing")
                
                print(f"  Lineout analysis complete: {len(offside_player_boxes)} offside players detected")
            else:
                # Generator exhausted - use current frame for annotation instead
                print("  Generator exhausted - using detection frame for annotation")
                
                # Get lineout box and centre from the initially detected frame  
                if lineout_box is None or (isinstance(lineout_box, np.ndarray) and lineout_box.size == 0):
                    if lineout_boxes:
                        lineout_box = lineout_boxes[np.argmax(lineout_confidences)]
                    else:
                        print("  ERROR: No lineout box available, skipping annotation")
                        continue
                
                # Get image size
                if not imsize:
                    imsize = (frame.shape[1], frame.shape[0])
                
                # Convert lineout_box to resized coordinates (800x450)2
                lineout_box = general.convert_coordinates(tuple(lineout_box), imsize)
                
                # Use current_display_frame (already resized to 800x450)
                resized_frame = current_display_frame.copy()

                player_result_gen = yolo_utils.perform_inference(resized_frame, player_model, False)

                try:
                    player_result = next(player_result_gen)
                except StopIteration:
                    player_result = None

                players = player.build_player_data(resized_frame, player_result)
                players = player.assign_teams_by_colour(players)
                players = player_tracker.update(players)




                player_dict = player.build_player_coord_dict(players)
                
                # IMPORTANT: Check for hooker first! If hooker detected, use hooker position (green dot)
                # Detect hooker on the current frame
                hooker_boxes_frame = [box for box, cls in zip(lineout_model_boxes, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Hooker']]
                hooker_confidences_frame = [conf for conf, cls in zip(lineout_model_confidences, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Hooker']]
                
                if hooker_boxes_frame and hooker_confidences_frame:
                    # Hooker detected! Use hooker's bottom center as lineout center (green dot)
                    hooker_box = hooker_boxes_frame[np.argmax(hooker_confidences_frame)]
                    hooker_box = general.convert_coordinates(tuple(hooker_box), imsize)
                    lineout_centre = general.box_bottom_centre(hooker_box)
                    print(f"  ✓ Hooker detected! Using hooker position (green dot): {lineout_centre}")
                else:
                    # No hooker - use lineout box bottom center
                    lineout_centre = general.box_bottom_centre(lineout_box)
                    print(f"  Using lineout box center: {lineout_centre}")
                
                # Get field lines on resized frame - these will give us the correct angle
                field_lines, field_outline = field.get_field_lines(resized_frame, lineout_centre=lineout_centre, exclusion_box=lineout_box, thresh=ui_thresh, is_path=False, visualise_steps=False)
                
                # Find intersection point - this is where the field lines converge
                intersection_point = line.find_average_intersection_point(field_lines)
                
                print(f"  Field lines detected: {len(field_lines)} lines, intersection at {intersection_point}")
                
                # Pick 4 field line intersection points for homography
                # Use detected field lines to estimate proper field points
                frame_height, frame_width = resized_frame.shape[:2]
                
                # Extract angle from the first few field lines to match perspective
                if len(field_lines) >= 2:
                    # Use the detected field lines to guide our field point estimation
                    line1 = field_lines[0]
                    angle = np.arctan2(line1[3] - line1[1], line1[2] - line1[0])
                    print(f"  Using field line angle: {np.degrees(angle):.1f}°")
                
                # Estimate 4 corner points based on typical rugby field perspective
                field_points = [
                    (int(frame_width * 0.20), int(frame_height * 0.35)),  # Top-left
                    (int(frame_width * 0.80), int(frame_height * 0.35)),  # Top-right
                    (int(frame_width * 0.80), int(frame_height * 0.70)),  # Bottom-right
                    (int(frame_width * 0.20), int(frame_height * 0.70))   # Bottom-left
                ]
                
                point_locations = [
                    ('left_tryline', 'top_5m'),
                    ('right_tryline', 'top_5m'),
                    ('right_tryline', 'bottom_5m'),
                    ('left_tryline', 'bottom_5m')
                ]
                
                # Get homography matrix and offside points
                H = points.get_homography_matrix(field_points, point_locations)
                left_offside_point, right_offside_point = points.get_lineout_offside_points(lineout_centre, H)
                
                # Create annotated frame
                lineout_frame = draw.draw_points(resized_frame, [lineout_centre], (0, 255, 0), show_image=False)
                lineout_frame = draw.draw_points(lineout_frame, [left_offside_point, right_offside_point], show_image=False)

                for p in players:
                    x1, y1, x2, y2 = p["box"]
                    cv2.putText(
                        lineout_frame,
                        str(p["track_id"]),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
                
                # Create offside lines
                left_offside_line = (intersection_point[0], intersection_point[1], left_offside_point[0], left_offside_point[1])
                right_offside_line = (intersection_point[0], intersection_point[1], right_offside_point[0], right_offside_point[1])
                centre_offside_line = (intersection_point[0], intersection_point[1], lineout_centre[0], lineout_centre[1])
                
                # Fit lines to field
                left_offside_line = field.fit_line_to_field(left_offside_line, field_outline)
                right_offside_line = field.fit_line_to_field(right_offside_line, field_outline)
                
                lineout_frame = draw.draw_lines(lineout_frame, [left_offside_line, right_offside_line], (255, 0, 0), show_image=False)
                
    
         
                
                if lineout_box is not None:
                    player_dict = offside.filter_for_offside_detection(player_dict, lineout_box, 0)
                    player_dict = offside.filter_detections_off_the_field(player_dict, lineout_box, imsize)
                
                offside_player_boxes = offside.get_players_between_lines(player_dict, left_offside_line, right_offside_line)
                
                # Draw player boxes
                lineout_frame = draw.draw_boxes(lineout_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False)
                
                # Log detection
                overlay_duration_frames = int(fps * 5)
                detection_log.append({
                    'type': 'LINEOUT',
                    'timestamp': detection_timestamp,
                    'frame': detection_frame_index,
                    'offside_count': len(offside_player_boxes),
                    'confidence': float(max(lineout_confidences)) if lineout_confidences else 0.0,
                    'overlay_start': detection_frame_index,
                    'overlay_end': detection_frame_index + overlay_duration_frames,
                    'annotated_frame': lineout_frame.copy()
                })
                
                # Display annotated frame
                general.display_frame(lineout_frame, paused_state, fps, window_title="Displaying lineout offside detections. Press 'P' to continue playing")
                
                print(f"  Lineout analysis complete: {len(offside_player_boxes)} offside players detected")
    
    # End of Pass 1 - Detection complete
    print(f"\n{'='*80}")
    print(f"PASS 1 COMPLETE: Detection and Analysis")
    print(f"Total detections found: {len(detection_log)}")
    print(f"{'='*80}\n")

    # PASS 2: Create annotated video with ALL frames
    if detection_log:
        print(f"{'='*80}")
        print(f"PASS 2: Creating Annotated Video")
        print(f"{'='*80}\n")
        print(f"Reading video file again to create output with all frames...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video file for pass 2")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (800, 450))

        frame_num = 1
        total_frames_written = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = cv2.resize(frame, (800, 450))

            overlay_frame = None
            for detection in detection_log:
                if detection['overlay_start'] <= frame_num <= detection['overlay_end']:
                    overlay_frame = detection['annotated_frame'].copy()
                    break

            if overlay_frame is not None:
                out.write(overlay_frame)
            else:
                out.write(display_frame)

            total_frames_written += 1
            frame_num += 1

            if total_frames_written % 100 == 0:
                print(f"  Processed {total_frames_written} frames...")

        cap.release()
        out.release()
        
        print(f"\n✓ Output video saved: {output_video_path}")
        print(f"  Total frames: {total_frames_written}")
        print(f"  Duration: {int(total_frames_written/fps//60)}:{int((total_frames_written/fps)%60):02d}")
        print(f"  Resolution: 800x450")
        print(f"  Frame rate: {fps} fps\n")
    
    # Generate and save detection log
    if session_stats.events:

        print(f"\nGenerating detection log...")

        write_detection_report(
            output_log_path=output_log_path,
            video_name=video_name,
            session_stats=session_stats
        )

        print(f"✓ Detection log saved: {output_log_path}")
        print(f"\n{'='*80}\n")

        print("SUMMARY:")
        print("\nPLAYER DETECTION SUMMARY:")
        print(f"  Rucks detected: {session_stats.ruck_count}")
        print(f"  Lineouts detected: {session_stats.lineout_count}")
        print(f"  Total offside incidents: {session_stats.total_offside_players}")
        print(f"  Output video: {output_video_path}")
        print(f"  Log file: {output_log_path}")
        print(f"\n{'='*80}\n")

    else:
        print("\nNo ruck or lineout detections found in video.")
        
        cv2.destroyAllWindows()

def write_detection_report(output_log_path, video_name, session_stats):
    """
    Writes the analysis report using the structured DetectionSessionStats
    and DetectionEvent objects instead of the old detection_log dictionary list.
    """

    with open(output_log_path, 'w') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("RUGBY OFFSIDE DETECTION - ANALYSIS LOG\n")
        log_file.write("=" * 80 + "\n\n")

        log_file.write(f"Video: {video_name}\n")
        log_file.write(f"Total Frames Analysed: {session_stats.total_frames}\n")

        minutes = int(session_stats.video_duration // 60)
        seconds = int(session_stats.video_duration % 60)

        log_file.write(f"Video Duration: {minutes}:{seconds:02d}\n")
        log_file.write(f"Frame Rate: {session_stats.fps} fps\n\n")

        log_file.write("=" * 80 + "\n")
        log_file.write("DETECTION SUMMARY\n")
        log_file.write("=" * 80 + "\n\n")

        log_file.write(f"Total Detections: {session_stats.total_detections}\n")
        log_file.write(f"  - Rucks: {session_stats.ruck_count}\n")
        log_file.write(f"  - Lineouts: {session_stats.lineout_count}\n")
        log_file.write(f"Total Offside Players Detected: {session_stats.total_offside_players}\n\n")
        log_file.write(f"Average Detection Confidence: {session_stats.average_confidence:.2%}\n\n")

        log_file.write("=" * 80 + "\n")
        log_file.write("DETAILED DETECTION LOG\n")
        log_file.write("=" * 80 + "\n\n")

        for i, event in enumerate(session_stats.events, 1):
            timestamp = event.timestamp
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)

            log_file.write(f"Detection #{i}: {event.event_type.upper()}\n")
            log_file.write(f"  Time: {minutes}:{seconds:02d}\n")
            log_file.write(f"  Frame: {event.frame_number}\n")
            log_file.write(f"  Confidence: {event.confidence:.2%}\n")
            log_file.write(f"  Offside Players: {event.offside_count}\n")

            log_file.write("\n")
            log_file.write("  Team Summary:\n")

            counts = event.team_counts or {}

            log_file.write(f"    Team 0: {counts.get('team_0', 0)}\n")
            log_file.write(f"    Team 1: {counts.get('team_1', 0)}\n")
            log_file.write(f"    Unknown: {counts.get('unknown_team', 0)}\n")
            log_file.write(f"    Referees: {counts.get('refs', 0)}\n")

            log_file.write("\n")
            log_file.write("  Tracked Players:\n")

            for tracked_player in event.tracked_players:
                track_id = tracked_player.get("track_id")
                team = tracked_player.get("team")
                box = tracked_player.get("box")

                log_file.write(f"    ID {track_id} -> Team {team}, Box: {box}\n")

            log_file.write("\n")

        log_file.write("=" * 80 + "\n")
        log_file.write("END OF LOG\n")
        log_file.write("=" * 80 + "\n")


def build_offside_player_dict(frame, player_model, player_tracker=None):
    """
    Run the player detection pipeline and convert the result into the
    coordinate dictionary format used by offside_functions.
    """

    player_result_gen = yolo_utils.perform_inference(frame, player_model, False)

    try:
        player_result = next(player_result_gen)
    except StopIteration:
        raise SystemExit("Player YOLO inference failed")

    players = player.build_player_data(frame, player_result)
    players = player.assign_teams_by_colour(players)

    if player_tracker is not None:
        players = player_tracker.update(players)

    player_dict = player.build_player_coord_dict(players)

    return player_dict


if __name__ == "__main__":
    main()