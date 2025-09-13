import cv2
import numpy as np
import ui_functions as UI
import yolo_functions as YOLO
import field_functions as field
import ruck_functions as ruck
import lineout_functions as lineout
import drawing_functions as draw
import line_functions as line
import general_functions as general
import offside_functions as offside
import point_functions as points
from constants import RUCK_MODEL_CLASS_NUMBERS, LINEOUT_MODEL_CLASS_NUMBERS

def main():
    """
    Main function to execute the offside detection program.
    This function allows the user to select a video file for processing, validates the file type,
    loads the necessary YOLO models for detection, and provides an option to run the program
    in either manual or automatic mode.
    Steps:
    1. Prompts the user to select a video file through a graphical user interface.
    2. Validates that the selected file is a video.
    3. Loads YOLO models for ruck, lineout, and player detection.
    4. Asks the user to choose between manual or automatic processing mode.
    5. Executes the selected mode with the provided video and models.
    Raises:
        SystemExit: If the selected file is not a valid video file.
    Inputs:
        - Video file path selected by the user.
        - User input to choose between manual or automatic mode.
    Outputs:
        - Processes the video file using the selected mode and detection models.
    """

    # Get the video path from a graphical user input
    video_path = UI.select_file()
    is_video = UI.check_is_video(video_path)
    
    # Ensure the selected file was a video
    if not is_video:
        raise SystemExit("Selected file is not a video. Please select a video file (.mp4, .avi, .mov, .gif) for offside detection. Exiting...")

    fps = general.get_video_fps(video_path)
    
    print(f"Selected video path: {video_path}")

    # Loading ruck, lineout, ball and player detection models
    ruck_model = YOLO.load_model('./models/ruck.pt')
    lineout_model = YOLO.load_model('./models/lineout.pt')
    ball_model = YOLO.load_model('./models/ball.pt')
    player_model = YOLO.load_model('./models/yolo11n.pt') 

    print("YOLO models loaded")

    manual_or_auto = ''

    while True:
        manual_or_auto = input("Do you want to run in manual or auto mode: ").lower()
        if manual_or_auto in ['manual', 'auto', 'automatic']:
            break
        print("Please type manual, auto or automatic to start processing.")

    if manual_or_auto == 'manual':
        manual_mode(video_path, fps, ruck_model, lineout_model, player_model)
    else:
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
            ruck_result = YOLO.perform_inference(frame, ruck_model, False)

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

            players_result = YOLO.perform_inference(frame, player_model, show_output=False)

            player_dict = offside.get_player_coord_dict(players_result)

            offside_player_boxes = offside.get_players_between_lines(player_dict, left_ruck_line, right_ruck_line)

            # Draw the player box in red for offside players
            ruck_frame = draw.draw_boxes(ruck_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False)

            manual_pause_state['paused'] = True
            manual_pause_state['ruck'] = False

            # Display annotated ruck frame
            general.display_frame_manual(ruck_frame, manual_pause_state, fps, window_id='Frame', window_title="Displaying ruck offside detections. Press 'P' to continue playing.")


        if manual_pause_state['lineout']:
            # CHECK FOR HOOKER AND/OR LINEOUT IN IMAGE, OTHERWISE
            lineout_result = YOLO.perform_inference(frame, lineout_model, False)
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

            left_lineout_line = (intersection_point[0], intersection_point[1], left_offside_point[0], left_offside_point[1])
            right_lineout_line = (intersection_point[0], intersection_point[1], right_offside_point[0], right_offside_point[1])
        
            left_lineout_line = field.fit_line_to_field(left_lineout_line, field_outline)
            right_lineout_line = field.fit_line_to_field(right_lineout_line, field_outline)

            lineout_frame = draw.draw_lines(lineout_frame, [left_lineout_line, right_lineout_line], (255, 0, 0), show_image=False)

            players_result = YOLO.perform_inference(frame, player_model, show_output=False)

            player_dict = offside.get_player_coord_dict(players_result)

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
    Processes a video to detect and analyze rucks and lineouts in rugby matches using YOLO models.
    This function performs inference on a video file using pre-trained YOLO models for ruck, lineout, 
    ball, and player detection. It identifies rucks and lineouts, detects offside players, and visualizes 
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

    frame_threshold = fps // 10

    # Initialise ruck and lineout frame counts
    ruck_frame_count = 0
    lineout_frame_count = 0
    
    # Get the threshold value from user input from a frame in the middle of the video
    thresh_frame = UI.get_frame(video_path)
    ui_thresh = UI.threshold_slider(thresh_frame)

    print(f"Selected threshold value: {ui_thresh}")

    # Create generators for all inference predictions from the ruck and lineout models
    ruck_results = YOLO.perform_inference(video_path, ruck_model, False, 0.1)
    lineout_results = YOLO.perform_inference(video_path, lineout_model, False, 0.1) # Also includes hooker detection
    ball_results = YOLO.perform_inference(video_path, ball_model, False, 0.4) # Ball detection

    print("Video inference complete")
    print("Commencing analysis...")

    # Iterate through the ruck and lineout result generators (each generator will yield the same number of frames)
    for ruck_result, lineout_result, _ in zip(ruck_results, lineout_results, ball_results):

        # Ensure the generators are in sync by checking the original images
        if not np.array_equal(ruck_result.orig_img, lineout_result.orig_img):
            raise AssertionError("Generators are out of sync. Please check the video file and models. Exiting...")

        # Check if the user has requested to exit the video processing
        if paused_state['exit']:
            print("Exiting video processing.")
            break
        
        # Display the current frame
        frame = ruck_result.orig_img
        general.display_frame(frame, paused_state, 1000, window_title="Searching for a ruck or lineout. Press 'P' to pause or 'Q' to quit")

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
            continue
        
        # If both ruck and lineout boxes are detected, check the confidence levels
        elif ruck_boxes and lineout_boxes:
            if np.max(ruck_confidences) >= np.max(lineout_confidences):
                lineout_frame_count = 0 if lineout_frame_count > 0 else lineout_frame_count
                ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
                continue
            
            else:
                ruck_frame_count = 0 if ruck_frame_count > 0 else ruck_frame_count
                lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count
                continue

        # If only ruck boxes are detected, increment the ruck frame count
        elif ruck_boxes:
            ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
        
        # If only lineout boxes are detected, increment the lineout frame count
        elif lineout_boxes:
            lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count

        # If a ruck is detected for three consecutive frames, process the ruck results
        if ruck_frame_count >= frame_threshold:
            # Set a three second cooldown after a ruck is detected
            ruck_frame_count = -(fps * 3) + frame_threshold

            print('Ruck detected!')
            print('Starting ball detection...')

            # Continues iterating through ruck generator until the ball is released or the ruck is no longer detected
            ruck_result, ruck_box, imsize = ruck.get_ball_release(ruck_results, ball_results, lineout_results, paused_state, fps)

            # If the user has requested to exit the video processing, break the loop
            if paused_state['exit']:
                break

            if not ruck_box:
                ruck_box = ruck_boxes[np.argmax(ruck_confidences)] if ruck_boxes else None

                # This is a fallback but should never be reached
                if ruck_box is None or ruck_box.size == 0:
                    raise SystemExit('Error, no ruck could not be detected. Exiting ...')
            
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
                    players_result = YOLO.perform_inference(resized_frame, player_model, show_output=False, conf=0.1)

                    player_dict = offside.get_player_coord_dict(players_result)

                    if ruck_box is not None:
                        player_dict = offside.filter_for_offside_detection(player_dict, ruck_box, overlap_threshold=0.4, width_expansion_factor=0, height_expansion_factor=0)

                    offside_player_boxes = offside.get_players_between_lines(player_dict, left_ruck_line, right_ruck_line)

                    # Draw the player box in red for offside players
                    ruck_frame = draw.draw_boxes(ruck_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False, font_scale=0.5, font_thickness=1)

                    # Display annotated ruck frame
                    general.display_frame(ruck_frame, paused_state, fps, window_title="Displaying ruck offside detections. Press 'P' to continue playing")

                    if isinstance(ruck_result, list):
                        for result in ruck_result[1:]:
                            frame = result.orig_img
                            general.display_frame(frame, paused_state, fps, window_title="Searching for a ruck or lineout. Press 'P' to pause or 'Q' to quit")

                else:
                    print("Offside lines not drawn as no valid intersection was detected.")

        # If a lineout is detected for three consecutive frames, process the lineout results
        if lineout_frame_count >= frame_threshold:
            # Set a 10 second cooldown after a lineout is detected
            lineout_frame_count = -(fps * 10) + frame_threshold
                
            print('Lineout detected!')
            print('Starting ball detection...')

            # Continues iterating through lineout generator until the ball is released or the lineout is no longer detected
            lineout_result, lineout_box, lineout_centre, imsize = lineout.get_ball_release(lineout_results, ruck_results, ball_results, paused_state, fps)

            # If the user has requested to exit the video processing, break the loop
            if paused_state['exit']:
                break

            # If the centre of the lineout could not be found during the ball release processing, use the initially detected frame instead
            if not lineout_centre:
                if lineout_box:
                    lineout_centre = general.box_bottom_centre(lineout_box)
                else:
                    lineout_box = lineout_boxes[np.argmax(lineout_confidences)] if lineout_boxes else None

                    if lineout_box:
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

                field_points = []

                point_frame = resized_frame.copy()

                for _ in range(4):
                    field_points.append(UI.get_coordinates(point_frame, window_id='Frame', window_title='Select four crosspoints of the field lines'))
                
                point_locations = UI.get_point_locations(point_frame, field_points)

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

                # Run player detection on the current frame
                players_result = YOLO.perform_inference(resized_frame, player_model, show_output=False)

                player_dict = offside.get_player_coord_dict(players_result)

                if lineout_box is not None: 
                    player_dict = offside.filter_for_offside_detection(player_dict, lineout_box, 0)
                    player_dict = offside.filter_detections_off_the_field(player_dict, lineout_box, imsize)

                offside_player_boxes = offside.get_players_between_lines(player_dict, left_offside_line, right_offside_line)

                # Draw the player box in red for offside players
                lineout_frame = draw.draw_boxes(lineout_frame, offside_player_boxes, box_annotation='Offside', outline_colour=(0, 0, 255), show_image=False)

                # Display annotated lineout frame
                general.display_frame(lineout_frame, paused_state, fps, window_title="Displaying lineout offside detections. Press 'P' to continue playing")

                if isinstance(lineout_result, list):
                    for result in lineout_result[1:]:
                        frame = result.orig_img
                        general.display_frame(frame, paused_state, fps, window_title="Searching for a ruck or lineout. Press 'P' to pause or 'Q' to quit")


if __name__ == "__main__":
    main()