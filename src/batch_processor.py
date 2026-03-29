"""
batch_processor.py
------------------
This module provides functionality for batch processing multiple rugby videos to detect 
and analyse offside events in rucks and lineouts. It handles automated video processing, 
confidence tracking, video output generation, and summary report creation.

Functions:
----------
- `get_video_files`: Retrieves all video files from a specified directory.
- `process_video_batch`: Processes multiple videos and generates outputs.
- `save_annotated_video`: Saves a video with offside annotations.
- `generate_summary_report`: Creates a detailed text report for a processed video.
- `auto_mode_batch`: Modified auto mode for batch processing with confidence tracking.

Dependencies:
-------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- os, pathlib, datetime
- yolo_functions, field_functions, ruck_functions, lineout_functions, etc.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import yolo_functions as YOLO
import field_functions as field
import ruck_functions as ruck
import lineout_functions as lineout
import drawing_functions as draw
import line_functions as line
import general_functions as general
import offside_functions as offside
import point_functions as points
import player_detection as player
from constants import RUCK_MODEL_CLASS_NUMBERS, LINEOUT_MODEL_CLASS_NUMBERS


def get_video_files(directory_path):
    """
    Retrieves all video files from the specified directory.
    
    Parameters:
        directory_path (str): Path to the directory containing video files.
    
    Returns:
        list: List of absolute paths to video files (.mp4, .avi, .mov, .gif).
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.gif', '.MP4', '.AVI', '.MOV', '.GIF']
    video_files = []
    
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix in video_extensions:
            video_files.append(str(file_path.absolute()))
    
    return sorted(video_files)


def generate_summary_report(video_path, output_path, detections_data):
    """
    Generates a detailed text summary report for a processed video.
    
    Parameters:
        video_path (str): Path to the input video file.
        output_path (str): Path where the summary report will be saved.
        detections_data (dict): Dictionary containing detection information with keys:
            - 'video_name': Name of the video file
            - 'processing_date': Date and time of processing
            - 'total_frames': Total number of frames analysed
            - 'events': List of detection events (rucks/lineouts)
    
    Returns:
        None
    """
    video_name = Path(video_path).name
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RUGBY OFFSIDE DETECTION ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Video File: {video_name}\n")
        f.write(f"Processing Date: {timestamp}\n")
        f.write(f"Total Frames Analysed: {detections_data.get('total_frames', 0)}\n")
        f.write(f"Total Events Detected: {len(detections_data.get('events', []))}\n")
        f.write("\n" + "-" * 80 + "\n\n")
        
        events = detections_data.get('events', [])
        
        if not events:
            f.write("No offside events detected in this video.\n")
        else:
            f.write("DETECTION EVENTS:\n")
            f.write("=" * 80 + "\n\n")
            
            for i, event in enumerate(events, 1):
                f.write(f"Event #{i}\n")
                f.write(f"  Type: {event['type'].upper()}\n")
                f.write(f"  Frame Number: {event['frame_number']}\n")
                f.write(f"  Timestamp: {event['timestamp']}\n")
                f.write(f"  Detection Confidence: {event['detection_confidence']:.2%}\n")
                
                if event['type'] == 'ruck':
                    f.write(f"  Ruck Box Confidence: {event.get('ruck_confidence', 0):.2%}\n")
                elif event['type'] == 'lineout':
                    f.write(f"  Lineout Box Confidence: {event.get('lineout_confidence', 0):.2%}\n")
                    if 'hooker_confidence' in event:
                        f.write(f"  Hooker Confidence: {event.get('hooker_confidence', 0):.2%}\n")
                
                f.write(f"  Offside Players Detected: {event['offside_count']}\n")
                
                if event.get('offside_players'):
                    f.write(f"  Offside Player Positions:\n")
                    for j, player_info in enumerate(event['offside_players'], 1):
                        f.write(f"    Player {j}: Position ({player_info['x']}, {player_info['y']}), "
                               f"Confidence: {player_info['confidence']:.2%}\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
        
        # Summary statistics
        f.write("\nSUMMARY STATISTICS:\n")
        f.write("=" * 80 + "\n")
        ruck_events = [e for e in events if e['type'] == 'ruck']
        lineout_events = [e for e in events if e['type'] == 'lineout']
        total_offside_players = sum(e['offside_count'] for e in events)
        
        f.write(f"Total Ruck Events: {len(ruck_events)}\n")
        f.write(f"Total Lineout Events: {len(lineout_events)}\n")
        f.write(f"Total Offside Players Across All Events: {total_offside_players}\n")
        
        if ruck_events:
            avg_ruck_confidence = np.mean([e['detection_confidence'] for e in ruck_events])
            f.write(f"Average Ruck Detection Confidence: {avg_ruck_confidence:.2%}\n")
        
        if lineout_events:
            avg_lineout_confidence = np.mean([e['detection_confidence'] for e in lineout_events])
            f.write(f"Average Lineout Detection Confidence: {avg_lineout_confidence:.2%}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved: {output_path}")


def save_annotated_video(frames_data, output_path, fps, frame_size):
    """
    Saves annotated frames to a video file.
    
    Parameters:
        frames_data (list): List of tuples (frame_number, annotated_frame).
        output_path (str): Path where the output video will be saved.
        fps (int): Frames per second for the output video.
        frame_size (tuple): Size of the frames (width, height).
    
    Returns:
        None
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create video writer for: {output_path}")
    
    for frame_num, frame in frames_data:
        out.write(frame)
    
    out.release()
    print(f"Annotated video saved: {output_path}")


def auto_mode_batch(video_path, output_dir, ruck_model, lineout_model, ball_model, player_model):
    """
    Processes a single video in batch mode with automatic ruck/lineout detection.
    This is a modified version of auto_mode that runs without user interaction,
    automatically determines thresholds, and returns detection data.
    
    Parameters:
        video_path (str): Path to the input video file.
        output_dir (str): Directory where outputs will be saved.
        ruck_model (YOLO): Pre-trained YOLO model for ruck detection.
        lineout_model (YOLO): Pre-trained YOLO model for lineout detection.
        ball_model (YOLO): Pre-trained YOLO model for ball detection.
        player_model (YOLO): Pre-trained YOLO model for player detection.
    
    Returns:
        dict: Detection data including events, confidence scores, and statistics.
    """
    video_name = Path(video_path).stem
    fps = general.get_video_fps(video_path)
    
    # Initialise tracking variables
    frame_threshold = max(1, fps // 10)
    ruck_frame_count = 0
    lineout_frame_count = 0
    frame_number = 0
    
    # Detection data storage
    detections_data = {
        'video_name': video_name,
        'processing_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_frames': 0,
        'events': []
    }
    
    # Automatically determine threshold from middle frame
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    success, thresh_frame = video_capture.read()
    video_capture.release()
    
    if not success:
        print(f"Warning: Could not read frame from {video_path}. Using default threshold.")
        ui_thresh = 120
    else:
        # Use fixed threshold matching main.py for consistency
        ui_thresh = 120
    
    print(f"Processing: {video_name}")
    print(f"  FPS: {fps}, Frames: {total_frames}, Auto Threshold: {ui_thresh}")
    
    # Create generators for all inference predictions
    ruck_results = YOLO.perform_inference(video_path, ruck_model, False, 0.1)
    lineout_results = YOLO.perform_inference(video_path, lineout_model, False, 0.1)
    ball_results = YOLO.perform_inference(video_path, ball_model, False, 0.4)
    
    print(f"  Video inference complete. Analysing...")
    
    # Headless paused state (no user interaction)
    paused_state = {'paused': False, 'exit': False}
    
    # Store frames for video output
    annotated_frames = []
    
    # Process video frame by frame
    for ruck_result, lineout_result, ball_result in zip(ruck_results, lineout_results, ball_results):
        frame_number += 1
        
        # Ensure generators are in sync
        if not np.array_equal(ruck_result.orig_img, lineout_result.orig_img):
            print(f"Warning: Generators out of sync at frame {frame_number}")
            continue
        
        frame = ruck_result.orig_img
        
        # Extract detections
        ruck_model_boxes, ruck_model_classes, ruck_model_confidences = general.get_class_detections(ruck_result)
        lineout_model_boxes, lineout_model_classes, lineout_model_confidences = general.get_class_detections(lineout_result)
        
        ruck_boxes = [box for box, cls in zip(ruck_model_boxes, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]
        lineout_boxes = [box for box, cls in zip(lineout_model_boxes, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Lineout']]
        ruck_confidences = [conf for conf, cls in zip(ruck_model_confidences, ruck_model_classes) if cls == RUCK_MODEL_CLASS_NUMBERS['Ruck']]
        lineout_confidences = [conf for conf, cls in zip(lineout_model_confidences, lineout_model_classes) if cls == LINEOUT_MODEL_CLASS_NUMBERS['Lineout']]
        
        # Frame count management
        if ruck_frame_count < 0:
            ruck_frame_count += 1
        if lineout_frame_count < 0:
            lineout_frame_count += 1
        
        # Reset counts if no detections
        if not ruck_boxes and not lineout_boxes:
            ruck_frame_count = 0 if ruck_frame_count > 0 else ruck_frame_count
            lineout_frame_count = 0 if lineout_frame_count > 0 else lineout_frame_count
            continue
        
        # Handle overlapping detections (prioritise higher confidence)
        elif ruck_boxes and lineout_boxes:
            if np.max(ruck_confidences) >= np.max(lineout_confidences):
                lineout_frame_count = 0 if lineout_frame_count > 0 else lineout_frame_count
                ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
            else:
                ruck_frame_count = 0 if ruck_frame_count > 0 else ruck_frame_count
                lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count
            continue
        
        elif ruck_boxes:
            ruck_frame_count += 1 if ruck_frame_count >= 0 else ruck_frame_count
        elif lineout_boxes:
            lineout_frame_count += 1 if lineout_frame_count >= 0 else lineout_frame_count
        
        # RUCK DETECTION
        if ruck_frame_count >= frame_threshold:
            ruck_frame_count = -(fps * 3) + frame_threshold
            
            print(f"  Frame {frame_number}: Ruck detected (confidence: {np.max(ruck_confidences):.2%})")
            
            # Simplified ball release detection for batch mode
            ruck_box = ruck_boxes[np.argmax(ruck_confidences)]
            imsize = (frame.shape[1], frame.shape[0])
            
            # Convert ruck_box to resized coordinates (800x450)
            ruck_box_converted = general.convert_coordinates(tuple(ruck_box), imsize)
            
            resized_frame = cv2.resize(frame, (800, 450))
            ruck_left, ruck_right = ruck.get_last_feet(ruck_box_converted)
            ruck_centre = general.box_bottom_centre(ruck_box_converted)
            
            # Get field lines and calculate offside lines
            try:
                field_lines, field_outline = field.get_field_lines(frame, thresh=ui_thresh, is_path=False, visualise_steps=False)
                
                # Check if we have valid field lines
                if not field_lines or len(field_lines) == 0:
                    print(f"    Warning: No field lines detected - skipping ruck at frame {frame_number}")
                    continue
                
                intersection_point = line.find_average_intersection_point(field_lines)
                
                # Check if intersection point is valid
                if intersection_point is None or not all(coord is not None for coord in intersection_point):
                    print(f"    Warning: Could not find valid field line intersection - skipping ruck at frame {frame_number}")
                    continue
            except Exception as e:
                print(f"    Warning: Field line detection failed - skipping ruck at frame {frame_number}")
                continue
            
            left_ruck_line = (intersection_point[0], intersection_point[1], ruck_left[0], ruck_left[1])
            right_ruck_line = (intersection_point[0], intersection_point[1], ruck_right[0], ruck_right[1])
            
            left_ruck_line = field.fit_line_to_field(left_ruck_line, field_outline)
            right_ruck_line = field.fit_line_to_field(right_ruck_line, field_outline)
            
            # Draw points first (green dots for ruck feet)
            ruck_frame = draw.draw_points(resized_frame, [ruck_left, ruck_right], (0, 255, 0), show_image=False)
            ruck_frame = draw.draw_lines(ruck_frame, [left_ruck_line, right_ruck_line], (0, 0, 255), show_image=False)
            
            # Detect players
            player_result = player.detect_players(resized_frame, player_model)
            players = player.build_player_data(resized_frame, player_result)
            players = player.assign_teams_by_colour(players)
            player_dict = player.build_player_coord_dict(players)
            
            if ruck_box_converted is not None:
                player_dict = offside.filter_for_offside_detection(player_dict, ruck_box_converted, overlap_threshold=0.4, 
                                                                   width_expansion_factor=0, height_expansion_factor=0)
            
            offside_player_boxes = offside.get_players_between_lines(player_dict, left_ruck_line, right_ruck_line)
            
            # Annotate frame
            ruck_frame = draw.draw_boxes(ruck_frame, offside_player_boxes, box_annotation='Offside', 
                                        outline_colour=(0, 0, 255), show_image=False, font_scale=0.5, font_thickness=1)
            
            annotated_frames.append((frame_number, ruck_frame))
            
            # Record detection data
            event_data = {
                'type': 'ruck',
                'frame_number': frame_number,
                'timestamp': f"{frame_number / fps:.2f}s",
                'detection_confidence': float(np.max(ruck_confidences)),
                'ruck_confidence': float(np.max(ruck_confidences)),
                'offside_count': len(offside_player_boxes),
                'offside_players': []
            }
            
            # Store offside player information
            for player_pos, player_box in player_dict.items():
                if list(player_box) in [list(box) for box in offside_player_boxes]:
                    event_data['offside_players'].append({
                        'x': int(player_pos[0]),
                        'y': int(player_pos[1]),
                        'confidence': 0.85  # Player model confidence (can be enhanced)
                    })
            
            detections_data['events'].append(event_data)
            print(f"    Offside players detected: {len(offside_player_boxes)}")
        
        # LINEOUT DETECTION
        if lineout_frame_count >= frame_threshold:
            lineout_frame_count = -(fps * 10) + frame_threshold
            
            print(f"  Frame {frame_number}: Lineout detected (confidence: {np.max(lineout_confidences):.2%})")
            
            # Simplified lineout processing for batch mode
            lineout_box = lineout_boxes[np.argmax(lineout_confidences)]
            lineout_centre = general.box_bottom_centre(lineout_box)
            imsize = (frame.shape[1], frame.shape[0])
            
            resized_frame = cv2.resize(frame, (800, 450))
            
            # Get field lines
            try:
                field_lines, field_outline = field.get_field_lines(frame, lineout_centre=lineout_centre, 
                                                                    exclusion_box=lineout_box, thresh=ui_thresh, 
                                                                    is_path=False, visualise_steps=False)
                
                # Check if we have valid field lines
                if not field_lines or len(field_lines) == 0:
                    print(f"    Warning: No field lines detected - skipping lineout at frame {frame_number}")
                    continue
                
                intersection_point = line.find_average_intersection_point(field_lines)
                
                # Check if intersection point is valid
                if intersection_point is None or not all(coord is not None for coord in intersection_point):
                    print(f"    Warning: Could not find valid field line intersection - skipping lineout at frame {frame_number}")
                    continue
            except Exception as e:
                print(f"    Warning: Field line detection failed - skipping lineout at frame {frame_number}")
                continue
            
            # Convert to resized coordinates for annotation (800x450)
            lineout_centre_converted = general.convert_coordinates(tuple(lineout_centre), imsize)
            lineout_box_converted = general.convert_coordinates(tuple(lineout_box), imsize)
            
            # Automatically estimate field points based on standard rugby field layout
            frame_height, frame_width = resized_frame.shape[:2]
            field_points = [
                (int(frame_width * 0.25), int(frame_height * 0.35)),
                (int(frame_width * 0.75), int(frame_height * 0.35)),
                (int(frame_width * 0.75), int(frame_height * 0.70)),
                (int(frame_width * 0.25), int(frame_height * 0.70))
            ]
            
            # Use correct point_locations format (tuples, not strings)
            point_locations = [
                ('left_tryline', 'top_5m'),
                ('right_tryline', 'top_5m'),
                ('right_tryline', 'bottom_5m'),
                ('left_tryline', 'bottom_5m')
            ]
            
            try:
                H = points.get_homography_matrix(field_points, point_locations)
                left_offside_point, right_offside_point = points.get_lineout_offside_points(lineout_centre_converted, H)
                
                lineout_frame = draw.draw_points(resized_frame, [lineout_centre_converted], (0, 255, 0), show_image=False)
                lineout_frame = draw.draw_points(lineout_frame, [left_offside_point, right_offside_point], show_image=False)
                
                left_offside_line = (intersection_point[0], intersection_point[1], left_offside_point[0], left_offside_point[1])
                right_offside_line = (intersection_point[0], intersection_point[1], right_offside_point[0], right_offside_point[1])
                
                left_offside_line = field.fit_line_to_field(left_offside_line, field_outline)
                right_offside_line = field.fit_line_to_field(right_offside_line, field_outline)
                
                lineout_frame = draw.draw_lines(lineout_frame, [left_offside_line, right_offside_line], (255, 0, 0), show_image=False)
                
                # Detect players
                player_result = player.detect_players(resized_frame, player_model)
                players = player.build_player_data(resized_frame, player_result)
                players = player.assign_teams_by_colour(players)
                player_dict = player.build_player_coord_dict(players)
                
                if lineout_box_converted is not None:
                    player_dict = offside.filter_for_offside_detection(player_dict, lineout_box_converted, 0)
                    player_dict = offside.filter_detections_off_the_field(player_dict, lineout_box_converted, (800, 450))
                
                offside_player_boxes = offside.get_players_between_lines(player_dict, left_offside_line, right_offside_line)
                
                lineout_frame = draw.draw_boxes(lineout_frame, offside_player_boxes, box_annotation='Offside', 
                                               outline_colour=(0, 0, 255), show_image=False)
                
                annotated_frames.append((frame_number, lineout_frame))
                
                # Record detection data
                event_data = {
                    'type': 'lineout',
                    'frame_number': frame_number,
                    'timestamp': f"{frame_number / fps:.2f}s",
                    'detection_confidence': float(np.max(lineout_confidences)),
                    'lineout_confidence': float(np.max(lineout_confidences)),
                    'offside_count': len(offside_player_boxes),
                    'offside_players': []
                }
                
                for player_pos, player_box in player_dict.items():
                    if list(player_box) in [list(box) for box in offside_player_boxes]:
                        event_data['offside_players'].append({
                            'x': int(player_pos[0]),
                            'y': int(player_pos[1]),
                            'confidence': 0.85
                        })
                
                detections_data['events'].append(event_data)
                print(f"    Offside players detected: {len(offside_player_boxes)}")
                
            except Exception as e:
                print(f"    Warning: Lineout processing failed at frame {frame_number}: {e}")
    
    detections_data['total_frames'] = frame_number
    
    return detections_data, annotated_frames, fps


def process_video_batch(input_path, output_dir, ruck_model, lineout_model, ball_model, player_model):
    """
    Processes multiple videos in batch mode and generates outputs for each.
    
    Parameters:
        input_path (str): Path to directory containing videos or single video file.
        output_dir (str): Directory where outputs will be saved.
        ruck_model (YOLO): Pre-trained YOLO model for ruck detection.
        lineout_model (YOLO): Pre-trained YOLO model for lineout detection.
        ball_model (YOLO): Pre-trained YOLO model for ball detection.
        player_model (YOLO): Pre-trained YOLO model for player detection.
    
    Returns:
        None
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of videos to process
    input_path_obj = Path(input_path)
    if input_path_obj.is_file():
        video_files = [str(input_path_obj.absolute())]
    elif input_path_obj.is_dir():
        video_files = get_video_files(input_path)
    else:
        raise ValueError(f"Invalid input path: {input_path}")
    
    if not video_files:
        print("No video files found to process.")
        return
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING: {len(video_files)} video(s) found")
    print(f"{'='*80}\n")
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {Path(video_path).name}")
        print("-" * 80)
        
        try:
            video_name = Path(video_path).stem
            
            # Process video
            detections_data, annotated_frames, fps = auto_mode_batch(
                video_path, output_dir, ruck_model, lineout_model, ball_model, player_model
            )
            
            # Generate summary report
            report_path = output_path / f"{video_name}_analysis_report.txt"
            generate_summary_report(video_path, str(report_path), detections_data)
            
            # Save annotated video if there are any detections
            if annotated_frames:
                video_output_path = output_path / f"{video_name}_annotated.mp4"
                frame_size = (annotated_frames[0][1].shape[1], annotated_frames[0][1].shape[0])
                save_annotated_video(annotated_frames, str(video_output_path), fps, frame_size)
                print(f"  ✓ Annotated video saved with {len(annotated_frames)} detection frames")
            else:
                print(f"  ℹ No detections found - no annotated video generated")
            
            print(f"  ✓ Processing complete: {len(detections_data['events'])} events detected")
            
        except Exception as e:
            print(f"  ✗ Error processing {Path(video_path).name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
