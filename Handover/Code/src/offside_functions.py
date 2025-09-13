"""
offside_functions.py
---------------------

This module provides functions for detecting offside players in rugby using computer vision techniques. It includes utilities for extracting player positions, determining their relative positions on the field, and identifying offside players based on field lines. The module is designed to work with bounding box outputs from YOLO models and supports robust error handling for edge cases.

Functions:
----------
- `get_player_coord_dict`: Extracts the bottom-center coordinates of player bounding boxes from YOLO inference results.
- `check_between_lines`: Checks if a player is between two lines (i.e., offside) based on their coordinates.
- `get_players_between_lines`: Identifies players who are offside based on their positions relative to the offside lines.
- `filter_for_offside_detection`: Filters players who are outside the lineout region based on the lineout bounding box.
- `filter_detections_off_the_field`: Filters out player detections that are off the field based on their positions relative to a lineout box and the image size.

Dependencies:
-------------
- `general_functions`: A module providing utility functions for bounding box operations and geometric calculations.
"""
import general_functions as general

def get_player_coord_dict(players_result):
    """
    Extracts the bottom-center coordinates of player bounding boxes from YOLO inference results.

    Parameters:
        players_result (generator): YOLO inference results for player detection on a single frame.

    Returns:
        dict: Mapping of player bottom-centre coordinates to their bounding boxes.
    """

    player_dict = {}

    try:
        players = next(players_result)
    except:
        print("No players detected.")
        return player_dict
    
    player_boxes = players.boxes.xyxy.cpu().numpy()
    player_boxes = general.round_list_values(player_boxes)

    for player_box in player_boxes:
        player_coord = general.box_bottom_centre(player_box)
        player_dict.update({player_coord: player_box})
    
    return player_dict

def check_between_lines(player_points, left_line, right_line):
    """
    Checks if a player is between two lines (i.e., offside) based on their coordinates.

    Parameters:
        player_points (tuple): (x, y) coordinates of the player.
        left_line (tuple): (x1, y1, x2, y2) coordinates of the left line.
        right_line (tuple): (x1, y1, x2, y2) coordinates of the right line.

    Returns:
        bool: True if player is between the lines (offside), False otherwise.
    """

    threshold = 1e-9

    player_x, player_y = player_points
    left_line_x1, left_line_y1, left_line_x2, left_line_y2 = left_line
    right_line_x1, right_line_y1, right_line_x2, right_line_y2 = right_line

    left_val = (left_line_y2 - left_line_y1) * (player_x - left_line_x1) - (left_line_x2 - left_line_x1) * (player_y - left_line_y1)
    right_val = (right_line_y2 - right_line_y1) * (player_x - right_line_x1) - (right_line_x2 - right_line_x1) * (player_y - right_line_y1)
        
    return left_val >= threshold and right_val <= -threshold
   
    
def get_players_between_lines(player_dict, left_offside_line, right_offside_line):
    """
    Identify players who are offside based on their positions relative to the offside lines.

    Parameters:
        player_dict (dict): Mapping of player bottom-centre coordinates to their bounding boxes.
        left_offside_line (tuple): (x1, y1, x2, y2) coordinates of the left offside line.
        right_offside_line (tuple): (x1, y1, x2, y2) coordinates of the right offside line.

    Returns:
        list: List of bounding boxes for players who are offside.
    """

    offside_player_boxes = []

    for player_position, box in player_dict.items():
        if check_between_lines(player_position, left_offside_line, right_offside_line):
            
            offside_player_boxes.append(box)

    return offside_player_boxes


def filter_for_offside_detection(player_dict, roi, overlap_threshold=0.5, width_expansion_factor=0.5, height_expansion_factor=1.0):
    """
    Filters players who are outside the lineout region based on the lineout bounding box.

    Parameters:
        player_dict (dict): Dictionary mapping player coordinates to their bounding boxes.
        roi (tuple): Bounding box of the lineout region (x1, y1, x2, y2).
        overlap_threshold (float, optional): Minimum overlap ratio to consider a player outside the region. Defaults to 0.5.
        width_expansion_factor (float, optional): Factor to expand the width of the ROI for filtering. Defaults to 0.5.
        height_expansion_factor (float, optional): Factor to expand the height of the ROI for filtering. Defaults to 1.0.

    Returns:
        dict: Filtered player dictionary with players outside the expanded lineout region.
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1

    # Expand the ROI region slightly
    expanded_roi = [
        roi_x1 - roi_width * width_expansion_factor,
        roi_y1 - roi_height * height_expansion_factor,
        roi_x2 + roi_width * width_expansion_factor,
        roi_y2 + roi_height * height_expansion_factor
    ]

    player_coords = list(player_dict.keys())
    player_boxes = list(player_dict.values())

    # Filter players whose bounding boxes do not overlap with the ROI region
    filtered_player_dict = {
        coord: box for coord, box in zip(player_coords, player_boxes)
        if not general.box_overlap(box, expanded_roi, overlap_threshold)
    }

    return filtered_player_dict

def filter_detections_off_the_field(player_dict, lineout_box, imsize):
    """
    Filters out player detections that are off the field based on their positions 
    relative to a lineout box and the image size.
    This function determines whether the lineout box is in the top or bottom half 
    of the image and filters player detections accordingly. Players are retained 
    if their positions are within the valid region defined by the lineout box.

    Parameters:
        player_dict (dict): A dictionary where keys are player coordinates (tuples of x, y)
                            and values are bounding boxes (tuples representing box dimensions).
        lineout_box (tuple): A tuple representing the bounding box of the lineout area.
        imsize (tuple): A tuple (width, height) representing the size of the image.
        
    Returns:
        dict: A filtered dictionary of player detections where keys are player coordinates 
              and values are bounding boxes, containing only players within the valid region.
    """
    lineout_centre = general.box_centre(lineout_box)
    lineout_bottom_centre = general.box_bottom_centre(lineout_box)
    lineout_top_centre = general.box_top_centre(lineout_box)
    _, im_height = imsize

    player_coords = list(player_dict.keys())
    player_boxes = list(player_dict.values())

    if lineout_centre[1] >= im_height // 2:
        filtered_player_dict = {
            coord: box for coord, box in zip(player_coords, player_boxes)
            if coord[1] <= lineout_bottom_centre[1]
        }
    
    else:
        filtered_player_dict = {
            coord: box for coord, box in zip(player_coords, player_boxes)
            if coord[1] >= lineout_top_centre[1]
        }

    return filtered_player_dict