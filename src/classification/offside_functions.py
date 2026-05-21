"""
offside_functions.py
---------------------

This module provides functions for detecting offside players in rugby using
player coordinate dictionaries and geometric checks.

Functions:
----------
- `check_between_lines`: Checks if a player is between two lines.
- `get_players_between_lines`: Identifies players who are offside.
- `filter_for_offside_detection`: Filters players outside a region of interest.
- `filter_detections_off_the_field`: Filters out players off the field.

Dependencies:
-------------
- `general_functions`: Utility functions for bounding box operations.
"""

from src.utils import general_functions as general


def check_between_lines(player_points, left_line, right_line):
    """
    Checks if a player is between two lines (i.e., offside) based on their coordinates.

    Parameters:
        player_points (tuple): (x, y) coordinates of the player.
        left_line (tuple): (x1, y1, x2, y2) coordinates of the left line.
        right_line (tuple): (x1, y1, x2, y2) coordinates of the right line.

    Returns:
        bool: True if player is between the lines, False otherwise.
    """
    threshold = 1e-9

    player_x, player_y = player_points
    left_line_x1, left_line_y1, left_line_x2, left_line_y2 = left_line
    right_line_x1, right_line_y1, right_line_x2, right_line_y2 = right_line

    left_val = (
        (left_line_y2 - left_line_y1) * (player_x - left_line_x1)
        - (left_line_x2 - left_line_x1) * (player_y - left_line_y1)
    )

    right_val = (
        (right_line_y2 - right_line_y1) * (player_x - right_line_x1)
        - (right_line_x2 - right_line_x1) * (player_y - right_line_y1)
    )

    return left_val >= threshold and right_val <= -threshold


def get_players_between_lines(player_dict, left_offside_line, right_offside_line):
    """
    Identify players who are offside based on their positions relative to the offside lines.

    Parameters:
        player_dict (dict): Mapping of player bottom-centre coordinates to bounding boxes.
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


def filter_for_offside_detection(
    player_dict,
    roi,
    overlap_threshold=0.5,
    width_expansion_factor=0.5,
    height_expansion_factor=1.0
):
    """
    Filters players whose boxes overlap with the expanded ROI.

    Parameters:
        player_dict (dict): Dictionary mapping player coordinates to bounding boxes.
        roi (tuple): Bounding box of the region of interest (x1, y1, x2, y2).
        overlap_threshold (float, optional): Minimum overlap ratio to filter out a player.
        width_expansion_factor (float, optional): Width expansion factor for ROI.
        height_expansion_factor (float, optional): Height expansion factor for ROI.

    Returns:
        dict: Filtered player dictionary.
    """
    if roi is None:
        return player_dict

    roi_x1, roi_y1, roi_x2, roi_y2 = roi

    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1

    expanded_roi = [
        roi_x1 - roi_width * width_expansion_factor,
        roi_y1 - roi_height * height_expansion_factor,
        roi_x2 + roi_width * width_expansion_factor,
        roi_y2 + roi_height * height_expansion_factor
    ]

    filtered_player_dict = {
        coord: box
        for coord, box in player_dict.items()
        if not general.box_overlap(box, expanded_roi, overlap_threshold)
    }

    return filtered_player_dict


def filter_detections_off_the_field(player_dict, lineout_box, imsize):
    """
    Filters out player detections that are off the field based on their positions
    relative to a lineout box and the image size.

    Parameters:
        player_dict (dict): Dictionary where keys are player coordinates (x, y)
                            and values are bounding boxes.
        lineout_box (tuple): Bounding box of the lineout area.
        imsize (tuple): (width, height) of the image.

    Returns:
        dict: Filtered dictionary of player detections.
    """
    if lineout_box is None:
        return player_dict

    lineout_centre = general.box_centre(lineout_box)
    lineout_bottom_centre = general.box_bottom_centre(lineout_box)
    lineout_top_centre = general.box_top_centre(lineout_box)
    _, im_height = imsize

    if lineout_centre[1] >= im_height // 2:
        filtered_player_dict = {
            coord: box
            for coord, box in player_dict.items()
            if coord[1] <= lineout_bottom_centre[1]
        }
    else:
        filtered_player_dict = {
            coord: box
            for coord, box in player_dict.items()
            if coord[1] >= lineout_top_centre[1]
        }

    return filtered_player_dict