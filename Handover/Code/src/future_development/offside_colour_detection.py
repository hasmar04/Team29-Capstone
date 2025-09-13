"""
offside_colour_detection.py
--------------------------
This module provides functionality for sorting players into teams based on the dominant colours of their jerseys and processing offsides based on this information. It uses computer vision techniques, including YOLO for player detection and k-means clustering for colour analysis, to classify players into teams or mark them as 'Other' if their jersey colours do not match the identified team colours.

Key Functions:
---------------
- `get_player_colour_dict`: Extracts dominant colours from player bounding boxes and sorts players into teams using colour clustering.
- `get_dominant_colour_kmeans`: Determines the dominant colour in a region of interest (ROI) using k-means clustering.
- `find_team_colours`: Identifies the two dominant team colours from a list of player colours using k-means clustering.
- `sort_into_teams`: Assigns players to teams or 'Other' based on proximity of their dominant colour to the team colours.
- `get_player_side`: Determines which side of a line a player is on using the cross product.
- `get_offside_players`: Identifies offside players for each team based on their positions relative to the ruck and offside lines.

Dependencies:
---------------
- OpenCV (`cv2`)
- NumPy (`numpy`)
- scikit-learn (`KMeans`)
- general_functions

Note: This module is part of future development and is not currently integrated into the main project.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
import general_functions as general


def get_player_colour_dict(players_result, frame):
    """
    Extracts the bottom-center points and dominant colours of player bounding boxes from YOLO inference results, then sorts players into teams based on colour clustering.

    Parameters:
        players_result (generator): YOLO inference results for player detection on a single frame.
        frame (numpy.ndarray): The image frame from which to extract player ROIs for colour analysis.

    Returns:
        dict: Dictionary with keys 'Team 1', 'Team 2', and 'Other', each mapping to dicts of player box coordinates and their dominant colours.
    """

    try:
        players = next(players_result)
    except:
        print("No players detected.")
        return {'Team 1': {}, 'Team 2': {}, 'Other': {}}
    
    player_dict = {}
    player_colours = []
    
    player_boxes = players.boxes.xyxy.cpu().numpy()
    player_boxes = general.round_list_values(player_boxes)

    for player_box in player_boxes:
        x1, y1, x2, y2 = player_box
        # Extract ROI for dominant colour detection

        height = y2 - y1
        top_half_y2 = y1 + height // 2

        roi_width = (x2 - x1) // 2
        roi_height = (top_half_y2 - y1) // 2

        roi_y1 = y1 + (top_half_y2 - y1 - roi_height) // 2
        roi_y2 = roi_y1 + roi_height
        roi_x1 = x1 + (x2 - x1 - roi_width) // 2
        roi_x2 = roi_x1 + roi_width
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        dominant_colour = None

        if len(roi) >= 4:
            # cv2.imshow(f"ROI", roi)
            # cv2.waitKey(0)
            dominant_colour = get_dominant_colour_kmeans(roi, k=4, brightness_threshold=0)
            player_colours.append(dominant_colour)

        player_dict.update({tuple(player_box): dominant_colour})

    # Determine team colours if not already set
    if len(player_colours) >= 2:
        team_colours = find_team_colours(player_colours)

        return sort_into_teams(player_dict, team_colours, 40)

    else:
        return {'Team 1': {}, 'Team 2': {}, 'Other': {}}


def get_dominant_colour_kmeans(roi, k=8, brightness_threshold=0):
    """
    Get the dominant colour from the region of interest (ROI) using k-means clustering.

    Parameters:
        roi (numpy.ndarray): The region of interest (image) to analyse.
        k (int, optional): The number of clusters for k-means. Defaults to 8.
        brightness_threshold (int, optional): Minimum brightness threshold for valid colours. Defaults to 0.

    Returns:
        tuple: The dominant colour in BGR format as a tuple of integers.
    """
    # Reshape the ROI to a 2D array (each pixel as a row)
    roi_flattened = roi.reshape((-1, 3))
    
    # Check if the ROI is empty
    if roi_flattened.size == 0:
        return (0, 0, 0)  # Return black if the ROI is empty
    
    # Convert to float32 for k-means
    roi_flattened = np.float32(roi_flattened)
    
    # Define criteria and apply k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centres = cv2.kmeans(roi_flattened, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Count the number of pixels assigned to each cluster
    _, counts = np.unique(labels, return_counts=True)
    
    # Filter out dark colours based on the brightness threshold
    filtered_centres = [centre for centre in centres if np.mean(centre) > brightness_threshold]
    filtered_counts = [counts[i] for i, centre in enumerate(centres) if np.mean(centre) > brightness_threshold]
    
    if not filtered_centres:
        return (0, 0, 0)  # Return black if no valid colours are found
    
    # Get the index of the most frequent cluster
    dominant_index = np.argmax(filtered_counts)
    
    # Get the dominant colour (convert to integer values)
    dominant_colour_bgr = filtered_centres[dominant_index].astype(int)
    
    return tuple(dominant_colour_bgr)



def find_team_colours(colours):
    """
    Identifies the dominant team colours from a given list of colours using k-means clustering.

    Parameters:
        colours (list of list or tuple of int): List of BGR colour values for each player.

    Returns:
        list: Two dominant team colours as tuples (B, G, R).
    """

    colours_array = np.array(colours)

    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(colours_array)

    dominant_colors = kmeans.cluster_centers_.astype(int)
        
    return [tuple(colour) for colour in dominant_colors]

    
def sort_into_teams(player_dict, team_colours, threshold=120):
    """
    Sorts players into two teams or 'Other' based on proximity of their dominant colour to the team colours.

    Parameters:
        player_dict (dict): Mapping of player bounding boxes to their dominant colours.
        team_colours (list): List of two team colour tuples.
        threshold (float, optional): Distance threshold for assigning to 'Other'. Defaults to 120.

    Returns:
        dict: Dictionary with keys 'Team 1', 'Team 2', and 'Other', each mapping to dicts of player centre coordinates and their boxes.
    """

    team_1 = {}
    team_2 = {}
    other = {}

    for box, colour in player_dict.items():
        box_centre = general.box_bottom_centre(box)

        if colour is not None:
            distance_1 = np.linalg.norm(np.array(team_colours[0]) - np.array(colour))
            distance_2 = np.linalg.norm(np.array(team_colours[1] - np.array(colour)))

            if distance_1 > threshold and distance_2 > threshold:
                other.update({box_centre: box})
            elif distance_1 < distance_2:
                team_1.update({box_centre: box})
            else:
                team_2.update({box_centre: box})
        else:
            other.update({box_centre: box})
    
    return {"Team 1": team_1, "Team 2": team_2, "Other": other}

def get_player_side(player_points, line):
    """
    Determines which side of a line a player is on using the cross product.

    Parameters:
        player_points (tuple): (x, y) coordinates of the player.
        line (tuple): (x1, y1, x2, y2) coordinates of the line.

    Returns:
        str: 'right' if player is on the right side, 'left' if on the left.
    """

    threshold = 1e-9  # A small threshold to avoid floating-point precision issues
    player_x, player_y = player_points
    line_x1, line_y1, line_x2, line_y2 = line

    # Calculate the cross product to determine the relative position
    val = (line_y2 - line_y1) * (player_x - line_x1) - (line_x2 - line_x1) * (player_y - line_y1)

    # If the value is greater than or equal to 0, the player is in front of the left line
    if val >= threshold:
        return "right"
    elif val <= -threshold:
        return "left"

def get_offside_players(centre_line, sorted_player_dict, left_line, right_line):
    """
    Identifies offside players for each team based on their positions relative to the ruck and offside lines.

    Parameters:
        centre_line (tuple): (x1, y1, x2, y2) coordinates of the ruck centre line.
        sorted_player_dict (dict): Dictionary of players sorted into teams.
        left_line (tuple): (x1, y1, x2, y2) coordinates of the left offside line.
        right_line (tuple): (x1, y1, x2, y2) coordinates of the right offside line.

    Returns:
        list: List of bounding boxes for offside players.
    """

    # List to keep track of offside player boxes
    offside_player_boxes = []

    # Count players on each side of the ruck for each team
    team_counts = {
        'left': {'Team 1': 0, 'Team 2': 0},
        'right': {'Team 1': 0, 'Team 2': 0}
    }

    for team in ['Team 1', 'Team 2']:
        team_players = sorted_player_dict[team].items()
        for player_centre, _ in team_players:
            side = get_player_side(player_centre, centre_line)
            team_counts[side][team] += 1
    
    # Determine which team dominates each side
    left_dominant = 'Team 1' if team_counts['left']['Team 1'] > team_counts['left']['Team 2'] else 'Team 2'
    right_dominant = 'Team 1' if team_counts['right']['Team 1'] > team_counts['right']['Team 2'] else 'Team 2'

    for team in ['Team 1', 'Team 2']:
        # Get all players from the team
        team_players = sorted_player_dict[team].items()

        # Iterate through each player on the team
        for player_centre, player_box in team_players:
            # If the current team is the left dominant one, then anyone in front of the left line is offside
            if team == left_dominant:
                if get_player_side(player_centre, right_line) == 'right':
                    offside_player_boxes.append(player_box)
            
            # If the current team is the right dominant one, then anyone in front of the right line is offside
            elif team == right_dominant:
                if get_player_side(player_centre, left_line) == 'left':
                    offside_player_boxes.append(player_box)


    return offside_player_boxes
