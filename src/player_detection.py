from sklearn.cluster import KMeans
import numpy as np


def detect_players(frame, player_model):
    """Detect players in the given frame using the provided player detection model.

    Args:
        frame (numpy.ndarray): The input video frame.
        player_model: The pre-trained player detection model.

    Returns:
        object: The raw detection result from the model for the frame.
    """
    results = player_model(frame)

    if not results:
        return None
    
    print("Player detection resultsdasds:", results)

    return results[0]


def filter_player_boxes(result):
    """Filter the detected player bounding boxes based on confidence scores.

    Args:
        result: The raw output from the player detection model.

    Returns:
        list: A list of filtered player bounding boxes.
    """
    if result is None:
        return []

    filtered_boxes = []
    CONF_THRESHOLD = 0.6

    for box in result.boxes:
        conf = float(box.conf[0])

        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        filtered_boxes.append((x1, y1, x2, y2))

    return filtered_boxes


def get_player_bottom_centre(filtered_boxes):
    """
    Calculate the bottom center point of each player's bounding box.

    Args:
        filtered_boxes (list): A list of filtered player bounding boxes.

    Returns:
        list: A list of player dictionaries with box and bottom centre.
    """
    players = []

    for box in filtered_boxes:
        x1, y1, x2, y2 = box
        bottom_centre = ((x1 + x2) // 2, y2)

        players.append({
            "box": box,
            "bottom_centre": bottom_centre
        })

    return players


def get_jersey_crop(frame, players):
    """
    Extract the jersey region for each player and add it to the player data.

    Args:
        frame (numpy.ndarray): The input video frame.
        players (list): A list of player dictionaries containing bounding boxes.

    Returns:
        list: The updated player list with jersey crops added.
    """
    frame_height, frame_width = frame.shape[:2]

    for player in players:
        x1, y1, x2, y2 = player["box"]
        box_height = y2 - y1

        jersey_x1 = max(0, x1)
        jersey_x2 = min(frame_width, x2)
        jersey_y1 = max(0, y1 + int(box_height * 0.2))
        jersey_y2 = min(frame_height, y1 + int(box_height * 0.55))

        jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]
        player["jersey_crop"] = jersey_crop

    return players


def extract_jersey_colour(players):
    """
    Extract the dominant jersey colour for each player and add it to the player data.

    Args:
        players (list): A list of player dictionaries containing jersey crops.

    Returns:
        list: The updated player list with dominant jersey colours added.
    """
    for player in players:
        jersey_crop = player.get("jersey_crop")

        if jersey_crop is None or jersey_crop.size == 0:
            player["jersey_colour"] = None
            continue

        avg_bgr = np.mean(jersey_crop, axis=(0, 1))
        b, g, r = avg_bgr

        player["jersey_colour"] = (int(r), int(g), int(b))

    return players


def build_player_data(frame, player_result):
    """
    Build player data including bounding box, bottom centre point,
    jersey crop, and jersey colour.

    Args:
        frame (numpy.ndarray): The input video frame.
        player_result: The raw output from the player detection model.

    Returns:
        list: A list of dictionaries containing player data.
    """
    filtered_boxes = filter_player_boxes(player_result)
    players = get_player_bottom_centre(filtered_boxes)
    players = get_jersey_crop(frame, players)
    players = extract_jersey_colour(players)

    return players


def assign_teams_by_colour(players):
    """
    Assign teams to players based on their dominant jersey colour.

    Args:
        players (list): A list of player dictionaries containing jersey colours.

    Returns:
        list: The updated player list with assigned team labels.
    """
    valid_players = [player for player in players if player.get("jersey_colour") is not None]

    if len(valid_players) < 2:
        for player in players:
            player["team"] = None
        return players

    colours = np.array([player["jersey_colour"] for player in valid_players])

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    team_labels = kmeans.fit_predict(colours)

    for player, label in zip(valid_players, team_labels):
        player["team"] = int(label)

    for player in players:
        if "team" not in player:
            player["team"] = None

    return players

def build_player_coord_dict(players):
    """
    Convert player list into a coordinate dictionary for offside detection.

    Args:
        players (list): A list of player dictionaries.

    Returns:
        dict: Mapping of player bottom-centre coordinates to their bounding boxes.
    """
    player_dict = {}

    for player in players:
        player_coord = player["bottom_centre"]
        player_box = player["box"]
        player_dict[player_coord] = player_box

    return player_dict