from sklearn.cluster import KMeans
import numpy as np
import offside_functions as offside
import cv2


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
    


    return results[0]


def filter_player_boxes(result):
    """Filter detected bounding boxes based on confidence score."""
    if result is None:
        return []

    filtered_boxes = []
    CONF_THRESHOLD = 0.6

    for box in result.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        class_id = None
        class_name = "unknown"

        if hasattr(box, "cls") and box.cls is not None:
            class_id = int(box.cls[0])

            if hasattr(result, "names") and result.names is not None:
                class_name = str(result.names[class_id]).lower()

        filtered_boxes.append({
            "box": (x1, y1, x2, y2),
            "class_id": class_id,
            "class_name": class_name
        })

    return filtered_boxes

def get_player_bottom_centre(filtered_boxes):
    """
    Calculate the bottom centre point of each detected box.
    """
    players = []

    for item in filtered_boxes:
        x1, y1, x2, y2 = item["box"]
        bottom_centre = ((x1 + x2) // 2, y2)

        players.append({
            "box": item["box"],
            "bottom_centre": bottom_centre,
            "class_id": item["class_id"],
            "class_name": item["class_name"]
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
        jersey_y1 = max(0, y1 + int(box_height * 0.15))
        jersey_y2 = min(frame_height, y1 + int(box_height * 0.50))

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
        crop = player.get("jersey_crop")

        if crop is None or crop.size == 0:
            player["jersey_colour"] = None
            continue

        if crop.dtype != np.uint8:
            crop = crop.astype(np.uint8)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        _, _, v = cv2.split(hsv)
        dark_mask = v < 50

        invalid_mask = (green_mask > 0) | dark_mask
        valid_pixels = crop[~invalid_mask]

        pixels = valid_pixels.reshape(-1, 3)

        # fallback if everything got filtered out
        if len(pixels) == 0:
            pixels = crop.reshape(-1, 3)

        # still empty? (extreme edge case)
        if len(pixels) == 0:
            player["jersey_colour"] = None
            continue

        # small sample fallback
        if len(pixels) < 2:
            b, g, r = pixels[0]
            player["jersey_colour"] = (int(r), int(g), int(b))
            continue
        kmeans = KMeans(n_clusters=2, n_init=10)
        kmeans.fit(pixels)

        counts = np.bincount(kmeans.labels_)
        dominant_colour = kmeans.cluster_centers_[np.argmax(counts)]

        b, g, r = dominant_colour
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
    colours = [p["jersey_colour"] for p in players if p.get("jersey_colour") is not None]

    if len(colours) < 2:
        for p in players:
            p["team"] = None
        return players

    kmeans = KMeans(n_clusters=2, random_state=0).fit(colours)

    centres = kmeans.cluster_centers_

    # sort clusters by brightness so labels stay consistent
    brightness = [np.sum(c) for c in centres]
    sorted_indices = np.argsort(brightness)

    label_map = {
        sorted_indices[0]: 0,
        sorted_indices[1]: 1
    }

    i = 0
    for p in players:
        if p["jersey_colour"] is not None:
            raw_label = int(kmeans.labels_[i])
            p["team"] = label_map[raw_label]
            i += 1
        else:
            p["team"] = None

    return players

def get_offside_players(players, left_offside_line, right_offside_line):
    """
    Return the full player dictionaries for players between the two offside lines.

    Args:
        players (list): List of player dictionaries.
        left_offside_line (tuple): Left offside line.
        right_offside_line (tuple): Right offside line.

    Returns:
        list: Player dictionaries for offending players.
    """
    offside_players = []

    for player in players:
        player_position = player["bottom_centre"]

        if offside.check_between_lines(player_position, left_offside_line, right_offside_line):
            offside_players.append(player)

    return offside_players


def get_offside_player_boxes(offside_players):
    """
    Extract just the boxes for drawing.

    Args:
        offside_players (list): List of offside player dictionaries.

    Returns:
        list: Bounding boxes for offside players.
    """
    return [player["box"] for player in offside_players]


def get_offside_team_labels(offside_players):
    """
    Extract team labels for offending players.

    Args:
        offside_players (list): List of offside player dictionaries.

    Returns:
        list: Team labels for offending players.
    """
    return [player.get("team") for player in offside_players]

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

def count_teams_and_refs(players):
    """
    Count detected team members and referees.

    Returns:
        dict: {
            "team_0": int,
            "team_1": int,
            "unknown_team": int,
            "refs": int
        }
    """
    counts = {
        "team_0": 0,
        "team_1": 0,
        "unknown_team": 0,
        "refs": 0
    }

    for player in players:
        class_name = str(player.get("class_name", "")).lower()

        # Referee detection only works if your model has a separate ref class
        if class_name in ["ref", "referee", "official"]:
            counts["refs"] += 1
            continue

        team = player.get("team")

        if team == 0:
            counts["team_0"] += 1
        elif team == 1:
            counts["team_1"] += 1
        else:
            counts["unknown_team"] += 1

    return counts