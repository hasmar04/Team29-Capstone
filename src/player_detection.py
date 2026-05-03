from sklearn.cluster import KMeans
import numpy as np
from ultralytics import YOLO
import offside_functions as offside
import cv2


def filter_player_boxes(result):
    if result is None:
        return []

    filtered_boxes = []
    CONF_THRESHOLD = 0.4

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
    players = []

    for item in filtered_boxes:
        if item["class_name"] not in ["player", "person"]:
            continue

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
    frame_height, frame_width = frame.shape[:2]

    for player in players:
        x1, y1, x2, y2 = player["box"]

        box_width = x2 - x1
        box_height = y2 - y1

        jersey_x1 = max(0, x1 + int(box_width * 0.30))
        jersey_x2 = min(frame_width, x2 - int(box_width * 0.30))

        jersey_y1 = max(0, y1 + int(box_height * 0.20))
        jersey_y2 = min(frame_height, y1 + int(box_height * 0.60))

        jersey_crop = frame[jersey_y1:jersey_y2, jersey_x1:jersey_x2]

        player["jersey_crop"] = jersey_crop
        player["jersey_crop_box"] = (jersey_x1, jersey_y1, jersey_x2, jersey_y2)

    return players


def extract_jersey_colour(players):
    """
    Extract jersey colour with adaptive crop shifting if colour is unreliable.
    """

    def get_colour_from_crop(crop):
        if crop is None or crop.size == 0:
            return None, 0

        crop = crop.astype(np.uint8)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Remove grass
        green_mask = (h > 35) & (h < 85) & (s > 60)
        valid_mask = ~green_mask

        pixels = crop[valid_mask].reshape(-1, 3)

        if len(pixels) == 0:
            pixels = crop.reshape(-1, 3)

        if len(pixels) == 0:
            return None, 0

        hsv_pixels = cv2.cvtColor(
            pixels.reshape(-1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2HSV
        ).reshape(-1, 3)

        sat = hsv_pixels[:, 1]

        # Confidence = how many high-sat pixels we have
        high_sat_pixels = pixels[sat > 60]
        confidence = len(high_sat_pixels)

        if len(high_sat_pixels) > 10:
            pixels = high_sat_pixels

        median_colour = np.median(pixels, axis=0)
        b, g, r = median_colour

        return (int(r), int(g), int(b)), confidence


    for player in players:
        crop = player.get("jersey_crop")

        if crop is None or crop.size == 0:
            player["jersey_colour"] = None
            continue

        h, w = crop.shape[:2]

        candidates = []

        # Center
        candidates.append(crop[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)])

        # Slightly left
        candidates.append(crop[int(h*0.2):int(h*0.6), int(w*0.1):int(w*0.6)])

        # Slightly right
        candidates.append(crop[int(h*0.2):int(h*0.6), int(w*0.4):int(w*0.9)])

        # Higher (avoid shorts)
        candidates.append(crop[int(h*0.1):int(h*0.5), int(w*0.2):int(w*0.8)])

        best_colour = None
        best_conf = -1

        for c in candidates:
            colour, conf = get_colour_from_crop(c)

            if colour is None:
                continue

            if conf > best_conf:
                best_conf = conf
                best_colour = colour

        player["jersey_colour"] = best_colour

    return players
def build_player_data(frame, player_result):
    filtered_boxes = filter_player_boxes(player_result)
    players = get_player_bottom_centre(filtered_boxes)
    players = get_jersey_crop(frame, players)
    players = extract_jersey_colour(players)

    # ✅ DEBUG (FIXED)
    for p in players:
        colour = p.get("jersey_colour")

        if colour is not None:
            bgr = np.uint8([[colour[::-1]]])
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]

            print(
                "ID:", p.get("track_id"),
                "RGB:", colour,
                "HSV:", hsv,
                "team:", p.get("team")
            )

    return players


def assign_teams_by_colour(players):
    colours = []
    player_indices = []

    for i, p in enumerate(players):
        colour = p.get("jersey_colour")
        if colour is None:
            continue

        colours.append(colour)
        player_indices.append(i)

    if len(colours) < 2:
        for p in players:
            p["team"] = None
        return players

    rgb_colours = np.array(colours, dtype=np.uint8)

    hsv_colours = cv2.cvtColor(
        rgb_colours.reshape(-1, 1, 3),
        cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)

    features = []

    for h, s, v in hsv_colours:
        hue_angle = (float(h) / 180.0) * 2.0 * np.pi
        hue_x = np.cos(hue_angle)
        hue_y = np.sin(hue_angle)

        sat = float(s) / 255.0
        val = float(v) / 255.0

        features.append([
            hue_x * sat,
            hue_y * sat,
            sat * 0.5,
            val * 0.3
        ])

    features = np.array(features, dtype=float)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = kmeans.fit_predict(features)

    for player_index, label in zip(player_indices, labels):
        players[player_index]["team"] = int(label)

    for p in players:
        if p.get("jersey_colour") is None:
            p["team"] = None

    return players


def get_offside_players(players, left_offside_line, right_offside_line):
    offside_players = []

    for player in players:
        player_position = player["bottom_centre"]

        if offside.check_between_lines(player_position, left_offside_line, right_offside_line):
            offside_players.append(player)

    return offside_players


def get_offside_player_boxes(offside_players):
    return [player["box"] for player in offside_players]


def get_offside_team_labels(offside_players):
    return [player.get("team") for player in offside_players]


def build_player_coord_dict(players):
    player_dict = {}

    for player in players:
        player_coord = player["bottom_centre"]
        player_box = player["box"]
        player_dict[player_coord] = player_box

    return player_dict


def count_teams_and_refs(players):
    counts = {
        "team_0": 0,
        "team_1": 0,
        "unknown_team": 0,
        "refs": 0
    }

    for player in players:
        class_name = str(player.get("class_name", "")).lower()

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