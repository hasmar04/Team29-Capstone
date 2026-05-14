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
    Extract jersey colour using multi-band consensus voting.
    Also stores alternate band colours so team assignment can retry if a player is an outlier.
    """

    def dominant_colour_from_crop(crop):
        if crop is None or crop.size == 0:
            return None, 0

        crop = crop.astype(np.uint8)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        green_mask = (h > 35) & (h < 85) & (s > 60)
        low_sat_mask = s < 40
        dark_mask = v < 40

        valid_mask = ~(green_mask | low_sat_mask | dark_mask)
        pixels = crop[valid_mask]

        if len(pixels) < 20:
            return None, 0

        pixels_float = np.float32(pixels)
        k = min(3, len(pixels))

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20,
            1.0
        )

        _, labels, centers = cv2.kmeans(
            pixels_float,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        labels = labels.flatten()
        counts = np.bincount(labels)
        dominant_index = np.argmax(counts)

        b, g, r = centers[dominant_index]
        confidence = counts[dominant_index]

        return (int(r), int(g), int(b)), int(confidence)

    def colours_similar(c1, c2, threshold=45):
        if c1 is None or c2 is None:
            return False

        r1, g1, b1 = c1
        r2, g2, b2 = c2

        distance = np.sqrt(
            (r1 - r2) ** 2 +
            (g1 - g2) ** 2 +
            (b1 - b2) ** 2
        )

        return distance < threshold

    for player in players:
        crop = player.get("jersey_crop")

        player["jersey_colour_candidates"] = []

        if crop is None or crop.size == 0:
            player["jersey_colour"] = None
            continue

        h, w = crop.shape[:2]

        bands = []
        num_bands = 5
        band_height = max(1, h // num_bands)

        for i in range(num_bands):
            y1 = i * band_height
            y2 = h if i == num_bands - 1 else (i + 1) * band_height
            bands.append(crop[y1:y2, :])

        band_results = []

        for i, band in enumerate(bands):
            colour, confidence = dominant_colour_from_crop(band)

            if colour is None:
                continue

            if i == 0:
                weight = 1.5
            elif i == 1:
                weight = 1.4
            elif i == 2:
                weight = 1.2
            elif i == 3:
                weight = 0.8
            else:
                weight = 0.5

            band_results.append({
                "band": i,
                "colour": colour,
                "confidence": confidence * weight
            })

        player["jersey_colour_candidates"] = band_results

        if not band_results:
            player["jersey_colour"] = None
            continue

        groups = []

        for result in band_results:
            placed = False

            for group in groups:
                if colours_similar(result["colour"], group["representative"]):
                    group["members"].append(result)
                    group["total_confidence"] += result["confidence"]
                    placed = True
                    break

            if not placed:
                groups.append({
                    "representative": result["colour"],
                    "members": [result],
                    "total_confidence": result["confidence"]
                })

        best_group = max(groups, key=lambda g: g["total_confidence"])
        player["jersey_colour"] = best_group["representative"]

    return players

def build_player_data(frame, player_result):
    filtered_boxes = filter_player_boxes(player_result)
    players = get_player_bottom_centre(filtered_boxes)
    players = get_jersey_crop(frame, players)
    players = extract_jersey_colour(players)

    return players


def assign_teams_by_colour(players):
    """
    Assign players to exactly two teams, then sanity-check outliers against
    all other players. If a player's colour does not match its assigned team,
    try alternate band colours before finalising.
    """

    def colour_distance(c1, c2):
        if c1 is None or c2 is None:
            return float("inf")

        r1, g1, b1 = c1
        r2, g2, b2 = c2

        return float(np.sqrt(
            (r1 - r2) ** 2 +
            (g1 - g2) ** 2 +
            (b1 - b2) ** 2
        ))

    def average_team_colour(team_players):
        colours = [
            p.get("jersey_colour")
            for p in team_players
            if p.get("jersey_colour") is not None
        ]

        if not colours:
            return None

        return tuple(np.mean(np.array(colours), axis=0).astype(int))

    valid_indices = []
    colours = []

    for i, p in enumerate(players):
        class_name = str(p.get("class_name", "")).lower()

        if class_name in ["ref", "referee", "official"]:
            p["team"] = None
            continue

        colour = p.get("jersey_colour")

        if colour is not None:
            valid_indices.append(i)
            colours.append(colour)

    if len(colours) < 2:
        for p in players:
            class_name = str(p.get("class_name", "")).lower()
            if class_name not in ["ref", "referee", "official"]:
                p["team"] = 0
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

    for player_index, label in zip(valid_indices, labels):
        players[player_index]["team"] = int(label)

    # Force uncoloured non-ref players into a temporary team.
    for p in players:
        class_name = str(p.get("class_name", "")).lower()
        if class_name not in ["ref", "referee", "official"] and p.get("team") not in [0, 1]:
            p["team"] = 0

    OUTLIER_THRESHOLD = 65

    for player in players:
        class_name = str(player.get("class_name", "")).lower()

        if class_name in ["ref", "referee", "official"]:
            continue

        current_team = player.get("team")

        if current_team not in [0, 1]:
            continue

        other_team = 1 - current_team

        same_team_players = [
            p for p in players
            if p is not player and p.get("team") == current_team
        ]

        opposite_team_players = [
            p for p in players
            if p.get("team") == other_team
        ]

        same_team_colour = average_team_colour(same_team_players)
        opposite_team_colour = average_team_colour(opposite_team_players)

        current_colour = player.get("jersey_colour")

        same_distance = colour_distance(current_colour, same_team_colour)
        opposite_distance = colour_distance(current_colour, opposite_team_colour)

        # If it already fits its team, keep it.
        if same_distance <= OUTLIER_THRESHOLD:
            continue

        # If it fits the other team better, swap it.
        if opposite_distance < same_distance:
            player["team"] = other_team
            continue

        # Otherwise, try alternate band colours from the crop.
        best_candidate = current_colour
        best_team = current_team
        best_distance = same_distance

        for candidate in player.get("jersey_colour_candidates", []):
            candidate_colour = candidate.get("colour")

            candidate_same_distance = colour_distance(candidate_colour, same_team_colour)
            candidate_opposite_distance = colour_distance(candidate_colour, opposite_team_colour)

            if candidate_same_distance < best_distance:
                best_candidate = candidate_colour
                best_team = current_team
                best_distance = candidate_same_distance

            if candidate_opposite_distance < best_distance:
                best_candidate = candidate_colour
                best_team = other_team
                best_distance = candidate_opposite_distance

        player["jersey_colour"] = best_candidate
        player["team"] = best_team

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