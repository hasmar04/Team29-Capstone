import supervision as sv
import numpy as np


class PlayerTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, players):
        """
        players = list of dicts from player_detection.py

        Each player must contain:
        player["box"] = (x1,y1,x2,y2)
        """

        if not players:
            return []

        xyxy = []
        confidence = []
        class_id = []

        for p in players:
            x1, y1, x2, y2 = p["box"]
            xyxy.append([x1, y1, x2, y2])
            confidence.append(0.99)
            class_id.append(0)

        detections = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_id)
        )

        tracked = self.tracker.update_with_detections(detections)

        tracked_players = []

        for i, track_id in enumerate(tracked.tracker_id):
            player = players[i].copy()
            player["track_id"] = int(track_id)
            tracked_players.append(player)

        return tracked_players