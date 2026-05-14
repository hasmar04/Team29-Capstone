import supervision as sv
import numpy as np


class PlayerTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.track_teams = {}  # persistent team per track_id

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interW = max(0, xB - xA)
        interH = max(0, yB - yA)

        interArea = interW * interH

        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, players):
        if not players:
            return []

        boxes = []
        confidences = []

        for p in players:
            x1, y1, x2, y2 = p["box"]

            boxes.append([x1, y1, x2, y2])

            # Use real confidence if available
            confidences.append(float(p.get("confidence", 1.0)))

        boxes = np.array(boxes, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=np.zeros(len(boxes), dtype=int)
        )

        tracks = self.tracker.update_with_detections(detections)
        tracker_ids = tracks.tracker_id

        tracked_players = []

        # Match returned tracked boxes back to original players
        for track_box, track_id in zip(tracks.xyxy, tracker_ids):

            best_player = None
            best_iou = 0.0

            for p in players:
                iou = self.compute_iou(track_box, p["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_player = p

            if best_player is None:
                continue

            p = best_player.copy()
            p["track_id"] = int(track_id)

            new_team = p.get("team")

            if track_id in self.track_teams:

                # Reuse persistent team assignment
                p["team"] = self.track_teams[track_id]

            else:

                # Store team assignment once
                if new_team is not None:
                    self.track_teams[track_id] = new_team
                    p["team"] = new_team

                else:
                    p["team"] = None

            tracked_players.append(p)

        return tracked_players