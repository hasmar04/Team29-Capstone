import supervision as sv
import numpy as np

class PlayerTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.track_teams = {}  # persistent team per track_id

    def update(self, players):
        print("[TRACKER] update called with", len(players), "players")

        if not players:
            return []

        print("[TRACKER] building detections...")

        # Build detections for ByteTrack
        boxes = []
        confidences = []

        for p in players:
            x1, y1, x2, y2 = p["box"]
            boxes.append([x1, y1, x2, y2])
            confidences.append(1.0)  # dummy confidence (already filtered)

        boxes = np.array(boxes)
        confidences = np.array(confidences)

        detections = sv.Detections(
            xyxy=boxes,
            confidence=confidences,
            class_id=np.zeros(len(boxes))
        )

        print("[TRACKER] calling ByteTrack...")

        tracks = self.tracker.update_with_detections(detections)

        print("[TRACKER] ByteTrack returned")

        tracker_ids = tracks.tracker_id
        print("[TRACKER] tracker_ids:", tracker_ids)

        tracked_players = []

        for i, track_id in enumerate(tracker_ids):
            print(f"[TRACKER] processing tracked player {i}")

            # Find matching detection (simple index match)
            if i >= len(players):
                continue

            p = players[i]
            p["track_id"] = int(track_id)

            print("[TRACKER] comparisons done:", len(players))

            track_id = p["track_id"]
            new_team = p.get("team")

            print(f"[TRACKER] track_id {track_id} colour: {p.get('jersey_colour')}")

            if track_id in self.track_teams:
                # Reuse existing team
                p["team"] = self.track_teams[track_id]
                print(f"[TRACKER] reused team {p['team']} for track_id {track_id}")

            else:
                # Assign only once
                if new_team is not None:
                    self.track_teams[track_id] = new_team
                    p["team"] = new_team
                    print(f"[TRACKER] assigned NEW team {new_team} to track_id {track_id}")
                else:
                    p["team"] = None
                    print(f"[TRACKER] no team assigned for track_id {track_id}")

            print("[TRACKER] history size:", len(self.track_teams))

            tracked_players.append(p)

        print(f"[TRACKER] returning {len(tracked_players)} tracked players")

        return tracked_players