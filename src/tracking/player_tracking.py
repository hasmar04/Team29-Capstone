import numpy as np
import warnings

try:
    import supervision as sv
except ModuleNotFoundError:
    sv = None


class PlayerTracker:
    """Persist player identities and team labels with ByteTrack."""

    def __init__(self):
        """
        Initialise a ByteTrack tracker and persistent team cache.

        Parameters:
            None

        Returns:
            None

        Error handling:
        Falls back to deterministic per-frame IDs if ``supervision`` is not
            installed, allowing tests and documentation builds to run.
        """
        if sv is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*ByteTrack.*deprecated.*",
                    category=FutureWarning,
                )
                self.tracker = sv.ByteTrack()
        else:
            self.tracker = None
        self.track_teams = {}  # persistent team per track_id

    def compute_iou(self, boxA, boxB):
        """
        Compute intersection-over-union for two bounding boxes.

        Parameters:
            boxA (tuple/list): ``(x1, y1, x2, y2)`` box coordinates.
            boxB (tuple/list): ``(x1, y1, x2, y2)`` box coordinates.

        Returns:
            float: IoU score in the 0.0-1.0 range.

        Error handling:
            Degenerate non-overlapping boxes return ``0.0``. Invalid coordinate
            structures will raise normal indexing/type errors for callers to log.

        Example:
            ``tracker.compute_iou((0, 0, 10, 10), (5, 5, 15, 15))``
        """
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
        """
        Update tracks from current-frame player detections.

        Parameters:
            players (list): Player dictionaries containing at least ``box`` and
                optionally ``confidence`` and ``team``.

        Returns:
            list: Player dictionaries enriched with ``track_id`` and stable
            ``team`` values.

        Error handling:
        Empty input returns an empty list. If ByteTrack is unavailable, players
        receive deterministic IDs based on their order in the frame.
        """
        if not players:
            return []

        if self.tracker is None:
            tracked_players = []
            for index, player in enumerate(players):
                tracked = player.copy()
                tracked["track_id"] = index + 1
                tracked_players.append(tracked)
            return tracked_players

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
