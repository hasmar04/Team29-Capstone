"""Session-level detection statistics for reporting and GUI integration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DetectionSessionStats:
    """
    Accumulate detection statistics for one processed video.

    Parameters:
        total_detections (int): Number of detected ruck/lineout events.
        ruck_count (int): Number of ruck detections.
        lineout_count (int): Number of lineout detections.
        total_offside_players (int): Sum of offside players across events.
        average_confidence (float): Running mean event confidence.
        total_frames (int): Processed frame count.
        video_duration (float): Video duration in seconds.
        fps (float): Source video frame rate.
        events (list): DetectionEvent objects or event-like dictionaries.
        total_confidence_sum (float): Internal accumulator.
        logs (list): Structured processing logs.

    Returns:
        DetectionSessionStats: Mutable session statistics object.

    Error handling:
        ``add_event`` accepts dataclasses or dictionaries. Missing event fields
        are treated as safe defaults so summary generation does not crash the GUI.
    """

    total_detections: int = 0
    ruck_count: int = 0
    lineout_count: int = 0
    total_offside_players: int = 0
    average_confidence: float = 0.0
    total_frames: int = 0
    video_duration: float = 0.0
    fps: float = 0.0
    events: List[Any] = field(default_factory=list)
    total_confidence_sum: float = 0.0
    logs: List[Dict[str, Any]] = field(default_factory=list)

    def add_event(self, event) -> None:
        """
        Add one detection event and update aggregate statistics.

        Parameters:
            event (DetectionEvent or dict): Event with type, confidence, and
                offside count fields.

        Returns:
            None

        Error handling:
            Missing fields default to ``"unknown"``, ``0.0``, and ``0``.
        """
        self.events.append(event)
        event_type = self._get(event, "event_type", self._get(event, "type", "unknown"))
        confidence = float(self._get(event, "confidence", self._get(event, "detection_confidence", 0.0)))
        offside_count = int(self._get(event, "offside_count", 0))

        self.total_detections += 1
        self.total_offside_players += offside_count
        self.total_confidence_sum += confidence
        self.average_confidence = self.total_confidence_sum / self.total_detections

        if event_type == "ruck":
            self.ruck_count += 1
        elif event_type == "lineout":
            self.lineout_count += 1

    def add_log(self, log_entry: Dict[str, Any]) -> None:
        """
        Add a structured processing log entry.

        Parameters:
            log_entry (dict): JSON-safe log entry with stage/message fields.

        Returns:
            None

        Error handling:
            The entry is stored as provided; validation is left to API/frontend
            schema layers.
        """
        self.logs.append(log_entry)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert session statistics to a frontend-safe dictionary.

        Parameters:
            None

        Returns:
            dict: Summary, events, and logs for GUI or JSON output.

        Error handling:
            Event objects with ``to_dict`` are converted; other values are
            returned as-is.
        """
        return {
            "total_detections": self.total_detections,
            "ruck_count": self.ruck_count,
            "lineout_count": self.lineout_count,
            "total_offside_players": self.total_offside_players,
            "average_confidence": self.average_confidence,
            "total_frames": self.total_frames,
            "video_duration": self.video_duration,
            "fps": self.fps,
            "events": [
                event.to_dict() if hasattr(event, "to_dict") else event
                for event in self.events
            ],
            "logs": self.logs,
        }

    @staticmethod
    def _get(event, key, default=None):
        """Read a key from a dictionary or object with a default."""
        if isinstance(event, dict):
            return event.get(key, default)
        return getattr(event, key, default)
