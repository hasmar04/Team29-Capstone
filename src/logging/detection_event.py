"""Structured detection event model used by reports and frontend payloads."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class DetectionEvent:
    """
    Store one detected ruck or lineout event.

    Parameters:
        event_type (str): Event label, usually ``"ruck"`` or ``"lineout"``.
        confidence (float): Primary detector confidence in the 0.0-1.0 range.
        frame_number (int): Frame where the event was detected.
        timestamp (float): Timestamp in seconds.
        offside_count (int): Number of offside players identified.
        detection_id (int, optional): Stable identifier when assigned by callers.
        processing_stage (str, optional): Pipeline stage that produced the event.
        offside_players (list, optional): Player positions and confidences.
        team_counts (dict, optional): Team/referee counts at the event frame.
        tracked_players (list, optional): ByteTrack player summaries.
        metadata (dict, optional): Additional model or geometry metadata.
        errors (list, optional): Recoverable errors associated with the event.
        created_at (str, optional): ISO timestamp for logging.

    Returns:
        DetectionEvent: Dataclass instance. Use ``to_dict`` for JSON output.

    Error handling:
        The dataclass stores error details as strings/dicts supplied by callers;
        it does not raise during construction unless required fields are missing.
    """

    event_type: str
    confidence: float
    frame_number: int
    timestamp: float
    offside_count: int
    detection_id: int = 0
    processing_stage: str = "offside_detection"
    offside_players: List[Dict[str, Any]] = field(default_factory=list)
    team_counts: Dict[str, int] = field(default_factory=dict)
    tracked_players: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the event to a JSON-serialisable dictionary.

        Parameters:
            None

        Returns:
            dict: Event payload for reports, GUI display, or API responses.

        Error handling:
            Relies on stored values being JSON-compatible.
        """
        data = asdict(self)
        data["type"] = data.pop("event_type")
        data["detection_confidence"] = data["confidence"]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionEvent":
        """
        Build an event from a dictionary.

        Parameters:
            data (dict): Event payload using either ``event_type`` or ``type``.

        Returns:
            DetectionEvent: Normalised event instance.

        Error handling:
            Missing optional fields receive defaults. Missing required event,
            confidence, frame, timestamp, or offside count values raise normal
            Python ``TypeError``/``KeyError`` exceptions for the caller to log.
        """
        payload = dict(data)
        if "event_type" not in payload and "type" in payload:
            payload["event_type"] = payload.pop("type")
        if "confidence" not in payload and "detection_confidence" in payload:
            payload["confidence"] = payload.pop("detection_confidence")

        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in payload.items() if key in allowed})

    def add_error(self, stage: str, message: str, error: Optional[Exception] = None) -> None:
        """
        Attach recoverable error information to the event.

        Parameters:
            stage (str): Stage where the issue occurred.
            message (str): Human-readable description.
            error (Exception, optional): Original exception.

        Returns:
            None

        Error handling:
            Error objects are converted to strings for JSON output.
        """
        self.errors.append({
            "stage": stage,
            "message": message,
            "error": str(error) if error else None,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
