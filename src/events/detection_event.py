from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DetectionEvent:

    # Core event info
    event_type: str
    confidence: float

    # Video info
    frame_number: int
    timestamp: float

    # Offside info
    offside_count: int

    # Detection metadata
    detection_id: int = 0

    # Optional detailed player data
    offside_players: List[Dict[str, Any]] = field(default_factory=list)

    # Optional team statistics
    team_counts: Dict[str, int] = field(default_factory=dict)

    # Optional tracking data
    tracked_players: List[Dict[str, Any]] = field(default_factory=list)

    # Optional extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)