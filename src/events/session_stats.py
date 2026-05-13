from dataclasses import dataclass, field


@dataclass
class DetectionSessionStats:

    total_detections: int = 0

    ruck_count: int = 0
    lineout_count: int = 0

    total_offside_players: int = 0

    average_confidence: float = 0.0

    total_frames: int = 0

    video_duration: float = 0.0

    fps: float = 0.0

    events: list = field(default_factory=list)

    total_confidence_sum: float = 0.0


    def add_event(self, event):

        self.events.append(event)

        self.total_detections += 1

        self.total_offside_players += event.offside_count

        self.total_confidence_sum += event.confidence

        self.average_confidence = (
            self.total_confidence_sum / self.total_detections
        )

        if event.event_type == "ruck":
            self.ruck_count += 1

        elif event.event_type == "lineout":
            self.lineout_count += 1