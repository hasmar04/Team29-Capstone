# Backend Pipeline

## Entry Points

- `run_gui.py`: desktop GUI entry point.
- `python -m src.processing.main`: CLI entry point.
- `src.processing.batch_processor.process_video_batch`: programmatic batch entry point.

## Pipeline Stages

1. Model loading
   - Loads `ruck.pt`, `lineout.pt`, `ball.pt`, and `player-id.pt`.
   - Uses `src.detection.yolo_functions.load_model`.

2. Video preparation
   - Reads FPS and frame count.
   - Chooses a default threshold for field-line extraction.
   - Creates a rolling pre-event frame buffer.

3. YOLO inference
   - Runs ruck, lineout, and ball inference streams.
   - Extracts boxes/classes/confidences through utility functions.

4. Event validation
   - Requires consecutive detections.
   - Applies cooldowns after ruck and lineout detections.
   - Resolves overlapping ruck/lineout candidates by confidence.

5. Field geometry
   - Detects field lines.
   - Finds average intersection points.
   - Fits offside lines to the field outline.
   - Logs and skips candidate events when geometry is not valid.

6. Player detection and classification
   - Detects players on the event frame.
   - Filters low-confidence boxes.
   - Calculates bottom-centre points.
   - Extracts jersey crops.
   - Assigns teams using colour clustering.

7. Tracking
   - Uses `supervision.ByteTrack`.
   - Adds `track_id` values.
   - Reuses previous team labels by track ID to reduce flicker.

8. Offside filtering
   - Removes players inside the active ruck or lineout region.
   - Removes likely off-field lineout detections.
   - Checks player bottom-centres between the offside lines.

9. Output generation
   - Holds annotated event frames for review.
   - Writes annotated MP4 when detections exist.
   - Writes text reports for human review.
   - Writes JSON payloads for GUI/frontend integration.

## Structured Return Format

`process_video_batch` returns a list of per-video payloads:

```python
{
    "video_name": "clip_001",
    "source_video": ".../clip_001.mp4",
    "processing_date": "2026-05-21T14:00:00",
    "total_frames": 1500,
    "fps": 25,
    "events": [],
    "summary": {},
    "logs": [],
    "output_files": {},
}
```

Failed clips return the same general shape with `status: "failed"` and an error log entry.

## Edge Cases

- Invalid frame: skipped and logged.
- Empty detections: counters reset.
- Overlapping players: ByteTrack and IoU matching preserve likely identity.
- Occlusion: team label is reused by track ID when available.
- Poor lighting: jersey colour extraction ignores grass, low-saturation, and dark pixels where possible.
- Inconsistent camera angles: geometry failures are logged and skipped rather than crashing the batch.

## Handover Guidance

Keep the JSON schema stable. If new fields are added, add them as optional fields first and update GUI parsing after the backend payload is confirmed.
