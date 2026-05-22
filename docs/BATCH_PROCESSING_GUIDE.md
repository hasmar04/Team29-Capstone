# Batch Processing Guide

Batch mode processes one or more rugby clips without user interaction and writes frontend-ready outputs.

## Inputs

Supported video extensions:

- `.mp4`
- `.avi`
- `.mov`
- `.gif`

Input can be either a single video file or a directory. Directory input is processed sequentially in sorted file-name order.

## Running Batch Mode

GUI:

```bash
python run_gui.py
```

Select `Batch Mode`, choose the input directory, choose an output directory, then start processing.

Programmatic:

```python
from src.detection import yolo_functions as yolo
from src.processing import batch_processor

ruck_model = yolo.load_model("./models/ruck.pt")
lineout_model = yolo.load_model("./models/lineout.pt")
ball_model = yolo.load_model("./models/ball.pt")
player_model = yolo.load_model("./models/player-id.pt")

results = batch_processor.process_video_batch(
    "./clips",
    "./batch_output",
    ruck_model,
    lineout_model,
    ball_model,
    player_model,
)
```

## Outputs

For each input video:

```text
{video_name}_analysis_report.txt
{video_name}_analysis.json
{video_name}_annotated.mp4
```

Annotated MP4 files are only written when at least one event is detected.

## JSON Contract

`*_analysis.json` is the preferred frontend integration output.

Top-level fields:

- `video_name`
- `source_video`
- `processing_date`
- `total_frames`
- `fps`
- `events`
- `summary`
- `logs`
- `output_files`

Event fields:

- `type`
- `frame_number`
- `timestamp`
- `detection_confidence`
- `ruck_confidence` or `lineout_confidence`
- `processing_stage`
- `offside_count`
- `offside_players`
- `team_counts`
- `tracked_players`
- `errors`

## Stability Behaviour

Batch mode is designed to keep moving:

- Invalid frames are skipped and logged.
- Empty detections reset event counters.
- Overlapping ruck and lineout detections are resolved by higher confidence.
- Field-line or homography failures skip the current candidate event.
- Per-video exceptions are returned as failed payloads and do not stop later clips.

## Performance Notes

Long clips are model-inference heavy. Use CUDA-capable PyTorch where available. Batch mode still runs separate ruck, lineout, and ball inference streams; future teams can improve performance by caching shared frame data or combining model passes where technically practical.
