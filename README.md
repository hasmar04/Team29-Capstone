# P541 Offside Monitoring in Rugby

QUT IFB398 IT Capstone project for Queensland Rugby Union and the Queensland Reds.

This repository contains a Python computer-vision pipeline and desktop GUI for reviewing rugby match footage, detecting rucks and lineouts, tracking players, classifying teams by jersey colour, and producing offside event outputs that can be consumed by the frontend.

## Project Overview

The system is designed as an assistive review tool, not as an autonomous referee. It combines custom YOLO models, field-line geometry, ByteTrack player tracking, jersey-colour team classification, and report generation to highlight possible offside incidents in rugby footage.

## End-user Installation

1. Dowload the latest release zip from the GitHub repository for your operating system: [Team29-Capstone Releases](https://github.com/hasmar04/Team29-Capstone/releases)
2. Extract the zip to a location on your local machine that is easy to access. 
   - Note! Do not separate the files within the zip archive. Moving individual files will break the program. 
3. Click on the 'app' file to start the program

### Development Installation

Primary use cases:

- Process a single clip in manual or automatic mode.
- Process a directory of clips in batch mode.
- Generate annotated clips and reports for coaching, review, and future model development.
- Provide structured backend outputs for GUI display and future frontend integration.

## System Architecture

The current implementation is organised into logical backend layers under `src/`:

```text
src/
  detection/        YOLO model loading and ruck, lineout, ball detection helpers
  tracking/         ByteTrack-based player identity persistence
  classification/   player detection, jersey colour extraction, team assignment, offside checks
  processing/       CLI, GUI orchestration, batch processing
  logging/          structured event and session statistics dataclasses
  events/           event dataclass mirror for integration experiments
  utils/            field, line, point, drawing, UI, and general helpers
```

Legacy top-level modules such as `src/player_detection.py` and `src/batch_processor.py` are retained for compatibility with older tests and scripts. New development should prefer the modular folders above.

See [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md) for the full architecture notes.

## Installation

Use Python 3.8 or newer. A virtual environment is strongly recommended.

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `requirements.txt` file installs the core dependencies:

- `ultralytics`, `torch`, `torchvision` for YOLO inference.
- `opencv-python`, `opencv-contrib-python`, `numpy` for video and geometry processing.
- `supervision` for ByteTrack.
- `scikit-learn` for jersey-colour team clustering.
- `customtkinter`, `Pillow` for the desktop GUI.
- `pytest`, `pytest-cov` for test execution.

CUDA is optional but recommended for long video batches. Install the PyTorch build that matches the target GPU environment if GPU acceleration is required.

## Environment Setup

Required model files must be present in `models/`:

```text
models/
  ruck.pt
  lineout.pt
  ball.pt
  player-id.pt
```python
import batch_processor as batch
import yolo_functions as YOLO

# Load models
ruck_model = YOLO.load_model('./models/ruck.pt')
lineout_model = YOLO.load_model('./models/lineout.pt')
ball_model = YOLO.load_model('./models/ball.pt')
player_model = YOLO.load_model('./models/player-id.pt')

# Process videos
batch.process_video_batch(
    input_path='./my_videos',
    output_dir='./results',
    ruck_model=ruck_model,
    lineout_model=lineout_model,
    ball_model=ball_model,
    player_model=player_model
)
```

Run commands from the repository root so `src` imports resolve consistently:

```bash
cd Team29-Capstone
```

VS Code users should select the virtual environment interpreter and run tests with `python -m pytest`, which honours `pytest.ini`.

## Running the System

Launch the GUI:

```bash
python run_gui.py
```

or:

```bash
.\run_gui.bat
```

Run the CLI pipeline:

```bash
python -m src.processing.main
```

The CLI offers:

- `manual`: user-triggered ruck or lineout analysis on a single clip.
- `auto`: automatic event detection on a single clip.
- `batch`: sequential processing of every supported video in a directory.

## Backend Pipeline

The backend pipeline follows this flow:

1. Load YOLO models from `models/`.
2. Read video frames with OpenCV.
3. Run ruck, lineout, and ball inference.
4. Validate events using consecutive-frame thresholds and cooldowns.
5. Estimate field lines and field-line intersections.
6. Calculate ruck or lineout offside lines.
7. Run player detection on the event frame.
8. Extract jersey crops and assign teams by colour.
9. Stabilise player identities with ByteTrack.
10. Filter players inside active ruck or lineout regions.
11. Identify players between offside lines.
12. Generate annotated clips, text reports, JSON summaries, and GUI payloads.

See [BACKEND_PIPELINE.md](BACKEND_PIPELINE.md) for module-level detail.

## Frontend Integration

Batch processing now produces structured dictionaries and JSON files alongside the human-readable text reports. GUI and future frontend work should consume these payloads instead of scraping report text where possible.

Standard batch output per clip:

```text
{video_name}_analysis_report.txt
{video_name}_analysis.json
{video_name}_annotated.mp4        only when detections exist
```

`*_analysis.json` includes:

- `video_name`
- `processing_date`
- `total_frames`
- `fps`
- `events`
- `summary`
- `logs`
- `output_files`

Each event includes event type, frame number, timestamp, confidence values, offside count, offside player positions, tracked player IDs when available, team counts, processing stage, and error metadata where relevant.

## Batch Processing

Batch mode processes clips sequentially to avoid GUI freezes and uncontrolled GPU memory growth. Each clip is isolated: errors are captured in the batch result, logged, and the next clip continues.

Programmatic usage:

```python
from src.detection import yolo_functions as yolo
from src.processing import batch_processor as batch

models = {
    "ruck": yolo.load_model("./models/ruck.pt"),
    "lineout": yolo.load_model("./models/lineout.pt"),
    "ball": yolo.load_model("./models/ball.pt"),
    "player": yolo.load_model("./models/player-id.pt"),
}

results = batch.process_video_batch(
    input_path="./clips",
    output_dir="./batch_output",
    ruck_model=models["ruck"],
    lineout_model=models["lineout"],
    ball_model=models["ball"],
    player_model=models["player"],
)
```

See [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md).

## YOLO Models

The project uses custom-trained YOLO model files:

- `ruck.pt`: detects ruck formations.
- `lineout.pt`: detects lineout formations and hooker classes where available.
- `ball.pt`: detects the ball for release/context checks.
- `player-id.pt`: detects players, people, and referee/official classes depending on training labels.

Inference is centralised in `src/detection/yolo_functions.py`. Model training guidance and handover notes are in [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md).

## Player Detection, Tracking, and Team Classification

Player processing is implemented in `src/classification/player_detection.py`:

- YOLO boxes are filtered by confidence.
- Player bottom-centres are used as the field contact point for offside geometry.
- Jersey crops are extracted from the upper torso.
- Dominant colours are estimated from multiple crop bands.
- KMeans groups players into two teams.
- Referee/official labels are excluded from team assignment where the model provides them.

Tracking is implemented in `src/tracking/player_tracking.py` using `supervision.ByteTrack`. Track IDs and team labels persist across frames, which helps reduce flicker during occlusion, overlapping players, and noisy jersey crops.

## Logging and Confidence Outputs

Structured event/session objects live in `src/logging/` and are mirrored in `src/events/`.

Logs and reports include:

- timestamps
- processing stage
- event type
- frame number
- confidence values
- player IDs where available
- team counts
- offside counts
- error information for skipped or failed stages

Batch mode also writes JSON logs for frontend consumption.

## Testing

Run all tests:

```bash
python -m pytest
```

Run with coverage:

```bash
python -m pytest --cov=src
```

Run a focused test file:

```bash
python -m pytest src/unit_testing/player_detection_unit_testing.py
```

The test suite is configured by `pytest.ini` to discover files ending in `_unit_testing.py`.

## Known Limitations
See [Team29-Capstone-Training](https://github.com/hasmar04/Team29-Capstone-Training)

- Offside decisions are candidate detections and still require human review.
- Accuracy depends heavily on camera angle, frame resolution, lighting, occlusion, and model training coverage.
- Field-line detection can fail when markings are hidden, washed out, or filmed from unusual perspectives.
- Team classification can be unstable when jersey colours are similar, lighting is poor, or players are heavily occluded.
- Batch mode still runs separate ruck, lineout, and ball inference streams; future optimisation could combine or cache more model passes.
- Manual mode still relies on OpenCV windows and is less suitable for headless environments.

## Future Improvements

- Add a formal API layer for frontend consumers.
- Store event outputs in SQLite or a lightweight local database.
- Add schema validation for JSON payloads.
- Expand model training data for more camera angles, weather, stadium lighting, and jersey combinations.
- Improve homography estimation with automatic field landmark detection.
- Add confidence calibration across model, field-line, and offside geometry stages.
- Add CI with a lightweight dependency profile and mocked YOLO models.

## Handover Notes

Future teams should start with the modular backend folders and avoid extending the legacy top-level duplicates unless maintaining old tests or scripts. The most important integration contract is the structured batch result in `src/processing/batch_processor.py`; keep it stable and version it if frontend expectations change.

Recommended next handover tasks:

- Confirm model files match the documented class mappings in `src/utils/constants.py`.
- Run tests in the target VS Code environment.
- Process a short known clip and inspect both the annotated video and JSON payload.
- Review `2025-2026-Handover/` for project partner and design context.
- Keep new documentation in root-level Markdown files so future teams can find it quickly.

## Contributors

Team 29, QUT IFB398 IT Capstone, P541 Offside Monitoring in Rugby for Queensland Rugby Union.
