# Team29-Capstone

## Queensland Reds Rugby - Offside Detection System

An automated computer vision system for detecting and analysing offside events in rugby matches, specifically designed for the Queensland Reds at QUT.

## Quick Start

### Installation

1. Clone the repository:

```bash
git clone https://github.com/hasmar04/Team29-Capstone.git
cd Team29-Capstone
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure YOLO models are in the `models/` directory

### Running Batch Processing

```bash
cd src
python main.py
```

When prompted:

1. Select mode: `batch`
2. Choose your videos directory
3. Select output directory (or use default)
4. Wait for processing to complete

See [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md) for detailed instructions.

### Example Programmatic Usage

```python
import batch_processor as batch
import yolo_functions as YOLO

# Load models
ruck_model = YOLO.load_model('./models/ruck.pt')
lineout_model = YOLO.load_model('./models/lineout.pt')
ball_model = YOLO.load_model('./models/ball.pt')
player_model = YOLO.load_model('./models/yolo11n.pt')

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

### Processing Modes

#### 1. Batch Processing

Process multiple videos automatically with comprehensive reporting:

- Mass input of video footage from directories
- Fully automated detection without user interaction
- Confidence level tracking for all detections
- Annotated video output with highlighted offside players and lines
- Detailed text reports for each video with statistics

**Use case**: Analysing multiple match recordings overnight

#### 2. Auto Mode

Single video processing with automatic detection:

- Real-time ruck and lineout detection
- Automatic offside line calculation
- Interactive threshold selection

**Use case**: Quick analysis of a single match

#### 3. Manual Mode

Frame-by-frame control for detailed analysis:

- Manual trigger for ruck/lineout detection
- Step-by-step offside analysis
- Full user control over detection timing

**Use case**: Detailed review of specific match moments

## System Architecture - Batch Processing

See [docs/System Achitecture.md](<docs/System Architecture.md>)

## Processing Performance

```text
Video Processing Timeline (Example):
────────────────────────────────────────────────────────────────────────

Video: 10 minutes @ 30 FPS = 18,000 frames

Step                           Time          % of Total
──────────────────────────── ─────────────── ───────────
YOLO Inference (all frames)   15-30 minutes     60-70%
Frame-by-frame analysis        5-10 minutes      20-30%
Output generation              1-2 minutes        5-10%
──────────────────────────── ─────────────── ───────────
TOTAL PROCESSING TIME          21-42 minutes     100%

CPU vs GPU:
• CPU only: ~40-50 minutes (5x real-time)
• With GPU: ~8-15 minutes (2x real-time)
```

## Data Flow Summary

```text
Input Video -> YOLO Inference -> Detection Logic -> Field Analysis -> 
Player Tracking -> Offside Calculation -> Frame Annotation -> 
Video Compilation -> Report Generation -> Output Files
```

## Output Examples

### Annotated Videos

- Red lines indicate ruck offside boundaries
- Blue lines indicate lineout offside zones
- Red boxes highlight players in offside positions
- "Offside" labels on detected players

### Analysis Reports

```text
==================================================================================
RUGBY OFFSIDE DETECTION ANALYSIS REPORT
==================================================================================

Video File: match1.mp4
Total Events Detected: 8
Total Offside Players: 12

Event #1
  Type: RUCK
  Timestamp: 7.80s
  Detection Confidence: 87.50%
  Offside Players Detected: 2
```

## Project Structure

```text
Team29-Capstone/
├── src/
│   ├── main.py                    # Entry point with mode selection
│   ├── batch_processor.py         # NEW: Batch processing module
│   ├── ruck_functions.py          # Ruck detection logic
│   ├── lineout_functions.py       # Lineout detection logic
│   ├── offside_functions.py       # Offside player identification
│   ├── field_functions.py         # Field line detection
│   ├── yolo_functions.py          # YOLO model inference
│   └── ...
├── models/
│   ├── ruck.pt                    # Ruck detection model
│   ├── lineout.pt                 # Lineout detection model
│   ├── ball.pt                    # Ball tracking model
│   └── yolo11n.pt                 # Player detection model
├── inference/                      # Test videos and images
├── Handover/                       # Project documentation
├── BATCH_PROCESSING_GUIDE.md      # NEW: Detailed batch processing guide
└── example_batch_usage.py         # NEW: Example script

```

## Models

The system uses custom-trained YOLO models:

- **ruck.pt**: Detects ruck formations and ball position
- **lineout.pt**: Detects lineout formations, hooker position
- **ball.pt**: Specialised ball tracking
- **yolo11n.pt**: General player detection (YOLO11 nano)

Training notebooks available in `colab_notebooks/`

## Documentation

- **[BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md)**: Complete batch processing documentation
- **Manual Testing Plan.pdf**: Testing procedures
- **Handover/Docs/Troubleshooting Guide.pdf**: Common issues and solutions
- **Handover/Docs/Training and Exporting Datasets Guide.pdf**: Model training guide
- **System Architecture.txt**: System design overview

## Key Technical Details

### Detection Pipeline

1. YOLO inference on video frames
2. Consecutive frame detection for stability
3. Ball release detection
4. Field line extraction and perspective correction
5. Offside line calculation using homography
6. Player position analysis
7. Offside player identification

### Automatic Features

- **Adaptive thresholding**: Automatically adjusts based on video brightness
- **Consecutive frame validation**: Prevents false positives
- **Cooldown periods**: Avoids duplicate detections (3s for rucks, 10s for lineouts)
- **Field point estimation**: Heuristic-based field intersection detection

### Confidence Tracking

Every detection includes:

- Detection confidence (YOLO model score)
- Player detection confidence
- Offside line calculation success rate
- Overall event confidence

## Performance

- **Processing speed**: 2-5x real-time (depending on hardware)
- **Detection accuracy**: ~85-90% for clear footage
- **GPU acceleration**: Supported via PyTorch CUDA
- **Recommended specs**:
  - CPU: Multi-core processor (4+ cores)
  - RAM: 8GB minimum
  - GPU: NVIDIA GPU with CUDA support (optional but recommended)

## Development

### Running Tests

```bash
python -m pytest src/unit_testing/
```

### Training New Models

See notebooks in `colab_notebooks/`:

- `ball_dataset.ipynb`
- `lineout_dataset.ipynb`
- `ruck_dataset.ipynb`
- `training_template.ipynb`

## Contributors

**Team 29** - QUT Capstone Project 2025

- Project Partner: Queensland Reds
- University: Queensland University of Technology

## License

This project is developed as part of a university capstone project for the Queensland Reds.

## Support

For issues or questions:

1. Check [BATCH_PROCESSING_GUIDE.md](BATCH_PROCESSING_GUIDE.md)
2. Review `Handover/Docs/Troubleshooting Guide.pdf`
3. Contact the development team

---
