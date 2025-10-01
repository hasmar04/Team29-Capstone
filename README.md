# Team29-Capstone

## Queensland Reds Rugby - Offside Detection System

An automated computer vision system for detecting and analysing offside events in rugby matches, specifically designed for the Queensland Reds at QUT.

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

# System Architecture - Batch Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     QUEENSLAND REDS OFFSIDE DETECTION SYSTEM                │
│                              (Batch Processing Mode)                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. USER INPUT                                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User runs: python main.py                                                 │
│   Selects: "batch" mode                                                     │
│   Provides: Directory with rugby match videos                               │
│                                                                             │
│   Input Directory:          Output Directory:                               │
│   ├── match1.mp4            └── batch_output/                               │
│   ├── match2.mp4                                                            │
│   └── match3.mp4                                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. MODEL LOADING (One-time initialisation)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │  ruck.pt     │  │ lineout.pt   │  │  ball.pt     │  │ yolo11n.pt   │    │
│   │              │  │              │  │              │  │              │    │
│   │ Ruck         │  │ Lineout +    │  │ Ball         │  │ Player       │    │
│   │ Detection    │  │ Hooker       │  │ Tracking     │  │ Detection    │    │
│   │              │  │ Detection    │  │              │  │              │    │
│   └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. VIDEO PROCESSING LOOP (For each video)                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │ A. AUTOMATIC THRESHOLD DETECTION                                   │    │
│   │    • Extract middle frame                                          │    │
│   │    • Calculate average brightness                                  │    │
│   │    • Adaptive threshold = brightness * 0.7 (clamped 150-250)       │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │ B. YOLO INFERENCE (All frames)                                     │    │
│   │    ┌─────────────────────────────────────────────┐                 │    │
│   │    │  Frame 1 -> Ruck/Lineout/Ball/Player YOLO   │                 │    │
│   │    │  Frame 2 -> Ruck/Lineout/Ball/Player YOLO   │                 │    │
│   │    │  Frame 3 -> Ruck/Lineout/Ball/Player YOLO   │                 │    │
│   │    │  ...                                        │                 │    │
│   │    │  Frame N -> Ruck/Lineout/Ball/Player YOLO   │                 │    │
│   │    └─────────────────────────────────────────────┘                 │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │ C. FRAME-BY-FRAME ANALYSIS                                         │    │
│   │                                                                    │    │
│   │    For each frame:                                                 │    │
│   │                                                                    │    │
│   │    ┌───────────────────────────────────────────────────────────┐   │    │
│   │    │ 1. DETECTION LOGIC                                        │   │    │
│   │    │    • Track consecutive ruck detections (≥ fps/10)         │   │    │
│   │    │    • Track consecutive lineout detections (≥ fps/10)      │   │    │
│   │    │    • Prioritise higher confidence if both detected        │   │    │
│   │    │    • Apply cooldown after detection (3s ruck, 10s LO)     │   │    │
│   │    └───────────────────────────────────────────────────────────┘   │    │
│   │                            │                                       │    │
│   │                            ▼                                       │    │
│   │    ┌───────────────────────────────────────────────────────────┐   │    │
│   │    │ 2. IF RUCK DETECTED                                       │   │    │
│   │    │    a. Get last feet positions                             │   │    │
│   │    │    b. Extract field lines (Canny + Hough)                 │   │    │
│   │    │    c. Calculate intersection point                        │   │    │
│   │    │    d. Draw offside lines (last feet -> intersection)      │   │    │
│   │    │    e. Detect players with YOLO                            │   │    │
│   │    │    f. Filter players in ruck region                       │   │    │
│   │    │    g. Identify offside players between lines              │   │    │
│   │    │    h. Annotate frame with red lines & boxes               │   │    │
│   │    │    i. Record detection data + confidence scores           │   │    │
│   │    └───────────────────────────────────────────────────────────┘   │    │
│   │                            │                                       │    │
│   │                            ▼                                       │    │
│   │    ┌───────────────────────────────────────────────────────────┐   │    │
│   │    │ 3. IF LINEOUT DETECTED                                    │   │    │
│   │    │    a. Get hooker/lineout center position                  │   │    │
│   │    │    b. Extract field lines                                 │   │    │
│   │    │    c. Estimate field intersection points (heuristic)      │   │    │
│   │    │    d. Calculate homography matrix                         │   │    │
│   │    │    e. Get 10m offside points using homography             │   │    │
│   │    │    f. Draw offside lines (10m points -> intersection)     │   │    │
│   │    │    g. Detect players with YOLO                            │   │    │
│   │    │    h. Filter players in lineout region                    │   │    │
│   │    │    i. Identify offside players between lines              │   │    │
│   │    │    j. Annotate frame with blue lines & boxes              │   │    │
│   │    │    k. Record detection data + confidence scores           │   │    │
│   │    └───────────────────────────────────────────────────────────┘   │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │ D. DATA COLLECTION                                                 │    │
│   │                                                                    │    │
│   │    Stored for each detection:                                      │    │
│   │    • Event type (RUCK/LINEOUT)                                     │    │
│   │    • Frame number & timestamp                                      │    │
│   │    • Detection confidence (YOLO score)                             │    │
│   │    • Offside player count                                          │    │
│   │    • Player positions & confidences                                │    │
│   │    • Annotated frame (for video output)                            │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. OUTPUT GENERATION (Per video)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │ A. ANNOTATED VIDEO                                                 │    │
│   │    • Compile all annotated frames                                  │    │
│   │    • Save as: {video_name}_annotated.mp4                           │    │
│   │    • Format: MP4V codec, original FPS                              │    │
│   │    • Contains:                                                     │    │
│   │      - Red lines (ruck offside)                                    │    │
│   │      - Blue lines (lineout offside)                                │    │
│   │      - Red boxes around offside players                            │    │
│   │      - "Offside" labels                                            │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                    │                                        │
│                                    ▼                                        │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │ B. ANALYSIS REPORT                                                 │    │
│   │    • Generate text report                                          │    │
│   │    • Save as: {video_name}_analysis_report.txt                     │    │
│   │    • Contains:                                                     │    │
│   │      ┌──────────────────────────────────────────────────────────┐  │    │
│   │      │ HEADER                                                   │  │    │
│   │      │ • Video name                                             │  │    │
│   │      │ • Processing date/time                                   │  │    │
│   │      │ • Total frames analysed                                  │  │    │
│   │      │ • Total events detected                                  │  │    │
│   │      └──────────────────────────────────────────────────────────┘  │    │
│   │      ┌──────────────────────────────────────────────────────────┐  │    │
│   │      │ DETECTION EVENTS (for each event)                        │  │    │
│   │      │ • Event number & type                                    │  │    │
│   │      │ • Frame number & timestamp                               │  │    │
│   │      │ • Detection confidence                                   │  │    │
│   │      │ • Ruck/Lineout/Hooker confidences                        │  │    │
│   │      │ • Offside player count                                   │  │    │
│   │      │ • Player positions & confidences                         │  │    │
│   │      └──────────────────────────────────────────────────────────┘  │    │
│   │      ┌──────────────────────────────────────────────────────────┐  │    │
│   │      │ SUMMARY STATISTICS                                       │  │    │
│   │      │ • Total ruck/lineout events                              │  │    │
│   │      │ • Total offside players                                  │  │    │
│   │      │ • Average detection confidences                          │  │    │
│   │      └──────────────────────────────────────────────────────────┘  │    │
│   │                                                                    │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. FINAL OUTPUT (Batch complete)                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   batch_output/                                                             │
│   ├── match1_annotated.mp4        ← Video with offside annotations          │
│   ├── match1_analysis_report.txt  ← Detailed text report                    │
│   ├── match2_annotated.mp4                                                  │
│   ├── match2_analysis_report.txt                                            │
│   ├── match3_annotated.mp4                                                  │
│   └── match3_analysis_report.txt                                            │
│                                                                             │
│   Console Output:                                                           │
│   ════════════════════════════════════════════════════════════════════════  │
│   BATCH PROCESSING COMPLETE                                                 │
│   Output directory: ./batch_output                                          │
│   ════════════════════════════════════════════════════════════════════════  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ KEY TECHNICAL COMPONENTS                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Module              Function                    Purpose                    │
│  ─────────────────── ─────────────────────────── ────────────────────────── │
│  batch_processor.py  process_video_batch()       Main orchestrator          │
│  batch_processor.py  auto_mode_batch()           Headless auto detection    │
│  batch_processor.py  generate_summary_report()   Create text reports        │
│  batch_processor.py  save_annotated_video()      Save annotated videos      │
│  main.py             main()                      Entry point with modes     │
│  ui_functions.py     select_directory()          Folder selection GUI       │
│  yolo_functions.py   perform_inference()         YOLO model inference       │
│  field_functions.py  get_field_lines()           Field line extraction      │
│  ruck_functions.py   get_last_feet()             Ruck feet detection        │
│  lineout_functions.py get_ball_release()         Ball release detection     │
│  offside_functions.py get_players_between_lines() Offside identification    │
│  point_functions.py  get_homography_matrix()     Perspective transform      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│ CONFIDENCE SCORE FLOW                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  YOLO Model Output -> Confidence Score (0.0 - 1.0)                           │
│         │                                                                   │
│         ├─> Ruck detection: 0.87  ────┐                                     │
│         ├─> Lineout detection: 0.92 ──┼──> Stored in event data             │
│         ├─> Player detection: 0.85  ──┤                                     │
│         └─> Ball detection: 0.76  ────┘                                     │
│                                         │                                   │
│                                         ▼                                   │
│                          Reported in analysis_report.txt                    │
│                          (as percentages: 87%, 92%, 85%, 76%)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Processing Performance

```
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

```
Input Video -> YOLO Inference -> Detection Logic -> Field Analysis -> 
Player Tracking -> Offside Calculation -> Frame Annotation -> 
Video Compilation -> Report Generation -> Output Files
```

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

## Output Examples

### Annotated Videos
- Red lines indicate ruck offside boundaries
- Blue lines indicate lineout offside zones
- Red boxes highlight players in offside positions
- "Offside" labels on detected players

### Analysis Reports
```
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

```
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

**Note**: This system is designed for rugby analysis and coaching purposes. Detections should be verified by qualified officials for official match decisions.