```text
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