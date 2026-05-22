# Queensland Reds Rugby Offside Detection System - GUI

## Features

- **Modern Interface**: Clean, intuitive GUI built with customtkinter
- **Multiple Processing Modes**: Manual, Automatic, and Batch processing
- **Real-time Progress Tracking**: Live updates and progress bars
- **File Management**: Easy video file and directory selection
- **Logging**: Comprehensive processing logs with timestamps
- **Model Status**: Real-time model loading status

## Quick Start - End User

1. Dowload the latest release zip from the GitHub repository for your operating system: [Team29-Capstone Releases](https://github.com/hasmar04/Team29-Capstone/releases)
2. Extract the zip to a location on your local machine that is easy to access. 
   - Note! Do not separate the files within the zip archive. Moving individual files will break the program. 
3. Click on the 'app' file to start the program

## Quick Start - Development

Python needs to be installed on your system. It can be downloaded from the [Python Website](https://www.python.org/downloads/)

### Windows Users
1. Double-click `run_gui.bat` to launch the GUI
2. Or run `python run_gui.py` from command prompt

### Mac and Linux Users
1. Open the Terminal in the current folder where this repository is downloaded
2. Run `python run_gui.py`

## GUI Components

### 1. Processing Mode Selection
- **Manual Mode**: Manually select when to detect rucks/lineouts
- **Auto Mode**: Automatic detection of rucks/lineouts  
- **Batch Mode**: Process multiple videos automatically

### 2. File Selection
- **Video File**: Select single video file (Manual/Auto modes)
- **Batch Directory**: Select directory containing videos (Batch mode)
- **Output Directory**: Choose where to save results

### 3. Processing Controls
- **Start Processing**: Begin video analysis
- **Stop**: Cancel current processing
- **Progress Bar**: Visual progress indicator

### 4. Real-time Feedback
- **Status Updates**: Current processing status
- **Processing Log**: Detailed log with timestamps
- **Model Status**: YOLO model loading status

## Usage Instructions

### Manual Mode
1. Select "Manual Mode"
2. Choose a video file
3. Click "Start Processing"
4. Use keyboard controls in the video window:
   - `L` - Detect lineout
   - `R` - Detect ruck
   - `P` - Pause/Play
   - `Q` - Quit

### Auto Mode
1. Select "Auto Mode"
2. Choose a video file
3. Click "Start Processing"
4. The system will automatically detect rucks and lineouts
5. Use `P` to pause or `Q` to quit during processing

### Batch Mode
1. Select "Batch Mode"
2. Choose a directory containing video files
3. Optionally select an output directory
4. Click "Start Processing"
5. All videos will be processed automatically

## Output Files

### Single Video Modes
- `{video_name}_offside_detection.mp4` - Annotated video
- `{video_name}_detection_log.txt` - Detailed analysis log

### Batch Mode
- Processed videos saved to `batch_output/` directory
- Individual analysis logs for each video

## Troubleshooting

See [Troubleshooting Guide](<docs/Troubleshooting Guide.md>)

### System Requirements

- Python 3.8+
- All dependencies from `requirements.txt`
- YOLO model files in `models/` directory
- Sufficient disk space for output videos

## Advanced Configuration

### Model Files Required
- **ball.pt**: Specialised ball tracking
- **lineout.pt**: Detects lineout formations, hooker position
- **player-id**: Detects players and refs
- **ruck.pt**: Detects ruck formations and ball position

### Output Settings
- Videos are processed at 800x450 resolution
- Overlay duration: 5 seconds per detection
- Automatic threshold: 120 for field line detection
