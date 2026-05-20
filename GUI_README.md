# Queensland Reds Rugby Offside Detection System - GUI

## Features

- **Modern Interface**: Clean, intuitive GUI built with customtkinter
- **Multiple Processing Modes**: Manual, Automatic, and Batch processing
- **Real-time Progress Tracking**: Live updates and progress bars
- **File Management**: Easy video file and directory selection
- **Logging**: Comprehensive processing logs with timestamps
- **Model Status**: Real-time model loading status

## Quick Start

Python needs to be installed on your system. It can be downloaded from the [Python Website](https://www.python.org/downloads/)

### Windows Users
1. Double-click `run_gui.bat` to launch the GUI
2. Or run `python run_gui.py` from command prompt

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

### Common Issues

1. **"Models are still loading"**
   - Wait for YOLO models to load completely
   - Check that model files exist in `models/` directory

2. **"No video file selected"**
   - Ensure you've selected a valid video file
   - Check file format (supports .mp4, .avi, .mov, .gif)

3. **"Processing failed"**
   - Check the processing log for detailed error messages
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

### System Requirements

- Python 3.8+
- All dependencies from `requirements.txt`
- YOLO model files in `models/` directory
- Sufficient disk space for output videos

## Advanced Configuration

### Model Files Required
- `models/ruck.pt` - Ruck detection model
- `models/lineout.pt` - Lineout detection model  
- `models/ball.pt` - Ball detection model
- `models/yolo11n.pt` - Player detection model

### Output Settings
- Videos are processed at 800x450 resolution
- Overlay duration: 5 seconds per detection
- Automatic threshold: 120 for field line detection

## Support

For technical support or questions:
1. Check the processing log for error details
2. Ensure all dependencies are properly installed
3. Verify model files are present and valid
4. Check that video files are in supported formats

## Keyboard Shortcuts (During Video Processing)

- `P` - Pause/Resume video playback
- `Q` - Quit processing
- `L` - Detect lineout (Manual mode only)
- `R` - Detect ruck (Manual mode only)
