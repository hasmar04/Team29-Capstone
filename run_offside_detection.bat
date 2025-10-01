@echo off
REM Quick Start Script for Queensland Reds Offside Detection System
REM Windows Batch Script

echo ================================================================================
echo Queensland Reds - Rugby Offside Detection System
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version
echo.

REM Check if models exist
if not exist "models\ruck.pt" (
    echo ERROR: Model files not found in 'models' directory
    echo Please ensure all YOLO model files are present:
    echo   - ruck.pt
    echo   - lineout.pt
    echo   - ball.pt
    echo   - yolo11n.pt
    pause
    exit /b 1
)

echo [2/3] Model files found
echo.

REM Navigate to src directory
cd src

echo [3/3] Starting application...
echo.
echo Available modes:
echo   - manual: Manual ruck/lineout detection
echo   - auto:   Automatic detection (single video)
echo   - batch:  Process multiple videos automatically
echo.
echo ================================================================================
echo.

REM Run the main application
python main.py

REM Return to original directory
cd ..

echo.
echo ================================================================================
echo Application closed
echo ================================================================================
pause
