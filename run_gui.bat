@echo off
REM Queensland Reds Rugby Offside Detection System - GUI Launcher
REM This batch file launches the GUI application on Windows

echo Starting Queensland Reds Rugby Offside Detection System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Run the GUI application
python run_gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo An error occurred. Press any key to exit...
    pause
)

