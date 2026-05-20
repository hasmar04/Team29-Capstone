#!/usr/bin/env python3
"""
run_gui.py
----------
Launcher script for the Queensland Reds Rugby Offside Detection System GUI.
This script ensures the correct Python path and launches the GUI application.

Usage:
    python run_gui.py
"""

import sys
import os

# Add the src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)

# Import and run the GUI
try:
    from src.gui_app import run_gui
    run_gui()
except ImportError as e:
    print(f"Error importing GUI application: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error running GUI application: {e}")
    sys.exit(1)
