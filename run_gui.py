#!/usr/bin/env python3
"""
run_gui.py
----------
Launcher script for the Queensland Reds Rugby Offside Detection System GUI.
This script ensures the correct Python path and launches the GUI application.

Usage:
    python run_gui.py
"""

import os
import sys

# Add the project root to Python path so imports work from any PowerShell cwd.
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

def main():
    """Import and run the GUI application."""
    try:
        from src.gui_app import run_gui
        run_gui()
    except ImportError as e:
        print(f"Error importing GUI application: {e}")
        print("Make sure all dependencies are installed:")
        print(f"{sys.executable} -m pip install -r {os.path.join(script_dir, 'requirements.txt')}")
        sys.exit(1)
    except Exception as e:
        print(f"Error running GUI application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
