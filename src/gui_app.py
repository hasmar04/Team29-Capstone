"""Compatibility wrapper for the modular GUI application.

New code should import from ``src.processing.gui_app``. This module keeps
existing entry points, tests, and ``run_gui.py`` imports stable.
"""

from src.processing.gui_app import *  # noqa: F401,F403
