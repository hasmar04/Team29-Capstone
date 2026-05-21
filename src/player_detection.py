"""Compatibility wrapper for player classification helpers.

New code should import from ``src.classification.player_detection``. This module
keeps older unit tests and scripts working during handover.
"""

from src.classification.player_detection import *  # noqa: F401,F403
