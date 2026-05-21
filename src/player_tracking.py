"""Compatibility wrapper for player tracking helpers.

New code should import from ``src.tracking.player_tracking``. This module keeps
older scripts that import ``src.player_tracking`` working during handover.
"""

from src.tracking.player_tracking import *  # noqa: F401,F403
