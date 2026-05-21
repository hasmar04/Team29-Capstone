"""Compatibility wrapper for the modular batch processor.

New code should import from ``src.processing.batch_processor``. This module is
kept so older scripts and tests that import ``src.batch_processor`` continue to
work during handover.
"""

from src.processing.batch_processor import *  # noqa: F401,F403
