"""Pytest configuration file.

This file adjusts the Python import path during test discovery so that the
``app`` package is importable when running tests from the repository root.
"""

import os
import sys

# Add the repository root to sys.path to allow ``app`` imports
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
