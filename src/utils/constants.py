"""
constants.py
----------------
This module defines global constant dictionaries and variables that are used across the program. 
These constants include mappings for model class numbers, pose estimation keypoints, and a 
scale representation of a rugby field. Additionally, it provides a list of strings for user 
interaction during point selection.

Constants:
-----------
- `RUCK_MODEL_CLASS_NUMBERS`: Dictionary mapping class names to IDs for the ruck model.
- `LINEOUT_MODEL_CLASS_NUMBERS`: Dictionary mapping class names to IDs for the lineout model.
- `BALL_MODEL_CLASS_NUMBERS`: Dictionary mapping class names to IDs for the ball model.
- `ANKLE_POINT_NUMBERS`: Dictionary mapping keypoint names to IDs for ankle positions in pose estimation.
- `FIELD_POINTS_DICT`: Dictionary providing a scaled representation of key points and lines on a rugby field.
- `POINT_SELECTION_STRING`: List of strings used for user interaction to select specific points on the field.
"""

# Constant dictionary giving class IDs for the ruck model
RUCK_MODEL_CLASS_NUMBERS = {
    "Ball": 0,
    "Ruck": 1
}

# Constant dictionary giving class IDs for the lineout model
LINEOUT_MODEL_CLASS_NUMBERS = {
    "Ball": 0,
    "Hooker": 1,
    "Lineout": 2,
    "Referee": 3
}

BALL_MODEL_CLASS_NUMBERS = {
    "Ball": 0
}

# Constant dictionary giving IDs for the ankles in the pose estimation model
ANKLE_POINT_NUMBERS = {
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16
}

# Constant dictionary providing a scale replica of all keypoints on a rugby field
# 10 pixels/metre
FIELD_POINTS_DICT = {
    "left_tryline": {
        "top": (0, 0),
        "top_5m": (0, 50),
        "top_15m": (0, 150),
        "bottom_15m": (0, 550),
        "bottom_5m": (0, 650),
        "bottom": (0, 700),
        "line": (0, 0, 0, 700)
    },
    "left_5m_line": {
        "top": (50, 0),
        "top_5m": (50, 50),
        "top_15m": (50, 150),
        "bottom_15m": (50, 550),
        "bottom_5m": (50, 650),
        "bottom": (50, 700),
        "line": (50, 0, 50, 700)
    },
    "left_22m_line": {
        "top": (220, 0),
        "top_5m": (220, 50),
        "top_15m": (220, 150),
        "bottom_15m": (220, 550),
        "bottom_5m": (220, 650),
        "bottom": (220, 700),
        "line": (220, 0, 220, 700)
    },
    "left_10m_line": {
        "top": (400, 0),
        "top_5m": (400, 50),
        "top_15m": (400, 150),
        "bottom_15m": (400, 550),
        "bottom_5m": (400, 650),
        "bottom": (400, 700),
        "line": (400, 0, 400, 700)
    },
    "halfway_line": {
        "top": (500, 0),
        "top_5m": (500, 50),
        "top_15m": (500, 150),
        "centre": (500, 350),
        "bottom_15m": (500, 550),
        "bottom_5m": (500, 650),
        "bottom": (500, 700),
        "line": (500, 0, 500, 700)
    },
    "right_10m_line": {
        "top": (600, 0),
        "top_5m": (600, 50),
        "top_15m": (600, 150),
        "bottom_15m": (600, 550),
        "bottom_5m": (600, 650),
        "bottom": (600, 700),
        "line": (600, 0, 600, 700)
    },
    "right_22m_line": {
        "top": (780, 0),
        "top_5m": (780, 50),
        "top_15m": (780, 150),
        "bottom_15m": (780, 550),
        "bottom_5m": (780, 650),
        "bottom": (780, 700),
        "line": (780, 0, 780, 700)
    },
    "right_5m_line": {
        "top": (950, 0),
        "top_5m": (950, 50),
        "top_15m": (950, 150),
        "bottom_15m": (950, 550),
        "bottom_5m": (950, 650),
        "bottom": (950, 700),
        "line": (950, 0, 950, 700)
    },
    "right_tryline": {
        "top": (1000, 0),
        "top_5m": (1000, 50),
        "top_15m": (1000, 150),
        "bottom_15m": (1000, 550),
        "bottom_5m": (1000, 650),
        "bottom": (1000, 700),
        "line": (1000, 0, 1000, 700)
    },
    "top_sideline": (0, 0, 1000, 0),
    "bottom_sideline": (0, 700, 1000, 700)
}

POINT_SELECTION_STRING = [
        "Is the dot on the left (1), right (2) or middle of the 50m line (3)?",
        "Is it the try (1), 5m (2), 22m (3), 10m (4) or halfway (5) line?",
        "Is it on the top (1) or bottom (2) half of the field?",
        "Is it on the side (1), 5m (2), 15m (3) or halfway (4) line?"
    ]