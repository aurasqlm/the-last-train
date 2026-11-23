"""THE LAST RAIN TRAIN — 100% pure-code cinema-grade animated short film pipeline."""

from .storyboard import (
    ACT_RANGES,
    FPS,
    HEIGHT,
    SHOTS,
    TOTAL_DURATION_S,
    TOTAL_FRAMES,
    WIDTH,
    ShotSpec,
    get_act_shots,
    get_shot,
    get_shot_at_frame,
    hex_to_rgb,
    verify_storyboard,
)

__all__ = [
    "ACT_RANGES",
    "FPS",
    "HEIGHT",
    "SHOTS",
    "TOTAL_DURATION_S",
    "TOTAL_FRAMES",
    "WIDTH",
    "ShotSpec",
    "get_act_shots",
    "get_shot",
    "get_shot_at_frame",
    "hex_to_rgb",
    "verify_storyboard",
]
