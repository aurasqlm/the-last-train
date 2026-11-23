"""Unified dispatcher for Acts I–VIII of THE LAST RAIN TRAIN."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from . import act1, act2, act3, act4, act5, act6, act7, act8
from ..storyboard import ShotSpec

ACT_MODULES = {
    1: act1,
    2: act2,
    3: act3,
    4: act4,
    5: act5,
    6: act6,
    7: act7,
    8: act8,
}


def get_act_module(act_num: int):
    if act_num not in ACT_MODULES:
        raise ValueError(f"Unknown act number: {act_num}")
    return ACT_MODULES[act_num]


def prepare_shot_cache(
    shot_spec: ShotSpec, width: int = 1920, height: int = 1080
) -> Dict[str, Any]:
    return get_act_module(shot_spec.act).prepare_shot_cache(shot_spec, width, height)


def render_shot_frame(
    shot_spec: ShotSpec,
    local_t: float,
    global_t: float,
    width: int = 1920,
    height: int = 1080,
    cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    return get_act_module(shot_spec.act).render_shot_frame(
        shot_spec, local_t, global_t, width=width, height=height, cache=cache
    )
