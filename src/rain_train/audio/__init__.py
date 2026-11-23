"""Pure-code 300.0s stereo soundtrack & shot-accurate foley package for THE LAST RAIN TRAIN."""

from __future__ import annotations

SR: int = 48000
DURATION_S: float = 300.0
N_SAMPLES: int = int(round(SR * DURATION_S))  # 14,400,000 exact

__all__ = ["SR", "DURATION_S", "N_SAMPLES"]
