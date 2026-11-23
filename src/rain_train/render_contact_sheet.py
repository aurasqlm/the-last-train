"""Render the 8x5 40-shot widescreen contact sheet (`contact_sheet_5min.png`)."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rain_train.acts import prepare_shot_cache, render_shot_frame
from rain_train.storyboard import SHOTS


def render_contact_sheet(
    out_path: str = "contact_sheet_5min.png",
    tile_w: int = 480,
    tile_h: int = 270,
    cols: int = 8,
    rows: int = 5,
) -> None:
    t0 = time.time()
    assert len(SHOTS) == cols * rows, f"Expected {cols*rows} shots, got {len(SHOTS)}"
    pad = 8
    header_h = 44
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = header_h + rows * tile_h + (rows + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 18, dtype=np.uint8)

    # Header bar
    cv2.putText(
        canvas,
        "THE LAST RAIN TRAIN  |  40-SHOT 8-ACT CONTACT SHEET (300.0s / 1920x1080 / 60fps)",
        (pad + 6, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 228, 214),
        2,
        cv2.LINE_AA,
    )

    for idx, shot in enumerate(SHOTS):
        r = idx // cols
        c = idx % cols
        local_t = 0.55 * shot.duration_s
        global_t = shot.start_s + local_t
        cache = prepare_shot_cache(shot, tile_w, tile_h)
        frame = render_shot_frame(shot, local_t, global_t, tile_w, tile_h, cache=cache)
        if frame.dtype != np.uint8:
            u8 = np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
        else:
            u8 = frame

        # Subtle dark label bar at bottom-left of tile
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        label = f"{shot.id} A{shot.act} {shot.start_s:05.1f}s ({shot.duration_s:.1f}s)"
        cv2.rectangle(bgr, (0, tile_h - 22), (230, tile_h), (12, 14, 18), -1)
        cv2.putText(
            bgr,
            label,
            (6, tile_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (240, 232, 218),
            1,
            cv2.LINE_AA,
        )

        y0 = header_h + pad + r * (tile_h + pad)
        x0 = pad + c * (tile_w + pad)
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = bgr
        print(f"  [{idx+1:02d}/40] {shot.id} (Act {shot.act}) rendered", flush=True)

    cv2.imwrite(out_path, canvas)
    dt = time.time() - t0
    print(f"Saved {out_path} ({canvas_w}x{canvas_h}) in {dt:.1f}s", flush=True)


if __name__ == "__main__":
    render_contact_sheet()
