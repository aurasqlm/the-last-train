"""Act IV — THROUGH THE PINE VALLEY & TUNNEL (Shots S17–S21, 01:50.0–02:25.0).

Palette: Midnight Pine (#0E1A1F), Tunnel Obsidian (#07090D), Flickering Amber (#E8A852), Exit Twilight (#EEDEC7 / #122347).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..painterly.clouds import render_sky_gradient
from ..painterly.ink import draw_lantern, draw_train_profile
from ..painterly.optics import (
    apply_atmospheric_haze,
    apply_filmic_grade,
    apply_halation_bloom,
    apply_vignette,
)
from ..painterly.paper_grain import (
    apply_paper_grain,
    apply_pigment_granulation,
    generate_paper_grain,
)
from ..painterly.weather import render_slanted_rain
from ..storyboard import ShotSpec


def _over(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(bg * (1.0 - a) + fg * a, 0.0, 1.0)


def _draw_pine_ridge(
    img: np.ndarray,
    base_y_frac: float,
    amp_frac: float,
    color: Tuple[float, float, float],
    seed: int,
) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    profile = np.zeros(w, dtype=np.float32)
    for freq, amp in ((3.0, 0.45), (9.0, 0.25), (27.0, 0.16), (65.0, 0.14)):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        profile += amp * np.sin(xs * freq * np.pi + phase)
    ridge_y = (base_y_frac - amp_frac * (0.5 + 0.5 * profile)) * h
    yy = np.arange(h, dtype=np.float32)[:, None]
    mask = np.clip((yy - ridge_y[None, :]) / 2.5, 0.0, 1.0)
    col = np.array(color, dtype=np.float32)[None, None, :]
    return _over(img, np.broadcast_to(col, img.shape), mask)


def _finalize(
    img: np.ndarray,
    grain: np.ndarray,
    vig_strength: float = 0.30,
    bloom_thresh: float = 0.60,
    bloom_tight: float = 0.48,
    bloom_wide: float = 0.34,
) -> np.ndarray:
    h, w = img.shape[:2]
    bw, bh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
    bright = np.maximum(small - bloom_thresh, 0.0)
    tight = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(1.5, bw / 160.0))
    wide = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(4.0, bw / 48.0))
    bloom = cv2.resize(tight * bloom_tight + wide * bloom_wide, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.clip(img + bloom, 0.0, 1.0)
    g = (grain[..., None] - 0.5) * 0.050
    img = (img + g - 0.5) * 1.07 + 0.5
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    vig = np.clip(1.0 - vig_strength * 0.5 * (xx * xx + yy * yy), 0.45, 1.0)[..., None]
    return (np.clip(img * vig, 0.0, 1.0) * 255.0).astype(np.uint8)


def prepare_shot_cache(
    shot_spec: ShotSpec,
    width: int = 1920,
    height: int = 1080,
) -> Dict[str, Any]:
    """Pre-render static painterly layers for Act IV shots (S17–S21)."""
    sid = shot_spec.id
    grain = generate_paper_grain(height, width, seed=int(sid[1:]) * 29)
    cache: Dict[str, Any] = {"id": sid, "width": width, "height": height, "grain": grain}

    if sid == "S17":
        # High exterior wide: illuminated train crossing high steel trestle bridge over mist-filled midnight pine valley
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.03, 0.06, 0.09)),
                (0.42, (0.06, 0.11, 0.15)),
                (0.72, (0.09, 0.16, 0.20)),
                (1.00, (0.04, 0.08, 0.10)),
            ],
            seed=417,
            brush_strength=0.026,
        )
        bg = _draw_pine_ridge(bg, 0.56, 0.18, (0.05, 0.10, 0.13), seed=41)
        bg = apply_atmospheric_haze(bg, 0.62, (0.14, 0.22, 0.27), strength=0.45, band_height=0.18)
        bg = _draw_pine_ridge(bg, 0.78, 0.14, (0.03, 0.06, 0.08), seed=42)

        # High steel trestle bridge deck & vertical truss piers
        u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
        deck_y = int(height * 0.52)
        cv2.rectangle(u8, (0, deck_y), (width, deck_y + max(6, height // 80)), (18, 26, 32), -1)
        for px_f in (0.14, 0.34, 0.54, 0.74, 0.90):
            px = int(px_f * width)
            pw = max(6, int(width * 0.016))
            cv2.line(u8, (px - pw, deck_y), (px - pw * 2, height), (14, 20, 25), max(3, width // 360), cv2.LINE_AA)
            cv2.line(u8, (px + pw, deck_y), (px + pw * 2, height), (14, 20, 25), max(3, width // 360), cv2.LINE_AA)
            for k in range(5):
                y0 = deck_y + int(k * (height - deck_y) / 5)
                y1 = deck_y + int((k + 1) * (height - deck_y) / 5)
                cv2.line(u8, (px - pw, y0), (px + pw, y1), (16, 22, 28), max(1, width // 640), cv2.LINE_AA)
                cv2.line(u8, (px + pw, y0), (px - pw, y1), (16, 22, 28), max(1, width // 640), cv2.LINE_AA)
        cache["valley_bridge"] = u8.astype(np.float32) / 255.0

    elif sid == "S18":
        # Exterior side tracking: dense dark pine silhouettes rushing past warm strobing carriage windows
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.04, 0.07, 0.10)),
                (0.55, (0.07, 0.12, 0.16)),
                (1.00, (0.03, 0.05, 0.07)),
            ],
            seed=418,
            brush_strength=0.025,
        )
        # Warm glowing train profile in mid-ground
        bg = draw_train_profile(
            bg,
            x_start_frac=-0.05,
            x_end_frac=1.05,
            y_bottom_frac=0.72,
            height_frac=0.22,
            body_color=(0.10, 0.14, 0.19),
            stripe_color=(0.52, 0.24, 0.19),
            window_color=(1.00, 0.80, 0.42),
            num_cars=3,
        )
        cache["train_bg"] = bg
        rng = np.random.default_rng(418)
        cache["pines"] = [
            (rng.uniform(0.0, 1.5), rng.uniform(0.65, 1.15), rng.uniform(0.035, 0.085), rng.uniform(0.7, 1.4))
            for _ in range(18)
        ]

    elif sid == "S19":
        # Front-quarter low exterior: train plunging into cavernous stone tunnel mouth
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.03, 0.05, 0.08)),
                (0.50, (0.06, 0.10, 0.13)),
                (1.00, (0.03, 0.04, 0.06)),
            ],
            seed=419,
            brush_strength=0.025,
        )
        bg = _draw_pine_ridge(bg, 0.36, 0.20, (0.04, 0.08, 0.10), seed=49)
        u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
        # Stone cliffside & cavernous arched tunnel portal
        cv2.fillPoly(u8, [np.array([[0, int(height * 0.25)], [width, int(height * 0.18)], [width, height], [0, height]], dtype=np.int32)], (20, 25, 29), cv2.LINE_AA)
        cx, cy = int(width * 0.56), int(height * 0.62)
        rx, ry = int(width * 0.22), int(height * 0.34)
        # Stone arch rim
        cv2.ellipse(u8, (cx, cy), (rx + int(width * 0.035), ry + int(height * 0.045)), 0, 180, 360, (44, 48, 52), -1, cv2.LINE_AA)
        # Obsidian tunnel mouth interior
        cv2.ellipse(u8, (cx, cy), (rx, ry), 0, 180, 360, (7, 9, 13), -1, cv2.LINE_AA)
        cv2.rectangle(u8, (cx - rx, cy), (cx + rx, height), (7, 9, 13), -1)
        cache["tunnel_portal"] = u8.astype(np.float32) / 255.0

    elif sid == "S20":
        # Interior medium close-up: obsidian dark cabin with passenger silhouette ready for strobing amber tunnel lamps
        u8 = np.full((height, width, 3), (8, 10, 14), dtype=np.uint8)
        # Window frame & dark seat silhouette
        cv2.rectangle(u8, (int(width * 0.08), int(height * 0.10)), (int(width * 0.68), int(height * 0.72)), (5, 7, 10), -1)
        cv2.rectangle(u8, (int(width * 0.08), int(height * 0.10)), (int(width * 0.68), int(height * 0.72)), (26, 20, 16), max(4, width // 240))
        # Passenger head & shoulders silhouette on right
        cv2.circle(u8, (int(width * 0.76), int(height * 0.44)), int(height * 0.13), (18, 14, 12), -1, cv2.LINE_AA)
        cv2.ellipse(u8, (int(width * 0.78), int(height * 0.85)), (int(width * 0.20), int(height * 0.32)), 0, 180, 360, (16, 12, 10), -1, cv2.LINE_AA)
        cache["dark_cabin"] = u8.astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cache["xx_norm"] = xx / float(width)
        cache["yy_norm"] = yy / float(height)

    elif sid == "S21":
        # Interior forward POV: dark tunnel walls converging to a pinpoint of silver-blue twilight
        u8 = np.full((height, width, 3), (7, 9, 13), dtype=np.uint8)
        cx, cy = width // 2, int(height * 0.50)
        for r_frac in (0.48, 0.36, 0.26, 0.18, 0.12):
            rx = int(width * r_frac)
            ry = int(height * r_frac * 1.15)
            cv2.ellipse(u8, (cx, cy), (rx, ry), 0, 175, 365, (18, 21, 26), max(2, width // 360), cv2.LINE_AA)
        cache["tunnel_rings"] = u8.astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cache["dist2"] = ((xx - width * 0.5) / (width * 0.5)) ** 2 + ((yy - height * 0.5) / (height * 0.5)) ** 2

    return cache


def render_shot_frame(
    shot_spec: ShotSpec,
    local_t: float,
    global_t: float,
    width: int = 1920,
    height: int = 1080,
    cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Render one 60fps RGB uint8 frame for Act IV (S17–S21)."""
    if cache is None or cache.get("width") != width or cache.get("height") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(0.001, shot_spec.duration_s)
    u = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]

    if sid == "S17":
        # Train gliding smoothly across high trestle bridge over mist-filled midnight pine valley
        img = cache["valley_bridge"].copy()
        x_start = 0.04 + 0.36 * u
        img = draw_train_profile(
            img,
            x_start_frac=x_start,
            x_end_frac=x_start + 0.52,
            y_bottom_frac=0.52,
            height_frac=0.075,
            body_color=(0.09, 0.13, 0.18),
            stripe_color=(0.56, 0.25, 0.20),
            window_color=(1.00, 0.80, 0.42),
            num_cars=4,
        )
        img = apply_atmospheric_haze(img, 0.70 + 0.02 * np.sin(local_t * 0.6), (0.14, 0.22, 0.27), strength=0.32, band_height=0.16)
        img = render_slanted_rain(img, time_s=global_t, density=0.95, angle_deg=14.0, seed=417)
        return _finalize(img, grain, vig_strength=0.30)

    if sid == "S18":
        # Fast lateral whip-pan: dark pine silhouettes rushing past warm strobing carriage windows
        img = cache["train_bg"].copy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        for px0, h_f, w_f, spd in cache["pines"]:
            px = int((1.3 - ((px0 + local_t * spd * 1.15) % 1.6)) * width)
            pw = int(w_f * width)
            # Dark pine trunk & jagged conifer boughs
            cv2.rectangle(u8, (px - pw // 5, int(height * (1.0 - h_f))), (px + pw // 5, height), (6, 10, 12), -1)
            for tier in range(4):
                ty = int(height * (1.0 - h_f + tier * 0.18))
                tw = int(pw * (0.7 + tier * 0.45))
                tri = np.array([[px, ty - int(height * 0.14)], [px - tw, ty + int(height * 0.08)], [px + tw, ty + int(height * 0.08)]], dtype=np.int32)
                cv2.fillPoly(u8, [tri], (8, 14, 16), cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        img = render_slanted_rain(img, time_s=global_t, density=0.80, angle_deg=24.0, seed=418)
        return _finalize(img, grain, vig_strength=0.32)

    if sid == "S19":
        # Train plunges suddenly into cavernous stone tunnel mouth; rain cuts off into obsidian darkness
        img = cache["tunnel_portal"].copy()
        # Train moving into the tunnel mouth
        x_end = 0.62 - 0.14 * u
        img = draw_train_profile(
            img,
            x_start_frac=x_end - 0.56,
            x_end_frac=x_end,
            y_bottom_frac=0.76,
            height_frac=0.19,
            body_color=(0.10, 0.14, 0.19),
            stripe_color=(0.56, 0.25, 0.20),
            window_color=(1.00, 0.80, 0.42),
            num_cars=3,
        )
        # Re-mask the right side of the tunnel mouth so the train disappears inside
        rain_fade = max(0.0, 1.0 - max(0.0, (u - 0.65) / 0.35))
        if rain_fade > 0.05:
            img = render_slanted_rain(img, time_s=global_t, density=1.0 * rain_fade, angle_deg=14.0, seed=419)
        # Plunge to darkness in final 20%
        dark = 1.0 - 0.82 * max(0.0, (u - 0.72) / 0.28)
        img = np.clip(img * dark, 0.0, 1.0)
        return _finalize(img, grain, vig_strength=0.35)

    if sid == "S20":
        # Tight handheld cabin rumble + rhythmic strobing amber tunnel lamps sweeping across glass & face
        rumble_x = int(np.sin(local_t * 34.0) * 3.0 * (width / 1920.0))
        rumble_y = int(np.cos(local_t * 42.0) * 2.5 * (height / 1080.0))
        img = np.roll(np.roll(cache["dark_cabin"], rumble_x, axis=1), rumble_y, axis=0)
        # Sweeping amber tunnel lamp beam (~1.7 Hz)
        sweep = (local_t * 1.7) % 1.0
        beam_x = 1.15 - 1.35 * sweep
        dx = (cache["xx_norm"] - beam_x) / 0.18
        dy = (cache["yy_norm"] - 0.46) / 0.42
        glow = np.exp(-(dx * dx + dy * dy))
        amber = np.array([0.96, 0.64, 0.26], dtype=np.float32)[None, None, :]
        img = np.clip(img + glow[..., None] * amber * 0.85, 0.0, 1.0)
        return _finalize(img, grain, vig_strength=0.38, bloom_thresh=0.45, bloom_tight=0.56, bloom_wide=0.42)

    # S21: Interior forward POV — pinpoint of silver-blue twilight expands into blinding soft optical exit flare
    img = cache["tunnel_rings"].copy()
    # Exponential expansion of the exit light
    expand = 0.015 + 0.95 * (u ** 2.8)
    dist2 = cache["dist2"]
    flare = np.exp(-dist2 / max(0.0008, expand * expand * 1.4))
    core = np.exp(-dist2 / max(0.0002, expand * expand * 0.28))
    twilight = np.array([0.48, 0.64, 0.86], dtype=np.float32)[None, None, :]
    cream = np.array([0.98, 0.94, 0.86], dtype=np.float32)[None, None, :]
    img = np.clip(img + flare[..., None] * twilight * 0.85 + core[..., None] * cream * (0.95 + 0.4 * u), 0.0, 1.0)
    # Soft blinding whiteout bloom at the very end of the shot
    if u > 0.78:
        w_out = (u - 0.78) / 0.22
        img = np.clip(img * (1.0 - w_out * 0.75) + cream * (w_out * 0.75), 0.0, 1.0)
    return _finalize(img, grain, vig_strength=0.22 * (1.0 - u), bloom_thresh=0.42, bloom_tight=0.60, bloom_wide=0.50)
