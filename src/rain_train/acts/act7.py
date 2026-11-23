"""Act VII — THE DISTANT HORIZON (Shots S34–S37, 03:55.0–04:35.0).

Palette: Soft Lavender Dusk #3E3A5C, Sunset Peach #F6C7A8, Deep Indigo #191D32.
Each shot has its own distinct camera angle, composition, framing, lighting,
and purposeful 60fps motion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..painterly.clouds import render_cumulus_cathedral, render_sky_gradient
from ..painterly.ink import draw_lantern, draw_train_profile
from ..painterly.optics import (
    apply_atmospheric_haze,
    apply_filmic_grade,
    apply_halation_bloom,
)
from ..painterly.paper_grain import generate_paper_grain
from ..painterly.weather import render_sea_wake


def _over(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(bg * (1.0 - a) + fg * a, 0.0, 1.0)


def _make_vignette_mask(h: int, w: int, strength: float = 0.24) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    return np.clip(1.0 - strength * np.clip(r2 - 0.35, 0.0, 2.0) * 0.65, 0.56, 1.0).astype(np.float32)


def _fast_finish(
    img: np.ndarray,
    grain: np.ndarray,
    vig: np.ndarray,
    grain_strength: float = 0.028,
    bloom_thresh: float = 0.72,
    bloom_tight: float = 0.36,
    bloom_wide: float = 0.24,
) -> np.ndarray:
    h, w = img.shape[:2]
    qw, qh = max(32, w // 4), max(18, h // 4)
    small = cv2.resize(img, (qw, qh), interpolation=cv2.INTER_AREA)
    hi = np.clip((small - bloom_thresh) / max(1.0 - bloom_thresh, 1e-4), 0.0, 1.0)
    b1 = cv2.GaussianBlur(hi, (0, 0), sigmaX=max(2.0, qw / 90.0))
    b2 = cv2.GaussianBlur(hi, (0, 0), sigmaX=max(5.0, qw / 32.0))
    bloom = cv2.resize(b1 * bloom_tight + b2 * bloom_wide, (w, h), interpolation=cv2.INTER_LINEAR)
    img = 1.0 - (1.0 - img) * (1.0 - np.clip(bloom, 0.0, 0.85))
    img = np.clip((img - 0.5) * 1.05 + 0.5 + (grain[..., None] - 0.5) * grain_strength, 0.0, 1.0)
    img *= vig[..., None]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _fast_dusk_water(
    img: np.ndarray,
    horizon_y_frac: float,
    time_s: float,
    ripple_amp: float = 1.6,
    darkness: float = 0.86,
) -> np.ndarray:
    h, w = img.shape[:2]
    hy = int(np.clip(horizon_y_frac * h, 1, h - 2))
    water_h = h - hy
    if water_h <= 1:
        return img
    out = img.copy()
    src_h = min(hy, water_h)
    ref = img[hy - src_h : hy][::-1]
    if src_h < water_h:
        ref = cv2.resize(ref, (w, water_h), interpolation=cv2.INTER_LINEAR)

    hw, hh = max(32, w // 2), max(16, water_h // 2)
    yy = np.linspace(0.0, 1.0, hh, dtype=np.float32)
    xx = np.arange(hw, dtype=np.float32)
    depth = 0.12 + 0.88 * yy
    scale_x = w / 960.0
    dx_half = (
        np.sin(yy[:, None] * 42.0 - time_s * 1.6 + xx[None, :] * 0.024) * 0.65
        + np.sin(yy[:, None] * 92.0 + time_s * 2.4) * 0.35
    ) * (ripple_amp * scale_x * depth[:, None])
    dx = cv2.resize(dx_half, (w, water_h), interpolation=cv2.INTER_LINEAR)

    map_x = np.clip(np.arange(w, dtype=np.float32)[None, :] + dx, 0, w - 1).astype(np.float32)
    map_y = np.broadcast_to(np.arange(water_h, dtype=np.float32)[:, None], (water_h, w)).copy()
    warped = cv2.remap(ref, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    yy_full = np.linspace(0.0, 1.0, water_h, dtype=np.float32)
    fade = (darkness - 0.12 * yy_full)[:, None, None]
    out[hy:] = np.clip(warped * fade, 0.0, 1.0)
    return out


def _draw_stars(
    u8: np.ndarray,
    stars: List[Tuple[float, float, float, float]],
    local_t: float,
    y_max_frac: float,
    alpha_scale: float = 1.0,
) -> None:
    h, w = u8.shape[:2]
    for sx, sy, rad_px, phase in stars:
        if sy >= y_max_frac:
            continue
        tw = (0.55 + 0.45 * np.sin(local_t * 2.2 + phase)) * alpha_scale
        if tw <= 0.05:
            continue
        px, py = int(sx * w), int(sy * h)
        r = max(1, int(round(rad_px * w / 960.0)))
        col = (int(248 * tw), int(242 * tw), int(255 * tw))
        cv2.circle(u8, (px, py), r, col, -1, cv2.LINE_AA)


def prepare_shot_cache(shot_spec: Any, width: int = 1920, height: int = 1080) -> Dict[str, Any]:
    """Pre-render static lavender-peach dusk skies, clouds, starfields, and paper grain."""
    sid = shot_spec.id
    seed = 700 + int(sid[1:])
    grain = generate_paper_grain(height, width, seed=seed)
    vig = _make_vignette_mask(height, width, strength=0.24)
    rng = np.random.default_rng(seed)
    stars = [
        (float(rng.uniform(0.03, 0.97)), float(rng.uniform(0.03, 0.55)), float(rng.uniform(1.0, 2.3)), float(rng.uniform(0, 6.28)))
        for _ in range(55)
    ]
    cache: Dict[str, Any] = {"id": sid, "w": width, "h": height, "grain": grain, "vig": vig, "stars": stars}

    if sid == "S34":
        # Ultra-Wide Pullback: warm-lit train growing smaller across mirrored lavender-peach sea
        horizon_y = 0.52
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.10, 0.11, 0.21)),
                (0.24, (0.24, 0.23, 0.36)),
                (0.42, (0.64, 0.46, 0.52)),
                (0.52, (0.96, 0.78, 0.66)),
                (1.00, (0.12, 0.14, 0.24)),
            ],
            seed=34,
            brush_strength=0.025,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[(0.20, 0.26, 0.24, 0.24), (0.76, 0.24, 0.26, 0.26)],
            base_y_frac=horizon_y - 0.015,
            color_ramp=[
                (0.18, 0.18, 0.30),
                (0.42, 0.35, 0.48),
                (0.84, 0.62, 0.58),
                (0.98, 0.84, 0.72),
            ],
            sun_dir=(0.0, -1.0),
            rim_color=(0.98, 0.84, 0.70),
            rim_strength=0.62,
            seed=340,
        )
        bg = _over(sky, c_rgb, c_a * 0.85)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.86, 0.70, 0.64), strength=0.30, band_height=0.075)
        cache["bg"] = bg
        cache["horizon_y"] = horizon_y

    elif sid == "S35":
        # Water-Reflection Medium: glassy twilight water reflecting first pale stars & peach clouds
        horizon_y = 0.26
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.09, 0.11, 0.20)),
                (0.14, (0.22, 0.21, 0.35)),
                (0.26, (0.94, 0.76, 0.65)),
                (1.00, (0.10, 0.12, 0.22)),
            ],
            seed=35,
            brush_strength=0.024,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[(0.30, 0.12, 0.26, 0.14), (0.74, 0.14, 0.22, 0.12)],
            base_y_frac=horizon_y - 0.01,
            color_ramp=[
                (0.20, 0.20, 0.34),
                (0.50, 0.40, 0.52),
                (0.88, 0.68, 0.62),
                (0.98, 0.84, 0.72),
            ],
            sun_dir=(0.2, -0.9),
            rim_color=(0.98, 0.84, 0.70),
            rim_strength=0.55,
            seed=350,
        )
        bg = _over(sky, c_rgb, c_a * 0.80)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.76, 0.64, 0.62), strength=0.25, band_height=0.06)
        cache["bg"] = bg
        cache["horizon_y"] = horizon_y

    elif sid == "S36":
        # Long Telephoto Horizon: tiny warm amber cabin lantern glow receding into deep indigo distance
        horizon_y = 0.50
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.07, 0.09, 0.17)),
                (0.32, (0.16, 0.16, 0.28)),
                (0.47, (0.36, 0.31, 0.44)),
                (0.50, (0.64, 0.52, 0.54)),
                (1.00, (0.08, 0.10, 0.18)),
            ],
            seed=36,
            brush_strength=0.022,
        )
        bg = apply_atmospheric_haze(sky, horizon_y, (0.44, 0.38, 0.50), strength=0.34, band_height=0.07)
        cache["bg"] = bg
        cache["horizon_y"] = horizon_y

    elif sid == "S37":
        # Vast Horizon Silhouette: lavender-indigo sea & sky horizon, faint train trail dissolving
        horizon_y = 0.58
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.06, 0.08, 0.16)),
                (0.30, (0.14, 0.15, 0.27)),
                (0.50, (0.30, 0.27, 0.42)),
                (0.58, (0.78, 0.64, 0.60)),
                (1.00, (0.08, 0.10, 0.18)),
            ],
            seed=37,
            brush_strength=0.024,
        )
        bg = apply_atmospheric_haze(sky, horizon_y, (0.52, 0.44, 0.54), strength=0.28, band_height=0.07)
        cache["bg"] = bg
        cache["horizon_y"] = horizon_y

    return cache


def render_shot_frame(
    shot_spec: Any,
    local_t: float,
    global_t: float,
    width: int = 1920,
    height: int = 1080,
    cache: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Render one 60fps frame for Act VII (S34–S37) as uint8 RGB (height, width, 3)."""
    if cache is None or cache.get("w") != width or cache.get("h") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(shot_spec.duration_s, 1e-3)
    prog = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]
    vig = cache["vig"]

    if sid == "S34":
        # Ultra-Wide Pullback: warm-lit train growing smaller across mirrored lavender-peach sea
        hy = cache["horizon_y"]
        img = cache["bg"].copy()
        # Train shrinks smoothly as camera pulls back
        train_len = 0.26 * (1.0 - 0.52 * prog)
        train_h = 0.042 * (1.0 - 0.50 * prog)
        cx = 0.50 + 0.08 * prog
        img = draw_train_profile(
            img,
            x_start_frac=cx - train_len * 0.5,
            x_end_frac=cx + train_len * 0.5,
            y_bottom_frac=hy + 0.002,
            height_frac=train_h,
            body_color=(0.12, 0.11, 0.20),
            stripe_color=(0.44, 0.24, 0.26),
            window_color=(1.00, 0.86, 0.56),
            headlight_left=False,
            num_cars=3,
        )
        img = _fast_dusk_water(img, hy, local_t, ripple_amp=1.8, darkness=0.86)
        img = render_sea_wake(
            img,
            bow_xy=(cx + train_len * 0.5, hy + 0.006),
            tail_xy=(max(0.05, cx - train_len * 0.9), hy + 0.010),
            wake_width_frac=0.028 * (1.0 - 0.4 * prog),
            foam_color=(0.94, 0.84, 0.80),
            opacity=0.65 * (1.0 - 0.3 * prog),
            seed=34,
        )
        return _fast_finish(img, grain, vig)

    elif sid == "S35":
        # Water-Reflection Medium: glassy twilight water reflecting first pale stars
        hy = cache["horizon_y"]
        u8 = np.clip(cache["bg"] * 255.0, 0, 255).astype(np.uint8)
        _draw_stars(u8, cache["stars"], local_t, y_max_frac=hy - 0.02, alpha_scale=0.35 + 0.65 * prog)
        img = u8.astype(np.float32) / 255.0
        img = _fast_dusk_water(img, hy, local_t, ripple_amp=1.4, darkness=0.88)
        # Also draw reflected pale stars in the glassy twilight water below
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        ref_stars = [(sx, hy + (0.65 - sy) * 0.95, rad * 1.15, ph) for sx, sy, rad, ph in cache["stars"] if sy < 0.52]
        _draw_stars(u8, ref_stars, local_t + 0.4, y_max_frac=0.96, alpha_scale=0.30 + 0.55 * prog)
        img = u8.astype(np.float32) / 255.0
        return _fast_finish(img, grain, vig, bloom_thresh=0.70, bloom_tight=0.40, bloom_wide=0.26)

    elif sid == "S36":
        # Long Telephoto Horizon: tiny warm amber cabin lantern glow receding into deep indigo distance
        hy = cache["horizon_y"]
        u8 = np.clip(cache["bg"] * 255.0, 0, 255).astype(np.uint8)
        _draw_stars(u8, cache["stars"], local_t, y_max_frac=hy - 0.04, alpha_scale=0.65)
        img = u8.astype(np.float32) / 255.0
        img = _fast_dusk_water(img, hy, local_t, ripple_amp=1.5, darkness=0.86)
        # Tiny receding amber lantern cluster on the horizon
        lx = 0.50 + 0.04 * prog
        glow_str = 0.90 * (1.0 - 0.55 * prog)
        rad = 0.008 * (1.0 - 0.45 * prog)
        img = draw_lantern(img, lx, hy - 0.004, radius_frac=rad, glow_radius_mult=10.0, glow_strength=glow_str, hanging=False, water_reflection=True)
        return _fast_finish(img, grain, vig, bloom_thresh=0.66, bloom_tight=0.46, bloom_wide=0.34)

    else:
        # S37: Vast Horizon Silhouette — ultra-slow crane-up, faint train trail dissolving into starlight
        hy = cache["horizon_y"] + 0.04 * prog
        u8 = np.clip(cache["bg"] * 255.0, 0, 255).astype(np.uint8)
        _draw_stars(u8, cache["stars"], local_t, y_max_frac=hy - 0.03, alpha_scale=0.85)
        img = u8.astype(np.float32) / 255.0
        img = _fast_dusk_water(img, hy, local_t, ripple_amp=1.1, darkness=0.88)
        # Faint dissolving wake line on the glassy sea
        fade_wake = max(0.0, 1.0 - prog * 1.15)
        if fade_wake > 0.02:
            img = render_sea_wake(
                img,
                bow_xy=(0.68, hy + 0.005),
                tail_xy=(0.30, hy + 0.008),
                wake_width_frac=0.016,
                foam_color=(0.86, 0.80, 0.86),
                opacity=0.45 * fade_wake,
                seed=37,
            )
        return _fast_finish(img, grain, vig, bloom_thresh=0.74, bloom_tight=0.34, bloom_wide=0.22)
