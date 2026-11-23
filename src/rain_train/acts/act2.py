"""Act II — THE DOWNPOUR & ARRIVAL (Shots S06–S10, 00:35.0–01:10.0).

Palette: Deep Rain Slate (#1B2430), Misty Teal (#466170), Lantern Amber (#F5B969), Headlight (#FFF1D0).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from ..painterly.clouds import render_cumulus_cathedral, render_sky_gradient
from ..painterly.ink import (
    draw_lantern,
    draw_station_perspective,
    draw_train_perspective,
    draw_umbrella_figure,
)
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
from ..painterly.weather import (
    apply_puddle_reflection,
    render_slanted_rain,
    render_steam_plume,
)
from ..storyboard import ShotSpec


def _over(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(bg * (1.0 - a) + fg * a, 0.0, 1.0)


def _draw_hills(
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
    for freq, amp in ((2.0, 0.50), (5.0, 0.28), (11.0, 0.14), (23.0, 0.08)):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        profile += amp * np.sin(xs * freq * np.pi + phase)
    ridge_y = (base_y_frac - amp_frac * (0.5 + 0.5 * profile)) * h
    yy = np.arange(h, dtype=np.float32)[:, None]
    mask = np.clip((yy - ridge_y[None, :]) / 3.0, 0.0, 1.0)
    col = np.array(color, dtype=np.float32)[None, None, :]
    return _over(img, np.broadcast_to(col, img.shape), mask)


def _crop_pan_zoom(
    canvas: np.ndarray,
    w: int,
    h: int,
    cx_frac: float,
    cy_frac: float,
    zoom: float,
) -> np.ndarray:
    ch, cw = canvas.shape[:2]
    crop_w = cw / zoom
    crop_h = ch / zoom
    x0 = int(np.clip(cx_frac * cw - crop_w * 0.5, 0, max(0, cw - int(crop_w))))
    y0 = int(np.clip(cy_frac * ch - crop_h * 0.5, 0, max(0, ch - int(crop_h))))
    x1 = min(cw, x0 + max(2, int(crop_w)))
    y1 = min(ch, y0 + max(2, int(crop_h)))
    sub = canvas[y0:y1, x0:x1]
    if sub.shape[1] == w and sub.shape[0] == h:
        return sub.copy()
    return cv2.resize(sub, (w, h), interpolation=cv2.INTER_LINEAR)


def _finalize(
    img: np.ndarray,
    grain: np.ndarray,
    vig_strength: float = 0.28,
    bloom_thresh: float = 0.64,
    bloom_tight: float = 0.46,
    bloom_wide: float = 0.32,
) -> np.ndarray:
    h, w = img.shape[:2]
    bw, bh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
    bright = np.maximum(small - bloom_thresh, 0.0)
    tight = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(1.5, bw / 160.0))
    wide = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(4.0, bw / 48.0))
    bloom = cv2.resize(tight * bloom_tight + wide * bloom_wide, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.clip(img + bloom, 0.0, 1.0)
    g = (grain[..., None] - 0.5) * 0.055
    img = (img + g - 0.5) * 1.06 + 0.5
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    vig = np.clip(1.0 - vig_strength * 0.5 * (xx * xx + yy * yy), 0.45, 1.0)[..., None]
    return (np.clip(img * vig, 0.0, 1.0) * 255.0).astype(np.uint8)


def prepare_shot_cache(
    shot_spec: ShotSpec,
    width: int = 1920,
    height: int = 1080,
) -> Dict[str, Any]:
    """Pre-render static painterly layers for Act II shots (S06–S10)."""
    sid = shot_spec.id
    grain = generate_paper_grain(height, width, seed=int(sid[1:]) * 19)
    cache: Dict[str, Any] = {"id": sid, "width": width, "height": height, "grain": grain}

    if sid == "S06":
        # Wide empty wooden platform & awning in torrential slanted rain
        cw, ch = int(width * 1.14), int(height * 1.06)
        sky = render_sky_gradient(
            cw, ch,
            stops=[
                (0.00, (0.06, 0.09, 0.14)),
                (0.38, (0.12, 0.19, 0.26)),
                (0.68, (0.22, 0.32, 0.39)),
                (1.00, (0.09, 0.13, 0.18)),
            ],
            seed=206,
            brush_strength=0.030,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            cw, ch,
            towers=[(0.24, 0.24, 0.32, 0.20), (0.68, 0.20, 0.36, 0.24)],
            base_y_frac=0.54,
            color_ramp=[(0.07, 0.11, 0.16), (0.14, 0.21, 0.28), (0.24, 0.34, 0.42), (0.36, 0.46, 0.54)],
            sun_dir=(-0.3, -0.9),
            rim_color=None,
            seed=2060,
        )
        bg = _over(sky, c_rgb, c_a * 0.72)
        bg = _draw_hills(bg, 0.63, 0.12, (0.09, 0.14, 0.19), seed=61)
        bg = apply_atmospheric_haze(bg, 0.62, (0.22, 0.31, 0.38), strength=0.42, band_height=0.16)
        bg = draw_station_perspective(bg, platform_y_frac=0.75, vp_xy=(1.18, 0.56))
        for lx, ly, lr in [(0.12, 0.27, 0.019), (0.34, 0.30, 0.015), (0.53, 0.32, 0.012)]:
            bg = draw_lantern(bg, lx, ly, radius_frac=lr, glow_radius_mult=8.0, glow_strength=0.68, hanging=True)
        cache["canvas"] = bg

    elif sid == "S07":
        # Macro low-angle puddle reflecting glowing amber lanterns on wet concrete
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.07, 0.10, 0.15)),
                (0.35, (0.11, 0.16, 0.22)),
                (0.75, (0.08, 0.11, 0.16)),
                (1.00, (0.05, 0.07, 0.11)),
            ],
            seed=207,
            brush_strength=0.025,
        )
        # Warm glowing lantern reflections embedded in the puddle
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        for lx, ly, rx, ry, col in [
            (0.28, 0.46, 0.16, 0.26, (0.98, 0.72, 0.36)),
            (0.64, 0.40, 0.13, 0.22, (0.96, 0.68, 0.32)),
            (0.84, 0.52, 0.09, 0.16, (0.92, 0.62, 0.28)),
        ]:
            glow = np.exp(-(((xx - lx * width) / (rx * width)) ** 2 + ((yy - ly * height) / (ry * height)) ** 2))
            bg = np.clip(bg + glow[..., None] * np.array(col, dtype=np.float32)[None, None, :] * 0.75, 0.0, 1.0)
        cache["puddle_base"] = bg
        rng = np.random.default_rng(207)
        cache["rings"] = [
            (rng.uniform(0.10, 0.90), rng.uniform(0.18, 0.88), rng.uniform(0.0, 2.4), rng.uniform(0.65, 1.15))
            for _ in range(16)
        ]

    elif sid == "S08":
        # Extreme low-angle wet steel rails gleaming in rain
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.06, 0.09, 0.14)),
                (0.45, (0.14, 0.21, 0.29)),
                (0.60, (0.20, 0.29, 0.36)),
                (0.62, (0.08, 0.10, 0.13)),
                (1.00, (0.05, 0.07, 0.10)),
            ],
            seed=208,
            brush_strength=0.028,
        )
        bg = apply_atmospheric_haze(bg, 0.56, (0.24, 0.34, 0.42), strength=0.48, band_height=0.16)
        u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
        vp_x, vp_y = int(width * 0.52), int(height * 0.56)
        # Wooden railroad sleepers receding in perspective
        for k in range(18):
            t = (k / 18.0) ** 1.8
            y = int(vp_y + t * (height - vp_y))
            half_w = int((0.04 + 0.52 * t) * width)
            thick = max(2, int((2 + 16 * t) * (height / 1080.0)))
            cv2.line(u8, (vp_x - half_w, y), (vp_x + half_w, y), (26, 22, 20), thick, cv2.LINE_AA)
        # Gleaming wet steel rails
        for side in (-1, 1):
            bx = int(vp_x + side * width * 0.34)
            tx = int(vp_x + side * width * 0.015)
            cv2.line(u8, (tx, vp_y), (bx, height), (34, 42, 50), max(4, int(width * 0.014)), cv2.LINE_AA)
            cv2.line(u8, (tx, vp_y), (bx, height), (145, 168, 185), max(2, int(width * 0.005)), cv2.LINE_AA)
        cache["track_bg"] = u8.astype(np.float32) / 255.0
        cache["vp"] = (0.52, 0.56)

    elif sid == "S09":
        # Medium-long telephoto: twin blurred golden headlights piercing blue-teal rain fog
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.05, 0.08, 0.13)),
                (0.45, (0.11, 0.18, 0.25)),
                (0.70, (0.18, 0.28, 0.35)),
                (1.00, (0.06, 0.09, 0.13)),
            ],
            seed=209,
            brush_strength=0.028,
        )
        bg = _draw_hills(bg, 0.60, 0.16, (0.07, 0.11, 0.15), seed=91)
        bg = apply_atmospheric_haze(bg, 0.58, (0.20, 0.30, 0.38), strength=0.55, band_height=0.22)
        cache["fog_bg"] = bg

    elif sid == "S10":
        # Hero S10: 3/4 perspective vintage train braking into lantern-lit station
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.07, 0.10, 0.15)),
                (0.34, (0.13, 0.20, 0.27)),
                (0.66, (0.24, 0.34, 0.41)),
                (1.00, (0.10, 0.14, 0.19)),
            ],
            seed=10,
            brush_strength=0.032,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[(0.22, 0.24, 0.30, 0.20), (0.66, 0.18, 0.36, 0.24), (0.92, 0.26, 0.24, 0.18)],
            base_y_frac=0.54,
            color_ramp=[(0.08, 0.12, 0.17), (0.16, 0.23, 0.30), (0.26, 0.36, 0.44), (0.40, 0.50, 0.57)],
            sun_dir=(-0.3, -0.9),
            rim_color=None,
            seed=110,
        )
        bg = _over(sky, c_rgb, c_a * 0.72)
        bg = _draw_hills(bg, 0.62, 0.13, (0.10, 0.15, 0.19), seed=12)
        bg = _draw_hills(bg, 0.66, 0.08, (0.07, 0.11, 0.15), seed=15)
        bg = apply_atmospheric_haze(bg, 0.62, (0.22, 0.31, 0.38), strength=0.40, band_height=0.16)
        cache["sky_hills"] = bg

        roof_occ = np.zeros((height, width), dtype=np.float32)
        roof_poly = np.array([
            [0, 0],
            [int(width * 0.725), 0],
            [int(width * 0.642), int(height * 0.265)],
            [0, int(height * 0.165)],
        ], dtype=np.int32)
        cv2.fillPoly(roof_occ, [roof_poly], 1.0)
        roof_occ = np.maximum(roof_occ, cv2.GaussianBlur(roof_occ, (0, 0), sigmaX=1.2, sigmaY=1.2))
        cache["roof_occ"] = roof_occ

    return cache


def render_shot_frame(
    shot_spec: ShotSpec,
    local_t: float,
    global_t: float,
    width: int = 1920,
    height: int = 1080,
    cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Render one 60fps RGB uint8 frame for Act II (S06–S10)."""
    if cache is None or cache.get("width") != width or cache.get("height") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(0.001, shot_spec.duration_s)
    u = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]

    if sid == "S06":
        # Slow lateral track across torrential rain-swept empty platform
        cx = 0.45 + 0.10 * u
        img = _crop_pan_zoom(cache["canvas"], width, height, cx, 0.50, zoom=1.04)
        ripples = [(0.20, 0.85, local_t), (0.50, 0.82, local_t + 1.1), (0.78, 0.88, local_t + 2.3)]
        img = apply_puddle_reflection(img, y_start_frac=0.755, time_s=global_t, ripple_centers=ripples, darkness=0.76)
        warm_lights = [
            (0.12, 0.27, 0.16, (1.0, 0.78, 0.42)),
            (0.34, 0.30, 0.15, (1.0, 0.78, 0.42)),
            (0.53, 0.32, 0.14, (1.0, 0.78, 0.42)),
        ]
        img = render_slanted_rain(img, time_s=global_t, density=1.45, angle_deg=16.0, warm_lights=warm_lights, seed=206)
        return _finalize(img, grain)

    if sid == "S07":
        # Macro puddle ripples distorting glowing amber lantern reflections
        img = cache["puddle_base"].copy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        drift_x = int(width * 0.015 * np.sin(local_t * 0.5))
        for rx_f, ry_f, phase, speed in cache["rings"]:
            cycle = (local_t * speed + phase) % 2.2
            norm = cycle / 2.2
            fade = (1.0 - norm) ** 1.4
            cx_px = int(rx_f * width) + drift_x
            cy_px = int(ry_f * height)
            rad_x = max(4, int((0.015 + 0.12 * norm) * width))
            rad_y = max(2, int(rad_x * 0.36))  # perspective foreshortened ellipse
            col = (int(245 * fade + 20), int(185 * fade + 25), int(105 * fade + 35))
            cv2.ellipse(u8, (cx_px, cy_px), (rad_x, rad_y), 0, 0, 360, col, max(1, height // 420), cv2.LINE_AA)
            if norm < 0.25:
                # Bright splash bead at impact
                cv2.circle(u8, (cx_px, cy_px - int(height * 0.012 * (1.0 - norm * 4.0))), max(2, height // 240), (250, 225, 175), -1, cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        img = render_slanted_rain(img, time_s=global_t, density=0.85, angle_deg=12.0, seed=207)
        return _finalize(img, grain, vig_strength=0.32, bloom_thresh=0.58, bloom_tight=0.50, bloom_wide=0.36)

    if sid == "S08":
        # Extreme low-angle vibrating wet steel rails with micro-droplets jumping
        img = cache["track_bg"].copy()
        # Subtle train-approaching tremor
        vib_y = int(np.sin(local_t * 48.0) * (1.0 + 2.5 * u) * (height / 1080.0))
        if vib_y != 0:
            img = np.roll(img, vib_y, axis=0)
        # Distant warm headlight glow waking up at vanishing point
        vp_x, vp_y = cache["vp"]
        img = draw_lantern(img, vp_x, vp_y - 0.01, radius_frac=0.004 + 0.008 * u, glow_radius_mult=10.0, glow_strength=0.25 + 0.55 * u, hanging=False)
        # Jumping micro-droplets along rails
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        for k in range(18):
            t_r = (k + 0.5) / 18.0
            side = -1 if k % 2 == 0 else 1
            rx = int((vp_x + side * (0.015 + 0.325 * t_r)) * width)
            ry = int((vp_y + t_r * (1.0 - vp_y)) * height)
            bounce = abs(np.sin(local_t * (12.0 + k * 0.7) + k)) * height * 0.022 * t_r
            cv2.circle(u8, (rx, int(ry - bounce)), max(2, int(3.5 * t_r * height / 540.0)), (215, 228, 240), -1, cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        img = render_slanted_rain(img, time_s=global_t, density=1.15, angle_deg=14.0, seed=208)
        return _finalize(img, grain, vig_strength=0.30)

    if sid == "S09":
        # Twin blurred golden headlights piercing thick blue-teal rain fog
        img = cache["fog_bg"].copy()
        # Headlights grow and separate slightly as train approaches
        sep = 0.045 + 0.055 * (u ** 1.3)
        rad = 0.016 + 0.022 * (u ** 1.3)
        cy = 0.58 + 0.03 * u
        cx = 0.50 - 0.02 * u
        for sx in (cx - sep, cx + sep):
            img = draw_lantern(img, sx, cy, radius_frac=rad, glow_radius_mult=11.0, glow_strength=0.82, hanging=False)
        warm_lights = [
            (cx - sep, cy, 0.28, (1.0, 0.90, 0.68)),
            (cx + sep, cy, 0.28, (1.0, 0.90, 0.68)),
        ]
        img = render_slanted_rain(img, time_s=global_t, density=1.35, angle_deg=13.0, warm_lights=warm_lights, seed=209)
        return _finalize(img, grain, vig_strength=0.34, bloom_thresh=0.52, bloom_tight=0.58, bloom_wide=0.45)

    # S10: Dramatic decelerating side-pan hero shot — vintage train braking smoothly into station
    ease_out = 1.0 - (1.0 - u) ** 2.3
    front_x = 0.78 - 0.30 * ease_out  # smoothly decelerates into 0.48
    img = cache["sky_hills"].copy()
    img = draw_train_perspective(
        img,
        front_x_frac=front_x,
        front_y_bottom_frac=0.75,
        front_h_frac=0.35,
        vp_xy=(1.18, 0.56),
        body_color=(0.12, 0.17, 0.23),
        stripe_color=(0.58, 0.25, 0.20),
        window_color=(1.00, 0.82, 0.46),
        headlight_beams=True,
    )
    img = draw_station_perspective(img, platform_y_frac=0.75, vp_xy=(1.18, 0.56))

    # Billowing brake steam plumes rising as train brakes
    steam_lift = 0.03 * u
    img = render_steam_plume(img, (front_x - 0.05, 0.72 - steam_lift), (0.14, 0.085), (0.94, 0.86, 0.72), opacity=0.48 * (0.4 + 0.6 * ease_out), seed=101)
    img = render_steam_plume(img, (front_x + 0.12, 0.73 - steam_lift * 0.7), (0.17, 0.075), (0.88, 0.83, 0.75), opacity=0.38 * (0.4 + 0.6 * ease_out), seed=102)

    lanterns = [(0.11, 0.27, 0.019), (0.33, 0.30, 0.015), (0.52, 0.32, 0.012)]
    for lx, ly, lr in lanterns:
        img = draw_lantern(img, lx, ly, radius_frac=lr, glow_radius_mult=8.0, glow_strength=0.66, hanging=True)

    img = draw_umbrella_figure(img, 0.21, 0.76, height_frac=0.28, rim_color=(0.98, 0.78, 0.44))

    ripples = [(0.18, 0.86, local_t + 0.2), (0.40, 0.82, local_t + 1.8), (0.64, 0.89, local_t + 3.4)]
    img = apply_puddle_reflection(img, y_start_frac=0.755, time_s=global_t, ripple_centers=ripples, darkness=0.78)

    warm_lights = [
        (0.11, 0.27, 0.16, (1.0, 0.78, 0.42)),
        (0.33, 0.30, 0.15, (1.0, 0.78, 0.42)),
        (0.52, 0.32, 0.14, (1.0, 0.78, 0.42)),
        (front_x - 0.12, 0.62, 0.26, (1.0, 0.90, 0.66)),
    ]
    img = render_slanted_rain(
        img,
        time_s=global_t,
        density=1.25,
        angle_deg=14.0,
        warm_lights=warm_lights,
        seed=10,
        occlusion_mask=cache["roof_occ"],
    )
    return _finalize(img, grain, vig_strength=0.28, bloom_thresh=0.64, bloom_tight=0.48, bloom_wide=0.34)
