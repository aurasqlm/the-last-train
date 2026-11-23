"""Act VI — THE CLOUDBREAK & GOLDEN HOUR (Shots S28–S33, 03:10.0–03:55.0).

Palette: Radiant Apricot #F49D6E, Rose Gold #E77D8B, Cerulean Sky #5C8FD6, Sunlight #FFF1D0.
Each shot has its own distinct camera angle, composition, framing, lighting,
and purposeful 60fps motion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..painterly.clouds import render_cumulus_cathedral, render_sky_gradient
from ..painterly.ink import draw_petals_and_motes, draw_train_profile
from ..painterly.optics import (
    apply_atmospheric_haze,
    apply_filmic_grade,
    apply_halation_bloom,
    render_aperture_bokeh,
    render_god_rays,
)
from ..painterly.paper_grain import generate_paper_grain
from ..painterly.weather import render_sea_wake


def _over(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(bg * (1.0 - a) + fg * a, 0.0, 1.0)


def _make_vignette_mask(h: int, w: int, strength: float = 0.20) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    return np.clip(1.0 - strength * np.clip(r2 - 0.35, 0.0, 2.0) * 0.65, 0.58, 1.0).astype(np.float32)


def _fast_finish(
    img: np.ndarray,
    grain: np.ndarray,
    vig: np.ndarray,
    grain_strength: float = 0.028,
    bloom_thresh: float = 0.68,
    bloom_tight: float = 0.42,
    bloom_wide: float = 0.30,
) -> np.ndarray:
    h, w = img.shape[:2]
    qw, qh = max(32, w // 4), max(18, h // 4)
    small = cv2.resize(img, (qw, qh), interpolation=cv2.INTER_AREA)
    hi = np.clip((small - bloom_thresh) / max(1.0 - bloom_thresh, 1e-4), 0.0, 1.0)
    b1 = cv2.GaussianBlur(hi, (0, 0), sigmaX=max(2.0, qw / 90.0))
    b2 = cv2.GaussianBlur(hi, (0, 0), sigmaX=max(5.0, qw / 32.0))
    bloom = cv2.resize(b1 * bloom_tight + b2 * bloom_wide, (w, h), interpolation=cv2.INTER_LINEAR)
    img = 1.0 - (1.0 - img) * (1.0 - np.clip(bloom, 0.0, 0.85))
    img = np.clip((img - 0.5) * 1.06 + 0.5 + (grain[..., None] - 0.5) * grain_strength, 0.0, 1.0)
    img *= vig[..., None]
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def _fast_golden_water(
    img: np.ndarray,
    horizon_y_frac: float,
    time_s: float,
    shimmer_x_frac: float = 0.48,
    shimmer_strength: float = 0.85,
) -> np.ndarray:
    """Fast vectorized golden-hour mirror-sea reflection with sparkling shimmer path."""
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
    depth = 0.14 + 0.86 * yy
    scale_x = w / 960.0
    dx_half = (
        np.sin(yy[:, None] * 52.0 - time_s * 2.6 + xx[None, :] * 0.028) * 0.6
        + np.sin(yy[:, None] * 118.0 + time_s * 4.1) * 0.4
    ) * (2.3 * scale_x * depth[:, None])
    dx = cv2.resize(dx_half, (w, water_h), interpolation=cv2.INTER_LINEAR)

    map_x = np.clip(np.arange(w, dtype=np.float32)[None, :] + dx, 0, w - 1).astype(np.float32)
    map_y = np.broadcast_to(np.arange(water_h, dtype=np.float32)[:, None], (water_h, w)).copy()
    warped = cv2.remap(ref, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    nx = (xx / max(hw - 1, 1)) - shimmer_x_frac
    cone_w = 0.05 + 0.22 * yy[:, None]
    path = np.exp(-((nx[None, :] / cone_w) ** 2))
    wave = 0.5 + 0.5 * np.sin(yy[:, None] * 85.0 - time_s * 5.0 + nx[None, :] * 38.0)
    sparkle_half = np.power(np.clip(wave, 0.0, 1.0), 4.0) * path * shimmer_strength
    sparkle = cv2.resize(sparkle_half, (w, water_h), interpolation=cv2.INTER_LINEAR)
    shimmer_rgb = np.array([1.00, 0.92, 0.68], dtype=np.float32)[None, None, :]

    yy_full = np.linspace(0.0, 1.0, water_h, dtype=np.float32)
    fade = (0.88 - 0.10 * yy_full)[:, None, None]
    out[hy:] = np.clip(warped * fade + sparkle[..., None] * shimmer_rgb * 0.72, 0.0, 1.0)
    return out


def prepare_shot_cache(shot_spec: Any, width: int = 1920, height: int = 1080) -> Dict[str, Any]:
    """Pre-render static golden-hour skies, cloud cathedrals, god-rays, and paper grain."""
    sid = shot_spec.id
    seed = 600 + int(sid[1:])
    grain = generate_paper_grain(height, width, seed=seed)
    vig = _make_vignette_mask(height, width, strength=0.20)
    cache: Dict[str, Any] = {"id": sid, "w": width, "h": height, "grain": grain, "vig": vig}

    if sid == "S28":
        # Macro / Close-Up Exterior: rain tapering to single sparkling drops in warm golden-apricot light
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.20, 0.42, 0.74)),
                (0.38, (0.58, 0.56, 0.74)),
                (0.72, (0.94, 0.62, 0.46)),
                (1.00, (0.98, 0.84, 0.62)),
            ],
            seed=28,
            brush_strength=0.024,
        )
        bokeh = render_aperture_bokeh(
            width, height,
            palette=[
                (1.00, 0.94, 0.76),
                (0.98, 0.72, 0.52),
                (0.92, 0.56, 0.60),
                (0.60, 0.80, 0.98),
            ],
            num_discs=42,
            y_range=(0.08, 0.92),
            radius_range=(0.022, 0.075),
            seed=28,
        )
        bg = 1.0 - (1.0 - bg) * (1.0 - np.clip(bokeh * 0.55, 0.0, 0.85))
        cache["bg"] = cv2.GaussianBlur(bg, (0, 0), sigmaX=max(1.5, width / 400.0))

    elif sid == "S29":
        # Wide Skyward Establishing: colossal storm clouds tearing open
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.12, 0.34, 0.72)),
                (0.32, (0.34, 0.56, 0.84)),
                (0.65, (0.92, 0.56, 0.54)),
                (1.00, (1.00, 0.86, 0.62)),
            ],
            seed=29,
            brush_strength=0.026,
        )
        left_rgb, left_a = render_cumulus_cathedral(
            width, height,
            towers=[(0.12, 0.18, 0.28, 0.48), (0.34, 0.26, 0.24, 0.42)],
            base_y_frac=0.92,
            color_ramp=[
                (0.16, 0.18, 0.34),
                (0.46, 0.34, 0.48),
                (0.92, 0.58, 0.48),
                (1.00, 0.92, 0.74),
            ],
            sun_pos=(0.50, 0.44),
            rim_color=(1.00, 0.94, 0.76),
            rim_strength=0.90,
            seed=291,
        )
        right_rgb, right_a = render_cumulus_cathedral(
            width, height,
            towers=[(0.66, 0.24, 0.24, 0.44), (0.88, 0.16, 0.28, 0.50)],
            base_y_frac=0.92,
            color_ramp=[
                (0.16, 0.18, 0.34),
                (0.46, 0.34, 0.48),
                (0.92, 0.58, 0.48),
                (1.00, 0.92, 0.74),
            ],
            sun_pos=(0.50, 0.44),
            rim_color=(1.00, 0.94, 0.76),
            rim_strength=0.90,
            seed=292,
        )
        cache["sky"] = sky
        cache["left_rgb"] = left_rgb
        cache["left_a"] = left_a
        cache["right_rgb"] = right_rgb
        cache["right_a"] = right_a

    elif sid == "S30":
        # Wide Horizon Shot: sweeping golden god-rays piercing cloud gaps onto sea tracks
        horizon_y = 0.62
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.14, 0.32, 0.66)),
                (0.28, (0.34, 0.50, 0.76)),
                (0.48, (0.88, 0.54, 0.50)),
                (0.62, (1.00, 0.84, 0.58)),
                (1.00, (0.24, 0.34, 0.52)),
            ],
            seed=30,
            brush_strength=0.026,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[
                (0.12, 0.16, 0.24, 0.32),
                (0.32, 0.24, 0.18, 0.24),
                (0.76, 0.18, 0.26, 0.34),
                (0.92, 0.28, 0.18, 0.22),
            ],
            base_y_frac=horizon_y - 0.05,
            color_ramp=[
                (0.20, 0.20, 0.38),
                (0.56, 0.38, 0.48),
                (0.95, 0.64, 0.44),
                (1.00, 0.94, 0.74),
            ],
            sun_pos=(0.52, 0.24),
            rim_color=(1.00, 0.95, 0.78),
            rim_strength=0.88,
            seed=300,
        )
        bg = _over(sky, c_rgb, c_a)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.98, 0.76, 0.52), strength=0.30, band_height=0.08)
        rays = render_god_rays(
            width, height,
            origin_xy=(0.52, 0.16),
            color=(1.00, 0.90, 0.58),
            num_rays=5,
            intensity=0.88,
            spread_deg=(30.0, 150.0),
            seed=30,
            cloud_alpha=c_a,
        )
        cache["bg"] = bg
        cache["rays"] = rays
        cache["horizon_y"] = horizon_y

    elif sid == "S31":
        # Interior Medium Shot: warm apricot sunlight flooding wooden carriage & silhouette
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.18, 0.12, 0.11)),
                (0.40, (0.34, 0.22, 0.16)),
                (0.75, (0.48, 0.30, 0.20)),
                (1.00, (0.20, 0.13, 0.10)),
            ],
            seed=31,
            brush_strength=0.022,
        )
        cache["bg"] = bg

    elif sid == "S32":
        # Macro Water Surface: golden-apricot & rose-gold bokeh shimmer on sea ripples
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.98, 0.76, 0.54)),
                (0.32, (0.92, 0.54, 0.54)),
                (0.68, (0.58, 0.40, 0.56)),
                (1.00, (0.22, 0.26, 0.44)),
            ],
            seed=32,
            brush_strength=0.024,
        )
        bokeh = render_aperture_bokeh(
            width, height,
            palette=[
                (1.00, 0.95, 0.76),
                (0.98, 0.68, 0.46),
                (0.94, 0.52, 0.58),
            ],
            num_discs=55,
            y_range=(0.05, 0.95),
            radius_range=(0.012, 0.044),
            seed=32,
        )
        cache["bg"] = bg
        cache["bokeh"] = bokeh

    elif sid == "S33":
        # Euphoric Wide Tracking (Hero): golden-hour sea, god-rays, backlit train & sakura petals
        horizon_y = 0.65
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.14, 0.34, 0.70)),
                (0.24, (0.30, 0.50, 0.78)),
                (0.44, (0.82, 0.48, 0.52)),
                (0.56, (0.94, 0.62, 0.40)),
                (0.65, (1.00, 0.86, 0.64)),
                (1.00, (0.28, 0.38, 0.56)),
            ],
            seed=33,
            brush_strength=0.028,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[
                (0.08, 0.24, 0.22, 0.34),
                (0.24, 0.34, 0.16, 0.24),
                (0.88, 0.20, 0.24, 0.36),
                (0.72, 0.32, 0.17, 0.24),
            ],
            base_y_frac=horizon_y - 0.04,
            color_ramp=[
                (0.22, 0.20, 0.40),
                (0.58, 0.38, 0.50),
                (0.94, 0.62, 0.44),
                (1.00, 0.92, 0.72),
            ],
            sun_pos=(0.48, 0.34),
            rim_color=(1.00, 0.95, 0.78),
            rim_strength=0.88,
            seed=330,
        )
        bg = _over(sky, c_rgb, c_a)
        rays = render_god_rays(
            width, height,
            origin_xy=(0.48, 0.20),
            color=(1.00, 0.88, 0.54),
            num_rays=4,
            intensity=0.84,
            spread_deg=(36.0, 144.0),
            seed=33,
            cloud_alpha=c_a,
        )
        bg = 1.0 - (1.0 - bg) * (1.0 - rays)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.98, 0.78, 0.54), strength=0.30, band_height=0.075)
        cache["bg"] = bg
        cache["rays"] = rays
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
    """Render one 60fps frame for Act VI (S28–S33) as uint8 RGB (height, width, 3)."""
    if cache is None or cache.get("w") != width or cache.get("h") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(shot_spec.duration_s, 1e-3)
    prog = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]
    vig = cache["vig"]

    if sid == "S28":
        # Macro / Close-Up Exterior: rain tapering to single sparkling golden droplets
        img = cache["bg"].copy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        rng = np.random.default_rng(280)
        # Number of visible drops tapers from 24 down to 6 over the shot
        num_drops = max(6, int(24 * (1.0 - 0.72 * prog)))
        for i in range(num_drops):
            bx = rng.uniform(0.08, 0.92)
            by0 = rng.uniform(0.0, 1.0)
            spd = rng.uniform(0.14, 0.28)
            rad = int(rng.uniform(3.5, 11.0) * width / 960.0)
            dy = (by0 + local_t * spd) % 1.15 - 0.07
            dx = bx - local_t * 0.018
            px, py = int((dx % 1.0) * width), int(dy * height)
            if 0 <= px < width and 0 <= py < height:
                # Sparkling golden-apricot droplet with specular highlight
                cv2.ellipse(u8, (px, py), (max(2, int(rad * 0.62)), max(3, rad)), 6, 0, 360, (255, 224, 178), -1, cv2.LINE_AA)
                cv2.circle(u8, (px - max(1, rad // 4), py - max(1, rad // 3)), max(1, rad // 3), (255, 250, 232), -1, cv2.LINE_AA)
                # Subtle golden trail
                cv2.line(u8, (px, py - rad), (px + max(1, rad // 2), py - rad * 3), (255, 215, 155), max(1, rad // 4), cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        return _fast_finish(img, grain, vig, bloom_thresh=0.64, bloom_tight=0.48, bloom_wide=0.34)

    elif sid == "S29":
        # Wide Skyward Establishing: colossal storm clouds physically tearing open
        img = cache["sky"].copy()
        shift_px = int(width * 0.14 * prog)
        l_rgb = np.roll(cache["left_rgb"], -shift_px, axis=1)
        l_a = np.roll(cache["left_a"], -shift_px, axis=1)
        if shift_px > 0:
            l_a[:, -shift_px:] = 0.0
        r_rgb = np.roll(cache["right_rgb"], shift_px, axis=1)
        r_a = np.roll(cache["right_a"], shift_px, axis=1)
        if shift_px > 0:
            r_a[:, :shift_px] = 0.0
        img = _over(img, l_rgb, l_a)
        img = _over(img, r_rgb, r_a)
        # Expanding radiant golden light bloom in the central tear
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx, cy = width * 0.50, height * (0.46 - 0.06 * prog)
        r2 = ((xx - cx) / (width * (0.18 + 0.16 * prog))) ** 2 + ((yy - cy) / (height * (0.22 + 0.16 * prog))) ** 2
        glow = np.exp(-r2) * (0.42 + 0.35 * prog)
        img = np.clip(img + glow[..., None] * np.array([1.00, 0.90, 0.66], dtype=np.float32)[None, None, :], 0.0, 1.0)
        return _fast_finish(img, grain, vig, bloom_thresh=0.66, bloom_tight=0.46, bloom_wide=0.34)

    elif sid == "S30":
        # Wide Horizon Shot: sweeping volumetric golden god-rays piercing cloud gaps onto sea tracks
        hy = cache["horizon_y"]
        img = cache["bg"].copy()
        # Gently shifting god-ray shafts
        ray_shift = int(width * 0.05 * np.sin(prog * np.pi))
        rays = np.roll(cache["rays"], ray_shift, axis=1)
        pulse = 0.85 + 0.15 * np.sin(local_t * 1.1)
        img = 1.0 - (1.0 - img) * (1.0 - np.clip(rays * pulse, 0.0, 0.95))
        # Distant silhouetted train crossing beneath the god-rays
        tx0 = 0.18 + 0.24 * prog
        img = draw_train_profile(
            img,
            x_start_frac=tx0,
            x_end_frac=tx0 + 0.26,
            y_bottom_frac=hy + 0.002,
            height_frac=0.040,
            body_color=(0.12, 0.10, 0.18),
            stripe_color=(0.48, 0.22, 0.20),
            window_color=(1.00, 0.92, 0.66),
            headlight_left=False,
            num_cars=3,
            rim_light_color=(1.00, 0.92, 0.60),
            rim_light_strength=0.95,
        )
        img = _fast_golden_water(img, hy, local_t, shimmer_x_frac=0.52 + 0.04 * np.sin(prog * np.pi), shimmer_strength=0.90)
        img = 1.0 - (1.0 - img) * (1.0 - rays * 0.38)
        return _fast_finish(img, grain, vig, bloom_thresh=0.66, bloom_tight=0.44, bloom_wide=0.32)

    elif sid == "S31":
        # Interior Medium Shot: warm apricot sunlight flooding wooden carriage & silhouette
        img = cache["bg"].copy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        # Tall golden-lit carriage windows along the left/center wall
        dolly = int(width * 0.04 * prog)
        for i in range(3):
            wx0 = int(width * (0.08 + i * 0.28)) - dolly
            wx1 = wx0 + int(width * 0.22)
            wy0, wy1 = int(height * 0.12), int(height * 0.56)
            cv2.rectangle(u8, (wx0, wy0), (wx1, wy1), (255, 224, 168), -1)
            cv2.rectangle(u8, (wx0, wy0), (wx1, wy1), (62, 40, 28), max(3, int(width / 240)), cv2.LINE_AA)
        # Wooden bench seat & silhouette bathed in gold
        cv2.rectangle(u8, (0, int(height * 0.68)), (width, height), (54, 34, 24), -1)
        # Seated silhouette on right-center catching warm golden rim-light
        cx = int(width * (0.72 - 0.02 * prog))
        cv2.ellipse(u8, (cx, int(height * 0.44)), (int(width * 0.048), int(height * 0.095)), 0, 0, 360, (44, 28, 22), -1, cv2.LINE_AA)
        cv2.ellipse(u8, (cx + int(width * 0.01), int(height * 0.72)), (int(width * 0.095), int(height * 0.20)), 0, 0, 360, (52, 32, 24), -1, cv2.LINE_AA)
        # Golden rim-light on silhouette's left edge
        cv2.ellipse(u8, (cx, int(height * 0.44)), (int(width * 0.048), int(height * 0.095)), 0, 120, 245, (255, 220, 148), max(2, int(width / 320)), cv2.LINE_AA)
        # Hanging straps swaying gently
        for s_i in range(4):
            sx = int(width * (0.16 + s_i * 0.20)) - dolly
            sway = int(width * 0.012 * np.sin(local_t * 1.8 + s_i * 0.6))
            cv2.line(u8, (sx, 0), (sx + sway, int(height * 0.20)), (110, 72, 46), max(2, int(width / 480)), cv2.LINE_AA)
            cv2.circle(u8, (sx + sway, int(height * 0.23)), max(5, int(height * 0.026)), (210, 160, 102), max(2, int(width / 480)), cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        # Sweeping diagonal apricot sunbeams flooding across carriage
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        beam_pos = (xx / max(width, 1)) - (yy / max(height, 1)) * 0.55 - 0.10 * prog
        beams = (
            np.exp(-((beam_pos - 0.05) / 0.11) ** 2)
            + np.exp(-((beam_pos - 0.34) / 0.11) ** 2)
            + np.exp(-((beam_pos - 0.62) / 0.11) ** 2)
        )
        img = np.clip(img + beams[..., None] * np.array([0.46, 0.30, 0.14], dtype=np.float32)[None, None, :], 0.0, 1.0)
        # Glowing golden dust motes swirling in sunbeams
        img = draw_petals_and_motes(
            img, time_s=local_t, num_petals=0, num_motes=65, mote_color=(1.00, 0.94, 0.72), seed=31
        )
        return _fast_finish(img, grain, vig, bloom_thresh=0.64, bloom_tight=0.48, bloom_wide=0.34)

    elif sid == "S32":
        # Macro Water Surface: golden-apricot & rose-gold bokeh shimmer on silky sea ripples
        bg = cache["bg"].copy()
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        xx = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        # Silky animated sea wavelets
        w1 = np.sin(yy * 42.0 - local_t * 2.8 + xx * 10.0)
        w2 = np.sin(yy * 78.0 + local_t * 3.6 - xx * 16.0)
        crest = np.power(np.clip(0.5 + 0.3 * w1 + 0.2 * w2, 0.0, 1.0), 3.0)
        img = np.clip(bg * (0.78 + 0.32 * crest[..., None]), 0.0, 1.0)
        # Shifting sparkling bokeh shimmer
        shift_x = int(local_t * width * 0.03)
        bokeh = np.roll(cache["bokeh"], shift_x, axis=1)
        twinkle = 0.72 + 0.28 * np.sin(local_t * 4.2 + xx * 20.0)
        img = 1.0 - (1.0 - img) * (1.0 - np.clip(bokeh * twinkle[..., None] * 0.85, 0.0, 0.95))
        return _fast_finish(img, grain, vig, bloom_thresh=0.62, bloom_tight=0.48, bloom_wide=0.36)

    else:
        # S33: Euphoric Wide Tracking (Hero) — golden hour sea, backlit train & 3-plane sakura petals
        hy = cache["horizon_y"]
        img = cache["bg"].copy()
        # Train gliding smoothly rightward across golden-hour sea
        tx0 = 0.44 + 0.12 * prog
        tx1 = tx0 + 0.40
        img = draw_train_profile(
            img,
            x_start_frac=tx0,
            x_end_frac=tx1,
            y_bottom_frac=hy + 0.003,
            height_frac=0.058,
            body_color=(0.11, 0.09, 0.16),
            stripe_color=(0.48, 0.22, 0.20),
            window_color=(1.00, 0.92, 0.66),
            headlight_left=False,
            num_cars=3,
            rim_light_color=(1.00, 0.90, 0.56),
            rim_light_strength=1.00,
        )
        img = _fast_golden_water(img, hy, local_t, shimmer_x_frac=0.48, shimmer_strength=0.92)
        img = 1.0 - (1.0 - img) * (1.0 - cache["rays"] * 0.36)
        img = render_sea_wake(
            img,
            bow_xy=(min(0.99, tx1), hy + 0.008),
            tail_xy=(max(0.02, tx0 - 0.08), hy + 0.013),
            wake_width_frac=0.042,
            foam_color=(1.00, 0.93, 0.76),
            opacity=0.82,
            seed=33,
        )
        img = draw_petals_and_motes(
            img,
            time_s=local_t,
            num_petals=56,
            num_motes=68,
            petal_color=(1.00, 0.78, 0.83),
            mote_color=(1.00, 0.95, 0.74),
            seed=33,
        )
        return _fast_finish(img, grain, vig, bloom_thresh=0.68, bloom_tight=0.44, bloom_wide=0.32)
