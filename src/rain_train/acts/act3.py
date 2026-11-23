"""Act III — THE WINDOW SEAT (Shots S11–S16, 01:10.0–01:50.0).

Palette: Warm Tungsten Cabin (#281D19 / #F2C57C) vs. Indigo Night (#142036).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..painterly.clouds import render_sky_gradient
from ..painterly.ink import draw_lantern
from ..painterly.optics import (
    apply_filmic_grade,
    apply_halation_bloom,
    apply_vignette,
    render_aperture_bokeh,
)
from ..painterly.paper_grain import (
    apply_paper_grain,
    apply_pigment_granulation,
    generate_paper_grain,
)
from ..painterly.weather import (
    render_macro_window_droplets,
    render_slanted_rain,
)
from ..storyboard import ShotSpec


def _catmull_rom(
    ctrl: List[Tuple[float, float]],
    w: int,
    h: int,
    closed: bool = False,
    samples_per_seg: int = 10,
) -> np.ndarray:
    pts = np.array([(p[0] * w, p[1] * h) for p in ctrl], dtype=np.float32)
    n = len(pts)
    if n < 3:
        return pts.astype(np.int32)
    if closed:
        ext = np.vstack([pts[-1:], pts, pts[:2]])
        seg_count = n
    else:
        ext = np.vstack([pts[0:1], pts, pts[-1:]])
        seg_count = n - 1
    t_vals = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False, dtype=np.float32)[:, None]
    t2 = t_vals * t_vals
    t3 = t2 * t_vals
    out_pts = []
    for i in range(seg_count):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        seg = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t_vals
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
        )
        out_pts.append(seg)
    if not closed:
        out_pts.append(pts[-1:])
    return np.vstack(out_pts).astype(np.int32)


def _render_passenger_reflection_layer(w: int, h: int) -> np.ndarray:
    """Pre-render graceful Catmull-Rom seated anime passenger reflection layer (float32 RGB)."""
    sil_u8 = np.zeros((h, w, 3), dtype=np.uint8)
    # Hanging cabin hand-strap reflected in upper-center
    cv2.line(sil_u8, (int(w * 0.42), 0), (int(w * 0.43), int(h * 0.20)), (195, 148, 92), max(2, w // 320), cv2.LINE_AA)
    cv2.circle(sil_u8, (int(w * 0.43), int(h * 0.24)), max(6, int(h * 0.036)), (225, 178, 118), max(2, w // 320), cv2.LINE_AA)

    face_ctrl = [
        (0.662, 0.305), (0.644, 0.352), (0.638, 0.388), (0.624, 0.428),
        (0.638, 0.448), (0.635, 0.468), (0.637, 0.488), (0.646, 0.526),
        (0.688, 0.556), (0.742, 0.528), (0.768, 0.435), (0.748, 0.312),
    ]
    cv2.fillPoly(sil_u8, [_catmull_rom(face_ctrl, w, h, closed=True)], (228, 178, 124), cv2.LINE_AA)
    cv2.ellipse(sil_u8, (int(w * 0.658), int(h * 0.402)), (max(3, int(w * 0.013)), max(2, int(h * 0.007))), -10, 180, 360, (68, 42, 28), max(1, h // 300), cv2.LINE_AA)
    cv2.circle(sil_u8, (int(w * 0.654), int(h * 0.409)), max(2, int(h * 0.008)), (78, 52, 38), -1, cv2.LINE_AA)

    neck_ctrl = [(0.690, 0.540), (0.686, 0.595), (0.682, 0.642), (0.752, 0.638), (0.750, 0.585), (0.744, 0.530)]
    cv2.fillPoly(sil_u8, [_catmull_rom(neck_ctrl, w, h, closed=True)], (202, 152, 102), cv2.LINE_AA)

    sh_ctrl = [
        (0.515, 1.000), (0.555, 0.780), (0.605, 0.665), (0.685, 0.618),
        (0.785, 0.612), (0.865, 0.665), (0.920, 0.815), (0.950, 1.000),
    ]
    cv2.fillPoly(sil_u8, [_catmull_rom(sh_ctrl, w, h, closed=True)], (122, 82, 52), cv2.LINE_AA)

    hair_ctrl = [
        (0.635, 0.345), (0.652, 0.285), (0.705, 0.242), (0.772, 0.248),
        (0.822, 0.305), (0.848, 0.415), (0.862, 0.555), (0.835, 0.645),
        (0.785, 0.658), (0.745, 0.600), (0.748, 0.485), (0.732, 0.382), (0.692, 0.325),
    ]
    cv2.fillPoly(sil_u8, [_catmull_rom(hair_ctrl, w, h, closed=True)], (96, 62, 42), cv2.LINE_AA)

    hand_ctrl = [(0.602, 0.645), (0.618, 0.578), (0.636, 0.538), (0.652, 0.555), (0.650, 0.612), (0.628, 0.672)]
    cv2.fillPoly(sil_u8, [_catmull_rom(hand_ctrl, w, h, closed=True)], (218, 168, 118), cv2.LINE_AA)

    rim_spline = _catmull_rom(
        [(0.665, 0.268), (0.718, 0.242), (0.778, 0.252), (0.826, 0.312), (0.852, 0.425), (0.860, 0.562)],
        w, h, closed=False,
    )
    cv2.polylines(sil_u8, [rim_spline], False, (252, 212, 148), max(2, h // 260), cv2.LINE_AA)
    return cv2.GaussianBlur(sil_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=max(1.2, w / 640.0))


def _finalize(
    img: np.ndarray,
    grain: np.ndarray,
    vig_strength: float = 0.28,
    bloom_thresh: float = 0.62,
    bloom_tight: float = 0.45,
    bloom_wide: float = 0.30,
) -> np.ndarray:
    h, w = img.shape[:2]
    bw, bh = max(8, w // 4), max(8, h // 4)
    small = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
    bright = np.maximum(small - bloom_thresh, 0.0)
    tight = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(1.5, bw / 160.0))
    wide = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(4.0, bw / 48.0))
    bloom = cv2.resize(tight * bloom_tight + wide * bloom_wide, (w, h), interpolation=cv2.INTER_LINEAR)
    img = np.clip(img + bloom, 0.0, 1.0)
    g = (grain[..., None] - 0.5) * 0.052
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
    """Pre-render static painterly layers for Act III shots (S11–S16)."""
    sid = shot_spec.id
    grain = generate_paper_grain(height, width, seed=int(sid[1:]) * 23)
    cache: Dict[str, Any] = {"id": sid, "width": width, "height": height, "grain": grain}

    bokeh_pal = [
        (1.00, 0.80, 0.44),
        (0.98, 0.60, 0.28),
        (0.94, 0.34, 0.28),
        (0.58, 0.78, 0.94),
        (1.00, 0.93, 0.74),
    ]

    if sid == "S11":
        # Interior wide one-point perspective vintage wooden carriage
        u8 = np.zeros((height, width, 3), dtype=np.uint8)
        u8[:, :] = (36, 26, 22)  # Deep warm cedar wood
        vp_x, vp_y = width // 2, int(height * 0.47)

        # Ceiling, floor, left & right walls
        cv2.fillPoly(u8, [np.array([[0, 0], [width, 0], [int(width * 0.62), int(height * 0.34)], [int(width * 0.38), int(height * 0.34)]], dtype=np.int32)], (52, 38, 30), cv2.LINE_AA)
        cv2.fillPoly(u8, [np.array([[0, height], [width, height], [int(width * 0.62), int(height * 0.64)], [int(width * 0.38), int(height * 0.64)]], dtype=np.int32)], (42, 30, 24), cv2.LINE_AA)
        # End door with window
        cv2.rectangle(u8, (int(width * 0.38), int(height * 0.34)), (int(width * 0.62), int(height * 0.64)), (32, 22, 18), -1)
        cv2.rectangle(u8, (int(width * 0.46), int(height * 0.38)), (int(width * 0.54), int(height * 0.54)), (22, 34, 56), -1)

        # Side windows (indigo night outside) + velvet bench seats
        for side in (-1, 1):
            for k in range(3):
                t0 = 0.08 + k * 0.28
                t1 = t0 + 0.20
                x0 = int(vp_x + side * width * (0.13 + 0.37 * t0))
                x1 = int(vp_x + side * width * (0.13 + 0.37 * t1))
                y_top0 = int(height * (0.37 - 0.16 * t0))
                y_top1 = int(height * (0.37 - 0.16 * t1))
                y_bot0 = int(height * (0.52 + 0.06 * t0))
                y_bot1 = int(height * (0.52 + 0.06 * t1))
                win_poly = np.array([[x0, y_top0], [x1, y_top1], [x1, y_bot1], [x0, y_bot0]], dtype=np.int32)
                cv2.fillPoly(u8, [win_poly], (20, 32, 54), cv2.LINE_AA)
                cv2.polylines(u8, [win_poly], True, (72, 50, 38), max(2, width // 360), cv2.LINE_AA)
            # Plush crimson-brown bench seat
            sx_near = 0 if side == -1 else width
            sx_far = int(vp_x + side * width * 0.13)
            seat_poly = np.array([
                [sx_near, int(height * 0.66)],
                [sx_far, int(height * 0.55)],
                [sx_far, int(height * 0.63)],
                [sx_near, int(height * 0.88)],
            ], dtype=np.int32)
            cv2.fillPoly(u8, [seat_poly], (88, 36, 32), cv2.LINE_AA)

        bg = u8.astype(np.float32) / 255.0
        # Glowing tungsten ceiling globes receding down the aisle
        for gy, gr in [(0.12, 0.028), (0.24, 0.019), (0.31, 0.013)]:
            bg = draw_lantern(bg, 0.50, gy, radius_frac=gr, glow_radius_mult=7.5, glow_strength=0.68, hanging=False)
        cache["carriage"] = bg

    elif sid == "S12":
        # Medium close-up upward: warm cedar ceiling & overhead rail for swaying hand-straps
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.24, 0.17, 0.13)),
                (0.45, (0.16, 0.11, 0.09)),
                (1.00, (0.09, 0.12, 0.20)),
            ],
            seed=312,
            brush_strength=0.028,
        )
        bg = draw_lantern(bg, 0.78, 0.16, radius_frac=0.036, glow_radius_mult=9.0, glow_strength=0.72, hanging=False)
        cache["bg"] = bg

    elif sid == "S13":
        # Medium close-up OTS: seated silhouette by fogged window
        cw = int(width * 1.5)
        sky = render_sky_gradient(
            cw, height,
            stops=[
                (0.00, (0.04, 0.07, 0.14)),
                (0.55, (0.08, 0.13, 0.23)),
                (1.00, (0.05, 0.08, 0.14)),
            ],
            seed=313,
            brush_strength=0.024,
        )
        bokeh = render_aperture_bokeh(cw, height, bokeh_pal, num_discs=65, y_range=(0.40, 0.88), radius_range=(0.012, 0.042), seed=313)
        cache["ext_strip"] = np.clip(1.0 - (1.0 - sky) * (1.0 - bokeh * 0.85), 0.0, 1.0)
        cache["passenger"] = _render_passenger_reflection_layer(width, height)

    elif sid == "S14":
        # Hero S14: Macro window glass droplets + refracted town bokeh + passenger reflection
        cw = int(width * 1.6)
        sky = render_sky_gradient(
            cw, height,
            stops=[
                (0.00, (0.04, 0.07, 0.14)),
                (0.42, (0.07, 0.12, 0.22)),
                (0.70, (0.11, 0.16, 0.27)),
                (1.00, (0.05, 0.07, 0.12)),
            ],
            seed=14,
            brush_strength=0.024,
        )
        bokeh = render_aperture_bokeh(cw, height, bokeh_pal, num_discs=95, y_range=(0.44, 0.94), radius_range=(0.014, 0.050), seed=14)
        ext = 1.0 - (1.0 - sky) * (1.0 - np.clip(bokeh * 0.90, 0.0, 0.96))
        cache["ext_strip"] = cv2.GaussianBlur(ext, (0, 0), sigmaX=max(2.0, width / 320.0))
        cache["passenger"] = _render_passenger_reflection_layer(width, height)

    elif sid == "S15":
        # Interior-to-Exterior POV: fast 60fps parallax glide passing rainy town bokeh & utility poles
        cw = int(width * 2.0)
        sky = render_sky_gradient(
            cw, height,
            stops=[
                (0.00, (0.04, 0.07, 0.15)),
                (0.45, (0.08, 0.14, 0.26)),
                (0.78, (0.12, 0.16, 0.24)),
                (1.00, (0.05, 0.07, 0.12)),
            ],
            seed=315,
            brush_strength=0.025,
        )
        bokeh = render_aperture_bokeh(cw, height, bokeh_pal, num_discs=120, y_range=(0.48, 0.92), radius_range=(0.010, 0.044), seed=315)
        cache["town_strip"] = np.clip(1.0 - (1.0 - sky) * (1.0 - bokeh * 0.92), 0.0, 1.0)

    elif sid == "S16":
        # Close-up glass reflection: warm translucent passenger reflection as town lights fade to forest
        cw = int(width * 1.5)
        sky = render_sky_gradient(
            cw, height,
            stops=[
                (0.00, (0.04, 0.07, 0.13)),
                (0.55, (0.07, 0.12, 0.20)),
                (1.00, (0.04, 0.06, 0.10)),
            ],
            seed=316,
            brush_strength=0.024,
        )
        bokeh = render_aperture_bokeh(cw, height, bokeh_pal, num_discs=55, y_range=(0.45, 0.90), radius_range=(0.015, 0.045), seed=316)
        cache["town_strip"] = np.clip(1.0 - (1.0 - sky) * (1.0 - bokeh * 0.80), 0.0, 1.0)
        cache["forest_bg"] = sky
        cache["passenger"] = _render_passenger_reflection_layer(width, height)

    return cache


def render_shot_frame(
    shot_spec: ShotSpec,
    local_t: float,
    global_t: float,
    width: int = 1920,
    height: int = 1080,
    cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Render one 60fps RGB uint8 frame for Act III (S11–S16)."""
    if cache is None or cache.get("width") != width or cache.get("height") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(0.001, shot_spec.duration_s)
    u = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]

    if sid == "S11":
        # Gentle rhythmic train sway inside warm wooden carriage
        sway_x = int(width * 0.008 * np.sin(local_t * 2.2))
        sway_y = int(height * 0.004 * np.sin(local_t * 4.4))
        img = np.roll(np.roll(cache["carriage"], sway_x, axis=1), sway_y, axis=0)
        return _finalize(img, grain, vig_strength=0.30, bloom_thresh=0.60, bloom_tight=0.48, bloom_wide=0.34)

    if sid == "S12":
        # Row of leather/ring hanging hand-straps swaying in smooth 60fps damped pendulum motion
        img = cache["bg"].copy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        # Overhead metal bar receding in perspective
        cv2.line(u8, (0, int(height * 0.16)), (width, int(height * 0.30)), (175, 160, 145), max(4, height // 140), cv2.LINE_AA)
        for i in range(6):
            frac = 0.12 + i * 0.15
            ax = int(width * frac)
            ay = int(height * (0.16 + 0.14 * frac))
            scale = 1.15 - 0.12 * i
            strap_len = int(height * 0.24 * scale)
            ring_r = max(10, int(height * 0.072 * scale))
            ang = 0.22 * np.sin(local_t * 2.8 - i * 0.35) + 0.06 * np.sin(local_t * 5.6 - i * 0.5)
            bx = int(ax + strap_len * np.sin(ang))
            by = int(ay + strap_len * np.cos(ang))
            # Leather strap
            cv2.line(u8, (ax, ay), (bx, by), (96, 62, 40), max(4, int(10 * scale * height / 1080.0)), cv2.LINE_AA)
            # Cream/wood hand ring
            rcx = int(bx + ring_r * np.sin(ang))
            rcy = int(by + ring_r * np.cos(ang))
            cv2.circle(u8, (rcx, rcy), ring_r, (230, 198, 152), max(3, int(9 * scale * height / 1080.0)), cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        return _finalize(img, grain, vig_strength=0.28, bloom_thresh=0.62, bloom_tight=0.45, bloom_wide=0.30)

    if sid == "S13":
        # Seated silhouette leaning chin on hand by fogged window
        strip = cache["ext_strip"]
        cw = strip.shape[1]
        x0 = int((local_t * width * 0.18) % max(1, cw - width))
        bg = strip[:, x0:x0 + width].copy()
        # Add rain streaks outside window
        bg = render_slanted_rain(bg, time_s=global_t, density=0.65, angle_deg=18.0, seed=313)
        # Overlay warm cabin window frame on left + seated passenger silhouette on right
        u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
        cv2.rectangle(u8, (0, 0), (int(width * 0.06), height), (36, 25, 20), -1)
        cv2.rectangle(u8, (0, int(height * 0.84)), (width, height), (42, 29, 22), -1)
        bg = u8.astype(np.float32) / 255.0
        # Composite passenger silhouette at strong foreground presence
        p = cache["passenger"]
        alpha = np.clip(p.max(axis=2) * 1.4, 0.0, 0.92)[..., None]
        img = np.clip(bg * (1.0 - alpha) + p * 0.85 * alpha, 0.0, 1.0)
        return _finalize(img, grain, vig_strength=0.30)

    if sid == "S14":
        # Hero S14: Macro window droplets beading & sliding with refracted town bokeh + passenger reflection
        strip = cache["ext_strip"]
        cw = strip.shape[1]
        x0 = int((local_t * width * 0.22) % max(1, cw - width))
        bg = strip[:, x0:x0 + width].copy()
        # Translucent Fresnel reflection of passenger (~40%)
        img = np.clip(bg * 0.78 + cache["passenger"] * 0.42, 0.0, 1.0)
        # Macro refractive droplets sliding smoothly
        img = render_macro_window_droplets(img, time_s=local_t, num_large=28, num_med=68, num_micro=160, fog_strength=0.14, seed=14)
        return _finalize(img, grain, vig_strength=0.26, bloom_thresh=0.60, bloom_tight=0.46, bloom_wide=0.32)

    if sid == "S15":
        # Interior-to-Exterior POV: fast 60fps parallax glide passing rainy town lights & dark utility poles
        strip = cache["town_strip"]
        cw = strip.shape[1]
        x0 = int((local_t * width * 0.55) % max(1, cw - width))
        img = strip[:, x0:x0 + width].copy()
        # Dark utility poles whipping past foreground at high speed
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        for p_idx in range(3):
            px = int((1.2 - ((local_t * 1.35 + p_idx * 0.38) % 1.4)) * width)
            pw = max(6, int(width * 0.022))
            cv2.rectangle(u8, (px, 0), (px + pw, height), (14, 18, 26), -1)
        # Window sill frame
        cv2.rectangle(u8, (0, int(height * 0.88)), (width, height), (34, 24, 19), -1)
        img = u8.astype(np.float32) / 255.0
        img = render_slanted_rain(img, time_s=global_t, density=0.95, angle_deg=26.0, seed=315)
        return _finalize(img, grain, vig_strength=0.28)

    # S16: Close-up glass reflection — warm passenger reflection overlaid on rainy glass as town fades to forest
    strip = cache["town_strip"]
    cw = strip.shape[1]
    x0 = int((local_t * width * 0.16) % max(1, cw - width))
    town = strip[:, x0:x0 + width]
    # Cross-fade town bokeh out into dark forest silhouettes
    forest = cache["forest_bg"][:, x0:x0 + width]
    bg = town * (1.0 - u * 0.75) + forest * (u * 0.75)
    img = np.clip(bg * 0.68 + cache["passenger"] * (0.58 + 0.12 * u), 0.0, 1.0)
    img = render_slanted_rain(img, time_s=global_t, density=0.60, angle_deg=18.0, seed=316)
    return _finalize(img, grain, vig_strength=0.30, bloom_thresh=0.60, bloom_tight=0.44, bloom_wide=0.30)
