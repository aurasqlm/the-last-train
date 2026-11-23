"""Act V — THE FLOODED SEA TRACKS (Shots S22–S27, 02:25.0–03:10.0).

Palette: Twilight Cobalt #122347, Mirror Silver #8FA9C4, Cream Cloudlight #EEDEC7.
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
from ..painterly.weather import apply_sea_reflection, render_sea_wake


def _over(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(bg * (1.0 - a) + fg * a, 0.0, 1.0)


def _make_vignette_mask(h: int, w: int, strength: float = 0.24) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    return np.clip(1.0 - strength * np.clip(r2 - 0.35, 0.0, 2.0) * 0.65, 0.55, 1.0).astype(np.float32)


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


def _draw_catenary_poles_with_sag(
    img: np.ndarray,
    horizon_y_frac: float,
    pole_color: Tuple[int, int, int],
    wire_color: Tuple[int, int, int],
    num_poles: int = 8,
    pole_h_frac: float = 0.11,
    x_offset_frac: float = 0.0,
) -> np.ndarray:
    h, w = img.shape[:2]
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hy = int(horizon_y_frac * h)
    spacing = 1.0 / max(num_poles, 1)
    pole_xs = []
    for i in range(-1, num_poles + 2):
        fx = (i * spacing + (x_offset_frac % spacing))
        if -0.15 <= fx <= 1.15:
            pole_xs.append(int(fx * w))
    pole_xs.sort()
    pole_top = hy - int(pole_h_frac * h)
    wire_y = pole_top + int(h * 0.015)

    lw = max(1, int(round(w / 960.0)))
    for i in range(len(pole_xs) - 1):
        x0, x1 = pole_xs[i], pole_xs[i + 1]
        pts = []
        for s in np.linspace(0.0, 1.0, 12):
            px = int(x0 * (1.0 - s) + x1 * s)
            sag = int(4.0 * s * (1.0 - s) * (h * 0.016))
            pts.append([px, wire_y + sag])
        cv2.polylines(u8, [np.array(pts, dtype=np.int32)], False, wire_color, lw, cv2.LINE_AA)

    arm = max(4, int(round(7 * w / 960.0)))
    for px in pole_xs:
        if 0 <= px < w:
            cv2.line(u8, (px, hy + 2), (px, pole_top), pole_color, max(1, lw + 1), cv2.LINE_AA)
            cv2.line(u8, (px - arm, pole_top + arm), (px + arm, pole_top + arm), pole_color, lw, cv2.LINE_AA)
            cv2.line(u8, (px, pole_top + arm * 2), (px + arm, pole_top + arm), pole_color, lw, cv2.LINE_AA)

    return u8.astype(np.float32) / 255.0


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
    # Fast quarter-res optical halation bloom (visually identical soft 35mm glow, 16x faster at 1080p)
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


def _fast_water_reflect(
    img: np.ndarray,
    horizon_y_frac: float,
    time_s: float,
    ripple_amp: float = 2.2,
    darkness: float = 0.85,
) -> np.ndarray:
    """Fast vectorized mirror-sea reflection for 60fps compositing."""
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

    # Compute smooth ripple displacement at half-res then upsample (4x faster at 1080p)
    hw, hh = max(32, w // 2), max(16, water_h // 2)
    yy = np.linspace(0.0, 1.0, hh, dtype=np.float32)
    xx = np.arange(hw, dtype=np.float32)
    depth = 0.15 + 0.85 * yy
    scale_x = w / 960.0
    dx_half = (
        np.sin(yy[:, None] * 48.0 - time_s * 2.4 + xx[None, :] * 0.030) * 0.6
        + np.sin(yy[:, None] * 110.0 + time_s * 3.8) * 0.4
    ) * (ripple_amp * scale_x * depth[:, None])
    dx = cv2.resize(dx_half, (w, water_h), interpolation=cv2.INTER_LINEAR)

    map_x = np.clip(np.arange(w, dtype=np.float32)[None, :] + dx, 0, w - 1).astype(np.float32)
    map_y = np.broadcast_to(np.arange(water_h, dtype=np.float32)[:, None], (water_h, w)).copy()
    warped = cv2.remap(ref, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    yy_full = np.linspace(0.0, 1.0, water_h, dtype=np.float32)
    tint = np.array([0.84, 0.90, 1.00], dtype=np.float32)[None, None, :]
    fade = (darkness - 0.14 * yy_full)[:, None, None]
    out[hy:] = np.clip(warped * tint * fade, 0.0, 1.0)
    return out


def prepare_shot_cache(shot_spec: Any, width: int = 1920, height: int = 1080) -> Dict[str, Any]:
    """Pre-render static painterly backgrounds, clouds, and paper grain once per shot."""
    sid = shot_spec.id
    seed = 500 + int(sid[1:])
    grain = generate_paper_grain(height, width, seed=seed)
    vig = _make_vignette_mask(height, width, strength=0.22)
    cache: Dict[str, Any] = {"id": sid, "w": width, "h": height, "grain": grain, "vig": vig}

    if sid == "S22":
        # Wide Exterior Reveal: rails descending into endless shallow mirror-sea
        horizon_y = 0.46
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.06, 0.12, 0.27)),
                (0.22, (0.14, 0.25, 0.44)),
                (0.42, (0.42, 0.55, 0.70)),
                (0.48, (0.86, 0.84, 0.79)),
                (1.00, (0.09, 0.16, 0.30)),
            ],
            seed=22,
            brush_strength=0.026,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[
                (0.18, 0.22, 0.24, 0.24),
                (0.52, 0.16, 0.28, 0.30),
                (0.84, 0.24, 0.22, 0.22),
            ],
            base_y_frac=horizon_y - 0.01,
            color_ramp=[
                (0.18, 0.26, 0.44),
                (0.44, 0.48, 0.64),
                (0.76, 0.76, 0.82),
                (0.98, 0.94, 0.84),
            ],
            sun_dir=(0.5, -0.85),
            rim_color=(0.98, 0.93, 0.82),
            rim_strength=0.68,
            seed=220,
        )
        bg = _over(sky, c_rgb, c_a * 0.90)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.66, 0.76, 0.85), strength=0.32, band_height=0.08)
        cache["bg"] = bg
        cache["horizon_y"] = horizon_y

    elif sid == "S23":
        # Low Water-Level Close-Up: train wheels slicing through shallow water
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.07, 0.13, 0.26)),
                (0.42, (0.18, 0.30, 0.48)),
                (0.58, (0.56, 0.68, 0.78)),
                (1.00, (0.08, 0.15, 0.28)),
            ],
            seed=23,
            brush_strength=0.025,
        )
        cache["bg"] = sky
        cache["horizon_y"] = 0.56

    elif sid == "S24":
        # Monumental Wide (Hero): towering cumulus cloud cathedrals mirrored in sea
        horizon_y = 0.55
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.06, 0.12, 0.26)),
                (0.24, (0.12, 0.22, 0.40)),
                (0.43, (0.34, 0.47, 0.64)),
                (0.55, (0.82, 0.82, 0.80)),
                (1.00, (0.08, 0.14, 0.28)),
            ],
            seed=24,
            brush_strength=0.028,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[
                (0.20, 0.26, 0.18, 0.28),
                (0.36, 0.14, 0.22, 0.40),
                (0.54, 0.28, 0.16, 0.24),
                (0.82, 0.30, 0.17, 0.22),
            ],
            base_y_frac=horizon_y - 0.015,
            color_ramp=[
                (0.16, 0.23, 0.40),
                (0.42, 0.44, 0.60),
                (0.76, 0.74, 0.80),
                (1.00, 0.96, 0.85),
            ],
            sun_dir=(0.65, -0.76),
            rim_color=(1.00, 0.95, 0.84),
            rim_strength=0.78,
            seed=240,
        )
        bg = _over(sky, c_rgb, c_a)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.68, 0.76, 0.84), strength=0.34, band_height=0.075)
        bg = _draw_catenary_poles_with_sag(
            bg, horizon_y_frac=horizon_y, pole_color=(30, 44, 68), wire_color=(42, 58, 84), num_poles=8, pole_h_frac=0.11
        )
        cache["bg"] = bg
        cache["horizon_y"] = horizon_y

    elif sid == "S25":
        # Interior Medium Close-Up: silhouette leaning closer to glass in awe
        horizon_y = 0.58
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.07, 0.14, 0.30)),
                (0.35, (0.24, 0.38, 0.58)),
                (0.56, (0.80, 0.84, 0.86)),
                (1.00, (0.12, 0.22, 0.38)),
            ],
            seed=25,
            brush_strength=0.024,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[(0.26, 0.22, 0.24, 0.32), (0.60, 0.28, 0.20, 0.26)],
            base_y_frac=horizon_y - 0.02,
            color_ramp=[
                (0.20, 0.28, 0.46),
                (0.48, 0.54, 0.68),
                (0.82, 0.84, 0.88),
                (0.98, 0.95, 0.86),
            ],
            sun_dir=(0.6, -0.8),
            rim_color=(0.98, 0.95, 0.86),
            rim_strength=0.72,
            seed=250,
        )
        ext = _over(sky, c_rgb, c_a * 0.88)
        ext = _fast_water_reflect(ext, horizon_y, 0.0, ripple_amp=1.8, darkness=0.88)
        cache["ext"] = ext
        cache["horizon_y"] = horizon_y

    elif sid == "S26":
        # Medium-Wide Water-Level: drifting warm floating sea-lanterns
        horizon_y = 0.42
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.05, 0.10, 0.22)),
                (0.26, (0.11, 0.20, 0.36)),
                (0.42, (0.36, 0.48, 0.64)),
                (1.00, (0.07, 0.13, 0.26)),
            ],
            seed=26,
            brush_strength=0.024,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[(0.24, 0.20, 0.22, 0.22), (0.72, 0.18, 0.26, 0.24)],
            base_y_frac=horizon_y - 0.01,
            color_ramp=[
                (0.14, 0.20, 0.36),
                (0.34, 0.40, 0.56),
                (0.68, 0.70, 0.78),
                (0.94, 0.90, 0.82),
            ],
            sun_dir=(0.5, -0.85),
            rim_color=(0.96, 0.90, 0.80),
            rim_strength=0.60,
            seed=260,
        )
        bg = _over(sky, c_rgb, c_a * 0.82)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.48, 0.58, 0.72), strength=0.28, band_height=0.07)
        cache["bg"] = bg
        cache["horizon_y"] = horizon_y

    elif sid == "S27":
        # Sweeping Ghibli Side-Track: classic wide profile of glowing train gliding across infinite sea
        horizon_y = 0.52
        sky = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.05, 0.11, 0.25)),
                (0.24, (0.14, 0.25, 0.44)),
                (0.42, (0.42, 0.56, 0.72)),
                (0.52, (0.86, 0.85, 0.81)),
                (1.00, (0.08, 0.15, 0.28)),
            ],
            seed=27,
            brush_strength=0.026,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            width, height,
            towers=[
                (0.14, 0.22, 0.20, 0.28),
                (0.46, 0.16, 0.26, 0.34),
                (0.78, 0.20, 0.22, 0.30),
            ],
            base_y_frac=horizon_y - 0.01,
            color_ramp=[
                (0.18, 0.25, 0.42),
                (0.45, 0.48, 0.64),
                (0.80, 0.78, 0.84),
                (1.00, 0.96, 0.86),
            ],
            sun_dir=(0.6, -0.8),
            rim_color=(1.00, 0.95, 0.84),
            rim_strength=0.76,
            seed=270,
        )
        bg = _over(sky, c_rgb, c_a * 0.92)
        bg = apply_atmospheric_haze(bg, horizon_y, (0.70, 0.78, 0.86), strength=0.30, band_height=0.07)
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
    """Render one 60fps frame for Act V (S22–S27) as uint8 RGB (height, width, 3)."""
    if cache is None or cache.get("w") != width or cache.get("h") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(shot_spec.duration_s, 1e-3)
    prog = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]
    vig = cache["vig"]

    if sid == "S22":
        # Wide Exterior Reveal: sloping rails submerging into mirror-sea + slow crane-down
        hy = cache["horizon_y"] - 0.025 * prog
        img = _fast_water_reflect(cache["bg"], hy, local_t, ripple_amp=2.4, darkness=0.86)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        # Perspective steel rails descending from bottom-left into vanishing point on mirror-sea
        vp_x = int(width * (0.62 - 0.03 * prog))
        vp_y = int(height * (hy + 0.01))
        lw = max(1, int(round(width / 640.0)))
        # Submerged wooden ties fading into the water
        for k in range(18):
            f = (k / 18.0) ** 1.6
            ty = int(vp_y + (height * 1.02 - vp_y) * f)
            tx_c = int(vp_x + (width * 0.24 - vp_x) * f)
            half_w = int(width * (0.008 + 0.14 * f))
            alpha_t = float(np.clip(1.0 - (1.0 - f) * 0.8, 0.15, 0.85))
            col = (int(42 * alpha_t + 90 * (1 - alpha_t)), int(52 * alpha_t + 115 * (1 - alpha_t)), int(70 * alpha_t + 145 * (1 - alpha_t)))
            cv2.line(u8, (tx_c - half_w, ty), (tx_c + half_w, ty), col, max(1, int(lw * (1 + 2 * f))), cv2.LINE_AA)
        # Twin gleaming steel rails
        cv2.line(u8, (int(width * 0.12), height), (vp_x - 2, vp_y), (175, 198, 222), lw + 1, cv2.LINE_AA)
        cv2.line(u8, (int(width * 0.38), height), (vp_x + 2, vp_y), (175, 198, 222), lw + 1, cv2.LINE_AA)
        # Water lapping ripples over submerged rails
        for r_i in range(5):
            rf = (r_i * 0.18 + local_t * 0.06) % 1.0
            ry = int(vp_y + (height * 0.92 - vp_y) * (0.15 + 0.75 * rf))
            rx = int(vp_x + (width * 0.25 - vp_x) * (0.15 + 0.75 * rf))
            rw = int(width * (0.04 + 0.12 * rf))
            cv2.ellipse(u8, (rx, ry), (rw, max(2, int(rw * 0.12))), 0, 0, 360, (195, 212, 230), 1, cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        # Distant warm headlight glint on horizon
        img = draw_lantern(img, vp_x / width, (vp_y - height * 0.015) / height, radius_frac=0.006, glow_radius_mult=10.0, glow_strength=0.45 + 0.25 * prog, hanging=False)
        return _fast_finish(img, grain, vig)

    elif sid == "S23":
        # Low Water-Level Close-Up: train wheels slicing through shallow water & fanning wake spray
        hy = cache["horizon_y"]
        img = _fast_water_reflect(cache["bg"], hy, local_t * 1.8, ripple_amp=3.5, darkness=0.84)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        # Lower train carriage body & glowing windows at top of frame
        body_bot = int(height * 0.44)
        cv2.rectangle(u8, (0, 0), (width, body_bot), (26, 36, 54), -1)
        cv2.rectangle(u8, (0, int(height * 0.34)), (width, int(height * 0.38)), (132, 62, 52), -1)
        # Warm glowing windows passing along top edge
        win_shift = int((local_t * 0.18 % 1.0) * width * 0.25)
        for wx in range(-int(width * 0.25), width + int(width * 0.25), int(width * 0.22)):
            x0 = wx - win_shift
            cv2.rectangle(u8, (x0, int(height * 0.04)), (x0 + int(width * 0.14), int(height * 0.26)), (248, 210, 128), -1)
            cv2.rectangle(u8, (x0, int(height * 0.04)), (x0 + int(width * 0.14), int(height * 0.26)), (44, 52, 68), max(2, int(width / 320)), cv2.LINE_AA)
        # Bogie frame & two large spinning steel wheels slicing the water surface
        wheel_y = int(height * 0.52)
        wheel_r = int(height * 0.13)
        cv2.rectangle(u8, (int(width * 0.16), int(height * 0.42)), (int(width * 0.84), int(height * 0.52)), (18, 24, 36), -1)
        for cx_frac in (0.32, 0.68):
            cx = int(width * cx_frac)
            cv2.circle(u8, (cx, wheel_y), wheel_r, (34, 42, 56), -1, cv2.LINE_AA)
            cv2.circle(u8, (cx, wheel_y), wheel_r, (148, 168, 192), max(2, int(width / 320)), cv2.LINE_AA)
            cv2.circle(u8, (cx, wheel_y), int(wheel_r * 0.32), (82, 96, 116), -1, cv2.LINE_AA)
            # Rotating spokes
            angle0 = local_t * 9.5
            for sp in range(6):
                ang = angle0 + sp * (np.pi / 3.0)
                ex = int(cx + np.cos(ang) * wheel_r * 0.88)
                ey = int(wheel_y + np.sin(ang) * wheel_r * 0.88)
                cv2.line(u8, (cx, wheel_y), (ex, ey), (98, 114, 136), max(2, int(width / 480)), cv2.LINE_AA)
        # Silky concentric wake ripples & arcing silver water spray droplets
        rng = np.random.default_rng(230 + int(local_t * 12))
        for cx_frac in (0.32, 0.68):
            cx = int(width * cx_frac)
            for k in range(4):
                phase = (local_t * 1.4 + k * 0.25) % 1.0
                rx = int(wheel_r * (1.1 + phase * 2.2))
                ry = int(wheel_r * (0.22 + phase * 0.45))
                col_v = int(235 - 90 * phase)
                cv2.ellipse(u8, (cx, int(height * 0.60)), (rx, ry), 0, 0, 180, (col_v, col_v + 8, min(255, col_v + 18)), max(1, int(width / 640)), cv2.LINE_AA)
            # Spray droplets fanning outward
            for _ in range(28):
                ang = rng.uniform(-0.85 * np.pi, -0.15 * np.pi)
                dist = rng.uniform(wheel_r * 0.6, wheel_r * 2.1)
                px = int(cx + np.cos(ang) * dist * 1.4)
                py = int(height * 0.58 + np.sin(ang) * dist * 0.7)
                pr = max(1, int(rng.uniform(1.0, 3.2) * width / 960.0))
                if 0 <= px < width and 0 <= py < height:
                    cv2.circle(u8, (px, py), pr, (235, 244, 255), -1, cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        img = render_sea_wake(img, bow_xy=(0.20, 0.60), tail_xy=(0.95, 0.66), wake_width_frac=0.14, foam_color=(0.92, 0.96, 1.00), opacity=0.78, seed=23)
        return _fast_finish(img, grain, vig)

    elif sid == "S24":
        # Monumental Wide (Hero): cumulus cloud cathedrals mirrored in sea, train gliding left
        hy = cache["horizon_y"]
        img = cache["bg"].copy()
        # Train glides smoothly right-to-left across mirror-sea
        x_shift = -0.14 * prog
        x0 = 0.28 + x_shift
        x1 = 0.76 + x_shift
        img = draw_train_profile(
            img,
            x_start_frac=x0,
            x_end_frac=x1,
            y_bottom_frac=hy + 0.004,
            height_frac=0.076,
            body_color=(0.13, 0.18, 0.27),
            stripe_color=(0.58, 0.28, 0.24),
            window_color=(1.00, 0.85, 0.52),
            headlight_left=True,
            num_cars=3,
        )
        img = _fast_water_reflect(img, hy, local_t, ripple_amp=2.4, darkness=0.86)
        # Submerged parallel rails
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        cv2.line(u8, (0, int((hy + 0.007) * height)), (width, int((hy + 0.007) * height)), (155, 178, 204), 1, cv2.LINE_AA)
        cv2.line(u8, (0, int((hy + 0.015) * height)), (width, int((hy + 0.015) * height)), (125, 150, 178), 1, cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        img = render_sea_wake(
            img,
            bow_xy=(x0, hy + 0.009),
            tail_xy=(min(0.98, x1 + 0.14), hy + 0.015),
            wake_width_frac=0.068,
            foam_color=(0.94, 0.97, 1.00),
            opacity=0.92,
            seed=24,
        )
        for lx, ly, lr in ((0.14, 0.77, 0.015), (0.35, 0.69, 0.011), (0.66, 0.74, 0.014), (0.84, 0.82, 0.017)):
            bob = 0.004 * np.sin(local_t * 1.6 + lx * 12.0)
            img = draw_lantern(img, lx, ly + bob, radius_frac=lr, glow_radius_mult=8.0, glow_strength=0.82, hanging=False, water_reflection=True)
        return _fast_finish(img, grain, vig, bloom_thresh=0.70, bloom_tight=0.40, bloom_wide=0.28)

    elif sid == "S25":
        # Interior Medium Close-Up: silhouette leans closer to glass in quiet awe
        ext = cache["ext"]
        # Parallax scroll of exterior sea & clouds
        shift_px = int((local_t * 0.045 % 1.0) * width)
        img = np.roll(ext, -shift_px, axis=1)
        # Passing catenary pole outside window
        pole_x = int(((0.85 - local_t * 0.28) % 1.2 - 0.1) * width)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if 0 <= pole_x < width:
            cv2.rectangle(u8, (pole_x, 0), (pole_x + max(4, int(width * 0.012)), height), (28, 40, 62), -1)
        # Wooden train window frame (left/right/top/bottom)
        cv2.rectangle(u8, (0, 0), (int(width * 0.06), height), (32, 24, 20), -1)
        cv2.rectangle(u8, (int(width * 0.94), 0), (width, height), (32, 24, 20), -1)
        cv2.rectangle(u8, (0, int(height * 0.84)), (width, height), (36, 26, 20), -1)
        cv2.rectangle(u8, (0, 0), (width, int(height * 0.06)), (32, 24, 20), -1)
        # Passenger leaning gently forward toward window glass (Catmull-Rom smooth profile)
        lean = 0.032 * prog
        face_ctrl = [
            (0.68 - lean, 0.30), (0.66 - lean, 0.35), (0.65 - lean, 0.39),
            (0.635 - lean, 0.43), (0.65 - lean, 0.45), (0.648 - lean, 0.49),
            (0.66 - lean, 0.53), (0.71 - lean, 0.56), (0.76 - lean, 0.52),
            (0.78 - lean, 0.42), (0.76 - lean, 0.31),
        ]
        cv2.fillPoly(u8, [_catmull_rom(face_ctrl, width, height, closed=True)], (42, 46, 62), cv2.LINE_AA)
        hair_ctrl = [
            (0.65 - lean, 0.34), (0.67 - lean, 0.27), (0.73 - lean, 0.23),
            (0.80 - lean, 0.25), (0.85 - lean, 0.33), (0.87 - lean, 0.48),
            (0.85 - lean, 0.64), (0.78 - lean, 0.66), (0.74 - lean, 0.55),
            (0.75 - lean, 0.40), (0.71 - lean, 0.32),
        ]
        cv2.fillPoly(u8, [_catmull_rom(hair_ctrl, width, height, closed=True)], (24, 26, 38), cv2.LINE_AA)
        sh_ctrl = [
            (0.56 - lean * 0.5, 1.00), (0.60 - lean * 0.6, 0.72), (0.70 - lean * 0.7, 0.62),
            (0.84 - lean * 0.5, 0.64), (0.94, 0.80), (0.98, 1.00),
        ]
        cv2.fillPoly(u8, [_catmull_rom(sh_ctrl, width, height, closed=True)], (28, 24, 30), cv2.LINE_AA)
        # Cool silver-blue sea rim-light washing over face & nose bridge
        rim_pts = _catmull_rom(face_ctrl[:7], width, height, closed=False)
        cv2.polylines(u8, [rim_pts], False, (198, 224, 248), max(2, int(width / 480)), cv2.LINE_AA)
        img = u8.astype(np.float32) / 255.0
        return _fast_finish(img, grain, vig, bloom_thresh=0.74)

    elif sid == "S26":
        # Medium-Wide Water-Level: drifting warm floating sea-lanterns passing silently
        hy = cache["horizon_y"]
        img = cache["bg"].copy()
        # Distant glowing train gliding softly in background
        tx0 = 0.08 + 0.16 * prog
        img = draw_train_profile(
            img,
            x_start_frac=tx0,
            x_end_frac=tx0 + 0.34,
            y_bottom_frac=hy + 0.002,
            height_frac=0.042,
            body_color=(0.11, 0.16, 0.25),
            stripe_color=(0.48, 0.24, 0.22),
            window_color=(1.00, 0.84, 0.50),
            headlight_left=False,
            num_cars=3,
        )
        img = _fast_water_reflect(img, hy, local_t, ripple_amp=2.6, darkness=0.84)
        # Drifting warm Toro Nagashi sea-lanterns with 3-plane parallax
        lantern_specs = [
            (0.18, 0.58, 0.012, 0.018),
            (0.46, 0.64, 0.016, 0.025),
            (0.76, 0.60, 0.013, 0.020),
            (0.28, 0.78, 0.024, 0.040),
            (0.64, 0.84, 0.028, 0.048),
            (0.88, 0.74, 0.020, 0.032),
        ]
        for bx, by, br, spd in lantern_specs:
            lx = (bx - local_t * spd) % 1.12 - 0.06
            ly = by + 0.006 * np.sin(local_t * 1.4 + bx * 10.0)
            if 0.02 < lx < 0.98:
                img = draw_lantern(img, lx, ly, radius_frac=br, glow_radius_mult=8.5, glow_strength=0.88, hanging=False, water_reflection=True)
        return _fast_finish(img, grain, vig, bloom_thresh=0.66, bloom_tight=0.44, bloom_wide=0.32)

    else:
        # S27: Sweeping Ghibli Side-Track — classic wide profile of glowing train gliding across infinite sea
        hy = cache["horizon_y"]
        img = cache["bg"].copy()
        img = _draw_catenary_poles_with_sag(
            img,
            horizon_y_frac=hy,
            pole_color=(28, 42, 66),
            wire_color=(40, 56, 82),
            num_poles=7,
            pole_h_frac=0.10,
            x_offset_frac=-local_t * 0.09,
        )
        # Train centered in silky side-tracking shot, subtle drift
        x0 = 0.22 - 0.03 * np.sin(prog * np.pi)
        x1 = x0 + 0.56
        img = draw_train_profile(
            img,
            x_start_frac=x0,
            x_end_frac=x1,
            y_bottom_frac=hy + 0.004,
            height_frac=0.068,
            body_color=(0.12, 0.17, 0.26),
            stripe_color=(0.58, 0.27, 0.23),
            window_color=(1.00, 0.86, 0.54),
            headlight_left=True,
            num_cars=4,
        )
        img = _fast_water_reflect(img, hy, local_t * 1.3, ripple_amp=2.2, darkness=0.87)
        img = render_sea_wake(
            img,
            bow_xy=(x0, hy + 0.008),
            tail_xy=(min(0.99, x1 + 0.18), hy + 0.014),
            wake_width_frac=0.056,
            foam_color=(0.94, 0.97, 1.00),
            opacity=0.88,
            seed=27,
        )
        return _fast_finish(img, grain, vig, bloom_thresh=0.70, bloom_tight=0.40, bloom_wide=0.28)
