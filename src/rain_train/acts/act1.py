"""Act I — BEFORE THE STORM (Shots S01–S05, 00:00.0–00:35.0).

Palette: Overcast Pearl (#CFD6DC), Slate Blue (#3D4E5E), Warm Cedar (#6E4B3A), Lantern Amber (#F5B969).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..painterly.clouds import render_cumulus_cathedral, render_sky_gradient
from ..painterly.ink import (
    draw_lantern,
    draw_station_perspective,
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
from ..painterly.weather import render_slanted_rain
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
    for freq, amp in ((2.0, 0.48), (5.0, 0.28), (11.0, 0.15), (21.0, 0.09)):
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
    """Fast bilinear pan/zoom crop from an oversized float32 canvas."""
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


def _make_vignette_mask(h: int, w: int, strength: float = 0.26) -> np.ndarray:
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = xx * xx + yy * yy
    return np.clip(1.0 - strength * 0.5 * r2, 0.45, 1.0)[..., None]


def _finalize(
    img: np.ndarray,
    grain: np.ndarray,
    vig_strength: float = 0.24,
    bloom_thresh: float = 0.72,
    bloom_tight: float = 0.32,
    bloom_wide: float = 0.22,
    contrast: float = 1.05,
    saturation: float = 1.04,
    vig_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = img.shape[:2]
    if bloom_tight > 0.0 or bloom_wide > 0.0:
        bw, bh = max(8, w // 4), max(8, h // 4)
        small = cv2.resize(img, (bw, bh), interpolation=cv2.INTER_AREA)
        bright = np.maximum(small - bloom_thresh, 0.0)
        tight = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(1.5, bw / 160.0))
        wide = cv2.GaussianBlur(bright, (0, 0), sigmaX=max(4.0, bw / 48.0))
        bloom_s = tight * bloom_tight + wide * bloom_wide
        bloom = cv2.resize(bloom_s, (w, h), interpolation=cv2.INTER_LINEAR)
        img = np.clip(img + bloom, 0.0, 1.0)
    # Fused paper grain + contrast + vignette
    g = (grain[..., None] - 0.5) * 0.055
    img = (img + g - 0.5) * contrast + 0.5
    if vig_mask is None:
        vig_mask = _make_vignette_mask(h, w, vig_strength)
    img = np.clip(img * vig_mask, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def prepare_shot_cache(
    shot_spec: ShotSpec,
    width: int = 1920,
    height: int = 1080,
) -> Dict[str, Any]:
    """Pre-render static painterly layers for Act I shots (S01–S05)."""
    sid = shot_spec.id
    grain = generate_paper_grain(height, width, seed=int(sid[1:]) * 17)
    cache: Dict[str, Any] = {"id": sid, "width": width, "height": height, "grain": grain}

    if sid == "S01":
        # Wide establishing high valley dusk — oversized canvas for slow L->R pan + push-in
        cw, ch = int(width * 1.18), int(height * 1.14)
        sky = render_sky_gradient(
            cw, ch,
            stops=[
                (0.00, (0.20, 0.26, 0.33)),
                (0.32, (0.38, 0.45, 0.51)),
                (0.62, (0.72, 0.76, 0.80)),
                (1.00, (0.28, 0.34, 0.40)),
            ],
            seed=101,
            brush_strength=0.030,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            cw, ch,
            towers=[
                (0.18, 0.28, 0.32, 0.18),
                (0.54, 0.22, 0.38, 0.22),
                (0.86, 0.26, 0.30, 0.19),
            ],
            base_y_frac=0.52,
            color_ramp=[
                (0.22, 0.28, 0.35),
                (0.42, 0.48, 0.54),
                (0.68, 0.73, 0.77),
                (0.83, 0.86, 0.88),
            ],
            sun_dir=(-0.4, -0.8),
            rim_color=(0.88, 0.90, 0.92),
            rim_strength=0.35,
            seed=1010,
        )
        bg = _over(sky, c_rgb, c_a * 0.78)
        bg = _draw_hills(bg, 0.62, 0.15, (0.25, 0.31, 0.37), seed=11)
        bg = apply_atmospheric_haze(bg, 0.58, (0.68, 0.73, 0.77), strength=0.38, band_height=0.14)
        bg = _draw_hills(bg, 0.70, 0.12, (0.17, 0.23, 0.28), seed=12)
        bg = apply_atmospheric_haze(bg, 0.67, (0.58, 0.64, 0.70), strength=0.32, band_height=0.11)
        bg = _draw_hills(bg, 0.79, 0.08, (0.12, 0.16, 0.19), seed=13)

        # Valley floor, curving rail line, and lone wooden station
        u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
        # Curving rail embankment
        rail_pts = []
        for s in np.linspace(0.0, 1.0, 40):
            rx = int((0.02 + 0.96 * s) * cw)
            ry = int((0.83 - 0.06 * np.sin(s * np.pi) + 0.03 * s) * ch)
            rail_pts.append([rx, ry])
        pts_arr = np.array(rail_pts, dtype=np.int32)
        cv2.polylines(u8, [pts_arr], False, (52, 58, 64), max(2, ch // 180), cv2.LINE_AA)
        cv2.polylines(u8, [pts_arr + np.array([0, max(2, ch // 240)])], False, (158, 168, 178), max(1, ch // 360), cv2.LINE_AA)

        # Lone wooden station silhouette at center-right
        sx, sy = int(cw * 0.62), int(ch * 0.75)
        sw, sh = int(cw * 0.075), int(ch * 0.042)
        cv2.rectangle(u8, (sx, sy), (sx + sw, sy + sh), (62, 44, 34), -1)
        roof = np.array([
            [sx - sw // 6, sy + 2],
            [sx + sw // 2, sy - sh // 2],
            [sx + sw + sw // 6, sy + 2],
        ], dtype=np.int32)
        cv2.fillPoly(u8, [roof], (40, 32, 30), cv2.LINE_AA)
        # Tiny waiting platform
        cv2.rectangle(u8, (sx - sw // 2, sy + sh), (sx + int(sw * 1.6), sy + sh + max(3, ch // 160)), (76, 62, 54), -1)
        cache["canvas"] = u8.astype(np.float32) / 255.0
        cache["station_xy"] = (0.655, 0.768)

    elif sid == "S02":
        # Close-up tall roadside pampas grass bending in pre-storm gusts
        bg = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.22, 0.29, 0.36)),
                (0.45, (0.50, 0.56, 0.62)),
                (0.75, (0.28, 0.35, 0.34)),
                (1.00, (0.12, 0.16, 0.15)),
            ],
            seed=102,
            brush_strength=0.035,
        )
        bg = _draw_hills(bg, 0.64, 0.12, (0.20, 0.27, 0.30), seed=22)
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=max(2.0, width / 220.0))
        cache["bg"] = bg

        rng = np.random.default_rng(202)
        blades = []
        for i in range(44):
            bx = rng.uniform(-0.05, 1.05)
            bh = rng.uniform(0.38, 0.82)
            stiff = rng.uniform(0.7, 1.35)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            has_plume = rng.random() < 0.42
            depth = rng.uniform(0.35, 1.0)
            blades.append((bx, bh, stiff, phase, has_plume, depth))
        blades.sort(key=lambda b: b[5])
        cache["blades"] = blades

        leaves = []
        for i in range(18):
            leaves.append((
                rng.uniform(0.0, 1.0),
                rng.uniform(0.15, 0.85),
                rng.uniform(0.28, 0.65),
                rng.uniform(0.0, 2.0 * np.pi),
                rng.uniform(0.008, 0.018),
            ))
        cache["leaves"] = leaves

    elif sid == "S03":
        # Low-angle upward: swaying black power lines & wooden poles against darkening fast storm clouds
        cw, ch = int(width * 1.35), int(height * 1.25)
        sky = render_sky_gradient(
            cw, ch,
            stops=[
                (0.00, (0.12, 0.17, 0.23)),
                (0.38, (0.21, 0.28, 0.36)),
                (0.72, (0.40, 0.47, 0.54)),
                (1.00, (0.56, 0.62, 0.68)),
            ],
            seed=103,
            brush_strength=0.030,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            cw, ch,
            towers=[
                (0.16, 0.35, 0.34, 0.26),
                (0.48, 0.28, 0.42, 0.30),
                (0.82, 0.34, 0.36, 0.25),
            ],
            base_y_frac=0.72,
            color_ramp=[
                (0.10, 0.14, 0.19),
                (0.18, 0.24, 0.31),
                (0.33, 0.40, 0.48),
                (0.56, 0.62, 0.68),
            ],
            sun_dir=(-0.3, -0.9),
            rim_color=(0.74, 0.79, 0.84),
            rim_strength=0.28,
            seed=1030,
        )
        cache["sky"] = _over(sky, c_rgb, c_a * 0.86)

    elif sid == "S04":
        # Macro top-down: weathered grey platform stone tiles
        rng = np.random.default_rng(104)
        base = np.full((height, width, 3), [0.36, 0.39, 0.42], dtype=np.float32)
        # Subtle stone variation + mortar lines
        noise_low = rng.normal(0.0, 0.045, (max(8, height // 24), max(8, width // 24), 1)).astype(np.float32)
        noise_low = cv2.resize(noise_low, (width, height), interpolation=cv2.INTER_CUBIC)[..., None]
        base = np.clip(base + noise_low, 0.15, 0.65)
        u8 = np.clip(base * 255.0, 0, 255).astype(np.uint8)

        # Draw 4x3 weathered stone slab joints
        cols, rows = 4, 3
        for c in range(1, cols):
            x = int(c * width / cols)
            cv2.line(u8, (x, 0), (x, height), (48, 52, 56), max(2, width // 320), cv2.LINE_AA)
        for r in range(1, rows):
            y = int(r * height / rows)
            cv2.line(u8, (0, y), (width, y), (46, 50, 54), max(2, height // 180), cv2.LINE_AA)

        cache["stone_base"] = u8.astype(np.float32) / 255.0
        # Pre-generate 32 staggered raindrop impact events over 6.5s
        drops = []
        for i in range(36):
            t0 = (i / 36.0) * 5.8 + rng.uniform(0.0, 0.25)
            dx = rng.uniform(0.08, 0.92)
            dy = rng.uniform(0.10, 0.90)
            rad = rng.uniform(0.022, 0.048)
            drops.append((t0, dx, dy, rad))
        cache["drops"] = drops

    elif sid == "S05":
        # Medium shot eye-level: lone silhouette on platform opening translucent umbrella as lanterns click on
        cw, ch = int(width * 1.12), int(height * 1.12)
        sky = render_sky_gradient(
            cw, ch,
            stops=[
                (0.00, (0.14, 0.19, 0.26)),
                (0.42, (0.26, 0.34, 0.42)),
                (0.75, (0.42, 0.48, 0.54)),
                (1.00, (0.18, 0.22, 0.27)),
            ],
            seed=105,
            brush_strength=0.028,
        )
        c_rgb, c_a = render_cumulus_cathedral(
            cw, ch,
            towers=[(0.25, 0.26, 0.34, 0.20), (0.72, 0.22, 0.36, 0.22)],
            base_y_frac=0.56,
            color_ramp=[(0.14, 0.19, 0.26), (0.24, 0.31, 0.39), (0.38, 0.45, 0.52), (0.52, 0.58, 0.64)],
            sun_dir=(-0.3, -0.9),
            rim_color=None,
            seed=1050,
        )
        bg = _over(sky, c_rgb, c_a * 0.72)
        bg = _draw_hills(bg, 0.65, 0.11, (0.13, 0.18, 0.23), seed=51)
        bg = apply_atmospheric_haze(bg, 0.64, (0.36, 0.43, 0.49), strength=0.35, band_height=0.14)
        bg = draw_station_perspective(bg, platform_y_frac=0.78, vp_xy=(1.15, 0.58))
        cache["canvas"] = bg

    return cache


def render_shot_frame(
    shot_spec: ShotSpec,
    local_t: float,
    global_t: float,
    width: int = 1920,
    height: int = 1080,
    cache: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Render one 60fps RGB uint8 frame for Act I (S01–S05)."""
    if cache is None or cache.get("width") != width or cache.get("height") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(0.001, shot_spec.duration_s)
    u = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]

    if sid == "S01":
        # Slow left-to-right pan + gentle push-in
        cx = 0.46 + 0.08 * u
        cy = 0.50 + 0.02 * u
        zoom = 1.06 + 0.08 * u
        img = _crop_pan_zoom(cache["canvas"], width, height, cx, cy, zoom)
        # Subtle drifting valley mist band
        mist_y = 0.66 + 0.012 * np.sin(local_t * 0.6)
        img = apply_atmospheric_haze(img, mist_y, (0.72, 0.77, 0.81), strength=0.18 + 0.05 * u, band_height=0.09)
        # Faint warm station light waking up in dusk
        sx, sy = cache["station_xy"]
        lx = (sx - (cx - 0.5 / zoom)) * zoom
        ly = (sy - (cy - 0.5 / zoom)) * zoom
        if 0.05 < lx < 0.95 and 0.05 < ly < 0.95:
            glow_str = 0.25 + 0.35 * u
            img = draw_lantern(img, lx, ly, radius_frac=0.005, glow_radius_mult=7.0, glow_strength=glow_str, hanging=False)
        return _finalize(img, grain, vig_strength=0.25, bloom_thresh=0.74, bloom_tight=0.28, bloom_wide=0.20)

    if sid == "S02":
        # Tall roadside pampas grass bending violently in pre-storm gusts
        img = cache["bg"].copy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        gust = 0.65 + 0.35 * np.sin(local_t * 2.4) + 0.22 * np.sin(local_t * 5.7)
        sway_cam_x = int(width * 0.004 * np.sin(local_t * 1.9))

        for bx, bh, stiff, phase, has_plume, depth in cache["blades"]:
            bend = (0.14 + 0.16 * gust) * stiff + 0.05 * np.sin(local_t * 4.2 + phase + bx * 5.0)
            x0 = bx * width + sway_cam_x
            y0 = height * 1.02
            pts = []
            for k in range(7):
                s = k / 6.0
                px = x0 + (s * s) * bend * width
                py = y0 - s * bh * height + (s * s * 0.18) * abs(bend) * height
                pts.append([int(px), int(py)])
            pts_arr = np.array(pts, dtype=np.int32)
            # Depth-graded olive/slate grass color
            c_r = int(28 + 38 * depth)
            c_g = int(42 + 46 * depth)
            c_b = int(38 + 40 * depth)
            thick = max(2, int((2.5 + 3.5 * depth) * (height / 1080.0)))
            cv2.polylines(u8, [pts_arr], False, (c_r, c_g, c_b), thick, cv2.LINE_AA)
            if has_plume:
                tx, ty = pts[-1]
                plume_len = int(height * (0.06 + 0.05 * depth))
                cv2.ellipse(
                    u8,
                    (tx + int(plume_len * 0.35), ty + int(plume_len * 0.12)),
                    (plume_len, max(4, plume_len // 4)),
                    18 + int(bend * 45),
                    0, 360,
                    (int(155 + 45 * depth), int(162 + 42 * depth), int(166 + 40 * depth)),
                    -1,
                    cv2.LINE_AA,
                )

        # Stray leaves sweeping across frame
        for lx0, ly0, spd, lph, lsz in cache["leaves"]:
            lx = ((lx0 + local_t * spd) % 1.25 - 0.12) * width
            ly = (ly0 + 0.08 * np.sin(local_t * 3.8 + lph)) * height
            ang = (local_t * 140.0 + lph * 60.0) % 360.0
            rx = max(4, int(lsz * width))
            ry = max(2, rx // 3)
            cv2.ellipse(u8, (int(lx), int(ly)), (rx, ry), ang, 0, 360, (44, 52, 42), -1, cv2.LINE_AA)

        img = u8.astype(np.float32) / 255.0
        return _finalize(img, grain, vig_strength=0.28, bloom_thresh=0.78, bloom_tight=0.20, bloom_wide=0.14)

    if sid == "S03":
        # Low-angle upward: fast-moving darkening storm clouds + swaying power lines & wooden pole
        cx = 0.36 + 0.24 * u
        cy = 0.56 - 0.10 * u  # slow tilt-up
        img = _crop_pan_zoom(cache["sky"], width, height, cx, cy, zoom=1.12)
        # Progressive storm darkening
        img = np.clip(img * (1.0 - 0.18 * u), 0.0, 1.0)
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        # Bold wooden utility pole & crossarms on right-third
        pole_x = int(width * (0.72 - 0.02 * u))
        pole_w = max(8, int(width * 0.022))
        pole_top = int(height * (0.16 + 0.04 * u))
        cv2.rectangle(u8, (pole_x - pole_w // 2, pole_top), (pole_x + pole_w // 2, height), (22, 26, 32), -1)
        # Crossarms
        for arm_off in (0.06, 0.16):
            ay = pole_top + int(height * arm_off)
            aw = int(width * 0.14)
            cv2.rectangle(u8, (pole_x - aw, ay - max(3, height // 140)), (pole_x + aw // 2, ay + max(3, height // 140)), (26, 30, 36), -1)

        # 6 swaying catenary power lines
        for idx, (y_left, y_right, sag_base) in enumerate([
            (0.24, 0.22, 0.09),
            (0.29, 0.23, 0.11),
            (0.35, 0.32, 0.12),
            (0.41, 0.33, 0.10),
            (0.50, 0.34, 0.14),
            (0.58, 0.36, 0.13),
        ]):
            sway = 0.022 * np.sin(local_t * (2.6 + 0.3 * idx) + idx * 1.1)
            pts = []
            for s in np.linspace(0.0, 1.0, 24):
                px = int(s * width)
                py = int(((y_left * (1.0 - s) + y_right * s) + 4.0 * s * (1.0 - s) * (sag_base + sway)) * height)
                pts.append([px, py])
            cv2.polylines(u8, [np.array(pts, dtype=np.int32)], False, (18, 22, 28), max(2, height // 360), cv2.LINE_AA)

        img = u8.astype(np.float32) / 255.0
        return _finalize(img, grain, vig_strength=0.30, bloom_thresh=0.76, bloom_tight=0.18, bloom_wide=0.12)

    if sid == "S04":
        # Macro top-down: first heavy raindrops splashing & darkening weathered grey stone tiles
        img = cache["stone_base"].copy()
        u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        for t0, dx, dy, rad in cache["drops"]:
            age = local_t - t0
            if age < 0.0:
                continue
            cx_px = int(dx * width)
            cy_px = int(dy * height)
            r_wet = int(rad * min(width, height) * min(1.0, 0.55 + age * 0.45))
            # Darkened wet stone blotch
            overlay = u8.copy()
            cv2.circle(overlay, (cx_px, cy_px), max(3, r_wet), (32, 37, 44), -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.45, u8, 0.55, 0.0, u8)
            # Brief bright splash crown & concentric ripple
            if age < 0.45:
                fade = 1.0 - age / 0.45
                r_ring = int(r_wet * (0.4 + 1.6 * (age / 0.45)))
                ring_col = (int(180 * fade + 40), int(195 * fade + 40), int(210 * fade + 45))
                cv2.circle(u8, (cx_px, cy_px), max(2, r_ring), ring_col, max(1, height // 450), cv2.LINE_AA)

        img = u8.astype(np.float32) / 255.0
        return _finalize(img, grain, vig_strength=0.26, bloom_thresh=0.76, bloom_tight=0.16, bloom_wide=0.10)

    # S05: Medium shot eye-level — lone silhouette pops open umbrella as warm amber lanterns click on
    zoom = 1.02 + 0.09 * u
    img = _crop_pan_zoom(cache["canvas"], width, height, 0.50, 0.50, zoom)

    # Lanterns click on around local_t = 3.2s with brief warm flicker
    lantern_on = 0.0
    if local_t >= 3.2:
        dt = local_t - 3.2
        if dt < 0.12:
            lantern_on = 0.85
        elif dt < 0.22:
            lantern_on = 0.25
        else:
            lantern_on = min(1.0, (dt - 0.22) * 3.5 + 0.4)

    lanterns = [(0.16, 0.28, 0.018), (0.42, 0.30, 0.015), (0.64, 0.32, 0.012)]
    for lx, ly, lr in lanterns:
        if lantern_on > 0.01:
            img = draw_lantern(img, lx, ly, radius_frac=lr, glow_radius_mult=8.0, glow_strength=0.65 * lantern_on, hanging=True)

    # Passenger silhouette with umbrella (rim light warms up when lanterns click on)
    rim_col = (
        0.48 + 0.50 * lantern_on,
        0.54 + 0.24 * lantern_on,
        0.62 - 0.18 * lantern_on,
    )
    img = draw_umbrella_figure(img, 0.34, 0.78, height_frac=0.31, rim_color=rim_col)

    # Light initial rain picking up through S05
    rain_density = 0.25 + 0.55 * u
    warm_lights = [(lx, ly, 0.16, (1.0, 0.78, 0.42)) for lx, ly, _ in lanterns] if lantern_on > 0.1 else None
    img = render_slanted_rain(img, time_s=global_t, density=rain_density, angle_deg=11.0, warm_lights=warm_lights, seed=105)
    return _finalize(img, grain, vig_strength=0.26, bloom_thresh=0.66, bloom_tight=0.42, bloom_wide=0.28)
