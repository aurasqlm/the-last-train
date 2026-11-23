"""Render Gate 1 Hero 4-Panel Proof Sheet (style_frame_check.png, exact 1920x1080).

Renders 4 cinema-grade hand-painted Studio Ghibli / Makoto Shinkai hero frames
(S10, S14, S24, S33) at 960x540 each and composites them into a 1920x1080 2x2 grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .painterly.clouds import render_cumulus_cathedral, render_sky_gradient
from .painterly.ink import (
    draw_lantern,
    draw_petals_and_motes,
    draw_station_perspective,
    draw_train_perspective,
    draw_train_profile,
    draw_umbrella_figure,
)
from .painterly.optics import (
    apply_atmospheric_haze,
    apply_filmic_grade,
    apply_halation_bloom,
    apply_vignette,
    render_aperture_bokeh,
    render_god_rays,
)
from .painterly.paper_grain import (
    apply_paper_grain,
    apply_pigment_granulation,
    generate_paper_grain,
)
from .painterly.weather import (
    apply_puddle_reflection,
    apply_sea_reflection,
    render_macro_window_droplets,
    render_sea_wake,
    render_slanted_rain,
    render_steam_plume,
)


def _over(bg: np.ndarray, fg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None]
    return np.clip(bg * (1.0 - a) + fg * a, 0.0, 1.0)


def _draw_distant_hills(
    img: np.ndarray,
    base_y_frac: float,
    amp_frac: float,
    color: Tuple[float, float, float],
    seed: int = 1,
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


def _draw_catenary_poles_with_sag(
    img: np.ndarray,
    horizon_y_frac: float,
    pole_color: Tuple[int, int, int],
    wire_color: Tuple[int, int, int],
    num_poles: int = 8,
    pole_h_frac: float = 0.11,
) -> np.ndarray:
    """Draw railway catenary poles with natural catenary wire sag (never rigid straight 1px lines)."""
    h, w = img.shape[:2]
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    hy = int(horizon_y_frac * h)
    pole_xs = [int(f * w) for f in np.linspace(0.05, 0.95, num_poles)]
    pole_top = hy - int(pole_h_frac * h)
    wire_y = pole_top + int(h * 0.015)

    # Natural parabolic catenary sag between each pair of poles
    all_xs = [0] + pole_xs + [w]
    for i in range(len(all_xs) - 1):
        x0, x1 = all_xs[i], all_xs[i + 1]
        pts = []
        for s in np.linspace(0.0, 1.0, 14):
            px = int(x0 * (1.0 - s) + x1 * s)
            sag = int(4.0 * s * (1.0 - s) * (h * 0.016))
            pts.append([px, wire_y + sag])
        cv2.polylines(u8, [np.array(pts, dtype=np.int32)], False, wire_color, 1, cv2.LINE_AA)

    for px in pole_xs:
        cv2.line(u8, (px, hy + 2), (px, pole_top), pole_color, 2, cv2.LINE_AA)
        cv2.line(u8, (px - 7, pole_top + 7), (px + 7, pole_top + 7), pole_color, 1, cv2.LINE_AA)
        cv2.line(u8, (px, pole_top + 14), (px + 6, pole_top + 7), pole_color, 1, cv2.LINE_AA)

    return u8.astype(np.float32) / 255.0


def render_hero_s10(w: int = 960, h: int = 540) -> np.ndarray:
    """Top-Left: Act II S10 — 3/4 perspective vintage rain train entering lantern-lit station in heavy downpour."""
    grain = generate_paper_grain(h, w, seed=10)

    # Stormy deep rain-slate & misty teal gouache sky
    img = render_sky_gradient(
        w, h,
        stops=[
            (0.00, (0.07, 0.10, 0.15)),
            (0.34, (0.13, 0.20, 0.27)),
            (0.66, (0.24, 0.34, 0.41)),
            (1.00, (0.10, 0.14, 0.19)),
        ],
        seed=10,
        brush_strength=0.032,
    )

    # Sculpted rain-laden storm clouds
    c_rgb, c_a = render_cumulus_cathedral(
        w, h,
        towers=[(0.22, 0.24, 0.30, 0.20), (0.66, 0.18, 0.36, 0.24), (0.92, 0.26, 0.24, 0.18)],
        base_y_frac=0.54,
        color_ramp=[(0.08, 0.12, 0.17), (0.16, 0.23, 0.30), (0.26, 0.36, 0.44), (0.40, 0.50, 0.57)],
        sun_dir=(-0.3, -0.9),
        rim_color=None,
        seed=110,
    )
    img = _over(img, c_rgb, c_a * 0.72)

    # Distant misty cedar/pine hills
    img = _draw_distant_hills(img, 0.62, 0.13, (0.10, 0.15, 0.19), seed=12)
    img = _draw_distant_hills(img, 0.66, 0.08, (0.07, 0.11, 0.15), seed=15)
    img = apply_atmospheric_haze(img, 0.62, (0.22, 0.31, 0.38), strength=0.40, band_height=0.16)

    # Dramatic 3/4 perspective vintage Japanese train braking into station with volumetric headlights
    img = draw_train_perspective(
        img,
        front_x_frac=0.49,
        front_y_bottom_frac=0.75,
        front_h_frac=0.35,
        vp_xy=(1.18, 0.56),
        body_color=(0.12, 0.17, 0.23),
        stripe_color=(0.58, 0.25, 0.20),
        window_color=(1.00, 0.82, 0.46),
        headlight_beams=True,
    )

    # Rustic wooden station awning, pillars, and receding wet platform
    img = draw_station_perspective(img, platform_y_frac=0.75, vp_xy=(1.18, 0.56))

    # Billowing multi-layered painterly white-amber brake steam plumes erupting from undercarriage
    img = render_steam_plume(img, (0.43, 0.72), (0.14, 0.085), (0.94, 0.86, 0.72), opacity=0.52, seed=101)
    img = render_steam_plume(img, (0.60, 0.73), (0.17, 0.075), (0.88, 0.83, 0.75), opacity=0.42, seed=102)
    img = render_steam_plume(img, (0.76, 0.69), (0.12, 0.055), (0.80, 0.82, 0.84), opacity=0.30, seed=103)

    # Warm hanging paper station lanterns receding in perspective
    lanterns = [(0.11, 0.27, 0.019), (0.33, 0.30, 0.015), (0.52, 0.32, 0.012)]
    for lx, ly, lr in lanterns:
        img = draw_lantern(img, lx, ly, radius_frac=lr, glow_radius_mult=8.0, glow_strength=0.66, hanging=True)

    # Graceful anime passenger silhouette with ribbed translucent vinyl umbrella
    img = draw_umbrella_figure(img, 0.21, 0.76, height_frac=0.28, rim_color=(0.98, 0.78, 0.44))

    # Broken wet-stone platform reflections with organic painterly ripples
    ripples = [(0.18, 0.86, 0.2), (0.40, 0.82, 1.8), (0.64, 0.89, 3.4), (0.83, 0.84, 5.1)]
    img = apply_puddle_reflection(img, y_start_frac=0.755, time_s=1.4, ripple_centers=ripples, darkness=0.78)

    # Strict opaque wooden station roof occlusion mask so ZERO rain streaks ever pass through the solid roof!
    roof_occ = np.zeros((h, w), dtype=np.float32)
    roof_poly = np.array([
        [0, 0],
        [int(w * 0.725), 0],
        [int(w * 0.642), int(h * 0.265)],
        [0, int(h * 0.165)],
    ], dtype=np.int32)
    cv2.fillPoly(roof_occ, [roof_poly], 1.0)
    # Keep interior strictly 1.0 while feathering only 1px outside the eave
    roof_occ = np.maximum(roof_occ, cv2.GaussianBlur(roof_occ, (0, 0), sigmaX=1.2, sigmaY=1.2))

    # 4-depth-layer slanted rain catching warm lantern & headlight cones (masked behind opaque roof!)
    warm_lights = [
        (0.11, 0.27, 0.16, (1.0, 0.78, 0.42)),
        (0.33, 0.30, 0.15, (1.0, 0.78, 0.42)),
        (0.52, 0.32, 0.14, (1.0, 0.78, 0.42)),
        (0.34, 0.62, 0.26, (1.0, 0.90, 0.66)),
    ]
    splash_segments = [
        (0.13, 0.43, 0.30, 0.43, 0.95),  # Umbrella canopy arc
        (0.00, 0.16, 0.64, 0.26, 0.75),  # Wooden station roof eaves
        (0.46, 0.38, 0.96, 0.50, 0.85),  # Train roof ridge
        (0.02, 0.755, 0.98, 0.755, 0.80), # Wet stone platform
    ]
    img = render_slanted_rain(
        img,
        time_s=1.4,
        density=1.25,
        angle_deg=14.0,
        warm_lights=warm_lights,
        seed=10,
        occlusion_mask=roof_occ,
        splash_segments=splash_segments,
    )

    # 35mm optical halation, pigment granulation, cold-press watercolor paper grain
    img = apply_halation_bloom(img, threshold=0.64, tight_strength=0.48, wide_strength=0.34)
    img = apply_pigment_granulation(img, grain, strength=0.022)
    img = apply_paper_grain(img, grain, strength=0.032)
    img = apply_filmic_grade(img, contrast=1.06, saturation=1.08)
    img = apply_vignette(img, strength=0.28)
    return img


def render_hero_s14(w: int = 960, h: int = 540) -> np.ndarray:
    """Top-Right: Act III S14 — Macro refractive rainy train window, clear warm passenger reflection & town bokeh."""
    grain = generate_paper_grain(h, w, seed=14)

    # Deep indigo rainy night sky & distant town silhouette
    bg = render_sky_gradient(
        w, h,
        stops=[
            (0.00, (0.04, 0.07, 0.14)),
            (0.42, (0.07, 0.12, 0.22)),
            (0.70, (0.11, 0.16, 0.27)),
            (1.00, (0.05, 0.07, 0.12)),
        ],
        seed=14,
        brush_strength=0.024,
    )
    bg = _draw_distant_hills(bg, 0.65, 0.12, (0.04, 0.06, 0.10), seed=140)

    # Crisp vintage cinema lens circular aperture bokeh discs (warm streetlamps & town lights)
    bokeh = render_aperture_bokeh(
        w, h,
        palette=[
            (1.00, 0.80, 0.44),  # Warm tungsten
            (0.98, 0.60, 0.28),  # Streetlamp amber
            (0.94, 0.34, 0.28),  # Distant signal crimson
            (0.58, 0.78, 0.94),  # Cool cyan shopfront
            (1.00, 0.93, 0.74),  # Bright window cream
        ],
        num_discs=68,
        y_range=(0.44, 0.94),
        radius_range=(0.014, 0.050),
        seed=14,
    )
    bg = 1.0 - (1.0 - bg) * (1.0 - np.clip(bokeh * 0.90, 0.0, 0.96))

    # Passing out-of-focus dark utility pole & wires outside
    u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
    cv2.rectangle(u8, (int(w * 0.19), 0), (int(w * 0.22), h), (12, 16, 26), -1)
    cv2.line(u8, (0, int(h * 0.24)), (w, int(h * 0.18)), (16, 22, 34), 3, cv2.LINE_AA)
    bg = cv2.GaussianBlur(u8.astype(np.float32) / 255.0, (0, 0), sigmaX=3.2, sigmaY=3.2)

    # Clear, graceful Fresnel reflection of warm tungsten cabin & seated anime passenger (~40% opacity)
    refl = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Warm tungsten overhead cabin lamp glow from upper-right
    cabin_glow = np.exp(-(((xx - w * 0.82) / (w * 0.44)) ** 2 + ((yy - h * 0.12) / (h * 0.38)) ** 2))
    refl += cabin_glow[..., None] * np.array([0.48, 0.32, 0.15], dtype=np.float32)[None, None, :]

    # Seated anime passenger reflection (clear, poetic 3/4 profile looking left, eye-line, hair, collar, resting hand)
    sil_u8 = np.zeros((h, w, 3), dtype=np.uint8)
    # Hanging cabin hand-strap reflected in upper-center
    cv2.line(sil_u8, (int(w * 0.42), 0), (int(w * 0.43), int(h * 0.20)), (195, 148, 92), 3, cv2.LINE_AA)
    cv2.circle(sil_u8, (int(w * 0.43), int(h * 0.24)), int(h * 0.036), (225, 178, 118), 3, cv2.LINE_AA)

    def _catmull_rom(ctrl: list[Tuple[float, float]], closed: bool = False, samples_per_seg: int = 12) -> np.ndarray:
        """Evaluate a smooth C1 Catmull-Rom spline through control points (eliminates polygonal corners)."""
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

    # Graceful organic anime profile head & jawline looking wistfully out toward left
    face_ctrl = [
        (0.662, 0.305),  # forehead top
        (0.644, 0.352),  # gentle curved forehead
        (0.638, 0.388),  # brow bridge
        (0.624, 0.428),  # delicate petite anime nose tip
        (0.638, 0.448),  # under nose
        (0.635, 0.468),  # soft upper lip
        (0.637, 0.488),  # soft lower lip
        (0.646, 0.526),  # rounded chin
        (0.688, 0.556),  # soft jawline curve
        (0.742, 0.528),  # jaw corner
        (0.768, 0.435),  # cheek/ear
        (0.748, 0.312),  # upper temple
    ]
    face_spline = _catmull_rom(face_ctrl, closed=True, samples_per_seg=10)
    cv2.fillPoly(sil_u8, [face_spline], (228, 178, 124), cv2.LINE_AA)

    # Subtle anime eye-line & wistful lash in profile
    cv2.ellipse(sil_u8, (int(w * 0.658), int(h * 0.402)), (int(w * 0.013), int(h * 0.007)), -10, 180, 360, (68, 42, 28), 2, cv2.LINE_AA)
    cv2.circle(sil_u8, (int(w * 0.654), int(h * 0.409)), max(2, int(h * 0.008)), (78, 52, 38), -1, cv2.LINE_AA)

    # Slender neck curving naturally into collar
    neck_ctrl = [
        (0.690, 0.540),
        (0.686, 0.595),
        (0.682, 0.642),
        (0.752, 0.638),
        (0.750, 0.585),
        (0.744, 0.530),
    ]
    cv2.fillPoly(sil_u8, [_catmull_rom(neck_ctrl, closed=True, samples_per_seg=8)], (202, 152, 102), cv2.LINE_AA)

    # Softly draped coat shoulders leaning gently toward the window glass
    sh_ctrl = [
        (0.515, 1.000),
        (0.555, 0.780),
        (0.605, 0.665),
        (0.685, 0.618),
        (0.785, 0.612),
        (0.865, 0.665),
        (0.920, 0.815),
        (0.950, 1.000),
    ]
    cv2.fillPoly(sil_u8, [_catmull_rom(sh_ctrl, closed=True, samples_per_seg=10)], (122, 82, 52), cv2.LINE_AA)

    # Flowing, organic shoulder-length anime hair with curved crown & sweeping back locks
    hair_ctrl = [
        (0.635, 0.345),  # front wispy bangs tip
        (0.652, 0.285),  # front crown curve
        (0.705, 0.242),  # smooth rounded top of head
        (0.772, 0.248),  # upper back crown
        (0.822, 0.305),  # back of head curve
        (0.848, 0.415),  # sweeping mid-hair volume
        (0.862, 0.555),  # flowing shoulder-length lock
        (0.835, 0.645),  # curved lower hair tip
        (0.785, 0.658),  # inner hair curl
        (0.745, 0.600),  # tucked behind neck
        (0.748, 0.485),  # side-lock framing jaw
        (0.732, 0.382),  # ear/temple sweep
        (0.692, 0.325),  # brow-framing fringe
    ]
    hair_spline = _catmull_rom(hair_ctrl, closed=True, samples_per_seg=12)
    cv2.fillPoly(sil_u8, [hair_spline], (96, 62, 42), cv2.LINE_AA)

    # Wispy curved side-lock framing the cheek
    bang_ctrl = [
        (0.688, 0.318),
        (0.678, 0.382),
        (0.692, 0.468),
        (0.708, 0.518),
        (0.722, 0.455),
        (0.718, 0.368),
    ]
    cv2.fillPoly(sil_u8, [_catmull_rom(bang_ctrl, closed=True, samples_per_seg=8)], (88, 56, 38), cv2.LINE_AA)

    # Coat lapel / collar V-line
    collar_spline = _catmull_rom([(0.630, 0.632), (0.662, 0.688), (0.695, 0.735), (0.735, 0.682), (0.768, 0.622)], closed=False, samples_per_seg=8)
    cv2.polylines(sil_u8, [collar_spline], False, (248, 204, 138), 2, cv2.LINE_AA)

    # Delicate hand resting softly near chin/cheek
    hand_ctrl = [
        (0.602, 0.645),
        (0.618, 0.578),
        (0.636, 0.538),
        (0.652, 0.555),
        (0.650, 0.612),
        (0.628, 0.672),
    ]
    cv2.fillPoly(sil_u8, [_catmull_rom(hand_ctrl, closed=True, samples_per_seg=8)], (218, 168, 118), cv2.LINE_AA)

    # Soft warm golden tungsten rim-light following the exact smooth spline hair & facial contours!
    rim_hair_spline = _catmull_rom(
        [(0.665, 0.268), (0.718, 0.242), (0.778, 0.252), (0.826, 0.312), (0.852, 0.425), (0.860, 0.562), (0.836, 0.640)],
        closed=False,
        samples_per_seg=12,
    )
    cv2.polylines(sil_u8, [rim_hair_spline], False, (255, 228, 162), 3, cv2.LINE_AA)
    rim_face_spline = _catmull_rom(face_ctrl[:8], closed=False, samples_per_seg=10)
    cv2.polylines(sil_u8, [rim_face_spline], False, (255, 234, 178), 2, cv2.LINE_AA)

    # Soft double-glass Fresnel optical reflection feathering (organic & dreamy, zero hard cutout!)
    sil_f = cv2.GaussianBlur(sil_u8.astype(np.float32) / 255.0, (0, 0), sigmaX=1.8, sigmaY=1.8)
    refl = np.clip(refl + sil_f * 0.74, 0.0, 1.0)

    # Composite warm Fresnel cabin reflection onto dark night glass
    glass_bg = 1.0 - (1.0 - bg) * (1.0 - refl * 0.62)

    # Realistic luminous refractive water droplets with clean translucent wiped trails
    img = render_macro_window_droplets(
        glass_bg,
        time_s=2.8,
        num_large=30,
        num_med=76,
        num_micro=220,
        fog_strength=0.15,
        seed=14,
    )
    # Physically accurate inner-surface Fresnel reflection (bounces off the dry inner glass face)
    img = 1.0 - (1.0 - img) * (1.0 - sil_f * 0.26)

    # Clean, straight macro depth-of-field wooden train window frame & dark rubber gasket
    frame_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    frame_a = np.zeros((h, w), dtype=np.float32)
    # Left wooden pillar & dark rubber gasket
    cv2.rectangle(frame_rgb, (0, 0), (int(w * 0.044), h), (26, 20, 16), -1)
    cv2.rectangle(frame_rgb, (int(w * 0.044), 0), (int(w * 0.058), h), (12, 11, 12), -1)
    cv2.rectangle(frame_a, (0, 0), (int(w * 0.058), h), 1.0, -1)
    # Right wooden pillar & dark rubber gasket (straight vertical framing!)
    cv2.rectangle(frame_rgb, (int(w * 0.942), 0), (int(w * 0.956), h), (12, 11, 12), -1)
    cv2.rectangle(frame_rgb, (int(w * 0.956), 0), (w, h), (26, 20, 16), -1)
    cv2.rectangle(frame_a, (int(w * 0.942), 0), (w, h), 1.0, -1)
    # Bottom wooden sill & rubber gasket with soft warm cabin specular bounce
    cv2.rectangle(frame_rgb, (0, int(h * 0.905)), (w, int(h * 0.924)), (12, 11, 12), -1)
    cv2.rectangle(frame_rgb, (0, int(h * 0.924)), (w, h), (30, 22, 16), -1)
    cv2.line(frame_rgb, (int(w * 0.058), int(h * 0.926)), (int(w * 0.942), int(h * 0.926)), (96, 68, 44), 3, cv2.LINE_AA)
    cv2.rectangle(frame_a, (0, int(h * 0.905)), (w, h), 1.0, -1)

    frame_f = cv2.GaussianBlur(frame_rgb.astype(np.float32) / 255.0, (0, 0), sigmaX=3.6, sigmaY=3.6)
    frame_a = cv2.GaussianBlur(frame_a, (0, 0), sigmaX=3.0, sigmaY=3.0)
    img = _over(img, frame_f, frame_a)

    # 35mm optical halation, paper grain, filmic grade
    img = apply_halation_bloom(img, threshold=0.62, tight_strength=0.45, wide_strength=0.32)
    img = apply_paper_grain(img, grain, strength=0.030)
    img = apply_filmic_grade(img, contrast=1.07, saturation=1.10)
    img = apply_vignette(img, strength=0.30)
    return img


def render_hero_s24(w: int = 960, h: int = 540) -> np.ndarray:
    """Bottom-Left: Act V S24 — Spirited-Away flooded mirror-sea tracks, asymmetric cumulus cathedrals & wake."""
    grain = generate_paper_grain(h, w, seed=24)
    horizon_y = 0.55

    # Twilight cobalt to mirror-silver & warm cream horizon gouache sky
    img = render_sky_gradient(
        w, h,
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

    # Asymmetric, awe-inspiring Ghibli sculpted gouache cumulus cloud cathedral (left-center spire + right turret)
    c_rgb, c_a = render_cumulus_cathedral(
        w, h,
        towers=[
            (0.20, 0.26, 0.18, 0.28),
            (0.36, 0.14, 0.22, 0.40),
            (0.54, 0.28, 0.16, 0.24),
            (0.82, 0.30, 0.17, 0.22),
        ],
        base_y_frac=horizon_y - 0.015,
        color_ramp=[
            (0.16, 0.23, 0.40),  # Deep cobalt-indigo flat cloud base (#2A3B66)
            (0.42, 0.44, 0.60),  # Smooth lavender-slate midtone
            (0.76, 0.74, 0.80),  # Soft warm silver-rose body
            (1.00, 0.96, 0.85),  # Sunlit cream-gold crown (#FFF4D8)
        ],
        sun_dir=(0.65, -0.76),
        rim_color=(1.00, 0.95, 0.84),
        rim_strength=0.78,
        seed=240,
    )
    img = _over(img, c_rgb, c_a)

    # Soft horizon mist
    img = apply_atmospheric_haze(img, horizon_y, (0.68, 0.76, 0.84), strength=0.34, band_height=0.075)

    # Catenary poles with natural wire sag
    img = _draw_catenary_poles_with_sag(
        img,
        horizon_y_frac=horizon_y,
        pole_color=(30, 44, 68),
        wire_color=(42, 58, 84),
        num_poles=8,
        pole_h_frac=0.11,
    )

    # Warm-lit vintage train gliding across the submerged mirror-sea tracks (true side-tracking shot)
    img = draw_train_profile(
        img,
        x_start_frac=0.18,
        x_end_frac=0.66,
        y_bottom_frac=horizon_y + 0.004,
        height_frac=0.076,
        body_color=(0.13, 0.18, 0.27),
        stripe_color=(0.58, 0.28, 0.24),
        window_color=(1.00, 0.85, 0.52),
        headlight_left=True,
        num_cars=3,
    )

    # Glassy perspective-foreshortened mirror-sea reflection with concentric pole & lantern ripples
    pole_ripples = [(float(f), horizon_y + 0.005, 16.0) for f in np.linspace(0.05, 0.95, 8)]
    lantern_ripples = [(0.14, 0.77, 26.0), (0.35, 0.69, 20.0), (0.66, 0.74, 24.0), (0.84, 0.82, 30.0)]
    img = apply_sea_reflection(
        img,
        horizon_y_frac=horizon_y,
        time_s=2.0,
        ripple_strength=0.82,
        darkness=0.86,
        ripple_points=pole_ripples + lantern_ripples,
    )

    # Unified side-tracking submerged steel rails running PARALLEL to the horizon right beneath the train wheels!
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    rail_y1 = int((horizon_y + 0.007) * h)
    rail_y2 = int((horizon_y + 0.015) * h)
    cv2.line(u8, (0, rail_y1), (w, rail_y1), (155, 178, 204), 1, cv2.LINE_AA)
    cv2.line(u8, (0, rail_y2), (w, rail_y2), (125, 150, 178), 1, cv2.LINE_AA)
    img = u8.astype(np.float32) / 255.0

    # Lush frothy white-foam & silver-cobalt V-shaped water wake + fine mist spray fanning from wheels
    img = render_sea_wake(
        img,
        bow_xy=(0.18, horizon_y + 0.009),
        tail_xy=(0.78, horizon_y + 0.015),
        wake_width_frac=0.068,
        foam_color=(0.94, 0.97, 1.00),
        opacity=0.95,
        seed=24,
    )

    # Delicate floating Toro Nagashi lanterns casting long, shimmering broken vertical amber light columns
    sea_lanterns = [
        (0.14, 0.77, 0.015),
        (0.35, 0.69, 0.011),
        (0.66, 0.74, 0.014),
        (0.84, 0.82, 0.017),
    ]
    for lx, ly, lr in sea_lanterns:
        img = draw_lantern(img, lx, ly, radius_frac=lr, glow_radius_mult=8.5, glow_strength=0.84, hanging=False, water_reflection=True)

    # 35mm halation, pigment granulation, cold-press watercolor paper grain
    img = apply_halation_bloom(img, threshold=0.70, tight_strength=0.40, wide_strength=0.30)
    img = apply_pigment_granulation(img, grain, strength=0.022)
    img = apply_paper_grain(img, grain, strength=0.032)
    img = apply_filmic_grade(img, contrast=1.05, saturation=1.08)
    img = apply_vignette(img, strength=0.22)
    return img


def render_hero_s33(w: int = 960, h: int = 540) -> np.ndarray:
    """Bottom-Right: Act VI S33 — Euphoric golden-hour cloudbreak, sweeping god-rays, rim-lit train & sakura petals."""
    grain = generate_paper_grain(h, w, seed=33)
    horizon_y = 0.65

    # Radiant cerulean -> luminous sky -> rose-gold -> apricot -> warm golden horizon sky
    img = render_sky_gradient(
        w, h,
        stops=[
            (0.00, (0.14, 0.34, 0.70)),  # Deep cerulean zenith (rich contrast for golden god-rays!)
            (0.24, (0.30, 0.50, 0.78)),  # Luminous sky blue
            (0.44, (0.82, 0.48, 0.52)),  # Rose gold
            (0.56, (0.94, 0.62, 0.40)),  # Radiant apricot
            (0.65, (1.00, 0.86, 0.64)),  # Warm golden horizon (not blown-out white!)
            (1.00, (0.28, 0.38, 0.56)),
        ],
        seed=33,
        brush_strength=0.028,
    )

    # Colossal sculpted storm clouds parting dramatically, radially lit from the central golden cloudbreak!
    c_rgb, c_a = render_cumulus_cathedral(
        w, h,
        towers=[
            (0.08, 0.24, 0.22, 0.34),
            (0.24, 0.34, 0.16, 0.24),
            (0.88, 0.20, 0.24, 0.36),
            (0.72, 0.32, 0.17, 0.24),
        ],
        base_y_frac=horizon_y - 0.04,
        color_ramp=[
            (0.22, 0.20, 0.40),  # Deep violet-indigo storm belly
            (0.58, 0.38, 0.50),  # Rich rose-lavender midtone
            (0.94, 0.62, 0.44),  # Warm apricot lit face (#F6A97A)
            (1.00, 0.92, 0.72),  # Incandescent cream-gold rim (#FFF4D8)
        ],
        sun_pos=(0.48, 0.34),
        rim_color=(1.00, 0.95, 0.78),
        rim_strength=0.88,
        seed=330,
    )
    img = _over(img, c_rgb, c_a)

    # 4 irregular, wide, softly feathered volumetric golden crepuscular light shafts (never geometric spokes!)
    rays = render_god_rays(
        w, h,
        origin_xy=(0.48, 0.20),
        color=(1.00, 0.88, 0.54),
        num_rays=4,
        intensity=0.84,
        spread_deg=(36.0, 144.0),
        seed=33,
        cloud_alpha=c_a,
    )
    img = 1.0 - (1.0 - img) * (1.0 - rays)

    # Soft golden atmospheric horizon haze
    img = apply_atmospheric_haze(img, horizon_y, (0.98, 0.78, 0.54), strength=0.30, band_height=0.075)

    # Dramatically backlit train (rich dark violet-charcoal silhouette + glowing golden-apricot rim-light!)
    img = draw_train_profile(
        img,
        x_start_frac=0.52,
        x_end_frac=0.92,
        y_bottom_frac=horizon_y + 0.003,
        height_frac=0.058,
        body_color=(0.11, 0.09, 0.16),
        stripe_color=(0.48, 0.22, 0.20),
        window_color=(1.00, 0.92, 0.66),
        headlight_left=False,
        num_cars=3,
        rim_light_color=(1.00, 0.90, 0.56),
        rim_light_strength=1.00,
    )

    # Glassy golden-hour mirror-sea with sparkling golden bokeh shimmer path
    img = apply_sea_reflection(
        img,
        horizon_y_frac=horizon_y,
        time_s=3.2,
        ripple_strength=0.76,
        darkness=0.88,
        shimmer_color=(1.00, 0.94, 0.74),
        shimmer_x_frac=0.48,
        shimmer_strength=0.92,
    )

    # Extend sweeping golden god-ray shafts down across the shimmering mirror-sea surface
    img = 1.0 - (1.0 - img) * (1.0 - rays * 0.42)

    # Glowing sunlit water wake trailing behind the golden-hour train
    img = render_sea_wake(
        img,
        bow_xy=(0.92, horizon_y + 0.008),
        tail_xy=(0.45, horizon_y + 0.013),
        wake_width_frac=0.042,
        foam_color=(1.00, 0.93, 0.76),
        opacity=0.82,
        seed=33,
    )

    # Unmistakable graceful pink-white & golden-backlit sakura petals across 3 DoF planes + golden dust motes
    img = draw_petals_and_motes(
        img,
        time_s=3.2,
        num_petals=62,
        num_motes=75,
        petal_color=(1.00, 0.78, 0.83),
        mote_color=(1.00, 0.95, 0.74),
        seed=33,
    )

    # Lush 35mm optical halation, pigment granulation, cold-press watercolor paper grain
    img = apply_halation_bloom(img, threshold=0.68, tight_strength=0.44, wide_strength=0.34)
    img = apply_pigment_granulation(img, grain, strength=0.022)
    img = apply_paper_grain(img, grain, strength=0.032)
    img = apply_filmic_grade(img, contrast=1.06, saturation=1.10)
    img = apply_vignette(img, strength=0.20)
    return img


def _add_label(panel_u8: np.ndarray, label: str) -> np.ndarray:
    """Add a subtle, cinema-grade serif shot label in the bottom-left corner of a panel."""
    pil = Image.fromarray(panel_u8)
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerif.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = 12, 6
    x0, y0 = 14, panel_u8.shape[0] - th - pad_y * 2 - 12
    draw.rounded_rectangle(
        [x0, y0, x0 + tw + pad_x * 2, y0 + th + pad_y * 2],
        radius=4,
        fill=(14, 16, 22, 165),
    )
    draw.text((x0 + pad_x, y0 + pad_y - 2), label, fill=(244, 239, 230, 235), font=font)
    return np.array(pil)


def main() -> None:
    pw, ph = 960, 540
    p1 = _add_label(np.clip(render_hero_s10(pw, ph) * 255.0, 0, 255).astype(np.uint8), "ACT II · S10 — Lantern Station Downpour")
    p2 = _add_label(np.clip(render_hero_s14(pw, ph) * 255.0, 0, 255).astype(np.uint8), "ACT III · S14 — Macro Rainy Window & Reflection")
    p3 = _add_label(np.clip(render_hero_s24(pw, ph) * 255.0, 0, 255).astype(np.uint8), "ACT V · S24 — Flooded Mirror-Sea & Cloud Cathedrals")
    p4 = _add_label(np.clip(render_hero_s33(pw, ph) * 255.0, 0, 255).astype(np.uint8), "ACT VI · S33 — Golden Hour Cloudbreak & Petals")

    canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
    canvas[0:540, 0:960] = p1
    canvas[0:540, 960:1920] = p2
    canvas[540:1080, 0:960] = p3
    canvas[540:1080, 960:1920] = p4

    # Crisp 2px dark hairline divider between the 4 panels
    canvas[539:541, :] = (18, 20, 26)
    canvas[:, 959:961] = (18, 20, 26)

    out_path = Path(__file__).resolve().parents[2] / "style_frame_check.png"
    Image.fromarray(canvas).save(out_path, optimize=True)
    print(f"Saved {out_path} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
