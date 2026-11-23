"""35mm optical halation/bloom, volumetric god-rays, aperture bokeh, atmospheric haze, filmic grade."""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


def apply_halation_bloom(
    img: np.ndarray,
    threshold: float = 0.65,
    tight_strength: float = 0.45,
    wide_strength: float = 0.30,
) -> np.ndarray:
    """Simulate 35mm cinema film optical halation (warm reddish-amber wide scatter + crisp tight bloom)."""
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    mask = np.clip((lum - threshold) / (1.0 - threshold + 1e-5), 0.0, 1.0) ** 1.5
    bright = img * mask[..., None]

    # Tight optical glow
    tight = cv2.GaussianBlur(bright, (0, 0), sigmaX=7.0, sigmaY=7.0)
    # Wide 35mm red-amber halation scatter
    wide = cv2.GaussianBlur(bright, (0, 0), sigmaX=28.0, sigmaY=28.0)
    halation_tint = np.array([1.18, 0.94, 0.78], dtype=np.float32)[None, None, :]
    bloom = tight * tight_strength + wide * wide_strength * halation_tint

    # Screen blend to preserve highlights without harsh clipping
    return np.clip(1.0 - (1.0 - img) * (1.0 - np.clip(bloom, 0.0, 0.92)), 0.0, 1.0)


def render_god_rays(
    w: int,
    h: int,
    origin_xy: Tuple[float, float] = (0.50, 0.22),
    color: Tuple[float, float, float] = (1.0, 0.92, 0.72),
    num_rays: int = 4,
    intensity: float = 0.65,
    spread_deg: Tuple[float, float] = (34.0, 146.0),
    seed: int = 33,
    cloud_alpha: np.ndarray | None = None,
) -> np.ndarray:
    """Render 3-4 irregular, wide, softly feathered volumetric crepuscular light shafts (never geometric spokes)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ox, oy = origin_xy[0] * w, origin_xy[1] * h
    dx = xx - ox
    dy = yy - oy
    dist = np.sqrt(dx * dx + dy * dy) / max(w, h)
    angle = np.degrees(np.arctan2(dy, dx))

    # 3-4 irregular, organic wide volumetric light shafts (asymmetric spacing, wide soft angular falloff)
    n_shafts = max(3, min(5, num_rays))
    lo, hi = spread_deg
    # Irregular organic shaft angles (avoiding uniform mechanical spacing)
    shaft_fracs = np.sort(rng.uniform(0.08, 0.92, size=n_shafts))
    rays = np.zeros((h, w), dtype=np.float32)
    for idx_r, frac in enumerate(shaft_fracs):
        a0 = float(lo + (hi - lo) * frac)
        # Wide, sweeping volumetric beam widths (8 to 16 degrees)
        w_shaft = float(rng.uniform(8.5, 16.5))
        amp = float(rng.uniform(0.45, 0.95)) if (idx_r != 1) else float(rng.uniform(0.75, 1.05))
        diff = np.abs(((angle - a0 + 180.0) % 360.0) - 180.0)
        rays += np.exp(-0.5 * (diff / w_shaft) ** 2) * amp * 0.58

    # Soft broad atmospheric veil across the cloudbreak opening
    center_a = 0.5 * (lo + hi)
    diff_c = np.abs(((angle - center_a + 180.0) % 360.0) - 180.0)
    rays += np.exp(-0.5 * (diff_c / 32.0) ** 2) * 0.28

    # Radial cloud-gap occlusion marching with generous light-wrap bleeding over cloud edges
    if cloud_alpha is not None:
        gap = 1.0 - np.clip(cloud_alpha, 0.0, 1.0)
        occ_shafts = np.zeros((h, w), dtype=np.float32)
        steps = 12
        for s_i in range(steps):
            frac = 0.14 + 0.86 * (s_i / max(1, steps - 1))
            sx = np.clip(ox + dx * frac, 0, w - 1).astype(np.float32)
            sy = np.clip(oy + dy * frac, 0, h - 1).astype(np.float32)
            occ_shafts += cv2.remap(gap, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        occ_shafts /= float(steps)
        occ_shafts = cv2.GaussianBlur(occ_shafts, (0, 0), sigmaX=14.0, sigmaY=14.0)
        rays = rays * (0.38 + 0.92 * occ_shafts)

    # Smooth atmospheric radial falloff & warm sunlit cloudbreak glow
    radial = np.exp(-dist * 1.15) * (1.0 - np.exp(-dist * 9.0))
    core_glow = np.exp(-((dist / 0.22) ** 2)) * 0.36
    field = np.clip((rays * radial * 0.95 + core_glow) * intensity, 0.0, 0.88)
    # Heavy volumetric Gaussian feathering so shafts read as luminous atmospheric haze, never sharp spokes
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=14.0, sigmaY=14.0)

    col = np.array(color, dtype=np.float32)[None, None, :]
    return field[..., None] * col


def render_aperture_bokeh(
    w: int,
    h: int,
    palette: List[Tuple[float, float, float]],
    num_discs: int = 65,
    y_range: Tuple[float, float] = (0.42, 0.95),
    radius_range: Tuple[float, float] = (0.012, 0.052),
    seed: int = 14,
) -> np.ndarray:
    """Render crisp-edged cinema lens circular aperture bokeh discs with bright spherical-aberration rims."""
    rng = np.random.default_rng(seed)
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    for _ in range(num_discs):
        cx = int(rng.uniform(0.02 * w, 0.98 * w))
        cy = int(rng.uniform(y_range[0] * h, y_range[1] * h))
        r = max(4, int(rng.uniform(radius_range[0] * h, radius_range[1] * h)))
        col = np.array(palette[int(rng.integers(0, len(palette)))], dtype=np.float32)
        amp = float(rng.uniform(0.28, 0.85))

        x0, x1 = max(0, cx - r - 3), min(w, cx + r + 4)
        y0, y1 = max(0, cy - r - 3), min(h, cy + r + 4)
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / float(r)
        # Soft-edged disc with bright outer ring (classic vintage anime lens bokeh)
        disc = np.clip((1.04 - d) / 0.14, 0.0, 1.0)
        ring = np.exp(-((d - 0.86) / 0.12) ** 2) * 0.42
        profile = (disc * 0.65 + ring * disc) * amp

        patch = canvas[y0:y1, x0:x1]
        # Additive-screen blend for luminous overlapping bokeh
        add = profile[..., None] * col[None, None, :]
        canvas[y0:y1, x0:x1] = 1.0 - (1.0 - patch) * (1.0 - np.clip(add, 0.0, 0.95))

    return cv2.GaussianBlur(canvas, (0, 0), sigmaX=1.4, sigmaY=1.4)


def apply_atmospheric_haze(
    img: np.ndarray,
    y_center_frac: float,
    color: Tuple[float, float, float],
    strength: float = 0.35,
    band_height: float = 0.12,
) -> np.ndarray:
    """Apply soft horizontal atmospheric mist/haze band around horizon or valley."""
    h, w = img.shape[:2]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    band = np.exp(-0.5 * ((yy - y_center_frac) / max(0.01, band_height)) ** 2) * strength
    col = np.array(color, dtype=np.float32)[None, None, :]
    return np.clip(img * (1.0 - band[..., None]) + col * band[..., None], 0.0, 1.0)


def apply_filmic_grade(
    img: np.ndarray,
    contrast: float = 1.06,
    saturation: float = 1.08,
    shadow_lift: float = 0.012,
) -> np.ndarray:
    """Apply subtle cinema S-curve contrast, saturation, and lifted filmic shadows."""
    # Gentle S-curve around 0.5
    graded = np.clip((img - 0.5) * contrast + 0.5, 0.0, 1.0)
    # Saturation boost
    lum = (0.299 * graded[..., 0] + 0.587 * graded[..., 1] + 0.114 * graded[..., 2])[..., None]
    graded = np.clip(lum + (graded - lum) * saturation, 0.0, 1.0)
    # Subtle warm filmic shadow lift
    graded = shadow_lift + graded * (1.0 - shadow_lift)
    return np.clip(graded, 0.0, 1.0)


def apply_vignette(img: np.ndarray, strength: float = 0.26) -> np.ndarray:
    """Apply gentle 35mm lens falloff vignette."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / w - 0.5) * 2.0
    ny = (yy / h - 0.5) * 2.0
    r2 = nx * nx + ny * ny
    vig = 1.0 - strength * np.clip((r2 - 0.35) / 1.45, 0.0, 1.0) ** 1.4
    return np.clip(img * vig[..., None], 0.0, 1.0)
