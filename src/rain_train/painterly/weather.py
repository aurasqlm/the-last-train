"""Cinematic anime rain, refractive glass droplets, broken puddle reflections, mirror-sea & wakes."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def render_macro_window_droplets(
    glass_bg: np.ndarray,
    time_s: float = 0.0,
    num_large: int = 34,
    num_med: int = 85,
    num_micro: int = 240,
    fog_strength: float = 0.16,
    seed: int = 14,
) -> np.ndarray:
    """Render realistic refractive transparent water droplets on a fogged train window.

    Features:
    - Subtle condensation mist on the glass, wiped clear along vertical gravity slide-trails.
    - Organic gravity-sagged meniscus shapes (wider bottom, tapered top), ZERO black outlines.
    - True inverted optical refraction of the background bokeh inside each clear water lens.
    - Soft lower external shadow, internal lower-rim caustic glow, and crisp upper-left specular pinpoint.
    """
    h, w = glass_bg.shape[:2]
    rng = np.random.default_rng(seed)

    # Clear background vs fogged/condensed glass with realistic perimeter condensation gradient
    yy_full, xx_full = np.mgrid[0:h, 0:w].astype(np.float32)
    nx_f = (xx_full / max(1.0, float(w)) - 0.5) * 2.0
    ny_f = (yy_full / max(1.0, float(h)) - 0.5) * 2.0
    edge_dist = np.clip(np.sqrt(nx_f * nx_f * 0.88 + ny_f * ny_f * 0.92), 0.0, 1.38)
    sill_fog = np.clip((yy_full / max(1.0, float(h)) - 0.68) / 0.28, 0.0, 1.0) * 0.52
    # Subtle organic breath-like condensation mottling
    mottle = cv2.resize(rng.standard_normal((max(6, h // 32), max(6, w // 32))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    mottle = np.clip(0.85 + 0.25 * mottle, 0.55, 1.25)
    fog_map = np.clip(fog_strength * (0.48 + 1.05 * (edge_dist ** 1.25) + sill_fog) * mottle, 0.0, 0.72)

    fogged = cv2.GaussianBlur(glass_bg, (0, 0), sigmaX=6.0, sigmaY=6.0)
    # Delicate translucent cool-blue condensation veil (subtle enough that wiped trails stay luminous!)
    mist_tint = np.array([0.16, 0.21, 0.30], dtype=np.float32)[None, None, :]
    fogged = np.clip(fogged * (1.0 - fog_map[..., None] * 0.24) + mist_tint * (fog_map[..., None] * 0.42), 0.0, 1.0)

    # Mask of where condensation has been wiped clear by sliding droplets (0 = fogged, 1 = clear)
    clear_mask = np.zeros((h, w), dtype=np.float32)

    # Generate organic droplets: (x, y, rx, ry, trail_len, lob_amp, lob_phase, skew)
    drops: List[Tuple[float, float, float, float, float, float, float, float]] = []
    for _ in range(num_micro):
        x = rng.uniform(0.05 * w, 0.95 * w)
        y = rng.uniform(0.02 * h, 0.91 * h)
        r = rng.uniform(1.0, 2.5)
        drops.append((x, y, r, r * rng.uniform(0.85, 1.30), 0.0, 0.0, 0.0, 0.0))

    for _ in range(num_med):
        x = rng.uniform(0.05 * w, 0.95 * w)
        y = rng.uniform(0.04 * h, 0.90 * h)
        rx = rng.uniform(3.0, 6.5)
        aspect = rng.uniform(0.85, 1.75)
        ry = rx * aspect
        trail = rng.uniform(h * 0.06, h * 0.22) if (aspect > 1.25 and rng.random() < 0.55) else 0.0
        drops.append((x, y, rx, ry, trail, float(rng.uniform(0.06, 0.20)), float(rng.uniform(0, 6.28)), float(rng.uniform(-0.14, 0.14))))

    for _ in range(num_large):
        x = rng.uniform(0.06 * w, 0.93 * w)
        y = rng.uniform(0.10 * h, 0.88 * h)
        rx = rng.uniform(6.5, 14.5)
        aspect = rng.uniform(1.15, 2.00)
        ry = rx * aspect
        trail = rng.uniform(h * 0.14, h * 0.42) if rng.random() < 0.80 else 0.0
        drops.append((x, y, rx, ry, trail, float(rng.uniform(0.08, 0.22)), float(rng.uniform(0, 6.28)), float(rng.uniform(-0.18, 0.18))))

    # Carve soft, luminous wiped-clean vertical wet trails through the pale condensation fog (NEVER dark lines!)
    trail_sheen = np.zeros((h, w), dtype=np.float32)
    for (x, y, rx, ry, trail, _la, _lp, _sk) in drops:
        if trail <= 2.0:
            continue
        x0 = max(0, int(x - rx * 1.10))
        x1 = min(w, int(x + rx * 1.10) + 1)
        y0 = max(0, int(y - trail))
        y1 = min(h, int(y) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        wobble = np.sin((yy - y) * 0.042 + x * 0.13) * (rx * 0.24) + np.cos((yy - y) * 0.019 + y * 0.07) * (rx * 0.14)
        dx = np.abs(xx - (x + wobble)) / max(1.4, rx * 0.54)
        fade_top = np.clip((yy - y0) / max(14.0, trail * 0.42), 0.0, 1.0)
        tmask = np.clip((1.0 - dx * dx) * 1.35, 0.0, 1.0) * fade_top
        clear_mask[y0:y1, x0:x1] = np.maximum(clear_mask[y0:y1, x0:x1], tmask)
        # Soft luminous wet-glass sheen along the wiped trail so it reads as clear glistening water
        center_sheen = np.exp(-((dx / 0.65) ** 2)) * fade_top * 0.065
        edge_glint = np.exp(-((dx - 0.85) / 0.24) ** 2) * fade_top * 0.055
        trail_sheen[y0:y1, x0:x1] = np.maximum(trail_sheen[y0:y1, x0:x1], center_sheen + edge_glint)

    clear_mask = cv2.GaussianBlur(clear_mask, (0, 0), sigmaX=1.1, sigmaY=1.1)
    # Wiped glass is slightly brighter/clearer than fogged glass (never dark!)
    clear_glass = np.clip(glass_bg * 1.10 + 0.018, 0.0, 1.0)
    out = fogged * (1.0 - clear_mask[..., None]) + clear_glass * clear_mask[..., None]
    out = np.clip(out + trail_sheen[..., None] * np.array([0.80, 0.88, 0.98], dtype=np.float32), 0.0, 1.0)

    # Render each luminous, transparent refractive water droplet (ZERO dark outlines!)
    for (cx, cy, rx, ry, _trail, lob_amp, lob_phase, skew) in drops:
        pad = int(max(rx, ry) * 0.72) + 3
        x0 = max(0, int(cx - rx - pad))
        x1 = min(w, int(cx + rx + pad) + 1)
        y0 = max(0, int(cy - ry - pad))
        y1 = min(h, int(cy + ry + pad) + 1)
        if x1 <= x0 or y1 <= y0:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        ny = (yy - cy) / max(1.0, ry)
        sag_x = rx * (1.0 + 0.24 * np.clip(ny, -1.0, 1.0))
        nx = (xx - (cx + skew * rx * ny)) / np.maximum(1.0, sag_x)
        ang = np.arctan2(ny, nx)
        lobe = (
            1.0
            + lob_amp * np.sin(2.0 * ang + lob_phase)
            + 0.50 * lob_amp * np.cos(3.0 * ang - lob_phase * 1.3)
            + 0.22 * lob_amp * np.sin(5.0 * ang + lob_phase * 0.7)
        )
        r = np.sqrt(nx * nx + ny * ny) / np.maximum(0.68, lobe)

        # Clean antialiased droplet meniscus
        edge_w = max(0.07, 1.3 / max(rx, ry))
        mask = np.clip((1.0 - r) / edge_w, 0.0, 1.0)
        if np.max(mask) < 1e-3:
            continue

        # True spherical lens normal-map optical refraction of the composited world (bokeh + reflection)
        r_clamped = np.clip(r, 0.0, 0.995)
        nz = np.sqrt(np.maximum(0.08, 1.0 - r_clamped * r_clamped))
        # Spherical Snell-like displacement: stronger magnification & edge fisheye warp near meniscus rim
        lens_mag = 2.65 if rx >= 6.0 else 1.85
        refr_map_x = np.clip(cx - (nx * rx * lens_mag) / (nz + 0.36), 0.0, float(w - 1)).astype(np.float32)
        refr_map_y = np.clip(cy - (ny * ry * lens_mag) / (nz + 0.36), 0.0, float(h - 1)).astype(np.float32)
        refr_col = cv2.remap(glass_bg, refr_map_x, refr_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # Luminous transparent water lens (clear optical transmission + subtle Fresnel depth)
        drop_col = np.clip(refr_col * 1.42 + np.array([0.03, 0.04, 0.06], dtype=np.float32), 0.0, 1.0)

        # Delicate internal lower-rim caustic glow (focused refracted light at bottom meniscus)
        caustic = np.exp(-((r - 0.78) / 0.16) ** 2) * np.clip(ny * 1.15 + 0.15, 0.0, 1.0)
        caustic_col = np.clip(refr_col * 1.80 + np.array([0.22, 0.20, 0.16], dtype=np.float32), 0.0, 1.0)
        drop_col = np.clip(drop_col + caustic[..., None] * caustic_col * 0.48, 0.0, 1.0)

        # Soft upper-left Fresnel glass rim catchlight (no uniform bright outline!)
        rim_glint = np.exp(-((r - 0.91) / 0.08) ** 2) * np.clip(-0.55 * nx - 0.55 * ny + 0.35, 0.0, 1.0) * 0.22
        drop_col = np.clip(drop_col + rim_glint[..., None] * np.array([0.82, 0.90, 0.98], dtype=np.float32), 0.0, 1.0)

        # Tiny crisp white specular pinpoint on upper-left
        sp_d2 = ((nx + 0.30) / 0.13) ** 2 + ((ny + 0.34) / 0.12) ** 2
        spec = np.exp(-sp_d2) * (0.90 if rx >= 4.0 else 0.54)
        drop_col = np.clip(drop_col + spec[..., None] * np.array([1.0, 0.98, 0.94], dtype=np.float32), 0.0, 1.0)

        a = (mask * 0.90)[..., None]
        out[y0:y1, x0:x1] = out[y0:y1, x0:x1] * (1.0 - a) + drop_col * a

    return np.clip(out, 0.0, 1.0)


def render_slanted_rain(
    img: np.ndarray,
    time_s: float = 0.0,
    density: float = 1.0,
    angle_deg: float = 14.0,
    warm_lights: Optional[Sequence[Tuple[float, float, float, Tuple[float, float, float]]]] = None,
    seed: int = 10,
    occlusion_mask: Optional[np.ndarray] = None,
    splash_segments: Optional[Sequence[Tuple[float, float, float, float, float]]] = None,
) -> np.ndarray:
    """Render 4-depth-layer cinematic anime rain with wind-gust angle jitter, lantern cones & surface splash halos.

    `splash_segments`: optional list of `(x0_f, y0_f, x1_f, y1_f, strength)` normalized segments where
    rebounding raindrop splash halos and misty spray bounce off roofs, umbrella canopy, eaves, and platform.
    """
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    base_rad = np.radians(angle_deg)
    dx_base, dy_base = -np.sin(base_rad), np.cos(base_rad)

    # Precompute warm light illumination map so rain streaks glow golden inside lantern/headlight cones
    light_map = np.zeros((h, w, 3), dtype=np.float32)
    if warm_lights:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for (lx, ly, lr, lcol) in warm_lights:
            d2 = ((xx - lx * w) / (lr * w)) ** 2 + ((yy - ly * h) / (lr * h * 1.45)) ** 2
            glow = np.exp(-d2 * 1.35)
            light_map += glow[..., None] * np.array(lcol, dtype=np.float32)[None, None, :]
        light_map = np.clip(light_map, 0.0, 1.25)

    out = img.copy()

    # 4 depth layers: (count_base, len_min, len_max, thick, base_alpha, blur_sigma, occ_weight, angle_jitter_deg)
    layers = [
        (int(360 * density), 7, 24, 1, 0.12, 0.9, 1.00, 3.6),    # Fine distant misty rain sheet
        (int(240 * density), 14, 44, 1, 0.21, 0.0, 1.00, 2.8),   # Varied mid-background rain
        (int(155 * density), 26, 72, 1, 0.33, 0.0, 1.00, 2.2),   # Crisp midground streaks
        (int(40 * density),  54, 118, 2, 0.22, 2.2, 1.00, 1.8),  # Sparse fast foreground bokeh streaks (100% roof-masked!)
    ]

    for layer_idx, (count, lmin, lmax, thick, base_a, blur_s, occ_w, ang_jit) in enumerate(layers):
        layer_mask = np.zeros((h, w), dtype=np.float32)
        for _ in range(count):
            x0 = rng.uniform(-0.08 * w, 1.08 * w)
            y0 = rng.uniform(-0.10 * h, 1.05 * h)
            speed = (180.0 + layer_idx * 140.0) * rng.uniform(0.75, 1.28)
            x0 = (x0 + dx_base * speed * time_s) % (w * 1.15) - 0.075 * w
            y0 = (y0 + dy_base * speed * time_s) % (h * 1.15) - 0.075 * h

            # Individual wind-gust angle jitter & varied length/opacity (never mechanical parallel lines!)
            r_rad = base_rad + np.radians(rng.normal(0.0, ang_jit))
            dx_u, dy_u = -np.sin(r_rad), np.cos(r_rad)
            length = rng.uniform(lmin, lmax) * (h / 540.0)
            x1 = x0 + dx_u * length
            y1 = y0 + dy_u * length
            val = float(rng.uniform(0.24, 1.0))
            cv2.line(
                layer_mask,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                val,
                thick,
                cv2.LINE_AA,
            )

        if blur_s > 0.0:
            layer_mask = cv2.GaussianBlur(layer_mask, (0, 0), sigmaX=blur_s, sigmaY=blur_s)

        if occlusion_mask is not None:
            layer_mask *= np.clip(1.0 - occlusion_mask * occ_w, 0.0, 1.0)

        cool_col = np.array([0.74, 0.82, 0.90], dtype=np.float32)[None, None, :]
        rain_rgb = cool_col + light_map * 0.98
        rain_a = np.clip(layer_mask * (base_a + 0.38 * np.mean(light_map, axis=-1)), 0.0, 0.85)[..., None]
        out = np.clip(out + rain_rgb * rain_a * (1.0 - out * 0.45), 0.0, 1.0)

    # Misty rain-splash halos, tiny white splash crowns & rebounding micro-droplets on umbrella, eaves, train roof & platform
    if splash_segments:
        splash_mist = np.zeros((h, w), dtype=np.float32)
        splash_dots = np.zeros((h, w), dtype=np.float32)
        for (sx0, sy0, sx1, sy1, s_str) in splash_segments:
            seg_len = float(np.hypot((sx1 - sx0) * w, (sy1 - sy0) * h))
            n_sp = max(8, int(seg_len / 12.0))
            for _ in range(n_sp):
                t = float(rng.uniform(0.0, 1.0))
                px = (sx0 * (1.0 - t) + sx1 * t) * w
                py = (sy0 * (1.0 - t) + sy1 * t) * h
                # Soft rebounding mist halo just above the impact point
                mr = max(3, int(rng.uniform(4.5, 12.5)))
                cv2.ellipse(
                    splash_mist,
                    (int(px), int(py - mr * 0.35)),
                    (mr, max(2, int(mr * 0.55))),
                    0, 0, 360,
                    float(rng.uniform(0.26, 0.62) * s_str),
                    -1, cv2.LINE_AA,
                )
                # Tiny white V-crown & rebounding splash beads arcing upward
                crown_h = int(rng.uniform(3.0, 7.0))
                crown_w = int(rng.uniform(3.0, 6.0))
                cv2.line(splash_dots, (int(px), int(py)), (int(px - crown_w), int(py - crown_h)), float(0.65 * s_str), 1, cv2.LINE_AA)
                cv2.line(splash_dots, (int(px), int(py)), (int(px + crown_w), int(py - crown_h)), float(0.65 * s_str), 1, cv2.LINE_AA)
                for _b in range(2):
                    bx = int(px + rng.uniform(-8.0, 8.0))
                    by = int(py - rng.uniform(3.0, 13.0))
                    cv2.circle(splash_dots, (bx, by), 1, float(rng.uniform(0.50, 0.98) * s_str), -1, cv2.LINE_AA)
        splash_mist = cv2.GaussianBlur(splash_mist, (0, 0), sigmaX=4.5, sigmaY=3.0)
        splash_dots = cv2.GaussianBlur(splash_dots, (0, 0), sigmaX=0.7, sigmaY=0.7)
        sp_field = np.clip(splash_mist * 0.58 + splash_dots * 0.78, 0.0, 0.92)
        sp_col = np.array([0.84, 0.88, 0.94], dtype=np.float32)[None, None, :] + light_map * 0.65
        out = np.clip(out + sp_col * sp_field[..., None] * (1.0 - out * 0.38), 0.0, 1.0)

    return out


def apply_puddle_reflection(
    img: np.ndarray,
    y_start_frac: float = 0.75,
    time_s: float = 0.0,
    ripple_centers: Optional[Sequence[Tuple[float, float, float]]] = None,
    darkness: float = 0.72,
) -> np.ndarray:
    """Render realistic broken wet-stone platform reflections with organic painterly water ripples."""
    h, w = img.shape[:2]
    y0 = int(np.clip(y_start_frac, 0.05, 0.95) * h)
    ref_h = h - y0
    if ref_h <= 2:
        return img

    src_h = min(y0, ref_h)
    flipped = img[y0 - src_h : y0, :][::-1, :, :].copy()
    if src_h < ref_h:
        pad = np.repeat(flipped[-1:, :, :], ref_h - src_h, axis=0)
        flipped = np.concatenate([flipped, pad], axis=0)

    rng = np.random.default_rng(77)
    puddle_noise = cv2.resize(rng.standard_normal((max(4, ref_h // 10), max(8, w // 24))).astype(np.float32), (w, ref_h), interpolation=cv2.INTER_CUBIC)
    stone_micro = cv2.resize(rng.standard_normal((max(8, ref_h // 3), max(16, w // 6))).astype(np.float32), (w, ref_h), interpolation=cv2.INTER_LINEAR)

    refl = cv2.GaussianBlur(flipped, (0, 0), sigmaX=1.8, sigmaY=4.2)

    yy, xx = np.mgrid[0:ref_h, 0:w].astype(np.float32)
    disp_x = stone_micro * 2.4
    disp_y = stone_micro * 1.5

    # Physically distort the reflection around organic raindrop ripple rings
    ripple_highlight = np.zeros((ref_h, w), dtype=np.float32)
    if ripple_centers:
        for idx, (rx_f, ry_f, phase) in enumerate(ripple_centers):
            cx = rx_f * w
            cy = (ry_f - y_start_frac) * h
            if cy < 0 or cy >= ref_h:
                continue
            age = (time_s + phase) % 1.8
            fade = (1.0 - age / 1.8) ** 1.15
            max_r = 16.0 + age * 28.0
            dx_r = (xx - cx) / max(1.0, max_r)
            dy_r = (yy - cy) / max(1.0, max_r * 0.27)
            d = np.sqrt(dx_r * dx_r + dy_r * dy_r)
            ang = np.arctan2(dy_r, dx_r)
            # Organic broken-ring angular modulation (NEVER sterile unbroken vector ellipses!)
            breakup = np.clip(0.45 + 0.55 * np.sin(ang * 3.0 + idx * 1.7) * np.cos(ang * 5.0 - idx * 2.3) + 0.35 * stone_micro, 0.0, 1.0)
            wave = np.sin((d - 1.0) * 14.0) * np.exp(-((d - 0.85) / 0.38) ** 2) * fade
            disp_x += wave * (dx_r / (d + 0.1)) * 4.5
            disp_y += wave * (dy_r / (d + 0.1)) * 2.4
            ring_hi = np.exp(-((d - 1.0) / 0.08) ** 2) * breakup * fade * 0.32
            ring_hi2 = np.exp(-((d - 0.64) / 0.09) ** 2) * breakup * fade * 0.18
            ripple_highlight += ring_hi + ring_hi2

    map_x = np.clip(xx + disp_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + disp_y, 0, ref_h - 1).astype(np.float32)
    refl = cv2.remap(refl, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    wet_stone = np.array([0.07, 0.09, 0.12], dtype=np.float32)[None, None, :]
    puddle_mask = np.clip(0.55 + 0.35 * puddle_noise, 0.18, 0.92)[..., None]
    lum = (0.299 * refl[..., 0] + 0.587 * refl[..., 1] + 0.114 * refl[..., 2])[..., None]
    reflectivity = np.clip(puddle_mask * (0.45 + 0.78 * lum) * darkness, 0.0, 0.94)

    out_strip = wet_stone * (1.0 - reflectivity) + refl * reflectivity
    # Tint broken ripple highlights with the underlying warm reflection color
    r_tint = np.clip(refl * 1.4 + np.array([0.35, 0.30, 0.24], dtype=np.float32), 0.0, 1.0)
    out_strip = np.clip(out_strip + ripple_highlight[..., None] * r_tint, 0.0, 1.0)

    out = img.copy()
    out[y0:, :] = out_strip
    return out


def apply_sea_reflection(
    img: np.ndarray,
    horizon_y_frac: float = 0.56,
    time_s: float = 0.0,
    ripple_strength: float = 0.65,
    darkness: float = 0.88,
    shimmer_color: Optional[Tuple[float, float, float]] = None,
    shimmer_x_frac: float = 0.50,
    shimmer_strength: float = 0.75,
    ripple_points: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> np.ndarray:
    """Render glassy Ghibli mirror-sea reflection with perspective-foreshortened ripples & bokeh glints."""
    h, w = img.shape[:2]
    hy = int(np.clip(horizon_y_frac, 0.10, 0.90) * h)
    sea_h = h - hy
    if sea_h <= 2:
        return img

    src_h = min(hy, sea_h)
    flipped = img[hy - src_h : hy, :][::-1, :, :].copy()
    if src_h < sea_h:
        pad = np.repeat(flipped[-1:, :, :], sea_h - src_h, axis=0)
        flipped = np.concatenate([flipped, pad], axis=0)

    yy, xx = np.mgrid[0:sea_h, 0:w].astype(np.float32)
    depth = yy / max(1.0, float(sea_h))  # 0 at horizon, 1 at bottom foreground

    # Perspective-foreshortened horizontal ripple bands (tight near horizon, wider/softer in foreground)
    z = 1.0 / (depth + 0.075)
    rng = np.random.default_rng(24)
    w_noise = cv2.resize(rng.standard_normal((max(8, sea_h // 5), max(8, w // 24))).astype(np.float32), (w, sea_h), interpolation=cv2.INTER_CUBIC)
    rip = (
        np.sin(z * 4.4 + time_s * 1.6 + w_noise * 2.1) * 0.52
        + np.sin(z * 9.8 - time_s * 2.2 + xx * 0.018) * 0.33
        + np.sin(z * 18.0 + xx * 0.032 + w_noise) * 0.15
    )

    disp_x = rip * ripple_strength * (2.6 + 11.5 * depth)
    disp_y = rip * ripple_strength * (0.9 + 4.2 * depth)
    map_x = np.clip(xx + disp_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + disp_y, 0, sea_h - 1).astype(np.float32)
    sea = cv2.remap(flipped, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Subtle horizontal water-band specular modulation (breaks mechanical mirror flip!)
    band_sheen = np.clip(rip, 0.0, 1.0) * (0.045 + 0.055 * depth) * ripple_strength
    sea = np.clip(sea + band_sheen[..., None], 0.0, 1.0)

    # Subtle Fresnel depth gradient (slightly deeper/cooler toward foreground)
    fresnel = darkness * (1.0 - 0.16 * (depth ** 1.3))
    sea = sea * fresnel[..., None]

    # Delicate concentric water ripples around submerged utility poles & floating lanterns
    if ripple_points:
        ring_buf = np.zeros((sea_h, w), dtype=np.float32)
        for (rx_f, ry_f, rad_px) in ripple_points:
            cx = rx_f * w
            cy = (ry_f - horizon_y_frac) * h
            if cy < -4 or cy >= sea_h + 4:
                continue
            for scale_r, amp_r in ((0.55, 0.24), (1.0, 0.18), (1.55, 0.10)):
                rx_e = max(3, int(rad_px * scale_r))
                ry_e = max(1, int(rad_px * scale_r * 0.22))
                cv2.ellipse(ring_buf, (int(cx), int(max(0, cy))), (rx_e, ry_e), 0, 0, 360, amp_r, 1, cv2.LINE_AA)
        ring_buf = cv2.GaussianBlur(ring_buf, (0, 0), sigmaX=1.0, sigmaY=0.7)
        sea = np.clip(sea + ring_buf[..., None] * np.array([0.82, 0.88, 0.96], dtype=np.float32), 0.0, 1.0)

    # Delicate perspective-foreshortened golden/silver bokeh glints (NEVER harsh white slashes!)
    if shimmer_color is not None and shimmer_strength > 0.0:
        s_rng = np.random.default_rng(333)
        glint_buf = np.zeros((sea_h, w), dtype=np.float32)
        num_glints = 280
        for _ in range(num_glints):
            d_f = float(s_rng.uniform(0.03, 0.96) ** 1.15)
            gy = int(d_f * (sea_h - 2))
            # Spread widens with perspective depth
            spread = (0.04 + 0.24 * d_f) * w
            gx = int(shimmer_x_frac * w + s_rng.normal(0.0, spread))
            if gx < 2 or gx >= w - 2:
                continue
            rx = max(2, int((2.0 + 11.0 * d_f) * s_rng.uniform(0.5, 1.3)))
            ry = max(1, int((0.8 + 2.4 * d_f) * s_rng.uniform(0.6, 1.2)))
            amp = float(s_rng.uniform(0.28, 0.92) * (1.0 - 0.25 * d_f))
            cv2.ellipse(glint_buf, (gx, gy), (rx, ry), 0, 0, 360, amp, -1, cv2.LINE_AA)

        glint_buf = cv2.GaussianBlur(glint_buf, (0, 0), sigmaX=1.4, sigmaY=0.9)
        sc = np.array(shimmer_color, dtype=np.float32)[None, None, :]
        sea = 1.0 - (1.0 - sea) * (1.0 - np.clip(glint_buf[..., None] * sc * shimmer_strength, 0.0, 0.92))

    # Subtle horizon waterline gleam
    line_glow = np.exp(-((yy / 3.5) ** 2)) * 0.18
    sea = np.clip(sea + line_glow[..., None], 0.0, 1.0)

    out = img.copy()
    out[hy:, :] = np.clip(sea, 0.0, 1.0)
    return out


def render_sea_wake(
    img: np.ndarray,
    bow_xy: Tuple[float, float],
    tail_xy: Tuple[float, float],
    wake_width_frac: float = 0.055,
    foam_color: Tuple[float, float, float] = (0.88, 0.94, 0.99),
    opacity: float = 0.80,
    seed: int = 24,
) -> np.ndarray:
    """Render a lush V-shaped white-foam & silver-blue water wake + fine mist spray fanning from train wheels."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    wake = np.zeros((h, w), dtype=np.float32)
    spray = np.zeros((h, w), dtype=np.float32)

    bx, by = bow_xy[0] * w, bow_xy[1] * h
    tx, ty = tail_xy[0] * w, tail_xy[1] * h
    steps = 90
    for i in range(steps):
        t = i / max(1, steps - 1)
        px = bx * (1.0 - t) + tx * t
        py = by * (1.0 - t) + ty * t
        # Primary waterline frothy foam directly under the wheels
        rx0 = max(2, int((5.0 + 10.0 * t) * rng.uniform(0.7, 1.3)))
        ry0 = max(1, int((1.5 + 2.2 * t) * rng.uniform(0.7, 1.2)))
        cv2.ellipse(wake, (int(px), int(py)), (rx0, ry0), 0, 0, 360, float((1.0 - 0.35 * t) * 0.85), -1, cv2.LINE_AA)

        # V-shaped fanning wake arms (primarily below waterline into foreground sea)
        spread = (0.14 + 0.95 * (t ** 1.05)) * wake_width_frac * h
        for arm_scale in (0.55, 1.0):
            wy = py + spread * arm_scale * rng.uniform(0.82, 1.15)
            rx = max(2, int((5.0 + 20.0 * t) * rng.uniform(0.7, 1.3)))
            ry = max(1, int((1.4 + 3.6 * t) * rng.uniform(0.6, 1.2)))
            amp = float((1.0 - 0.45 * t) * rng.uniform(0.45, 0.92) * (1.0 if arm_scale == 1.0 else 0.65))
            cv2.ellipse(wake, (int(px), int(wy)), (rx, ry), 0, 0, 360, amp, -1, cv2.LINE_AA)

        # Fine mist spray arcing up slightly from the slicing wheels
        if rng.random() < 0.65:
            sx = int(px + rng.uniform(-6.0, 6.0))
            sy = int(py - rng.uniform(0.0, wake_width_frac * h * 0.42))
            cv2.circle(spray, (sx, sy), max(1, int(rng.uniform(1.0, 2.5))), float(rng.uniform(0.35, 0.85)), -1, cv2.LINE_AA)

    wake = cv2.GaussianBlur(wake, (0, 0), sigmaX=1.8, sigmaY=1.0)
    spray = cv2.GaussianBlur(spray, (0, 0), sigmaX=2.2, sigmaY=1.6)
    combined = np.clip(wake + spray * 0.65, 0.0, 1.0)
    fc = np.array(foam_color, dtype=np.float32)[None, None, :]
    a = np.clip(combined * opacity, 0.0, 0.92)[..., None]
    return np.clip(img * (1.0 - a) + fc * a, 0.0, 1.0)


def render_steam_plume(
    img: np.ndarray,
    center_xy: Tuple[float, float],
    radius_xy: Tuple[float, float],
    color: Tuple[float, float, float] = (0.86, 0.84, 0.80),
    opacity: float = 0.42,
    seed: int = 101,
) -> np.ndarray:
    """Render billowing multi-layered painterly locomotive steam clouds."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    cx, cy = center_xy[0] * w, center_xy[1] * h
    rx, ry = radius_xy[0] * w, radius_xy[1] * h

    plume = np.zeros((h, w), dtype=np.float32)
    for _ in range(18):
        px = int(cx + rng.normal(0.0, rx * 0.45))
        py = int(cy + rng.normal(0.0, ry * 0.40))
        prx = max(6, int(rx * rng.uniform(0.24, 0.52)))
        pry = max(5, int(ry * rng.uniform(0.26, 0.55)))
        amp = float(rng.uniform(0.25, 0.65))
        cv2.ellipse(plume, (px, py), (prx, pry), float(rng.uniform(-25, 25)), 0, 360, amp, -1, cv2.LINE_AA)

    plume = cv2.GaussianBlur(plume, (0, 0), sigmaX=max(3.0, rx * 0.14), sigmaY=max(3.0, ry * 0.14))
    plume = np.clip(plume / (np.max(plume) + 1e-6), 0.0, 1.0) * opacity
    col = np.array(color, dtype=np.float32)[None, None, :]
    return np.clip(img * (1.0 - plume[..., None]) + col * plume[..., None], 0.0, 1.0)
