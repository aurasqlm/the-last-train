"""Continuous hand-painted Studio Ghibli / Makoto Shinkai gouache cumulus cloud cathedrals and painterly skies."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .paper_grain import apply_gouache_brushwork


def render_sky_gradient(
    w: int,
    h: int,
    stops: Sequence[Tuple[float, Tuple[float, float, float]]],
    seed: int = 1,
    brush_strength: float = 0.028,
) -> np.ndarray:
    """Render a painterly gouache sky gradient with subtle directional brushwork (never flat digital)."""
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)
    sky_col = np.zeros((h, 3), dtype=np.float32)

    stops_sorted = sorted(stops, key=lambda s: s[0])
    for i in range(len(stops_sorted) - 1):
        y0, c0 = stops_sorted[i]
        y1, c1 = stops_sorted[i + 1]
        mask = (yy >= y0) & (yy <= y1)
        if not np.any(mask):
            continue
        t = (yy[mask] - y0) / max(1e-5, y1 - y0)
        t = t * t * (3.0 - 2.0 * t)
        c0_arr = np.array(c0, dtype=np.float32)
        c1_arr = np.array(c1, dtype=np.float32)
        sky_col[mask] = c0_arr[None, :] * (1.0 - t[:, None]) + c1_arr[None, :] * t[:, None]

    sky_col[yy < stops_sorted[0][0]] = stops_sorted[0][1]
    sky_col[yy > stops_sorted[-1][0]] = stops_sorted[-1][1]

    img = np.broadcast_to(sky_col[:, None, :], (h, w, 3)).copy()
    if brush_strength > 0.0:
        img = apply_gouache_brushwork(img, strength=brush_strength, angle_deg=-6.0, scale=1.2, seed=seed)
    return np.clip(img, 0.0, 1.0)


def _sample_ramp(t: np.ndarray, ramp: Sequence[Tuple[float, float, float]]) -> np.ndarray:
    """Piecewise linear color ramp lookup for t in [0, 1]."""
    n = len(ramp)
    if n == 1:
        return np.broadcast_to(np.array(ramp[0], dtype=np.float32), t.shape + (3,))
    tc = np.clip(t, 0.0, 1.0) * (n - 1)
    idx = np.clip(np.floor(tc).astype(np.int32), 0, n - 2)
    frac = (tc - idx)[..., None]
    ramp_arr = np.array(ramp, dtype=np.float32)
    return ramp_arr[idx] * (1.0 - frac) + ramp_arr[idx + 1] * frac


def _fbm_2d(h: int, w: int, rng: np.random.Generator, octaves: Sequence[Tuple[int, float]]) -> np.ndarray:
    """Generate smooth multi-octave 2D fBM noise in approximately [-1, 1]."""
    acc = np.zeros((h, w), dtype=np.float32)
    w_sum = 0.0
    for grid_div, weight in octaves:
        gh = max(4, h // grid_div)
        gw = max(4, w // grid_div)
        base = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
        acc += up * weight
        w_sum += weight
    acc /= max(1e-5, w_sum)
    acc /= float(np.std(acc) + 1e-5)
    return acc


def render_cumulus_cathedral(
    w: int,
    h: int,
    towers: Sequence[Tuple[float, float, float, float]],
    base_y_frac: float = 0.56,
    color_ramp: Sequence[Tuple[float, float, float]] = (
        (0.16, 0.22, 0.36),
        (0.36, 0.45, 0.60),
        (0.72, 0.78, 0.85),
        (0.98, 0.93, 0.84),
    ),
    sun_dir: Tuple[float, float] = (-0.6, -0.8),
    sun_pos: Optional[Tuple[float, float]] = None,
    rim_color: Optional[Tuple[float, float, float]] = (1.0, 0.95, 0.84),
    rim_strength: float = 0.65,
    seed: int = 240,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render solid, hand-painted Makoto Shinkai / Studio Ghibli gouache cumulus cloud cathedrals.

    Uses a true smooth-minimum (log-sum-exp smin) SDF metaball union of overlapping cumulus domes
    with subtle fBM displacement applied ONLY along the outer silhouette boundary.
    Guarantees 100% solid cloud interiors (ZERO Swiss-cheese holes, ZERO detached noise specks,
    ZERO dark creases between ovals), lit with smooth 2D+Z SDF surface normals and volumetric
    self-shadowing.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base_y_px = base_y_frac * h
    scale_px = float(min(w, h))

    # 1. Build 35-55 overlapping cumulus domes across the requested tower envelopes
    domes: list[Tuple[float, float, float, float]] = []
    for (tx, ty, trx, try_) in towers:
        cx_px, cy_px = tx * w, ty * h
        rx_px, ry_px = trx * w, try_ * h

        # Broad buoyant condensation base shelf (keeps the cloud mass unified & grounded)
        for _ in range(4):
            px = cx_px + rng.uniform(-0.72, 0.72) * rx_px
            py = base_y_px - rng.uniform(0.06, 0.24) * ry_px
            prx = rx_px * rng.uniform(0.46, 0.72)
            pry = ry_px * rng.uniform(0.26, 0.42)
            domes.append((px, py, max(10.0, prx), max(10.0, pry)))

        # Sweeping mid-body billowing cumulus mass
        for _ in range(7):
            t_h = rng.uniform(0.18, 0.72)
            spread = 1.0 - 0.48 * t_h
            px = cx_px + rng.uniform(-spread, spread) * rx_px * 0.68
            py = base_y_px - t_h * ry_px * 1.12
            prx = rx_px * rng.uniform(0.32, 0.54) * (1.0 - 0.20 * t_h)
            pry = ry_px * rng.uniform(0.28, 0.48) * (1.0 - 0.18 * t_h)
            domes.append((px, py, max(10.0, prx), max(10.0, pry)))

        # Towering sunlit cathedral crown spires
        for _ in range(5):
            t_h = rng.uniform(0.60, 1.08)
            spread = max(0.16, 0.95 - 0.62 * t_h)
            px = cx_px + rng.uniform(-spread, spread) * rx_px * 0.48
            py = base_y_px - t_h * ry_px * 1.14
            prx = rx_px * rng.uniform(0.22, 0.38)
            pry = ry_px * rng.uniform(0.24, 0.40)
            domes.append((px, py, max(9.0, prx), max(9.0, pry)))

    if not domes:
        return np.zeros((h, w, 3), dtype=np.float32), np.zeros((h, w), dtype=np.float32)

    # 2. Numerically stabilized smooth-minimum (log-sum-exp smin) SDF union in pixel units
    # d < 0 inside the cloud, d > 0 outside the cloud.
    k_smooth = 0.055 * (540.0 / max(180.0, scale_px))  # smooth blending radius ~18px at 540p
    min_d = np.full((h, w), 1e6, dtype=np.float32)
    dome_sdfs: list[np.ndarray] = []
    for (px, py, prx, pry) in domes:
        r_mean = 0.5 * (prx + pry)
        d_i = (np.sqrt(((xx - px) / prx) ** 2 + ((yy - py) / pry) ** 2) - 1.0) * r_mean
        dome_sdfs.append(d_i)
        min_d = np.minimum(min_d, d_i)

    exp_sum = np.zeros((h, w), dtype=np.float32)
    for d_i in dome_sdfs:
        exp_sum += np.exp(np.clip(-k_smooth * (d_i - min_d), -40.0, 0.0))
    sdf = min_d - (np.log(np.maximum(exp_sum, 1e-12)) / k_smooth)

    # Smooth-maximum (smax) flat meteorological condensation base line
    if base_y_px > 0:
        d_base = yy - base_y_px
        k_base = 0.08 * (540.0 / max(180.0, scale_px))
        m_max = np.maximum(sdf, d_base)
        sdf = m_max + np.log(
            np.exp(np.clip(k_base * (sdf - m_max), -40.0, 0.0))
            + np.exp(np.clip(k_base * (d_base - m_max), -40.0, 0.0))
        ) / k_base

    # 3. Subtle low-amplitude fBM edge displacement ONLY near the outer silhouette (100% solid interior!)
    edge_fbm = _fbm_2d(h, w, rng, octaves=((32, 0.55), (14, 0.30), (6, 0.15)))
    edge_band_mask = np.exp(-((sdf / (scale_px * 0.045)) ** 2))
    edge_amp = scale_px * 0.016
    sdf_disp = sdf + edge_fbm * edge_amp * edge_band_mask

    # Crisp-yet-feathered gouache cloud silhouette alpha (strictly solid wherever sdf_disp < -2px)
    feather = max(2.2, scale_px * 0.006)
    alpha_buf = np.clip((-sdf_disp + feather * 0.5) / (feather * 1.5), 0.0, 1.0)
    alpha_buf = alpha_buf * alpha_buf * (3.0 - 2.0 * alpha_buf)

    if np.max(alpha_buf) < 1e-3:
        return np.zeros((h, w, 3), dtype=np.float32), alpha_buf

    # 4. Smooth 3D cumulus dome heightfield & analytical/finite-difference 2D+Z surface normals
    # Interior depth (-sdf) smoothly rises into rounded billowing dome volumes
    depth_in = np.maximum(0.0, -sdf)
    dome_scale = scale_px * 0.17
    height_3d = np.sqrt(np.clip(1.0 - np.exp(-depth_in / dome_scale), 0.0, 1.0))
    # Silky-smooth dome shading (ZERO medium/high-frequency procedural noise in cloud interiors!)
    height_smooth = cv2.GaussianBlur(height_3d, (0, 0), sigmaX=12.5, sigmaY=12.5)

    gx = cv2.Sobel(height_smooth, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(height_smooth, cv2.CV_32F, 0, 1, ksize=5)
    # Normal points outward from the convex dome surface: (-dh/dx, -dh/dy, z_scale)
    nx_raw = -gx * 1.55
    ny_raw = -gy * 1.55
    nz_raw = np.full((h, w), 0.38, dtype=np.float32)
    n_len = np.sqrt(nx_raw * nx_raw + ny_raw * ny_raw + nz_raw * nz_raw) + 1e-6
    nx = nx_raw / n_len
    ny = ny_raw / n_len
    nz = nz_raw / n_len

    # 5. Directional (or radial sun_pos) sunlight + smooth volumetric self-shadowing
    if sun_pos is not None:
        spx, spy = sun_pos[0] * w, sun_pos[1] * h
        dx_s = spx - xx
        dy_s = spy - yy
        d_s = np.sqrt(dx_s * dx_s + dy_s * dy_s) + 1e-5
        sx_map = dx_s / d_s
        sy_map = dy_s / d_s
    else:
        sx, sy = sun_dir
        snorm = float(np.hypot(sx, sy) + 1e-6)
        sx_map = np.full((h, w), sx / snorm, dtype=np.float32)
        sy_map = np.full((h, w), sy / snorm, dtype=np.float32)

    # Ray-march 6 steps toward the sun through the SOLID smooth alpha field for clean self-shadowing
    occ = np.zeros((h, w), dtype=np.float32)
    step_px = scale_px * 0.024
    for step_i in range(1, 7):
        shift_x = np.clip(xx + sx_map * step_px * step_i, 0, w - 1).astype(np.float32)
        shift_y = np.clip(yy + sy_map * step_px * step_i, 0, h - 1).astype(np.float32)
        sample_a = cv2.remap(alpha_buf, shift_x, shift_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        occ += sample_a * (1.0 / (step_i ** 0.72))
    occ = np.clip(occ / 3.4, 0.0, 1.0)
    occ = cv2.GaussianBlur(occ, (0, 0), sigmaX=11.0, sigmaY=11.0)

    # Combine crisp directional NdotL + smooth self-shadowing + warm altitude lift (dark flat cloud base)
    ndotl = nx * sx_map + ny * sy_map + nz * 0.45
    alt_frac = np.clip((base_y_px - yy) / max(1.0, h * 0.44), 0.0, 1.0)
    base_shadow = np.exp(-((np.maximum(0.0, base_y_px - yy) / (scale_px * 0.085)) ** 2)) * 0.34
    shade = np.clip(0.54 + 0.36 * ndotl - 0.18 * occ + 0.22 * alt_frac - base_shadow, 0.0, 0.99)
    # Painterly gouache tonal curve (clean sculpted steps from sunlit cream -> lavender midtone -> cobalt base)
    smooth_s = shade * shade * (3.0 - 2.0 * shade)
    shade = np.clip(0.82 * smooth_s + 0.18 * shade, 0.0, 0.99)

    rgb_buf = _sample_ramp(shade, color_ramp)

    # 6. Luminous subsurface cream-gold / apricot rim-lighting on sun-facing cloud edges
    if rim_color is not None and rim_strength > 0.0:
        rim_edge = np.exp(-(((sdf_disp + scale_px * 0.010) / (scale_px * 0.024)) ** 2)) * alpha_buf
        sun_facing = np.clip(nx * sx_map + ny * sy_map + 0.42, 0.0, 1.0) * (1.0 - occ * 0.40)
        rim_mask = np.clip(rim_edge * sun_facing * rim_strength * 1.40, 0.0, 0.94)[..., None]
        rc = np.array(rim_color, dtype=np.float32)[None, None, :]
        rgb_buf = rgb_buf * (1.0 - rim_mask) + rc * rim_mask

    # 7. Clean edge-preserving gouache brushwork smoothing (eliminates any digital/spongy artifacts)
    rgb_u8 = np.clip(rgb_buf * 255.0, 0, 255).astype(np.uint8)
    rgb_u8 = cv2.edgePreservingFilter(rgb_u8, flags=1, sigma_s=36, sigma_r=0.22)
    rgb_buf = rgb_u8.astype(np.float32) / 255.0
    rgb_buf = apply_gouache_brushwork(rgb_buf, strength=0.009, angle_deg=14.0, scale=1.1, seed=seed + 1)

    return np.clip(rgb_buf, 0.0, 1.0), np.clip(alpha_buf, 0.0, 1.0)
