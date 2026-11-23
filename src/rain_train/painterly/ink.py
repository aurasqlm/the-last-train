"""Hand-drawn anime ink, 3/4 perspective vintage train, station architecture, graceful anime figure, lanterns & petals."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def apply_line_boil(img: np.ndarray, frame_idx: int, strength: float = 0.6) -> np.ndarray:
    """Subtle 12fps hand-drawn ink line-boil micro-displacement."""
    boil_step = frame_idx // 5
    rng = np.random.default_rng(boil_step + 9001)
    h, w = img.shape[:2]
    dx = cv2.resize(rng.standard_normal((max(4, h // 48), max(4, w // 48))).astype(np.float32), (w, h)) * strength
    dy = cv2.resize(rng.standard_normal((max(4, h // 48), max(4, w // 48))).astype(np.float32), (w, h)) * strength
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(img, np.clip(xx + dx, 0, w - 1), np.clip(yy + dy, 0, h - 1), cv2.INTER_LINEAR)


def _add_glow(
    img: np.ndarray,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    color: Tuple[float, float, float],
    strength: float,
) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = max(0, int(cx - rx * 3.0))
    x1 = min(w, int(cx + rx * 3.0) + 1)
    y0 = max(0, int(cy - ry * 3.0))
    y1 = min(h, int(cy + ry * 3.0) + 1)
    if x1 <= x0 or y1 <= y0:
        return img
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d2 = ((xx - cx) / max(1.0, rx)) ** 2 + ((yy - cy) / max(1.0, ry)) ** 2
    g = np.exp(-d2 * 0.9) * strength
    col = np.array(color, dtype=np.float32)[None, None, :]
    patch = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = 1.0 - (1.0 - patch) * (1.0 - np.clip(g[..., None] * col, 0.0, 0.95))
    return img


def draw_lantern(
    img: np.ndarray,
    x_frac: float,
    y_frac: float,
    radius_frac: float = 0.016,
    glow_radius_mult: float = 7.5,
    glow_strength: float = 0.62,
    hanging: bool = True,
    water_reflection: bool = False,
) -> np.ndarray:
    """Draw a detailed 3D Japanese wooden/paper lantern with warm 35mm optical glow."""
    h, w = img.shape[:2]
    cx, cy = x_frac * w, y_frac * h
    r = max(3.5, radius_frac * h)

    out = img.copy()
    # Wide warm amber atmospheric halo + tight incandescent core glow
    out = _add_glow(out, cx, cy, r * glow_radius_mult, r * glow_radius_mult, (1.00, 0.66, 0.24), glow_strength * 0.65)
    out = _add_glow(out, cx, cy, r * 2.8, r * 2.8, (1.00, 0.90, 0.56), glow_strength * 0.95)

    if water_reflection:
        # Subtle warm amber alpha-tinted halo so glow reads unmistakably even over bright mirror-sea clouds!
        yy_g, xx_g = np.mgrid[0:h, 0:w].astype(np.float32)
        d_halo = np.exp(-(((xx_g - cx) / max(1.0, r * 4.8)) ** 2 + ((yy_g - cy) / max(1.0, r * 4.2)) ** 2))
        amber_tint = np.array([1.00, 0.72, 0.26], dtype=np.float32)[None, None, :]
        h_a = np.clip(d_halo * glow_strength * 0.52, 0.0, 0.65)[..., None]
        out = np.clip(out * (1.0 - h_a * 0.35) + amber_tint * h_a, 0.0, 1.0)

        # Delicate concentric water ripple rings around the floating raft base
        rip_buf = np.zeros((h, w), dtype=np.float32)
        for s_r, a_r in ((1.8, 0.42), (2.9, 0.28), (4.2, 0.16)):
            cv2.ellipse(
                rip_buf,
                (int(cx), int(cy + r * 0.95)),
                (max(4, int(r * s_r)), max(1, int(r * s_r * 0.24))),
                0, 0, 360, a_r, 1, cv2.LINE_AA,
            )
        rip_buf = cv2.GaussianBlur(rip_buf, (0, 0), sigmaX=0.9, sigmaY=0.6)
        out = np.clip(out + rip_buf[..., None] * np.array([1.00, 0.88, 0.62], dtype=np.float32) * 0.55, 0.0, 1.0)

        # Long, shimmering, broken vertical amber reflection column dancing across the rippled mirror-sea
        col_h = int(min(h - cy - 2, r * 14.0))
        if col_h > 4:
            rng_l = np.random.default_rng(int(cx * 13 + cy * 7))
            shimmer = np.zeros((h, w), dtype=np.float32)
            for s_i in range(28):
                t_s = s_i / 27.0
                sy = int(cy + r * 1.05 + t_s * col_h)
                if sy >= h - 1:
                    break
                wobble = float(rng_l.normal(0.0, r * (0.22 + 0.52 * t_s)))
                rx_s = max(2, int(r * (1.9 + 1.5 * t_s) * rng_l.uniform(0.55, 1.35)))
                ry_s = max(1, int(r * (0.30 + 0.26 * t_s)))
                amp_s = float((1.0 - t_s) ** 1.15 * rng_l.uniform(0.55, 1.0))
                cv2.ellipse(shimmer, (int(cx + wobble), sy), (rx_s, ry_s), 0, 0, 360, amp_s, -1, cv2.LINE_AA)
            shimmer = cv2.GaussianBlur(shimmer, (0, 0), sigmaX=1.5, sigmaY=0.9)
            scol = np.array([1.00, 0.72, 0.26], dtype=np.float32)[None, None, :]
            sh_a = np.clip(shimmer * glow_strength * 0.95, 0.0, 0.88)[..., None]
            # Hybrid alpha + screen blend so the golden reflection column glows richly on both dark & bright water
            out = np.clip(out * (1.0 - sh_a * 0.45) + scol * sh_a * 0.95, 0.0, 1.0)
        # Soft warm water-surface glow pool right under the raft
        out = _add_glow(out, cx, cy + r * 1.1, r * 4.2, r * 1.3, (1.00, 0.74, 0.32), glow_strength * 0.78)

    u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    ix, iy, ir = int(cx), int(cy), max(3, int(r))

    if hanging:
        # Iron cord & wooden pagoda cap
        cv2.line(u8, (ix, max(0, iy - int(ir * 3.8))), (ix, iy - ir), (32, 26, 24), max(1, ir // 5), cv2.LINE_AA)
        cap_pts = np.array([
            [ix - int(ir * 1.25), iy - int(ir * 0.75)],
            [ix + int(ir * 1.25), iy - int(ir * 0.75)],
            [ix + int(ir * 0.7), iy - int(ir * 1.25)],
            [ix - int(ir * 0.7), iy - int(ir * 1.25)],
        ], dtype=np.int32)
        cv2.fillPoly(u8, [cap_pts], (38, 28, 22), cv2.LINE_AA)
        # Warm glowing paper lantern body
        cv2.ellipse(u8, (ix, iy), (int(ir * 0.85), int(ir * 1.05)), 0, 0, 360, (255, 222, 148), -1, cv2.LINE_AA)
        cv2.ellipse(u8, (ix, iy), (int(ir * 0.55), int(ir * 0.72)), 0, 0, 360, (255, 246, 212), -1, cv2.LINE_AA)
        # Subtle wooden ribs
        cv2.ellipse(u8, (ix, iy), (int(ir * 0.85), int(ir * 1.05)), 0, 0, 360, (115, 78, 48), 1, cv2.LINE_AA)
        cv2.rectangle(u8, (ix - int(ir * 0.55), iy + int(ir * 0.85)), (ix + int(ir * 0.55), iy + int(ir * 1.15)), (42, 30, 22), -1)
    else:
        # Delicate floating Toro Nagashi (dark cedar raft + softly tapered glowing shoji lantern)
        raft_pts = np.array([
            [ix - int(ir * 1.45), iy + int(ir * 0.65)],
            [ix + int(ir * 1.45), iy + int(ir * 0.65)],
            [ix + int(ir * 1.10), iy + int(ir * 0.98)],
            [ix - int(ir * 1.10), iy + int(ir * 0.98)],
        ], dtype=np.int32)
        cv2.fillPoly(u8, [raft_pts], (42, 28, 22), cv2.LINE_AA)
        # Tapered glowing shoji paper body
        body_pts = np.array([
            [ix - int(ir * 0.68), iy - int(ir * 0.95)],
            [ix + int(ir * 0.68), iy - int(ir * 0.95)],
            [ix + int(ir * 0.82), iy + int(ir * 0.65)],
            [ix - int(ir * 0.82), iy + int(ir * 0.65)],
        ], dtype=np.int32)
        cv2.fillPoly(u8, [body_pts], (255, 214, 126), cv2.LINE_AA)
        # Warm incandescent candle core
        cv2.ellipse(u8, (ix, iy - int(ir * 0.05)), (max(2, int(ir * 0.48)), max(2, int(ir * 0.58))), 0, 0, 360, (255, 250, 226), -1, cv2.LINE_AA)
        # Delicate wooden rim & top lip
        cv2.polylines(u8, [body_pts], True, (110, 68, 38), 1, cv2.LINE_AA)

    return u8.astype(np.float32) / 255.0


def draw_umbrella_figure(
    img: np.ndarray,
    x_frac: float,
    y_bottom_frac: float,
    height_frac: float = 0.28,
    rim_color: Tuple[float, float, float] = (0.98, 0.78, 0.44),
) -> np.ndarray:
    """Draw a graceful, anatomically proportioned anime silhouette with a ribbed translucent vinyl umbrella."""
    h, w = img.shape[:2]
    cx = x_frac * w
    yb = y_bottom_frac * h
    fh = height_frac * h  # total figure height (feet to head top)

    out = img.copy()

    # 1. Translucent vinyl umbrella canopy (drawn with alpha blend so warm lantern light transmits through!)
    umb_cx = cx + fh * 0.04
    umb_cy = yb - fh * 0.96
    umb_rx = fh * 0.56
    umb_ry = fh * 0.24

    # Soft warm light glow transmitting through the translucent vinyl canopy
    out = _add_glow(out, umb_cx, umb_cy - umb_ry * 0.1, umb_rx * 1.15, umb_ry * 1.5, rim_color, 0.24)

    # Draw canopy polygon (curved dome)
    angles = np.linspace(np.pi, 2.0 * np.pi, 36)
    dome_pts = []
    for a in angles:
        px = umb_cx + np.cos(a) * umb_rx
        py = umb_cy + np.sin(a) * umb_ry
        dome_pts.append([int(px), int(py)])
    # Subtle scalloped vinyl bottom edge
    scallop_xs = np.linspace(umb_cx + umb_rx, umb_cx - umb_rx, 13)
    for i, sx in enumerate(scallop_xs):
        dip = -fh * 0.022 if (i % 2 == 1) else 0.0
        dome_pts.append([int(sx), int(umb_cy + dip)])
    dome_arr = np.array(dome_pts, dtype=np.int32)

    canopy_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(canopy_mask, [dome_arr], 1.0, cv2.LINE_AA)
    # Translucent warm-tinted vinyl (lets background lantern glow show through!)
    vinyl_col = np.array([0.82, 0.76, 0.68], dtype=np.float32)[None, None, :]
    a_canopy = (canopy_mask * 0.42)[..., None]
    out = np.clip(out * (1.0 - a_canopy) + vinyl_col * a_canopy, 0.0, 1.0)

    u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    rim_bgr = tuple(int(np.clip(c * 255, 0, 255)) for c in rim_color)

    # Umbrella ribs & crisp warm rim highlight along top curve
    top_pt = (int(umb_cx), int(umb_cy - umb_ry - fh * 0.03))
    for rib_f in (-0.82, -0.45, 0.0, 0.45, 0.82):
        rx_end = int(umb_cx + rib_f * umb_rx)
        ry_end = int(umb_cy - fh * 0.01)
        cv2.line(u8, top_pt, (rx_end, ry_end), (190, 170, 145), 1, cv2.LINE_AA)
    cv2.polylines(u8, [dome_arr[:36]], False, rim_bgr, 2, cv2.LINE_AA)

    # Umbrella shaft (angled naturally toward right hand)
    hand_x = int(cx + fh * 0.11)
    hand_y = int(yb - fh * 0.62)
    cv2.line(u8, (int(umb_cx), int(umb_cy - umb_ry)), (hand_x, hand_y), (32, 34, 40), 2, cv2.LINE_AA)

    # 2. Graceful anime silhouette (head, windswept hair, neck, sloped shoulders, belted trench coat, boots)
    dark_coat = (18, 22, 30)
    mid_coat = (28, 34, 44)

    # Boots & legs (natural stance, slightly apart)
    leg_l = np.array([
        [int(cx - fh * 0.07), int(yb - fh * 0.22)],
        [int(cx - fh * 0.02), int(yb - fh * 0.22)],
        [int(cx - fh * 0.03), int(yb)],
        [int(cx - fh * 0.08), int(yb)],
    ], dtype=np.int32)
    leg_r = np.array([
        [int(cx + fh * 0.01), int(yb - fh * 0.22)],
        [int(cx + fh * 0.06), int(yb - fh * 0.22)],
        [int(cx + fh * 0.08), int(yb)],
        [int(cx + fh * 0.03), int(yb)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [leg_l, leg_r], (14, 17, 24), cv2.LINE_AA)

    # Draped trench coat with natural waist taper & windswept flared hem
    coat_pts = np.array([
        [int(cx - fh * 0.11), int(yb - fh * 0.72)],  # left shoulder
        [int(cx + fh * 0.10), int(yb - fh * 0.73)],  # right shoulder
        [int(cx + fh * 0.13), int(yb - fh * 0.60)],  # right sleeve reaching to umbrella
        [int(cx + fh * 0.08), int(yb - fh * 0.52)],  # waist right
        [int(cx + fh * 0.15), int(yb - fh * 0.18)],  # flared hem right
        [int(cx + fh * 0.04), int(yb - fh * 0.16)],  # hem fold
        [int(cx - fh * 0.04), int(yb - fh * 0.19)],  # hem fold
        [int(cx - fh * 0.17), int(yb - fh * 0.17)],  # windswept hem left
        [int(cx - fh * 0.10), int(yb - fh * 0.52)],  # waist left
        [int(cx - fh * 0.14), int(yb - fh * 0.62)],  # left sleeve
    ], dtype=np.int32)
    cv2.fillPoly(u8, [coat_pts], dark_coat, cv2.LINE_AA)

    # Subtle coat collar & warm rim-light along right shoulder/coat edge (facing train/lanterns)
    cv2.line(u8, (int(cx + fh * 0.10), int(yb - fh * 0.73)), (int(cx + fh * 0.13), int(yb - fh * 0.60)), rim_bgr, 1, cv2.LINE_AA)
    cv2.line(u8, (int(cx + fh * 0.08), int(yb - fh * 0.52)), (int(cx + fh * 0.15), int(yb - fh * 0.18)), rim_bgr, 1, cv2.LINE_AA)

    # Slender neck
    cv2.rectangle(u8, (int(cx - fh * 0.025), int(yb - fh * 0.78)), (int(cx + fh * 0.025), int(yb - fh * 0.72)), dark_coat, -1)

    # Anime head (slightly tilted up looking toward the arriving train)
    head_cx = int(cx + fh * 0.005)
    head_cy = int(yb - fh * 0.835)
    cv2.ellipse(u8, (head_cx, head_cy), (int(fh * 0.062), int(fh * 0.072)), -6, 0, 360, dark_coat, -1, cv2.LINE_AA)

    # Windswept shoulder-length anime hair strands
    hair_pts = np.array([
        [int(cx - fh * 0.06), int(yb - fh * 0.88)],
        [int(cx + fh * 0.05), int(yb - fh * 0.89)],
        [int(cx + fh * 0.07), int(yb - fh * 0.81)],
        [int(cx + fh * 0.02), int(yb - fh * 0.74)],
        [int(cx - fh * 0.09), int(yb - fh * 0.72)],
        [int(cx - fh * 0.12), int(yb - fh * 0.76)],
        [int(cx - fh * 0.08), int(yb - fh * 0.82)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [hair_pts], mid_coat, cv2.LINE_AA)

    return u8.astype(np.float32) / 255.0


def draw_train_perspective(
    img: np.ndarray,
    front_x_frac: float = 0.36,
    front_y_bottom_frac: float = 0.74,
    front_h_frac: float = 0.36,
    vp_xy: Tuple[float, float] = (1.18, 0.56),
    body_color: Tuple[float, float, float] = (0.12, 0.17, 0.23),
    stripe_color: Tuple[float, float, float] = (0.58, 0.25, 0.20),
    window_color: Tuple[float, float, float] = (1.00, 0.82, 0.46),
    headlight_beams: bool = True,
) -> np.ndarray:
    """Draw a dramatic 3/4 perspective vintage Japanese rain train with recessed lit windows & volumetric headlight cones."""
    h, w = img.shape[:2]
    out = img.copy()

    fx = front_x_frac * w
    fyb = front_y_bottom_frac * h
    fh = front_h_frac * h
    fyt = fyb - fh
    vpx, vpy = vp_xy[0] * w, vp_xy[1] * h

    # Front cab width in 3/4 view
    cab_w = fh * 0.52
    # Receding side length toward vanishing point
    t_end = 0.72
    bx = fx + (vpx - fx) * t_end
    byb = fyb + (vpy - fyb) * t_end
    byt = fyt + (vpy - fyt) * t_end

    # 1. Strong volumetric conical headlight beams slicing diagonally down-left through the misty rain
    if headlight_beams:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        for hl_off in (0.28, 0.72):
            hx = fx - cab_w * hl_off
            hy = fyb - fh * 0.26
            dx = hx - xx
            dy = yy - hy
            r_px = np.sqrt(dx * dx + dy * dy) + 1e-5
            dist = r_px / w
            # Wide atmospheric cone + tight bright core slicing down-left
            beam_dir = (dx * 0.92 + dy * 0.39) / r_px
            wide_cone = np.clip((beam_dir - 0.80) / 0.20, 0.0, 1.0) ** 1.35
            core_cone = np.clip((beam_dir - 0.93) / 0.07, 0.0, 1.0) ** 1.4
            falloff = np.exp(-dist * 1.95) * (dx > -4.0)
            beam = (wide_cone * 0.56 + core_cone * 0.42) * falloff
            bcol = np.array([1.00, 0.90, 0.64], dtype=np.float32)[None, None, :]
            out = 1.0 - (1.0 - out) * (1.0 - np.clip(beam[..., None] * bcol, 0.0, 0.92))

    # Warm golden window light spill cast downward onto the wet platform/air beside the train
    out = _add_glow(out, fx + (bx - fx) * 0.38, fyb + fh * 0.04, (bx - fx) * 0.52, fh * 0.28, window_color, 0.34)

    u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    b_col = tuple(int(np.clip(c * 255, 0, 255)) for c in body_color)
    b_dark = tuple(int(np.clip(c * 165, 0, 255)) for c in body_color)
    b_roof = tuple(int(np.clip(c * 310 + 22, 0, 255)) for c in body_color)
    s_col = tuple(int(np.clip(c * 255, 0, 255)) for c in stripe_color)
    w_col = tuple(int(np.clip(c * 255, 0, 255)) for c in window_color)

    def _lerp_side(t: float, v_frac: float) -> Tuple[int, int]:
        px = fx + (vpx - fx) * t
        py_top = fyt + (vpy - fyt) * t
        py_bot = fyb + (vpy - fyb) * t
        return int(px), int(py_top + (py_bot - py_top) * v_frac)

    # 2. Receding gravel trackbed, wooden sleepers, wet glinting steel rails & deep contact shadow
    rail_y_left_0 = fyb + fh * 0.085
    rail_y_right_0 = fyb + fh * 0.105
    rail_x_left_0 = fx - cab_w * 0.88
    rail_x_right_0 = fx - cab_w * 0.06
    # Trackbed polygon extending from bottom-left foreground under the train to vanishing point
    t_fore = -0.55
    tb_l_fore = (int(rail_x_left_0 + (vpx - rail_x_left_0) * t_fore - fh * 0.25), int(rail_y_left_0 + (vpy - rail_y_left_0) * t_fore + fh * 0.04))
    tb_r_fore = (int(rail_x_right_0 + (vpx - rail_x_right_0) * t_fore + fh * 0.35), int(rail_y_right_0 + (vpy - rail_y_right_0) * t_fore + fh * 0.06))
    tb_pts = np.array([
        tb_l_fore,
        [int(vpx), int(vpy + 2)],
        [int(vpx + 40), int(vpy + 12)],
        tb_r_fore,
    ], dtype=np.int32)
    cv2.fillPoly(u8, [tb_pts], (18, 22, 28), cv2.LINE_AA)

    # Wooden railway sleepers receding in perspective
    for s_i in range(28):
        st = -0.45 + 1.38 * ((s_i / 27.0) ** 0.72)
        sl_x = rail_x_left_0 + (vpx - rail_x_left_0) * st
        sl_y = rail_y_left_0 + (vpy - rail_y_left_0) * st
        sr_x = rail_x_right_0 + (vpx - rail_x_right_0) * st
        sr_y = rail_y_right_0 + (vpy - rail_y_right_0) * st
        pad_x = (sr_x - sl_x) * 0.22
        thick_s = max(1, int(3.5 * (1.0 - np.clip(st, 0.0, 0.92))))
        cv2.line(
            u8,
            (int(sl_x - pad_x), int(sl_y + 1)),
            (int(sr_x + pad_x), int(sr_y + 2)),
            (30, 26, 24),
            thick_s,
            cv2.LINE_AA,
        )

    # Two wet glinting steel rails receding straight to the vanishing point
    for (rx0, ry0) in ((rail_x_left_0, rail_y_left_0), (rail_x_right_0, rail_y_right_0)):
        p_fore = (int(rx0 + (vpx - rx0) * t_fore), int(ry0 + (vpy - ry0) * t_fore))
        p_vp = (int(vpx), int(vpy))
        cv2.line(u8, p_fore, p_vp, (22, 26, 34), 4, cv2.LINE_AA)
        cv2.line(u8, (p_fore[0], p_fore[1] - 1), p_vp, (148, 168, 188), 2, cv2.LINE_AA)

    # Deep dark contact shadow directly under the train on the trackbed
    shadow_pts = np.array([
        [int(fx - cab_w * 0.98), int(fyb + fh * 0.075)],
        [int(fx + 4), int(fyb + fh * 0.115)],
        [int(bx), int(byb + fh * 0.042)],
        [int(bx - cab_w * 0.25), int(byb + fh * 0.025)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [shadow_pts], (8, 10, 14), cv2.LINE_AA)

    # Deep dark undercarriage, steel bogie frames & spoked steel wheels resting directly on the rails
    under_pts = np.array([
        [int(fx - cab_w * 0.96), int(fyb - fh * 0.04)],
        [int(fx), int(fyb - fh * 0.01)],
        [int(bx), int(byb)],
        [int(bx), int(byb + fh * 0.036)],
        [int(fx), int(fyb + fh * 0.095)],
        [int(fx - cab_w * 0.96), int(fyb + fh * 0.070)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [under_pts], (12, 15, 20), cv2.LINE_AA)
    # Visible spoked bogie wheels resting solidly on the steel rail
    for wt in (0.08, 0.17, 0.36, 0.44, 0.56):
        wx, wy = _lerp_side(wt, 1.035)
        wr = max(3, int(fh * 0.064 * (1.0 - wt * 0.70)))
        cv2.circle(u8, (wx, wy), wr, (22, 28, 38), -1, cv2.LINE_AA)
        cv2.circle(u8, (wx, wy), max(2, wr - 1), (82, 96, 114), 1, cv2.LINE_AA)
        if wr >= 6:
            cv2.circle(u8, (wx, wy), max(1, wr // 3), (62, 74, 88), -1, cv2.LINE_AA)

    # 3. Receding side body wall (quad tapering toward vanishing point)
    side_pts = np.array([
        [int(fx), int(fyt)],
        [int(bx), int(byt)],
        [int(bx), int(byb)],
        [int(fx), int(fyb)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [side_pts], b_col, cv2.LINE_AA)
    # Subtle wet rain-sheen metallic highlight band along upper side wall & lower shadow skirt
    sheen_pts = np.array([
        _lerp_side(0.0, 0.02),
        _lerp_side(t_end, 0.02),
        _lerp_side(t_end, 0.14),
        _lerp_side(0.0, 0.14),
    ], dtype=np.int32)
    cv2.fillPoly(u8, [sheen_pts], tuple(min(255, c + 18) for c in b_col), cv2.LINE_AA)
    skirt_pts = np.array([
        _lerp_side(0.0, 0.84),
        _lerp_side(t_end, 0.84),
        _lerp_side(t_end, 1.00),
        _lerp_side(0.0, 1.00),
    ], dtype=np.int32)
    cv2.fillPoly(u8, [skirt_pts], b_dark, cv2.LINE_AA)

    # Receding vermilion/rust vintage stripe along side
    stripe_pts = np.array([
        _lerp_side(0.0, 0.68),
        _lerp_side(t_end, 0.68),
        _lerp_side(t_end, 0.80),
        _lerp_side(0.0, 0.80),
    ], dtype=np.int32)
    cv2.fillPoly(u8, [stripe_pts], s_col, cv2.LINE_AA)

    # Subtle vertical rain-streak weathering & door seams along steel body panels
    rng_w = np.random.default_rng(109)
    for dt in (0.145, 0.355, 0.525):
        cv2.line(u8, _lerp_side(dt, 0.04), _lerp_side(dt, 0.98), (18, 24, 32), 1, cv2.LINE_AA)
        cv2.line(u8, _lerp_side(dt + 0.004, 0.04), _lerp_side(dt + 0.004, 0.98), (52, 66, 80), 1, cv2.LINE_AA)
    for _ in range(34):
        wt = float(rng_w.uniform(0.02, t_end * 0.94))
        v0 = float(rng_w.uniform(0.04, 0.40))
        v1 = min(0.98, v0 + float(rng_w.uniform(0.25, 0.65)))
        c_shift = int(rng_w.integers(-10, 14))
        st_col = tuple(int(np.clip(c + c_shift, 10, 220)) for c in b_col)
        cv2.line(u8, _lerp_side(wt, v0), _lerp_side(wt, v1), st_col, 1, cv2.LINE_AA)

    # Curved metallic roof receding in perspective with wet specular ridge highlight & AC units
    roof_pts = np.array([
        [int(fx - cab_w * 0.92), int(fyt + fh * 0.02)],
        [int(fx - cab_w * 0.45), int(fyt - fh * 0.065)],
        [int(fx), int(fyt - fh * 0.055)],
        [int(bx), int(byt - fh * 0.018)],
        [int(bx), int(byt)],
        [int(fx), int(fyt)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [roof_pts], b_roof, cv2.LINE_AA)
    cv2.line(u8, (int(fx - cab_w * 0.35), int(fyt - fh * 0.055)), (int(bx), int(byt - fh * 0.016)), (132, 156, 178), 2, cv2.LINE_AA)

    # Rooftop AC vents & electrical insulators receding in perspective
    for (at0, at1) in ((0.06, 0.13), (0.28, 0.36), (0.48, 0.55)):
        ac_pts = np.array([
            _lerp_side(at0, -0.04),
            _lerp_side(at1, -0.04),
            _lerp_side(at1, -0.10),
            _lerp_side(at0, -0.10),
        ], dtype=np.int32)
        cv2.fillPoly(u8, [ac_pts], (42, 52, 64), cv2.LINE_AA)
        cv2.line(u8, _lerp_side(at0, -0.10), _lerp_side(at1, -0.10), (96, 114, 132), 1, cv2.LINE_AA)

    # 4. Recessed warm-lit passenger windows with wooden mullions, seats & hanging hand-straps
    win_ts = [(0.05, 0.13), (0.16, 0.24), (0.27, 0.34), (0.37, 0.43), (0.46, 0.51), (0.54, 0.59), (0.61, 0.66)]
    for wi, (t0, t1) in enumerate(win_ts):
        w_pts = np.array([
            _lerp_side(t0, 0.20),
            _lerp_side(t1, 0.20),
            _lerp_side(t1, 0.58),
            _lerp_side(t0, 0.58),
        ], dtype=np.int32)
        # Warm glowing interior with subtle per-window warmth variation
        w_tint = (
            min(255, int(w_col[0])),
            min(255, int(w_col[1] * (0.94 + 0.06 * ((wi % 3) / 2.0)))),
            min(255, int(w_col[2] * (0.88 + 0.12 * ((wi % 2))))),
        )
        cv2.fillPoly(u8, [w_pts], w_tint, cv2.LINE_AA)
        # Bright ceiling tungsten strip at top of window
        ceil_pts = np.array([
            _lerp_side(t0, 0.20),
            _lerp_side(t1, 0.20),
            _lerp_side(t1, 0.29),
            _lerp_side(t0, 0.29),
        ], dtype=np.int32)
        cv2.fillPoly(u8, [ceil_pts], (255, 240, 185), cv2.LINE_AA)
        # Interior seat-back silhouette in lower third of window
        seat_pts = np.array([
            _lerp_side(t0 + (t1 - t0) * 0.06, 0.46),
            _lerp_side(t1 - (t1 - t0) * 0.06, 0.46),
            _lerp_side(t1 - (t1 - t0) * 0.06, 0.58),
            _lerp_side(t0 + (t1 - t0) * 0.06, 0.58),
        ], dtype=np.int32)
        cv2.fillPoly(u8, [seat_pts], (118, 74, 42), cv2.LINE_AA)
        # Hanging hand-straps inside warm window
        for strap_f in (0.32, 0.68):
            ts = t0 + (t1 - t0) * strap_f
            p_top = _lerp_side(ts, 0.20)
            p_mid = _lerp_side(ts, 0.34)
            cv2.line(u8, p_top, p_mid, (105, 68, 38), 1, cv2.LINE_AA)
            cv2.circle(u8, p_mid, max(1, int(2.8 * (1.0 - ts))), (105, 68, 38), 1, cv2.LINE_AA)
        # Wooden window mullions (horizontal transom bar + vertical center sash)
        cv2.line(u8, _lerp_side(t0, 0.32), _lerp_side(t1, 0.32), (62, 42, 28), 1, cv2.LINE_AA)
        cv2.line(u8, _lerp_side(0.5 * (t0 + t1), 0.20), _lerp_side(0.5 * (t0 + t1), 0.58), (62, 42, 28), 1, cv2.LINE_AA)
        # Dark recessed window frame
        cv2.polylines(u8, [w_pts], True, (22, 28, 36), 2, cv2.LINE_AA)

    # 5. Front cab face (3/4 view) with destination headsign, wipers, grab-rails, coupler/skirt & headlight housings
    front_pts = np.array([
        [int(fx - cab_w), int(fyt + fh * 0.03)],
        [int(fx), int(fyt)],
        [int(fx), int(fyb)],
        [int(fx - cab_w), int(fyb - fh * 0.03)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [front_pts], b_dark, cv2.LINE_AA)
    # Metallic specular edge highlight along the front-right corner pillar & brow
    cv2.line(u8, (int(fx), int(fyt + fh * 0.01)), (int(fx), int(fyb)), (98, 118, 138), 2, cv2.LINE_AA)
    cv2.line(u8, (int(fx - cab_w), int(fyt + fh * 0.03)), (int(fx), int(fyt)), (118, 138, 158), 2, cv2.LINE_AA)

    # Illuminated vintage destination headsign plate centered above the windshields
    dest_pts = np.array([
        [int(fx - cab_w * 0.72), int(fyt + fh * 0.075)],
        [int(fx - cab_w * 0.28), int(fyt + fh * 0.062)],
        [int(fx - cab_w * 0.28), int(fyt + fh * 0.145)],
        [int(fx - cab_w * 0.72), int(fyt + fh * 0.158)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [dest_pts], (242, 212, 148), cv2.LINE_AA)
    cv2.polylines(u8, [dest_pts], True, (26, 30, 38), 2, cv2.LINE_AA)
    # Subtle vintage typography dashes inside destination board
    cv2.line(
        u8,
        (int(fx - cab_w * 0.64), int(fyt + fh * 0.115)),
        (int(fx - cab_w * 0.36), int(fyt + fh * 0.104)),
        (58, 44, 34),
        2,
        cv2.LINE_AA,
    )

    # Front stripe & steel bumper/coupler/skirt hardware
    f_stripe = np.array([
        [int(fx - cab_w), int(fyt + fh * 0.69)],
        [int(fx), int(fyt + fh * 0.68)],
        [int(fx), int(fyt + fh * 0.80)],
        [int(fx - cab_w), int(fyt + fh * 0.80)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [f_stripe], s_col, cv2.LINE_AA)
    # Lower steel snowplow/skirt & heavy coupler assembly
    skirt_f = np.array([
        [int(fx - cab_w * 0.96), int(fyb - fh * 0.03)],
        [int(fx - cab_w * 0.04), int(fyb)],
        [int(fx - cab_w * 0.12), int(fyb + fh * 0.07)],
        [int(fx - cab_w * 0.88), int(fyb + fh * 0.05)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [skirt_f], (20, 24, 30), cv2.LINE_AA)
    cv2.line(u8, (int(fx - cab_w * 0.92), int(fyb - fh * 0.02)), (int(fx - cab_w * 0.08), int(fyb + fh * 0.005)), (76, 90, 106), 1, cv2.LINE_AA)
    cv2.rectangle(u8, (int(fx - cab_w * 0.64), int(fyb - fh * 0.02)), (int(fx - cab_w * 0.36), int(fyb + fh * 0.055)), (32, 38, 48), -1)
    cv2.rectangle(u8, (int(fx - cab_w * 0.64), int(fyb - fh * 0.02)), (int(fx - cab_w * 0.36), int(fyb + fh * 0.055)), (86, 100, 118), 1)

    # Split cab windshield (cool wet glass reflection + angled windshield wipers!)
    for (u0, u1) in ((0.12, 0.45), (0.55, 0.88)):
        ws_pts = np.array([
            [int(fx - cab_w * u1), int(fyt + fh * 0.22)],
            [int(fx - cab_w * u0), int(fyt + fh * 0.20)],
            [int(fx - cab_w * u0), int(fyt + fh * 0.52)],
            [int(fx - cab_w * u1), int(fyt + fh * 0.53)],
        ], dtype=np.int32)
        cv2.fillPoly(u8, [ws_pts], (68, 84, 100), cv2.LINE_AA)
        # Diagonal wet rain-sheen glass reflection band
        sheen_w = np.array([
            [int(fx - cab_w * (u1 - 0.04)), int(fyt + fh * 0.23)],
            [int(fx - cab_w * (u1 - 0.14)), int(fyt + fh * 0.22)],
            [int(fx - cab_w * (u0 + 0.04)), int(fyt + fh * 0.51)],
            [int(fx - cab_w * (u0 + 0.14)), int(fyt + fh * 0.51)],
        ], dtype=np.int32)
        cv2.fillPoly(u8, [sheen_w], (102, 122, 140), cv2.LINE_AA)
        cv2.polylines(u8, [ws_pts], True, (22, 26, 34), 2, cv2.LINE_AA)
        # Crisp angled windshield wiper blade
        wp_base = (int(fx - cab_w * (0.5 * (u0 + u1) - 0.04)), int(fyt + fh * 0.52))
        wp_tip = (int(fx - cab_w * (u1 - 0.06)), int(fyt + fh * 0.29))
        cv2.line(u8, wp_base, wp_tip, (18, 22, 28), 2, cv2.LINE_AA)

    # Vertical metallic side grab-rails flanking the cab front
    for rail_u in (0.04, 0.96):
        rx_g = int(fx - cab_w * rail_u)
        cv2.line(u8, (rx_g, int(fyt + fh * 0.54)), (rx_g, int(fyt + fh * 0.84)), (134, 152, 170), 1, cv2.LINE_AA)

    # Recessed metallic headlight housings with chrome bezel
    for hl_off in (0.28, 0.72):
        hx_i = int(fx - cab_w * hl_off)
        hy_i = int(fyb - fh * 0.26)
        cv2.circle(u8, (hx_i, hy_i), max(4, int(fh * 0.045)), (26, 30, 38), -1, cv2.LINE_AA)
        cv2.circle(u8, (hx_i, hy_i), max(4, int(fh * 0.045)), (120, 134, 150), 1, cv2.LINE_AA)
        cv2.circle(u8, (hx_i, hy_i), max(3, int(fh * 0.032)), (255, 244, 206), -1, cv2.LINE_AA)

    # Pantograph on roof
    p_base = _lerp_side(0.18, -0.04)
    cv2.line(u8, p_base, (p_base[0] - int(fh * 0.08), p_base[1] - int(fh * 0.14)), (28, 34, 42), 2, cv2.LINE_AA)
    cv2.line(u8, (p_base[0] - int(fh * 0.08), p_base[1] - int(fh * 0.14)), (p_base[0] + int(fh * 0.06), p_base[1] - int(fh * 0.22)), (28, 34, 42), 2, cv2.LINE_AA)
    cv2.line(u8, (p_base[0] - int(fh * 0.12), p_base[1] - int(fh * 0.22)), (p_base[0] + int(fh * 0.18), p_base[1] - int(fh * 0.22)), (36, 44, 54), 2, cv2.LINE_AA)

    out = u8.astype(np.float32) / 255.0

    # Soft atmospheric rain-mist blending the distant receding rear cars into the storm
    yy_m, xx_m = np.mgrid[0:h, 0:w].astype(np.float32)
    rear_mist = np.clip((xx_m - (fx + (bx - fx) * 0.45)) / max(1.0, (bx - fx) * 0.65), 0.0, 1.0)
    rear_mist *= np.clip((yy_m - (byt - fh * 0.15)) / max(1.0, fh * 0.35), 0.0, 1.0)
    rear_mist *= np.clip((fyb + fh * 0.10 - yy_m) / max(1.0, fh * 0.35), 0.0, 1.0) * 0.38
    mist_col = np.array([0.16, 0.23, 0.30], dtype=np.float32)[None, None, :]
    out = out * (1.0 - rear_mist[..., None]) + mist_col * rear_mist[..., None]

    # Twin incandescent headlights with bright golden-white core + 35mm optical halation
    for hl_off in (0.28, 0.72):
        hx = fx - cab_w * hl_off
        hy = fyb - fh * 0.26
        out = _add_glow(out, hx, hy, fh * 0.14, fh * 0.14, (1.00, 0.86, 0.56), 0.90)
        out = _add_glow(out, hx, hy, fh * 0.045, fh * 0.045, (1.00, 0.98, 0.92), 0.99)

    return np.clip(out, 0.0, 1.0)


def draw_station_perspective(
    img: np.ndarray,
    platform_y_frac: float = 0.75,
    vp_xy: Tuple[float, float] = (1.18, 0.56),
) -> np.ndarray:
    """Draw a rustic wooden Japanese station awning, pillars, and receding wet stone platform in perspective."""
    h, w = img.shape[:2]
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    py = int(platform_y_frac * h)
    vpx, vpy = int(vp_xy[0] * w), int(vp_xy[1] * h)

    # Sloped wooden station roof awning receding from top-left toward right vanishing point
    roof_pts = np.array([
        [0, 0],
        [int(w * 0.72), 0],
        [int(w * 0.64), int(h * 0.26)],
        [0, int(h * 0.16)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [roof_pts], (24, 20, 20), cv2.LINE_AA)
    # Visible underside wooden rafters receding in perspective
    for rf in np.linspace(0.06, 0.62, 10):
        rx_eave = int(w * rf)
        ry_eave = int(h * (0.16 + 0.10 * (rf / 0.64)))
        rx_top = int(w * (rf + 0.08))
        cv2.line(u8, (rx_eave, ry_eave), (rx_top, 0), (42, 33, 26), 2, cv2.LINE_AA)
    # Wooden eaves trim & subtle warm-lit rain drip beads along lower edge
    cv2.line(u8, (0, int(h * 0.16)), (int(w * 0.64), int(h * 0.26)), (68, 52, 38), 3, cv2.LINE_AA)
    for df in np.linspace(0.04, 0.61, 14):
        dx_d = int(w * df)
        dy_d = int(h * (0.16 + 0.10 * (df / 0.64)))
        cv2.circle(u8, (dx_d, dy_d + 2), 1, (190, 170, 136), -1, cv2.LINE_AA)

    # Timber support pillars receding in perspective
    for (px_f, top_y_f, thick) in ((0.08, 0.17, 10), (0.31, 0.21, 7), (0.51, 0.24, 5)):
        px = int(px_f * w)
        ty = int(top_y_f * h)
        cv2.line(u8, (px, ty), (px, py + 4), (32, 25, 22), thick, cv2.LINE_AA)
        # Diagonal timber roof brace
        cv2.line(u8, (px, ty + int(h * 0.08)), (px + int(w * 0.04), ty), (38, 30, 26), max(2, thick // 2), cv2.LINE_AA)

    # Wet stone platform on the left receding cleanly beside the trackbed (never cutting under the train!)
    plat_edge_x = int(w * 0.44)
    plat_top_pts = np.array([
        [0, py],
        [plat_edge_x, py],
        [int(w * 0.56), int(py - h * 0.035)],
        [0, int(py - h * 0.035)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [plat_top_pts], (32, 37, 45), cv2.LINE_AA)
    # Yellow tactile safety strip along platform edge
    cv2.line(u8, (0, py + 2), (plat_edge_x, py + 2), (148, 120, 56), 2, cv2.LINE_AA)
    cv2.line(u8, (plat_edge_x, py + 2), (int(w * 0.56), int(py - h * 0.033)), (135, 108, 50), 2, cv2.LINE_AA)
    # Dark vertical concrete platform face dropping into the trackbed
    plat_face = np.array([
        [plat_edge_x, py],
        [int(w * 0.56), int(py - h * 0.035)],
        [int(w * 0.56), int(py + h * 0.015)],
        [plat_edge_x, int(py + h * 0.045)],
    ], dtype=np.int32)
    cv2.fillPoly(u8, [plat_face], (16, 19, 24), cv2.LINE_AA)

    return u8.astype(np.float32) / 255.0


def draw_train_profile(
    img: np.ndarray,
    x_start_frac: float,
    x_end_frac: float,
    y_bottom_frac: float,
    height_frac: float = 0.075,
    body_color: Tuple[float, float, float] = (0.14, 0.18, 0.26),
    stripe_color: Tuple[float, float, float] = (0.58, 0.28, 0.24),
    window_color: Tuple[float, float, float] = (1.00, 0.86, 0.54),
    headlight_left: bool = False,
    num_cars: int = 3,
    scale_detail: float = 1.0,
    rim_light_color: Optional[Tuple[float, float, float]] = None,
    rim_light_strength: float = 0.0,
) -> np.ndarray:
    """Draw a detailed multi-car vintage Japanese train with roof vents, pantograph, interior life & optional rim-light."""
    h, w = img.shape[:2]
    out = img.copy()

    x0 = x_start_frac * w
    x1 = x_end_frac * w
    yb = y_bottom_frac * h
    th = height_frac * h
    yt = yb - th

    # Soft warm window light spilling outward into the mist around the train
    out = _add_glow(out, 0.5 * (x0 + x1), yt + th * 0.5, (x1 - x0) * 0.58, th * 1.4, window_color, 0.28)

    u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    b_col = tuple(int(np.clip(c * 255, 0, 255)) for c in body_color)
    b_dark = tuple(int(np.clip(c * 165, 0, 255)) for c in body_color)
    r_col = tuple(int(np.clip(c * 285 + 16, 0, 255)) for c in body_color)
    s_col = tuple(int(np.clip(c * 255, 0, 255)) for c in stripe_color)
    w_col = tuple(int(np.clip(c * 255, 0, 255)) for c in window_color)
    rim_bgr = None
    if rim_light_color is not None and rim_light_strength > 0.0:
        rim_bgr = tuple(int(np.clip((c * rim_light_strength + b_col[i] / 255.0 * (1.0 - rim_light_strength)) * 255, 0, 255)) for i, c in enumerate(rim_light_color))

    gap = max(3.0, th * 0.11)
    car_w = ((x1 - x0) - gap * (num_cars - 1)) / max(1, num_cars)

    for c in range(num_cars):
        cx0 = int(x0 + c * (car_w + gap))
        cx1 = int(cx0 + car_w)
        # Dark undercarriage & bogie wheel assemblies
        cv2.rectangle(u8, (cx0 + 3, int(yb - th * 0.14)), (cx1 - 3, int(yb)), (16, 20, 28), -1)
        for bf in (0.20, 0.80):
            bx = int(cx0 + car_w * bf)
            cv2.rectangle(u8, (bx - int(th * 0.22), int(yb - th * 0.12)), (bx + int(th * 0.22), int(yb + 1)), (10, 13, 18), -1)

        # Car-to-car gangway bellows
        if c < num_cars - 1:
            cv2.rectangle(u8, (cx1, int(yt + th * 0.22)), (int(cx1 + gap + 1), int(yb - th * 0.16)), b_dark, -1)

        # Main car body with subtle vertical bevel
        cv2.rectangle(u8, (cx0, int(yt + th * 0.12)), (cx1, int(yb - th * 0.12)), b_col, -1)
        cv2.rectangle(u8, (cx0, int(yt + th * 0.12)), (cx1, int(yt + th * 0.20)), tuple(min(255, v + 14) for v in b_col), -1)
        # Curved roof cap & rooftop AC vents
        cv2.rectangle(u8, (cx0 + 2, int(yt)), (cx1 - 2, int(yt + th * 0.14)), r_col, -1)
        for vf in (0.35, 0.65):
            vx = int(cx0 + car_w * vf)
            cv2.rectangle(u8, (vx - int(th * 0.14), int(yt - th * 0.08)), (vx + int(th * 0.14), int(yt + 1)), r_col, -1)

        # Vintage stripe
        cv2.rectangle(u8, (cx0, int(yt + th * 0.66)), (cx1, int(yt + th * 0.78)), s_col, -1)

        # Recessed warm glowing windows with mullions & subtle passenger silhouette
        n_win = 5
        margin = car_w * 0.08
        win_span = (car_w - 2 * margin) / n_win
        for wi in range(n_win):
            wx0 = int(cx0 + margin + wi * win_span + win_span * 0.12)
            wx1 = int(cx0 + margin + (wi + 1) * win_span - win_span * 0.12)
            wy0 = int(yt + th * 0.24)
            wy1 = int(yt + th * 0.58)
            # Subtle per-window warmth variation
            wt = (
                min(255, int(w_col[0])),
                min(255, int(w_col[1] * (0.92 + 0.08 * (((c + wi) % 3) / 2.0)))),
                min(255, int(w_col[2] * (0.85 + 0.15 * ((c + wi) % 2)))),
            )
            cv2.rectangle(u8, (wx0, wy0), (wx1, wy1), wt, -1)
            # Subtle interior seat & seated silhouette in 2 windows
            cv2.rectangle(u8, (wx0, int(wy1 - (wy1 - wy0) * 0.28)), (wx1, wy1), (135, 92, 54), -1)
            if (c == 0 and wi == 2) or (c == 1 and wi == 1):
                sx = (wx0 + wx1) // 2
                sy = int(wy0 + (wy1 - wy0) * 0.56)
                cv2.circle(u8, (sx, sy), max(1, int(th * 0.08)), (72, 50, 38), -1, cv2.LINE_AA)
                cv2.ellipse(u8, (sx, int(sy + th * 0.14)), (max(2, int(th * 0.12)), max(1, int(th * 0.10))), 0, 180, 360, (66, 46, 36), -1, cv2.LINE_AA)
            # Window frame & transom bar
            cv2.line(u8, (wx0, int(wy0 + (wy1 - wy0) * 0.34)), (wx1, int(wy0 + (wy1 - wy0) * 0.34)), (78, 56, 38), 1, cv2.LINE_AA)

        # Detailed diamond pantograph on lead car
        if c == 0:
            px = int(cx0 + car_w * 0.28)
            p_col = rim_bgr if rim_bgr is not None else b_col
            cv2.line(u8, (px, int(yt)), (px - int(th * 0.16), int(yt - th * 0.15)), p_col, 1, cv2.LINE_AA)
            cv2.line(u8, (px - int(th * 0.16), int(yt - th * 0.15)), (px + int(th * 0.06), int(yt - th * 0.30)), p_col, 1, cv2.LINE_AA)
            cv2.line(u8, (px - int(th * 0.24), int(yt - th * 0.30)), (px + int(th * 0.28), int(yt - th * 0.30)), p_col, 1, cv2.LINE_AA)

        # Optional luminous golden-hour rim-light along roof & sunlit edges
        if rim_bgr is not None:
            cv2.line(u8, (cx0 + 1, int(yt)), (cx1 - 1, int(yt)), rim_bgr, 3, cv2.LINE_AA)
            cv2.line(u8, (cx0, int(yt + th * 0.08)), (cx0, int(yb - th * 0.12)), rim_bgr, 2, cv2.LINE_AA)
            cv2.line(u8, (cx1, int(yt + th * 0.08)), (cx1, int(yb - th * 0.12)), rim_bgr, 1, cv2.LINE_AA)

    out = u8.astype(np.float32) / 255.0
    if rim_light_color is not None and rim_light_strength > 0.0:
        out = _add_glow(out, 0.5 * (x0 + x1), yt, (x1 - x0) * 0.52, th * 0.42, rim_light_color, 0.42 * rim_light_strength)
    if headlight_left:
        out = _add_glow(out, x0, yt + th * 0.62, th * 1.1, th * 0.55, (1.0, 0.92, 0.72), 0.78)
    return np.clip(out, 0.0, 1.0)


def draw_station_silhouette(img: np.ndarray, platform_y_frac: float = 0.73, awning_y_frac: float = 0.21) -> np.ndarray:
    """Backwards-compatible alias calling draw_station_perspective."""
    return draw_station_perspective(img, platform_y_frac=platform_y_frac)


def draw_petals_and_motes(
    img: np.ndarray,
    time_s: float = 0.0,
    num_petals: int = 58,
    num_motes: int = 75,
    petal_color: Tuple[float, float, float] = (1.00, 0.80, 0.84),
    mote_color: Tuple[float, float, float] = (1.00, 0.94, 0.72),
    seed: int = 33,
) -> np.ndarray:
    """Render unmistakable, graceful pink-white & golden-backlit sakura petals across 3 DoF planes + golden dust motes."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = img.copy()

    # 1. Luminous golden sunlit dust motes (soft bokeh specks)
    mote_layer = np.zeros((h, w), dtype=np.float32)
    for _ in range(num_motes):
        mx = int(rng.uniform(0.02 * w, 0.98 * w))
        my = int(rng.uniform(0.04 * h, 0.92 * h))
        mr = max(1, int(rng.uniform(1.4, 4.5)))
        amp = float(rng.uniform(0.28, 0.85))
        cv2.circle(mote_layer, (mx, my), mr, amp, -1, cv2.LINE_AA)
    mote_layer = cv2.GaussianBlur(mote_layer, (0, 0), sigmaX=1.8, sigmaY=1.8)
    mc = np.array(mote_color, dtype=np.float32)[None, None, :]
    out = 1.0 - (1.0 - out) * (1.0 - np.clip(mote_layer[..., None] * mc * 0.72, 0.0, 0.90))

    # 2. Unmistakable, graceful notched sakura petals across 3 distinct DoF planes
    planes = [
        (int(num_petals * 0.30), 3.0, 5.5, 0.3, 0.58),    # Tiny distant drifting petals
        (int(num_petals * 0.55), 7.5, 13.5, 0.4, 0.88),   # Crisp translucent mid-ground petals with backlit edges
        (6,                      26.0, 42.0, 6.8, 0.52),  # 6 large, softly Gaussian-blurred foreground bokeh petals!
    ]
    pc_pink = np.array(petal_color, dtype=np.float32)[None, None, :]
    pc_gold = np.array([1.00, 0.94, 0.78], dtype=np.float32)[None, None, :]

    for (count, rmin, rmax, blur_s, alpha_s) in planes:
        p_body = np.zeros((h, w), dtype=np.float32)
        p_rim = np.zeros((h, w), dtype=np.float32)
        for _ in range(count):
            px = rng.uniform(0.02 * w, 0.98 * w)
            py = rng.uniform(0.04 * h, 0.90 * h)
            rx = rng.uniform(rmin, rmax)
            # Natural 3D tumbling perspective foreshortening
            foreshorten = rng.uniform(0.36, 0.78)
            ry = rx * foreshorten
            curl = rng.uniform(-0.26, 0.26) * ry
            ang = rng.uniform(-55.0, 35.0)
            rad = np.radians(ang)
            ca, sa = np.cos(rad), np.sin(rad)

            # Classic curved sakura petal with tapered base and delicate V-notch at the tip
            local = np.array([
                [-rx, 0.0],                               # tapered stem base
                [-rx * 0.48, -ry * 0.68 + curl * 0.3],    # upper-left curve
                [rx * 0.18, -ry * 0.96 + curl],           # upper lobe peak
                [rx * 0.94, -ry * 0.56 + curl * 0.5],     # upper tip lobe
                [rx * 0.68, 0.0 + curl * 0.2],            # unmistakable sakura V-notch!
                [rx * 0.94, ry * 0.56 + curl * 0.5],      # lower tip lobe
                [rx * 0.18, ry * 0.96 + curl],            # lower lobe peak
                [-rx * 0.48, ry * 0.68 + curl * 0.3],     # lower-left curve
            ], dtype=np.float32)
            rot = np.stack([
                local[:, 0] * ca - local[:, 1] * sa + px,
                local[:, 0] * sa + local[:, 1] * ca + py,
            ], axis=-1).astype(np.int32)
            val = float(rng.uniform(0.72, 1.00))
            cv2.fillPoly(p_body, [rot], val, cv2.LINE_AA)
            # Warm golden backlit rim along upper curve
            cv2.polylines(p_rim, [rot[:5]], False, val, max(1, int(rx / 10.0)), cv2.LINE_AA)

        if blur_s > 0.0:
            p_body = cv2.GaussianBlur(p_body, (0, 0), sigmaX=blur_s, sigmaY=blur_s)
            p_rim = cv2.GaussianBlur(p_rim, (0, 0), sigmaX=blur_s, sigmaY=blur_s)

        a_b = np.clip(p_body * alpha_s, 0.0, 0.90)[..., None]
        a_r = np.clip(p_rim * alpha_s * 0.95, 0.0, 0.88)[..., None]
        # Translucent warm-backlit blend (lets golden sky glow through!)
        out = np.clip(out * (1.0 - a_b * 0.56) + pc_pink * a_b * 0.84 + pc_gold * a_r * 0.62, 0.0, 1.0)

    return out
