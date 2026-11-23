"""Act VIII — QUIET RESOLUTION & TITLE (Shots S38–S40, 04:35.0–05:00.0).

Palette: Handmade Paper Cream #F4EFE6, Charcoal Ink #232120.
S38: Final gentle water ripple settling to absolute stillness.
S39: Soft watercolor wash dissolve into warm handmade paper #F4EFE6.
S40: Delicate minimal serif title card: THE LAST RAIN TRAIN (最後の雨列車)
     using FreeSerif.ttf + NotoSerifCJK-Regular.ttc holding gracefully to 300.0s.
"""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..painterly.clouds import render_sky_gradient
from ..painterly.paper_grain import generate_paper_grain

PAPER_CREAM_RGB = (244.0 / 255.0, 239.0 / 255.0, 230.0 / 255.0)  # #F4EFE6
CHARCOAL_INK_RGB = (35, 33, 32)  # #232120

LATIN_FONT_PATH = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
CJK_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"


def _make_handmade_paper(width: int, height: int, grain: np.ndarray) -> np.ndarray:
    """Create warm handmade cold-press watercolor paper (#F4EFE6) with subtle deckle warmth."""
    base = np.empty((height, width, 3), dtype=np.float32)
    base[..., 0] = PAPER_CREAM_RGB[0]
    base[..., 1] = PAPER_CREAM_RGB[1]
    base[..., 2] = PAPER_CREAM_RGB[2]

    # Subtle warm edge deckle vignette
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    nx = (xx / max(width - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(height - 1, 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    edge_warm = np.clip((r2 - 0.45) * 0.045, 0.0, 0.07)
    base[..., 0] -= edge_warm * 0.45
    base[..., 1] -= edge_warm * 0.65
    base[..., 2] -= edge_warm * 0.95

    # Tactile cold-press watercolor paper fibers
    paper = np.clip(base + (grain[..., None] - 0.5) * 0.048, 0.0, 1.0)
    return paper


def _render_title_card_rgb(width: int, height: int, paper: np.ndarray) -> np.ndarray:
    """Pre-render crisp minimal serif title card on handmade paper."""
    pil = Image.fromarray(np.clip(paper * 255.0, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil)

    scale = height / 1080.0
    en_size = max(14, int(round(68 * scale)))
    jp_size = max(12, int(round(36 * scale)))

    en_font = ImageFont.truetype(LATIN_FONT_PATH, en_size)
    jp_font = ImageFont.truetype(CJK_FONT_PATH, jp_size)

    en_text = "THE LAST RAIN TRAIN"
    jp_text = "最後の雨列車"

    en_bbox = draw.textbbox((0, 0), en_text, font=en_font)
    en_w = en_bbox[2] - en_bbox[0]
    en_h = en_bbox[3] - en_bbox[1]

    jp_bbox = draw.textbbox((0, 0), jp_text, font=jp_font)
    jp_w = jp_bbox[2] - jp_bbox[0]
    jp_h = jp_bbox[3] - jp_bbox[1]

    cy = height * 0.47
    en_x = (width - en_w) * 0.5 - en_bbox[0]
    en_y = cy - en_h - 24 * scale - en_bbox[1]
    draw.text((en_x, en_y), en_text, fill=CHARCOAL_INK_RGB, font=en_font)

    # Delicate thin charcoal ink hairline divider
    line_half = int(round(width * 0.085))
    line_y = int(round(cy + 2 * scale))
    line_w = max(1, int(round(1.5 * scale)))
    draw.line(
        [(width // 2 - line_half, line_y), (width // 2 + line_half, line_y)],
        fill=(78, 74, 70),
        width=line_w,
    )

    jp_x = (width - jp_w) * 0.5 - jp_bbox[0]
    jp_y = cy + 24 * scale - jp_bbox[1]
    draw.text((jp_x, jp_y), jp_text, fill=(52, 49, 47), font=jp_font)

    return np.array(pil, dtype=np.float32) / 255.0


def prepare_shot_cache(shot_spec: Any, width: int = 1920, height: int = 1080) -> Dict[str, Any]:
    """Pre-render handmade paper, twilight water base, watercolor bleed field, and title card."""
    sid = shot_spec.id
    seed = 800 + int(sid[1:])
    grain = generate_paper_grain(height, width, seed=seed)
    paper = _make_handmade_paper(width, height, grain)
    cache: Dict[str, Any] = {"id": sid, "w": width, "h": height, "grain": grain, "paper": paper}

    if sid in ("S38", "S39"):
        # Quiet twilight glassy water base
        water = render_sky_gradient(
            width, height,
            stops=[
                (0.00, (0.09, 0.11, 0.20)),
                (0.45, (0.20, 0.19, 0.32)),
                (0.78, (0.36, 0.30, 0.44)),
                (1.00, (0.14, 0.15, 0.26)),
            ],
            seed=38,
            brush_strength=0.022,
        )
        # Subtle warm reflected sky glow in center
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        nx = (xx / max(width - 1, 1)) - 0.5
        ny = (yy / max(height - 1, 1)) - 0.5
        glow = np.exp(-(nx * nx / 0.18 + ny * ny / 0.14)) * 0.22
        water = np.clip(water + glow[..., None] * np.array([0.96, 0.82, 0.74], dtype=np.float32)[None, None, :], 0.0, 1.0)
        cache["water"] = water

        if sid == "S39":
            # Organic watercolor wash bleed field (low-frequency multi-octave noise + radial)
            rng = np.random.default_rng(839)
            low = rng.uniform(0.0, 1.0, (max(8, height // 48), max(8, width // 48))).astype(np.float32)
            mid = rng.uniform(0.0, 1.0, (max(16, height // 18), max(16, width // 18))).astype(np.float32)
            n1 = cv2.resize(low, (width, height), interpolation=cv2.INTER_CUBIC)
            n2 = cv2.resize(mid, (width, height), interpolation=cv2.INTER_CUBIC)
            radial = np.sqrt(nx * nx * 1.4 + ny * ny * 1.4)
            bleed_field = np.clip(0.48 * radial + 0.34 * n1 + 0.18 * n2, 0.0, 1.0)
            cache["bleed_field"] = bleed_field

    elif sid == "S40":
        cache["title_rgb"] = _render_title_card_rgb(width, height, paper)

    return cache


def render_shot_frame(
    shot_spec: Any,
    local_t: float,
    global_t: float,
    width: int = 1920,
    height: int = 1080,
    cache: Dict[str, Any] | None = None,
) -> np.ndarray:
    """Render one 60fps frame for Act VIII (S38–S40) as uint8 RGB (height, width, 3)."""
    if cache is None or cache.get("w") != width or cache.get("h") != height or cache.get("id") != shot_spec.id:
        cache = prepare_shot_cache(shot_spec, width, height)

    sid = shot_spec.id
    dur = max(shot_spec.duration_s, 1e-3)
    prog = float(np.clip(local_t / dur, 0.0, 1.0))
    grain = cache["grain"]

    if sid == "S38":
        # Final gentle water ripple expanding and settling to absolute glassy stillness
        img = cache["water"].copy()
        decay = float(np.exp(-2.8 * prog))
        if decay > 0.01:
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            dx = (xx - width * 0.5) / max(height, 1)
            dy = (yy - height * 0.52) / max(height, 1)
            r = np.sqrt(dx * dx + (dy * 1.45) ** 2)
            front = 0.05 + 0.52 * prog
            ring = np.exp(-((r - front) / 0.045) ** 2) * np.cos((r - front) * 58.0) * decay
            highlight = np.clip(ring, 0.0, 1.0) * 0.28
            shadow = np.clip(-ring, 0.0, 1.0) * 0.14
            img = np.clip(
                img + highlight[..., None] * np.array([0.96, 0.94, 0.90], dtype=np.float32)[None, None, :] - shadow[..., None],
                0.0,
                1.0,
            )
        img = np.clip(img + (grain[..., None] - 0.5) * 0.026, 0.0, 1.0)
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)

    elif sid == "S39":
        # Soft watercolor wash dissolve into warm handmade paper #F4EFE6
        water = cache["water"]
        paper = cache["paper"]
        field = cache["bleed_field"]
        # Smooth organic watercolor front expanding from 0 to 1
        thresh = -0.15 + 1.30 * (prog * prog * (3.0 - 2.0 * prog))
        mask = np.clip((thresh - field) / 0.22, 0.0, 1.0)
        # Subtle warm peach wet-on-wet pigment edge along the dissolving boundary
        edge = np.exp(-((mask - 0.5) / 0.16) ** 2) * (1.0 - prog * 0.65)
        wash = water * (1.0 - mask[..., None]) + paper * mask[..., None]
        wash = np.clip(
            wash + edge[..., None] * np.array([0.14, 0.08, 0.04], dtype=np.float32)[None, None, :],
            0.0,
            1.0,
        )
        return np.clip(wash * 255.0, 0, 255).astype(np.uint8)

    else:
        # S40: Minimal serif title card holding gracefully to 300.0s with gentle paper-grain breathe
        paper = cache["paper"]
        title_rgb = cache["title_rgb"]
        # Smooth ink fade-in over first 2.2s, then serene hold to 300.0s
        fade_in = float(np.clip(local_t / 2.2, 0.0, 1.0))
        fade_in = fade_in * fade_in * (3.0 - 2.0 * fade_in)
        breathe = 1.0 + 0.004 * np.sin(local_t * 1.1)
        img = np.clip((paper * (1.0 - fade_in) + title_rgb * fade_in) * breathe, 0.0, 1.0)
        return np.clip(img * 255.0, 0, 255).astype(np.uint8)
