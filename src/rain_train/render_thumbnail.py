"""Single-pass high-impact 1920x1080 YouTube thumbnail renderer for THE LAST RAIN TRAIN."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rain_train.acts import prepare_shot_cache, render_shot_frame
from rain_train.storyboard import SHOTS


def _grade_thumbnail(img_f32: np.ndarray) -> np.ndarray:
    """Boost warm-vs-cool separation, micro-contrast, and cinematic vignette for tiny thumbnail pop."""
    h, w, _ = img_f32.shape
    out = np.clip(img_f32, 0.0, 1.0).copy()

    # S-curve contrast boost
    out = np.clip((out - 0.5) * 1.18 + 0.5, 0.0, 1.0)

    # Split-tone: cool deep twilight cobalt in shadows/water, warm golden-apricot in highlights
    luma = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    shadow_w = np.clip(1.0 - luma * 1.6, 0.0, 1.0)[..., None]
    hi_w = np.clip((luma - 0.42) * 1.7, 0.0, 1.0)[..., None]

    # Deepen shadows toward rich cobalt
    out[..., 0:1] = np.clip(out[..., 0:1] - 0.05 * shadow_w + 0.05 * hi_w, 0.0, 1.0)
    out[..., 1:2] = np.clip(out[..., 1:2] - 0.01 * shadow_w + 0.02 * hi_w, 0.0, 1.0)
    out[..., 2:3] = np.clip(out[..., 2:3] + 0.09 * shadow_w - 0.04 * hi_w, 0.0, 1.0)

    # Saturation boost in HSV
    u8 = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.24, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # Subtle top-sky darkening gradient so title typography has razor-sharp contrast
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    top_grad = np.clip(1.0 - yy / 0.36, 0.0, 1.0) ** 1.6
    out = out * (1.0 - 0.34 * top_grad[..., None])

    # Cinematic radial vignette
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    yy_c = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    r2 = xx * xx * 0.75 + yy_c * yy_c
    vig = np.clip(1.0 - 0.32 * np.clip(r2 - 0.35, 0.0, 2.0), 0.55, 1.0)
    out = np.clip(out * vig[..., None], 0.0, 1.0)

    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _draw_tracked_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    y: int,
    font: ImageFont.FreeTypeFont,
    fill: tuple,
    tracking: int = 0,
) -> None:
    widths = [draw.textbbox((0, 0), ch, font=font)[2] for ch in text]
    total_w = sum(widths) + tracking * max(0, len(text) - 1)
    x = int(center_x - total_w / 2)
    for ch, cw in zip(text, widths):
        draw.text((x, y), ch, font=font, fill=fill)
        x += cw + tracking


def render_thumbnail() -> None:
    w, h = 1920, 1080
    # Hero golden-hour flooded sea tracks shot (S33)
    shot = next(s for s in SHOTS if s.id == "S33")
    cache = prepare_shot_cache(shot, w, h)
    local_t = 0.52 * shot.duration_s
    global_t = shot.start_s + local_t
    frame = render_shot_frame(shot, local_t, global_t, w, h, cache=cache)
    if frame.dtype == np.uint8:
        frame = frame.astype(np.float32) / 255.0

    graded = _grade_thumbnail(frame)
    base = Image.fromarray(graded, mode="RGB").convert("RGBA")

    title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", 112)
    jp_font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 44)

    title_text = "THE LAST RAIN TRAIN"
    jp_text = "最後の雨列車"

    # Soft multi-pass shadow layer for razor-sharp legibility at mobile sizes
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    _draw_tracked_text(sdraw, title_text, w // 2, 74, title_font, (8, 12, 26, 220), tracking=12)
    _draw_tracked_text(sdraw, jp_text, w // 2, 206, jp_font, (8, 12, 26, 210), tracking=16)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=14))

    # Crisp foreground typography in warm ivory cream
    fg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(fg)
    _draw_tracked_text(fdraw, title_text, w // 2, 70, title_font, (255, 248, 232, 255), tracking=12)
    # Subtle thin golden hairline divider
    fdraw.rectangle((w // 2 - 140, 194, w // 2 + 140, 196), fill=(248, 208, 138, 210))
    _draw_tracked_text(fdraw, jp_text, w // 2, 204, jp_font, (255, 240, 214, 240), tracking=16)

    composed = Image.alpha_composite(Image.alpha_composite(base, shadow), fg).convert("RGB")
    composed.save("thumbnail.png", format="PNG", optimize=True)
    composed.save("thumbnail.jpg", format="JPEG", quality=92, optimize=True, progressive=True)

    png_mb = Path("thumbnail.png").stat().st_size / (1024 * 1024)
    jpg_mb = Path("thumbnail.jpg").stat().st_size / (1024 * 1024)
    print(f"Saved thumbnail.png ({png_mb:.2f} MB) and thumbnail.jpg ({jpg_mb:.2f} MB)", flush=True)


if __name__ == "__main__":
    render_thumbnail()
