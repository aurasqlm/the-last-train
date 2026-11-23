"""Authentic cold-press watercolor paper grain, pigment granulation, and gouache brushwork."""

from __future__ import annotations

import cv2
import numpy as np


def _value_noise(h: int, w: int, grid_h: int, grid_w: int, rng: np.random.Generator) -> np.ndarray:
    small = rng.standard_normal((max(2, grid_h), max(2, grid_w))).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def generate_paper_grain(height: int, width: int, seed: int = 42) -> np.ndarray:
    """Generate multi-frequency cold-press watercolor paper tooth with directional fibers.

    Returns a float32 array in roughly [-1, 1] combining:
    - Coarse cotton paper tooth (dimples & deckle valleys)
    - Medium anisotropic horizontal/vertical cellulose fibers
    - Fine micro-grain with subtle raking-light emboss relief
    """
    rng = np.random.default_rng(seed)

    # Coarse cotton dimples (subtle)
    coarse = _value_noise(height, width, max(4, height // 24), max(4, width // 24), rng)
    # Medium cold-press tooth
    medium = _value_noise(height, width, max(8, height // 8), max(8, width // 8), rng)
    # Fine tooth
    fine = _value_noise(height, width, max(16, height // 3), max(16, width // 3), rng)
    # Micro paper fibers (anisotropic horizontal + vertical weave)
    raw = rng.standard_normal((height, width)).astype(np.float32)
    fib_h = cv2.GaussianBlur(raw, (0, 0), sigmaX=4.0, sigmaY=0.6)
    fib_v = cv2.GaussianBlur(rng.standard_normal((height, width)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=3.6)

    tooth = 0.14 * coarse + 0.30 * medium + 0.32 * fine + 0.12 * fib_h + 0.12 * fib_v

    # Subtle raking-light emboss (top-left light catching cold-press ridges)
    gx = cv2.Sobel(tooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tooth, cv2.CV_32F, 0, 1, ksize=3)
    relief = -0.6 * gx - 0.8 * gy
    relief = relief / (np.std(relief) + 1e-6)

    combined = 0.72 * (tooth / (np.std(tooth) + 1e-6)) + 0.28 * relief
    combined = combined / (np.std(combined) + 1e-6)
    return np.clip(combined, -2.5, 2.5).astype(np.float32)


def apply_paper_grain(
    img: np.ndarray,
    grain: np.ndarray,
    strength: float = 0.055,
    warm_tint: float = 0.015,
) -> np.ndarray:
    """Apply cold-press watercolor paper tooth strongest in midtones and highlights."""
    if grain.shape[:2] != img.shape[:2]:
        grain = cv2.resize(grain, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    # Paper tooth shows most naturally in midtones & warm highlights, gently attenuated in deep shadows
    weight = (0.35 + 0.65 * np.clip(lum * 1.4, 0.0, 1.0)) * (1.0 - 0.25 * np.clip((lum - 0.85) * 4.0, 0.0, 1.0))
    delta = (grain * weight * strength)[..., None]
    # Slight warm cream paper bias in ridges
    tint = np.array([1.0 + warm_tint, 1.0, 1.0 - warm_tint * 0.8], dtype=np.float32)[None, None, :]
    return np.clip(img + delta * tint, 0.0, 1.0)


def apply_pigment_granulation(
    img: np.ndarray,
    grain: np.ndarray,
    strength: float = 0.045,
) -> np.ndarray:
    """Simulate watercolor/gouache pigment settling into paper valleys and edge darkening."""
    if grain.shape[:2] != img.shape[:2]:
        grain = cv2.resize(grain, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Pigment pools in valleys (negative grain)
    valleys = np.clip(-grain, 0.0, 2.0) / 2.0
    # Watercolor edge-darkening bloom along tonal gradients
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.sqrt(gx * gx + gy * gy) * 2.5, 0.0, 1.0)
    edge = cv2.GaussianBlur(edge, (0, 0), sigmaX=1.8, sigmaY=1.8)

    darken = 1.0 - (valleys * strength + edge * strength * 0.45)[..., None]
    return np.clip(img * darken, 0.0, 1.0)


def apply_gouache_brushwork(
    img: np.ndarray,
    strength: float = 0.032,
    angle_deg: float = -12.0,
    scale: float = 1.0,
    seed: int = 7,
) -> np.ndarray:
    """Overlay subtle directional gouache dry-brush impasto texture to eliminate flat digital gradients."""
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)

    # Broad horizontal-ish painterly strokes
    gh = max(4, int(h / (22 * scale)))
    gw = max(4, int(w / (90 * scale)))
    broad = _value_noise(h, w, gh, gw, rng)

    # Fine bristle streaks
    raw = rng.standard_normal((h, w)).astype(np.float32)
    bristle = cv2.GaussianBlur(raw, (0, 0), sigmaX=11.0 * scale, sigmaY=1.1 * scale)
    bristle = bristle / (np.std(bristle) + 1e-6)

    strokes = 0.55 * (broad / (np.std(broad) + 1e-6)) + 0.45 * bristle

    if abs(angle_deg) > 0.5:
        M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle_deg, 1.0)
        strokes = cv2.warpAffine(strokes, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Slight warm/cool bristle color variation
    ch_var = np.stack([strokes, strokes * 0.92, strokes * 0.85], axis=-1)
    return np.clip(img + ch_var * strength, 0.0, 1.0)
