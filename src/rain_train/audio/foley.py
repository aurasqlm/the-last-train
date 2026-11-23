"""Shot-accurate procedural stereo foley synced directly to storyboard.py across all 8 acts."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal

from rain_train.storyboard import ACT_RANGES, get_shot

from . import N_SAMPLES, SR
from .mastering import pan_mono_to_stereo


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Fast 1/f pink noise via Voss-McCartney / spectral shaping."""
    white = rng.standard_normal(n).astype(np.float32)
    # 3-pole Paul Kellet approximation in vectorized SOS/IIR or fast FFT
    # For 14.4M samples, a 1st-order + 2nd-order shelf approximation via sosfilt is super fast and low-memory
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786], dtype=np.float64)
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float64)
    sos = signal.tf2sos(b, a)
    pink = signal.sosfilt(sos, white).astype(np.float32)
    pink /= float(np.std(pink)) + 1e-9
    return pink


def _smooth_window(n_samples: int, start_s: float, end_s: float, fade_in_s: float = 1.0, fade_out_s: float = 1.0, sr: int = SR) -> np.ndarray:
    """Create a smooth raised-cosine [0, 1] envelope for a time interval."""
    env = np.zeros(n_samples, dtype=np.float32)
    i0 = max(0, int(round(start_s * sr)))
    i1 = min(n_samples, int(round(end_s * sr)))
    if i1 <= i0:
        return env
    env[i0:i1] = 1.0
    fi = min(int(round(fade_in_s * sr)), (i1 - i0) // 2)
    fo = min(int(round(fade_out_s * sr)), (i1 - i0) // 2)
    if fi > 1:
        env[i0 : i0 + fi] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, fi, dtype=np.float32)))
    if fo > 1:
        env[i1 - fo : i1] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fo, dtype=np.float32)))
    return env


def _add_event(bus: np.ndarray, stereo_event: np.ndarray, time_s: float, sr: int = SR) -> None:
    """Mix a stereo event into bus at time_s."""
    idx = int(round(time_s * sr))
    if idx < 0 or idx >= bus.shape[1]:
        return
    end = min(idx + stereo_event.shape[1], bus.shape[1])
    length = end - idx
    if length > 0:
        bus[:, idx:end] += stereo_event[:, :length]


def _synth_thunder(dur_s: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Distant rolling thunder: 35-110 Hz filtered noise with multi-tap rumble."""
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.standard_normal(n).astype(np.float64)
    sos = signal.butter(3, [32.0, 115.0], btype="band", fs=sr, output="sos")
    rumble = signal.sosfilt(sos, noise).astype(np.float32)

    # Multi-tap envelope for rolling echoes
    env = np.exp(-t / (dur_s * 0.35)) * (1.0 - np.exp(-t / 0.18))
    env *= 0.65 + 0.35 * np.sin(2.0 * np.pi * 0.7 * t) ** 2
    rumble *= env * 0.32
    return pan_mono_to_stereo(rumble, pan=float(rng.uniform(-0.3, 0.3)), sr=sr)


def _synth_raindrop_impact(
    band_lo: float,
    band_hi: float,
    dur_s: float,
    amp: float,
    pan: float,
    rng: np.random.Generator,
    sr: int = SR,
) -> np.ndarray:
    """Realistic raindrop impact on stone or glass: crisp broadband noise-burst transient (ZERO sine bloops)."""
    n = max(int(round(dur_s * sr)), 64)
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.standard_normal(n).astype(np.float64)
    sos_hi = signal.butter(2, [band_lo, min(band_hi, sr * 0.44)], btype="band", fs=sr, output="sos")
    burst = signal.sosfilt(sos_hi, noise).astype(np.float32)
    # Sharp 0.8 ms attack, crisp 4-10 ms decay
    env = (1.0 - np.exp(-t / 0.0008)) * np.exp(-t / max(dur_s * 0.22, 0.004))
    sig = burst * env * amp
    return pan_mono_to_stereo(sig, pan=pan, sr=sr)


def _synth_puddle_splash(amp: float, pan: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Close-up puddle splash: crisp high-frequency water spray + tiny damped splash body."""
    dur_s = float(rng.uniform(0.045, 0.085))
    n = int(round(dur_s * sr))
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.standard_normal(n).astype(np.float64)
    sos_spray = signal.butter(2, [2400.0, 11000.0], btype="band", fs=sr, output="sos")
    sos_body = signal.butter(2, [650.0, 1800.0], btype="band", fs=sr, output="sos")
    spray = signal.sosfilt(sos_spray, noise).astype(np.float32) * np.exp(-t / 0.014)
    body = signal.sosfilt(sos_body, noise).astype(np.float32) * np.exp(-t / 0.009)
    sig = (spray * 0.75 + body * 0.45) * (1.0 - np.exp(-t / 0.001)) * amp
    return pan_mono_to_stereo(sig, pan=pan, sr=sr)


def _synth_umbrella_open(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Umbrella opening whoosh + taut fabric snap."""
    n = int(round(0.75 * sr))
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.standard_normal(n).astype(np.float64)
    sos = signal.butter(2, [450.0, 3800.0], btype="band", fs=sr, output="sos")
    whoosh = signal.sosfilt(sos, noise).astype(np.float32)
    env = np.maximum(np.sin(np.pi * np.clip(t / 0.35, 0.0, 1.0)), 0.0) ** 1.6 * np.exp(-np.maximum(t - 0.18, 0.0) / 0.12)
    # Taut fabric snap at 0.24s
    snap_idx = int(0.24 * sr)
    snap_len = int(0.06 * sr)
    t_s = np.arange(snap_len, dtype=np.float32) / sr
    snap_noise = signal.sosfilt(signal.butter(2, [180.0, 1200.0], btype="band", fs=sr, output="sos"), rng.standard_normal(snap_len)).astype(np.float32)
    snap = snap_noise * np.exp(-t_s / 0.012) * 0.32
    mono = whoosh * env * 0.28
    mono[snap_idx : snap_idx + snap_len] += snap
    return pan_mono_to_stereo(mono, pan=-0.10, sr=sr)


def _synth_lantern_click(sr: int = SR) -> np.ndarray:
    """Warm brass station lantern relay click."""
    n = int(round(0.18 * sr))
    t = np.arange(n, dtype=np.float32) / sr
    click = (
        np.sin(2.0 * np.pi * 1650.0 * t) * np.exp(-t / 0.007) * 0.18
        + np.sin(2.0 * np.pi * 3400.0 * t) * np.exp(-t / 0.004) * 0.12
        + np.sin(2.0 * np.pi * 420.0 * t) * np.exp(-t / 0.016) * 0.12
    ).astype(np.float32)
    return pan_mono_to_stereo(click, pan=0.16, sr=sr)


def _synth_station_bell(sr: int = SR) -> np.ndarray:
    """Clear, resonant Japanese rural station chime/bell (two warm strikes 1.7s apart)."""
    n = int(round(5.5 * sr))
    t_full = np.arange(n, dtype=np.float32) / sr
    mono = np.zeros(n, dtype=np.float32)
    # Two strikes: E5 (659.25 Hz) then A5 (880 Hz) gentle rural station chime
    strikes = [
        (0.0, 659.25, 0.20),
        (1.7, 880.00, 0.24),
    ]
    for t0, f0, g in strikes:
        idx = int(round(t0 * sr))
        t = t_full[: n - idx]
        partials = [
            (f0 * 1.000, 1.00, 2.2),
            (f0 * 2.004, 0.45, 1.4),
            (f0 * 2.996, 0.22, 0.85),
            (f0 * 4.012, 0.10, 0.45),
        ]
        strike_sig = np.zeros_like(t)
        for f, a, tau in partials:
            strike_sig += (a * np.sin(2.0 * np.pi * f * t) * np.exp(-t / tau)).astype(np.float32)
        att = min(int(0.005 * sr), len(strike_sig))
        strike_sig[:att] *= np.linspace(0.0, 1.0, att, dtype=np.float32)
        mono[idx:] += strike_sig * g
    return pan_mono_to_stereo(mono, pan=-0.22, sr=sr)


def _synth_rail_clack_pair(amp: float, pan: float, rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Authentic 2-bogie rail joint 'ba-dum ... ba-dum' (low-mid thump + crisp steel rail-joint click)."""
    n = int(round(0.55 * sr))
    mono = np.zeros(n, dtype=np.float32)
    noise = rng.standard_normal(n).astype(np.float64)
    sos_low = signal.butter(2, [75.0, 360.0], btype="band", fs=sr, output="sos")
    sos_click = signal.butter(2, [900.0, 2800.0], btype="band", fs=sr, output="sos")
    thump_src = signal.sosfilt(sos_low, noise).astype(np.float32)
    click_src = signal.sosfilt(sos_click, noise).astype(np.float32)
    for offset_s, g in [(0.0, 1.0), (0.18, 0.84)]:
        idx = int(round(offset_s * sr))
        dur = int(round(0.11 * sr))
        t = np.arange(dur, dtype=np.float32) / sr
        env_lo = np.exp(-t / 0.022) * (1.0 - np.exp(-t / 0.0025))
        env_hi = np.exp(-t / 0.008) * (1.0 - np.exp(-t / 0.0010))
        if idx + dur <= n:
            mono[idx : idx + dur] += (thump_src[idx : idx + dur] * env_lo + click_src[idx : idx + dur] * env_hi * 0.55) * amp * g
    return pan_mono_to_stereo(mono, pan=pan, sr=sr)


def _synth_doppler_whoosh(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    """Crisp tunnel lamp Doppler light-whoosh sweeping L -> R."""
    n = int(round(0.52 * sr))
    t = np.arange(n, dtype=np.float32) / sr
    noise = rng.standard_normal(n).astype(np.float64)
    sos = signal.butter(2, [320.0, 4200.0], btype="band", fs=sr, output="sos")
    base = signal.sosfilt(sos, noise).astype(np.float32)
    env = np.maximum(np.sin(np.pi * t / 0.52), 0.0) ** 2.0
    pan_curve = np.linspace(-0.65, 0.65, n, dtype=np.float32)
    angle = (pan_curve + 1.0) * (np.pi / 4.0)
    sig = base * env * 0.30
    out = np.empty((2, n), dtype=np.float32)
    out[0] = sig * np.cos(angle)
    out[1] = sig * np.sin(angle)
    return out


def render_foley_buses(sr: int = SR, n_samples: int = N_SAMPLES) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render (foley_ambience, foley_events, tunnel_mask) synced to storyboard.py."""
    rng = np.random.default_rng(98765)
    ambience = np.zeros((2, n_samples), dtype=np.float32)
    events = np.zeros((2, n_samples), dtype=np.float32)

    # Pre-generate phase-coherent stereo pink & white noise beds (correlation ~ +0.62, Side/Mid ~ 0.48)
    pink_c = _pink_noise(n_samples, rng)
    pink_a = _pink_noise(n_samples, rng)
    pink_b = _pink_noise(n_samples, rng)
    pink_l = (0.79 * pink_c + 0.61 * pink_a).astype(np.float32)
    pink_r = (0.79 * pink_c + 0.61 * pink_b).astype(np.float32)
    del pink_c, pink_a, pink_b

    white_c = rng.standard_normal(n_samples).astype(np.float32)
    white_a = rng.standard_normal(n_samples).astype(np.float32)
    white_b = rng.standard_normal(n_samples).astype(np.float32)
    white_l = (0.79 * white_c + 0.61 * white_a).astype(np.float32)
    white_r = (0.79 * white_c + 0.61 * white_b).astype(np.float32)
    del white_c, white_a, white_b

    t_full = np.arange(n_samples, dtype=np.float32) / sr

    # =========================================================================
    # ACT I (00:00 - 00:35): Dusk wind through grass, distant thunder, stone raindrops, umbrella, lantern
    # =========================================================================
    s01 = get_shot("S01")
    s02 = get_shot("S02")
    s04 = get_shot("S04")
    s05 = get_shot("S05")

    env_act1_wind = _smooth_window(n_samples, s01.start_s, s05.end_s + 2.0, fade_in_s=1.2, fade_out_s=2.8, sr=sr)
    sos_wind = signal.butter(2, [260.0, 1200.0], btype="band", fs=sr, output="sos")
    sos_grass = signal.butter(2, [2800.0, 9500.0], btype="band", fs=sr, output="sos")
    gust_l = 0.68 + 0.32 * np.sin(2.0 * np.pi * 0.14 * t_full)
    gust_r = 0.68 + 0.32 * np.sin(2.0 * np.pi * 0.16 * t_full + 0.7)

    i_end_a1 = int(round((s05.end_s + 2.5) * sr))
    ambience[0, :i_end_a1] += (
        signal.sosfilt(sos_wind, pink_l[:i_end_a1]).astype(np.float32) * gust_l[:i_end_a1] * 0.10
        + signal.sosfilt(sos_grass, white_l[:i_end_a1]).astype(np.float32) * gust_l[:i_end_a1] * 0.022
    ) * env_act1_wind[:i_end_a1]
    ambience[1, :i_end_a1] += (
        signal.sosfilt(sos_wind, pink_r[:i_end_a1]).astype(np.float32) * gust_r[:i_end_a1] * 0.10
        + signal.sosfilt(sos_grass, white_r[:i_end_a1]).astype(np.float32) * gust_r[:i_end_a1] * 0.022
    ) * env_act1_wind[:i_end_a1]

    # Distant thunder rumbles (S02 @ 8.0s, S04 @ 21.0s)
    _add_event(events, _synth_thunder(5.0, rng, sr) * 0.65, s02.start_s + 1.0, sr)
    _add_event(events, _synth_thunder(5.5, rng, sr) * 0.70, s04.start_s + 1.0, sr)

    # Realistic accelerating raindrops hitting stone (S04: 20.0 - 28.5s) — crisp noise bursts, ZERO bloops
    t_drop = s04.start_s + 0.3
    while t_drop < s05.start_s + 2.5:
        progress = (t_drop - s04.start_s) / 8.5
        lo = float(rng.uniform(1600.0, 2800.0))
        hi = float(rng.uniform(6500.0, 11500.0))
        pan = float(rng.uniform(-0.55, 0.55))
        _add_event(
            events,
            _synth_raindrop_impact(lo, hi, 0.028, 0.14 + 0.09 * progress, pan, rng, sr),
            t_drop,
            sr,
        )
        t_drop += float(rng.uniform(0.16, 0.48) * (1.0 - 0.65 * np.clip(progress, 0.0, 1.0)))

    # Umbrella open whoosh (S05 @ 28.5s) & brass lantern click (S05 @ 32.5s)
    _add_event(events, _synth_umbrella_open(rng, sr), s05.start_s + 2.0, sr)
    _add_event(events, _synth_lantern_click(sr), s05.start_s + 6.0, sr)

    # =========================================================================
    # ACT II (00:35 - 01:10): Rich 3-layer airy stereo heavy rain, puddle splashes, station bell, rail rumble, brake hiss
    # =========================================================================
    s06 = get_shot("S06")
    s07 = get_shot("S07")
    s08 = get_shot("S08")
    s09 = get_shot("S09")
    s10 = get_shot("S10")

    a2_i0 = int(round((s06.start_s - 2.0) * sr))
    a2_i1 = int(round((s10.end_s + 1.5) * sr))
    env_heavy_rain = _smooth_window(n_samples, s06.start_s - 1.5, s10.end_s + 1.0, fade_in_s=2.2, fade_out_s=1.2, sr=sr)
    sos_rain_body = signal.butter(2, [450.0, 2400.0], btype="band", fs=sr, output="sos")
    sos_rain_sheet = signal.butter(2, [2400.0, 7800.0], btype="band", fs=sr, output="sos")
    sos_rain_air = signal.butter(2, [7800.0, 14500.0], btype="band", fs=sr, output="sos")

    ambience[0, a2_i0:a2_i1] += (
        signal.sosfilt(sos_rain_body, pink_l[a2_i0:a2_i1]).astype(np.float32) * 0.14
        + signal.sosfilt(sos_rain_sheet, white_l[a2_i0:a2_i1]).astype(np.float32) * 0.055
        + signal.sosfilt(sos_rain_air, white_l[a2_i0:a2_i1]).astype(np.float32) * 0.032
    ) * env_heavy_rain[a2_i0:a2_i1]
    ambience[1, a2_i0:a2_i1] += (
        signal.sosfilt(sos_rain_body, pink_r[a2_i0:a2_i1]).astype(np.float32) * 0.14
        + signal.sosfilt(sos_rain_sheet, white_r[a2_i0:a2_i1]).astype(np.float32) * 0.055
        + signal.sosfilt(sos_rain_air, white_r[a2_i0:a2_i1]).astype(np.float32) * 0.032
    ) * env_heavy_rain[a2_i0:a2_i1]

    # Close-up puddle splashes (S07: 42.0 - 48.5s)
    for _ in range(44):
        td = float(rng.uniform(s07.start_s, s07.end_s))
        _add_event(events, _synth_puddle_splash(0.18, float(rng.uniform(-0.6, 0.6)), rng, sr), td, sr)

    # Clear resonant Japanese station chime/bell (S08 @ 49.0s)
    _add_event(events, _synth_station_bell(sr), s08.start_s + 0.8, sr)

    # Approaching steel rail rumble (S08-S09: 51.0 - 63.0s)
    r_i0 = int(round(51.0 * sr))
    r_i1 = int(round(63.5 * sr))
    t_r = np.linspace(0.0, 1.0, r_i1 - r_i0, dtype=np.float32)
    env_rumble = (t_r**1.8) * np.clip((1.0 - t_r) * 8.0, 0.0, 1.0)
    sos_rail = signal.butter(2, [65.0, 260.0], btype="band", fs=sr, output="sos")
    ambience[0, r_i0:r_i1] += signal.sosfilt(sos_rail, pink_l[r_i0:r_i1]).astype(np.float32) * env_rumble * 0.18
    ambience[1, r_i0:r_i1] += signal.sosfilt(sos_rail, pink_r[r_i0:r_i1]).astype(np.float32) * env_rumble * 0.18

    # Realistic pneumatic air-brake hiss & steam release (S10: 62.5 - 69.5s)
    b_len = int(round(6.5 * sr))
    t_b = np.arange(b_len, dtype=np.float32) / sr
    sos_steam = signal.butter(2, [1800.0, 11500.0], btype="band", fs=sr, output="sos")
    steam_raw = signal.sosfilt(sos_steam, rng.standard_normal(b_len)).astype(np.float32)
    # Initial pneumatic valve release puff + tapering steam hiss
    valve_puff = np.exp(-t_b / 0.35) * (1.0 - np.exp(-t_b / 0.03)) * 0.55
    steam_tail = np.maximum(np.sin(np.pi * np.clip(t_b / 6.5, 0.0, 1.0)), 0.0) ** 1.2 * np.exp(-np.maximum(t_b - 1.5, 0.0) / 2.2)
    _add_event(events, pan_mono_to_stereo(steam_raw * (valve_puff + steam_tail) * 0.15, pan=0.10, sr=sr), s10.start_s + 1.0, sr)

    # =========================================================================
    # ACT III (01:10 - 01:50): Cozy 2-bogie rail clack-clack, muffled interior rain, tactile window glass ticks
    # =========================================================================
    a3_start, a3_end, _, _ = ACT_RANGES[3]
    a3_i0 = int(round(a3_start * sr))
    a3_i1 = int(round(a3_end * sr))
    env_a3 = _smooth_window(n_samples, a3_start, a3_end, fade_in_s=1.0, fade_out_s=1.0, sr=sr)

    sos_muffled_rain = signal.butter(2, [350.0, 2600.0], btype="band", fs=sr, output="sos")
    sos_cabin_hum = signal.butter(2, [75.0, 190.0], btype="band", fs=sr, output="sos")
    ambience[0, a3_i0:a3_i1] += (
        signal.sosfilt(sos_muffled_rain, pink_l[a3_i0:a3_i1]).astype(np.float32) * 0.09
        + signal.sosfilt(sos_cabin_hum, pink_l[a3_i0:a3_i1]).astype(np.float32) * 0.08
    ) * env_a3[a3_i0:a3_i1]
    ambience[1, a3_i0:a3_i1] += (
        signal.sosfilt(sos_muffled_rain, pink_r[a3_i0:a3_i1]).astype(np.float32) * 0.09
        + signal.sosfilt(sos_cabin_hum, pink_r[a3_i0:a3_i1]).astype(np.float32) * 0.08
    ) * env_a3[a3_i0:a3_i1]

    # Distinct cozy 2-bogie rail clack-clack ('ba-dum ... ba-dum') every ~1.80s across Act III
    t_clack = a3_start + 1.0
    while t_clack < a3_end - 1.0:
        _add_event(events, _synth_rail_clack_pair(0.24, float(rng.uniform(-0.12, 0.12)), rng, sr), t_clack, sr)
        t_clack += float(rng.uniform(1.75, 1.90))

    # Tactile rain pattering on window glass (heightened in S14 macro 90.0-96.5s)
    s14 = get_shot("S14")
    for _ in range(75):
        td = float(rng.uniform(a3_start + 0.8, a3_end - 0.8))
        amp = 0.14 if (s14.start_s <= td <= s14.end_s) else 0.075
        _add_event(
            events,
            _synth_raindrop_impact(2600.0, 9500.0, 0.022, amp, float(rng.uniform(0.05, 0.45)), rng, sr),
            td,
            sr,
        )

    # =========================================================================
    # ACT IV (01:50 - 02:25): Bridge resonance, pine wind, tamed tunnel roar, crisp Doppler whooshes, exit burst
    # =========================================================================
    s17 = get_shot("S17")
    s18 = get_shot("S18")
    s19 = get_shot("S19")
    s20 = get_shot("S20")
    s21 = get_shot("S21")

    # S17 Bridge resonance (110.0 - 117.5s)
    b_i0, b_i1 = int(round(s17.start_s * sr)), int(round(s17.end_s * sr))
    sos_bridge = signal.butter(2, [140.0, 480.0], btype="band", fs=sr, output="sos")
    env_b = _smooth_window(n_samples, s17.start_s, s17.end_s, fade_in_s=0.8, fade_out_s=0.8, sr=sr)
    ambience[0, b_i0:b_i1] += signal.sosfilt(sos_bridge, pink_l[b_i0:b_i1]).astype(np.float32) * env_b[b_i0:b_i1] * 0.14
    ambience[1, b_i0:b_i1] += signal.sosfilt(sos_bridge, pink_r[b_i0:b_i1]).astype(np.float32) * env_b[b_i0:b_i1] * 0.14

    # S18 Rushing pine wind (117.5 - 124.0s)
    p_i0, p_i1 = int(round(s18.start_s * sr)), int(round(s18.end_s * sr))
    sos_pine = signal.butter(2, [500.0, 5200.0], btype="band", fs=sr, output="sos")
    env_p = _smooth_window(n_samples, s18.start_s, s18.end_s, fade_in_s=0.6, fade_out_s=0.4, sr=sr)
    ambience[0, p_i0:p_i1] += signal.sosfilt(sos_pine, pink_l[p_i0:p_i1]).astype(np.float32) * env_p[p_i0:p_i1] * 0.14
    ambience[1, p_i0:p_i1] += signal.sosfilt(sos_pine, pink_r[p_i0:p_i1]).astype(np.float32) * env_p[p_i0:p_i1] * 0.14

    # S19-S21 Tunnel plunge & concrete cavity roar (124.0 - 141.5s) — tamed sub-bass, realistic 120-520 Hz echo
    tunnel_mask = _smooth_window(n_samples, s19.start_s, s21.start_s + 3.0, fade_in_s=0.35, fade_out_s=0.45, sr=sr)
    t_i0, t_i1 = int(round(s19.start_s * sr)), int(round((s21.start_s + 3.5) * sr))
    sos_tun = signal.butter(2, [120.0, 520.0], btype="band", fs=sr, output="sos")
    ambience[0, t_i0:t_i1] += signal.sosfilt(sos_tun, pink_l[t_i0:t_i1]).astype(np.float32) * tunnel_mask[t_i0:t_i1] * 0.16
    ambience[1, t_i0:t_i1] += signal.sosfilt(sos_tun, pink_r[t_i0:t_i1]).astype(np.float32) * tunnel_mask[t_i0:t_i1] * 0.16

    # Rhythmic Doppler tunnel light-whooshes every 0.85s during S20 (130.5 - 138.5s)
    t_w = s20.start_s + 0.3
    while t_w < s20.end_s - 0.3:
        _add_event(events, _synth_doppler_whoosh(rng, sr), t_w, sr)
        t_w += 0.85

    # Clean open-air exit burst at S21 (141.5s)
    n_exit = int(round(2.5 * sr))
    t_ex = np.arange(n_exit, dtype=np.float32) / sr
    sos_ex = signal.butter(2, [1000.0, 9000.0], btype="band", fs=sr, output="sos")
    ex_noise = signal.sosfilt(sos_ex, rng.standard_normal(n_exit)).astype(np.float32)
    ex_env = np.exp(-t_ex / 0.80) * (1.0 - np.exp(-t_ex / 0.035))
    _add_event(events, pan_mono_to_stereo(ex_noise * ex_env * 0.16, pan=0.0, sr=sr), s21.start_s + 3.0, sr)

    # =========================================================================
    # ACT V (02:25 - 03:10): Clearly audible, shimmering shallow-sea water wake hush & gentle lap
    # =========================================================================
    a5_start, a5_end, _, _ = ACT_RANGES[5]
    a5_i0, a5_i1 = int(round(a5_start * sr)), int(round(a5_end * sr))
    env_a5 = _smooth_window(n_samples, a5_start, a5_end + 3.0, fade_in_s=1.8, fade_out_s=3.0, sr=sr)
    sos_wake_body = signal.butter(2, [350.0, 2800.0], btype="band", fs=sr, output="sos")
    sos_wake_foam = signal.butter(2, [2800.0, 11500.0], btype="band", fs=sr, output="sos")
    swell_l = 0.72 + 0.28 * np.sin(2.0 * np.pi * 0.22 * t_full[a5_i0:a5_i1])
    swell_r = 0.72 + 0.28 * np.sin(2.0 * np.pi * 0.24 * t_full[a5_i0:a5_i1] + 0.9)
    ambience[0, a5_i0:a5_i1] += (
        signal.sosfilt(sos_wake_body, pink_l[a5_i0:a5_i1]).astype(np.float32) * 0.13
        + signal.sosfilt(sos_wake_foam, white_l[a5_i0:a5_i1]).astype(np.float32) * 0.042
    ) * swell_l * env_a5[a5_i0:a5_i1]
    ambience[1, a5_i0:a5_i1] += (
        signal.sosfilt(sos_wake_body, pink_r[a5_i0:a5_i1]).astype(np.float32) * 0.13
        + signal.sosfilt(sos_wake_foam, white_r[a5_i0:a5_i1]).astype(np.float32) * 0.042
    ) * swell_r * env_a5[a5_i0:a5_i1]

    # =========================================================================
    # ACT VI (03:10 - 03:55): Rain tapering to sparkling drops, warm golden-hour breeze
    # =========================================================================
    s28 = get_shot("S28")
    a6_start, a6_end, _, _ = ACT_RANGES[6]
    a6_i0, a6_i1 = int(round(a6_start * sr)), int(round(a6_end * sr))
    env_a6 = _smooth_window(n_samples, a6_start, a6_end, fade_in_s=2.0, fade_out_s=3.5, sr=sr)
    sos_breeze = signal.butter(2, [450.0, 2400.0], btype="band", fs=sr, output="sos")
    sos_breeze_air = signal.butter(2, [4500.0, 12500.0], btype="band", fs=sr, output="sos")
    ambience[0, a6_i0:a6_i1] += (
        signal.sosfilt(sos_breeze, pink_l[a6_i0:a6_i1]).astype(np.float32) * 0.075
        + signal.sosfilt(sos_breeze_air, white_l[a6_i0:a6_i1]).astype(np.float32) * 0.025
    ) * env_a6[a6_i0:a6_i1]
    ambience[1, a6_i0:a6_i1] += (
        signal.sosfilt(sos_breeze, pink_r[a6_i0:a6_i1]).astype(np.float32) * 0.075
        + signal.sosfilt(sos_breeze_air, white_r[a6_i0:a6_i1]).astype(np.float32) * 0.025
    ) * env_a6[a6_i0:a6_i1]

    # Sparkling final drops (S28: 190.0 - 197.0s)
    for _ in range(22):
        td = float(rng.uniform(s28.start_s + 0.3, s28.end_s))
        _add_event(
            events,
            _synth_raindrop_impact(3200.0, 11000.0, 0.025, 0.10, float(rng.uniform(-0.55, 0.55)), rng, sr),
            td,
            sr,
        )

    # =========================================================================
    # ACTS VII-VIII (03:55 - 05:00): Receding train lullaby -> glassy water stillness -> pure piano by 285s
    # =========================================================================
    a7_start, a7_end, _, _ = ACT_RANGES[7]
    s38 = get_shot("S38")
    a7_i0, a7_i1 = int(round(a7_start * sr)), int(round((s38.end_s + 2.0) * sr))
    env_a7 = _smooth_window(n_samples, a7_start, s38.end_s + 1.5, fade_in_s=1.5, fade_out_s=10.0, sr=sr)
    sos_still = signal.butter(2, [350.0, 2600.0], btype="band", fs=sr, output="sos")
    sos_still_air = signal.butter(2, [3500.0, 10000.0], btype="band", fs=sr, output="sos")
    ambience[0, a7_i0:a7_i1] += (
        signal.sosfilt(sos_still, pink_l[a7_i0:a7_i1]).astype(np.float32) * 0.055
        + signal.sosfilt(sos_still_air, white_l[a7_i0:a7_i1]).astype(np.float32) * 0.015
    ) * env_a7[a7_i0:a7_i1]
    ambience[1, a7_i0:a7_i1] += (
        signal.sosfilt(sos_still, pink_r[a7_i0:a7_i1]).astype(np.float32) * 0.055
        + signal.sosfilt(sos_still_air, white_r[a7_i0:a7_i1]).astype(np.float32) * 0.015
    ) * env_a7[a7_i0:a7_i1]

    # Soft receding train lullaby clacks in early Act VII (237-255s)
    t_rec = a7_start + 2.0
    while t_rec < a7_start + 20.0:
        fade = max(0.0, 1.0 - (t_rec - a7_start) / 20.0)
        _add_event(events, _synth_rail_clack_pair(0.08 * fade, 0.0, rng, sr), t_rec, sr)
        t_rec += 2.1

    # Final gentle water ripple laps (S38: 276-281s)
    for td in [276.2, 278.5, 280.6]:
        _add_event(events, _synth_puddle_splash(0.06, float(rng.uniform(-0.25, 0.25)), rng, sr), td, sr)

    return ambience, events, tunnel_mask
