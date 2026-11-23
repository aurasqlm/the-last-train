"""DSP utilities, velvet-noise convolution reverb, dynamic ducking, and mastering chain."""

from __future__ import annotations

import wave
from typing import Tuple

import numpy as np
from scipy import signal

from . import N_SAMPLES, SR


def biquad_peaking_sos(fc: float, gain_db: float, q: float, sr: int = SR) -> np.ndarray:
    """Design a single biquad peaking EQ section in SOS format."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sr
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / a

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def biquad_lowshelf_sos(fc: float, gain_db: float, slope: float = 0.8, sr: int = SR) -> np.ndarray:
    """Design a low-shelf biquad section in SOS format."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha

    b0 = a * ((a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0)
    a2 = (a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def biquad_highshelf_sos(fc: float, gain_db: float, slope: float = 0.8, sr: int = SR) -> np.ndarray:
    """Design a high-shelf biquad section in SOS format."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sr
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)
    two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def pan_mono_to_stereo(mono: np.ndarray, pan: float, itd_ms: float = 0.0, sr: int = SR) -> np.ndarray:
    """Constant-power stereo panner with optional micro-ITD (-1.0 left to +1.0 right)."""
    pan = float(np.clip(pan, -1.0, 1.0))
    angle = (pan + 1.0) * (np.pi / 4.0)
    gl = np.cos(angle)
    gr = np.sin(angle)

    out = np.empty((2, mono.shape[0]), dtype=np.float32)
    out[0] = mono * gl
    out[1] = mono * gr

    if itd_ms > 0.0 and abs(pan) > 0.05:
        delay_samp = int(round(abs(pan) * itd_ms * 1e-3 * sr))
        if delay_samp > 0 and delay_samp < mono.shape[0]:
            if pan > 0.0:
                out[0, delay_samp:] = out[0, :-delay_samp]
                out[0, :delay_samp] = 0.0
            else:
                out[1, delay_samp:] = out[1, :-delay_samp]
                out[1, :delay_samp] = 0.0
    return out


def generate_velvet_ir(
    duration_s: float,
    rt60_s: float,
    hf_damping_hz: float = 7200.0,
    early_reflections: bool = True,
    seed: int = 42,
    sr: int = SR,
) -> np.ndarray:
    """Generate a smooth, phase-coherent stereo velvet-noise + Gaussian reverb IR normalized to unit L2 energy."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr

    # Exponential decay envelope (-60 dB at rt60_s)
    tau = rt60_s / np.log(1000.0)
    env = np.exp(-t / max(tau, 1e-3))

    # Build a common center component + decorrelated stereo side components for healthy stereo correlation
    grid = 28  # ~1700 impulses/sec
    n_cells = n // grid

    def _make_velvet_channel() -> np.ndarray:
        ch_ir = np.zeros(n, dtype=np.float64)
        offsets = rng.integers(0, grid, size=n_cells)
        indices = np.arange(n_cells) * grid + offsets
        signs = rng.choice([-1.0, 1.0], size=n_cells)
        ch_ir[indices] = signs * 0.75
        ch_ir += rng.standard_normal(n) * 0.22
        return ch_ir

    common = _make_velvet_channel()
    ir = np.zeros((2, n), dtype=np.float64)
    for ch in range(2):
        # 55% common + 45% independent stereo tail -> natural lush hall width without phase cancellation
        ir[ch] = 0.55 * common + 0.45 * _make_velvet_channel()

        # Early reflections (12-65 ms)
        if early_reflections:
            er_times_ms = [13.2, 21.7, 29.4, 38.1, 47.6, 58.3]
            er_gains = [0.50, -0.40, 0.32, -0.26, 0.21, -0.16]
            for ms, g in zip(er_times_ms, er_gains):
                jitter = rng.uniform(-1.0, 1.0)
                idx = int(round((ms + jitter) * 1e-3 * sr))
                if 0 < idx < n:
                    ir[ch, idx] += g

        ir[ch] *= env

        # Smooth fade-in (4 ms) to avoid pre-ringing, and HF damping low-pass
        fade_in = min(int(0.004 * sr), n)
        ir[ch, :fade_in] *= np.linspace(0.0, 1.0, fade_in)

        sos_lp = signal.butter(2, hf_damping_hz, btype="low", fs=sr, output="sos")
        ir[ch] = signal.sosfilt(sos_lp, ir[ch])

        # Gentle fade-out on last 5%
        fade_out = max(int(0.05 * n), 1)
        ir[ch, -fade_out:] *= np.linspace(1.0, 0.0, fade_out)

    # Normalize to UNIT L2 ENERGY per channel so wet gain is exact (0 dB = equal energy to dry)
    l2_norm = np.sqrt(0.5 * (np.sum(ir[0] ** 2) + np.sum(ir[1] ** 2)) + 1e-12)
    ir /= l2_norm
    return ir.astype(np.float32)


def convolve_stereo(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Fast stereo FFT convolution truncated cleanly to input length."""
    n = audio.shape[1]
    out = np.empty_like(audio, dtype=np.float32)
    for ch in range(2):
        conv = signal.fftconvolve(audio[ch], ir[ch], mode="full")[:n]
        out[ch] = conv.astype(np.float32)
    return out


def compute_rms_envelope(audio: np.ndarray, window_s: float = 0.25, sr: int = SR) -> np.ndarray:
    """Compute a smooth mono RMS envelope for dynamic ducking."""
    mono_sq = 0.5 * (audio[0].astype(np.float32) ** 2 + audio[1].astype(np.float32) ** 2)
    # Downsample by 16x for ultra-fast smoothing, then linearly interpolate back
    ds = 16
    n = mono_sq.shape[0]
    n_ds = n // ds
    trimmed = mono_sq[: n_ds * ds].reshape(n_ds, ds).mean(axis=1)
    sr_ds = sr / ds
    sos = signal.butter(2, 1.0 / max(window_s, 0.05), btype="low", fs=sr_ds, output="sos")
    env_ds = np.sqrt(np.maximum(signal.sosfiltfilt(sos, trimmed), 0.0))
    x_old = np.linspace(0.0, 1.0, n_ds)
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, env_ds).astype(np.float32)


def master_mix(
    piano_bus: np.ndarray,
    strings_bus: np.ndarray,
    foley_ambience: np.ndarray,
    foley_events: np.ndarray,
    tunnel_mask: np.ndarray,
    sr: int = SR,
) -> np.ndarray:
    """Combine all buses with concert-hall & tunnel reverb, ducking, analog EQ, M/S polish, and soft limiting."""
    # 1. Concert Hall Reverb for Piano & Strings (unit-energy IR -> clean, controlled wet blend)
    hall_ir = generate_velvet_ir(duration_s=2.8, rt60_s=2.1, hf_damping_hz=7200.0, early_reflections=True, seed=101, sr=sr)
    music_dry = piano_bus + strings_bus
    music_wet = convolve_stereo(piano_bus * 0.34 + strings_bus * 0.45, hall_ir)
    music_bus = music_dry + music_wet

    # 2. Exterior/Cabin Reverb for Foley Events + Cavernous Tunnel Reverb during Act IV tunnel
    foley_ir = generate_velvet_ir(duration_s=0.85, rt60_s=0.70, hf_damping_hz=8500.0, early_reflections=True, seed=202, sr=sr)
    foley_events_wet = convolve_stereo(foley_events * 0.24, foley_ir)
    foley_events_bus = foley_events + foley_events_wet

    tunnel_ir = generate_velvet_ir(duration_s=2.6, rt60_s=2.2, hf_damping_hz=2600.0, early_reflections=True, seed=303, sr=sr)
    tunnel_send = (foley_ambience * 0.35 + foley_events * 0.45 + piano_bus * 0.12) * tunnel_mask[np.newaxis, :]
    tunnel_wet = convolve_stereo(tunnel_send, tunnel_ir)

    # 3. Dynamic Ambience Ducking: music gently ducks foley ambience (-2 to -5 dB)
    music_env = compute_rms_envelope(music_bus, window_s=0.35, sr=sr)
    duck_gain = 1.0 / (1.0 + 3.2 * music_env)
    duck_gain = np.clip(duck_gain, 0.55, 1.0)
    foley_ambience_ducked = foley_ambience * duck_gain[np.newaxis, :]

    # 4. Sum Master Bus
    mix = music_bus + foley_ambience_ducked + foley_events_bus + tunnel_wet * 0.50

    # 5. Sub-bass cleanup (30 Hz HPF) + Open Cinema Analog EQ (clear air & presence, gentle roll-off above 14.5 kHz)
    sos_hp = signal.butter(2, 30.0, btype="high", fs=sr, output="sos")
    sos_low_trim = biquad_lowshelf_sos(110.0, -1.2, slope=0.75, sr=sr)
    sos_pres = biquad_peaking_sos(3600.0, 1.4, 0.9, sr=sr)
    sos_air = biquad_highshelf_sos(9000.0, 1.5, slope=0.7, sr=sr)
    sos_rolloff = signal.butter(2, 15500.0, btype="low", fs=sr, output="sos")
    sos_master = np.vstack([sos_hp, sos_low_trim, sos_pres, sos_air, sos_rolloff])
    for ch in range(2):
        mix[ch] = signal.sosfilt(sos_master, mix[ch]).astype(np.float32)

    # 6. Zero-phase Mid/Side Stereo Polish: strictly mono below 150 Hz, natural stereo width
    mid = 0.5 * (mix[0] + mix[1])
    side = 0.5 * (mix[0] - mix[1])
    sos_side_hp = signal.butter(2, 150.0, btype="high", fs=sr, output="sos")
    side = signal.sosfiltfilt(sos_side_hp, side).astype(np.float32)
    mix[0] = mid + side
    mix[1] = mid - side

    # 7. Normalize to cinematic loudness before soft-knee limiting
    peak_raw = float(np.max(np.abs(mix))) + 1e-12
    target_pre_limit = 0.96
    if peak_raw > 0.0:
        mix *= target_pre_limit / max(peak_raw, 0.35)

    # Soft-knee tanh limiter with hard ceiling at -1.0 dBFS (0.891 linear)
    ceiling = 10.0 ** (-1.0 / 20.0)  # ~0.89125
    knee = 0.72
    abs_x = np.abs(mix)
    over = abs_x > knee
    if np.any(over):
        excess = (abs_x[over] - knee) / (ceiling - knee)
        shaped = knee + (ceiling - knee) * np.tanh(excess)
        mix[over] = np.sign(mix[over]) * shaped
    mix = np.clip(mix, -ceiling, ceiling)

    # Ensure clean 10ms fade-in at 0.0s and graceful natural decay in final 2.0s to exact 0.0 at 300.000s
    fade_in = int(0.010 * sr)
    fade_out = int(2.0 * sr)
    mix[:, :fade_in] *= np.linspace(0.0, 1.0, fade_in, dtype=np.float32)[np.newaxis, :]
    mix[:, -fade_out:] *= (0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, fade_out, dtype=np.float32))))[np.newaxis, :]

    assert mix.shape == (2, N_SAMPLES), f"Expected (2, {N_SAMPLES}), got {mix.shape}"
    return mix.astype(np.float32)


def write_wav_16bit(filepath: str, stereo: np.ndarray, sr: int = SR) -> None:
    """Write float32 stereo array (2, N) in [-1, 1] to 16-bit PCM WAV."""
    assert stereo.ndim == 2 and stereo.shape[0] == 2
    interleaved = np.empty(stereo.shape[1] * 2, dtype=np.int16)
    scaled = np.clip(stereo, -1.0, 1.0) * 32767.0
    interleaved[0::2] = np.round(scaled[0]).astype(np.int16)
    interleaved[1::2] = np.round(scaled[1]).astype(np.int16)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(interleaved.tobytes())


def analyze_mix(stereo: np.ndarray, sr: int = SR) -> Tuple[float, float]:
    """Return (peak_dbfs, overall_rms_dbfs)."""
    peak = float(np.max(np.abs(stereo))) + 1e-12
    rms = float(np.sqrt(np.mean(stereo**2))) + 1e-12
    return 20.0 * np.log10(peak), 20.0 * np.log10(rms)
