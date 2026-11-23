"""Warm solo cello & lush 4-part detuned bowed-string ensemble synthesizer (Acts IV-VII)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import signal

from . import N_SAMPLES, SR
from .mastering import biquad_peaking_sos, pan_mono_to_stereo
from .piano import midi_to_hz


@dataclass(frozen=True)
class StringNote:
    time_s: float
    midi: int
    velocity: float
    duration_s: float
    section: str  # "cello_solo", "vln1", "vln2", "vla", "cello", "bass"


_SECTION_PARAMS: Dict[str, Tuple[int, float, float, List[Tuple[float, float]]]] = {
    # section: (n_voices, pan, vib_cents, [(formant_hz, gain_db), ...])
    # Balanced L/R orchestral seating to keep Act VI L/R within +/- 0.3 dB
    "cello_solo": (3, 0.10, 20.0, [(210.0, 4.5), (480.0, 3.5), (1050.0, 2.2), (2400.0, 1.5)]),
    "vln1": (5, -0.26, 22.0, [(290.0, 3.5), (680.0, 3.2), (1500.0, 2.5), (3400.0, 2.0)]),
    "vln2": (4, 0.22, 18.0, [(280.0, 3.4), (640.0, 3.0), (1400.0, 2.2), (3100.0, 1.6)]),
    "vla": (4, -0.10, 16.0, [(245.0, 4.0), (570.0, 3.2), (1250.0, 2.0), (2700.0, 1.4)]),
    "cello": (4, 0.24, 17.0, [(210.0, 4.5), (490.0, 3.4), (1100.0, 2.2), (2300.0, 1.4)]),
    "bass": (2, 0.08, 8.0, [(105.0, 3.8), (240.0, 3.0), (580.0, 1.8)]),
}


def synthesize_bowed_note(
    midi: int,
    velocity: float,
    duration_s: float,
    section: str,
    rng: np.random.Generator,
    sr: int = SR,
) -> np.ndarray:
    """Synthesize a lush, warm bowed-string note with formant shaping, multi-voice chorus
    detuning, 5.2-5.8 Hz delayed vibrato, and silky bow-rosin air noise.
    """
    v = float(np.clip(velocity, 0.05, 1.0))
    f0 = midi_to_hz(midi)
    n_voices, pan, vib_cents, formants = _SECTION_PARAMS[section]

    # Smooth legato overlap tail
    rel_s = 0.72 if section != "cello_solo" else 0.55
    total_s = duration_s + rel_s
    n = int(round(total_s * sr))
    t = np.arange(n, dtype=np.float32) / sr

    # Bow ADSR envelope (smooth 0.30-0.55s attack, expressive swell, smooth release)
    att_s = min(0.30 + 0.24 * (1.0 - v), duration_s * 0.45)
    n_att = max(int(round(att_s * sr)), 64)
    n_rel = max(int(round(rel_s * sr)), 64)
    n_sus = max(n - n_att - n_rel, 0)

    env = np.ones(n, dtype=np.float32)
    env[:n_att] = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_att, dtype=np.float32)))
    if n_sus > 0:
        swell = 1.0 + 0.14 * np.sin(np.linspace(0.0, np.pi, n_sus, dtype=np.float32))
        env[n_att : n_att + n_sus] = swell
    env[-n_rel:] = 0.5 * (1.0 + np.cos(np.linspace(0.0, np.pi, n_rel, dtype=np.float32)))

    # Warm bowed-string harmonic profile (1/k^1.50 avoids reedy/buzzy sawtooth harshness)
    max_freq = min(10500.0, 2200.0 + 7500.0 * (v**1.05))
    k_max = min(14, max(4, int(max_freq / max(f0, 30.0))))
    ks = np.arange(1, k_max + 1, dtype=np.float64)
    amps = (1.0 / (ks**1.50)) * np.exp(-0.07 * (ks - 1.0)) * (0.80 + 0.20 * np.abs(np.sin(np.pi * ks * 0.13)))
    if len(amps) >= 2:
        amps[0] *= 1.25
        amps[1] *= 1.10
    amps /= np.sum(amps) + 1e-12

    # Delayed-onset vibrato envelope (ramps in smoothly from 0.20s to 0.65s)
    vib_env = np.clip((t - 0.20) / 0.45, 0.0, 1.0).astype(np.float32)
    vib_env = vib_env * vib_env * (3.0 - 2.0 * vib_env)  # smoothstep

    stereo = np.zeros((2, n), dtype=np.float32)
    if section == "cello_solo":
        voice_weights = [1.0, 0.28, 0.28]
        norm_w = float(np.sqrt(sum(w * w for w in voice_weights)))
        voice_weights = [w / norm_w for w in voice_weights]
    else:
        voice_weights = [1.0 / np.sqrt(n_voices)] * n_voices

    for vi in range(n_voices):
        if section == "cello_solo" and vi == 0:
            cents_off = 0.0
        else:
            spread = 5.5 if section == "cello_solo" else 8.5
            cents_off = float(rng.uniform(-spread, spread))
        detune = 2.0 ** (cents_off / 1200.0)
        vib_rate = float(rng.uniform(5.2, 5.8))
        vib_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        vib_depth_rad = (vib_cents / 1200.0) * np.log(2.0)

        # Instantaneous phase with smooth delayed vibrato
        vib_lfo = vib_env * np.sin(2.0 * np.pi * vib_rate * t + vib_phase)
        inst_freq = (f0 * detune) * (1.0 + vib_depth_rad * vib_lfo)
        base_phase = np.cumsum(2.0 * np.pi * inst_freq / sr, dtype=np.float64)

        voice_mono = np.zeros(n, dtype=np.float32)
        for idx in range(len(ks)):
            k = float(ks[idx])
            a_k = float(amps[idx])
            p0 = float(rng.uniform(0.0, 2.0 * np.pi))
            voice_mono += (a_k * np.sin(k * base_phase + p0)).astype(np.float32)

        # Staggered voice attack for lush ensemble chorus
        if vi > 0:
            shift = int(round(rng.uniform(0.012, 0.042) * sr))
            if shift < n:
                voice_mono[shift:] = voice_mono[:-shift]
                voice_mono[:shift] = 0.0

        voice_mono *= env * voice_weights[vi]
        v_pan = float(np.clip(pan + rng.uniform(-0.10, 0.10), -0.55, 0.55))
        stereo += pan_mono_to_stereo(voice_mono, pan=v_pan, itd_ms=0.10, sr=sr)

    # Silky bow-rosin air noise (3.5-9.5 kHz bandpass, modulated by bow envelope)
    rosin = rng.standard_normal(n).astype(np.float64)
    sos_rosin = signal.butter(2, [3500.0, 9500.0], btype="band", fs=sr, output="sos")
    rosin_f = signal.sosfilt(sos_rosin, rosin).astype(np.float32) * env * (0.022 * v)
    stereo += pan_mono_to_stereo(rosin_f, pan=pan, sr=sr)

    # Apply wooden instrument body formants + warm velocity-coupled low-pass (open up to 11 kHz)
    lp_cut = min(11000.0, 2600.0 + 7800.0 * v)
    sos_list = [biquad_peaking_sos(fc, g, 2.0, sr=sr) for fc, g in formants]
    sos_list.append(signal.butter(2, lp_cut, btype="low", fs=sr, output="sos"))
    sos_body = np.vstack(sos_list)
    for ch in range(2):
        stereo[ch] = signal.sosfilt(sos_body, stereo[ch]).astype(np.float32)

    # Overall velocity scaling
    stereo *= 0.30 * (v**1.30)
    return stereo


def build_strings_score() -> List[StringNote]:
    """Compose Act IV tunnel cello drone, Act V solo cello, Act VI full string climax, Act VII fade."""
    notes: List[StringNote] = []

    # =========================================================================
    # ACT IV TUNNEL (124.0 - 144.0s): Tense low D2 cello pedal drone
    # =========================================================================
    notes.append(StringNote(124.0, 38, 0.34, 11.5, "cello"))
    notes.append(StringNote(134.5, 40, 0.40, 4.5, "cello"))
    notes.append(StringNote(138.5, 45, 0.50, 5.5, "cello"))  # Rising into S21 exit flare

    # =========================================================================
    # ACT V: FLOODED SEA TRACKS (150.0 - 190.0s) | F major / Bb Lydian
    # Warm Solo Cello entering gradually in tenor register (A3-C4-D4-F4-E4-C4)
    # =========================================================================
    cello_melody: Sequence[Tuple[float, int, float, float]] = [
        (150.5, 57, 0.42, 3.6),  # A3
        (154.0, 60, 0.48, 3.2),  # C4
        (157.0, 62, 0.52, 4.5),  # D4
        (161.5, 65, 0.56, 3.5),  # F4
        (165.0, 64, 0.52, 3.2),  # E4
        (168.0, 60, 0.48, 4.8),  # C4
        (173.0, 62, 0.54, 3.5),  # D4
        (176.5, 64, 0.58, 3.5),  # E4
        (180.0, 65, 0.62, 4.5),  # F4
        (184.5, 64, 0.60, 2.8),  # E4
        (187.2, 62, 0.64, 3.2),  # D4 -> leads into D major cloudbreak!
    ]
    for t_s, m, v, d in cello_melody:
        notes.append(StringNote(t_s, m, v, d, "cello_solo"))

    # =========================================================================
    # ACT VI: CLOUDBREAK & GOLDEN HOUR (190.0 - 235.0s) | D MAJOR CLIMAX!
    # Full 4-part String Ensemble swelling from mp (190s) to soaring ff peak (212-228s)
    # =========================================================================
    # Harmony pads: (start_s, dur_s, bass, cello, vla, vln2, vel)
    act6_pads = [
        (190.0, 6.2, 38, 50, 57, 66, 0.52),  # Dmaj9
        (196.0, 6.2, 37, 49, 57, 64, 0.58),  # A/C#
        (202.0, 6.2, 35, 47, 54, 62, 0.64),  # Bm9
        (208.0, 6.2, 43, 50, 59, 66, 0.74),  # Gmaj7#11 (god-rays S30)
        (214.0, 5.8, 40, 52, 59, 67, 0.84),  # Em9 (sunlight floods cabin S31)
        (219.5, 5.8, 45, 52, 61, 69, 0.90),  # A13sus4 (peak swell S32)
        (225.0, 5.8, 38, 50, 57, 66, 0.92),  # Dmaj9 (euphoric hero S33)
        (230.5, 5.2, 43, 50, 59, 66, 0.72),  # Gmaj7
    ]
    for t_s, dur, m_b, m_c, m_va, m_v2, vel in act6_pads:
        notes.append(StringNote(t_s, m_b, vel * 0.85, dur, "bass"))
        notes.append(StringNote(t_s, m_c, vel * 0.92, dur, "cello"))
        notes.append(StringNote(t_s, m_va, vel * 0.88, dur, "vla"))
        notes.append(StringNote(t_s, m_v2, vel * 0.86, dur, "vln2"))

    # Soaring Violin I melody with high octave emotional climax (F#5-E5-D5-A4-B4-D5 -> A5 peak)
    vln1_melody: Sequence[Tuple[float, int, float, float]] = [
        (190.5, 78, 0.56, 3.2),  # F#5
        (193.5, 76, 0.58, 2.8),  # E5
        (196.2, 74, 0.62, 3.8),  # D5
        (200.0, 69, 0.60, 2.4),  # A4
        (202.2, 71, 0.66, 3.0),  # B4
        (205.0, 74, 0.70, 3.2),  # D5
        (208.2, 76, 0.78, 3.0),  # E5
        (211.0, 78, 0.84, 3.2),  # F#5
        (214.0, 81, 0.92, 3.5),  # A5 (soaring climax!)
        (217.3, 79, 0.90, 2.4),  # G5
        (219.5, 78, 0.92, 3.2),  # F#5
        (222.5, 76, 0.94, 2.8),  # E5
        (225.0, 78, 0.95, 5.5),  # F#5 (radiant peak hold over Dmaj9)
        (230.5, 74, 0.76, 5.0),  # D5
    ]
    for t_s, m, v, d in vln1_melody:
        notes.append(StringNote(t_s, m, v, d, "vln1"))

    # =========================================================================
    # ACT VII: THE DISTANT HORIZON (235.0 - 260.0s) | Strings decrescendo out
    # =========================================================================
    act7_fade = [
        (235.5, 7.5, 38, 50, 57, 66, 0.48),  # Dmaj9
        (243.0, 7.5, 43, 50, 59, 62, 0.34),  # Gmaj7
        (250.5, 8.0, 38, 50, 57, 62, 0.20),  # Dmaj9 dissolving into reverb
    ]
    for t_s, dur, m_b, m_c, m_va, m_v2, vel in act7_fade:
        notes.append(StringNote(t_s, m_b, vel * 0.75, dur, "bass"))
        notes.append(StringNote(t_s, m_c, vel * 0.85, dur, "cello"))
        notes.append(StringNote(t_s, m_va, vel * 0.80, dur, "vla"))
        notes.append(StringNote(t_s, m_v2, vel * 0.70, dur, "vln2"))

    return notes


def render_strings_bus(sr: int = SR, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Render the complete 300.0s cello & string ensemble stereo bus."""
    rng = np.random.default_rng(54321)
    notes = build_strings_score()
    bus = np.zeros((2, n_samples), dtype=np.float32)

    for note in notes:
        start_idx = int(round(note.time_s * sr))
        if start_idx >= n_samples:
            continue
        rendered = synthesize_bowed_note(note.midi, note.velocity, note.duration_s, note.section, rng=rng, sr=sr)
        end_idx = min(start_idx + rendered.shape[1], n_samples)
        length = end_idx - start_idx
        if length > 0:
            bus[:, start_idx:end_idx] += rendered[:, :length]

    return bus
