"""Physically modeled felt-piano synthesizer and 300.0s expressive rubato film score."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import interpolate, signal

from . import N_SAMPLES, SR
from .mastering import biquad_peaking_sos, pan_mono_to_stereo


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


@dataclass(frozen=True)
class PianoNote:
    time_s: float
    midi: int
    velocity: float  # 0.0 .. 1.0
    duration_s: float


def synthesize_felt_note(
    midi: int,
    velocity: float,
    duration_s: float,
    rng: np.random.Generator,
    sr: int = SR,
) -> np.ndarray:
    """Synthesize a single stereo felt-piano note with inharmonic partials, unison detuning,
    velocity-dependent felt brightness, two-stage decay, and mechanical felt hammer thump.
    """
    v = float(np.clip(velocity, 0.05, 1.0))
    f0 = midi_to_hz(midi)

    # Extra tail for natural decay
    total_s = min(duration_s + 2.4, 12.5)
    n = int(round(total_s * sr))
    t = np.arange(n, dtype=np.float32) / sr

    # Inharmonicity coefficient B scales with pitch register
    reg = np.clip((midi - 21.0) / 87.0, 0.0, 1.0)
    b_inharm = float(1.4e-4 + 2.8e-4 * (reg**1.6))

    # Number of partials (rich upper harmonic bloom up to 8-12 kHz on louder notes)
    max_freq = min(13500.0, 3200.0 + 9800.0 * (v**1.05))
    k_max = min(18, max(6, int(max_freq / max(f0, 25.0))))
    ks = np.arange(1, k_max + 1, dtype=np.float64)

    # Inharmonic partial frequencies
    freqs = ks * f0 * np.sqrt(1.0 + b_inharm * (ks**2))
    valid = freqs < (sr * 0.44)
    ks = ks[valid]
    freqs = freqs[valid]

    # Velocity-dependent felt hammer spectral slope: soft = warm & intimate, loud = bright bloom
    alpha = 1.95 - 0.85 * v
    amps = (1.0 / (ks**alpha)) * np.exp(-0.045 * (ks - 1.0) * (1.1 - 0.55 * v))
    if len(amps) >= 2:
        amps[0] *= 1.12
        amps[1] *= 1.08
    amps /= np.sum(amps) + 1e-12

    # Two-stage physical decay (prompt + aftersound), higher partials decay faster
    base_tau1 = 0.60 + 1.65 * (1.0 - reg)
    base_tau2 = 2.4 + 5.2 * (1.0 - reg)
    tau1 = base_tau1 / (ks**1.08)
    tau2 = base_tau2 / (ks**1.20)

    # Vectorized additive synthesis: (K, N) -> sum over K
    # For notes >= C3 (midi >= 48), add 2 slightly detuned unison strings for organic beating
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(ks))
    mono = np.zeros(n, dtype=np.float32)

    unison_cents = [0.0] if midi < 48 else [-rng.uniform(0.55, 1.05), rng.uniform(0.55, 1.05)]
    unison_gain = 1.0 / np.sqrt(len(unison_cents))

    for cents in unison_cents:
        detune = 2.0 ** (cents / 1200.0)
        for idx in range(len(ks)):
            f_k = float(freqs[idx] * detune)
            a_k = float(amps[idx] * unison_gain)
            env_k = 0.60 * np.exp(-t / float(tau1[idx])) + 0.40 * np.exp(-t / float(tau2[idx]))
            mono += (a_k * env_k * np.sin(2.0 * np.pi * f_k * t + float(phases[idx]))).astype(np.float32)

    # Smooth raised-cosine felt hammer attack (4-10 ms, crisp yet warm)
    attack_s = 0.004 + 0.006 * (1.0 - v)
    n_att = max(int(round(attack_s * sr)), 16)
    if n_att < n:
        att = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_att, dtype=np.float32)))
        mono[:n_att] *= att

    # Damper release envelope after duration_s (soft felt damper)
    rel_start = int(round(duration_s * sr))
    if rel_start < n:
        rel_len = n - rel_start
        rel_tau = 0.34 + 0.28 * (1.0 - reg)
        t_rel = np.arange(rel_len, dtype=np.float32) / sr
        mono[rel_start:] *= np.exp(-t_rel / rel_tau)

    # Two-component felt hammer transient: (a) warm felt thump (90-320 Hz) + (b) woody key-click (1.2-4.5 kHz)
    thump_len = min(int(0.065 * sr), n)
    noise = rng.standard_normal(thump_len).astype(np.float64)
    sos_thump = signal.butter(2, [90.0, 320.0], btype="band", fs=sr, output="sos")
    sos_click = signal.butter(2, [1200.0, 4500.0], btype="band", fs=sr, output="sos")
    thump = signal.sosfilt(sos_thump, noise).astype(np.float32)
    click = signal.sosfilt(sos_click, noise).astype(np.float32)
    t_thump = np.arange(thump_len, dtype=np.float32) / sr
    thump *= np.exp(-t_thump / 0.015) * (0.065 * (0.45 + 0.55 * v))
    click *= np.exp(-t_thump / 0.006) * (0.050 * (0.35 + 0.65 * v))
    mono[:thump_len] += thump + click

    # Velocity scaling (musical dynamic curve)
    gain = 0.44 * (v**1.30)
    mono *= gain

    # Pitch-mapped stereo soundboard image: bass slightly left (-0.30), treble slightly right (+0.30)
    pan = float(np.clip((midi - 60.0) / 36.0 * 0.30, -0.32, 0.32))
    return pan_mono_to_stereo(mono, pan=pan, itd_ms=0.12, sr=sr)


def _add_chord(
    notes: List[PianoNote],
    start_s: float,
    midis: Sequence[int],
    vel: float,
    dur_s: float,
    roll_ms: float,
    rng: np.random.Generator,
) -> None:
    """Add a naturally rolled chord with micro-timing & velocity humanization."""
    for i, m in enumerate(sorted(midis)):
        jitter_s = float(rng.normal(0.0, 0.008))
        v_jit = float(np.clip(vel * rng.uniform(0.94, 1.06), 0.08, 0.98))
        t_note = max(0.0, start_s + i * (roll_ms * 1e-3) + jitter_s)
        notes.append(PianoNote(time_s=t_note, midi=m, velocity=v_jit, duration_s=dur_s))


def _add_melody(
    notes: List[PianoNote],
    start_s: float,
    events: Sequence[Tuple[float, int, float, float]],
    tempo_scale: float,
    rng: np.random.Generator,
) -> None:
    """Add expressive rubato melody events: (beat_offset, midi, vel, dur_beats)."""
    # Smooth rubato warp: slight accelerando into phrase peak, gentle ritardando at end
    if not events:
        return
    max_b = max(e[0] for e in events) + 0.5
    ctrl_b = np.array([0.0, max_b * 0.35, max_b * 0.72, max_b], dtype=np.float64)
    ctrl_t = np.array([0.0, max_b * 0.34, max_b * 0.69, max_b * 1.05], dtype=np.float64) * tempo_scale
    spline = interpolate.CubicSpline(ctrl_b, ctrl_t)

    for b_off, m, v, d_b in events:
        t_on = start_s + float(spline(b_off)) + float(rng.normal(0.0, 0.009))
        v_h = float(np.clip(v * rng.uniform(0.95, 1.05), 0.08, 0.98))
        notes.append(PianoNote(time_s=max(0.0, t_on), midi=m, velocity=v_h, duration_s=d_b * tempo_scale))


def build_piano_score(seed: int = 77) -> List[PianoNote]:
    """Compose the full 300.0s 8-act felt-piano score synced to storyboard.py."""
    rng = np.random.default_rng(seed)
    notes: List[PianoNote] = []

    # =========================================================================
    # ACT I: BEFORE THE STORM (00:00 - 00:35) | D minor | 62-66 BPM | pp-mp
    # 00:00-00:06 is foley only (wind & grass). Sparse Motif A enters at 00:06.
    # =========================================================================
    act1_chords = [
        (6.0, [50, 57, 62, 64, 65], 0.30, 5.2, 28.0),   # Dm9
        (12.2, [46, 53, 57, 62, 64], 0.28, 5.0, 30.0),  # Bbmaj7#11
        (18.2, [43, 50, 55, 58, 62], 0.32, 5.0, 26.0),  # Gm9
        (24.0, [45, 52, 57, 62, 64], 0.34, 5.0, 24.0),  # A7sus4
        (29.5, [50, 57, 62, 65], 0.36, 5.2, 22.0),      # Dm7 (umbrella opens 28.5s)
    ]
    for t_s, ch, vel, dur, roll in act1_chords:
        _add_chord(notes, t_s, ch, vel, dur, roll, rng)

    # Sparse Motif A droplets (A4-F4-E4-D4)
    _add_melody(
        notes, 7.2,
        [
            (0.0, 69, 0.36, 1.8), (1.8, 65, 0.32, 1.4), (3.2, 64, 0.30, 1.2), (4.4, 62, 0.34, 2.5),
            (7.5, 69, 0.35, 1.6), (9.1, 70, 0.38, 1.5), (10.6, 69, 0.33, 1.4), (12.0, 65, 0.30, 2.6),
            (16.0, 64, 0.34, 1.5), (17.5, 65, 0.36, 1.4), (18.9, 69, 0.40, 2.2), (21.5, 67, 0.36, 2.4),
            (24.2, 64, 0.38, 2.8),
        ],
        tempo_scale=0.95,
        rng=rng,
    )

    # =========================================================================
    # ACT II: THE DOWNPOUR & ARRIVAL (00:35 - 01:10) | D minor | 68-72 BPM | mp-mf
    # Rolling left-hand arpeggios, crescendo at S09 headlights (55s), breath at S10 braking (67s)
    # =========================================================================
    act2_prog = [
        (35.0, [38, 45, 50, 57, 64, 65], 0.46),  # Dm9
        (40.5, [40, 48, 55, 60, 64], 0.48),      # C/E
        (46.0, [41, 48, 53, 57, 64], 0.50),      # Fmaj7
        (51.5, [43, 50, 55, 58, 62], 0.56),      # Gm9 (rails vibrating S08)
        (56.5, [45, 52, 57, 61, 67], 0.64),      # A7b9 (headlights S09)
        (61.5, [38, 45, 50, 57, 62, 65], 0.62),  # Dm (train braking S10)
        (66.5, [45, 52, 57, 62], 0.40),          # A7sus4 (fermata breath)
    ]
    for t_s, arp, vel in act2_prog:
        # Rolling LH arpeggio
        for idx, m in enumerate(arp):
            notes.append(PianoNote(t_s + idx * 0.24 + float(rng.normal(0, 0.006)), m, vel * (0.85 + 0.04 * idx), 3.6))
        # Repeat upper shimmer
        for idx, m in enumerate(arp[2:]):
            notes.append(PianoNote(t_s + 2.4 + idx * 0.28 + float(rng.normal(0, 0.007)), m, vel * 0.78, 2.4))

    _add_melody(
        notes, 35.8,
        [
            (0.0, 69, 0.52, 1.5), (1.5, 65, 0.48, 1.0), (2.5, 64, 0.46, 1.0), (3.5, 62, 0.50, 2.0),
            (6.0, 67, 0.54, 1.5), (7.5, 65, 0.50, 1.0), (8.5, 64, 0.52, 1.5), (10.0, 60, 0.48, 2.0),
            (12.5, 69, 0.58, 1.5), (14.0, 72, 0.62, 1.5), (15.5, 70, 0.58, 1.2), (16.7, 69, 0.55, 1.8),
            (19.0, 67, 0.62, 1.5), (20.5, 69, 0.66, 1.5), (22.0, 70, 0.68, 1.8), (24.0, 69, 0.70, 2.5),
            (27.5, 65, 0.60, 1.8), (29.5, 64, 0.52, 2.5), (32.5, 62, 0.44, 3.5),
        ],
        tempo_scale=0.88,
        rng=rng,
    )

    # =========================================================================
    # ACT III: THE WINDOW SEAT (01:10 - 01:50) | D Dorian / Dm | 6/8 rocking ostinato | pp-mp
    # =========================================================================
    act3_bars = [
        (70.0, [50, 57, 62, 65, 64, 62]),  # Dm9
        (74.8, [50, 55, 59, 64, 62, 59]),  # G13/D
        (79.6, [46, 53, 57, 62, 60, 57]),  # Bbmaj7
        (84.4, [45, 52, 57, 62, 60, 57]),  # A7sus4
        (89.2, [50, 57, 62, 65, 64, 62]),  # Dm9 (macro droplets S14)
        (94.0, [50, 55, 59, 64, 62, 59]),  # G13/D
        (98.8, [46, 53, 57, 62, 60, 57]),  # Bbmaj7 (passing town bokeh S15)
        (103.6, [45, 52, 57, 61, 64, 61]), # A7 (glass reflection S16)
    ]
    for t_s, pat in act3_bars:
        for rep in range(2):
            base_t = t_s + rep * 2.4
            for i, m in enumerate(pat):
                v = 0.34 if i == 0 else (0.28 if i == 3 else 0.23)
                notes.append(PianoNote(base_t + i * 0.40 + float(rng.normal(0, 0.006)), m, v, 1.8))

    # Tender upper-register counter-melody (F5-E5-D5-A4)
    _add_melody(
        notes, 75.5,
        [
            (0.0, 77, 0.38, 1.8), (1.8, 76, 0.35, 1.4), (3.2, 74, 0.34, 2.0), (5.2, 69, 0.32, 2.5),
            (9.5, 74, 0.36, 1.6), (11.1, 76, 0.38, 1.6), (12.7, 77, 0.40, 2.2), (15.0, 76, 0.34, 2.6),
            (19.5, 79, 0.42, 1.8), (21.3, 77, 0.38, 1.5), (22.8, 76, 0.36, 1.5), (24.3, 74, 0.38, 2.8),
            (28.0, 72, 0.35, 2.0), (30.0, 69, 0.34, 3.0),
        ],
        tempo_scale=0.92,
        rng=rng,
    )

    # =========================================================================
    # ACT IV: PINE VALLEY & TUNNEL (01:50 - 02:25) | D minor | Urgent drive -> Tunnel -> Exit Flare
    # =========================================================================
    act4_drive = [
        (110.0, 38, [50, 57, 62], 0.54),  # Dm (bridge S17)
        (114.0, 46, [53, 58, 62], 0.56),  # Bb
        (118.0, 43, [50, 55, 58], 0.58),  # Gm (pine silhouettes S18)
        (121.5, 45, [52, 57, 61], 0.60),  # A7
        # Tunnel plunge at 124.0 (S19-S20): darker, muted, sparse pulsing low octaves
        (124.5, 38, [50, 53], 0.36),
        (128.5, 38, [50, 57], 0.38),
        (132.5, 46, [53, 58], 0.44),      # Rising tension in tunnel
        (136.5, 48, [55, 60], 0.52),      # C
        (140.0, 49, [55, 57, 64], 0.64),  # A7/C# -> surge into S21 exit flare (145s)
    ]
    for t_s, bass, upper, vel in act4_drive:
        notes.append(PianoNote(t_s, bass, vel * 1.05, 3.4))
        step = 0.36 if t_s < 124.0 else 0.48
        count = 9 if t_s < 124.0 else 7
        for i in range(count):
            m = upper[i % len(upper)]
            notes.append(PianoNote(t_s + i * step + float(rng.normal(0, 0.005)), m, vel * 0.72, 0.85))

    # =========================================================================
    # ACT V: THE FLOODED SEA TRACKS (02:25 - 03:10) | F major / Bb Lydian | 60-64 BPM | Expansive
    # Flowing broken chords supporting solo cello (cello enters 150s)
    # =========================================================================
    act5_prog = [
        (145.0, [41, 48, 53, 57, 60, 64, 67]),  # Fmaj9 (rails submerge S22)
        (151.0, [40, 48, 55, 60, 64, 67, 72]),  # C/E
        (157.0, [38, 45, 50, 57, 60, 64, 69]),  # Dm9 (wake ripples S23)
        (163.0, [46, 53, 57, 62, 64, 69, 72]),  # Bbmaj7#11 (cloud cathedrals S24)
        (169.0, [41, 48, 53, 57, 60, 64, 69]),  # Fmaj9 (awe S25)
        (175.0, [40, 48, 55, 60, 64, 67, 72]),  # C/E (sea lanterns S26)
        (181.0, [46, 53, 57, 62, 64, 69, 72]),  # Bbmaj7#11 (Ghibli side-track S27)
        (186.0, [45, 52, 57, 62, 64, 69]),      # A7sus4 -> lift into D major!
    ]
    for t_s, arp in act5_prog:
        for i, m in enumerate(arp):
            notes.append(PianoNote(t_s + i * 0.34 + float(rng.normal(0, 0.008)), m, 0.42 + 0.03 * (i % 3), 4.2))
        # Gentle upper shimmer
        for i, m in enumerate(reversed(arp[2:6])):
            notes.append(PianoNote(t_s + 2.8 + i * 0.42 + float(rng.normal(0, 0.009)), m, 0.34, 3.0))

    # =========================================================================
    # ACT VI: CLOUDBREAK & GOLDEN HOUR (03:10 - 03:55) | D MAJOR BREAKTHROUGH!
    # Radiant Dmaj9 -> A/C# -> Bm9 -> Gmaj7#11 -> Em9 -> A13sus4 -> Dmaj9
    # =========================================================================
    act6_prog = [
        (190.0, [38, 45, 50, 54, 57, 61, 64, 66], 0.62),  # Dmaj9 (cloudbreak S28)
        (196.0, [37, 45, 52, 57, 61, 64, 69], 0.66),      # A/C# (clouds tearing S29)
        (202.0, [35, 42, 47, 50, 54, 57, 61], 0.70),      # Bm9
        (208.0, [43, 50, 54, 59, 61, 66, 71], 0.76),      # Gmaj7#11 (god-rays S30)
        (214.0, [40, 47, 52, 55, 59, 62, 66], 0.82),      # Em9 (sunlight floods cabin S31)
        (219.5, [45, 52, 55, 57, 62, 66, 69], 0.86),      # A13sus4 (peak swell)
        (225.0, [38, 45, 50, 54, 57, 61, 64, 66], 0.84),  # Dmaj9 (euphoric hero S33)
        (230.5, [43, 50, 54, 59, 62, 66], 0.68),          # Gmaj7
    ]
    for t_s, arp, vel in act6_prog:
        _add_chord(notes, t_s, arp[:4], vel, 4.8, 18.0, rng)
        for i, m in enumerate(arp[2:]):
            notes.append(PianoNote(t_s + 0.35 + i * 0.28 + float(rng.normal(0, 0.007)), m, vel * 0.85, 3.8))
        for i, m in enumerate(arp[3:]):
            notes.append(PianoNote(t_s + 2.6 + i * 0.30 + float(rng.normal(0, 0.007)), m, vel * 0.78, 3.0))

    # Radiant Motif C melody in piano doubling/interweaving with strings
    _add_melody(
        notes, 191.0,
        [
            (0.0, 78, 0.65, 2.0), (2.0, 76, 0.62, 1.5), (3.5, 74, 0.64, 2.2), (6.0, 69, 0.60, 2.0),
            (8.5, 71, 0.66, 1.5), (10.0, 74, 0.72, 2.5), (13.0, 76, 0.74, 2.0), (15.5, 78, 0.78, 2.8),
            (19.0, 81, 0.82, 2.2), (21.2, 78, 0.80, 1.8), (23.0, 76, 0.78, 2.0), (25.5, 74, 0.84, 3.2),
            (29.5, 76, 0.86, 2.2), (32.0, 78, 0.88, 3.5), (36.0, 74, 0.76, 4.0),
        ],
        tempo_scale=0.92,
        rng=rng,
    )

    # =========================================================================
    # ACT VII: THE DISTANT HORIZON (03:55 - 04:35) | D major | 60->52 BPM | Nostalgic
    # =========================================================================
    act7_chords = [
        (235.5, [38, 45, 50, 54, 61, 64], 0.46, 6.2, 30.0),  # Dmaj9
        (243.0, [43, 50, 54, 59, 62], 0.42, 6.0, 32.0),      # Gmaj7
        (250.5, [35, 42, 47, 50, 54, 61], 0.38, 6.0, 34.0),  # Bm9
        (258.0, [45, 52, 57, 61, 64, 66], 0.34, 6.5, 34.0),  # A6/9
        (266.0, [38, 45, 50, 54, 61], 0.30, 7.5, 36.0),      # Dmaj9
    ]
    for t_s, ch, vel, dur, roll in act7_chords:
        _add_chord(notes, t_s, ch, vel, dur, roll, rng)

    # Nostalgic major-key echo of Motif A (F#4-E4-D4-A3)
    _add_melody(
        notes, 237.5,
        [
            (0.0, 66, 0.46, 2.2), (2.2, 64, 0.42, 1.8), (4.0, 62, 0.40, 2.4), (6.5, 57, 0.36, 3.0),
            (11.0, 66, 0.40, 2.4), (13.4, 67, 0.42, 2.0), (15.4, 66, 0.38, 2.2), (17.8, 62, 0.34, 3.5),
            (23.0, 64, 0.34, 2.5), (25.8, 66, 0.32, 3.0), (29.2, 62, 0.28, 4.0),
        ],
        tempo_scale=1.08,
        rng=rng,
    )

    # =========================================================================
    # ACT VIII: QUIET RESOLUTION & TITLE (04:35 - 05:00) | D major | mp->pp | Free rubato
    # Three tender notes (A4 -> E4 -> F#4), then wide warm Dadd9 at 288.5s decaying naturally to 300.000s
    # =========================================================================
    notes.append(PianoNote(276.2, 69, 0.44, 4.8))  # A4
    notes.append(PianoNote(280.6, 64, 0.42, 4.6))  # E4
    notes.append(PianoNote(284.6, 66, 0.46, 5.0))  # F#4

    # Final wide-voiced Dadd9 (D2 - A2 - E3 - F#3 - A3 - D4 - E4) at 288.5s, natural decay to 300.000s
    _add_chord(notes, 288.5, [38, 45, 52, 54, 57, 62, 64], vel=0.52, dur_s=9.5, roll_ms=48.0, rng=rng)

    return notes


def render_piano_bus(sr: int = SR, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Render the complete 300.0s felt-piano stereo bus with wooden soundboard formants."""
    rng = np.random.default_rng(12345)
    notes = build_piano_score(seed=77)
    bus = np.zeros((2, n_samples), dtype=np.float32)

    for note in notes:
        start_idx = int(round(note.time_s * sr))
        if start_idx >= n_samples:
            continue
        rendered = synthesize_felt_note(note.midi, note.velocity, note.duration_s, rng=rng, sr=sr)
        end_idx = min(start_idx + rendered.shape[1], n_samples)
        length = end_idx - start_idx
        if length > 0:
            bus[:, start_idx:end_idx] += rendered[:, :length]

    # Wooden soundboard formant filters + crisp felt presence (2.6 kHz) + open 12.5 kHz roll-off
    sos_board = np.vstack(
        [
            biquad_peaking_sos(190.0, 1.5, 1.8, sr=sr),
            biquad_peaking_sos(450.0, 1.5, 2.0, sr=sr),
            biquad_peaking_sos(1100.0, 1.8, 1.8, sr=sr),
            biquad_peaking_sos(2600.0, 2.0, 1.4, sr=sr),
            signal.butter(1, 12500.0, btype="low", fs=sr, output="sos"),
        ]
    )
    for ch in range(2):
        bus[ch] = signal.sosfilt(sos_board, bus[ch]).astype(np.float32)

    # Subtle damper-pedal sympathetic resonance (short warm feedback delay 18ms & 29ms)
    symp = np.zeros_like(bus)
    d1, d2 = int(0.018 * sr), int(0.029 * sr)
    symp[0, d1:] += bus[1, :-d1] * 0.06
    symp[1, d2:] += bus[0, :-d2] * 0.06
    sos_symp = signal.butter(2, [160.0, 2600.0], btype="band", fs=sr, output="sos")
    for ch in range(2):
        symp[ch] = signal.sosfilt(sos_symp, symp[ch]).astype(np.float32)
    bus += symp

    return bus
