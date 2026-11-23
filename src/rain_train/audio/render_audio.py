"""CLI entry point to render and verify the 300.0s stereo soundtrack_300s.wav."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure src/ is on sys.path when executed directly
_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rain_train.audio import DURATION_S, N_SAMPLES, SR
from rain_train.audio.foley import render_foley_buses
from rain_train.audio.mastering import analyze_mix, master_mix, write_wav_16bit
from rain_train.audio.piano import render_piano_bus
from rain_train.audio.strings import render_strings_bus
from rain_train.storyboard import ACT_RANGES


def print_diagnostics(stereo: np.ndarray, sr: int = SR) -> None:
    """Print detailed per-act RMS, dynamic range, peak dBFS, and stereo correlation."""
    peak_db, overall_rms_db = analyze_mix(stereo, sr=sr)
    l_ch, r_ch = stereo[0].astype(np.float64), stereo[1].astype(np.float64)
    corr = float(np.dot(l_ch, r_ch) / (np.linalg.norm(l_ch) * np.linalg.norm(r_ch) + 1e-12))

    print("\n=== SOUNDTRACK DIAGNOSTICS ===")
    print(f"Samples      : {stereo.shape[1]} ({stereo.shape[1] / sr:.3f}s @ {sr} Hz, {stereo.shape[0]}ch)")
    print(f"Peak Level   : {peak_db:.2f} dBFS (ceiling -1.00 dBFS, zero clipping)")
    print(f"Overall RMS  : {overall_rms_db:.2f} dBFS")
    print(f"Crest Factor : {peak_db - overall_rms_db:.2f} dB")
    print(f"Stereo Corr  : {corr:+.3f}")
    print("\nPer-Act RMS & Peak:")
    act_rms_list = []
    for act in range(1, 9):
        s_s, e_s, _, _ = ACT_RANGES[act]
        i0, i1 = int(round(s_s * sr)), int(round(e_s * sr))
        seg = stereo[:, i0:i1]
        a_peak, a_rms = analyze_mix(seg, sr=sr)
        act_rms_list.append(a_rms)
        print(f"  Act {act} ({s_s:05.1f}s - {e_s:05.1f}s): RMS = {a_rms:6.2f} dBFS | Peak = {a_peak:6.2f} dBFS")
    print(f"Dynamic Range (Act VI peak vs quietest act): {max(act_rms_list) - min(act_rms_list):.2f} dB")


def render_soundtrack(output_path: str) -> None:
    t0 = time.perf_counter()
    print(f"[1/4] Synthesizing felt-piano score ({DURATION_S:.1f}s @ {SR} Hz)...")
    piano_bus = render_piano_bus(sr=SR, n_samples=N_SAMPLES)
    print(f"      Done in {time.perf_counter() - t0:.1f}s")

    t1 = time.perf_counter()
    print("[2/4] Synthesizing solo cello & 4-part string ensemble...")
    strings_bus = render_strings_bus(sr=SR, n_samples=N_SAMPLES)
    print(f"      Done in {time.perf_counter() - t1:.1f}s")

    t2 = time.perf_counter()
    print("[3/4] Synthesizing shot-accurate stereo foley across Acts I-VIII...")
    foley_amb, foley_evt, tunnel_mask = render_foley_buses(sr=SR, n_samples=N_SAMPLES)
    print(f"      Done in {time.perf_counter() - t2:.1f}s")

    t3 = time.perf_counter()
    print("[4/4] Mastering (velvet-noise hall & tunnel IRs, ducking, analog EQ, soft limiter)...")
    mastered = master_mix(piano_bus, strings_bus, foley_amb, foley_evt, tunnel_mask, sr=SR)
    write_wav_16bit(output_path, mastered, sr=SR)
    print(f"      Done in {time.perf_counter() - t3:.1f}s | Total: {time.perf_counter() - t0:.1f}s")
    print(f"Wrote: {output_path} ({os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")

    print_diagnostics(mastered, sr=SR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render 300.0s stereo soundtrack for THE LAST RAIN TRAIN")
    parser.add_argument(
        "--output",
        type=str,
        default=str(_ROOT / "soundtrack_300s.wav"),
        help="Output WAV path (default: soundtrack_300s.wav in repo root)",
    )
    args = parser.parse_args()
    render_soundtrack(args.output)


if __name__ == "__main__":
    main()
