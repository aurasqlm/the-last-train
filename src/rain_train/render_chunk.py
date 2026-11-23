"""Resumable, timeout-safe chunked 1080p60 renderer & stitcher for THE LAST RAIN TRAIN.

Usage:
  python3 src/rain_train/render_chunk.py --act 1
  python3 src/rain_train/render_chunk.py --motion-check
  python3 src/rain_train/render_chunk.py --stitch
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rain_train.acts import prepare_shot_cache, render_shot_frame
from rain_train.storyboard import FPS, HEIGHT, SHOTS, TOTAL_FRAMES, WIDTH, ShotSpec

# Worker-local state for multiprocessing
_WORKER_CACHE: Dict[str, Dict] = {}
_WORKER_W: int = WIDTH
_WORKER_H: int = HEIGHT
_INTERNAL_W: int = 960
_INTERNAL_H: int = 540


def _init_worker(w: int, h: int, iw: int, ih: int) -> None:
    global _WORKER_CACHE, _WORKER_W, _WORKER_H, _INTERNAL_W, _INTERNAL_H
    _WORKER_CACHE = {}
    _WORKER_W = w
    _WORKER_H = h
    _INTERNAL_W = iw
    _INTERNAL_H = ih


def _render_one_frame(args: Tuple[ShotSpec, int]) -> bytes:
    shot, frame_idx = args
    if shot.id not in _WORKER_CACHE:
        # Keep only the active shot in worker cache to bound memory
        _WORKER_CACHE.clear()
        _WORKER_CACHE[shot.id] = prepare_shot_cache(shot, _INTERNAL_W, _INTERNAL_H)
    global_t = frame_idx / float(FPS)
    local_t = (frame_idx - shot.start_frame) / float(FPS)
    frame = render_shot_frame(
        shot, local_t, global_t, _INTERNAL_W, _INTERNAL_H, cache=_WORKER_CACHE[shot.id]
    )
    if frame.dtype != np.uint8:
        u8 = np.clip(frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        u8 = frame
    if (_INTERNAL_W, _INTERNAL_H) != (_WORKER_W, _WORKER_H):
        u8 = cv2.resize(u8, (_WORKER_W, _WORKER_H), interpolation=cv2.INTER_CUBIC)
    return u8.tobytes()


def ffprobe_frame_count(path: Path) -> int:
    if not path.exists() or path.stat().st_size < 1024:
        return -1
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_packets",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                str(path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out)
    except Exception:
        return -1


def render_shot_list(
    shots: List[ShotSpec],
    out_path: Path,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
    crf: int = 18,
    preset: str = "veryfast",
    force: bool = False,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expected_frames = sum(s.num_frames for s in shots)
    if not force:
        existing = ffprobe_frame_count(out_path)
        if existing == expected_frames:
            print(
                f"[SKIP] {out_path} already exists with exact {existing} frames.",
                flush=True,
            )
            return

    t0 = time.time()
    n_workers = max(1, os.cpu_count() or 2)
    tmp_path = out_path.with_suffix(".tmp.mp4")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None

    tasks: List[Tuple[ShotSpec, int]] = []
    for s in shots:
        for f in range(s.start_frame, s.end_frame):
            tasks.append((s, f))

    done = 0
    try:
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(width, height, 960, 540)) as pool:
            for raw_bytes in pool.imap(_render_one_frame, tasks, chunksize=8):
                proc.stdin.write(raw_bytes)
                done += 1
                if done % 300 == 0 or done == expected_frames:
                    dt = time.time() - t0
                    rate = done / max(dt, 1e-6)
                    print(
                        f"  {out_path.name}: {done}/{expected_frames} frames ({rate:.1f} fps, {dt:.1f}s)",
                        flush=True,
                    )
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="ignore") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed (rc={rc}): {stderr[-500:]}")
    except Exception:
        proc.kill()
        raise

    tmp_path.replace(out_path)
    actual = ffprobe_frame_count(out_path)
    dt = time.time() - t0
    print(
        f"[DONE] {out_path} ({actual}/{expected_frames} frames, {out_path.stat().st_size/1e6:.2f} MB) in {dt:.1f}s",
        flush=True,
    )
    if actual != expected_frames:
        raise RuntimeError(f"Frame count mismatch for {out_path}: got {actual}, expected {expected_frames}")


def render_act(act_num: int, force: bool = False) -> None:
    shots = [s for s in SHOTS if s.act == act_num]
    if not shots:
        raise ValueError(f"No shots found for Act {act_num}")
    out_path = Path("chunks") / f"act_{act_num:02d}.mp4"
    render_shot_list(shots, out_path, force=force)


def render_motion_check() -> None:
    """Render 3 short 1080p60 motion-check clips (2s each) from Acts II, V, VI."""
    samples = [
        ("S10", 2, 120),  # Act II hero train braking
        ("S27", 5, 120),  # Act V Ghibli sea tracking
        ("S33", 6, 120),  # Act VI golden hour petals
    ]
    out_dir = Path("chunks") / "motion_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid, act_num, n_frames in samples:
        shot = next(s for s in SHOTS if s.id == sid)
        sub = ShotSpec(
            id=shot.id,
            act=shot.act,
            start_s=shot.start_s,
            end_s=shot.start_s + n_frames / float(FPS),
            start_frame=shot.start_frame,
            end_frame=shot.start_frame + n_frames,
            shot_type=shot.shot_type,
            camera_motion=shot.camera_motion,
            description=shot.description,
            palette=shot.palette,
        )
        render_shot_list([sub], out_dir / f"motion_{sid}_1080p60.mp4", force=True)


def stitch_final(audio_path: str = "soundtrack_300s.wav", out_path: str = "final_last_rain_train_1080p60.mp4") -> None:
    chunks_dir = Path("chunks")
    chunk_files = [chunks_dir / f"act_{i:02d}.mp4" for i in range(1, 9)]
    for cf in chunk_files:
        if not cf.exists():
            raise FileNotFoundError(f"Missing chunk: {cf}")
    list_file = chunks_dir / "chunks.txt"
    with open(list_file, "w") as f:
        for cf in chunk_files:
            f.write(f"file '{cf.resolve()}'\n")

    # First pass encode with target < 90MB (2100k maxrate -> ~78MB for 300s + 6MB audio = ~84MB)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "20",
        "-preset",
        "veryfast",
        "-maxrate",
        "2100k",
        "-bufsize",
        "4200k",
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-ac",
        "2",
        "-shortest",
        out_path,
    ]
    print("Stitching and muxing final MP4...", flush=True)
    subprocess.run(cmd, check=True)
    size_mb = Path(out_path).stat().st_size / (1024 * 1024)
    print(f"Created {out_path}: {size_mb:.2f} MB", flush=True)

    # If > 89MB, re-encode slightly tighter
    if size_mb > 89.0:
        print(f"Size {size_mb:.2f} MB > 89 MB, re-encoding tighter...", flush=True)
        tmp_tight = Path(out_path).with_suffix(".tight.mp4")
        cmd_tight = [
            "ffmpeg",
            "-y",
            "-i",
            out_path,
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            "-maxrate",
            "1850k",
            "-bufsize",
            "3700k",
            "-movflags",
            "+faststart",
            "-c:a",
            "copy",
            str(tmp_tight),
        ]
        subprocess.run(cmd_tight, check=True)
        tmp_tight.replace(out_path)
        size_mb = Path(out_path).stat().st_size / (1024 * 1024)
        print(f"Re-encoded {out_path}: {size_mb:.2f} MB", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--act", type=int, help="Render a single act (1..8)")
    p.add_argument("--motion-check", action="store_true", help="Render 1080p60 motion-check clips")
    p.add_argument("--stitch", action="store_true", help="Concatenate all 8 acts + mux audio")
    p.add_argument("--force", action="store_true", help="Force re-render even if chunk exists")
    args = p.parse_args()

    if args.motion_check:
        render_motion_check()
    elif args.act is not None:
        render_act(args.act, force=args.force)
    elif args.stitch:
        stitch_final()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
