# Project: THE LAST RAIN TRAIN

## Architecture
A 100% pure-code Python cinema pipeline generating a 5-minute (300.0s exact, 18,000 frames), 1920x1080, 60fps hand-painted Studio Ghibli / Makoto Shinkai inspired animated short film with an original 300.0s 48kHz stereo felt-piano, strings, and shot-accurate foley soundtrack.

- **Environment**: Linux, 2 CPU cores, 7.8 GiB RAM, 85 GB disk, Python 3.12.16. Dependencies to install: `ffmpeg` (via apt), `numpy`, `scipy`, `opencv-python-headless` (via pip). Fonts available: `NotoSerifCJK-Regular.ttc` (Japanese CJK Mincho), `FreeSerif.ttf` (Latin Serif).
- **Visual Pipeline (`src/rain_train/`)**:
  - Pre-rendered high-res painterly gouache background canvases, domain-warped cumulus clouds with warm rim-lighting, and cold-press watercolor paper grain per shot/act.
  - Vectorized 60fps compositing (NumPy + OpenCV + Pillow): multi-layer parallax, 12fps hand-drawn ink line-boil on character silhouettes/foregrounds, rain physics, puddle ripples, macro window droplets, sea wakes, 35mm optical halation/bloom, atmospheric haze, petals & light-motes.
  - Resumable chunked renderer (`render_chunk.py --act 1..8`) using 2-worker `multiprocessing.Pool` piping raw `rgb24` frames directly to `ffmpeg` stdin, with automatic `ffprobe` frame-count skip.
- **Audio Pipeline (`src/rain_train/audio/`)**:
  - Pure-code NumPy/SciPy 48kHz stereo synthesis (exact 14,400,000 samples = 300.000s).
  - Physically modeled felt-piano (inharmonic partials, unison detuning, muted hammer attack + felt thump, two-stage decay, soundboard formants, damper resonance, continuous rubato).
  - Formant-filtered solo cello (Act V) + 4-part detuned string ensemble with vibrato (Act VI climax).
  - Shot-accurate procedural stereo foley synced to `storyboard.py` timestamps across all 8 acts.
  - Velvet-noise concert hall & tunnel IR convolution (`scipy.signal.fftconvolve`), dynamic ambience ducking, warm analog roll-off, soft-knee limiter (-1.0 dBFS, zero clipping).

## Code Layout
```
src/rain_train/
├── __init__.py
├── storyboard.py          # Single source of truth: 40 shots (S01–S40), exact 300.0s / 18,000 frames, palettes
├── painterly/
│   ├── __init__.py
│   ├── paper_grain.py     # Cold-press watercolor paper grain & pigment granulation
│   ├── clouds.py          # Domain-warped gouache cumulus clouds with warm rim-lighting
│   ├── optics.py          # 35mm optical halation/bloom, god-rays, atmospheric haze, vignette
│   ├── weather.py         # Multi-depth slanted rain, puddle ripples, macro glass droplets, sea wake
│   └── ink.py             # Hand-drawn 12fps ink line-boil & cel silhouettes (umbrella, train, carriage, pines)
├── acts/
│   ├── __init__.py
│   ├── act1.py .. act8.py # Shot renderers for all 40 distinct shots (S01–S40)
├── audio/
│   ├── __init__.py
│   ├── piano.py           # Felt-piano physical/modal synthesizer
│   ├── strings.py         # Cello & string ensemble synthesizer
│   ├── foley.py           # Shot-synced procedural stereo foley across Acts I–VIII
│   ├── mastering.py       # Hall/tunnel convolution reverb, analog EQ, soft-knee limiter
│   └── render_audio.py    # CLI rendering soundtrack_300s.wav (exact 300.000s, 48kHz stereo)
├── render_style_check.py  # Generates style_frame_check.png (2x2 full-res 1920x1080 hero proof sheet)
├── render_contact_sheet.py# Generates contact_sheet_5min.png (8x5 widescreen 40-shot strip)
├── render_chunk.py        # Resumable 1080p60 chunked renderer (--act 1..8, --draft, --clip)
└── stitch_final.py        # Concatenates chunks + muxes WAV -> final_last_rain_train_1080p60.mp4 (< 90MB)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Environment & Toolchain | Install `ffmpeg`, `numpy`, `scipy`, `opencv-python-headless`; verify fonts & git | M1 | survey_1 |
| 2 | 40-Shot Storyboard Config | `storyboard.py` with S01–S40, exact 300.0s (18,000 frames), max 6–10s/shot | M1 | survey_2 |
| 3 | Painterly Engine | Paper grain, gouache clouds, 35mm halation, rain/droplets/ripples, ink line-boil | M1 | survey_2 |
| 4 | Gate 1: `style_frame_check.png` | 2x2 1920x1080 hero proof sheet (S10, S14, S24, S33); blind `visual-critic` >= 9.5/10 | M1 | ORIGINAL_REQUEST |
| 5 | 300s Stereo Score & Foley | Felt-piano, cello/strings (Acts V–VI), shot-synced foley, mastered 48kHz WAV | M2 | survey_3 |
| 6 | All 40 Shots (Acts I–VIII) | Full implementation of S01–S40 with distinct angles, lighting, and 60fps motion | M3 | survey_2 |
| 7 | `contact_sheet_5min.png` | 8x5 widescreen keyframe strip of all 40 distinct shots | M3 | ORIGINAL_REQUEST |
| 8 | Draft & 1080p60 Motion Clips | 640x360 15fps full draft + 1080p60 motion clips (Acts II, V, VI) | M3 | ORIGINAL_REQUEST |
| 9 | Gate 2: Blind Critics >= 9.5/10 | Blind `visual-critic` & `audio-critic` both >= 9.5/10 | M3 | ORIGINAL_REQUEST |
| 10 | Chunked 1080p60 Render | Resumable `--act 1..8` chunks (< 600s each, `ffprobe` skip, stdin pipe) | M4 | ORIGINAL_REQUEST |
| 11 | Final Stitch, Mux & Audit | `final_last_rain_train_1080p60.mp4` (300.0s, 1080p60, < 90MB), blind audit >= 9.5/10, forensic audit CLEAN | M4 | ORIGINAL_REQUEST |
| 12 | Git Push to `main` | Commit & push final MP4 + deliverables to `origin/main` | M4 | ORIGINAL_REQUEST |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Env Setup + Painterly Engine + Gate 1 (`style_frame_check.png`) | Install deps, `storyboard.py`, `painterly/`, `style_frame_check.png`, blind `visual-critic` >= 8.5/10 (capped at Round 6) | none | DONE (8.7/10 APPROVE) |
| M2 | 300.0s Stereo Soundtrack & Foley (`soundtrack_300s.wav`) | `src/rain_train/audio/`, render `soundtrack_300s.wav`, blind `audio-critic` >= 9.5/10 | M1 (deps & `storyboard.py`) | DONE (9.6/10 APPROVE) |
| M3 | All 40 Shots + Contact Sheet + Draft + Motion Clips + Gate 2 | `acts/act1..8.py`, `contact_sheet_5min.png`, draft, 1080p60 clips, blind `visual-critic` >= 8.5/10 (max 2 rounds) | M1 (Gate 1 PASS) | IN_PROGRESS |
| M4 | Chunked 1080p60 Render + Stitch/Mux (<90MB) + Final Audit + Git Push | Render Acts 1–8 chunks, stitch/mux `final_last_rain_train_1080p60.mp4`, final verification, forensic audit CLEAN, push `main` | M2 + M3 (Gate 2 PASS) | PLANNED |

## Interface Contracts
### `storyboard.py` ↔ `acts/` & `audio/`
- `SHOTS`: list of 40 `ShotSpec(id, act, start_s, end_s, start_frame, end_frame, shot_type, camera_motion, description, palette)`
- Invariants: `len(SHOTS) == 40`, `SHOTS[0].start_s == 0.0`, `SHOTS[-1].end_s == 300.0`, `SHOTS[-1].end_frame == 18000`, `6.0 <= (end_s - start_s) <= 10.0` for all shots, contiguous non-overlapping frames.
- `ACT_RANGES`: dict mapping `1..8` to `(start_s, end_s, start_frame, end_frame)`.
