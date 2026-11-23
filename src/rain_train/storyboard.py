"""Single source of truth for the 40-shot, 300.0s (18,000-frame) storyboard.

Every act, shot renderer, contact sheet, and audio/foley event imports from here
to guarantee exact frame-accurate synchronization across all 8 acts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

FPS: int = 60
WIDTH: int = 1920
HEIGHT: int = 1080
TOTAL_DURATION_S: float = 300.0
TOTAL_FRAMES: int = 18000


def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """Convert '#RRGGBB' hex string to float32-compatible (r, g, b) in [0.0, 1.0]."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


@dataclass(frozen=True)
class ShotSpec:
    """Metadata and timing contract for one shot in THE LAST RAIN TRAIN."""

    id: str
    act: int
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    shot_type: str
    camera_motion: str
    description: str
    palette: Tuple[str, ...]

    @property
    def duration_s(self) -> float:
        return round(self.end_s - self.start_s, 4)

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def rgb_palette(self) -> Tuple[Tuple[float, float, float], ...]:
        return tuple(hex_to_rgb(c) for c in self.palette)


_RAW_SHOTS: List[Tuple[str, int, float, float, str, str, str, Tuple[str, ...]]] = [
    # ACT I — BEFORE THE STORM (00:00.0 – 00:35.0 | 5 shots | 35.0s)
    (
        "S01", 1, 0.0, 7.0,
        "Wide Establishing (High)", "Slow left-to-right pan + gentle push-in",
        "Misty rural valley at dusk; lone wooden station & rail line under brooding overcast sky",
        ("#CFD6DC", "#3D4E5E", "#6E4B3A"),
    ),
    (
        "S02", 1, 7.0, 13.5,
        "Close-Up (Low)", "Subtle handheld sway + rack focus",
        "Tall roadside pampas grass bending violently in pre-storm gusts; stray leaves sweeping",
        ("#3D4E5E", "#5B6B5C", "#CFD6DC"),
    ),
    (
        "S03", 1, 13.5, 20.0,
        "Low-Angle Upward", "Slow tilt-up",
        "Swaying black power lines & wooden poles against darkening, fast-moving gouache storm clouds",
        ("#2A3744", "#3D4E5E", "#9AA7B2"),
    ),
    (
        "S04", 1, 20.0, 26.5,
        "Macro (Top-Down)", "Locked with micro-tremor",
        "First heavy raindrops splashing & darkening weathered grey platform stone tiles",
        ("#4A525A", "#2B333B", "#CFD6DC"),
    ),
    (
        "S05", 1, 26.5, 35.0,
        "Medium Shot (Eye-Level)", "Smooth slow dolly-in",
        "Lone silhouette on platform pops open translucent umbrella as warm amber station lanterns click on",
        ("#3D4E5E", "#6E4B3A", "#F5B969"),
    ),
    # ACT II — THE DOWNPOUR & ARRIVAL (00:35.0 – 01:10.0 | 5 shots | 35.0s)
    (
        "S06", 2, 35.0, 42.0,
        "Wide Platform Shot", "Slow lateral track",
        "Torrential multi-depth slanted rain sheets sweeping across empty wooden platform & awning",
        ("#1B2430", "#466170", "#F5B969"),
    ),
    (
        "S07", 2, 42.0, 48.5,
        "Macro (Low Angle)", "Gentle drift",
        "Expanding concentric puddle ripples distorting glowing amber lantern reflections on wet concrete",
        ("#1B2430", "#F5B969", "#466170"),
    ),
    (
        "S08", 2, 48.5, 55.0,
        "Extreme Low-Angle Track", "Micro-vibration + slow push",
        "Wet steel rails gleaming in rain, trembling with micro-droplets jumping as distant train approaches",
        ("#1B2430", "#6B8291", "#F5B969"),
    ),
    (
        "S09", 2, 55.0, 61.5,
        "Medium-Long Telephoto", "Subtle breathing zoom-in",
        "Twin blurred golden headlights piercing thick blue-teal rain fog down the curved tracks",
        ("#1B2430", "#466170", "#FFF1D0"),
    ),
    (
        "S10", 2, 61.5, 70.0,
        "Wide Side-Pan (Hero)", "Dramatic decelerating side-pan",
        "Vintage rain train brakes smoothly into station; warm glowing windows, hissing steam, rain spray",
        ("#1B2430", "#F5B969", "#466170"),
    ),
    # ACT III — THE WINDOW SEAT (01:10.0 – 01:50.0 | 6 shots | 40.0s)
    (
        "S11", 3, 70.0, 77.0,
        "Interior Wide (One-Point)", "Gentle rhythmic train sway",
        "Empty warm-lit vintage wooden carriage; tungsten ceiling globes, rain-streaked windows, indigo night outside",
        ("#281D19", "#F2C57C", "#142036"),
    ),
    (
        "S12", 3, 77.0, 83.5,
        "Medium Close-Up (Upward)", "Rhythmic carriage sway",
        "Row of leather/ring hanging hand-straps swaying in smooth 60fps damped pendulum motion",
        ("#281D19", "#F2C57C", "#6E4B3A"),
    ),
    (
        "S13", 3, 83.5, 90.0,
        "Medium Close-Up (OTS)", "Slow gentle push-in",
        "Seated silhouette leaning chin on hand by fogged window; warm cabin rim-light on shoulder & hair",
        ("#281D19", "#F2C57C", "#142036"),
    ),
    (
        "S14", 3, 90.0, 96.5,
        "Macro Window Glass (Hero)", "Locked with train vibration",
        "Rain droplets beading, merging, and sliding down fogged glass with refracted town bokeh",
        ("#142036", "#F2C57C", "#8FA9C4"),
    ),
    (
        "S15", 3, 96.5, 103.5,
        "Interior-to-Exterior POV", "Fast 60fps parallax glide",
        "Passing rainy Japanese town lights as warm bokeh circles; dark utility poles whipping past foreground",
        ("#142036", "#F2C57C", "#E8A852"),
    ),
    (
        "S16", 3, 103.5, 110.0,
        "Close-Up (Glass Reflection)", "Slow subtle drift",
        "Warm translucent reflection of silhouette's face overlaid on rainy glass as town lights fade to forest",
        ("#281D19", "#F2C57C", "#142036"),
    ),
    # ACT IV — THROUGH THE PINE VALLEY & TUNNEL (01:50.0 – 02:25.0 | 5 shots | 35.0s)
    (
        "S17", 4, 110.0, 117.5,
        "High Exterior Wide", "Sweeping aerial track",
        "Illuminated train crossing high steel trestle bridge over a mist-filled midnight pine valley",
        ("#0E1A1F", "#1B2F38", "#E8A852"),
    ),
    (
        "S18", 4, 117.5, 124.0,
        "Exterior Side Tracking", "Fast lateral whip-pan",
        "Dense dark pine silhouettes rushing past foreground in multi-layer parallax; warm carriage windows strobing",
        ("#0E1A1F", "#07090D", "#E8A852"),
    ),
    (
        "S19", 4, 124.0, 130.5,
        "Front-Quarter Low Exterior", "Accelerating push-in",
        "Train plunges suddenly into a cavernous stone tunnel mouth; rain cuts off into obsidian darkness",
        ("#07090D", "#0E1A1F", "#E8A852"),
    ),
    (
        "S20", 4, 130.5, 138.5,
        "Interior Medium Close-Up", "Tight handheld cabin rumble",
        "Rhythmic strobing amber tunnel lamps sweeping across cabin glass, seats, and silhouette's face",
        ("#07090D", "#E8A852", "#281D19"),
    ),
    (
        "S21", 4, 138.5, 145.0,
        "Interior Forward POV", "Rapid surge forward",
        "Pinpoint of silver-blue twilight expands into blinding soft optical exit flare, bursting into open night air",
        ("#07090D", "#EEDEC7", "#122347"),
    ),
    # ACT V — THE FLOODED SEA TRACKS (02:25.0 – 03:10.0 | 6 shots | 45.0s)
    (
        "S22", 5, 145.0, 153.0,
        "Wide Exterior Reveal", "Slow sweeping crane-down",
        "Steel rails descend and submerge into an endless shallow mirror-sea reflecting the cobalt twilight sky",
        ("#122347", "#8FA9C4", "#EEDEC7"),
    ),
    (
        "S23", 5, 153.0, 160.0,
        "Low Water-Level Close-Up", "Smooth 60fps tracking",
        "Train wheels slicing through shallow water, fanning silky concentric wake ripples & silver spray",
        ("#122347", "#8FA9C4", "#F2C57C"),
    ),
    (
        "S24", 5, 160.0, 167.5,
        "Monumental Wide (Hero)", "Slow majestic tilt-up & pan",
        "Towering hand-painted gouache cumulus cloud cathedrals rim-lit in cream, perfectly mirrored in sea",
        ("#122347", "#8FA9C4", "#EEDEC7"),
    ),
    (
        "S25", 5, 167.5, 174.5,
        "Interior Medium Close-Up", "Gentle push-in",
        "Seated silhouette leans closer to the window in quiet awe; cool silver-blue sea light washes over face",
        ("#122347", "#EEDEC7", "#281D19"),
    ),
    (
        "S26", 5, 174.5, 181.5,
        "Medium-Wide Water-Level", "Smooth lateral glide",
        "Drifting warm floating sea-lanterns passing silently beside the submerged tracks with glowing water reflections",
        ("#122347", "#F5B969", "#8FA9C4"),
    ),
    (
        "S27", 5, 181.5, 190.0,
        "Sweeping Ghibli Side-Track", "Silky 60fps wide side-track",
        "Classic Spirited-Away-inspired wide profile of glowing train gliding across infinite mirrored sea & clouds",
        ("#122347", "#8FA9C4", "#EEDEC7"),
    ),
    # ACT VI — THE CLOUDBREAK & GOLDEN HOUR (03:10.0 – 03:55.0 | 6 shots | 45.0s)
    (
        "S28", 6, 190.0, 197.0,
        "Macro / Close-Up Exterior", "Slow-motion feel drift",
        "Rain tapering to single sparkling droplets catching first warm golden-apricot light in mid-air",
        ("#5C8FD6", "#F49D6E", "#FFF1D0"),
    ),
    (
        "S29", 6, 197.0, 205.0,
        "Wide Skyward Establishing", "Slow upward pan + pull",
        "Colossal storm clouds tearing open; cerulean sky & rose-gold cloud edges erupting with light",
        ("#5C8FD6", "#E77D8B", "#F49D6E"),
    ),
    (
        "S30", 6, 205.0, 212.5,
        "Wide Horizon Shot", "Slow sweeping pan",
        "Sweeping volumetric golden god-rays piercing cloud gaps and illuminating shimmering sea tracks below",
        ("#F49D6E", "#FFF1D0", "#5C8FD6"),
    ),
    (
        "S31", 6, 212.5, 220.0,
        "Interior Medium Shot", "Warm gliding dolly",
        "Radiant apricot sunlight floods across the wooden carriage, dust motes glowing, silhouette bathed in gold",
        ("#F49D6E", "#FFF1D0", "#281D19"),
    ),
    (
        "S32", 6, 220.0, 226.5,
        "Macro Water Surface", "Gentle gliding track",
        "Macro golden-apricot & rose-gold bokeh shimmer dancing across silky sea ripples",
        ("#F49D6E", "#E77D8B", "#FFF1D0"),
    ),
    (
        "S33", 6, 226.5, 235.0,
        "Euphoric Wide Tracking (Hero)", "Sweeping pull-back & pan",
        "Train gliding through golden hour sea; glowing wind-blown petals & light-motes drifting across lens",
        ("#F49D6E", "#E77D8B", "#5C8FD6", "#FFF1D0"),
    ),
    # ACT VII — THE DISTANT HORIZON (03:55.0 – 04:35.0 | 4 shots | 40.0s)
    (
        "S34", 7, 235.0, 245.0,
        "Ultra-Wide Pullback", "Slow continuous pull-back",
        "Lingering pullback as warm-lit train grows smaller across the vast mirrored lavender-peach sea",
        ("#3E3A5C", "#F6C7A8", "#191D32"),
    ),
    (
        "S35", 7, 245.0, 255.0,
        "Water-Reflection Medium", "Slow serene drift",
        "Glassy twilight water reflecting first pale stars blinking awake beside fading sunset peach clouds",
        ("#191D32", "#3E3A5C", "#F6C7A8"),
    ),
    (
        "S36", 7, 255.0, 265.0,
        "Long Telephoto Horizon", "Still with subtle breath",
        "Tiny warm amber cabin lantern glow receding steadily into the deep indigo distance",
        ("#191D32", "#3E3A5C", "#F5B969"),
    ),
    (
        "S37", 7, 265.0, 275.0,
        "Vast Horizon Silhouette", "Ultra-slow crane-up",
        "Vast serene lavender-indigo sea & sky horizon; faint train trail dissolving into starlight",
        ("#191D32", "#3E3A5C", "#F6C7A8"),
    ),
    # ACT VIII — QUIET RESOLUTION & TITLE (04:35.0 – 05:00.0 | 3 shots | 25.0s)
    (
        "S38", 8, 275.0, 283.0,
        "Macro Water Surface", "Locked serene",
        "Final gentle water ripple expanding and settling into absolute glassy stillness",
        ("#191D32", "#3E3A5C", "#F4EFE6"),
    ),
    (
        "S39", 8, 283.0, 291.0,
        "Abstract Painterly Dissolve", "Organic watercolor bloom",
        "Soft watercolor wash bleeds outward across the frame, dissolving indigo twilight into warm handmade paper",
        ("#F4EFE6", "#3E3A5C", "#F6C7A8"),
    ),
    (
        "S40", 8, 291.0, 300.0,
        "Title Card (Centered)", "Static paper-grain breathe",
        "Minimal serif title: THE LAST RAIN TRAIN (最後の雨列車) in charcoal ink on handmade paper, holding to 05:00.0",
        ("#F4EFE6", "#232120"),
    ),
]

SHOTS: List[ShotSpec] = [
    ShotSpec(
        id=sid,
        act=act,
        start_s=start_s,
        end_s=end_s,
        start_frame=int(round(start_s * FPS)),
        end_frame=int(round(end_s * FPS)),
        shot_type=shot_type,
        camera_motion=camera_motion,
        description=description,
        palette=palette,
    )
    for sid, act, start_s, end_s, shot_type, camera_motion, description, palette in _RAW_SHOTS
]

_SHOT_BY_ID: Dict[str, ShotSpec] = {s.id: s for s in SHOTS}

ACT_RANGES: Dict[int, Tuple[float, float, int, int]] = {}
for _act_num in range(1, 9):
    _act_shots = [s for s in SHOTS if s.act == _act_num]
    ACT_RANGES[_act_num] = (
        _act_shots[0].start_s,
        _act_shots[-1].end_s,
        _act_shots[0].start_frame,
        _act_shots[-1].end_frame,
    )


def get_shot(shot_id: str) -> ShotSpec:
    return _SHOT_BY_ID[shot_id]


def get_act_shots(act: int) -> List[ShotSpec]:
    return [s for s in SHOTS if s.act == act]


def get_shot_at_frame(frame_idx: int) -> ShotSpec:
    for s in SHOTS:
        if s.start_frame <= frame_idx < s.end_frame:
            return s
    raise IndexError(f"Frame {frame_idx} out of range [0, {TOTAL_FRAMES})")


def verify_storyboard() -> None:
    """Assert all strict storyboard invariants."""
    assert len(SHOTS) == 40, f"Expected 40 shots, got {len(SHOTS)}"
    assert SHOTS[0].start_s == 0.0, f"First shot must start at 0.0s, got {SHOTS[0].start_s}"
    assert SHOTS[0].start_frame == 0, f"First frame must be 0, got {SHOTS[0].start_frame}"
    assert SHOTS[-1].end_s == TOTAL_DURATION_S, f"Last shot must end at {TOTAL_DURATION_S}s, got {SHOTS[-1].end_s}"
    assert SHOTS[-1].end_frame == TOTAL_FRAMES, f"Last frame must be {TOTAL_FRAMES}, got {SHOTS[-1].end_frame}"

    prev_end_s = 0.0
    prev_end_f = 0
    for idx, s in enumerate(SHOTS, start=1):
        expected_id = f"S{idx:02d}"
        assert s.id == expected_id, f"Expected shot id {expected_id}, got {s.id}"
        assert s.start_s == prev_end_s, f"Non-contiguous start_s at {s.id}: {s.start_s} != {prev_end_s}"
        assert s.start_frame == prev_end_f, f"Non-contiguous start_frame at {s.id}: {s.start_frame} != {prev_end_f}"
        dur = s.duration_s
        assert 6.0 <= dur <= 10.0, f"Shot {s.id} duration {dur}s outside [6.0, 10.0]"
        assert s.end_frame - s.start_frame == int(round(dur * FPS)), f"Frame count mismatch at {s.id}"
        prev_end_s = s.end_s
        prev_end_f = s.end_frame

    assert len(ACT_RANGES) == 8
    total_act_frames = sum(ef - sf for _, _, sf, ef in ACT_RANGES.values())
    assert total_act_frames == TOTAL_FRAMES, f"Act frames sum {total_act_frames} != {TOTAL_FRAMES}"


verify_storyboard()
