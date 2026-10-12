"""Frame-drop detection and constant-frame-rate retiming plans.

Drops come in two forms:

* **Timestamp gaps** — variable-frame-rate captures simply omit frames; the gap between two timestamps is
  larger than the nominal frame interval. Retiming onto a constant grid fills them automatically.
* **Duplicated frames** — CFR encoders or capture tools repeat the previous image when a frame is missing.
  Signature: (almost) zero difference to the previous frame while the surrounding content is moving.
  Such frames are treated as holes and re-synthesised at their correct time. A *regular* cadence of
  repeats (animation drawn "on twos", 3:2 pulldown leftovers) is content, not a drop, and is left alone
  unless repair is forced.

The retiming plan maps every output frame on the target grid to: an exact copy of a source frame, an
interpolation between two unique source frames at time ``tau``, or the nearest frame when interpolation
would cross a scene cut or bridge an implausibly long gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from neuralframe.chunking.analysis import FrameAnalysis

COPY, INTERP, NEAREST = 0, 1, 2
_KIND_NAMES = {COPY: "copy", INTERP: "interp", NEAREST: "nearest"}

RepairMode = Literal["auto", "on", "off"]


_NTSC = {"23.976": Fraction(24000, 1001), "29.97": Fraction(30000, 1001), "47.952": Fraction(48000, 1001),
         "59.94": Fraction(60000, 1001), "119.88": Fraction(120000, 1001)}


def parse_fps(text: str | float | Fraction) -> Fraction:
    """"60", "59.94" (→ 60000/1001), "60000/1001", 48.0 ..."""
    if isinstance(text, Fraction):
        return text
    value = str(text).strip()
    if value in _NTSC:
        return _NTSC[value]
    if "/" in value:
        num, den = value.split("/", 1)
        result = Fraction(int(num), int(den))
    else:
        result = Fraction(value).limit_denominator(1000)
    if result <= 0 or result > 1000:
        raise ValueError(f"unsupported frame rate {text!r}")
    return result


def snap_fps(target: Fraction, source_fps: Fraction, *, ntsc_snap: bool = True) -> Fraction:
    """60 → 60000/1001 when the source runs on an NTSC clock (23.976/29.97/59.94), so the conversion is an
    exact ratio instead of drifting by one frame every ~16 s."""
    if ntsc_snap and source_fps.denominator == 1001 and target.denominator == 1:
        return Fraction(target.numerator * 1000, 1001)
    return target


@dataclass
class DropReport:
    duplicates: list[int] = field(default_factory=list)
    """Source indices whose image repeats the previous frame inside moving content (dropped frames)."""
    cadence_holds: int = 0
    """Repeats left alone because they follow a regular cadence (e.g. animation on twos)."""
    gaps: list[tuple[int, int]] = field(default_factory=list)
    """(index of the frame after the gap, number of missing frames) for timestamp gaps."""
    nominal_interval_s: float = 0.0

    @property
    def missing_from_gaps(self) -> int:
        return sum(n for _, n in self.gaps)

    def summary(self) -> dict[str, Any]:
        return {
            "duplicated_frames": len(self.duplicates),
            "duplicate_indices": self.duplicates[:50],
            "cadence_holds_kept": self.cadence_holds,
            "timestamp_gaps": len(self.gaps),
            "frames_missing_in_gaps": self.missing_from_gaps,
            "nominal_interval_ms": round(self.nominal_interval_s * 1000, 3),
        }


def detect_drops(
    analysis: FrameAnalysis,
    scene_cuts: list[int],
    *,
    repair: RepairMode = "auto",
    dup_threshold: float = 0.0025,
    motion_threshold: float = 0.008,
    window: int = 6,
    cadence_ratio: float = 0.3,
) -> DropReport:
    times = analysis.times()
    mad = analysis.mad
    n = analysis.frames
    deltas = np.diff(times)
    nominal = float(np.median(deltas)) if deltas.size else 0.0
    report = DropReport(nominal_interval_s=nominal)
    if nominal > 0:
        for i, d in enumerate(deltas, start=1):
            if d > 1.5 * nominal:
                report.gaps.append((i, round(d / nominal) - 1))
    if repair == "off" or n < 3:
        return report
    cut_set = set(scene_cuts)
    still = mad < dup_threshold
    for i in range(1, n):
        if not still[i] or i in cut_set:
            continue
        lo, hi = max(1, i - window), min(n, i + window + 1)
        neighbours = [mad[j] for j in range(lo, hi) if j != i and not still[j] and j not in cut_set]
        if len(neighbours) < 2 or float(np.median(neighbours)) <= motion_threshold:
            continue  # static shot: identical frames are content, not drops
        repeats_nearby = int(np.count_nonzero(still[lo:hi]))
        if repair == "auto" and repeats_nearby / (hi - lo) >= cadence_ratio:
            report.cadence_holds += 1  # regular holds (animation on twos), not sporadic drops
            continue
        report.duplicates.append(i)
    return report


@dataclass
class RetimePlan:
    fps: Fraction
    time_base: Fraction
    step: int
    """Output frame duration in ``time_base`` units."""
    pts: npt.NDArray[np.int64]
    src_a: npt.NDArray[np.int64]
    src_b: npt.NDArray[np.int64]
    tau: npt.NDArray[np.float32]
    kind: npt.NDArray[np.int8]
    drops: DropReport
    notes: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return int(self.pts.size)

    def needed_sources_from(self, out_index: int) -> int:
        """Smallest source index any output frame at or after ``out_index`` reads."""
        if out_index >= self.frames:
            return 10**12
        return int(np.min(self.src_a[out_index:]))

    def summary(self) -> dict[str, Any]:
        counts = {name: int(np.count_nonzero(self.kind == code)) for code, name in _KIND_NAMES.items()}
        return {
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "output_frames": self.frames,
            "copied": counts["copy"],
            "interpolated": counts["interp"],
            "nearest_at_cuts_or_long_gaps": counts["nearest"],
            "drops": self.drops.summary(),
            "notes": self.notes,
        }


def plan_retime(
    analysis: FrameAnalysis,
    scene_cuts: list[int],
    target_fps: Fraction,
    drops: DropReport,
    *,
    max_gap_s: float = 0.5,
) -> RetimePlan:
    """Map a constant-rate output grid onto the unique source frames."""
    times = analysis.times()
    n = analysis.frames
    if n == 0:
        raise ValueError("cannot retime an empty video")
    nominal = drops.nominal_interval_s or (float(times[1] - times[0]) if n > 1 else 1.0 / float(target_fps))
    dup = np.zeros(n, dtype=bool)
    dup[drops.duplicates] = True
    unique_idx = np.nonzero(~dup)[0]
    unique_t = times[unique_idx]
    cut_set = set(scene_cuts)

    # Integer PTS on a 1/numerator clock: one output frame is ``denominator`` ticks (60000/1001 → 1001).
    time_base = Fraction(1, target_fps.numerator)
    step = target_fps.denominator
    start = float(times[0])
    end = float(times[-1]) + nominal
    count = max(1, round((end - start) * float(target_fps)))
    start_pts = round(start * target_fps.numerator)
    out_t = start + np.arange(count) / float(target_fps)
    tolerance = 0.1 / max(float(target_fps), 1.0 / nominal if nominal > 0 else 1.0)

    pos = np.searchsorted(unique_t, out_t, side="right") - 1
    pos = np.clip(pos, 0, unique_t.size - 1)
    a = unique_idx[pos]
    nxt = np.clip(pos + 1, 0, unique_t.size - 1)
    b = unique_idx[nxt]
    ta, tb_next = unique_t[pos], unique_t[nxt]
    span = np.maximum(tb_next - ta, 1e-12)
    tau = np.clip((out_t - ta) / span, 0.0, 1.0).astype(np.float32)
    kind = np.full(count, INTERP, dtype=np.int8)

    exact_a = np.abs(out_t - ta) <= tolerance
    exact_b = (np.abs(tb_next - out_t) <= tolerance) & (nxt != pos)
    last = nxt == pos  # past the last unique frame: hold it
    kind[exact_a | last] = COPY
    a = np.where(exact_b & ~exact_a, b, a)
    kind[exact_b & ~exact_a] = COPY
    is_cut = np.zeros(n + 1, dtype=np.int64)
    for c in cut_set:
        if 0 < c < n:
            is_cut[c] = 1
    cuts_upto = np.cumsum(is_cut)
    crosses_cut = (cuts_upto[b] - cuts_upto[a]) > 0  # a cut starts somewhere in (a, b]
    too_long = span > max_gap_s
    nearest = (kind == INTERP) & (crosses_cut | too_long)
    pick_b = nearest & (tau >= 0.5)
    a = np.where(pick_b, b, a)
    kind[nearest] = NEAREST
    tau[kind != INTERP] = 0.0
    b = np.where(kind == INTERP, b, a)

    notes: list[str] = []
    src_fps = 1.0 / nominal if nominal > 0 else 0.0
    if src_fps > float(target_fps) * 1.05:
        notes.append(f"source ({src_fps:.3f} fps) is faster than the target; frames are selected, not blended")
    if drops.duplicates:
        notes.append(f"{len(drops.duplicates)} duplicated (dropped) frames re-synthesised")
    if drops.gaps:
        notes.append(f"{len(drops.gaps)} timestamp gaps ({drops.missing_from_gaps} missing frames) filled")
    pts = (start_pts + np.arange(count, dtype=np.int64) * step).astype(np.int64)
    return RetimePlan(
        fps=target_fps, time_base=time_base, step=step, pts=pts, src_a=a.astype(np.int64),
        src_b=b.astype(np.int64), tau=tau, kind=kind, drops=drops, notes=notes,
    )
