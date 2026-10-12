"""Low-resolution analysis pass: per-frame timestamps, motion and colour statistics, scene cuts.

One decode at thumbnail size through the *same* filter chain as the main decode (so frame counts and
timestamps line up exactly) yields everything the planners need:

* scene cuts — a cut changes the colour distribution, plain motion mostly does not, so a cut needs both a
  large histogram change and a large mean pixel change;
* motion (mean absolute difference between consecutive frames) — used by frame generation to tell a
  duplicated (dropped) frame from a genuinely static shot;
* exact presentation timestamps — gaps reveal dropped frames in variable-frame-rate captures.

Results are cached as ``.npz`` next to the job's chunk store.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import numpy.typing as npt

from neuralframe.io.decoder import DecodePlan, FrameDecoder
from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.io.rawvideo import RawVideoFormat
from neuralframe.log import get_logger

log = get_logger(component="analysis")

THUMB_W, THUMB_H = 64, 36
_BINS = 16


@dataclass
class FrameAnalysis:
    time_base: Fraction
    pts: npt.NDArray[np.int64]
    mad: npt.NDArray[np.float32]
    """Mean |frame_i - frame_{i-1}| in [0, 1] (0 for the first frame)."""
    hist: npt.NDArray[np.float32]
    """Half L1 distance of 3x16-bin RGB histograms between consecutive frames, in [0, 1]."""
    seconds: float = 0.0

    @property
    def frames(self) -> int:
        return int(self.pts.size)

    def times(self) -> npt.NDArray[np.float64]:
        return self.pts.astype(np.float64) * float(self.time_base)

    def scene_cuts(self, threshold: float = 0.35, min_mad: float = 0.06, min_gap: int = 4) -> list[int]:
        """Frame indices that start a new shot (never index 0)."""
        candidates = np.nonzero((self.hist > threshold) & (self.mad > min_mad))[0]
        cuts: list[int] = []
        for i in candidates.tolist():
            if i == 0:
                continue
            if cuts and i - cuts[-1] < min_gap:
                # keep the stronger of two cuts that are too close together (flash frames, fades)
                if self.hist[i] > self.hist[cuts[-1]]:
                    cuts[-1] = i
                continue
            cuts.append(i)
        return cuts

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, pts=self.pts, mad=self.mad, hist=self.hist,
            tb=np.array([self.time_base.numerator, self.time_base.denominator], dtype=np.int64),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> FrameAnalysis:
        data = np.load(path)
        tb = data["tb"]
        return cls(Fraction(int(tb[0]), int(tb[1])), data["pts"], data["mad"], data["hist"])


def _histogram(thumb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    planes = thumb.reshape(3, -1)
    hist = np.stack([np.bincount(p >> 4, minlength=_BINS) for p in planes]).astype(np.float32)
    return hist / float(planes.shape[1])


def analyse(
    tools: FFmpegTools,
    path: Path,
    stream_index: int,
    plan: DecodePlan,
    *,
    cache: Path | None = None,
    loglevel: str = "error",
) -> FrameAnalysis:
    if cache is not None and cache.is_file():
        try:
            return FrameAnalysis.load(cache)
        except Exception:
            log.warning("ignoring unreadable analysis cache", path=str(cache))
    t0 = time.perf_counter()
    fmt = RawVideoFormat(THUMB_W, THUMB_H, 8, 0, 0, rgb=True)  # planar GBR thumbnails
    thumb_plan = DecodePlan(
        fmt=fmt,
        hwaccel=plan.hwaccel,
        filters=(*plan.filters, f"scale={THUMB_W}:{THUMB_H}:flags=area"),
    )
    pts: list[int] = []
    mads: list[float] = []
    hists: list[float] = []
    prev: npt.NDArray[np.int16] | None = None
    prev_hist: npt.NDArray[np.float32] | None = None
    with FrameDecoder(tools, path, stream_index, thumb_plan, loglevel=loglevel) as dec:
        time_base = dec.time_base
        for frame in dec:
            thumb = frame.data.astype(np.uint8, copy=False)
            cur = thumb.astype(np.int16)
            h = _histogram(thumb)
            if prev is None or prev_hist is None:
                mads.append(0.0)
                hists.append(0.0)
            else:
                mads.append(float(np.abs(cur - prev).mean()) / 255.0)
                hists.append(float(np.abs(h - prev_hist).sum()) / 6.0)
            prev, prev_hist = cur, h
            pts.append(frame.pts)
    result = FrameAnalysis(
        time_base=time_base,
        pts=np.asarray(pts, dtype=np.int64),
        mad=np.asarray(mads, dtype=np.float32),
        hist=np.asarray(hists, dtype=np.float32),
        seconds=time.perf_counter() - t0,
    )
    log.info("analysis done", frames=result.frames, seconds=round(result.seconds, 2))
    if cache is not None:
        result.save(cache)
    return result
