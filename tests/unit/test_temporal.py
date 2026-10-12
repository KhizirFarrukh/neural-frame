"""TemporalStage: chunking with overlap is invisible for windowed models, never crosses scene cuts,
and cross-fades boundaries of recurrent models."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralframe.chunking.planner import plan_chunks  # noqa: E402
from neuralframe.colorspace import ChromaLocation, ColorPlan, ColorRange, Matrix, Primaries, Transfer  # noqa: E402
from neuralframe.core.runtime import CancelToken, Frame, StageQueue, StageRunner  # noqa: E402
from neuralframe.inference.temporal import TemporalStage, crossfade_weight  # noqa: E402
from neuralframe.io.rawvideo import RawVideoFormat  # noqa: E402
from neuralframe.ops import reference as ref  # noqa: E402

FMT = RawVideoFormat(16, 12, 8, 0, 0)  # 4:4:4 keeps the colour round trip simple
OUT = RawVideoFormat(32, 24, 8, 0, 0)
COLOR = ColorPlan(Matrix.BT709, ColorRange.FULL, Primaries.BT709, Transfer.BT709, ChromaLocation.LEFT,
                  Matrix.BT709, ColorRange.FULL, Primaries.BT709, Transfer.BT709)


class WindowMean:
    """Fake windowed model: output_t = mean of inputs t-r..t+r (clamped), nearest x2."""

    def __init__(self, radius: int, recurrent: bool = False) -> None:
        self.radius, self.recurrent = radius, recurrent
        self.device = torch.device("cpu")
        self.frames_processed, self.seconds, self.calls = 0, 0.0, 0

    def run(self, frames: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
        self.calls += 1
        t = frames.shape[0]
        state = frames[0]
        for i in range(t):
            if self.recurrent:  # depends on where the sequence started (like a hidden state)
                state = 0.7 * state + 0.3 * frames[i]
                out = state
            else:
                lo, hi = max(0, i - self.radius), min(t, i + self.radius + 1)
                out = frames[lo:hi].mean(dim=0)
            yield i, out.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        self.frames_processed += t

    def close(self) -> None:
        pass


def make_frames(values: list[float]) -> list[Frame[Any]]:
    frames = []
    for i, v in enumerate(values):
        rgb = np.full((3, FMT.height, FMT.width), v, dtype=np.float32)
        rgb[1] *= 0.5
        planes = ref.rgb_to_yuv(rgb, matrix=Matrix.BT709, full_range=True, bit_depth=8, log2_cw=0, log2_ch=0)
        frames.append(Frame(i, i * 1000, 1000, FMT.join(planes)))
    return frames


def run_stage(frames: list[Frame[Any]], model: Any, chunk: int, overlap: int, cuts: list[int],
              replicate: bool, done: set[int] | None = None) -> tuple[dict[int, np.ndarray], TemporalStage]:
    cancel = CancelToken()
    q_in: StageQueue[Any] = StageQueue("in", 4, cancel)
    q_out: StageQueue[Any] = StageQueue("out", 1000, cancel)
    plan = plan_chunks(len(frames), chunk, scene_cuts=cuts, force_cut_boundaries=True)
    stage = TemporalStage(cancel, q_in, q_out, model=model, chunks=plan, overlap=overlap, in_fmt=FMT,
                          out_fmt=OUT, color=COLOR, kernel="lanczos3", done_chunks=done, replicate_edges=replicate)

    class Feed:
        name = "feed"

        def __init__(self) -> None:
            from neuralframe.core.runtime import StageMetrics

            self.metrics = StageMetrics("feed")

        def run(self) -> None:
            for f in frames:
                q_in.put(f)
            q_in.close()

    StageRunner([Feed(), stage], cancel).run()  # type: ignore[list-item]
    out: dict[int, np.ndarray] = {}
    while (item := q_out.get()) is not None:
        assert item.index not in out
        out[item.index] = item.data.astype(np.int32)
    return out, stage


def test_windowed_model_chunked_equals_unchunked() -> None:
    values = list(np.linspace(0.1, 0.9, 40))
    whole, _ = run_stage(make_frames(values), WindowMean(2), chunk=1000, overlap=4, cuts=[], replicate=True)
    chunked, stage = run_stage(make_frames(values), WindowMean(2), chunk=10, overlap=4, cuts=[], replicate=True)
    assert sorted(chunked) == list(range(40))
    assert stage.blended_frames > 0
    for i in range(40):
        assert np.max(np.abs(chunked[i] - whole[i])) <= 1


def test_no_window_crosses_a_scene_cut() -> None:
    values = [0.2] * 12 + [0.8] * 12  # cut at 12: two flat shots
    out, _ = run_stage(make_frames(values), WindowMean(3), chunk=6, overlap=4, cuts=[12], replicate=True)
    first, second = out[0], out[23]
    for i in range(12):
        np.testing.assert_array_equal(out[i], first)
    for i in range(12, 24):
        np.testing.assert_array_equal(out[i], second)


def test_recurrent_boundaries_are_cross_faded_in_order() -> None:
    values = list(np.linspace(0.1, 0.9, 30))
    out, stage = run_stage(make_frames(values), WindowMean(0, recurrent=True), chunk=10, overlap=4, cuts=[],
                           replicate=False)
    assert sorted(out) == list(range(30))
    assert stage.blended_frames == 2 * 4  # two boundaries x (2 * half) frames


def test_resume_skips_finished_chunks_but_rebuilds_blend_zone() -> None:
    values = list(np.linspace(0.1, 0.9, 30))
    full, _ = run_stage(make_frames(values), WindowMean(0, recurrent=True), chunk=10, overlap=4, cuts=[],
                        replicate=False)
    model = WindowMean(0, recurrent=True)
    resumed, _ = run_stage(make_frames(values), model, chunk=10, overlap=4, cuts=[], replicate=False, done={0})
    assert sorted(resumed) == list(range(10, 30))  # chunk 0 is already in the store
    for i in range(10, 30):
        np.testing.assert_array_equal(resumed[i], full[i])
    assert model.calls == 3  # chunk 0 re-run only to rebuild the zone it shares with chunk 1


def test_crossfade_weights_partition_unity() -> None:
    w = [crossfade_weight(p, 8) for p in range(8)]
    assert w[0] > 0.9 and w[-1] < 0.1
    assert all(a > b for a, b in itertools.pairwise(w))
