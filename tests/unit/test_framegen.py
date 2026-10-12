from __future__ import annotations

from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from neuralframe.chunking.analysis import FrameAnalysis
from neuralframe.framegen.retime import COPY, INTERP, NEAREST, detect_drops, parse_fps, plan_retime, snap_fps

TB = Fraction(1, 30000)


def analysis(pts: list[int], mad: list[float]) -> FrameAnalysis:
    return FrameAnalysis(TB, np.array(pts, dtype=np.int64), np.array(mad, dtype=np.float32),
                         np.zeros(len(pts), np.float32))


def moving(n: int, step: int = 1001) -> FrameAnalysis:
    mad = [0.0] + [0.02] * (n - 1)
    return analysis([i * step for i in range(n)], mad)


@pytest.mark.parametrize(("text", "expected"), [
    ("60", Fraction(60)), ("59.94", Fraction(60000, 1001)), ("60000/1001", Fraction(60000, 1001)),
    ("23.976", Fraction(24000, 1001)), ("48", Fraction(48)), ("25.5", Fraction(51, 2)),
])
def test_parse_fps(text: str, expected: Fraction) -> None:
    assert parse_fps(text) == expected


def test_parse_fps_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        parse_fps("0")


def test_snap_to_ntsc_clock() -> None:
    assert snap_fps(Fraction(60), Fraction(30000, 1001)) == Fraction(60000, 1001)
    assert snap_fps(Fraction(60), Fraction(25)) == Fraction(60)
    assert snap_fps(Fraction(60), Fraction(30000, 1001), ntsc_snap=False) == Fraction(60)


def test_sporadic_duplicate_in_motion_is_a_drop() -> None:
    a = moving(30)
    a.mad[12] = 0.0004
    report = detect_drops(a, [])
    assert report.duplicates == [12]


def test_static_shot_repeats_are_not_drops() -> None:
    a = analysis([i * 1001 for i in range(30)], [0.0] + [0.0005] * 29)
    assert detect_drops(a, []).duplicates == []


def test_animation_on_twos_is_a_cadence_not_drops() -> None:
    mad = [0.0] + [0.0003 if i % 2 == 0 else 0.03 for i in range(1, 40)]
    a = analysis([i * 1001 for i in range(40)], mad)
    auto = detect_drops(a, [], repair="auto")
    assert auto.duplicates == [] and auto.cadence_holds > 10
    forced = detect_drops(a, [], repair="on")
    assert len(forced.duplicates) > 15


def test_timestamp_gaps_are_counted() -> None:
    pts = [i * 1001 for i in range(10)] + [(i + 3) * 1001 for i in range(10, 20)]
    report = detect_drops(analysis(pts, [0.0] + [0.02] * 19), [])
    assert report.gaps == [(10, 3)]
    assert report.missing_from_gaps == 3


def test_retime_doubles_ntsc_rate_with_copies_and_midpoints() -> None:
    a = moving(10)
    plan = plan_retime(a, [], Fraction(60000, 1001), detect_drops(a, []))
    assert plan.frames == 20
    assert list(plan.kind[:4]) == [COPY, INTERP, COPY, INTERP]
    np.testing.assert_allclose(plan.tau[1::2][:-1], 0.5)
    assert plan.time_base == Fraction(1, 60000) and plan.step == 1001
    assert np.all(np.diff(plan.pts) == 1001)


def test_duplicate_is_resynthesised_between_neighbours() -> None:
    a = moving(20)
    a.mad[8] = 0.0
    plan = plan_retime(a, [], Fraction(30000, 1001), detect_drops(a, []))
    k = 8  # same rate: output 8 sits exactly where the duplicated source frame was
    assert plan.kind[k] == INTERP and plan.src_a[k] == 7 and plan.src_b[k] == 9
    assert plan.tau[k] == pytest.approx(0.5)
    assert 8 not in plan.src_a[plan.kind == COPY]


def test_no_interpolation_across_scene_cut() -> None:
    a = moving(20)
    plan = plan_retime(a, [10], Fraction(60000, 1001), detect_drops(a, [10]))
    around = plan.kind[(plan.src_a == 9) | (plan.src_b == 10)]
    assert NEAREST in around
    assert not np.any((plan.kind == INTERP) & (plan.src_a == 9) & (plan.src_b == 10))


def test_source_faster_than_target_selects_frames() -> None:
    a = moving(60, step=500)  # 60 fps source
    plan = plan_retime(a, [], Fraction(30), detect_drops(a, []))
    assert plan.frames == 30
    assert np.all(plan.kind == COPY)
    assert any("faster than the target" in n for n in plan.notes)


def test_long_gap_holds_instead_of_morphing() -> None:
    pts = [i * 1001 for i in range(5)] + [(i + 60) * 1001 for i in range(5, 10)]  # 2 s hole
    a = analysis(pts, [0.0] + [0.02] * 9)
    plan = plan_retime(a, [], Fraction(30000, 1001), detect_drops(a, []), max_gap_s=0.5)
    assert np.count_nonzero(plan.kind == NEAREST) > 50


def test_retime_stage_copies_bit_exact_and_orders_output() -> None:
    torch = pytest.importorskip("torch")
    from neuralframe.colorspace import ChromaLocation, ColorPlan, ColorRange, Matrix, Primaries, Transfer
    from neuralframe.core.runtime import CancelToken, Frame, StageMetrics, StageQueue, StageRunner
    from neuralframe.framegen.interpolate import RetimeStage
    from neuralframe.io.rawvideo import RawVideoFormat

    class Blend:
        device = torch.device("cpu")
        seconds = 0.0

        def interpolate(self, a: Any, b: Any, taus: list[float]) -> Any:
            return torch.stack([a * (1 - t) + b * t for t in taus])

        def close(self) -> None:
            pass

    fmt = RawVideoFormat(8, 4, 8, 1, 1)
    color = ColorPlan(Matrix.BT709, ColorRange.LIMITED, Primaries.BT709, Transfer.BT709, ChromaLocation.LEFT,
                      Matrix.BT709, ColorRange.LIMITED, Primaries.BT709, Transfer.BT709)
    a = moving(6)
    plan = plan_retime(a, [], Fraction(60000, 1001), detect_drops(a, []))
    cancel = CancelToken()
    q_in: StageQueue[Any] = StageQueue("in", 3, cancel)
    q_out: StageQueue[Any] = StageQueue("out", 100, cancel)
    sources = [np.full(fmt.frame_samples, 16 + 20 * i, np.uint8) for i in range(6)]

    class Feed:
        name = "feed"
        metrics = StageMetrics("feed")

        def run(self) -> None:
            for i, data in enumerate(sources):
                q_in.put(Frame(i, int(a.pts[i]), 1001, data))
            q_in.close()

    stage = RetimeStage(cancel, q_in, q_out, plan=plan, interpolator=Blend(), fmt=fmt, color=color)  # type: ignore[arg-type]
    StageRunner([Feed(), stage], cancel).run()  # type: ignore[list-item]
    out = []
    while (f := q_out.get()) is not None:
        out.append(f)
    assert [f.index for f in out] == list(range(plan.frames))
    assert all(np.array_equal(out[2 * i].data, sources[i]) for i in range(6))  # copies untouched
    assert stage.interpolated == 5 and stage.copied == 7


def test_interpolator_splits_timesteps_after_oom(tmp_path: Any) -> None:
    torch = pytest.importorskip("torch")
    from neuralframe.framegen.interpolate import Interpolator
    from tests.models import tiny_rife_model

    spec, weights = tiny_rife_model(tmp_path)
    interp = Interpolator(spec, weights, torch.device("cpu"), "fp32")
    real = interp.net

    class Limited:
        scale = 1.0

        def multiple(self) -> int:
            return int(real.multiple())

        def __call__(self, a: Any, b: Any, t: Any) -> Any:
            if a.shape[0] > 2:
                raise torch.OutOfMemoryError("simulated")
            real.scale = self.scale
            return real(a, b, t)

    interp.net = Limited()
    a, b = torch.rand(3, 40, 72), torch.rand(3, 40, 72)
    out = interp.interpolate(a, b, [0.2, 0.4, 0.6, 0.8, 0.9])
    assert out.shape == (5, 3, 40, 72)
    assert interp.batch_cap == 2 and interp.oom_retries == 1
    torch.testing.assert_close(out, ((a + b) / 2).expand(5, -1, -1, -1), atol=1e-4, rtol=0)


@pytest.mark.parametrize(("model", "requested", "expected"), [
    ("lanczos", "auto", "interpolate_first"),
    ("realesr-animevideov3", "auto", "interpolate_first"),
    ("realesrgan-x4plus", "auto", "upscale_first"),
    ("realesrgan-x4plus", "interpolate_first", "interpolate_first"),
    ("realbasicvsr-x4", "interpolate_first", "upscale_first"),  # temporal models need the original frames
])
def test_framegen_order_resolution(settings: Any, tmp_path: Any, model: str, requested: str, expected: str) -> None:
    from neuralframe.core.job import JobRunner, JobSpec
    from neuralframe.models.registry import ModelRegistry

    registry = ModelRegistry.load()
    src = tmp_path / "in.mp4"
    src.write_bytes(b"")
    spec = JobSpec(input=src, output=tmp_path / "out.mkv", interpolate=True, framegen_order=requested,  # type: ignore[arg-type]
                   framegen_engine="rife")
    notes: list[str] = []
    engine, order = JobRunner(settings, registry=registry).resolve_framegen(spec, registry.get(model), notes)
    assert (engine, order) == ("rife", expected)
