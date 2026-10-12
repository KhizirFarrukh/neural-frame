"""Frame generation end to end: constant-rate output, drop repair, scene cuts, upscale ordering, resume."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralframe.config import Settings, default_cache_dir  # noqa: E402
from neuralframe.core.job import JobRunner, JobSpec  # noqa: E402
from neuralframe.core.planning import TargetSpec  # noqa: E402
from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError  # noqa: E402
from neuralframe.io.ffmpeg import FFmpegTools  # noqa: E402
from neuralframe.io.hwcaps import Capabilities  # noqa: E402
from neuralframe.io.probe import probe  # noqa: E402
from tests.conftest import ClipFactory  # noqa: E402
from tests.models import registry_with, tiny_rife_model, tiny_sr_model  # noqa: E402

LOSSLESS: dict[str, Any] = {"encoder": "libx264", "codec": "h264", "quality": 0}


@pytest.fixture
def fg_registry(settings: Settings) -> Any:
    settings.inference.device = "cpu"
    settings.framegen.model = "tiny-rife"
    rife, _ = tiny_rife_model(settings.paths.models())
    sr, _ = tiny_sr_model(settings.paths.models(), scale=2)
    return registry_with(rife, sr)


def lumas(path: Path) -> list[np.ndarray]:
    """Decoded Y planes (exact samples, no conversion)."""
    with av.open(str(path)) as container:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray()[: frame.height].astype(np.float32))
        return frames


def assert_cfr(times: list[float], fps: Fraction) -> None:
    np.testing.assert_allclose(np.diff(times), float(1 / fps), atol=1.5e-3)


def run(settings: Settings, caps: Capabilities, registry: Any, **fields: Any) -> dict[str, Any]:
    spec = JobSpec(overwrite=True, **{**LOSSLESS, **fields})
    return JobRunner(settings, capabilities=caps, registry=registry).run(spec).report


def test_doubles_ntsc_rate_with_bit_exact_copies(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, fg_registry: Any, clip: ClipFactory,
    tmp_path: Path, ffhelpers: Any,
) -> None:
    src = clip("fg-ntsc", width=160, height=96, fps="30000/1001", duration=2.0)
    out = tmp_path / "fg.mkv"
    report = run(settings, caps, fg_registry, input=src, output=out, upscale=False, interpolate=True,
                 target_fps="60")
    assert report["sync"]["ok"], report["sync"]
    fg = report["framegen"]
    assert fg["engine"] == "rife" and fg["fps"] == "60000/1001"  # snapped to the NTSC clock: exact 2x
    assert "inference" not in report["throughput"]["stages"]  # framegen only: no pixel processing pass
    video = probe(out, tools).require_video()
    assert (video.width, video.height) == (160, 96)
    times = ffhelpers.frame_times(out)
    assert len(times) == report["output"]["frames"] == fg["output_frames"] == 120
    assert_cfr(times, Fraction(60000, 1001))
    src_md5, out_md5 = ffhelpers.frame_md5s(src), ffhelpers.frame_md5s(out)
    assert out_md5[0::2] == src_md5  # every source frame survives untouched
    src_y, out_y = lumas(src), lumas(out)
    midpoint = (src_y[10] + src_y[11]) / 2  # the tiny RIFE is an exact midpoint blend
    assert np.abs(out_y[21] - midpoint).mean() < 0.75


def test_frozen_frame_is_detected_and_resynthesised(
    caps: Capabilities, settings: Settings, fg_registry: Any, clip: ClipFactory, tmp_path: Path, ffhelpers: Any,
) -> None:
    src = clip("fg-dup", width=160, height=96, fps="25", duration=2.0, duplicate_frames=(20,), crf=0)
    src_y = lumas(src)
    assert np.array_equal(src_y[20], src_y[19])  # fixture sanity: frame 20 is a frozen copy
    out = tmp_path / "dup.mkv"
    report = run(settings, caps, fg_registry, input=src, output=out, upscale=False, interpolate=True,
                 target_fps="25")
    assert report["framegen"]["drops"]["duplicate_indices"] == [20]
    out_y = lumas(out)
    assert len(out_y) == len(src_y) == 50
    assert not np.array_equal(out_y[20], out_y[19])
    assert np.abs(out_y[20] - (src_y[19] + src_y[21]) / 2).mean() < 0.75
    src_md5, out_md5 = ffhelpers.frame_md5s(src), ffhelpers.frame_md5s(out)
    assert [i for i in range(50) if src_md5[i] != out_md5[i]] == [20]


def test_timestamp_gap_is_filled_to_constant_rate(
    caps: Capabilities, settings: Settings, fg_registry: Any, clip: ClipFactory, tmp_path: Path, ffhelpers: Any,
) -> None:
    src = clip("fg-gap", width=160, height=96, fps="25", duration=2.0, drop_frames=(20, 21, 22))
    assert len(ffhelpers.frame_times(src)) == 47
    out = tmp_path / "gap.mkv"
    report = run(settings, caps, fg_registry, input=src, output=out, upscale=False, interpolate=True,
                 target_fps="25")
    drops = report["framegen"]["drops"]
    assert drops["timestamp_gaps"] == 1 and drops["frames_missing_in_gaps"] == 3
    times = ffhelpers.frame_times(out)
    assert len(times) == 50
    assert_cfr(times, Fraction(25))
    assert report["sync"]["ok"], report["sync"]


def test_no_blending_across_scene_cut(
    caps: Capabilities, settings: Settings, fg_registry: Any, clip: ClipFactory, tmp_path: Path, ffhelpers: Any,
) -> None:
    src = clip("fg-cut", width=160, height=96, fps="30000/1001", duration=2.0, scene_cut_at=1.0)
    out = tmp_path / "cut.mkv"
    report = run(settings, caps, fg_registry, input=src, output=out, upscale=False, interpolate=True,
                 target_fps="60")
    assert report["framegen"]["nearest_at_cuts_or_long_gaps"] >= 1
    src_md5, out_md5 = ffhelpers.frame_md5s(src), ffhelpers.frame_md5s(out)
    assert out_md5[59] in (src_md5[29], src_md5[30])  # between the last frame of shot 1 and the first of shot 2
    assert out_md5[58] == src_md5[29] and out_md5[60] == src_md5[30]


@pytest.mark.parametrize("order", ["upscale_first", "interpolate_first"])
def test_upscale_and_frame_generation(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, fg_registry: Any, clip: ClipFactory,
    tmp_path: Path, ffhelpers: Any, order: str,
) -> None:
    src = clip(f"fg-both-{order}", width=96, height=64, fps="24", duration=1.5)
    out = tmp_path / "both.mkv"
    report = run(settings, caps, fg_registry, input=src, output=out, model="tiny-srvgg-x2",
                 target=TargetSpec.factor(2), interpolate=True, target_fps="60", framegen_order=order)
    assert report["sync"]["ok"], report["sync"]
    assert report["framegen"]["order"] == order
    video = probe(out, tools).require_video()
    assert (video.width, video.height) == (192, 128)
    times = ffhelpers.frame_times(out)
    assert len(times) == report["framegen"]["output_frames"] == 90
    assert_cfr(times, Fraction(60))
    assert report["framegen"]["runtime"]["interpolated"] > 0


def test_kill_and_resume_with_frame_generation(
    caps: Capabilities, settings: Settings, fg_registry: Any, clip: ClipFactory, tmp_path: Path, ffhelpers: Any,
) -> None:
    src = clip("fg-resume", width=96, height=64, fps="25", duration=2.0, duplicate_frames=(30,))
    common: dict[str, Any] = {"input": src, "model": "tiny-srvgg-x2", "target": TargetSpec.factor(2),
                              "interpolate": True, "target_fps": "60", "chunk_frames": 32, **LOSSLESS}
    reference = tmp_path / "ref.mkv"
    JobRunner(settings, capabilities=caps, registry=fg_registry).run(JobSpec(output=reference, overwrite=True,
                                                                             **common))
    runner = JobRunner(settings, capabilities=caps, registry=fg_registry)

    def kill(chunk_id: int, _total: int) -> None:
        if chunk_id == 1:
            runner.cancel.cancel("simulated kill")

    runner.on_chunk_done = kill
    out = tmp_path / "resumed.mkv"
    with pytest.raises(JobCancelled):
        runner.run(JobSpec(output=out, overwrite=True, **common))
    report = JobRunner(settings, capabilities=caps, registry=fg_registry).run(
        JobSpec(output=out, resume=True, overwrite=True, **common)).report
    assert report["throughput"]["frames_resumed"] == 64
    assert ffhelpers.frame_md5s(out) == ffhelpers.frame_md5s(reference)


def test_minterpolate_fallback_engine(
    caps: Capabilities, settings: Settings, fg_registry: Any, clip: ClipFactory, tmp_path: Path, ffhelpers: Any,
) -> None:
    src = clip("fg-mi", width=160, height=96, fps="25", duration=2.0)
    out = tmp_path / "mi.mkv"
    report = run(settings, caps, fg_registry, input=src, output=out, upscale=False, interpolate=True,
                 target_fps="50", framegen_engine="minterpolate")
    assert report["framegen"]["engine"] == "minterpolate"
    times = ffhelpers.frame_times(out)
    assert abs(len(times) - 100) <= 2
    assert_cfr(times, Fraction(50))
    assert report["sync"]["ok"], report["sync"]


def test_images_cannot_be_frame_generated(
    caps: Capabilities, settings: Settings, fg_registry: Any, tmp_path: Path, ffhelpers: Any,
) -> None:
    src = tmp_path / "in.png"
    ffhelpers.text(["-f", "lavfi", "-i", "testsrc2=size=96x64:duration=0.04", "-frames:v", "1", "-update", "1",
                    "-y", str(src)])
    with pytest.raises(NeuralFrameError) as err:
        run(settings, caps, fg_registry, input=src, output=tmp_path / "out.png", interpolate=True)
    assert err.value.code is ErrorCode.INVALID_INPUT


@pytest.mark.gpu
@pytest.mark.network
def test_real_rife_beats_blending_on_cuda(
    caps: Capabilities, settings: Settings, tmp_path: Path, ffhelpers: Any,
) -> None:
    """Pan across a still at 50 fps, keep every other frame, regenerate the rest with RIFE and compare them
    with the real frames. Real motion is what interpolation is for; a cross-fade ghosts every edge."""
    settings.paths.models_dir = default_cache_dir() / "models"
    still = tmp_path / "still.png"
    ffhelpers.text(["-f", "lavfi", "-i", "testsrc2=size=640x400:duration=0.04", "-frames:v", "1", "-update", "1",
                    "-y", str(still)])
    truth = tmp_path / "truth.mkv"
    ffhelpers.text(["-loop", "1", "-framerate", "50", "-i", str(still), "-t", "2",
                    "-vf", "crop=320:192:x='t*100':y='t*50',format=yuv420p", "-c:v", "libx264", "-qp", "0",
                    "-y", str(truth)])
    half = tmp_path / "half.mkv"
    run(settings, caps, None, input=truth, output=half, upscale=False, interpolate=True, target_fps="25",
        repair_drops="off")
    out = tmp_path / "rife.mkv"
    report = run(settings, caps, None, input=half, output=out, upscale=False, interpolate=True, target_fps="50")
    assert report["framegen"]["model"] == "rife-4.22"
    truth_y, half_y, gen = lumas(truth), lumas(half), lumas(out)
    assert len(half_y) == 50 and len(gen) == len(truth_y) == 100

    def psnr(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a[8:-8, 8:-8], b[8:-8, 8:-8]  # content enters at the borders; nothing can predict it
        return float(10 * np.log10(255**2 / max(np.mean((a - b) ** 2), 1e-9)))

    odd = range(1, 98, 2)
    rife = float(np.mean([psnr(gen[i], truth_y[i]) for i in odd]))
    blend = float(np.mean([psnr((half_y[i // 2] + half_y[i // 2 + 1]) / 2, truth_y[i]) for i in odd]))
    assert rife > blend + 6.0, (rife, blend)
