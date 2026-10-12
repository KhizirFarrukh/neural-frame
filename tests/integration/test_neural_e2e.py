"""Neural upscaling end to end: tiny random model on CPU (CI), on CUDA (local), and a real model (local)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neuralframe.config import Settings, default_cache_dir  # noqa: E402
from neuralframe.core.job import JobRunner, JobSpec  # noqa: E402
from neuralframe.core.planning import TargetSpec  # noqa: E402
from neuralframe.io.ffmpeg import FFmpegTools  # noqa: E402
from neuralframe.io.hwcaps import Capabilities  # noqa: E402
from neuralframe.io.probe import probe  # noqa: E402
from tests.conftest import ClipFactory  # noqa: E402
from tests.models import registry_with, tiny_sr_model  # noqa: E402


def _run(settings: Settings, caps: Capabilities, registry: Any, **fields: Any) -> dict[str, Any]:
    fields.setdefault("overwrite", True)
    return JobRunner(settings, capabilities=caps, registry=registry).run(JobSpec(**fields)).report


@pytest.fixture
def tiny_registry(settings: Settings) -> Any:
    spec, _ = tiny_sr_model(settings.paths.models(), scale=2)
    return registry_with(spec)


def test_tiny_model_video_on_cpu(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, tiny_registry: Any, clip: ClipFactory,
    tmp_path: Path, ffhelpers: Any,
) -> None:
    settings.inference.device = "cpu"
    src = clip("nn-cpu", duration=1.0, subtitles=True)
    out = tmp_path / "nn.mkv"
    report = _run(settings, caps, tiny_registry, input=src, output=out, model="tiny-srvgg-x2",
                  target=TargetSpec.for_height(180), tile_size=64, tile_overlap=16)
    plan = report["plan"]
    assert plan["mode"] == "device"
    assert plan["inference"]["device"] == "cpu"
    assert plan["inference"]["tile"] == 64
    assert report["processor"]["runner"]["tiles"] > 30  # 30 frames x several tiles each
    assert report["sync"]["ok"]
    v = probe(out, tools).require_video()
    assert (v.width, v.height) == (320, 180)
    assert ffhelpers.packet_md5s(src, "a:0") == ffhelpers.packet_md5s(out, "a:0")


def test_host_mode_neural_matches_device_mode(
    caps: Capabilities, settings: Settings, tiny_registry: Any, clip: ClipFactory, tmp_path: Path, ffhelpers: Any
) -> None:
    """Host stages (NumPy colour/resize) + model vs. all-torch device path: same pixels within 1 LSB."""
    settings.inference.device = "cpu"
    src = clip("nn-modes", duration=0.3, audio=False)
    common: dict[str, Any] = {"input": src, "model": "tiny-srvgg-x2", "target": TargetSpec.factor(2),
                              "encoder": "libx264", "codec": "h264", "quality": 0}
    device_out = tmp_path / "device.mkv"
    _run(settings, caps, tiny_registry, output=device_out, **common)
    settings.pipeline.mode = "host"
    host_out = tmp_path / "host.mkv"
    report = _run(settings, caps, tiny_registry, output=host_out, **common)
    assert report["plan"]["mode"] == "host"
    import numpy as np

    from neuralframe.io.decoder import DecodePlan, FrameDecoder
    from neuralframe.io.ffmpeg import locate_ffmpeg
    from neuralframe.io.rawvideo import RawVideoFormat

    tools = locate_ffmpeg()
    frames = []
    for path in (device_out, host_out):
        v = probe(path, tools).require_video()
        fmt = RawVideoFormat.for_source(v.pixel_format, v.width, v.height)
        with FrameDecoder(tools, path, v.index, DecodePlan(fmt)) as dec:
            frames.append([f.data.astype(np.int32) for f in dec])
    assert len(frames[0]) == len(frames[1])
    worst = max(int(np.max(np.abs(a - b))) for a, b in zip(*frames, strict=True))
    assert worst <= 1


def test_tiny_model_image(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, tiny_registry: Any, tmp_path: Path, ffhelpers: Any
) -> None:
    settings.inference.device = "cpu"
    src = tmp_path / "in.png"
    ffhelpers.text(["-f", "lavfi", "-i", "testsrc2=size=96x64:duration=0.04", "-frames:v", "1", "-update", "1",
                    "-y", str(src)])
    out = tmp_path / "out.png"
    report = _run(settings, caps, tiny_registry, input=src, output=out, model="tiny-srvgg-x2",
                  target=TargetSpec.factor(4))
    assert report["kind"] == "image"
    v = probe(out, tools).require_video()
    assert (v.width, v.height) == (384, 256)
    assert report["plan"]["geometry"]["model_passes"] == 2  # x4 requested from an x2 model → cascaded


def test_classical_image_with_alpha_to_webp(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, tmp_path: Path, ffhelpers: Any
) -> None:
    src = tmp_path / "rgba.png"
    ffhelpers.text(["-f", "lavfi", "-i", "testsrc2=size=80x60:duration=0.04,format=rgba", "-frames:v", "1",
                    "-update", "1", "-y", str(src)])
    out = tmp_path / "out.png"
    report = JobRunner(settings, capabilities=caps).run(
        JobSpec(input=src, output=out, model="lanczos", target=TargetSpec.fit(1920, 1080), overwrite=True)
    ).report
    v = probe(out, tools).require_video()
    assert (v.width, v.height) == (1440, 1080)
    assert v.pixel_format.has_alpha
    assert report["output"]["alpha"] is True


@pytest.mark.gpu
def test_tiny_model_video_on_cuda(
    caps: Capabilities, settings: Settings, tiny_registry: Any, clip: ClipFactory, tmp_path: Path
) -> None:
    src = clip("nn-gpu", duration=1.0)
    report = _run(settings, caps, tiny_registry, input=src, output=tmp_path / "gpu.mp4", model="tiny-srvgg-x2",
                  target=TargetSpec.factor(2))
    assert report["plan"]["inference"]["device"].startswith("cuda")
    assert report["sync"]["ok"]


@pytest.mark.gpu
@pytest.mark.network
def test_real_anime_model_on_cuda(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path
) -> None:
    settings.paths.models_dir = default_cache_dir() / "models"  # reuse (or download once) real weights
    src = clip("nn-real", width=320, height=180, duration=1.0)
    out = tmp_path / "real.mkv"
    report = JobRunner(settings, capabilities=caps).run(
        JobSpec(input=src, output=out, model="realesr-animevideov3", target=TargetSpec.for_height(720),
                overwrite=True)
    ).report
    assert report["sync"]["ok"]
    assert (probe(out, tools).require_video().height) == 720


@pytest.mark.gpu
def test_cuda_stream_overlap_is_pixel_identical(
    caps: Capabilities, settings: Settings, tiny_registry: Any, clip: ClipFactory, tmp_path: Path, ffhelpers: Any
) -> None:
    src = clip("nn-streams", duration=1.0, subtitles=True)
    common: dict[str, Any] = {"input": src, "model": "tiny-srvgg-x2", "target": TargetSpec.for_height(180),
                              "encoder": "libx264", "codec": "h264", "quality": 0, "batch_size": 3}
    outputs = {}
    for streams in (False, True):
        settings.pipeline.cuda_streams = streams
        out = tmp_path / f"streams-{streams}.mkv"
        report = _run(settings, caps, tiny_registry, output=out, **common)
        names = [s["name"] for s in report["throughput"]["stages"]]
        assert ("upload" in names) is streams
        assert report["sync"]["ok"]
        outputs[streams] = ffhelpers.frame_md5s(out)
    assert outputs[True] == outputs[False]
