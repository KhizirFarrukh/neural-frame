"""Temporal model jobs end to end: scene-aware chunking, cross-fades, kill-and-resume through blend zones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neuralframe.config import Settings, default_cache_dir  # noqa: E402
from neuralframe.core.job import JobRunner, JobSpec  # noqa: E402
from neuralframe.core.planning import TargetSpec  # noqa: E402
from neuralframe.errors import JobCancelled  # noqa: E402
from neuralframe.io.ffmpeg import FFmpegTools  # noqa: E402
from neuralframe.io.hwcaps import Capabilities  # noqa: E402
from neuralframe.io.probe import probe  # noqa: E402
from tests.conftest import ClipFactory  # noqa: E402
from tests.models import registry_with, tiny_temporal_model  # noqa: E402


@pytest.fixture
def temporal_registry(settings: Settings) -> Any:
    settings.inference.device = "cpu"
    settings.chunking.temporal_chunk_frames = 16
    spec, _ = tiny_temporal_model(settings.paths.models())
    return registry_with(spec)


def test_temporal_job_respects_scene_cuts(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, temporal_registry: Any, clip: ClipFactory,
    tmp_path: Path,
) -> None:
    src = clip("tmp-cut", duration=2.0, scene_cut_at=1.0)
    out = tmp_path / "t.mkv"
    report = JobRunner(settings, capabilities=caps, registry=temporal_registry).run(
        JobSpec(input=src, output=out, model="tiny-realbasicvsr", target=TargetSpec.factor(2), overwrite=True)
    ).report
    assert report["sync"]["ok"]
    assert report["output"]["frames"] == 60
    plan = report["plan"]
    assert plan["chunks"]["scene_aligned_boundaries"] == 1  # the cut at frame 30 is a chunk boundary
    assert any("scene cuts at frames [30]" in n for n in plan["notes"])
    assert report["processor"]["temporal"]["blended_frames"] > 0
    assert probe(out, tools).require_video().height == 180


def test_temporal_kill_and_resume_is_identical(
    caps: Capabilities, settings: Settings, temporal_registry: Any, clip: ClipFactory, tmp_path: Path,
    ffhelpers: Any,
) -> None:
    src = clip("tmp-resume", duration=2.0)
    common: dict[str, Any] = {"input": src, "model": "tiny-realbasicvsr", "target": TargetSpec.factor(2),
                              "encoder": "libx264", "codec": "h264", "quality": 0}
    reference = tmp_path / "ref.mkv"
    JobRunner(settings, capabilities=caps, registry=temporal_registry).run(
        JobSpec(output=reference, overwrite=True, **common))

    runner = JobRunner(settings, capabilities=caps, registry=temporal_registry)

    def kill(chunk_id: int, _total: int) -> None:
        if chunk_id == 1:
            runner.cancel.cancel("simulated kill")

    runner.on_chunk_done = kill
    out = tmp_path / "resumed.mkv"
    with pytest.raises(JobCancelled):
        runner.run(JobSpec(output=out, overwrite=True, **common))
    report = JobRunner(settings, capabilities=caps, registry=temporal_registry).run(
        JobSpec(output=out, resume=True, overwrite=True, **common)).report
    assert report["throughput"]["frames_resumed"] == 32
    assert ffhelpers.frame_md5s(out) == ffhelpers.frame_md5s(reference)


@pytest.mark.gpu
@pytest.mark.network
def test_real_basicvsr_on_cuda(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path
) -> None:
    settings.paths.models_dir = default_cache_dir() / "models"
    settings.chunking.temporal_chunk_frames = 24
    src = clip("tmp-real", width=214, height=120, duration=2.0, scene_cut_at=1.0)
    out = tmp_path / "rbvsr.mkv"
    report = JobRunner(settings, capabilities=caps).run(
        JobSpec(input=src, output=out, model="realbasicvsr-x4", target=TargetSpec.for_height(480), overwrite=True)
    ).report
    assert report["sync"]["ok"]
    assert report["plan"]["inference"]["backend"] == "torch"
    assert probe(out, tools).require_video().height == 480
