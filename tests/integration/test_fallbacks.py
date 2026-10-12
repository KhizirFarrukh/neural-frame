"""Hardware I/O fallbacks: broken encoder → next encoder; hardware decode failure → software, gap-free."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neuralframe.config import Settings
from neuralframe.core.job import JobRunner, JobSpec
from neuralframe.core.planning import TargetSpec
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io import decoder as decoder_mod
from neuralframe.io.decoder import DecodePlan, ResilientDecoder
from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.io.hwcaps import Capabilities, CodecCheck
from neuralframe.io.probe import probe
from neuralframe.io.rawvideo import RawVideoFormat
from tests.conftest import ClipFactory


def test_encoder_preflight_falls_back(
    caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path
) -> None:
    # A "hardware" encoder that passed the capability probe but cannot encode this job (simulated with a
    # name FFmpeg does not know), ranked first by putting it in the most preferred family.
    broken = CodecCheck(name="hevc_brokenhw", codec="hevc", family="nvenc", listed=True, functional=True)
    fake_caps = caps.model_copy(update={"encoders": [broken, *caps.encoders]})
    settings.inference.device = "cpu"
    src = clip("fb-enc", duration=0.5)
    report = JobRunner(settings, capabilities=fake_caps).run(
        JobSpec(input=src, output=tmp_path / "o.mkv", model="lanczos", target=TargetSpec.factor(2), overwrite=True)
    ).report
    assert report["plan"]["encoder"]["name"] != "hevc_brokenhw"
    assert any("hevc_brokenhw failed preflight" in w for w in report["warnings"])
    assert report["sync"]["ok"]


def test_explicit_broken_encoder_is_an_error(
    caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path
) -> None:
    broken = CodecCheck(name="hevc_brokenhw", codec="hevc", family="nvenc", listed=True, functional=True)
    fake_caps = caps.model_copy(update={"encoders": [broken, *caps.encoders]})
    with pytest.raises(NeuralFrameError) as err:
        JobRunner(settings, capabilities=fake_caps).run(
            JobSpec(input=clip("fb-enc", duration=0.5), output=tmp_path / "o.mkv", model="lanczos",
                    target=TargetSpec.factor(2), encoder="hevc_brokenhw", overwrite=True)
        )
    assert err.value.code is ErrorCode.ENCODE_FAILED


def test_decoder_falls_back_mid_stream_without_gaps(
    tools: FFmpegTools, clip: ClipFactory, monkeypatch: Any
) -> None:
    src = clip("fb-dec", duration=2.0, gop=12)
    v = probe(src, tools).require_video()
    fmt = RawVideoFormat.for_source(v.pixel_format, v.width, v.height)

    reference = [(f.pts, f.data.tobytes()) for f in decoder_mod.FrameDecoder(tools, src, v.index, DecodePlan(fmt))]

    original = decoder_mod.FrameDecoder.frames

    def flaky(self: decoder_mod.FrameDecoder) -> Any:
        for i, frame in enumerate(original(self)):
            if self.plan.hwaccel is not None and i == 37:
                raise NeuralFrameError(ErrorCode.DECODE_FAILED, "simulated device loss")
            yield frame

    original_command = decoder_mod.FrameDecoder.command

    def software_command(self: decoder_mod.FrameDecoder) -> list[str]:
        # Keep the plan "hardware" (that is what the fallback logic keys on) but run FFmpeg without device
        # flags, so the injected mid-stream failure is the only failure, on any machine or CI runner.
        cmd = original_command(self)
        if "-hwaccel" in cmd:
            i = cmd.index("-hwaccel")
            del cmd[i : i + 2]
        return cmd

    monkeypatch.setattr(decoder_mod.FrameDecoder, "frames", flaky)
    monkeypatch.setattr(decoder_mod.FrameDecoder, "command", software_command)
    plan = DecodePlan(fmt, hwaccel="cuda")
    dec = ResilientDecoder(tools, src, v.index, plan).start()
    got = [(f.index, f.pts, f.data.tobytes()) for f in dec.frames()]
    dec.close()
    assert [g[0] for g in got] == list(range(len(reference)))
    assert [(g[1], g[2]) for g in got] == reference
    assert dec.fallbacks and "simulated device loss" in dec.fallbacks[0]
