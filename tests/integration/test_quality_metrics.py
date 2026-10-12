"""Video-level metrics against FFmpeg-generated clips."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.quality.metrics import compare_videos, frame_differences, iter_luma, psnr, vmaf
from tests.conftest import ClipFactory


def test_identical_and_degraded_clips(tools: FFmpegTools, clip: ClipFactory, tmp_path: Path, ffhelpers: Any) -> None:
    ref = clip("q-ref", width=128, height=96, fps="25", duration=1.0, crf=0, audio=False)
    same = compare_videos(tools, ref, ref)
    assert same.frames == 25 and same.psnr_y == 100.0 and same.ssim_y == pytest.approx(1.0)

    blurred = tmp_path / "blur.mkv"
    ffhelpers.text(["-i", str(ref), "-vf", "gblur=sigma=1.5", "-c:v", "libx264", "-qp", "0", "-y", str(blurred)])
    worse = compare_videos(tools, blurred, ref)
    assert 15 < worse.psnr_y < 40 and worse.ssim_y < 0.98

    # Baseline path: downscale, then let the metric scale back up with bicubic.
    small = tmp_path / "small.mkv"
    ffhelpers.text(["-i", str(ref), "-vf", "scale=64:48:flags=area", "-c:v", "libx264", "-qp", "0", "-y", str(small)])
    baseline = compare_videos(tools, small, ref, distorted_size=(128, 96))
    assert baseline.frames == 25 and baseline.psnr_y < same.psnr_y

    v_same = vmaf(tools, ref, ref)
    v_blur = vmaf(tools, blurred, ref)
    assert v_same > 95 and v_blur < v_same


def test_luma_is_exact(tools: FFmpegTools, clip: ClipFactory) -> None:
    ref = clip("q-ref", width=128, height=96, fps="25", duration=1.0, crf=0, audio=False)
    frames = list(iter_luma(tools, ref))
    assert len(frames) == 25 and frames[0].shape == (96, 128)
    assert psnr(frames[3], frames[3]) == 100.0
    diffs = frame_differences(iter(frames))
    assert diffs[0] == 0.0 and all(d > 0 for d in diffs[1:])  # testsrc2 moves every frame
