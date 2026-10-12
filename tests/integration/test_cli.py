"""The ``neuralframe`` command line, driven through Typer's test runner."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from neuralframe import __version__
from neuralframe.cli.app import app
from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.io.probe import probe
from tests.conftest import ClipFactory


@pytest.fixture
def config(tmp_path: Path, cache_dir: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"paths:\n  work_dir: '{(tmp_path / 'work').as_posix()}'\n  cache_dir: '{cache_dir.as_posix()}'\n"
        "inference:\n  device: cpu\n",
        encoding="utf-8",
    )
    return path


def invoke(*args: object) -> Result:
    return CliRunner().invoke(app, [str(a) for a in args])


def test_version() -> None:
    result = invoke("version")
    assert result.exit_code == 0 and __version__ in result.stdout


def test_probe_json(config: Path, clip: ClipFactory) -> None:
    src = clip("cli-probe", width=96, height=64, fps="24", duration=0.5)
    result = invoke("-c", config, "probe", src, "--json", "--frames")
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["timeline"]["frames"] == 12


def test_upscale_classical(tools: FFmpegTools, config: Path, clip: ClipFactory, tmp_path: Path) -> None:
    src = clip("cli-up", width=96, height=64, duration=0.5)
    out = tmp_path / "up.mkv"
    result = invoke("-c", config, "upscale", src, "-o", out, "-m", "lanczos", "--scale", "2",
                    "--encoder", "libx264", "--codec", "h264")
    assert result.exit_code == 0, result.output
    assert "done" in result.output
    video = probe(out, tools).require_video()
    assert (video.width, video.height) == (192, 128)


def test_interpolate_command(tools: FFmpegTools, config: Path, clip: ClipFactory, tmp_path: Path) -> None:
    src = clip("cli-fg", width=96, height=64, fps="25", duration=1.0)
    out = tmp_path / "fg.mkv"
    result = invoke("-c", config, "interpolate", src, "-o", out, "--fps", "50", "--engine", "minterpolate",
                    "--encoder", "libx264", "--codec", "h264")
    assert result.exit_code == 0, result.output
    assert "frame generation: 50/1 fps" in result.output
    video = probe(out, tools).require_video()
    assert video.fps == Fraction(50) and (video.width, video.height) == (96, 64)


def test_no_upscale_needs_frame_generation(config: Path, clip: ClipFactory, tmp_path: Path) -> None:
    src = clip("cli-up", width=96, height=64, duration=0.5)
    result = invoke("-c", config, "upscale", src, "-o", tmp_path / "x.mkv", "--no-upscale")
    assert result.exit_code != 0
    assert "--no-upscale" in result.output


def test_conflicting_targets_are_rejected(config: Path, clip: ClipFactory, tmp_path: Path) -> None:
    src = clip("cli-up", width=96, height=64, duration=0.5)
    result = invoke("-c", config, "upscale", src, "-o", tmp_path / "x.mkv", "--scale", "2", "--target", "4k")
    assert result.exit_code != 0
