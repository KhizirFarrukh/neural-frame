"""Packaged (frozen) builds find the FFmpeg and model weights shipped next to the executable."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from neuralframe import bundle
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io import ffmpeg as ffmpeg_mod
from neuralframe.models.download import ensure_weights
from neuralframe.models.registry import ModelSpec, WeightsSpec


@pytest.fixture
def frozen_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    app = tmp_path / "NeuralFrame"
    app.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(app / "NeuralFrame.exe"))
    return app


def test_not_frozen_means_no_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    assert bundle.app_dir() is None and bundle.bundled("ffmpeg") is None


def test_bundled_ffmpeg_wins_over_path(frozen_app: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    shipped = frozen_app / "ffmpeg" / exe
    shipped.parent.mkdir()
    shipped.write_bytes(b"")
    monkeypatch.delenv("NEURALFRAME_FFMPEG", raising=False)
    assert ffmpeg_mod._resolve(None, "NEURALFRAME_FFMPEG", "ffmpeg", None) == str(shipped)
    # An explicit setting still overrides the bundle.
    other = frozen_app / "custom-ffmpeg.exe"
    other.write_bytes(b"")
    assert ffmpeg_mod._resolve(str(other), "NEURALFRAME_FFMPEG", "ffmpeg", None) == str(other)


def test_bundled_weights_are_used_offline(frozen_app: Path, tmp_path: Path) -> None:
    data = b"pretend weights"
    shipped = frozen_app / "models" / "toy" / "toy.pth"
    shipped.parent.mkdir(parents=True)
    shipped.write_bytes(data)
    spec = ModelSpec(id="toy", kind="sr", scale=2, architecture="srvgg", licence="test",
                     weights=WeightsSpec(url="https://invalid.example/toy.pth",
                                         sha256=hashlib.sha256(data).hexdigest(), filename="toy.pth"))
    assert ensure_weights(spec, tmp_path / "cache-models") == shipped  # no download attempted


def test_tampered_bundled_weights_are_not_trusted(frozen_app: Path, tmp_path: Path) -> None:
    shipped = frozen_app / "models" / "toy" / "toy.pth"
    shipped.parent.mkdir(parents=True)
    shipped.write_bytes(b"tampered")
    spec = ModelSpec(id="toy", kind="sr", scale=2, architecture="srvgg", licence="test",
                     weights=WeightsSpec(url="https://invalid.example/toy.pth", sha256="0" * 64, filename="toy.pth"))
    with pytest.raises(NeuralFrameError) as err:  # falls through to a download, which fails (invalid URL)
        ensure_weights(spec, tmp_path / "cache-models", timeout_s=2)
    assert err.value.code is ErrorCode.DOWNLOAD_FAILED
