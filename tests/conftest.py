from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from neuralframe.config import Settings, load_settings
from neuralframe.io.ffmpeg import FFmpegTools, locate_ffmpeg, popen_kwargs
from neuralframe.io.hwcaps import Capabilities, detect_capabilities
from neuralframe.log import configure_logging
from neuralframe.testing.media import ClipSpec, make_clip


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _quiet_logs() -> None:
    configure_logging("WARNING")


@pytest.fixture(scope="session")
def tools() -> FFmpegTools:
    return locate_ffmpeg()


@pytest.fixture(scope="session")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("nf-cache")


@pytest.fixture(scope="session")
def caps(tools: FFmpegTools, cache_dir: Path) -> Capabilities:
    return detect_capabilities(tools, cache_dir=cache_dir)


@pytest.fixture
def settings(tmp_path: Path, cache_dir: Path) -> Settings:
    s = load_settings(None)
    s.paths.cache_dir = cache_dir
    s.paths.work_dir = tmp_path / "work"
    return s


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("media")


ClipFactory = Callable[..., Path]


@pytest.fixture(scope="session")
def clip(tools: FFmpegTools, media_dir: Path) -> ClipFactory:
    """``clip(name, **ClipSpec fields)`` → path of a generated (and cached) synthetic clip."""

    def factory(name: str, **fields: object) -> Path:
        spec = ClipSpec(**fields)  # type: ignore[arg-type]
        digest = hashlib.sha1(repr(spec).encode()).hexdigest()[:8]
        ext = "mkv" if spec.container == "mkv" else spec.container
        return make_clip(tools, media_dir / f"{name}-{digest}.{ext}", spec)

    return factory


# ---- helpers used by integration tests ---------------------------------------------------------------


def run_ffmpeg_text(tools: FFmpegTools, args: list[str]) -> str:
    proc = subprocess.run(
        [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *args],
        capture_output=True, timeout=300, check=True, **popen_kwargs(),
    )
    return proc.stdout.decode("utf-8", "replace")


def packet_md5s(tools: FFmpegTools, path: Path, selector: str) -> list[str]:
    """Per-packet MD5 of a stream copied out of ``path`` (bit-exactness checks)."""
    text = run_ffmpeg_text(tools, ["-i", str(path), "-map", f"0:{selector}", "-c", "copy", "-f", "framemd5", "-"])
    return [line.rsplit(",", 1)[-1].strip() for line in text.splitlines() if line and not line.startswith("#")]


def frame_md5s(tools: FFmpegTools, path: Path) -> list[str]:
    text = run_ffmpeg_text(tools, ["-i", str(path), "-map", "0:v:0", "-f", "framemd5", "-"])
    return [line.rsplit(",", 1)[-1].strip() for line in text.splitlines() if line and not line.startswith("#")]


def frame_times(tools: FFmpegTools, path: Path) -> list[float]:
    proc = subprocess.run(
        [tools.ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", str(path)],
        capture_output=True, timeout=300, check=True, **popen_kwargs(),
    )
    return [float(x.strip().rstrip(",")) for x in proc.stdout.decode().split() if x.strip()]


@pytest.fixture
def ffhelpers(tools: FFmpegTools) -> Iterator[object]:
    class Helpers:
        packet_md5s = staticmethod(lambda p, s: packet_md5s(tools, p, s))
        frame_md5s = staticmethod(lambda p: frame_md5s(tools, p))
        frame_times = staticmethod(lambda p: frame_times(tools, p))
        text = staticmethod(lambda args: run_ffmpeg_text(tools, args))

    yield Helpers()
