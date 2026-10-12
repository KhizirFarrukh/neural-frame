"""Locating and running the FFmpeg / ffprobe binaries."""

from __future__ import annotations

import collections
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from neuralframe.errors import ErrorCode, NeuralFrameError

_INSTALL_HINT = (
    "install FFmpeg (Windows: `winget install Gyan.FFmpeg`, Ubuntu: `apt install ffmpeg`) or set "
    "ffmpeg.ffmpeg / ffmpeg.ffprobe in config.yaml (env NEURALFRAME_FFMPEG__FFMPEG)"
)


@dataclass(frozen=True)
class FFmpegTools:
    ffmpeg: str
    ffprobe: str
    version: str
    major: int

    def describe(self) -> str:
        return f"FFmpeg {self.version} ({self.ffmpeg})"


def popen_kwargs() -> dict[str, Any]:
    """Platform flags for child processes (no console window flashes when running under the GUI)."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}


def _resolve(explicit: str | None, env_var: str, name: str, sibling_of: str | None) -> str | None:
    for candidate in (explicit, os.environ.get(env_var)):
        if candidate:
            found = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
            if found:
                return found
    if sibling_of is not None:
        exe = name + (".exe" if sys.platform == "win32" else "")
        sibling = Path(sibling_of).with_name(exe)
        if sibling.is_file():
            return str(sibling)
    return shutil.which(name)


@functools.lru_cache(maxsize=8)
def locate_ffmpeg(ffmpeg: str | None = None, ffprobe: str | None = None) -> FFmpegTools:
    ffmpeg_path = _resolve(ffmpeg, "NEURALFRAME_FFMPEG", "ffmpeg", None)
    if ffmpeg_path is None:
        raise NeuralFrameError(ErrorCode.TOOL_MISSING, "ffmpeg executable not found", hint=_INSTALL_HINT)
    ffprobe_path = _resolve(ffprobe, "NEURALFRAME_FFPROBE", "ffprobe", ffmpeg_path)
    if ffprobe_path is None:
        raise NeuralFrameError(ErrorCode.TOOL_MISSING, "ffprobe executable not found", hint=_INSTALL_HINT)
    try:
        out = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-version"],
            capture_output=True, check=True, timeout=30, **popen_kwargs(),
        ).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError) as exc:
        raise NeuralFrameError(ErrorCode.TOOL_MISSING, f"cannot run {ffmpeg_path}: {exc}") from exc
    match = re.search(r"ffmpeg version (\S+)", out)
    version = match.group(1) if match else "unknown"
    major_match = re.match(r"n?(\d+)", version)
    major = int(major_match.group(1)) if major_match else 0
    if 0 < major < 6:
        raise NeuralFrameError(
            ErrorCode.TOOL_MISSING,
            f"FFmpeg {version} is too old; NeuralFrame needs FFmpeg 6.0 or newer",
            hint=_INSTALL_HINT,
        )
    return FFmpegTools(ffmpeg=ffmpeg_path, ffprobe=ffprobe_path, version=version, major=major)


def run_tool(
    args: Sequence[str],
    *,
    error_code: ErrorCode,
    what: str,
    timeout: float | None = None,
    check: bool = True,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Run a short-lived ffmpeg/ffprobe command and turn failures into NeuralFrameError."""
    try:
        proc = subprocess.run(
            list(args), input=stdin, capture_output=True, timeout=timeout, **popen_kwargs()
        )
    except subprocess.TimeoutExpired as exc:
        raise NeuralFrameError(error_code, f"{what} timed out after {timeout}s") from exc
    except OSError as exc:
        raise NeuralFrameError(error_code, f"{what} could not start: {exc}") from exc
    if check and proc.returncode != 0:
        raise NeuralFrameError(
            error_code,
            f"{what} failed (exit {proc.returncode}): {tail(proc.stderr)}",
            details={"command": list(args)},
        )
    return proc


def run_ffprobe_json(tools: FFmpegTools, args: Sequence[str], *, what: str = "ffprobe") -> dict[str, Any]:
    proc = run_tool(
        [tools.ffprobe, "-v", "error", "-print_format", "json", *args],
        error_code=ErrorCode.PROBE_FAILED,
        what=what,
        timeout=600,
    )
    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:
        raise NeuralFrameError(ErrorCode.PROBE_FAILED, f"{what} returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise NeuralFrameError(ErrorCode.PROBE_FAILED, f"{what} returned unexpected output")
    return data


def tail(stderr: bytes | str, lines: int = 12) -> str:
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else stderr
    kept = [line for line in text.strip().splitlines() if line.strip()]
    return " | ".join(kept[-lines:]) if kept else "(no stderr output)"


class StderrCollector:
    """Drains a child's stderr on a daemon thread (prevents pipe deadlock) and keeps the last lines."""

    def __init__(self, stream: IO[bytes] | None, max_lines: int = 200) -> None:
        self._lines: collections.deque[str] = collections.deque(maxlen=max_lines)
        self._thread: threading.Thread | None = None
        if stream is not None:
            self._thread = threading.Thread(target=self._pump, args=(stream,), daemon=True, name="ffmpeg-stderr")
            self._thread.start()

    def _pump(self, stream: IO[bytes]) -> None:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._lines.append(line)
        stream.close()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def text(self, lines: int = 12) -> str:
        return tail("\n".join(self._lines), lines)

    def lines(self) -> list[str]:
        return list(self._lines)
