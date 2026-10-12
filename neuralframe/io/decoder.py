"""Frame decoder: an FFmpeg subprocess emitting raw frames in a NUT stream, demuxed with PyAV.

Why NUT instead of ``-f rawvideo``: NUT carries each frame's PTS in-band, so variable-frame-rate input
keeps its exact timestamps without a separate side channel. PyAV only *demuxes* (rawvideo packets are the
frames), so there is no decode cost on our side and FFmpeg does all codec work in its own process.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, NoReturn, cast

import av
import numpy as np

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, StderrCollector, popen_kwargs
from neuralframe.io.rawvideo import RawArray, RawVideoFormat
from neuralframe.log import get_logger

log = get_logger(component="decoder")

_PIPE_BUFFER = 1 << 20


class PipeReader:
    """Non-seekable read-only wrapper: PyAV must not try to seek a pipe."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream

    def read(self, n: int) -> bytes:
        return self._stream.read(n)


@dataclass(frozen=True)
class DecodePlan:
    """How to decode: hwaccel, filter chain (deinterlace/denoise), and the raw format handed to us."""

    fmt: RawVideoFormat
    hwaccel: str | None = None
    filters: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        parts = [self.fmt.pix_fmt, f"{self.fmt.width}x{self.fmt.height}", f"hwaccel={self.hwaccel or 'none'}"]
        if self.filters:
            parts.append("filters=" + ",".join(self.filters))
        return " ".join(parts)


@dataclass
class DecodedFrame:
    index: int
    pts: int
    duration: int
    data: RawArray


class FrameDecoder:
    """Iterate raw frames of one video stream. Use as a context manager."""

    def __init__(
        self,
        tools: FFmpegTools,
        path: Path,
        stream_index: int,
        plan: DecodePlan,
        *,
        seek_seconds: float | None = None,
        first_index: int = 0,
        loglevel: str = "error",
    ) -> None:
        self.tools = tools
        self.path = path
        self.stream_index = stream_index
        self.plan = plan
        self.seek_seconds = seek_seconds
        self.first_index = first_index
        self.loglevel = loglevel
        self.time_base: Fraction = Fraction(1, 1000)
        self.frame_rate: Fraction = Fraction(0)
        self.frames_read = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr: StderrCollector | None = None
        self._container: Any = None
        self._stream: Any = None

    # ---- lifecycle -----------------------------------------------------------------------------------

    def command(self) -> list[str]:
        cmd = [self.tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", self.loglevel]
        if self.plan.hwaccel:
            cmd += ["-hwaccel", self.plan.hwaccel]
            if self.plan.hwaccel == "vaapi":
                cmd += ["-hwaccel_device", "/dev/dri/renderD128"]
        if self.seek_seconds is not None and self.seek_seconds > 0:
            # Keyframe seek only; exact positioning is done by the caller on PTS (see resume logic).
            cmd += ["-ss", f"{self.seek_seconds:.6f}", "-noaccurate_seek"]
        cmd += ["-copyts", "-i", str(self.path), "-map", f"0:{self.stream_index}", "-an", "-sn", "-dn"]
        if self.plan.filters:
            cmd += ["-vf", ",".join(self.plan.filters)]
        cmd += [
            "-fps_mode", "passthrough",
            "-c:v", "rawvideo", "-pix_fmt", self.plan.fmt.pix_fmt,
            "-f", "nut", "-write_index", "0", "pipe:1",
        ]
        return cmd

    def start(self) -> FrameDecoder:
        cmd = self.command()
        log.debug("decoder start", cmd=" ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=_PIPE_BUFFER, **popen_kwargs()
            )
        except OSError as exc:
            raise NeuralFrameError(ErrorCode.DECODE_FAILED, f"cannot start ffmpeg decoder: {exc}") from exc
        self._stderr = StderrCollector(self._proc.stderr)
        assert self._proc.stdout is not None
        try:
            self._container = av.open(PipeReader(self._proc.stdout), mode="r", format="nut")
            self._stream = self._container.streams.video[0]
        except Exception as exc:
            self._fail(f"cannot read decoder output: {exc}")
        ctx = self._stream.codec_context
        width, height = int(ctx.width), int(ctx.height)
        expected = self.plan.fmt
        if (width, height) != (expected.width, expected.height):
            self._fail(
                f"decoder produced {width}x{height}, planned {expected.width}x{expected.height}"
            )
        self.time_base = Fraction(self._stream.time_base)
        rate = self._stream.average_rate or self._stream.guessed_rate
        self.frame_rate = Fraction(rate) if rate else Fraction(0)
        return self

    def __enter__(self) -> FrameDecoder:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
        proc = self._proc
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            if proc.stdout is not None:
                proc.stdout.close()
        if self._stderr is not None:
            self._stderr.join()

    # ---- reading -------------------------------------------------------------------------------------

    def _fail(self, message: str) -> NoReturn:
        stderr = ""
        if self._proc is not None:
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr is not None:
            self._stderr.join(1.0)
            stderr = self._stderr.text()
        self.close()
        raise NeuralFrameError(
            ErrorCode.DECODE_FAILED,
            f"{message}; ffmpeg: {stderr}" if stderr else message,
            details={"input": str(self.path), "hwaccel": self.plan.hwaccel},
        )

    def __iter__(self) -> Iterator[DecodedFrame]:
        return self.frames()

    def frames(self) -> Iterator[DecodedFrame]:
        if self._container is None:
            self.start()
        fmt = self.plan.fmt
        dtype = np.dtype(fmt.dtype).newbyteorder("<")
        expected = fmt.frame_samples
        nominal = 0
        if self.frame_rate > 0:
            nominal = max(1, round(1 / (self.frame_rate * self.time_base)))
        try:
            for packet in self._container.demux(self._stream):
                if packet.size == 0 or packet.pts is None:
                    continue
                arr = np.frombuffer(memoryview(packet), dtype=dtype)
                if arr.size != expected:
                    self._fail(f"raw frame has {arr.size} samples, expected {expected}")
                data = cast(RawArray, arr.astype(fmt.dtype, copy=True))
                duration = int(packet.duration or nominal)
                frame = DecodedFrame(
                    index=self.first_index + self.frames_read, pts=int(packet.pts), duration=duration, data=data
                )
                self.frames_read += 1
                yield frame
        except NeuralFrameError:
            raise
        except Exception as exc:
            self._fail(f"decoder stream error after {self.frames_read} frames: {exc}")
        proc = self._proc
        assert proc is not None
        rc = proc.wait()
        if rc != 0:
            self._fail(f"ffmpeg decoder exited with code {rc} after {self.frames_read} frames")


class ResilientDecoder:
    """Hardware decoding with transparent software fallback.

    If the hardware path fails (unsupported profile, driver hiccup, device lost) — even mid-stream — a
    software decoder is started, seeked to a keyframe shortly before the last delivered frame, and frames
    are skipped by exact PTS, so downstream stages see one continuous, gap-free sequence.
    """

    def __init__(
        self,
        tools: FFmpegTools,
        path: Path,
        stream_index: int,
        plan: DecodePlan,
        *,
        loglevel: str = "error",
        seek_margin_s: float = 5.0,
        resume_after: Fraction | None = None,
        first_index: int = 0,
    ) -> None:
        """``resume_after``: skip every frame at or before this time (seconds), seeking close to it first;
        delivered frames are numbered from ``first_index``."""
        self.tools = tools
        self.path = path
        self.stream_index = stream_index
        self.plan = plan
        self.loglevel = loglevel
        self.seek_margin_s = seek_margin_s
        self.resume_after = resume_after
        self.first_index = first_index
        self.fallbacks: list[str] = []
        seek = None if resume_after is None else max(0.0, float(resume_after) - seek_margin_s)
        self._current = FrameDecoder(tools, path, stream_index, plan, seek_seconds=seek, loglevel=loglevel)

    @property
    def time_base(self) -> Fraction:
        return self._current.time_base

    @property
    def frame_rate(self) -> Fraction:
        return self._current.frame_rate

    def start(self) -> ResilientDecoder:
        try:
            self._current.start()
        except NeuralFrameError as exc:
            if self.plan.hwaccel is None:
                raise
            seek = None if self.resume_after is None else max(0.0, float(self.resume_after) - self.seek_margin_s)
            self._switch_to_software(str(exc), seek_seconds=seek)
            self._current.start()
        return self

    def _switch_to_software(self, reason: str, seek_seconds: float | None) -> None:
        log.warning("hardware decoding failed; continuing in software", hwaccel=self.plan.hwaccel, error=reason)
        self.fallbacks.append(f"hardware decode ({self.plan.hwaccel}) failed: {reason[:160]}; using software")
        self._current.close()
        self.plan = DecodePlan(fmt=self.plan.fmt, hwaccel=None, filters=self.plan.filters, notes=self.plan.notes)
        self._current = FrameDecoder(
            self.tools, self.path, self.stream_index, self.plan, seek_seconds=seek_seconds, loglevel=self.loglevel
        )

    def frames(self) -> Iterator[DecodedFrame]:
        delivered = self.first_index
        last_time: Fraction | None = self.resume_after
        time_base = self._current.time_base
        while True:
            try:
                for frame in self._current.frames():
                    t = frame.pts * self._current.time_base
                    if last_time is not None and t <= last_time:
                        continue  # already delivered before the fallback
                    last_time = t
                    pts = frame.pts
                    if self._current.time_base != time_base:
                        pts = round(t / time_base)
                    yield DecodedFrame(index=delivered, pts=pts, duration=frame.duration, data=frame.data)
                    delivered += 1
                return
            except NeuralFrameError as exc:
                if self.plan.hwaccel is None:
                    raise
                seek = None if last_time is None else max(0.0, float(last_time) - self.seek_margin_s)
                self._switch_to_software(str(exc), seek_seconds=seek)
                self._current.start()

    def close(self) -> None:
        self._current.close()
