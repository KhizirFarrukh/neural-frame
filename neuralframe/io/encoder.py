"""Encoder planning (per-encoder argument mapping) and the raw-frame → FFmpeg segment encoder."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import IO, Any, NoReturn

import av
import numpy as np

from neuralframe.colorspace import ColorPlan
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, StderrCollector, popen_kwargs
from neuralframe.io.rawvideo import RawArray, RawVideoFormat
from neuralframe.log import get_logger

log = get_logger(component="encoder")

_SPEED: dict[str, dict[str, str]] = {
    "nvenc": {"slow": "p7", "medium": "p5", "fast": "p3"},
    "libx264": {"slow": "slow", "medium": "medium", "fast": "veryfast"},
    "libx265": {"slow": "slow", "medium": "medium", "fast": "fast"},
    "libsvtav1": {"slow": "4", "medium": "6", "fast": "9"},
    "libaom-av1": {"slow": "3", "medium": "5", "fast": "7"},
    "amf": {"slow": "quality", "medium": "balanced", "fast": "speed"},
    "qsv": {"slow": "veryslow", "medium": "medium", "fast": "veryfast"},
}

_DEFAULT_QUALITY = {"hevc": 21, "h264": 19, "av1": 28}


class PipeWriter:
    """Write-only, non-seekable wrapper so PyAV streams NUT into a pipe."""

    def __init__(self, stream: IO[bytes]) -> None:
        self._stream = stream

    def write(self, data: bytes) -> int:
        return self._stream.write(data)


@dataclass(frozen=True)
class EncoderPlan:
    name: str
    codec: str
    family: str
    quality: int
    speed: str
    ten_bit: bool
    color: ColorPlan
    extra_args: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def raw_format_depth(self) -> int:
        return 10 if self.ten_bit else 8

    def describe(self) -> str:
        bits = "10-bit" if self.ten_bit else "8-bit"
        return f"{self.name} ({self.codec}, {bits}, q={self.quality}, speed={self.speed})"

    def input_args(self) -> list[str]:
        if self.family == "vaapi":
            return ["-vaapi_device", "/dev/dri/renderD128"]
        return []

    def filters(self) -> list[str]:
        """Encoder-side filter chain.

        Our samples are already in the final colour space, so FFmpeg must never convert them. Frames are
        tagged with ``setparams`` *and* the encoder gets the same values as output options
        (:meth:`color_args`): when the two disagree, FFmpeg's format negotiation silently inserts a
        swscale pass (measured: full-range FFV1 got squeezed to limited range; untagged frames with
        tagged output options changed x264 output between runs).
        """
        tags = self.color.ffmpeg_tags()
        chain = [
            "setparams="
            f"range={tags['color_range']}:color_primaries={tags['color_primaries']}"
            f":color_trc={tags['color_trc']}:colorspace={tags['colorspace']}"
            f":chroma_location={tags['chroma_sample_location']}"
        ]
        if self.family == "vaapi":
            chain += [f"format={'p010' if self.ten_bit else 'nv12'}", "hwupload"]
        return chain

    def color_args(self) -> list[str]:
        tags = self.color.ffmpeg_tags()
        return [
            "-color_range", tags["color_range"],
            "-colorspace", tags["colorspace"],
            "-color_primaries", tags["color_primaries"],
            "-color_trc", tags["color_trc"],
            "-chroma_sample_location", tags["chroma_sample_location"],
        ]

    def codec_args(self) -> list[str]:
        name, codec, q = self.name, self.codec, str(self.quality)
        hw_fmt = "p010le" if self.ten_bit else "nv12"
        sw_fmt = "yuv420p10le" if self.ten_bit else "yuv420p"
        args: list[str]
        if self.family == "nvenc":
            args = [
                "-c:v", name, "-preset", _SPEED["nvenc"][self.speed], "-tune", "hq",
                "-rc", "vbr", "-cq", q, "-b:v", "0", "-spatial-aq", "1", "-rc-lookahead", "20",
                "-pix_fmt", hw_fmt,
            ]
            if codec in ("h264", "hevc"):
                args += ["-temporal-aq", "1"]
            if codec == "hevc" and self.ten_bit:
                args += ["-profile:v", "main10"]
        elif name == "libx264":
            args = ["-c:v", name, "-preset", _SPEED[name][self.speed], "-crf", q, "-pix_fmt", sw_fmt]
        elif name == "libx265":
            args = [
                "-c:v", name, "-preset", _SPEED[name][self.speed], "-crf", q, "-pix_fmt", sw_fmt,
                "-x265-params", "log-level=error",
            ]
        elif name == "libsvtav1":
            args = ["-c:v", name, "-preset", _SPEED[name][self.speed], "-crf", q, "-pix_fmt", sw_fmt]
        elif name == "libaom-av1":
            args = [
                "-c:v", name, "-cpu-used", _SPEED[name][self.speed], "-crf", q, "-b:v", "0",
                "-row-mt", "1", "-pix_fmt", sw_fmt,
            ]
        elif self.family == "amf":
            args = [
                "-c:v", name, "-quality", _SPEED["amf"][self.speed], "-rc", "cqp",
                "-qp_i", q, "-qp_p", q, "-pix_fmt", hw_fmt,
            ]
            if codec in ("h264", "hevc"):
                args += ["-qp_b", q]
        elif self.family == "qsv":
            args = [
                "-c:v", name, "-preset", _SPEED["qsv"][self.speed], "-global_quality", q, "-pix_fmt", hw_fmt,
            ]
        elif self.family == "vaapi":
            args = ["-c:v", name, "-rc_mode", "CQP", "-qp", q]
        elif self.family == "videotoolbox":
            args = ["-c:v", name, "-q:v", str(max(1, 100 - 2 * self.quality)), "-pix_fmt", hw_fmt]
        else:
            args = ["-c:v", name, "-pix_fmt", sw_fmt]
        return [*args, *self.color_args(), *self.extra_args]

    def container_args(self, container: str) -> list[str]:
        if container in ("mp4", "mov") and self.codec == "hevc":
            return ["-tag:v", "hvc1"]
        return []


def default_quality(codec: str) -> int:
    return _DEFAULT_QUALITY.get(codec, 21)


class SegmentEncoder:
    """Encode raw frames (with explicit PTS) into one file via ``ffmpeg -f nut -i pipe:0``."""

    def __init__(
        self,
        tools: FFmpegTools,
        out_path: Path,
        fmt: RawVideoFormat,
        time_base: Fraction,
        frame_rate: Fraction,
        plan: EncoderPlan,
        *,
        container: str = "mp4",
        loglevel: str = "error",
    ) -> None:
        self.tools = tools
        self.out_path = out_path
        self.fmt = fmt
        self.time_base = time_base
        self.frame_rate = frame_rate if frame_rate > 0 else Fraction(25)
        self.plan = plan
        self.container = container
        self.loglevel = loglevel
        self.frames_written = 0
        self.first_pts: int | None = None
        self.last_pts: int | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr: StderrCollector | None = None
        self._out: Any = None
        self._stream: Any = None

    def command(self) -> list[str]:
        timescale: list[str] = []
        if self.container in ("mp4", "mov") and self.time_base.numerator == 1:
            timescale = ["-video_track_timescale", str(self.time_base.denominator)]
        return [
            self.tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", self.loglevel, "-y",
            *self.plan.input_args(),
            # -copyts keeps our absolute PTS in each segment, so concatenation is a plain packet copy.
            "-copyts", "-f", "nut", "-i", "pipe:0",
            "-map", "0:v:0", "-fps_mode", "passthrough", "-enc_time_base", "demux",
            "-vf", ",".join(self.plan.filters()),
            *self.plan.codec_args(), *self.plan.container_args(self.container), *timescale,
            "-f", self.container, str(self.out_path),
        ]

    def start(self) -> SegmentEncoder:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.command()
        log.debug("encoder start", cmd=" ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1 << 20, **popen_kwargs()
            )
        except OSError as exc:
            raise NeuralFrameError(ErrorCode.ENCODE_FAILED, f"cannot start ffmpeg encoder: {exc}") from exc
        self._stderr = StderrCollector(self._proc.stderr)
        assert self._proc.stdin is not None
        self._out = av.open(PipeWriter(self._proc.stdin), mode="w", format="nut")
        stream = self._out.add_stream("rawvideo", rate=self.frame_rate)
        stream.width = self.fmt.width
        stream.height = self.fmt.height
        stream.pix_fmt = self.fmt.pix_fmt
        stream.codec_context.time_base = self.time_base
        stream.time_base = self.time_base
        self._stream = stream
        return self

    def __enter__(self) -> SegmentEncoder:
        return self.start()

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def write(self, pts: int, duration: int, data: RawArray) -> None:
        if self._out is None:
            self.start()
        if data.size != self.fmt.frame_samples:
            raise NeuralFrameError(
                ErrorCode.ENCODE_FAILED,
                f"frame has {data.size} samples, encoder expects {self.fmt.frame_samples}",
            )
        # PyAV copies any buffer-protocol object into the packet; the stub only names bytes.
        packet = av.Packet(np.ascontiguousarray(data).tobytes())
        packet.pts = pts
        packet.dts = pts
        packet.duration = max(int(duration), 1)
        packet.time_base = self.time_base
        packet.stream = self._stream
        try:
            self._out.mux(packet)
        except (OSError, av.error.FFmpegError) as exc:
            self._fail(f"encoder pipe closed after {self.frames_written} frames: {exc}")
        if self.first_pts is None:
            self.first_pts = pts
        self.last_pts = pts
        self.frames_written += 1

    def close(self) -> None:
        """Flush, wait for FFmpeg and verify it succeeded."""
        if self._out is None:
            return
        try:
            self._out.close()
        except (OSError, av.error.FFmpegError) as exc:
            self._fail(f"cannot finalise encoder input: {exc}")
        finally:
            self._out = None
        proc = self._proc
        assert proc is not None
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        rc = proc.wait()
        if self._stderr is not None:
            self._stderr.join()
        if rc != 0:
            self._fail(f"ffmpeg encoder exited with code {rc}")

    def abort(self) -> None:
        """Kill the encoder and remove its partial output."""
        if self._out is not None:
            try:
                self._out.close()
            except Exception:
                pass
            self._out = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr is not None:
            self._stderr.join(1.0)
        self.out_path.unlink(missing_ok=True)

    def _fail(self, message: str) -> NoReturn:
        stderr = self._stderr.text() if self._stderr is not None else ""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        if self._stderr is not None:
            self._stderr.join(1.0)
            stderr = self._stderr.text()
        raise NeuralFrameError(
            ErrorCode.ENCODE_FAILED,
            f"{message}; ffmpeg: {stderr}" if stderr else message,
            details={"encoder": self.plan.name, "output": str(self.out_path)},
        )


def preflight(
    tools: FFmpegTools, plan: EncoderPlan, fmt: RawVideoFormat, time_base: Fraction, frame_rate: Fraction,
    work_dir: Path,
) -> str | None:
    """Encode two frames with the real settings and output size; return None on success, else the error.

    Catches failures that capability probes at 320x240 cannot (size limits, session limits, profiles)
    before the pipeline starts, so the job can switch encoders instead of failing mid-way.
    """
    out = work_dir / f"preflight-{plan.name}.mp4"
    blank = np.zeros(fmt.frame_samples, dtype=fmt.dtype)
    blank[: fmt.width * fmt.height] = 16 << (fmt.bit_depth - 8)
    blank[fmt.width * fmt.height :] = 128 << (fmt.bit_depth - 8)
    step = max(1, round(1 / (frame_rate * time_base))) if frame_rate > 0 else 1
    try:
        with SegmentEncoder(tools, out, fmt, time_base, frame_rate, plan) as enc:
            for i in range(2):
                enc.write(i * step, step, blank)
    except NeuralFrameError as exc:
        return exc.message
    finally:
        out.unlink(missing_ok=True)
    return None
