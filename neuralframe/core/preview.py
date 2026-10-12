"""Single frames for before/after comparisons (web UI, desktop GUI)."""

from __future__ import annotations

import io
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, popen_kwargs
from neuralframe.io.media import MediaInfo

# Our colour enums -> names used by FFmpeg's colorspace filter (BT.601 525/625 share one matrix).
_CS_MATRIX = {"bt601": "smpte170m", "bt709": "bt709", "bt2020nc": "bt2020ncl", "smpte240m": "smpte240m", "fcc": "fcc"}


def colour_match_filter(report: dict[str, Any] | None) -> str | None:
    """FFmpeg filter giving a source frame the job's output colorimetry, so "before" and "after" compare
    like for like (a job may convert matrix and primaries, e.g. SD SMPTE-C to BT.709)."""
    color = (report or {}).get("plan", {}).get("color")
    if not color or color.get("hdr_passthrough"):
        return None
    src, dst = color["input"], color["output"]
    ispace, space = _CS_MATRIX.get(src["matrix"]), _CS_MATRIX.get(dst["colorspace"])
    if ispace is None or space is None:
        return None
    if (ispace, src["primaries"], src["range"]) == (space, dst["color_primaries"], dst["color_range"]):
        return None
    return (f"colorspace=ispace={ispace}:iprimaries={src['primaries']}:itrc={src['transfer']}:"
            f"irange={src['range']}:space={space}:primaries={dst['color_primaries']}:trc={dst['color_trc']}:"
            f"range={dst['color_range']}:fast=0")


def frame_time(media: MediaInfo, frame: int) -> float:
    """Seek position that lands on source frame ``frame`` (half a frame early, so rounding cannot skip it)."""
    if media.is_image:
        return 0.0
    video = media.require_video()
    fps = video.fps if video.fps and video.fps > 0 else Fraction(25)
    return max(0.0, float((frame - Fraction(1, 2)) / fps))


def frame_png(tools: FFmpegTools, path: Path, seconds: float, size: tuple[int, int] | None = None,
              colour: str | None = None) -> bytes:
    """One frame of ``path`` at ``seconds`` as PNG, optionally colour-converted and scaled (bicubic)."""
    args = [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(seconds, 0.0):.6f}",
            "-i", str(path), "-frames:v", "1", "-an", "-sn"]
    filters = [f for f in (colour, f"scale={size[0]}:{size[1]}:flags=bicubic" if size else None) if f]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-f", "image2pipe", "-c:v", "png", "pipe:1"]
    proc = subprocess.run(args, capture_output=True, timeout=120, check=False, **popen_kwargs())
    if proc.returncode != 0 or not proc.stdout:
        raise NeuralFrameError(ErrorCode.DECODE_FAILED, f"cannot extract a preview frame from {path.name}: "
                               + proc.stderr.decode(errors="replace").strip()[-300:])
    return bytes(proc.stdout)


def side_by_side(left_png: bytes, right_png: bytes) -> bytes:
    from PIL import Image

    left, right = Image.open(io.BytesIO(left_png)), Image.open(io.BytesIO(right_png))
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right.convert("RGB"), (left.width, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
