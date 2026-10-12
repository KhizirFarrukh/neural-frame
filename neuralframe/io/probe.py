"""ffprobe → :class:`MediaInfo`, plus packet-level frame timeline scans and interlace detection."""

from __future__ import annotations

import functools
import itertools
import json
import re
import statistics
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, run_ffprobe_json, run_tool
from neuralframe.io.media import (
    AttachmentStream,
    AudioStream,
    Chapter,
    ColorInfo,
    FrameTimeline,
    MediaInfo,
    PixelFormat,
    SubtitleStream,
    VideoStream,
    to_fraction,
)

_IMAGE_FORMATS = re.compile(r"(^image2$|_pipe$|^png$|^webp$|^apng$)")
_TEXT_SUBTITLES = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "microdvd", "subviewer"}


@functools.lru_cache(maxsize=4)
def _pixel_formats(ffprobe: str) -> dict[str, PixelFormat]:
    proc = run_tool(
        [ffprobe, "-v", "error", "-show_pixel_formats", "-print_format", "json"],
        error_code=ErrorCode.PROBE_FAILED,
        what="ffprobe -show_pixel_formats",
        timeout=60,
    )
    raw = json.loads(proc.stdout.decode("utf-8", "replace"))
    table: dict[str, PixelFormat] = {}
    for item in raw.get("pixel_formats", []):
        flags = item.get("flags", {})
        comps = item.get("components", [])
        depth = max((int(c.get("bit_depth", 8)) for c in comps), default=8)
        table[item["name"]] = PixelFormat(
            name=item["name"],
            bit_depth=depth,
            log2_chroma_w=int(item.get("log2_chroma_w", 0)),
            log2_chroma_h=int(item.get("log2_chroma_h", 0)),
            nb_components=int(item.get("nb_components", 0)),
            is_rgb=bool(flags.get("rgb", 0)),
            has_alpha=bool(flags.get("alpha", 0)),
            is_planar=bool(flags.get("planar", 0)),
            is_hwaccel=bool(flags.get("hwaccel", 0)),
            is_float=bool(flags.get("float", 0)) or item["name"].endswith(("f32le", "f32be", "f16le", "f16be")),
        )
    return table


def pixel_format(tools: FFmpegTools, name: str) -> PixelFormat:
    table = _pixel_formats(tools.ffprobe)
    if name in table:
        return table[name]
    return PixelFormat(name=name or "unknown", bit_depth=8, log2_chroma_w=1, log2_chroma_h=1)


def _float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "N/A", "") else None
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value) if value not in (None, "N/A", "") else None
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str | None:
    if value in (None, "", "unknown", "unspecified", "reserved", "N/A"):
        return None
    return str(value)


def _rotation(stream: dict[str, Any]) -> int:
    for side in stream.get("side_data_list", []) or []:
        if side.get("side_data_type") == "Display Matrix" and "rotation" in side:
            return round(float(side["rotation"])) % 360
    rotate_tag = (stream.get("tags") or {}).get("rotate")
    if rotate_tag is not None:
        # Legacy tag is clockwise; normalise to counter-clockwise like the display matrix.
        return (-int(float(rotate_tag))) % 360
    return 0


def _hdr_format(stream: dict[str, Any], transfer: str | None) -> str | None:
    side_types = {s.get("side_data_type", "") for s in stream.get("side_data_list", []) or []}
    if any("DOVI" in t or "Dolby Vision" in t for t in side_types):
        return "dolby_vision"
    if transfer == "smpte2084":
        return "hdr10"
    if transfer == "arib-std-b67":
        return "hlg"
    return None


def _parse_video(tools: FFmpegTools, s: dict[str, Any], format_name: str) -> VideoStream:
    pix_fmt = s.get("pix_fmt") or "unknown"
    tags = {str(k): str(v) for k, v in (s.get("tags") or {}).items()}
    disposition = s.get("disposition") or {}
    color = ColorInfo(
        matrix=_clean(s.get("color_space")),
        primaries=_clean(s.get("color_primaries")),
        transfer=_clean(s.get("color_transfer")),
        range=_clean(s.get("color_range")),
        chroma_location=_clean(s.get("chroma_location")),
    )
    frame_count = _int(s.get("nb_frames")) or _int(s.get("nb_read_packets"))
    still = bool(_IMAGE_FORMATS.search(format_name)) or (
        s.get("codec_name") in ("png", "mjpeg", "webp", "bmp", "tiff", "jpegxl")
        and (frame_count or 1) <= 1
        and _float(s.get("duration")) in (None, 0.0)
    )
    sar = to_fraction(s.get("sample_aspect_ratio") or "1:1")
    return VideoStream(
        index=int(s["index"]),
        codec=s.get("codec_name", "unknown"),
        profile=_clean(s.get("profile")),
        width=int(s.get("width") or 0),
        height=int(s.get("height") or 0),
        pix_fmt=pix_fmt,
        pixel_format=pixel_format(tools, pix_fmt),
        r_frame_rate=to_fraction(s.get("r_frame_rate") or "0/1"),
        avg_frame_rate=to_fraction(s.get("avg_frame_rate") or "0/1"),
        time_base=to_fraction(s.get("time_base") or "1/1000"),
        start_time=_float(s.get("start_time")),
        duration=_float(s.get("duration")),
        frame_count=frame_count,
        bit_rate=_int(s.get("bit_rate")),
        sar=sar if sar > 0 else Fraction(1),
        rotation=_rotation(s),
        field_order=_clean(s.get("field_order")),
        color=color,
        hdr_format=_hdr_format(s, color.transfer),
        is_still_image=still,
        is_attached_pic=bool(disposition.get("attached_pic", 0)),
        tags=tags,
    )


def probe(path: Path, tools: FFmpegTools) -> MediaInfo:
    """Describe a media file. Raises PROBE_FAILED for unreadable input."""
    if not path.is_file():
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"input file not found: {path}")
    data = run_ffprobe_json(
        tools,
        ["-show_format", "-show_streams", "-show_chapters", str(path)],
        what=f"ffprobe {path.name}",
    )
    fmt = data.get("format") or {}
    format_name = str(fmt.get("format_name", "unknown"))
    streams: list[dict[str, Any]] = data.get("streams") or []

    video: VideoStream | None = None
    extra_video: list[int] = []
    audio: list[AudioStream] = []
    subtitles: list[SubtitleStream] = []
    attachments: list[AttachmentStream] = []
    data_streams: list[int] = []

    for s in streams:
        kind = s.get("codec_type")
        tags = s.get("tags") or {}
        if kind == "video":
            parsed = _parse_video(tools, s, format_name)
            if video is None and not parsed.is_attached_pic:
                video = parsed
            else:
                extra_video.append(parsed.index)
        elif kind == "audio":
            audio.append(
                AudioStream(
                    index=int(s["index"]),
                    codec=s.get("codec_name", "unknown"),
                    profile=_clean(s.get("profile")),
                    channels=_int(s.get("channels")),
                    channel_layout=_clean(s.get("channel_layout")),
                    sample_rate=_int(s.get("sample_rate")),
                    bit_rate=_int(s.get("bit_rate")),
                    start_time=_float(s.get("start_time")),
                    duration=_float(s.get("duration")),
                    language=_clean(tags.get("language")),
                    title=_clean(tags.get("title")),
                )
            )
        elif kind == "subtitle":
            codec = s.get("codec_name", "unknown")
            subtitles.append(
                SubtitleStream(
                    index=int(s["index"]),
                    codec=codec,
                    language=_clean(tags.get("language")),
                    title=_clean(tags.get("title")),
                    is_text=codec in _TEXT_SUBTITLES,
                )
            )
        elif kind == "attachment":
            attachments.append(
                AttachmentStream(
                    index=int(s["index"]),
                    codec=_clean(s.get("codec_name")),
                    filename=_clean(tags.get("filename")),
                    mimetype=_clean(tags.get("mimetype")),
                )
            )
        else:
            data_streams.append(int(s["index"]))

    chapters = [
        Chapter(
            start=float(c.get("start_time", 0.0)),
            end=float(c.get("end_time", 0.0)),
            title=_clean((c.get("tags") or {}).get("title")),
        )
        for c in data.get("chapters") or []
    ]

    if video is None and not audio:
        raise NeuralFrameError(ErrorCode.PROBE_FAILED, f"{path} contains no audio or video streams")

    return MediaInfo(
        path=path,
        format_name=format_name,
        format_long_name=_clean(fmt.get("format_long_name")),
        duration=_float(fmt.get("duration")),
        size=_int(fmt.get("size")),
        bit_rate=_int(fmt.get("bit_rate")),
        start_time=_float(fmt.get("start_time")),
        video=video,
        audio=audio,
        subtitles=subtitles,
        attachments=attachments,
        data_streams=data_streams,
        extra_video_streams=extra_video,
        chapters=chapters,
        tags={str(k): str(v) for k, v in (fmt.get("tags") or {}).items()},
    )


def scan_frames(path: Path, tools: FFmpegTools, stream_index: int) -> FrameTimeline:
    """Demux-only scan of video packets: exact frame count, sorted PTS, keyframes and VFR detection."""
    proc = run_tool(
        [
            tools.ffprobe, "-v", "error", "-select_streams", str(stream_index),
            "-show_entries", "packet=pts,dts,flags:stream=time_base", "-of", "csv=p=0", str(path),
        ],
        error_code=ErrorCode.PROBE_FAILED,
        what="packet scan",
        timeout=3600,
    )
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    time_base = Fraction(1, 1000)
    entries: list[tuple[int, bool]] = []
    fallback = 0
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) == 1 and "/" in parts[0]:
            time_base = to_fraction(parts[0])
            continue
        if len(parts) < 3:
            continue
        pts = _int(parts[0])
        if pts is None:
            pts = _int(parts[1])
        if pts is None:
            pts = fallback
        fallback = pts + 1
        entries.append((pts, "K" in parts[2]))
    if not entries:
        raise NeuralFrameError(ErrorCode.PROBE_FAILED, f"no video packets found in {path}")

    order = sorted(range(len(entries)), key=lambda i: entries[i][0])
    pts_sorted = [entries[i][0] for i in order]
    keyframes = [rank for rank, i in enumerate(order) if entries[i][1]]
    deltas = [b - a for a, b in itertools.pairwise(pts_sorted) if b > a]
    nominal = int(statistics.median(deltas)) if deltas else 0
    # Timestamp jitter of a tick or two (container rounding, e.g. 33/34 ms) is not VFR.
    tolerance = max(2, nominal // 10)
    is_vfr = any(abs(d - nominal) > tolerance for d in deltas)
    return FrameTimeline(
        time_base=time_base, pts=pts_sorted, keyframes=keyframes, nominal_duration=nominal, is_vfr=is_vfr
    )


@dataclass(frozen=True)
class InterlaceReport:
    tff: int
    bff: int
    progressive: int
    undetermined: int

    @property
    def is_interlaced(self) -> bool:
        interlaced = self.tff + self.bff
        decided = interlaced + self.progressive
        return decided > 0 and interlaced / decided > 0.25

    @property
    def parity(self) -> str:
        return "tff" if self.tff >= self.bff else "bff"


_IDET_RE = re.compile(
    r"Multi frame detection:\s*TFF:\s*(\d+)\s*BFF:\s*(\d+)\s*Progressive:\s*(\d+)\s*Undetermined:\s*(\d+)"
)


def detect_interlacing(path: Path, tools: FFmpegTools, frames: int = 400) -> InterlaceReport:
    """Run FFmpeg's ``idet`` on the first ``frames`` frames and summarise its multi-frame verdict."""
    proc = run_tool(
        [
            tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-i", str(path),
            "-map", "0:v:0", "-frames:v", str(frames), "-vf", "idet", "-an", "-sn", "-f", "null", "-",
        ],
        error_code=ErrorCode.PROBE_FAILED,
        what="interlace detection",
        timeout=600,
    )
    matches = list(_IDET_RE.finditer(proc.stderr.decode("utf-8", "replace")))
    if not matches:
        return InterlaceReport(0, 0, 0, 0)
    tff, bff, prog, und = (int(g) for g in matches[-1].groups())
    return InterlaceReport(tff=tff, bff=bff, progressive=prog, undetermined=und)
