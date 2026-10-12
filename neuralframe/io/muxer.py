"""Segment concatenation, final stream-copy mux and A/V sync verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import av

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, run_tool
from neuralframe.io.media import MediaInfo
from neuralframe.io.probe import probe, scan_frames
from neuralframe.log import get_logger

log = get_logger(component="muxer")

CONTAINERS: dict[str, str] = {
    ".mkv": "matroska",
    ".mka": "matroska",
    ".mp4": "mp4",
    ".m4v": "mp4",
    ".mov": "mov",
    ".webm": "webm",
}

_TEXT_SUBS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
_SUB_COPY: dict[str, set[str]] = {
    "matroska": {"subrip", "srt", "ass", "ssa", "webvtt", "hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle"},
    "mp4": {"mov_text"},
    "mov": {"mov_text"},
    "webm": {"webvtt"},
}
_SUB_CONVERT_TARGET = {"matroska": "srt", "mp4": "mov_text", "mov": "mov_text", "webm": "webvtt"}
_AUDIO_OK: dict[str, set[str] | None] = {
    "matroska": None,  # Matroska holds anything FFmpeg can mux
    "mov": None,
    "mp4": {"aac", "mp3", "mp2", "ac3", "eac3", "opus", "flac", "alac", "dts", "truehd", "vorbis"},
    "webm": {"opus", "vorbis"},
}

Action = Literal["copy", "convert", "drop"]


def container_for(path: Path) -> str:
    try:
        return CONTAINERS[path.suffix.lower()]
    except KeyError:
        raise NeuralFrameError(
            ErrorCode.INVALID_INPUT,
            f"unsupported output extension {path.suffix!r}",
            hint=f"use one of {', '.join(sorted(CONTAINERS))}",
        ) from None


@dataclass(frozen=True)
class StreamMapping:
    input_index: int
    kind: Literal["audio", "subtitle", "attachment"]
    action: Action
    codec: str
    target_codec: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MuxPlan:
    container: str
    streams: tuple[StreamMapping, ...]
    copy_chapters: bool
    video_language: str | None = None
    video_title: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> dict[str, object]:
        return {
            "container": self.container,
            "streams": [
                {"index": s.input_index, "kind": s.kind, "action": s.action, "codec": s.codec,
                 "target_codec": s.target_codec, "reason": s.reason}
                for s in self.streams
            ],
            "chapters": self.copy_chapters,
            "warnings": list(self.warnings),
        }


def plan_mux(media: MediaInfo, output: Path) -> MuxPlan:
    """Decide what happens to every non-video stream. Audio is always copied bit-exactly or rejected."""
    container = container_for(output)
    streams: list[StreamMapping] = []
    warnings: list[str] = []

    allowed_audio = _AUDIO_OK[container]
    for a in media.audio:
        if allowed_audio is not None and a.codec not in allowed_audio:
            raise NeuralFrameError(
                ErrorCode.MUX_FAILED,
                f"audio stream #{a.index} ({a.codec}) cannot be stream-copied into {container}",
                hint="use a .mkv output to keep the original audio bit-exact",
            )
        streams.append(StreamMapping(a.index, "audio", "copy", a.codec))

    for s in media.subtitles:
        if s.codec in _SUB_COPY[container]:
            streams.append(StreamMapping(s.index, "subtitle", "copy", s.codec))
        elif s.codec in _TEXT_SUBS:
            target = _SUB_CONVERT_TARGET[container]
            msg = f"subtitle #{s.index} ({s.codec}) converted to {target}: {container} cannot hold {s.codec}"
            warnings.append(msg)
            streams.append(StreamMapping(s.index, "subtitle", "convert", s.codec, target, msg))
        else:
            msg = f"subtitle #{s.index} ({s.codec}) dropped: bitmap subtitles are not supported in {container}"
            warnings.append(msg)
            streams.append(StreamMapping(s.index, "subtitle", "drop", s.codec, reason=msg))

    for att in media.attachments:
        if container == "matroska":
            streams.append(StreamMapping(att.index, "attachment", "copy", att.codec or "attachment"))
        else:
            msg = f"attachment #{att.index} ({att.filename}) dropped: only Matroska stores attachments"
            warnings.append(msg)
            streams.append(StreamMapping(att.index, "attachment", "drop", att.codec or "?", reason=msg))

    if media.data_streams:
        warnings.append(f"data streams {media.data_streams} (e.g. timecode tracks) are not carried over")
    if media.extra_video_streams:
        warnings.append(f"additional video streams {media.extra_video_streams} (e.g. cover art) are not kept")

    video = media.video
    return MuxPlan(
        container=container,
        streams=tuple(streams),
        copy_chapters=bool(media.chapters),
        video_language=video.tags.get("language") if video else None,
        video_title=video.tags.get("title") if video else None,
        warnings=tuple(warnings),
    )


def concat_segments(segments: list[Path], output: Path) -> int:
    """Join video-only segments by packet copy. Segments carry absolute PTS, so nothing is re-timed.

    Returns the number of video packets written.
    """
    if not segments:
        raise NeuralFrameError(ErrorCode.MUX_FAILED, "no segments to concatenate")
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    last_dts: int | None = None
    try:
        with av.open(str(output), mode="w", format="mp4") as out:
            ostream = None
            for seg in segments:
                with av.open(str(seg)) as inp:
                    istream = inp.streams.video[0]
                    if ostream is None:
                        ostream = out.add_stream_from_template(istream)
                    for packet in inp.demux(istream):
                        if packet.size == 0:
                            continue
                        if packet.dts is not None and last_dts is not None and packet.dts <= last_dts:
                            raise NeuralFrameError(
                                ErrorCode.MUX_FAILED,
                                f"segment {seg.name} overlaps the previous one (dts {packet.dts} <= {last_dts})",
                            )
                        last_dts = packet.dts
                        packet.stream = ostream
                        out.mux(packet)
                        written += 1
    except NeuralFrameError:
        raise
    except Exception as exc:
        raise NeuralFrameError(ErrorCode.MUX_FAILED, f"segment concatenation failed: {exc}") from exc
    return written


def mux_final(
    tools: FFmpegTools,
    video: Path,
    source: MediaInfo,
    plan: MuxPlan,
    output: Path,
    *,
    loglevel: str = "error",
) -> None:
    """Combine the new video with the source's audio/subtitles/chapters/metadata, atomically."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    cmd = [
        tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", loglevel, "-y",
        # -copyts on both inputs keeps the original timeline: the video segments already carry source
        # PTS, so audio/subtitles line up exactly as they did in the source.
        "-copyts", "-i", str(video), "-i", str(source.path),
        "-map", "0:v:0",
    ]
    out_index = 1
    codec_overrides: list[str] = []
    for mapping in plan.streams:
        if mapping.action == "drop":
            continue
        cmd += ["-map", f"1:{mapping.input_index}"]
        if mapping.action == "convert" and mapping.target_codec:
            codec_overrides += [f"-c:{out_index}", mapping.target_codec]
        out_index += 1
    cmd += ["-map_metadata", "1", "-map_chapters", "1" if plan.copy_chapters else "-1", "-c", "copy"]
    cmd += codec_overrides
    if plan.video_language:
        cmd += ["-metadata:s:v:0", f"language={plan.video_language}"]
    if plan.video_title:
        cmd += ["-metadata:s:v:0", f"title={plan.video_title}"]
    if plan.container in ("mp4", "mov"):
        cmd += ["-movflags", "+faststart"]
    cmd += ["-f", plan.container, str(partial)]
    log.debug("final mux", cmd=" ".join(cmd))
    try:
        run_tool(cmd, error_code=ErrorCode.MUX_FAILED, what="final mux", timeout=None)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class SyncReport:
    ok: bool
    expected_frames: int
    output_frames: int
    source_offset_s: float | None
    output_offset_s: float | None
    frame_duration_s: float
    video_duration_s: float | None
    source_video_duration_s: float | None
    problems: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "expected_frames": self.expected_frames,
            "output_frames": self.output_frames,
            "source_av_offset_s": self.source_offset_s,
            "output_av_offset_s": self.output_offset_s,
            "frame_duration_s": self.frame_duration_s,
            "video_duration_s": self.video_duration_s,
            "source_video_duration_s": self.source_video_duration_s,
            "problems": list(self.problems),
        }


def verify_output(
    tools: FFmpegTools,
    source: MediaInfo,
    output: Path,
    *,
    expected_frames: int,
    frame_duration_s: float,
    expected_video_duration_s: float | None = None,
) -> SyncReport:
    """Check frame count, duration and audio/video start alignment (within one frame)."""
    out = probe(output, tools)
    problems: list[str] = []
    out_video = out.require_video()
    timeline = scan_frames(output, tools, out_video.index)
    if timeline.frame_count != expected_frames:
        problems.append(f"frame count {timeline.frame_count} != expected {expected_frames}")

    tolerance = frame_duration_s + 0.002  # one frame + container timestamp rounding
    src_video = source.video
    src_offset = out_offset = None
    if source.audio and out.audio and src_video is not None:
        src_a, out_a = source.audio[0], out.audio[0]
        if src_video.start_time is not None and src_a.start_time is not None:
            src_offset = src_video.start_time - src_a.start_time
        if out_video.start_time is not None and out_a.start_time is not None:
            out_offset = out_video.start_time - out_a.start_time
        if src_offset is not None and out_offset is not None and abs(src_offset - out_offset) > tolerance:
            problems.append(
                f"A/V start offset changed from {src_offset:.4f}s to {out_offset:.4f}s (> one frame)"
            )
        if len(out.audio) != len(source.audio):
            problems.append(f"audio streams {len(out.audio)} != source {len(source.audio)}")

    src_duration = expected_video_duration_s
    if src_duration is None and src_video is not None:
        src_duration = src_video.duration or source.duration
    # MKV has no per-stream duration and its container duration includes audio padding.
    out_duration = out_video.duration or timeline.span_s or out.duration
    if src_duration is not None and out_duration is not None and abs(src_duration - out_duration) > tolerance:
        problems.append(f"video duration {out_duration:.4f}s differs from {src_duration:.4f}s by > one frame")

    return SyncReport(
        ok=not problems,
        expected_frames=expected_frames,
        output_frames=timeline.frame_count,
        source_offset_s=src_offset,
        output_offset_s=out_offset,
        frame_duration_s=frame_duration_s,
        video_duration_s=out_duration,
        source_video_duration_s=src_duration,
        problems=tuple(problems),
    )
