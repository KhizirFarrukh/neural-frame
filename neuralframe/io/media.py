"""Typed media description produced by :mod:`neuralframe.io.probe`."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, WithJsonSchema


def to_fraction(value: Any) -> Fraction:
    """Parse ffprobe rationals ("30000/1001", "16:9"), numbers, or Fractions. Invalid/zero-den → 0."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a rational")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        return Fraction(value).limit_denominator(1_000_000)
    if isinstance(value, str):
        text = value.strip()
        for sep in ("/", ":"):
            if sep in text:
                num, den = text.split(sep, 1)
                try:
                    n, d = int(num), int(den)
                except ValueError:
                    return Fraction(0)
                return Fraction(n, d) if d != 0 else Fraction(0)
        try:
            return Fraction(text)
        except ValueError:
            return Fraction(0)
    raise TypeError(f"cannot interpret {value!r} as a rational")


Rational = Annotated[
    Fraction,
    PlainValidator(to_fraction),
    PlainSerializer(lambda f: f"{f.numerator}/{f.denominator}", return_type=str),
    WithJsonSchema({"type": "string", "examples": ["30000/1001"]}),
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class PixelFormat(_Model):
    name: str
    bit_depth: int
    log2_chroma_w: int = 0
    log2_chroma_h: int = 0
    nb_components: int = 3
    is_rgb: bool = False
    has_alpha: bool = False
    is_planar: bool = True
    is_hwaccel: bool = False
    is_float: bool = False


class ColorInfo(_Model):
    """Raw colour metadata as tagged in the stream (None = untagged)."""

    matrix: str | None = None
    primaries: str | None = None
    transfer: str | None = None
    range: str | None = None
    chroma_location: str | None = None


class VideoStream(_Model):
    index: int
    codec: str
    profile: str | None = None
    width: int
    height: int
    pix_fmt: str
    pixel_format: PixelFormat
    r_frame_rate: Rational = Fraction(0)
    avg_frame_rate: Rational = Fraction(0)
    time_base: Rational = Fraction(1, 1000)
    start_time: float | None = None
    duration: float | None = None
    frame_count: int | None = None
    bit_rate: int | None = None
    sar: Rational = Fraction(1)
    rotation: int = 0
    """Display rotation in degrees counter-clockwise (0, 90, 180, 270), as stored in the display matrix."""
    field_order: str | None = None
    color: ColorInfo = Field(default_factory=ColorInfo)
    hdr_format: str | None = None
    is_still_image: bool = False
    is_attached_pic: bool = False
    tags: dict[str, str] = Field(default_factory=dict)

    @property
    def bit_depth(self) -> int:
        return self.pixel_format.bit_depth

    @property
    def fps(self) -> Fraction:
        """Best guess of the nominal frame rate (avg if sane, else r_frame_rate)."""
        if self.avg_frame_rate > 0 and self.avg_frame_rate < 1000:
            return self.avg_frame_rate
        return self.r_frame_rate

    @property
    def display_aspect(self) -> Fraction:
        """Display aspect ratio of the upright picture (SAR and rotation applied)."""
        sar = self.sar if self.sar > 0 else Fraction(1)
        dar = Fraction(self.width, self.height) * sar
        return 1 / dar if self.rotation in (90, 270) else dar

    @property
    def display_size(self) -> tuple[int, int]:
        """Upright size with square pixels (width stretched/shrunk by SAR, then rotated)."""
        sar = self.sar if self.sar > 0 else Fraction(1)
        w = round(self.width * sar) if sar >= 1 else self.width
        h = self.height if sar >= 1 else round(self.height / sar)
        return (h, w) if self.rotation in (90, 270) else (w, h)

    @property
    def is_hdr(self) -> bool:
        return self.hdr_format is not None

    @property
    def is_interlaced_tagged(self) -> bool:
        return self.field_order in ("tt", "bb", "tb", "bt")


class AudioStream(_Model):
    index: int
    codec: str
    profile: str | None = None
    channels: int | None = None
    channel_layout: str | None = None
    sample_rate: int | None = None
    bit_rate: int | None = None
    start_time: float | None = None
    duration: float | None = None
    language: str | None = None
    title: str | None = None


class SubtitleStream(_Model):
    index: int
    codec: str
    language: str | None = None
    title: str | None = None
    is_text: bool = True


class AttachmentStream(_Model):
    index: int
    codec: str | None = None
    filename: str | None = None
    mimetype: str | None = None


class Chapter(_Model):
    start: float
    end: float
    title: str | None = None


class MediaInfo(_Model):
    path: Path
    format_name: str
    format_long_name: str | None = None
    duration: float | None = None
    size: int | None = None
    bit_rate: int | None = None
    start_time: float | None = None
    video: VideoStream | None = None
    audio: list[AudioStream] = Field(default_factory=list)
    subtitles: list[SubtitleStream] = Field(default_factory=list)
    attachments: list[AttachmentStream] = Field(default_factory=list)
    data_streams: list[int] = Field(default_factory=list)
    extra_video_streams: list[int] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        return self.video is not None and self.video.is_still_image

    def require_video(self) -> VideoStream:
        from neuralframe.errors import ErrorCode, NeuralFrameError

        if self.video is None:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"{self.path} has no video stream")
        return self.video


class FrameTimeline(_Model):
    """Presentation timestamps of every video packet (sorted), from a demux-only packet scan."""

    time_base: Rational
    pts: list[int]
    keyframes: list[int] = Field(default_factory=list)
    """Indices (into ``pts``) of keyframes, usable as seek points."""
    nominal_duration: int = 0
    is_vfr: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.pts)

    def seconds(self, index: int) -> float:
        return float(self.pts[index] * self.time_base)

    @property
    def span_s(self) -> float | None:
        """First frame start to last frame end (video only, unlike a container duration that includes audio)."""
        if not self.pts:
            return None
        return float((self.pts[-1] - self.pts[0] + self.nominal_duration) * self.time_base)
