"""Job planning: output geometry, colour handling, decode filters and encoder selection.

Every automatic decision is recorded as a human-readable note so it can be logged and written to the
job report.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from neuralframe.colorspace import (
    HDR_TRANSFERS,
    ChromaLocation,
    ColorPlan,
    ColorRange,
    Matrix,
    Primaries,
    Transfer,
    parse_chroma_location,
    parse_matrix,
    parse_primaries,
    parse_range,
    parse_transfer,
    same_gamut,
)
from neuralframe.config import ColorConfig, DecodeConfig
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.decoder import DecodePlan
from neuralframe.io.encoder import EncoderPlan
from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.io.hwcaps import Capabilities
from neuralframe.io.media import VideoStream
from neuralframe.io.probe import detect_interlacing
from neuralframe.io.rawvideo import RawVideoFormat

# ---------------------------------------------------------------------------------------------------------
# Target size
# ---------------------------------------------------------------------------------------------------------

TargetMode = Literal["box", "height", "scale", "size", "native"]

NAMED_BOXES: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2160p": (3840, 2160),
    "4k": (3840, 2160),
    "uhd": (3840, 2160),
    "4320p": (7680, 4320),
    "8k": (7680, 4320),
}


class TargetSpec(BaseModel):
    """What size the output should be.

    * ``box``    — fit inside ``width`` x ``height`` (orientation-aware), aspect preserved. Default: 4K.
    * ``height`` — exact output height, width from the display aspect ratio.
    * ``scale``  — multiply the display size by ``scale``.
    * ``size``   — exact ``width`` x ``height`` (stretches if the aspect differs).
    * ``native`` — the model's native scale factor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: TargetMode = "box"
    width: int | None = Field(default=3840, ge=16, le=16384)
    height: int | None = Field(default=2160, ge=16, le=16384)
    scale: float | None = Field(default=None, gt=0, le=16)
    allow_downscale: bool = False

    @classmethod
    def default(cls) -> TargetSpec:
        return cls()

    @classmethod
    def fit(cls, width: int, height: int) -> TargetSpec:
        return cls(mode="box", width=width, height=height)

    @classmethod
    def for_height(cls, height: int) -> TargetSpec:
        return cls(mode="height", width=None, height=height, allow_downscale=True)

    @classmethod
    def factor(cls, scale: float) -> TargetSpec:
        return cls(mode="scale", width=None, height=None, scale=scale, allow_downscale=True)

    @classmethod
    def exact(cls, width: int, height: int) -> TargetSpec:
        return cls(mode="size", width=width, height=height, allow_downscale=True)

    @classmethod
    def native(cls) -> TargetSpec:
        return cls(mode="native", width=None, height=None)

    @classmethod
    def parse(cls, text: str) -> TargetSpec:
        """Parse CLI/GUI strings: ``4k``, ``1080p``, ``1920x1080``, ``x2`` / ``2x``, ``native``, ``h1080``."""
        value = text.strip().lower()
        if value in NAMED_BOXES:
            return cls.fit(*NAMED_BOXES[value])
        if value == "native":
            return cls.native()
        if m := re.fullmatch(r"(\d+)\s*x\s*(\d+)", value):
            return cls.exact(int(m.group(1)), int(m.group(2)))
        if m := re.fullmatch(r"(?:x\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*x)", value):
            return cls.factor(float(m.group(1) or m.group(2)))
        if m := re.fullmatch(r"h?(\d+)p?", value):
            return cls.for_height(int(m.group(1)))
        raise NeuralFrameError(
            ErrorCode.INVALID_INPUT,
            f"cannot parse target size {text!r}",
            hint="examples: 4k, 1080p, 1920x1080, x2, 2.5x, native",
        )

    def describe(self) -> str:
        if self.mode == "box":
            return f"fit {self.width}x{self.height}"
        if self.mode == "height":
            return f"height {self.height}"
        if self.mode == "scale":
            return f"x{self.scale:g}"
        if self.mode == "size":
            return f"exactly {self.width}x{self.height}"
        return "model native scale"


_STANDARD_RATIOS = [
    Fraction(16, 9), Fraction(4, 3), Fraction(3, 2), Fraction(5, 4), Fraction(1, 1), Fraction(37, 20),
    Fraction(2, 1), Fraction(64, 27), Fraction(239, 100), Fraction(12, 5), Fraction(21, 9),
]


def snap_aspect(dar: Fraction, tolerance: float = 0.01) -> Fraction:
    """Snap a measured display aspect (e.g. 854/480) to a standard one (16:9) when within tolerance."""
    candidates = _STANDARD_RATIOS + [1 / r for r in _STANDARD_RATIOS]
    best = min(candidates, key=lambda r: abs(float(r) / float(dar) - 1.0))
    return best if abs(float(best) / float(dar) - 1.0) <= tolerance else dar


def _even(value: float) -> int:
    return max(2, 2 * round(value / 2))


@dataclass(frozen=True)
class Geometry:
    decoded_width: int
    decoded_height: int
    out_width: int
    out_height: int
    model_scale: int
    model_passes: int
    display_aspect: Fraction

    @property
    def model_out_width(self) -> int:
        return self.decoded_width * int(self.model_scale**self.model_passes)

    @property
    def model_out_height(self) -> int:
        return self.decoded_height * int(self.model_scale**self.model_passes)

    @property
    def upscale_factor(self) -> float:
        return max(self.out_width / self.decoded_width, self.out_height / self.decoded_height)

    def summary(self) -> dict[str, object]:
        return {
            "decoded": f"{self.decoded_width}x{self.decoded_height}",
            "model_output": f"{self.model_out_width}x{self.model_out_height}",
            "output": f"{self.out_width}x{self.out_height}",
            "model_scale": self.model_scale,
            "model_passes": self.model_passes,
            "display_aspect": f"{self.display_aspect.numerator}:{self.display_aspect.denominator}",
            "upscale_factor": round(self.upscale_factor, 4),
        }


def plan_geometry(
    video: VideoStream,
    target: TargetSpec,
    model_scale: int,
    *,
    even_dims: bool = True,
    cascade: bool = False,
    max_dim: int = 8192,
) -> tuple[Geometry, list[str]]:
    notes: list[str] = []
    rotated = video.rotation in (90, 270)
    dec_w, dec_h = (video.height, video.width) if rotated else (video.width, video.height)
    disp_w, disp_h = video.display_size
    measured = video.display_aspect
    dar = snap_aspect(measured)
    if dar != measured:
        notes.append(f"display aspect {float(measured):.4f} snapped to {dar.numerator}:{dar.denominator}")
    rnd = _even if even_dims else (lambda v: max(1, round(v)))

    w: float
    h: float
    if target.mode == "box":
        assert target.width is not None and target.height is not None
        long_edge, short_edge = max(target.width, target.height), min(target.width, target.height)
        bw, bh = (short_edge, long_edge) if dar < 1 else (long_edge, short_edge)
        if Fraction(bw, bh) > dar:
            h, w = float(bh), float(bh * dar)
        else:
            w, h = float(bw), float(bw / dar)
        if not target.allow_downscale and w <= disp_w and h <= disp_h:
            notes.append(
                f"source display size {disp_w}x{disp_h} already fills {target.describe()}; keeping source size"
            )
            w, h = disp_w, disp_h
    elif target.mode == "height":
        assert target.height is not None
        h, w = float(target.height), float(target.height * dar)
    elif target.mode == "scale":
        assert target.scale is not None
        w, h = disp_w * target.scale, disp_h * target.scale
    elif target.mode == "size":
        assert target.width is not None and target.height is not None
        w, h = target.width, target.height
        if abs(Fraction(target.width, target.height) / dar - 1) > 0.01:
            notes.append(f"exact size {target.width}x{target.height} changes the aspect ratio (stretch)")
    else:
        w, h = disp_w * max(model_scale, 1), disp_h * max(model_scale, 1)

    out_w, out_h = int(rnd(float(w))), int(rnd(float(h)))
    if max(out_w, out_h) > max_dim:
        shrink = max_dim / max(out_w, out_h)
        out_w, out_h = int(rnd(out_w * shrink)), int(rnd(out_h * shrink))
        notes.append(f"output limited to {max_dim}px on the long edge ({out_w}x{out_h})")

    factor = max(out_w / dec_w, out_h / dec_h)
    passes = 0
    if model_scale > 1:
        if factor <= 1.0:
            notes.append("no upscaling required for this target; the super-resolution model is skipped")
        else:
            passes = 1
            if cascade and factor > model_scale * 1.5:
                passes = 2
                notes.append(f"factor {factor:.2f} exceeds model scale x{model_scale}: running two passes")
            elif factor > model_scale * 1.05:
                notes.append(
                    f"factor {factor:.2f} exceeds model scale x{model_scale}; remaining "
                    f"x{factor / model_scale:.2f} is Lanczos-resampled"
                )
    geometry = Geometry(dec_w, dec_h, out_w, out_h, model_scale, passes, dar)
    return geometry, notes


# ---------------------------------------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------------------------------------


def _guess_sd_primaries(height: int) -> Primaries:
    return Primaries.BT470BG if 560 <= height <= 600 else Primaries.SMPTE170M


def plan_color(video: VideoStream, out_height: int, cfg: ColorConfig, *, ten_bit: bool) -> ColorPlan:
    notes: list[str] = []
    coded_h = video.height if video.rotation in (0, 180) else video.width
    is_sd = coded_h <= 576
    tagged = video.color

    if video.pixel_format.is_rgb:
        in_matrix = Matrix.GBR
    else:
        in_matrix = parse_matrix(tagged.matrix) or (Matrix.BT601 if is_sd else Matrix.BT709)
        if tagged.matrix is None:
            notes.append(f"untagged matrix: assuming {in_matrix.value} for {coded_h}-line video")
        if tagged.matrix == "bt2020c":
            notes.append("constant-luminance BT.2020 is treated as non-constant luminance")

    in_range = parse_range(tagged.range)
    if in_range is None:
        full = video.pix_fmt.startswith("yuvj") or video.pixel_format.is_rgb or video.is_still_image
        in_range = ColorRange.FULL if full else ColorRange.LIMITED
        notes.append(f"untagged range: assuming {'full' if full else 'limited'}")

    in_primaries = parse_primaries(tagged.primaries)
    if in_primaries is None:
        if in_matrix is Matrix.BT2020NC:
            in_primaries = Primaries.BT2020
        elif is_sd and in_matrix is Matrix.BT601:
            in_primaries = _guess_sd_primaries(coded_h)
        else:
            in_primaries = Primaries.BT709
        notes.append(f"untagged primaries: assuming {in_primaries.value}")

    in_transfer = parse_transfer(tagged.transfer) or Transfer.BT709
    default_loc = ChromaLocation.CENTER if video.codec in ("mjpeg", "jpeg", "png", "webp") else ChromaLocation.LEFT
    in_chroma = parse_chroma_location(tagged.chroma_location) or default_loc

    hdr = in_transfer in HDR_TRANSFERS
    if hdr:
        notes.append(
            f"HDR input ({in_transfer.value}) passed through without tone mapping; output forced to 10-bit. "
            "The models are trained on SDR content, so results may vary."
        )
        out_matrix = in_matrix if in_matrix is not Matrix.GBR else Matrix.BT2020NC
        return ColorPlan(
            in_matrix=in_matrix, in_range=in_range, in_primaries=in_primaries, in_transfer=in_transfer,
            in_chroma=in_chroma, out_matrix=out_matrix,
            out_range=in_range if cfg.output_range == "preserve" else ColorRange.LIMITED,
            out_primaries=in_primaries, out_transfer=in_transfer, out_bit_depth=10, convert_gamut=False,
            hdr=True, notes=tuple(notes),
        )

    if cfg.matrix == "bt709":
        out_matrix = Matrix.BT709
    elif cfg.matrix == "preserve":
        out_matrix = in_matrix if in_matrix is not Matrix.GBR else Matrix.BT709
    else:  # auto: HD output uses BT.709; SD output keeps the source matrix
        if out_height >= 720:
            out_matrix = Matrix.BT709
        else:
            out_matrix = in_matrix if in_matrix is not Matrix.GBR else Matrix.BT601

    if out_matrix is Matrix.BT709 and in_matrix is not Matrix.BT709 and in_matrix is not Matrix.GBR:
        notes.append(f"converting matrix {in_matrix.value} -> bt709 for HD output")

    convert_gamut = False
    out_primaries = in_primaries
    if out_matrix is Matrix.BT709 and in_matrix is not Matrix.BT709:
        if cfg.convert_primaries and not same_gamut(in_primaries, Primaries.BT709):
            convert_gamut = True
            out_primaries = Primaries.BT709
            notes.append(f"converting primaries {in_primaries.value} -> bt709 in linear light")
        elif same_gamut(in_primaries, Primaries.BT709):
            out_primaries = Primaries.BT709

    out_transfer = Transfer.BT709 if in_transfer in (Transfer.BT709, Transfer.SMPTE170M) else in_transfer
    if cfg.output_range == "preserve":
        out_range = in_range
    else:
        out_range = ColorRange.FULL if cfg.output_range == "full" else ColorRange.LIMITED
    return ColorPlan(
        in_matrix=in_matrix, in_range=in_range, in_primaries=in_primaries, in_transfer=in_transfer,
        in_chroma=in_chroma, out_matrix=out_matrix, out_range=out_range, out_primaries=out_primaries,
        out_transfer=out_transfer, out_bit_depth=10 if ten_bit else 8, convert_gamut=convert_gamut,
        hdr=False, notes=tuple(notes),
    )


# ---------------------------------------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------------------------------------

_DENOISE = {"light": "hqdn3d=2:1.5:3:2.25", "medium": "hqdn3d=4:3:6:4.5", "strong": "hqdn3d=8:6:12:9"}


def plan_decode(
    video: VideoStream,
    path: Path,
    tools: FFmpegTools,
    caps: Capabilities | None,
    cfg: DecodeConfig,
    geometry: Geometry,
    *,
    deinterlace: Literal["auto", "on", "off"] | None = None,
    denoise: Literal["off", "light", "medium", "strong"] | None = None,
) -> DecodePlan:
    notes: list[str] = []
    filters: list[str] = []

    if video.pix_fmt.startswith("yuvj"):
        # Keep full-range samples untouched; we apply the range ourselves exactly once.
        filters.append("scale=in_range=full:out_range=full")

    mode = deinterlace or cfg.deinterlace
    do_deint = False
    if mode == "on":
        do_deint = True
    elif mode == "auto" and not video.is_still_image:
        if video.is_interlaced_tagged:
            do_deint = True
            notes.append(f"stream tagged interlaced ({video.field_order}); deinterlacing")
        elif video.field_order is None:
            report = detect_interlacing(path, tools)
            if report.is_interlaced:
                do_deint = True
                notes.append(
                    f"idet detected interlacing (TFF {report.tff}, BFF {report.bff}, "
                    f"progressive {report.progressive}); deinterlacing"
                )
    if do_deint:
        # send_frame keeps the frame count and timestamps (one output frame per input frame).
        filters.append(f"{cfg.deinterlace_filter}=mode=send_frame:parity=auto:deint=all")

    level = denoise or cfg.denoise
    if level != "off":
        filters.append(_DENOISE[level])
        notes.append(f"denoise pre-filter: {level}")
    if cfg.deblock:
        filters.append("deblock")

    hwaccel: str | None = None
    if cfg.hwaccel == "auto":
        hwaccel = caps.select_hwaccel(video.codec) if caps is not None else None
    elif cfg.hwaccel != "none":
        functional = caps is None or any(h.name == cfg.hwaccel and h.functional for h in caps.hwaccels)
        if functional:
            hwaccel = cfg.hwaccel
        else:
            notes.append(f"requested hwaccel {cfg.hwaccel} is not functional here; using software decode")
    if video.is_still_image:
        hwaccel = None
    notes.append(f"decode: {'hardware ' + hwaccel if hwaccel else 'software'}")

    fmt = RawVideoFormat.for_source(video.pixel_format, geometry.decoded_width, geometry.decoded_height)
    return DecodePlan(fmt=fmt, hwaccel=hwaccel, filters=tuple(filters), notes=tuple(notes))


# ---------------------------------------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------------------------------------

_MAX_WIDTH = {"h264_nvenc": 4096, "h264_amf": 4096, "h264_qsv": 4096, "hevc_nvenc": 8192, "av1_nvenc": 8192}


def plan_encoder(
    caps: Capabilities,
    *,
    codec: str,
    encoder: str,
    quality: int,
    speed: str,
    ten_bit: bool,
    color: ColorPlan,
    out_width: int,
    out_height: int,
    extra_args: tuple[str, ...] = (),
) -> EncoderPlan:
    notes: list[str] = []
    ten = ten_bit or color.out_bit_depth == 10

    def fits(name: str) -> bool:
        return max(out_width, out_height) <= _MAX_WIDTH.get(name, 16384)

    if encoder == "auto":
        candidates = [e for e in caps.functional_encoders() if e.codec == codec and fits(e.name)]
        if not candidates:
            raise NeuralFrameError(
                ErrorCode.ENCODE_FAILED,
                f"no working {codec} encoder for {out_width}x{out_height}",
                hint="run `neuralframe doctor` to see available encoders",
            )
        from neuralframe.io.hwcaps import FAMILY_ORDER

        candidates.sort(key=lambda e: FAMILY_ORDER.index(e.family))
        chosen = next((e for e in candidates if e.ten_bit), candidates[0]) if ten else candidates[0]
        if chosen is not candidates[0]:
            notes.append(f"{candidates[0].name} lacks 10-bit support; using {chosen.name}")
    else:
        check = caps.encoder(encoder)
        if check is None or not check.functional:
            working = ", ".join(e.name for e in caps.functional_encoders()) or "none"
            raise NeuralFrameError(
                ErrorCode.ENCODE_FAILED,
                f"encoder {encoder!r} is not available on this machine",
                hint=f"working encoders: {working}",
            )
        if not fits(encoder):
            raise NeuralFrameError(
                ErrorCode.ENCODE_FAILED,
                f"{encoder} cannot encode {out_width}x{out_height} (max {_MAX_WIDTH[encoder]} px)",
            )
        chosen = check

    if ten and not chosen.ten_bit:
        notes.append(f"{chosen.name} has no 10-bit mode here; encoding 8-bit")
        ten = False
    notes.append(f"encoder: {chosen.name} ({chosen.family})")
    return EncoderPlan(
        name=chosen.name, codec=chosen.codec, family=chosen.family, quality=quality, speed=speed,
        ten_bit=ten, color=dataclasses.replace(color, out_bit_depth=10 if ten else 8),
        extra_args=extra_args, notes=tuple(notes),
    )
