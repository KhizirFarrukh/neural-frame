"""Raw planar frame layouts exchanged with FFmpeg over pipes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from neuralframe.io.media import PixelFormat

RawArray: TypeAlias = npt.NDArray[np.unsignedinteger[Any]]
"""Flat or planar sample buffer: uint8 at 8 bits, uint16 (LSB-aligned) above."""

_SUBSAMPLING_NAMES = {(1, 1): "420", (1, 0): "422", (0, 0): "444"}


def _depth_bucket(depth: int) -> int:
    for bucket in (8, 10, 12, 16):
        if depth <= bucket:
            return bucket
    return 16


@dataclass(frozen=True)
class RawVideoFormat:
    """Planar YUV (or GBR) layout; samples are uint8 at 8 bits, LSB-aligned little-endian uint16 above."""

    width: int
    height: int
    bit_depth: int = 8
    log2_cw: int = 1
    log2_ch: int = 1
    rgb: bool = False

    def __post_init__(self) -> None:
        if (self.log2_cw, self.log2_ch) not in _SUBSAMPLING_NAMES:
            raise ValueError(f"unsupported chroma subsampling {(self.log2_cw, self.log2_ch)}")
        if self.rgb and (self.log2_cw or self.log2_ch):
            raise ValueError("planar RGB cannot be chroma-subsampled")
        if self.bit_depth not in (8, 10, 12, 16):
            raise ValueError(f"unsupported bit depth {self.bit_depth}")

    # ---- construction --------------------------------------------------------------------------------

    @classmethod
    def for_source(cls, pix: PixelFormat, width: int, height: int) -> RawVideoFormat:
        """Pick the pipe format closest to the source so FFmpeg does no colour maths for us."""
        depth = _depth_bucket(pix.bit_depth)
        if pix.is_rgb:
            return cls(width, height, depth, 0, 0, rgb=True)
        sub = (pix.log2_chroma_w, pix.log2_chroma_h)
        if sub not in _SUBSAMPLING_NAMES or pix.nb_components < 3:
            sub = (1, 1)  # 4:1:1, 4:4:0, gray, … are converted to 4:2:0
        return cls(width, height, depth, sub[0], sub[1])

    @classmethod
    def yuv420(cls, width: int, height: int, bit_depth: int = 8) -> RawVideoFormat:
        return cls(width, height, bit_depth, 1, 1)

    # ---- properties ----------------------------------------------------------------------------------

    @property
    def pix_fmt(self) -> str:
        suffix = "" if self.bit_depth == 8 else f"{self.bit_depth}le"
        if self.rgb:
            return f"gbrp{suffix}"
        return f"yuv{_SUBSAMPLING_NAMES[(self.log2_cw, self.log2_ch)]}p{suffix}"

    @property
    def dtype(self) -> type[np.uint8] | type[np.uint16]:
        return np.uint8 if self.bit_depth == 8 else np.uint16

    @property
    def bytes_per_sample(self) -> int:
        return 1 if self.bit_depth == 8 else 2

    @property
    def chroma_width(self) -> int:
        return (self.width + (1 << self.log2_cw) - 1) >> self.log2_cw

    @property
    def chroma_height(self) -> int:
        return (self.height + (1 << self.log2_ch) - 1) >> self.log2_ch

    @property
    def plane_shapes(self) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
        chroma = (self.chroma_height, self.chroma_width)
        return (self.height, self.width), chroma, chroma

    @property
    def frame_samples(self) -> int:
        return self.width * self.height + 2 * self.chroma_width * self.chroma_height

    @property
    def frame_bytes(self) -> int:
        return self.frame_samples * self.bytes_per_sample

    def with_size(self, width: int, height: int) -> RawVideoFormat:
        return RawVideoFormat(width, height, self.bit_depth, self.log2_cw, self.log2_ch, self.rgb)

    # ---- buffers -------------------------------------------------------------------------------------

    def empty(self) -> RawArray:
        return cast(RawArray, np.empty(self.frame_samples, dtype=self.dtype))

    def split(
        self, buf: RawArray
    ) -> tuple[RawArray, RawArray, RawArray]:
        """Views of the three planes inside a flat frame buffer (no copy)."""
        if buf.size != self.frame_samples:
            raise ValueError(f"buffer has {buf.size} samples, expected {self.frame_samples}")
        (yh, yw), (ch, cw), _ = self.plane_shapes
        y_end = yh * yw
        c_size = ch * cw
        return (
            buf[:y_end].reshape(yh, yw),
            buf[y_end : y_end + c_size].reshape(ch, cw),
            buf[y_end + c_size :].reshape(ch, cw),
        )

    def from_bytes(self, data: bytes | memoryview) -> RawArray:
        arr = cast(RawArray, np.frombuffer(data, dtype=np.dtype(self.dtype).newbyteorder("<")))
        if arr.size != self.frame_samples:
            raise ValueError(f"frame has {arr.size} samples, expected {self.frame_samples} ({self})")
        return arr

    def join(
        self, planes: tuple[RawArray, RawArray, RawArray]
    ) -> RawArray:
        out = self.empty()
        dst = self.split(out)
        for d, p in zip(dst, planes, strict=True):
            d[...] = p
        return out
