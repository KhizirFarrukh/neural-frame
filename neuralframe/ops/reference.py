"""Pure-NumPy reference implementations of every hot-path frame operation.

These are the ground truth: the native C++ and Torch implementations are tested against them. They favour
clarity over speed, but are vectorised enough to run the CPU pipeline on small clips.

Conventions
-----------
* RGB frames are planar ``float32`` arrays shaped ``(3, H, W)`` with non-linear (gamma-encoded) values in
  ``[0, 1]``.
* YUV frames are three planes; samples are ``uint8`` for 8-bit and ``uint16`` (LSB-aligned) above 8 bits.
* Chroma subsampling is expressed as ``log2`` factors: 4:2:0 → (1, 1), 4:2:2 → (1, 0), 4:4:4 → (0, 0).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

from neuralframe.colorspace import KR_KB, ChromaLocation, Matrix, chroma_siting

F32 = npt.NDArray[np.float32]
Plane = npt.NDArray[np.unsignedinteger[Any]]
ResizeKernel = Literal["lanczos3", "bicubic", "bilinear"]


# ---------------------------------------------------------------------------------------------------------
# Chroma resampling (siting-aware, factor 2 per axis)
# ---------------------------------------------------------------------------------------------------------


def _f32(x: Any) -> F32:
    return np.asarray(x, dtype=np.float32)


def _clamped(n: int, offset: int, length: int) -> npt.NDArray[np.intp]:
    return np.clip(np.arange(n) + offset, 0, length - 1)


def _up2(c: F32, n_out: int, axis: int, cosited: bool) -> F32:
    m = c.shape[axis]
    nxt = np.take(c, _clamped(m, 1, m), axis=axis)
    if cosited:
        even = c
        odd = 0.5 * (c + nxt)
    else:
        prev = np.take(c, _clamped(m, -1, m), axis=axis)
        even = 0.75 * c + 0.25 * prev
        odd = 0.75 * c + 0.25 * nxt
    out = np.stack([even, odd], axis=axis + 1)
    shape = list(c.shape)
    shape[axis] = 2 * m
    out = out.reshape(shape)
    return np.ascontiguousarray(np.take(out, np.arange(n_out), axis=axis), dtype=np.float32)


def _down2(x: F32, axis: int, cosited: bool) -> F32:
    n = x.shape[axis]
    m = (n + 1) // 2
    base = np.arange(m) * 2
    centre = np.take(x, np.clip(base, 0, n - 1), axis=axis)
    right = np.take(x, np.clip(base + 1, 0, n - 1), axis=axis)
    if cosited:
        left = np.take(x, np.clip(base - 1, 0, n - 1), axis=axis)
        out = 0.25 * left + 0.5 * centre + 0.25 * right
    else:
        out = 0.5 * (centre + right)
    return np.ascontiguousarray(out, dtype=np.float32)


def upsample_chroma(
    c: F32, height: int, width: int, log2_cw: int, log2_ch: int, location: ChromaLocation
) -> F32:
    h_cosited, v_cosited = chroma_siting(location)
    out = c
    if log2_cw == 1:
        out = _up2(out, width, axis=1, cosited=h_cosited)
    if log2_ch == 1:
        out = _up2(out, height, axis=0, cosited=v_cosited)
    return out


def downsample_chroma(c: F32, log2_cw: int, log2_ch: int, location: ChromaLocation) -> F32:
    h_cosited, v_cosited = chroma_siting(location)
    out = c
    if log2_cw == 1:
        out = _down2(out, axis=1, cosited=h_cosited)
    if log2_ch == 1:
        out = _down2(out, axis=0, cosited=v_cosited)
    return out


# ---------------------------------------------------------------------------------------------------------
# YCbCr <-> RGB
# ---------------------------------------------------------------------------------------------------------


def range_params(bit_depth: int, full_range: bool) -> tuple[float, float, float, float]:
    """(y_offset, y_scale, c_offset, c_scale) mapping code values to Y' in [0,1] and Cb/Cr in [-.5,.5]."""
    s = float(1 << (bit_depth - 8))
    if full_range:
        peak = float((1 << bit_depth) - 1)
        return 0.0, peak, float(1 << (bit_depth - 1)), peak
    return 16.0 * s, 219.0 * s, 128.0 * s, 224.0 * s


def yuv_to_rgb(
    y: Plane,
    u: Plane,
    v: Plane,
    *,
    matrix: Matrix,
    full_range: bool,
    bit_depth: int,
    chroma_location: ChromaLocation = ChromaLocation.LEFT,
) -> F32:
    """Convert planar Y'CbCr to planar R'G'B' float32 in [0, 1]."""
    height, width = y.shape
    if matrix is Matrix.GBR:
        # Planar GBR (gbrp*): the three full-resolution planes are G, B, R at full range.
        peak = np.float32((1 << bit_depth) - 1)
        rgb = np.stack([v.astype(np.float32), y.astype(np.float32), u.astype(np.float32)]) / peak
        return _f32(np.clip(rgb, 0.0, 1.0))

    log2_cw = 1 if u.shape[1] < width else 0
    log2_ch = 1 if u.shape[0] < height else 0
    yo, ys, co, cs = range_params(bit_depth, full_range)

    yf = (y.astype(np.float32) - np.float32(yo)) / np.float32(ys)
    cb = (u.astype(np.float32) - np.float32(co)) / np.float32(cs)
    cr = (v.astype(np.float32) - np.float32(co)) / np.float32(cs)
    cb = upsample_chroma(cb, height, width, log2_cw, log2_ch, chroma_location)
    cr = upsample_chroma(cr, height, width, log2_cw, log2_ch, chroma_location)

    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    r = yf + np.float32(2.0 * (1.0 - kr)) * cr
    b = yf + np.float32(2.0 * (1.0 - kb)) * cb
    g = yf - np.float32(2.0 * kr * (1.0 - kr) / kg) * cr - np.float32(2.0 * kb * (1.0 - kb) / kg) * cb
    return _f32(np.clip(np.stack([r, g, b]), 0.0, 1.0))


def rgb_to_yuv(
    rgb: F32,
    *,
    matrix: Matrix,
    full_range: bool,
    bit_depth: int,
    log2_cw: int,
    log2_ch: int,
    chroma_location: ChromaLocation = ChromaLocation.LEFT,
) -> tuple[Plane, Plane, Plane]:
    """Convert planar R'G'B' float32 (clipped to [0, 1]) to quantised planar Y'CbCr."""
    clipped = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    r, g, b = clipped[0], clipped[1], clipped[2]
    dtype = np.uint8 if bit_depth <= 8 else np.uint16
    peak = (1 << bit_depth) - 1

    def quantise(x: F32, offset: float, scale: float) -> Plane:
        q = np.floor(x * np.float32(scale) + np.float32(offset) + np.float32(0.5))
        return cast(Plane, np.clip(q, 0, peak).astype(dtype))

    if matrix is Matrix.GBR:
        return quantise(g, 0.0, peak), quantise(b, 0.0, peak), quantise(r, 0.0, peak)

    yo, ys, co, cs = range_params(bit_depth, full_range)
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    yf = np.float32(kr) * r + np.float32(kg) * g + np.float32(kb) * b
    cb = (b - yf) / np.float32(2.0 * (1.0 - kb))
    cr = (r - yf) / np.float32(2.0 * (1.0 - kr))
    cb = downsample_chroma(cb.astype(np.float32), log2_cw, log2_ch, chroma_location)
    cr = downsample_chroma(cr.astype(np.float32), log2_cw, log2_ch, chroma_location)
    return quantise(yf, yo, ys), quantise(cb, co, cs), quantise(cr, co, cs)


# ---------------------------------------------------------------------------------------------------------
# Transfer functions and gamut conversion
# ---------------------------------------------------------------------------------------------------------


# Transcendental functions are evaluated with scalar ``math`` into cached tables. NumPy's SIMD
# ``sin``/``power`` can differ by an ULP between vector and scalar paths depending on memory alignment,
# which made outputs non-reproducible run to run; tables keep every run (and the native module, which
# consumes the same tables) bit-identical.

_LUT_SIZE = 1 << 16


@functools.cache
def transfer_luts() -> tuple[F32, F32]:
    """(BT.709 code value -> linear, linear -> BT.709 code value) sampled on a uniform [0, 1] grid."""
    n = _LUT_SIZE - 1
    to_linear = [
        (x / 4.5) if x < 0.081 else ((x + 0.099) / 1.099) ** (1.0 / 0.45) for x in (i / n for i in range(n + 1))
    ]
    to_code = [(4.5 * x) if x < 0.018 else 1.099 * x**0.45 - 0.099 for x in (i / n for i in range(n + 1))]
    lin = np.asarray(to_linear, dtype=np.float32)
    enc = np.asarray(to_code, dtype=np.float32)
    lin.setflags(write=False)
    enc.setflags(write=False)
    return lin, enc


def apply_lut(x: F32, lut: F32) -> F32:
    """Piecewise-linear lookup of ``x`` (clipped to [0, 1]) in a uniform table."""
    pos = np.clip(x, 0.0, 1.0).astype(np.float32) * np.float32(len(lut) - 1)
    i0 = np.minimum(pos.astype(np.int32), len(lut) - 2)
    frac = pos - i0.astype(np.float32)
    return _f32(lut[i0] + (lut[i0 + 1] - lut[i0]) * frac)


def bt709_to_linear(x: F32) -> F32:
    return apply_lut(x, transfer_luts()[0])


def linear_to_bt709(x: F32) -> F32:
    return apply_lut(x, transfer_luts()[1])


def apply_gamut(rgb: F32, matrix3: npt.NDArray[np.float32]) -> F32:
    """Convert gamma-encoded RGB between gamuts in linear light (BT.709 OETF on both sides)."""
    lin = bt709_to_linear(rgb)
    m = matrix3.astype(np.float32)
    out = np.stack([m[c, 0] * lin[0] + m[c, 1] * lin[1] + m[c, 2] * lin[2] for c in range(3)])
    return linear_to_bt709(_f32(out))


# ---------------------------------------------------------------------------------------------------------
# Separable resampling
# ---------------------------------------------------------------------------------------------------------


def _sinc(x: float) -> float:
    if x == 0.0:
        return 1.0
    px = math.pi * x
    return math.sin(px) / px


def _lanczos3(x: float) -> float:
    return _sinc(x) * _sinc(x / 3.0) if abs(x) < 3.0 else 0.0


def _bicubic(x: float) -> float:
    a = -0.5  # Keys / Catmull-Rom, the common "bicubic" baseline
    ax = abs(x)
    if ax <= 1.0:
        return ((a + 2.0) * ax - (a + 3.0)) * ax * ax + 1.0
    if ax < 2.0:
        return ((a * ax - 5.0 * a) * ax + 8.0 * a) * ax - 4.0 * a
    return 0.0


def _bilinear(x: float) -> float:
    return max(0.0, 1.0 - abs(x))


KERNELS: dict[str, tuple[Callable[[float], float], float]] = {
    "lanczos3": (_lanczos3, 3.0),
    "bicubic": (_bicubic, 2.0),
    "bilinear": (_bilinear, 1.0),
}


@functools.lru_cache(maxsize=128)
def resample_weights(
    in_size: int, out_size: int, kernel: str = "lanczos3"
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """Per-output-pixel source indices and normalised weights (antialiased when downscaling).

    Pixel centres sit at ``i + 0.5``; the kernel is truncated at the image border and renormalised.
    Rows are padded to a fixed tap count with zero weights. Returned arrays are cached and read-only.
    """
    fn, support = KERNELS[kernel]
    scale = in_size / out_size
    filter_scale = max(scale, 1.0)
    radius = support * filter_scale
    taps = math.ceil(radius) * 2 + 1
    idx = np.zeros((out_size, taps), dtype=np.int32)
    weights = np.zeros((out_size, taps), dtype=np.float32)
    for i in range(out_size):
        centre = (i + 0.5) * scale
        lo = max(math.floor(centre - radius + 0.5), 0)
        hi = min(math.floor(centre + radius + 0.5), in_size)
        js = list(range(lo, hi))[:taps]
        w = [fn((j + 0.5 - centre) / filter_scale) for j in js]
        total = math.fsum(w)
        if total == 0.0:
            nearest = min(range(len(js)), key=lambda k: abs(js[k] + 0.5 - centre))
            w = [1.0 if k == nearest else 0.0 for k in range(len(js))]
            total = 1.0
        for k, (j, wk) in enumerate(zip(js, w, strict=True)):
            idx[i, k] = j
            weights[i, k] = wk / total
        idx[i, len(js):] = js[-1] if js else 0
    idx.setflags(write=False)
    weights.setflags(write=False)
    return idx, weights


def _resample_axis(img: F32, out_size: int, axis: int, kernel: str) -> F32:
    in_size = img.shape[axis]
    if in_size == out_size:
        return img
    idx, w = resample_weights(in_size, out_size, kernel)
    shape = [1] * img.ndim
    shape[axis] = out_size
    out = np.zeros([*img.shape[:axis], out_size, *img.shape[axis + 1 :]], dtype=np.float32)
    for k in range(idx.shape[1]):
        out += np.take(img, idx[:, k], axis=axis) * w[:, k].reshape(shape)
    return out


def resize(img: F32, out_h: int, out_w: int, kernel: str = "lanczos3") -> F32:
    """Resize a planar ``(C, H, W)`` float32 image (horizontal pass first, then vertical)."""
    out = _resample_axis(img.astype(np.float32, copy=False), out_w, axis=img.ndim - 1, kernel=kernel)
    return _resample_axis(out, out_h, axis=img.ndim - 2, kernel=kernel)
