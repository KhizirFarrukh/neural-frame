"""FrameOps backed by the C++ extension ``neuralframe._native``.

Parameters and tables (range constants, kernel weights, transfer curves) come from
:mod:`neuralframe.ops.reference`, so both implementations share a single source of truth and produce
bit-identical results; the native code is multi-threaded and runs without the GIL.
"""

from __future__ import annotations

from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from neuralframe import _native
from neuralframe.colorspace import KR_KB, ChromaLocation, Matrix, chroma_siting
from neuralframe.ops import reference

F32 = npt.NDArray[np.float32]
Plane = npt.NDArray[np.unsignedinteger[Any]]


class _MatrixArgs(TypedDict):
    kr: float
    kb: float
    y_offset: float
    y_scale: float
    c_offset: float
    c_scale: float
    bit_depth: int
    gbr: bool


def _matrix_args(matrix: Matrix, full_range: bool, bit_depth: int) -> _MatrixArgs:
    kr, kb = KR_KB.get(matrix, (0.2126, 0.0722))
    yo, ys, co, cs = reference.range_params(bit_depth, full_range)
    return {
        "kr": kr, "kb": kb, "y_offset": yo, "y_scale": ys, "c_offset": co, "c_scale": cs,
        "bit_depth": bit_depth, "gbr": matrix is Matrix.GBR,
    }


class NativeOps:
    name = "native"

    def yuv_to_rgb(
        self, y: Plane, u: Plane, v: Plane, *, matrix: Matrix, full_range: bool, bit_depth: int,
        chroma_location: ChromaLocation,
    ) -> F32:
        h_cosited, v_cosited = chroma_siting(chroma_location)
        planes = [np.ascontiguousarray(p) for p in (y, u, v)]
        out: F32 = _native.yuv_to_rgb(
            *planes, **_matrix_args(matrix, full_range, bit_depth), h_cosited=h_cosited, v_cosited=v_cosited
        )
        return out

    def rgb_to_yuv(
        self, rgb: F32, *, matrix: Matrix, full_range: bool, bit_depth: int, log2_cw: int, log2_ch: int,
        chroma_location: ChromaLocation,
    ) -> tuple[Plane, Plane, Plane]:
        h_cosited, v_cosited = chroma_siting(chroma_location)
        y, u, v = _native.rgb_to_yuv(
            np.ascontiguousarray(rgb, dtype=np.float32), **_matrix_args(matrix, full_range, bit_depth),
            log2_cw=log2_cw, log2_ch=log2_ch, h_cosited=h_cosited, v_cosited=v_cosited,
        )
        return y, u, v

    def resize(self, img: F32, out_h: int, out_w: int, kernel: str) -> F32:
        _, h, w = img.shape
        if (h, w) == (out_h, out_w):
            return img
        idx_x, w_x = reference.resample_weights(w, out_w, kernel)
        idx_y, w_y = reference.resample_weights(h, out_h, kernel)
        out: F32 = _native.resize(np.ascontiguousarray(img, dtype=np.float32), out_h, out_w, idx_x, w_x, idx_y, w_y)
        return out

    def apply_gamut(self, rgb: F32, matrix3: F32) -> F32:
        to_linear, to_code = reference.transfer_luts()
        out: F32 = _native.apply_gamut(
            np.ascontiguousarray(rgb, dtype=np.float32), np.ascontiguousarray(matrix3, dtype=np.float32),
            to_linear, to_code,
        )
        return out
