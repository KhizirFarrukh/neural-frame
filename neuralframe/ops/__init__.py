"""Frame operations behind one interface, with interchangeable implementations.

* ``numpy``  — :mod:`neuralframe.ops.reference`, the ground truth used by tests.
* ``native`` — :mod:`neuralframe._native` (C++17, multi-threaded, releases the GIL).

:func:`get_ops` picks the native implementation when it is importable unless told otherwise.
"""

from __future__ import annotations

import importlib
from typing import Any, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt

from neuralframe.colorspace import ChromaLocation, Matrix
from neuralframe.ops import reference

F32 = npt.NDArray[np.float32]
Plane = npt.NDArray[np.unsignedinteger[Any]]
OpsKind = Literal["auto", "native", "numpy"]


class FrameOps(Protocol):
    name: str

    def yuv_to_rgb(
        self, y: Plane, u: Plane, v: Plane, *, matrix: Matrix, full_range: bool, bit_depth: int,
        chroma_location: ChromaLocation,
    ) -> F32: ...

    def rgb_to_yuv(
        self, rgb: F32, *, matrix: Matrix, full_range: bool, bit_depth: int, log2_cw: int, log2_ch: int,
        chroma_location: ChromaLocation,
    ) -> tuple[Plane, Plane, Plane]: ...

    def resize(self, img: F32, out_h: int, out_w: int, kernel: str) -> F32: ...

    def apply_gamut(self, rgb: F32, matrix3: F32) -> F32: ...


class NumpyOps:
    name = "numpy"

    def yuv_to_rgb(
        self, y: Plane, u: Plane, v: Plane, *, matrix: Matrix, full_range: bool, bit_depth: int,
        chroma_location: ChromaLocation,
    ) -> F32:
        return reference.yuv_to_rgb(
            y, u, v, matrix=matrix, full_range=full_range, bit_depth=bit_depth, chroma_location=chroma_location
        )

    def rgb_to_yuv(
        self, rgb: F32, *, matrix: Matrix, full_range: bool, bit_depth: int, log2_cw: int, log2_ch: int,
        chroma_location: ChromaLocation,
    ) -> tuple[Plane, Plane, Plane]:
        return reference.rgb_to_yuv(
            rgb, matrix=matrix, full_range=full_range, bit_depth=bit_depth, log2_cw=log2_cw,
            log2_ch=log2_ch, chroma_location=chroma_location,
        )

    def resize(self, img: F32, out_h: int, out_w: int, kernel: str) -> F32:
        return reference.resize(img, out_h, out_w, kernel)

    def apply_gamut(self, rgb: F32, matrix3: F32) -> F32:
        return reference.apply_gamut(rgb, matrix3)


def native_available() -> bool:
    try:
        native = importlib.import_module("neuralframe._native")
    except ImportError:
        return False
    return hasattr(native, "yuv_to_rgb")


def get_ops(kind: OpsKind = "auto") -> FrameOps:
    if kind in ("auto", "native") and native_available():
        module = importlib.import_module("neuralframe.ops.native")
        return cast(FrameOps, module.NativeOps())
    if kind == "native":
        from neuralframe.errors import ErrorCode, NeuralFrameError

        raise NeuralFrameError(
            ErrorCode.BACKEND_UNAVAILABLE,
            "native ops requested but neuralframe._native is not built",
            hint="reinstall with a C++17 compiler available (pip install -e .)",
        )
    return NumpyOps()
