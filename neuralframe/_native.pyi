"""Type stub for the compiled ``neuralframe._native`` extension (see native/src/bindings.cpp)."""

from typing import Any

import numpy as np
import numpy.typing as npt

__version__: str

def build_info() -> str: ...
def set_threads(n: int) -> None: ...
def get_threads() -> int: ...
def yuv_to_rgb(
    y: npt.NDArray[Any],
    u: npt.NDArray[Any],
    v: npt.NDArray[Any],
    *,
    kr: float,
    kb: float,
    y_offset: float,
    y_scale: float,
    c_offset: float,
    c_scale: float,
    bit_depth: int,
    gbr: bool,
    h_cosited: bool,
    v_cosited: bool,
) -> npt.NDArray[np.float32]: ...
def rgb_to_yuv(
    rgb: npt.NDArray[np.float32],
    *,
    kr: float,
    kb: float,
    y_offset: float,
    y_scale: float,
    c_offset: float,
    c_scale: float,
    bit_depth: int,
    gbr: bool,
    log2_cw: int,
    log2_ch: int,
    h_cosited: bool,
    v_cosited: bool,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]: ...
def resize(
    img: npt.NDArray[np.float32],
    out_h: int,
    out_w: int,
    idx_x: npt.NDArray[np.int32],
    w_x: npt.NDArray[np.float32],
    idx_y: npt.NDArray[np.int32],
    w_y: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]: ...
def apply_gamut(
    rgb: npt.NDArray[np.float32],
    matrix: npt.NDArray[np.float32],
    to_linear: npt.NDArray[np.float32],
    to_code: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]: ...
def merge_tile(
    canvas: npt.NDArray[np.float32],
    tile: npt.NDArray[np.float32],
    y0: int,
    x0: int,
    wy: npt.NDArray[np.float32],
    wx: npt.NDArray[np.float32],
) -> None: ...
def normalize(canvas: npt.NDArray[np.float32], total: npt.NDArray[np.float32]) -> None: ...
def blend(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32], weight_a: float) -> npt.NDArray[np.float32]: ...

class RingBuffer:
    def __init__(self, slots: int, slot_bytes: int) -> None: ...
    def acquire_write(self, timeout_ms: int = -1) -> int: ...
    def commit_write(self, slot: int, tag: int = 0) -> None: ...
    def acquire_read(self, timeout_ms: int = -1) -> tuple[int, int]: ...
    def release_read(self, slot: int) -> None: ...
    def close(self) -> None: ...
    def view(self, slot: int) -> npt.NDArray[np.uint8]: ...
    @property
    def slots(self) -> int: ...
    @property
    def slot_bytes(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    @property
    def data_ptr(self) -> int: ...
    @property
    def ready(self) -> int: ...
    @property
    def closed(self) -> bool: ...
