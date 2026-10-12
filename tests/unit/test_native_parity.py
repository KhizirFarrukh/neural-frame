"""The C++ module must reproduce the NumPy reference exactly (spec allows <= 1 LSB; we hold it to 0)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

from neuralframe.chunking.tiling import TileLayout, merge_tiles, split_tiles
from neuralframe.colorspace import ChromaLocation, Matrix, Primaries, gamut_matrix
from neuralframe.ops import NumpyOps, native_available
from neuralframe.ops import reference as ref

pytestmark = pytest.mark.skipif(not native_available(), reason="native module not built")

if native_available():
    from neuralframe import _native
    from neuralframe.ops.native import NativeOps


def _rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([0.5 + 0.45 * np.sin(xx / 5 + yy / 17), 0.5 + 0.45 * np.cos(yy / 6),
                     0.5 + 0.4 * np.sin((xx - yy) / 8)])
    return np.clip(base + rng.normal(0, 0.05, base.shape), -0.02, 1.02).astype(np.float32)  # incl. out-of-range


MATRICES = [Matrix.BT601, Matrix.BT709, Matrix.BT2020NC, Matrix.GBR]
LOCATIONS = [ChromaLocation.LEFT, ChromaLocation.CENTER, ChromaLocation.TOPLEFT]


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize(("bit_depth", "full_range"), [(8, False), (8, True), (10, False), (12, True)])
@pytest.mark.parametrize(("lcw", "lch"), [(1, 1), (1, 0), (0, 0)])
@pytest.mark.parametrize("loc", LOCATIONS)
def test_colour_conversion_is_bit_identical(
    matrix: Matrix, bit_depth: int, full_range: bool, lcw: int, lch: int, loc: ChromaLocation
) -> None:
    if matrix is Matrix.GBR and (lcw or lch):
        pytest.skip("planar GBR is never subsampled")
    rgb = _rgb(41, 67)
    numpy_ops, native_ops = NumpyOps(), NativeOps()
    kwargs = {"matrix": matrix, "full_range": full_range, "bit_depth": bit_depth, "chroma_location": loc}
    planes_ref = numpy_ops.rgb_to_yuv(rgb, log2_cw=lcw, log2_ch=lch, **kwargs)
    planes_nat = native_ops.rgb_to_yuv(rgb, log2_cw=lcw, log2_ch=lch, **kwargs)
    for a, b in zip(planes_ref, planes_nat, strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    back_ref = numpy_ops.yuv_to_rgb(*planes_ref, **kwargs)
    back_nat = native_ops.yuv_to_rgb(*planes_ref, **kwargs)
    np.testing.assert_array_equal(back_ref, back_nat)


@pytest.mark.parametrize(("h", "w", "oh", "ow"), [(41, 67, 160, 270), (480, 854, 1080, 1920), (90, 160, 37, 61),
                                                 (64, 64, 64, 200), (33, 17, 33, 17)])
@pytest.mark.parametrize("kernel", ["lanczos3", "bicubic", "bilinear"])
def test_resize_is_bit_identical(h: int, w: int, oh: int, ow: int, kernel: str) -> None:
    img = _rgb(h, w)
    np.testing.assert_array_equal(NumpyOps().resize(img, oh, ow, kernel), NativeOps().resize(img, oh, ow, kernel))


def test_gamut_is_bit_identical() -> None:
    rgb = np.clip(_rgb(50, 70), 0, 1)
    m = gamut_matrix(Primaries.SMPTE170M, Primaries.BT709)
    np.testing.assert_array_equal(NumpyOps().apply_gamut(rgb, m), NativeOps().apply_gamut(rgb, m))


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_native_tile_merge_matches_reference(scale: int) -> None:
    img = _rgb(97, 131)
    layout = TileLayout.plan(97, 131, 48, 16)
    tiles = np.repeat(np.repeat(split_tiles(img, layout), scale, axis=2), scale, axis=3) * 0.9
    expected = merge_tiles(tiles, layout, scale)
    wy, wx, total = layout.weight_maps(scale)
    canvas = np.zeros((3, 97 * scale, 131 * scale), dtype=np.float32)
    for (iy, ix, y, x), tile in zip(layout.tiles, tiles, strict=True):
        _native.merge_tile(canvas, np.ascontiguousarray(tile), y * scale, x * scale, wy[iy], wx[ix])
    _native.normalize(canvas, total)
    np.testing.assert_array_equal(canvas, expected)


def test_blend_matches_reference() -> None:
    a, b = _rgb(30, 40, 1), _rgb(30, 40, 2)
    for w in (0.0, 0.3, 0.5, 1.0):
        np.testing.assert_array_equal(_native.blend(a, b, w), ref.temporal_blend(a, b, w))


def test_ring_buffer_from_python() -> None:
    ring = _native.RingBuffer(4, 1000)
    assert ring.slot_bytes % 4096 == 0 and ring.data_ptr % 4096 == 0
    received: list[int] = []

    def consume() -> None:
        while True:
            slot, tag = ring.acquire_read()
            if slot < 0:
                return
            assert int(ring.view(slot)[:4].view(np.int32)[0]) == tag
            received.append(tag)
            ring.release_read(slot)

    t = threading.Thread(target=consume)
    t.start()
    for i in range(200):
        slot = ring.acquire_write()
        ring.view(slot)[:4] = np.array([i], dtype=np.int32).view(np.uint8)
        ring.commit_write(slot, i)
    ring.close()
    t.join(timeout=10)
    assert received == list(range(200))


def test_thread_count_does_not_change_results() -> None:
    rgb = _rgb(120, 200)
    before = _native.get_threads()
    try:
        _native.set_threads(1)
        single = NativeOps().resize(rgb, 300, 500, "lanczos3")
        _native.set_threads(max(before, 4))
        multi = NativeOps().resize(rgb, 300, 500, "lanczos3")
    finally:
        _native.set_threads(before)
    np.testing.assert_array_equal(single, multi)
