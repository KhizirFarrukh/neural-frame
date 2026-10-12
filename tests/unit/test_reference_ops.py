from __future__ import annotations

import numpy as np
import pytest

from neuralframe.colorspace import ChromaLocation, Matrix, Primaries, gamut_matrix
from neuralframe.ops import reference as ref

MATRICES = [Matrix.BT601, Matrix.BT709, Matrix.BT2020NC, Matrix.SMPTE240M]


def smooth_rgb(h: int = 36, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        0.5 + 0.4 * np.sin(xx / 9.0 + yy / 13.0),
        0.5 + 0.4 * np.cos(xx / 11.0 - yy / 7.0),
        0.5 + 0.3 * np.sin((xx + yy) / 15.0),
    ])
    return np.clip(base + rng.normal(0, 0.002, base.shape), 0, 1).astype(np.float32)


@pytest.mark.parametrize("matrix", MATRICES)
@pytest.mark.parametrize("full_range", [False, True])
@pytest.mark.parametrize("bit_depth", [8, 10])
def test_444_round_trip_is_within_one_code(matrix: Matrix, full_range: bool, bit_depth: int) -> None:
    rgb = smooth_rgb()
    planes = ref.rgb_to_yuv(rgb, matrix=matrix, full_range=full_range, bit_depth=bit_depth, log2_cw=0, log2_ch=0)
    back = ref.yuv_to_rgb(*planes, matrix=matrix, full_range=full_range, bit_depth=bit_depth)
    again = ref.rgb_to_yuv(back, matrix=matrix, full_range=full_range, bit_depth=bit_depth, log2_cw=0, log2_ch=0)
    for a, b in zip(planes, again, strict=True):
        assert np.max(np.abs(a.astype(np.int32) - b.astype(np.int32))) <= 1


@pytest.mark.parametrize("location", [ChromaLocation.LEFT, ChromaLocation.CENTER, ChromaLocation.TOPLEFT])
def test_420_round_trip_on_smooth_content(location: ChromaLocation) -> None:
    rgb = smooth_rgb(37, 63)  # odd sizes exercise edge replication
    y, u, v = ref.rgb_to_yuv(
        rgb, matrix=Matrix.BT709, full_range=False, bit_depth=8, log2_cw=1, log2_ch=1, chroma_location=location
    )
    assert u.shape == (19, 32)
    back = ref.yuv_to_rgb(y, u, v, matrix=Matrix.BT709, full_range=False, bit_depth=8, chroma_location=location)
    assert back.shape == rgb.shape
    assert np.mean(np.abs(back - rgb)) < 2.0 / 255


def test_reference_levels() -> None:
    def to_rgb(y: int, u: int, v: int, *, full: bool = False, depth: int = 8) -> np.ndarray:
        dtype = np.uint8 if depth == 8 else np.uint16
        planes = [np.full((2, 2), c, dtype) for c in (y, u, v)]
        return ref.yuv_to_rgb(*planes, matrix=Matrix.BT709, full_range=full, bit_depth=depth)[:, 0, 0]

    np.testing.assert_allclose(to_rgb(235, 128, 128), [1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(to_rgb(16, 128, 128), [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(to_rgb(255, 128, 128, full=True), [1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(to_rgb(940, 512, 512, depth=10), [1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(to_rgb(64, 512, 512, depth=10), [0, 0, 0], atol=1e-6)


def test_matrix_differences_for_pure_red() -> None:
    red = np.zeros((3, 2, 2), np.float32)
    red[0] = 1.0
    y601 = ref.rgb_to_yuv(red, matrix=Matrix.BT601, full_range=False, bit_depth=8, log2_cw=0, log2_ch=0)[0]
    y709 = ref.rgb_to_yuv(red, matrix=Matrix.BT709, full_range=False, bit_depth=8, log2_cw=0, log2_ch=0)[0]
    assert int(y601[0, 0]) == 81  # round(16 + 219 * 0.299) = round(81.48)
    assert int(y709[0, 0]) == 63  # round(16 + 219 * 0.2126) = round(62.56)


def test_gbr_planes_map_to_rgb() -> None:
    g, b, r = (np.full((2, 3), val, np.uint8) for val in (10, 20, 30))
    rgb = ref.yuv_to_rgb(g, b, r, matrix=Matrix.GBR, full_range=True, bit_depth=8)
    np.testing.assert_allclose(rgb[:, 0, 0], [30 / 255, 10 / 255, 20 / 255], atol=1e-6)


def test_chroma_upsampling_preserves_cosited_samples_and_constants() -> None:
    c = np.arange(12, dtype=np.float32).reshape(3, 4)
    up = ref.upsample_chroma(c, 6, 8, 1, 1, ChromaLocation.TOPLEFT)
    np.testing.assert_array_equal(up[0::2, 0::2], c)
    flat = np.full((3, 4), 0.25, np.float32)
    for loc in ChromaLocation:
        np.testing.assert_allclose(ref.upsample_chroma(flat, 5, 7, 1, 1, loc), 0.25)
        np.testing.assert_allclose(ref.downsample_chroma(np.full((5, 7), 0.25, np.float32), 1, 1, loc), 0.25)


@pytest.mark.parametrize("kernel", ["lanczos3", "bicubic", "bilinear"])
@pytest.mark.parametrize(("n_in", "n_out"), [(10, 37), (37, 10), (64, 64), (5, 3), (854, 1920)])
def test_resample_weights_are_normalised(kernel: str, n_in: int, n_out: int) -> None:
    idx, w = ref.resample_weights(n_in, n_out, kernel)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert idx.min() >= 0 and idx.max() < n_in


def test_resize_identity_and_constant() -> None:
    img = smooth_rgb()
    np.testing.assert_array_equal(ref.resize(img, 36, 64), img)
    const = np.full((3, 20, 30), 0.3, np.float32)
    np.testing.assert_allclose(ref.resize(const, 47, 71), 0.3, atol=1e-6)
    np.testing.assert_allclose(ref.resize(const, 7, 9), 0.3, atol=1e-6)


def test_bicubic_reproduces_linear_ramp_in_the_interior() -> None:
    ramp = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (1, 8, 1))
    up = ref.resize(ramp, 8, 128, "bicubic")
    centres = (np.arange(128) + 0.5) * 32 / 128 - 0.5
    expected = np.clip(centres / 31, 0, 1)
    np.testing.assert_allclose(up[0, 4, 8:-8], expected[8:-8], atol=1e-4)


def test_downscale_is_antialiased() -> None:
    checker = (np.indices((64, 64)).sum(axis=0) % 2).astype(np.float32)[None]
    small = ref.resize(checker, 16, 16, "lanczos3")
    assert np.all(np.abs(small - 0.5) < 0.05)


def test_gamut_matrix_preserves_white() -> None:
    m = gamut_matrix(Primaries.SMPTE170M, Primaries.BT709)
    np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(gamut_matrix(Primaries.BT709, Primaries.BT709), np.eye(3), atol=1e-6)
    grey = np.full((3, 2, 2), 0.5, np.float32)
    np.testing.assert_allclose(ref.apply_gamut(grey, m), 0.5, atol=1e-4)
