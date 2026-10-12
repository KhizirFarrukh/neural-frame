from __future__ import annotations

import itertools

import numpy as np
import pytest

from neuralframe.chunking.tiling import TileLayout, feather_1d, merge_tiles, split_tiles, tile_positions


@pytest.mark.parametrize(("size", "tile", "overlap"), [(100, 40, 8), (1080, 512, 32), (854, 256, 16), (64, 64, 8)])
def test_positions_cover_axis_with_fixed_tiles(size: int, tile: int, overlap: int) -> None:
    pos = tile_positions(size, tile, overlap)
    assert pos[0] == 0
    assert pos[-1] + min(tile, size) == size
    for a, b in itertools.pairwise(pos):
        assert b > a
        assert a + tile - b >= overlap - 1  # actual overlap never (meaningfully) below the request


def test_single_tile_when_frame_fits() -> None:
    layout = TileLayout.plan(90, 160, 256, 32)
    assert layout.count == 1 and (layout.tile_h, layout.tile_w) == (90, 160)
    assert TileLayout.plan(90, 160, None, 32).count == 1


def test_feather_weights_ignore_margin_and_ramp() -> None:
    w = feather_1d(64, start=10, total=200, overlap=16, scale=1)
    assert w[0] == 0.0 and w[3] == 0.0  # discarded margin (overlap // 4)
    assert np.all(np.diff(w[:16]) >= 0) and w[32] == 1.0
    border = feather_1d(64, start=0, total=64, overlap=16, scale=1)
    np.testing.assert_array_equal(border, 1.0)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_split_merge_is_partition_of_unity(scale: int) -> None:
    rng = np.random.default_rng(0)
    img = rng.random((3, 97, 131), dtype=np.float32)
    layout = TileLayout.plan(97, 131, 40, 12)
    assert layout.count > 1
    tiles = split_tiles(img, layout)
    up = np.repeat(np.repeat(tiles, scale, axis=2), scale, axis=3)  # "model" = nearest upsampling
    merged = merge_tiles(up, layout, scale)
    expected = np.repeat(np.repeat(img, scale, axis=1), scale, axis=2)
    np.testing.assert_allclose(merged, expected, atol=1e-5)
    _, _, total = layout.weight_maps(scale)
    assert total.min() > 0
