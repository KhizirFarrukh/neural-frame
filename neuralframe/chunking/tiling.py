"""Spatial tiling with fixed-size, overlapping tiles and feathered blending (NumPy reference).

Design
------
* Every tile has the same shape. Edge tiles are shifted inward instead of shrunk, so batches are uniform
  and TensorRT needs a single optimisation profile.
* Blending weights are built per axis and combined as an outer product. On an *interior* edge a tile
  ignores its outermost ``margin`` pixels (where convolutions saw padding instead of real context), then
  ramps linearly to full weight; image borders get full weight. Overlapping contributions are normalised
  by the summed weights, so the result is a partition of unity whatever the actual overlap is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


def tile_positions(size: int, tile: int, overlap: int) -> tuple[int, ...]:
    """Origins of equally sized tiles covering ``[0, size)`` with at least ``overlap`` px of overlap."""
    if tile >= size:
        return (0,)
    step = max(tile - overlap, 1)
    count = math.ceil((size - overlap) / step)
    if count <= 1:
        return (0,)
    span = size - tile
    return tuple(round(i * span / (count - 1)) for i in range(count))


def feather_1d(length: int, start: int, total: int, overlap: int, scale: int) -> F32:
    """Weights along one axis for a tile of ``length`` input px at ``start`` in an axis of ``total`` px.

    Returned at output resolution (``length * scale`` samples).
    """
    n = length * scale
    w = np.ones(n, dtype=np.float32)
    if overlap <= 0:
        return w
    margin = (overlap // 4) * scale
    ramp = max(overlap * scale - 2 * margin, 1)
    d = np.arange(n, dtype=np.float32)
    profile = np.clip((d - margin + 0.5) / ramp, 0.0, 1.0)
    if start > 0:
        w = np.minimum(w, profile)
    if start + length < total:
        w = np.minimum(w, profile[::-1])
    return w


@dataclass(frozen=True)
class TileLayout:
    height: int
    width: int
    tile_h: int
    tile_w: int
    overlap: int
    ys: tuple[int, ...]
    xs: tuple[int, ...]

    @classmethod
    def plan(cls, height: int, width: int, tile: int | None, overlap: int) -> TileLayout:
        th = height if tile is None else min(tile, height)
        tw = width if tile is None else min(tile, width)
        ov = min(overlap, max(th // 2 - 1, 0), max(tw // 2 - 1, 0))
        return cls(
            height=height, width=width, tile_h=th, tile_w=tw, overlap=ov,
            ys=tile_positions(height, th, ov), xs=tile_positions(width, tw, ov),
        )

    @property
    def count(self) -> int:
        return len(self.ys) * len(self.xs)

    @property
    def tiles(self) -> list[tuple[int, int, int, int]]:
        """(row index, col index, y, x) for every tile, row-major."""
        return [(iy, ix, y, x) for iy, y in enumerate(self.ys) for ix, x in enumerate(self.xs)]

    def weights_y(self, scale: int) -> list[F32]:
        return [feather_1d(self.tile_h, y, self.height, self.overlap, scale) for y in self.ys]

    def weights_x(self, scale: int) -> list[F32]:
        return [feather_1d(self.tile_w, x, self.width, self.overlap, scale) for x in self.xs]

    @cached_property
    def _weight_cache(self) -> dict[int, tuple[list[F32], list[F32], F32]]:
        return {}

    def weight_maps(self, scale: int) -> tuple[list[F32], list[F32], F32]:
        """Per-row and per-column 1-D weights plus the summed weight map (output resolution)."""
        cache = self._weight_cache
        if scale not in cache:
            wy, wx = self.weights_y(scale), self.weights_x(scale)
            total = np.zeros((self.height * scale, self.width * scale), dtype=np.float32)
            for iy, ix, y, x in self.tiles:
                ys, xs = y * scale, x * scale
                total[ys : ys + self.tile_h * scale, xs : xs + self.tile_w * scale] += np.outer(wy[iy], wx[ix])
            cache[scale] = (wy, wx, total)
        return cache[scale]


def split_tiles(img: F32, layout: TileLayout) -> F32:
    """``(C, H, W)`` → ``(T, C, tile_h, tile_w)`` in row-major tile order."""
    return np.stack(
        [img[:, y : y + layout.tile_h, x : x + layout.tile_w] for _, _, y, x in layout.tiles]
    ).astype(np.float32)


def merge_tiles(tiles: F32, layout: TileLayout, scale: int) -> F32:
    """Inverse of :func:`split_tiles` for upscaled tiles ``(T, C, tile_h*s, tile_w*s)`` with feathering."""
    wy, wx, total = layout.weight_maps(scale)
    channels = tiles.shape[1]
    out = np.zeros((channels, layout.height * scale, layout.width * scale), dtype=np.float32)
    for (iy, ix, y, x), tile in zip(layout.tiles, tiles, strict=True):
        ys, xs = y * scale, x * scale
        out[:, ys : ys + layout.tile_h * scale, xs : xs + layout.tile_w * scale] += tile * np.outer(wy[iy], wx[ix])
    return (out / total).astype(np.float32)
