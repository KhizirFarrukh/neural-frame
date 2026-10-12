"""Model execution over arbitrarily sized frames: padding, fixed-size tiles, batching, feathered merge."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from neuralframe.chunking.tiling import TileLayout
from neuralframe.inference.backend import InferenceBackend
from neuralframe.models.registry import ModelSpec
from neuralframe.ops.torch_ops import pad_to_multiple


@dataclass
class RunnerStats:
    frames: int = 0
    tiles: int = 0
    infer_calls: int = 0
    layouts: dict[str, int] = field(default_factory=dict)


class ModelRunner:
    """Runs a super-resolution backend on ``(N, 3, H, W)`` float tensors of any size.

    ``tile`` is the tile edge in input pixels (``None`` = whole frame), ``batch`` the number of tiles (or
    whole frames) per backend call. Weights and canvases stay on the backend's device.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        model: ModelSpec,
        *,
        tile: int | None,
        overlap: int,
        batch: int,
    ) -> None:
        self.backend = backend
        self.model = model
        self.scale = model.scale
        self.tile = tile
        self.overlap = overlap
        self.batch = max(1, batch)
        self.multiple = max(model.pad_multiple, 1)
        self.stats = RunnerStats()
        self._weights: dict[tuple[int, int], tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]] = {}

    @property
    def device(self) -> torch.device:
        return self.backend.device

    def layout_for(self, height: int, width: int) -> TileLayout:
        tile = self.tile
        if tile is not None:
            tile = max(self.multiple, tile - tile % self.multiple)
        return TileLayout.plan(height, width, tile, self.overlap)

    def _device_weights(
        self, layout: TileLayout
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
        key = (layout.height, layout.width)
        if key not in self._weights:
            wy, wx, total = layout.weight_maps(self.scale)
            self._weights[key] = (
                [torch.from_numpy(w).to(self.device) for w in wy],
                [torch.from_numpy(w).to(self.device) for w in wx],
                torch.from_numpy(total).to(self.device),
            )
        return self._weights[key]

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """``(N, 3, H, W)`` in [0, 1] → ``(N, 3, H*s, W*s)`` float32 in [0, 1]."""
        n, _, h, w = x.shape
        padded, _, _ = pad_to_multiple(x, self.multiple)
        hp, wp = padded.shape[-2:]
        layout = self.layout_for(hp, wp)
        key = f"{layout.tile_h}x{layout.tile_w}x{layout.count}"
        self.stats.layouts[key] = self.stats.layouts.get(key, 0) + n
        self.stats.frames += n
        s = self.scale

        if layout.count == 1:
            outs = []
            for i in range(0, n, self.batch):
                outs.append(self.backend.infer(padded[i : i + self.batch]).float())
                self.stats.infer_calls += 1
                self.stats.tiles += min(self.batch, n - i)
            full = torch.cat(outs, dim=0)
        else:
            wy, wx, total = self._device_weights(layout)
            full = torch.zeros((n, 3, hp * s, wp * s), dtype=torch.float32, device=self.device)
            jobs = [(f, iy, ix, ty, tx) for f in range(n) for iy, ix, ty, tx in layout.tiles]
            th, tw = layout.tile_h, layout.tile_w
            for start in range(0, len(jobs), self.batch):
                chunk = jobs[start : start + self.batch]
                tiles = torch.stack([padded[f, :, ty : ty + th, tx : tx + tw] for f, _, _, ty, tx in chunk])
                result = self.backend.infer(tiles).float()
                self.stats.infer_calls += 1
                self.stats.tiles += len(chunk)
                for (f, iy, ix, ty, tx), out in zip(chunk, result, strict=True):
                    weight = wy[iy][:, None] * wx[ix][None, :]
                    full[f, :, ty * s : (ty + th) * s, tx * s : (tx + tw) * s] += out * weight
            full /= total
        return full[:, :, : h * s, : w * s].clamp_(0.0, 1.0)
