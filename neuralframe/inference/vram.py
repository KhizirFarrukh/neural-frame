"""VRAM management: calibration before the first chunk and an out-of-memory recovery ladder.

Ladder (per OOM): free cached memory → halve the batch (repeatable until 1) → halve the tile (whole
frames become tiles first; at most ``max_levels`` tile reductions) → give up with OOM_UNRECOVERABLE, which
lets the job fall back to another backend. The failed batch is simply retried, so finished work is never
lost.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.backend import InferenceOOM
from neuralframe.log import get_logger

if TYPE_CHECKING:
    from neuralframe.inference.runner import ModelRunner

log = get_logger(component="vram")

MIN_TILE = 64


@dataclass
class Adaptation:
    level: int
    batch: int
    tile: int | None
    reason: str
    at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {"level": self.level, "batch": self.batch, "tile": self.tile, "reason": self.reason[:200]}


class VramManager:
    def __init__(
        self,
        runner: ModelRunner,
        *,
        max_levels: int = 3,
        min_tile: int = MIN_TILE,
        on_adapt: Callable[[Adaptation], None] | None = None,
    ) -> None:
        self.runner = runner
        self.max_levels = max_levels
        self.min_tile = min_tile
        self.on_adapt = on_adapt
        self.history: list[Adaptation] = []
        self.tile_levels = 0

    @property
    def level(self) -> int:
        return len(self.history)

    def _release(self) -> None:
        self.runner.backend.release_scratch()
        if self.runner.device.type == "cuda":
            torch.cuda.synchronize(self.runner.device)
            torch.cuda.empty_cache()

    def adapt(self, reason: str, frame_hw: tuple[int, int]) -> None:
        """Make the next attempt cheaper or raise OOM_UNRECOVERABLE."""
        self._release()
        r = self.runner
        if r.batch > 1:
            r.batch = max(1, r.batch // 2)
        else:
            if self.tile_levels >= self.max_levels:
                raise NeuralFrameError(
                    ErrorCode.OOM_UNRECOVERABLE,
                    f"out of memory after {self.level} adaptations (batch {r.batch}, tile {r.tile}): "
                    f"{reason[:160]}",
                    details={"adaptations": [a.as_dict() for a in self.history]},
                    hint="free VRAM used by other applications, use --tile-size 128, or --backend torch",
                )
            multiple = max(r.multiple, 8)
            current = r.tile if r.tile is not None else max(frame_hw)
            new_tile = max(self.min_tile, (current // 2) // multiple * multiple)
            if r.tile is not None and new_tile >= r.tile:
                raise NeuralFrameError(
                    ErrorCode.OOM_UNRECOVERABLE,
                    f"out of memory even at the minimum tile size {r.tile}px: {reason[:160]}",
                    details={"adaptations": [a.as_dict() for a in self.history]},
                )
            r.tile = new_tile
            self.tile_levels += 1
            # Whole-frame plans have no overlap; tiles need one to hide seams.
            r.overlap = min(r.overlap if r.overlap else 16, new_tile // 4)
        step = Adaptation(level=self.level + 1, batch=r.batch, tile=r.tile, reason=reason)
        self.history.append(step)
        log.warning("out of memory: adapting", level=step.level, batch=r.batch, tile=r.tile)
        if self.on_adapt is not None:
            self.on_adapt(step)

    def run_with_recovery(self, fn: Callable[[], torch.Tensor], frame_hw: tuple[int, int]) -> torch.Tensor:
        while True:
            try:
                return fn()
            except InferenceOOM as exc:
                self.adapt(str(exc), frame_hw)

    def calibrate(
        self,
        frame_hw: tuple[int, int],
        *,
        frames: int,
        safety_margin: float,
    ) -> dict[str, Any]:
        """Dry-run one batch of synthetic frames; adapt until it fits with ``safety_margin`` free VRAM."""
        r = self.runner
        t0 = time.perf_counter()
        h, w = frame_hw
        device = r.device
        while True:
            n = max(1, min(frames, r.batch if r.tile is None else 1))
            x = torch.rand((n, 3, h, w), device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            try:
                r.run(x)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except InferenceOOM as exc:
                self.adapt(str(exc), frame_hw)
                continue
            finally:
                del x
            if device.type != "cuda":
                break
            free, total = torch.cuda.mem_get_info(device)
            if free >= safety_margin * total or (r.batch == 1 and (r.tile or 10**9) <= self.min_tile):
                break
            self.adapt(f"only {free // 2**20} MB free after calibration (< {safety_margin:.0%})", frame_hw)
        self._release()
        return {
            "batch": r.batch,
            "tile": r.tile,
            "tile_overlap": r.overlap,
            "seconds": round(time.perf_counter() - t0, 2),
            "adaptations": [a.as_dict() for a in self.history],
        }
