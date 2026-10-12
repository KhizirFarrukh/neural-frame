"""Frame processors: what the inference stage applies to batches of RGB frames.

A processor receives planar float32 RGB frames ``(3, H, W)`` in ``[0, 1]`` and returns frames scaled by
:attr:`FrameProcessor.scale`. Neural processors (tiling, batching, backends) live in
:mod:`neuralframe.inference.upscaler`; classical models use :class:`IdentityProcessor` and leave all
resampling to the postprocess stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


class FrameProcessor(ABC):
    name: str = "processor"
    scale: int = 1
    batch_size: int = 1

    @abstractmethod
    def process(self, frames: list[F32]) -> list[F32]:
        """Process a batch (len <= batch_size). Must return one output per input, same order."""

    def stats(self) -> dict[str, object]:
        return {}

    def close(self) -> None:
        return None


class IdentityProcessor(FrameProcessor):
    name = "identity"
    scale = 1
    batch_size = 1

    def process(self, frames: list[F32]) -> list[F32]:
        return frames
