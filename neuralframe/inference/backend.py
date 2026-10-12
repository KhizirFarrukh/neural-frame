"""Inference backend interface shared by Torch, ONNX Runtime and TensorRT implementations.

Tensors at the boundary are ``torch.Tensor`` in NCHW layout, RGB in [0, 1], on :attr:`device` with
:attr:`dtype`; outputs are scaled by the model's native factor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import torch

from neuralframe.models.registry import ModelSpec


class InferenceOOM(RuntimeError):
    """Raised by backends when the device runs out of memory; the VRAM manager adapts and retries."""


@dataclass(frozen=True)
class ShapeProfile:
    """Range of input shapes a backend must support (used for TensorRT optimisation profiles)."""

    min_hw: tuple[int, int]
    opt_hw: tuple[int, int]
    max_hw: tuple[int, int]
    max_batch: int = 1
    opt_batch: int = 1

    @classmethod
    def fixed(cls, height: int, width: int, batch: int = 1) -> ShapeProfile:
        return cls((height, width), (height, width), (height, width), batch, batch)


def bytes_per_input_pixel(model: ModelSpec | None, precision: str) -> int:
    """Peak activation memory per *input* pixel per batch item.

    Measured with PyTorch eager FP16 on an RTX 4060 (peak allocator delta): RRDBNet x4 ≈ 8.8 KB/px,
    RRDBNet x2 ≈ 2.2 KB/px (features live at output resolution, so it scales with s²), SRVGGNetCompact
    ≈ 0.55 KB/px. Used before VRAM calibration has run; FP32 doubles it.
    """
    arch = model.architecture if model else "rrdbnet"
    scale = model.scale if model else 4
    per_px = 560 * scale * scale if arch == "rrdbnet" else 620
    return per_px * (1 if precision == "fp16" else 2)


class InferenceBackend(ABC):
    name: ClassVar[str] = "backend"

    def __init__(self, device: torch.device, precision: str = "fp16") -> None:
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.precision = "fp32" if device.type == "cpu" else precision
        self.model: ModelSpec | None = None

    @property
    def dtype(self) -> torch.dtype:
        return torch.float16 if self.precision == "fp16" else torch.float32

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @abstractmethod
    def load(self, model: ModelSpec, weights: Path, profile: ShapeProfile) -> None:
        """Prepare the model for inputs within ``profile``."""

    @abstractmethod
    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        """Run the network on ``batch`` (N, 3, H, W) → (N, 3, H*s, W*s). May raise :class:`InferenceOOM`."""

    def max_batch_for(self, tile_hw: tuple[int, int], free_vram_bytes: int) -> int:
        """Largest batch of ``tile_hw`` tiles expected to fit in ``free_vram_bytes`` (heuristic)."""
        per_item = bytes_per_input_pixel(self.model, self.precision) * tile_hw[0] * tile_hw[1]
        return max(1, int(free_vram_bytes // max(per_item, 1)))

    def release_scratch(self) -> None:
        """Free cached temporary memory (called on OOM and between jobs)."""
        if self.is_cuda:
            torch.cuda.empty_cache()

    def close(self) -> None:
        self.release_scratch()

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "device": str(self.device),
            "precision": self.precision,
            "model": self.model.id if self.model else None,
        }
