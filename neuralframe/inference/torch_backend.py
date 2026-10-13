"""PyTorch eager backend — the correctness reference for ONNX Runtime and TensorRT."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
from torch import nn

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.archs import load_network
from neuralframe.inference.backend import InferenceBackend, InferenceOOM, ShapeProfile
from neuralframe.models.registry import ModelSpec


def configure_determinism(deterministic: bool) -> None:
    """Deterministic cuDNN algorithms make kill-and-resume bit-identical; benchmark mode is faster."""
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


class TorchBackend(InferenceBackend):
    name: ClassVar[str] = "torch"

    def __init__(self, device: torch.device, precision: str = "fp16", *, deterministic: bool = True) -> None:
        super().__init__(device, precision)
        self.deterministic = deterministic
        self._net: nn.Module | None = None

    def load(self, model: ModelSpec, weights: Path, profile: ShapeProfile) -> None:
        configure_determinism(self.deterministic)
        net = load_network(model, weights).to(self.device)
        if self.dtype == torch.float16 and getattr(net, "nf_supports_half", True) is False:
            # The architecture overflows in FP16 (spandrel says so): fall back instead of producing NaNs.
            from neuralframe.inference.factory import bf16_supported

            self.precision = "bf16" if bf16_supported(self.device) else "fp32"
        if self.dtype in (torch.float16, torch.bfloat16):
            net = net.to(self.dtype)
        self._net = net
        self.model = model

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise NeuralFrameError(ErrorCode.INTERNAL, "TorchBackend.infer called before load()")
        try:
            with torch.inference_mode():
                out: torch.Tensor = self._net(batch.to(device=self.device, dtype=self.dtype))
                return out
        except torch.OutOfMemoryError as exc:
            raise InferenceOOM(str(exc)) from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise InferenceOOM(str(exc)) from exc
            raise

    def close(self) -> None:
        self._net = None
        super().close()
