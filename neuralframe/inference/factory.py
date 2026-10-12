"""Backend selection, device resolution and tile/batch planning for a job."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.backend import InferenceBackend, bytes_per_input_pixel
from neuralframe.log import get_logger
from neuralframe.models.registry import ModelSpec
from neuralframe.monitoring import host_memory

log = get_logger(component="inference")

TILE_CANDIDATES = (1024, 768, 640, 512, 384, 320, 256, 192, 128, 96, 64)
_BATCH_CAP_HEAVY = 4
_BATCH_CAP_LIGHT = 8


@dataclass(frozen=True)
class InferencePlan:
    backend: str
    device: str
    precision: str
    tile: int | None
    overlap: int
    batch: int
    frames_per_call: int
    budget_mb: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device": self.device,
            "precision": self.precision,
            "tile": self.tile,
            "tile_overlap": self.overlap,
            "batch": self.batch,
            "frames_per_call": self.frames_per_call,
            "budget_mb": self.budget_mb,
            "notes": list(self.notes),
        }


def torch_cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def resolve_device(gpus: list[int] | None, *, force_cpu: bool = False) -> torch.device:
    if force_cpu or not torch_cuda_available():
        if gpus and not force_cpu:
            log.warning("CUDA is not available; running inference on the CPU", requested_gpus=gpus)
        return torch.device("cpu")
    index = gpus[0] if gpus else 0
    if index >= torch.cuda.device_count():
        raise NeuralFrameError(
            ErrorCode.BACKEND_UNAVAILABLE,
            f"GPU {index} requested but only {torch.cuda.device_count()} CUDA device(s) are visible",
        )
    return torch.device("cuda", index)


def available_backends() -> list[str]:
    names = ["torch"]
    if importlib.util.find_spec("onnxruntime") is not None:
        names.append("onnxruntime")
    if importlib.util.find_spec("tensorrt") is not None and torch_cuda_available():
        names.append("tensorrt")
    return names


def resolve_backend_name(requested: str, device: torch.device) -> str:
    """auto: TensorRT on CUDA when installed, ONNX Runtime on CPU when installed, otherwise PyTorch."""
    available = available_backends()
    if requested == "auto":
        if device.type == "cuda" and "tensorrt" in available:
            return "tensorrt"
        if device.type == "cpu" and "onnxruntime" in available:
            return "onnxruntime"
        return "torch"
    if requested not in available:
        raise NeuralFrameError(
            ErrorCode.BACKEND_UNAVAILABLE,
            f"backend {requested!r} is not available here",
            hint=f"available: {', '.join(available)}",
        )
    if requested == "tensorrt" and device.type != "cuda":
        raise NeuralFrameError(ErrorCode.BACKEND_UNAVAILABLE, "TensorRT needs a CUDA GPU")
    return requested


def create_backend(
    name: str,
    device: torch.device,
    precision: str,
    *,
    deterministic: bool,
    models_dir: Path | None = None,
    engines_dir: Path | None = None,
    onnx_providers: list[str] | None = None,
    status: Callable[[str], None] | None = None,
) -> InferenceBackend:
    if name == "torch":
        from neuralframe.inference.torch_backend import TorchBackend

        return TorchBackend(device, precision, deterministic=deterministic)
    if name == "tensorrt":
        from neuralframe.inference.tensorrt_backend import TensorRTBackend

        assert models_dir is not None and engines_dir is not None
        return TensorRTBackend(device, precision, models_dir=models_dir, engines_dir=engines_dir, status=status)
    if name == "onnxruntime":
        from neuralframe.inference.onnx_backend import OnnxRuntimeBackend

        assert models_dir is not None
        return OnnxRuntimeBackend(device, precision, models_dir=models_dir, providers=onnx_providers)
    raise NeuralFrameError(ErrorCode.BACKEND_UNAVAILABLE, f"unknown backend {name!r}")


def memory_budget(device: torch.device, *, safety_margin: float, max_vram_gb: float | None) -> int:
    """Bytes the network may use: free device memory minus the safety margin, capped by the user limit."""
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        budget = int(free * (1.0 - safety_margin))
        if max_vram_gb is not None:
            used_by_others = torch.cuda.memory_reserved(device)
            budget = min(budget, int(max_vram_gb * 2**30) - used_by_others)
    else:
        _total, available = host_memory()
        budget = int(available * 0.5)
    return max(budget, 64 * 2**20)


def plan_inference(
    model: ModelSpec,
    frame_hw: tuple[int, int],
    *,
    device: torch.device,
    backend: str,
    precision: str,
    budget_bytes: int,
    tile: int | None = None,
    overlap: int = 32,
    batch: int | None = None,
) -> InferencePlan:
    """Choose tile size and batch so activations fit ``budget_bytes`` (pre-calibration heuristic)."""
    eff_precision = "fp32" if device.type == "cpu" else precision
    per_px = bytes_per_input_pixel(model, eff_precision)
    weights_bytes = (model.weights.size_bytes or 0) if model.weights else 0
    budget = max(budget_bytes - weights_bytes, 32 * 2**20)
    h, w = frame_hw
    notes: list[str] = []
    cap = _BATCH_CAP_HEAVY if per_px > 1500 else _BATCH_CAP_LIGHT

    chosen_tile = tile
    if chosen_tile is None:
        if per_px * h * w <= budget:
            chosen_tile = None
        else:
            chosen_tile = next((t for t in TILE_CANDIDATES if per_px * t * t <= budget), TILE_CANDIDATES[-1])
            notes.append(f"frame does not fit the {budget // 2**20} MB budget whole; tiling at {chosen_tile}px")
    tile_px = h * w if chosen_tile is None else min(chosen_tile, h) * min(chosen_tile, w)
    fit = max(1, budget // max(per_px * tile_px, 1))
    chosen_batch = batch if batch is not None else int(min(fit, cap))
    if batch is None and fit < 1:
        notes.append("even one tile exceeds the budget estimate; relying on OOM recovery")
    frames_per_call = chosen_batch if chosen_tile is None else 1
    return InferencePlan(
        backend=backend,
        device=str(device),
        precision=eff_precision,
        tile=chosen_tile,
        overlap=overlap if chosen_tile is not None else 0,
        batch=chosen_batch,
        frames_per_call=frames_per_call,
        budget_mb=budget // 2**20,
        notes=tuple(notes),
    )
