"""ONNX Runtime backend: CPU (CI fallback), CUDA, DirectML (AMD/Intel on Windows) and ROCm/MIGraphX."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.backend import InferenceBackend, InferenceOOM, ShapeProfile
from neuralframe.log import get_logger
from neuralframe.models.export import ensure_onnx
from neuralframe.models.registry import ModelSpec

log = get_logger(component="onnxruntime")

_GPU_PROVIDERS = ["CUDAExecutionProvider", "ROCMExecutionProvider", "MIGraphXExecutionProvider",
                  "DmlExecutionProvider"]


def _ort() -> Any:
    try:
        import onnxruntime as ort

        return ort
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NeuralFrameError(
            ErrorCode.BACKEND_UNAVAILABLE, "ONNX Runtime is not installed",
            hint="pip install neuralframe[onnx] (CPU) or neuralframe[onnx-gpu]",
        ) from exc


class OnnxRuntimeBackend(InferenceBackend):
    name: ClassVar[str] = "onnxruntime"

    def __init__(
        self,
        device: torch.device,
        precision: str = "fp32",
        *,
        models_dir: Path,
        providers: list[str] | None = None,
    ) -> None:
        # The exported graph is FP32; GPU providers run it as is (FP16 conversion is TensorRT's job).
        super().__init__(device, "fp32")
        del precision
        self.models_dir = models_dir
        self.requested_providers = providers
        self._ort = _ort()
        self._session: Any = None
        self.providers: list[str] = []

    def _choose_providers(self) -> list[str]:
        available = list(self._ort.get_available_providers())
        if self.requested_providers:
            chosen = [p for p in self.requested_providers if p in available]
            if not chosen:
                raise NeuralFrameError(
                    ErrorCode.BACKEND_UNAVAILABLE,
                    f"none of the requested ONNX Runtime providers {self.requested_providers} are available",
                    hint=f"available: {available}",
                )
            return chosen + (["CPUExecutionProvider"] if "CPUExecutionProvider" not in chosen else [])
        if self.device.type == "cuda":
            gpu = [p for p in _GPU_PROVIDERS if p in available]
            return [*gpu[:1], "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def load(self, model: ModelSpec, weights: Path, profile: ShapeProfile) -> None:
        onnx = ensure_onnx(model, weights, self.models_dir)
        providers = self._choose_providers()
        if "CUDAExecutionProvider" in providers:
            # onnxruntime-gpu can load the CUDA/cuDNN DLLs that ship with the torch wheels.
            with contextlib.suppress(Exception):
                self._ort.preload_dlls()
        options = self._ort.SessionOptions()
        options.graph_optimization_level = self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provider_options: list[dict[str, Any]] = []
        for p in providers:
            if p == "CUDAExecutionProvider":
                # Grow the memory arena by what is requested, not by powers of two: the default strategy
                # took the whole GPU next to PyTorch's colour/resize stages.
                provider_options.append({"device_id": self.device.index or 0, "cudnn_conv_algo_search": "HEURISTIC",
                                         "arena_extend_strategy": "kSameAsRequested"})
            else:
                provider_options.append({})
        try:
            self._session = self._ort.InferenceSession(
                str(onnx), sess_options=options, providers=providers, provider_options=provider_options
            )
        except Exception as exc:
            raise NeuralFrameError(ErrorCode.BACKEND_UNAVAILABLE, f"ONNX Runtime session failed: {exc}") from exc
        self.providers = list(self._session.get_providers())
        if self.device.type == "cuda" and self.providers[0] == "CPUExecutionProvider":
            log.warning("ONNX Runtime fell back to the CPU provider", requested=providers)
        self.model = model
        log.info("ONNX Runtime session ready", model=model.id, providers=self.providers)

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        if self._session is None:
            raise NeuralFrameError(ErrorCode.INTERNAL, "OnnxRuntimeBackend.infer called before load()")
        x = batch.to(dtype=torch.float32).contiguous()
        try:
            if self.providers and self.providers[0] == "CUDAExecutionProvider" and x.is_cuda:
                return self._infer_cuda(x)
            result = self._session.run(["output"], {"input": x.cpu().numpy()})[0]
            return torch.from_numpy(np.ascontiguousarray(result)).to(batch.device)
        except NeuralFrameError:
            raise
        except Exception as exc:
            if "memory" in str(exc).lower() or "alloc" in str(exc).lower():
                raise InferenceOOM(str(exc)) from exc
            raise NeuralFrameError(ErrorCode.INFERENCE_FAILED, f"ONNX Runtime inference failed: {exc}") from exc

    def _infer_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Zero-copy I/O binding on torch device memory (no host round trip)."""
        n, c, h, w = x.shape
        s = self.model.scale if self.model else 1
        out = torch.empty((n, c, h * s, w * s), dtype=torch.float32, device=x.device)
        binding = self._session.io_binding()
        device_id = x.device.index or 0
        binding.bind_input("input", "cuda", device_id, np.float32, tuple(x.shape), x.data_ptr())
        binding.bind_output("output", "cuda", device_id, np.float32, tuple(out.shape), out.data_ptr())
        torch.cuda.current_stream(x.device).synchronize()  # ORT uses its own stream
        self._session.run_with_iobinding(binding)
        return out

    def close(self) -> None:
        self._session = None
        super().close()

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data["providers"] = self.providers
        return data
