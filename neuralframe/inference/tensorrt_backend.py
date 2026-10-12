"""TensorRT backend: ONNX → FP16 engine with an optimisation profile, cached on disk, run with
``execute_async_v3`` on the current torch CUDA stream (device pointers from torch tensors)."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import torch

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.backend import InferenceBackend, InferenceOOM, ShapeProfile
from neuralframe.inference.engine_cache import EngineCache, EngineKey
from neuralframe.log import get_logger
from neuralframe.models.export import DEFAULT_OPSET, ensure_onnx
from neuralframe.models.registry import ModelSpec
from neuralframe.monitoring import driver_versions

log = get_logger(component="tensorrt")

StatusCallback = Callable[[str], None]


def _trt() -> Any:
    try:
        import tensorrt as trt

        return trt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise NeuralFrameError(
            ErrorCode.BACKEND_UNAVAILABLE, "TensorRT is not installed", hint="pip install neuralframe[tensorrt]"
        ) from exc


def _make_logger(trt: Any) -> Any:
    class _Logger(trt.ILogger):  # type: ignore[misc]
        def __init__(self) -> None:
            trt.ILogger.__init__(self)

        def log(self, severity: Any, msg: str) -> None:
            if severity <= trt.ILogger.Severity.ERROR:
                log.error("tensorrt", message=msg)
            elif severity == trt.ILogger.Severity.WARNING:
                log.debug("tensorrt warning", message=msg)

    return _Logger()


class TensorRTBackend(InferenceBackend):
    name: ClassVar[str] = "tensorrt"

    def __init__(
        self,
        device: torch.device,
        precision: str = "fp16",
        *,
        models_dir: Path,
        engines_dir: Path,
        workspace_bytes: int | None = None,
        status: StatusCallback | None = None,
    ) -> None:
        if device.type != "cuda":
            raise NeuralFrameError(ErrorCode.BACKEND_UNAVAILABLE, "TensorRT needs a CUDA device")
        super().__init__(device, precision)
        self.models_dir = models_dir
        self.cache = EngineCache(engines_dir)
        self.workspace_bytes = workspace_bytes
        self.status = status
        self._trt = _trt()
        self._logger = _make_logger(self._trt)
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self.engine_path: Path | None = None
        self.built = False
        self.build_seconds = 0.0

    # ---- engine management ---------------------------------------------------------------------------

    def environment(self) -> dict[str, str]:
        props = torch.cuda.get_device_properties(self.device)
        driver, _ = driver_versions()
        return {
            "gpu_name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "driver": driver or "unknown",
            "trt_version": str(self._trt.__version__),
        }

    def key_for(self, model: ModelSpec, profile: ShapeProfile) -> EngineKey:
        env = self.environment()
        return EngineKey(
            model_id=model.id,
            weights_sha256=model.weights.sha256 if model.weights else "none",
            opset=DEFAULT_OPSET,
            precision=self.precision,
            min_shape=(1, 3, *profile.min_hw),
            opt_shape=(profile.opt_batch, 3, *profile.opt_hw),
            max_shape=(profile.max_batch, 3, *profile.max_hw),
            **env,
        )

    def _build(self, onnx: Path, key: EngineKey) -> bytes:
        trt = self._trt
        builder = trt.Builder(self._logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, self._logger)
        if not parser.parse_from_file(str(onnx)):
            errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise NeuralFrameError(ErrorCode.ENGINE_BUILD_FAILED, f"TensorRT cannot parse {onnx.name}: {errors}")
        config = builder.create_builder_config()
        free, _ = torch.cuda.mem_get_info(self.device)
        workspace = self.workspace_bytes or int(min(free * 0.5, 4 * 2**30))
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace)
        if self.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
            # Half-precision I/O halves DRAM traffic for the (large) upscaled output. Setting I/O dtypes on
            # weakly typed networks is deprecated since TensorRT 10.12 but still supported in 10.x; if a
            # future release rejects it we keep FP32 I/O (infer() adapts to the engine's tensor dtype).
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                try:
                    network.get_input(0).dtype = trt.float16
                    network.get_output(0).dtype = trt.float16
                except Exception as exc:  # pragma: no cover - future TensorRT
                    log.info("keeping FP32 engine I/O", reason=str(exc))
        profile = builder.create_optimization_profile()
        profile.set_shape("input", key.min_shape, key.opt_shape, key.max_shape)
        config.add_optimization_profile(profile)

        timing_path = self.cache.timing_cache_path(key.environment())
        timing_bytes = timing_path.read_bytes() if timing_path.is_file() else b""
        timing = config.create_timing_cache(timing_bytes)
        config.set_timing_cache(timing, ignore_mismatch=False)

        log.info("building TensorRT engine (one-time per model/GPU/shape)", model=key.model_id,
                 opt=key.opt_shape, max=key.max_shape, workspace_mb=workspace // 2**20)
        if self.status:
            self.status(f"building TensorRT engine for {key.model_id} (one-time, can take a few minutes)")
        t0 = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise NeuralFrameError(
                ErrorCode.ENGINE_BUILD_FAILED,
                f"TensorRT engine build failed for {key.model_id} at {key.opt_shape}",
                hint="try a smaller --tile-size or --backend torch",
            )
        self.build_seconds = time.perf_counter() - t0
        timing_path.write_bytes(bytes(memoryview(config.get_timing_cache().serialize())))
        data = bytes(memoryview(serialized))
        self.engine_path = self.cache.store(key, data, self.build_seconds)
        self.built = True
        log.info("TensorRT engine built", model=key.model_id, seconds=round(self.build_seconds, 1),
                 size_mb=len(data) // 2**20)
        return data

    def load(self, model: ModelSpec, weights: Path, profile: ShapeProfile) -> None:
        torch.cuda.set_device(self.device)
        self.cache.prune_stale(self.environment())
        key = self.key_for(model, profile)
        data = self.cache.load(key)
        if data is None:
            onnx = ensure_onnx(model, weights, self.models_dir)
            data = self._build(onnx, key)
        else:
            self.engine_path = self.cache.path_for(key)
        self._runtime = self._trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(data)
        if self._engine is None:
            raise NeuralFrameError(ErrorCode.ENGINE_BUILD_FAILED, f"cannot deserialize engine {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise InferenceOOM("not enough device memory for the TensorRT execution context")
        io_dtype = self._engine.get_tensor_dtype("input")
        self._in_dtype = torch.float16 if io_dtype == self._trt.float16 else torch.float32
        self.model = model
        self.profile = profile

    @property
    def activation_bytes(self) -> int:
        return int(self._engine.device_memory_size_v2) if self._engine is not None else 0

    # ---- inference -----------------------------------------------------------------------------------

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        if self._context is None:
            raise NeuralFrameError(ErrorCode.INTERNAL, "TensorRTBackend.infer called before load()")
        x = batch.to(device=self.device, dtype=self._in_dtype).contiguous()
        ctx = self._context
        if not ctx.set_input_shape("input", tuple(x.shape)):
            raise NeuralFrameError(
                ErrorCode.INFERENCE_FAILED,
                f"input shape {tuple(x.shape)} is outside the engine profile",
                details={"profile": [self.profile.min_hw, self.profile.max_hw, self.profile.max_batch]},
            )
        shape = tuple(ctx.get_tensor_shape("output"))
        try:
            out = torch.empty(shape, dtype=self._in_dtype, device=self.device)
        except torch.OutOfMemoryError as exc:
            raise InferenceOOM(str(exc)) from exc
        ctx.set_tensor_address("input", x.data_ptr())
        ctx.set_tensor_address("output", out.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not ctx.execute_async_v3(stream.cuda_stream):
            raise NeuralFrameError(ErrorCode.INFERENCE_FAILED, "TensorRT execution failed")
        return out

    def release_scratch(self) -> None:
        torch.cuda.empty_cache()

    def close(self) -> None:
        self._context = None
        self._engine = None
        self._runtime = None
        super().close()

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update({
            "engine": str(self.engine_path) if self.engine_path else None,
            "engine_built_now": self.built,
            "build_seconds": round(self.build_seconds, 1),
            "activation_mb": self.activation_bytes // 2**20,
        })
        return data
