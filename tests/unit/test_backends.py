"""ONNX export, ONNX Runtime and TensorRT backends, engine cache keys/invalidation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralframe.inference.backend import ShapeProfile  # noqa: E402
from neuralframe.inference.engine_cache import EngineCache, EngineKey  # noqa: E402
from neuralframe.inference.torch_backend import TorchBackend  # noqa: E402
from neuralframe.models.export import compare_outputs, ensure_onnx, onnx_path, sample_tiles  # noqa: E402
from neuralframe.models.registry import ModelSpec  # noqa: E402
from tests.models import tiny_sr_model  # noqa: E402

try:
    import onnxruntime  # noqa: F401

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModelSpec, Path, Path]:
    models = tmp_path_factory.mktemp("models")
    spec, weights = tiny_sr_model(models, scale=2)
    return spec, weights, models


def _torch_reference(spec: ModelSpec, weights: Path, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    backend = TorchBackend(torch.device(device), "fp32", deterministic=True)
    backend.load(spec, weights, ShapeProfile.fixed(x.shape[2], x.shape[3], x.shape[0]))
    return backend.infer(x.to(device)).float().cpu()


def test_onnx_export_is_cached_and_reexported_on_weight_change(tiny: tuple[ModelSpec, Path, Path]) -> None:
    spec, weights, models = tiny
    path = ensure_onnx(spec, weights, models)
    assert path == onnx_path(spec, models) and path.is_file()
    mtime = path.stat().st_mtime_ns
    assert ensure_onnx(spec, weights, models).stat().st_mtime_ns == mtime  # cache hit
    changed = spec.model_copy(update={"weights": spec.weights.model_copy(update={"sha256": "0" * 64})})  # type: ignore[union-attr]
    ensure_onnx(changed, weights, models)
    assert path.stat().st_mtime_ns != mtime


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
def test_onnxruntime_cpu_matches_torch_with_dynamic_shapes(tiny: tuple[ModelSpec, Path, Path]) -> None:
    from neuralframe.inference.onnx_backend import OnnxRuntimeBackend

    spec, weights, models = tiny
    backend = OnnxRuntimeBackend(torch.device("cpu"), models_dir=models, providers=["CPUExecutionProvider"])
    backend.load(spec, weights, ShapeProfile.fixed(32, 48))
    for shape in ((2, 32, 48), (1, 57, 31)):  # dynamic batch / height / width
        x = sample_tiles(*shape)
        ref = _torch_reference(spec, weights, x)
        _, _, psnr = compare_outputs(ref, backend.infer(x))
        assert psnr > 80
    assert backend.describe()["providers"] == ["CPUExecutionProvider"]


def _key(**overrides: object) -> EngineKey:
    base = dict(model_id="m", weights_sha256="abc", opset=17, gpu_name="GPU A", compute_capability="8.9",
                driver="600.1", trt_version="10.16", precision="fp16", min_shape=(1, 3, 64, 64),
                opt_shape=(4, 3, 256, 256), max_shape=(4, 3, 256, 256))
    base.update(overrides)
    return EngineKey(**base)  # type: ignore[arg-type]


def test_engine_key_changes_with_environment_and_profile() -> None:
    k = _key()
    assert k.digest() == _key().digest()
    for change in ({"driver": "601.0"}, {"trt_version": "11.0"}, {"gpu_name": "GPU B"},
                   {"opt_shape": (8, 3, 256, 256)}, {"precision": "fp32"}, {"weights_sha256": "def"}):
        assert _key(**change).digest() != k.digest()
    assert k.filename.startswith("m-fp16-256x256b4-")


def test_engine_cache_store_load_and_prune(tmp_path: Path) -> None:
    cache = EngineCache(tmp_path)
    old = _key(driver="500.0")
    new = _key()
    cache.store(old, b"old-engine", 1.0)
    cache.store(new, b"new-engine", 2.0)
    assert cache.load(new) == b"new-engine"
    assert cache.load(_key(opt_shape=(2, 3, 256, 256))) is None
    removed = cache.prune_stale(new.environment())
    assert [p.name for p in removed] == [old.filename]
    assert cache.load(old) is None and cache.load(new) == b"new-engine"


@pytest.mark.gpu
def test_tensorrt_matches_torch_and_caches_engine(tiny: tuple[ModelSpec, Path, Path], tmp_path: Path) -> None:
    pytest.importorskip("tensorrt")
    from neuralframe.inference.tensorrt_backend import TensorRTBackend

    spec, weights, models = tiny
    profile = ShapeProfile(min_hw=(32, 32), opt_hw=(64, 64), max_hw=(64, 96), max_batch=2, opt_batch=2)
    x = sample_tiles(2, 64, 64)
    ref = _torch_reference(spec, weights, x, "cuda")
    first = TensorRTBackend(torch.device("cuda"), "fp16", models_dir=models, engines_dir=tmp_path)
    first.load(spec, weights, profile)
    assert first.built
    out = first.infer(x.cuda()).float().cpu()
    torch.cuda.synchronize()
    _, _, psnr = compare_outputs(ref, out)
    assert psnr > 45, psnr
    # other shapes inside the profile work too
    assert tuple(first.infer(sample_tiles(1, 32, 96).cuda()).shape) == (1, 3, 64, 192)
    first.close()
    second = TensorRTBackend(torch.device("cuda"), "fp16", models_dir=models, engines_dir=tmp_path)
    second.load(spec, weights, profile)
    assert not second.built  # loaded from the cache
    np.testing.assert_allclose(second.infer(x.cuda()).float().cpu().numpy(), out.numpy(), atol=0)
    second.close()
