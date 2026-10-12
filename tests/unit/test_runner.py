"""ModelRunner: tiling must not change results beyond a high PSNR (the tile-seam test)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralframe.inference.backend import InferenceBackend, InferenceOOM, ShapeProfile  # noqa: E402
from neuralframe.inference.runner import ModelRunner  # noqa: E402
from neuralframe.inference.torch_backend import TorchBackend  # noqa: E402
from neuralframe.models.registry import ModelSpec  # noqa: E402
from tests.models import tiny_sr_model  # noqa: E402


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def smooth_batch(n: int, h: int, w: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    frames = [
        torch.stack([0.5 + 0.4 * torch.sin(xx / (6 + k) + yy / 9), 0.5 + 0.3 * torch.cos(yy / 7),
                     0.5 + 0.2 * torch.sin((xx - yy) / 11)])
        for k in range(n)
    ]
    return torch.stack(frames).clamp(0, 1)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModelSpec, Path]:
    return tiny_sr_model(tmp_path_factory.mktemp("models"), scale=2)


def _backend(spec: ModelSpec, weights: Path) -> TorchBackend:
    b = TorchBackend(torch.device("cpu"), "fp32", deterministic=True)
    b.load(spec, weights, ShapeProfile.fixed(64, 64))
    return b


def test_tiled_matches_whole_frame(tiny: tuple[ModelSpec, Path]) -> None:
    spec, weights = tiny
    backend = _backend(spec, weights)
    x = smooth_batch(2, 101, 157)
    whole = ModelRunner(backend, spec, tile=None, overlap=0, batch=2).run(x)
    tiled_runner = ModelRunner(backend, spec, tile=48, overlap=16, batch=5)
    tiled = tiled_runner.run(x)
    assert tiled.shape == whole.shape == (2, 3, 202, 314)
    assert tiled_runner.stats.tiles > 2
    value = psnr(tiled.numpy(), whole.numpy())
    assert value > 45.0, f"tile seams visible: PSNR {value:.1f} dB"


def test_batching_does_not_change_results(tiny: tuple[ModelSpec, Path]) -> None:
    spec, weights = tiny
    backend = _backend(spec, weights)
    x = smooth_batch(3, 64, 80)
    a = ModelRunner(backend, spec, tile=None, overlap=0, batch=1).run(x)
    b = ModelRunner(backend, spec, tile=None, overlap=0, batch=3).run(x)
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-6)


def test_padding_to_model_multiple_is_cropped(tiny: tuple[ModelSpec, Path]) -> None:
    spec, weights = tiny
    spec4 = spec.model_copy(update={"pad_multiple": 4})
    out = ModelRunner(_backend(spec4, weights), spec4, tile=None, overlap=0, batch=1).run(smooth_batch(1, 33, 47))
    assert out.shape == (1, 3, 66, 94)


class _OOMBackend(InferenceBackend):
    name = "oom"

    def load(self, model: ModelSpec, weights: Path, profile: ShapeProfile) -> None:
        self.model = model

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        raise InferenceOOM("simulated")


def test_oom_is_surfaced_as_inference_oom(tiny: tuple[ModelSpec, Path]) -> None:
    spec, _ = tiny
    backend = _OOMBackend(torch.device("cpu"), "fp32")
    backend.load(spec, Path("unused"), ShapeProfile.fixed(8, 8))
    with pytest.raises(InferenceOOM):
        ModelRunner(backend, spec, tile=None, overlap=0, batch=1).run(smooth_batch(1, 16, 16))
