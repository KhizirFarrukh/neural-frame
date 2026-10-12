"""Simulated out-of-memory: the runner adapts (batch, then tile) and keeps producing correct output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from neuralframe.errors import ErrorCode, NeuralFrameError  # noqa: E402
from neuralframe.inference.backend import InferenceBackend, InferenceOOM, ShapeProfile  # noqa: E402
from neuralframe.inference.runner import ModelRunner  # noqa: E402
from neuralframe.models.registry import ModelSpec  # noqa: E402


class LimitedBackend(InferenceBackend):
    """Nearest-neighbour x2 'network' that runs out of memory above ``limit`` input pixels per call."""

    name = "limited"

    def __init__(self, limit: int) -> None:
        super().__init__(torch.device("cpu"), "fp32")
        self.limit = limit
        self.calls: list[tuple[int, ...]] = []

    def load(self, model: ModelSpec, weights: Path, profile: ShapeProfile) -> None:
        self.model = model

    def infer(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(batch.shape))
        if batch.shape[0] * batch.shape[2] * batch.shape[3] > self.limit:
            raise InferenceOOM("CUDA out of memory (simulated)")
        return batch.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


SPEC = ModelSpec(id="nearest-x2", kind="sr", scale=2, architecture="srvgg")


def frames(n: int = 4, h: int = 64, w: int = 96) -> torch.Tensor:
    return torch.rand(n, 3, h, w, generator=torch.Generator().manual_seed(0))


def expected(x: torch.Tensor) -> torch.Tensor:
    return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)


def runner(limit: int, *, batch: int, tile: int | None = None) -> ModelRunner:
    backend = LimitedBackend(limit)
    backend.load(SPEC, Path("unused"), ShapeProfile.fixed(64, 96))
    r = ModelRunner(backend, SPEC, tile=tile, overlap=0, batch=batch)
    r.enable_recovery(max_levels=3)
    return r


def test_batch_is_halved_until_it_fits() -> None:
    r = runner(limit=2 * 64 * 96, batch=8)
    x = frames()
    out = r.run(x)
    torch.testing.assert_close(out, expected(x))
    assert r.batch == 2
    assert [a.batch for a in r.vram.history] == [4, 2]  # type: ignore[union-attr]


def test_whole_frames_become_tiles_when_one_frame_does_not_fit() -> None:
    r = runner(limit=64 * 64, batch=1)
    x = frames(2)
    out = r.run(x)
    torch.testing.assert_close(out, expected(x), atol=1e-6, rtol=0)
    assert r.tile is not None and r.tile <= 64
    assert r.overlap > 0  # tiles need an overlap to hide seams


def test_gives_up_with_clear_error_after_max_levels() -> None:
    r = runner(limit=10, batch=2)
    with pytest.raises(NeuralFrameError) as err:
        r.run(frames(2))
    assert err.value.code is ErrorCode.OOM_UNRECOVERABLE
    assert err.value.details["adaptations"][0]["batch"] == 1  # batch first, then tiles


def test_calibration_records_final_configuration() -> None:
    r = runner(limit=64 * 96, batch=4)
    result = r.vram.calibrate((64, 96), frames=4, safety_margin=0.1)  # type: ignore[union-attr]
    assert result["batch"] == 1 and result["tile"] is None
    assert len(result["adaptations"]) == 2


def test_job_survives_simulated_oom(
    caps: Any, settings: Any, clip: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from neuralframe.core.job import JobRunner, JobSpec
    from neuralframe.core.planning import TargetSpec
    from neuralframe.inference.torch_backend import TorchBackend
    from tests.models import registry_with, tiny_sr_model

    settings.inference.device = "cpu"
    settings.inference.backend = "torch"
    spec, _ = tiny_sr_model(settings.paths.models(), scale=2)
    original = TorchBackend.infer

    def tight(self: TorchBackend, batch: torch.Tensor) -> torch.Tensor:
        if batch.shape[0] * batch.shape[2] * batch.shape[3] > 72 * 72:  # 64px tiles fit, 80px do not
            raise InferenceOOM("simulated")
        return original(self, batch)

    monkeypatch.setattr(TorchBackend, "infer", tight)
    report = JobRunner(settings, capabilities=caps, registry=registry_with(spec)).run(
        JobSpec(input=clip("oom", duration=0.5), output=tmp_path / "oom.mkv", model=spec.id,
                target=TargetSpec.factor(2), overwrite=True)
    ).report
    assert report["sync"]["ok"]
    assert report["vram"]["adaptations"], "expected the OOM ladder to adapt"
    assert report["vram"]["adaptations"][-1]["tile"] is not None
