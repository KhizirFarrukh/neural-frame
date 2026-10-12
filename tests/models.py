"""Tiny, randomly initialised models so neural code paths run in CI without downloading weights."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from neuralframe.inference.archs.srvgg import SRVGGNetCompact
from neuralframe.models.registry import ModelRegistry, ModelSpec, WeightsSpec


def tiny_sr_model(models_dir: Path, *, scale: int = 2, model_id: str | None = None) -> tuple[ModelSpec, Path]:
    model_id = model_id or f"tiny-srvgg-x{scale}"
    params = {"num_in_ch": 3, "num_out_ch": 3, "num_feat": 8, "num_conv": 2, "upscale": scale, "act_type": "prelu"}
    torch.manual_seed(1234)
    net = SRVGGNetCompact(**params)
    with torch.no_grad():
        for p in net.parameters():
            p.mul_(0.1)  # small residual on top of nearest upsampling keeps outputs well-behaved
    path = models_dir / model_id / f"{model_id}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"params": net.state_dict()}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = ModelSpec(
        id=model_id, kind="sr", scale=scale, architecture="srvgg", params=params, state_dict_key="params",
        weights=WeightsSpec(url="https://invalid.example/tiny.pth", sha256=sha, filename=path.name,
                            size_bytes=path.stat().st_size),
        licence="test-only",
    )
    return spec, path


def registry_with(*extra: ModelSpec) -> ModelRegistry:
    base = ModelRegistry.load()
    return ModelRegistry([*base.all(), *extra])


def tiny_temporal_model(models_dir: Path, model_id: str = "tiny-realbasicvsr") -> tuple[ModelSpec, Path]:
    """Tiny RealBasicVSR with zeroed optical flow (identity warps): exercises the full temporal path fast."""
    from neuralframe.inference.archs.basicvsr import RealBasicVSRNet

    params = {"mid_channels": 8, "num_propagation_blocks": 1, "num_cleaning_blocks": 1}
    torch.manual_seed(4321)
    net = RealBasicVSRNet(**params)
    with torch.no_grad():
        for name, p in net.named_parameters():
            p.mul_(0.0 if "spynet" in name else 0.1)
    path = models_dir / model_id / f"{model_id}.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {f"generator_ema.{k}": v for k, v in net.state_dict().items()}}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = ModelSpec(
        id=model_id, kind="sr", scale=4, input="recurrent", window=5, architecture="realbasicvsr", params=params,
        pad_multiple=4, state_dict_key="state_dict", state_dict_prefix="generator_ema.",
        weights=WeightsSpec(url="https://invalid.example/t.pth", sha256=sha, filename=path.name,
                            size_bytes=path.stat().st_size),
        licence="test-only",
    )
    return spec, path
