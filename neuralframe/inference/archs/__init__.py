"""Network architectures (PyTorch) and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.models.registry import ModelSpec


def build_network(spec: ModelSpec) -> nn.Module:
    params: dict[str, Any] = dict(spec.params)
    if spec.architecture == "rrdbnet":
        from neuralframe.inference.archs.rrdbnet import RRDBNet

        return RRDBNet(**params)
    if spec.architecture == "srvgg":
        from neuralframe.inference.archs.srvgg import SRVGGNetCompact

        return SRVGGNetCompact(**params)
    if spec.architecture == "realbasicvsr":
        from neuralframe.inference.archs.basicvsr import RealBasicVSRNet

        return RealBasicVSRNet(**params)
    if spec.architecture == "rife":
        from neuralframe.inference.archs.rife import IFNet

        return IFNet(**params)
    raise NeuralFrameError(
        ErrorCode.MODEL_UNAVAILABLE, f"architecture {spec.architecture!r} of model {spec.id!r} is not implemented"
    )


SPANDREL_HINT = 'pip install "neuralframe[models]" (install torchvision from the same PyTorch index as torch first)'


def load_spandrel(weights: Path, *, expect_scale: int | None = None, model_id: str = "model") -> Any:
    """Build a network from a checkpoint with spandrel (architecture detected from the weights).

    Returns spandrel's ``ImageModelDescriptor``; only 3-channel super-resolution models are accepted."""
    try:
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError as exc:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"{model_id} needs the spandrel package",
                               hint=SPANDREL_HINT) from exc
    try:
        descriptor = ModelLoader(device="cpu").load_from_file(str(weights))
    except Exception as exc:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"cannot load {weights.name}: {exc}",
                               hint="the file is not a supported image model (see spandrel's list)") from exc
    if not isinstance(descriptor, ImageModelDescriptor):
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"{weights.name} is not an image upscaling model")
    if descriptor.input_channels != 3 or descriptor.output_channels != 3:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE,
                               f"{weights.name} works on {descriptor.input_channels}-channel images; "
                               "only colour (3-channel) models are supported")
    if expect_scale is not None and descriptor.scale != expect_scale:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE,
                               f"{weights.name} is a {descriptor.scale}x model, the registry says {expect_scale}x")
    return descriptor


class SizeGuard(nn.Module):
    """Pads the input to the size a network needs (multiple of N, minimum side, square) and crops the
    output back, so a registry entry can never send a tile the network cannot take."""

    def __init__(self, net: nn.Module, *, scale: int, multiple: int, minimum: int, square: bool) -> None:
        super().__init__()
        self.net = net
        self.scale = scale
        self.multiple = max(1, multiple)
        self.minimum = max(1, minimum)
        self.square = square

    def _target(self, size: int) -> int:
        size = max(size, self.minimum)
        return -(-size // self.multiple) * self.multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        th, tw = self._target(h), self._target(w)
        if self.square:
            th = tw = max(th, tw)
        if (th, tw) == (h, w):
            return self.net(x)
        padded = torch.nn.functional.pad(x, (0, tw - w, 0, th - h), mode="replicate")
        out = self.net(padded)
        return out[..., : h * self.scale, : w * self.scale]


def load_network(spec: ModelSpec, weights: Path) -> nn.Module:
    """Build the network and load a checkpoint safely (``weights_only=True``: tensors only, no code)."""
    if spec.architecture == "spandrel":
        descriptor = load_spandrel(weights, expect_scale=spec.scale, model_id=spec.id)
        net: nn.Module = descriptor.model
        need = descriptor.size_requirements
        if int(need.multiple_of) > 1 or int(need.minimum) > 1 or bool(need.square):
            net = SizeGuard(net, scale=int(descriptor.scale), multiple=int(need.multiple_of),
                            minimum=int(need.minimum), square=bool(need.square))
        net.nf_supports_half = bool(descriptor.supports_half)  # type: ignore[assignment]
        net.nf_supports_bf16 = bool(descriptor.supports_bfloat16)  # type: ignore[assignment]
        return net.eval()
    net = build_network(spec)
    try:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"cannot read weights {weights}: {exc}") from exc
    if isinstance(state, dict) and spec.state_dict_key and spec.state_dict_key in state:
        state = state[spec.state_dict_key]
    if isinstance(state, dict) and spec.state_dict_prefix:
        prefix = spec.state_dict_prefix
        state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if isinstance(state, dict) and spec.state_dict_exclude:
        state = {k: v for k, v in state.items() if not k.startswith(tuple(spec.state_dict_exclude))}
    elif isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise NeuralFrameError(
            ErrorCode.MODEL_UNAVAILABLE, f"weights do not match the {spec.architecture} architecture: {exc}"
        ) from exc
    return net.eval()
