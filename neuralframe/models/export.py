"""PyTorch weights → ONNX (opset ≥ 17, dynamic batch/H/W), with a numeric parity check."""

from __future__ import annotations

import contextlib
import io
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.archs import load_network
from neuralframe.log import get_logger
from neuralframe.models.download import sha256_of
from neuralframe.models.registry import ModelSpec

log = get_logger(component="export")

DEFAULT_OPSET = 17


def onnx_path(spec: ModelSpec, models_dir: Path, opset: int = DEFAULT_OPSET) -> Path:
    return models_dir / spec.id / f"{spec.id}.opset{opset}.onnx"


def _sidecar(path: Path) -> Path:
    return path.with_suffix(".json")


def _export(net: torch.nn.Module, example: torch.Tensor, out: Path, opset: int, multiple: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        # The TorchScript exporter handles these pure-convolution networks quickly and robustly.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            torch.onnx.export(
                net, (example,), str(out), dynamo=False, opset_version=opset,
                input_names=["input"], output_names=["output"],
                dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                              "output": {0: "batch", 2: "out_height", 3: "out_width"}},
            )
        return
    except (TypeError, RuntimeError) as exc:  # legacy exporter removed in a future torch
        log.info("TorchScript ONNX export unavailable, using torch.export", reason=str(exc)[:120])
    dim = torch.export.Dim
    h = dim("h", min=8, max=8192) if multiple == 1 else multiple * dim("h_m", min=4, max=8192 // multiple)
    w = dim("w", min=8, max=8192) if multiple == 1 else multiple * dim("w_m", min=4, max=8192 // multiple)
    # The torch.export path prints progress (with emoji) that a cp1252 console cannot encode.
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(
            net, (example,), str(out), dynamo=True, opset_version=max(opset, 18),
            input_names=["input"], output_names=["output"],
            dynamic_shapes={"x": {0: dim("batch", min=1, max=64), 2: h, 3: w}},
        )


def export_onnx(spec: ModelSpec, weights: Path, out: Path, *, opset: int = DEFAULT_OPSET) -> Path:
    if opset < 17:
        raise NeuralFrameError(ErrorCode.CONFIG_INVALID, "ONNX export needs opset 17 or newer")
    t0 = time.perf_counter()
    net = load_network(spec, weights).float().eval()
    multiple = max(spec.pad_multiple, 1)
    example = torch.rand(1, 3, 16 * multiple, 24 * multiple)
    tmp = out.with_suffix(".tmp.onnx")
    try:
        _export(net, example, tmp, opset, multiple)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise NeuralFrameError(ErrorCode.ENGINE_BUILD_FAILED, f"ONNX export of {spec.id} failed: {exc}") from exc
    tmp.replace(out)
    meta = {"model": spec.id, "weights_sha256": sha256_of(weights), "opset": opset,
            "torch": torch.__version__, "exported_at": time.time()}
    _sidecar(out).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("exported ONNX", model=spec.id, path=str(out), seconds=round(time.perf_counter() - t0, 1))
    return out


def ensure_onnx(spec: ModelSpec, weights: Path, models_dir: Path, *, opset: int = DEFAULT_OPSET) -> Path:
    """Return a current ONNX export, re-exporting if the weights changed since the last export."""
    out = onnx_path(spec, models_dir, opset)
    side = _sidecar(out)
    if out.is_file() and side.is_file():
        with contextlib.suppress(Exception):
            meta = json.loads(side.read_text(encoding="utf-8"))
            if meta.get("weights_sha256") == (spec.weights.sha256 if spec.weights else None):
                return out
    return export_onnx(spec, weights, out, opset=opset)


@dataclass(frozen=True)
class ParityReport:
    reference: str
    candidate: str
    samples: int
    max_abs: float
    mean_abs: float
    psnr_db: float

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def compare_outputs(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float]:
    ref = reference.float().clamp(0, 1)
    cand = candidate.float().clamp(0, 1)
    diff = (ref - cand).abs()
    mse = float((diff**2).mean())
    psnr = math.inf if mse == 0 else 10 * math.log10(1.0 / mse)
    return float(diff.max()), float(diff.mean()), psnr


def sample_tiles(n: int, height: int, width: int, seed: int = 0) -> torch.Tensor:
    """Natural-ish test tiles: smooth gradients, edges and mild noise (not pure noise, which SR ignores)."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    tiles = []
    for i in range(n):
        f = 3 + 5 * torch.rand(3, generator=g)
        base = torch.stack([0.5 + 0.4 * torch.sin(f[0] * 6.28 * xx + i), 0.5 + 0.4 * torch.cos(f[1] * 6.28 * yy),
                            ((xx * f[2] + yy * 2).floor() % 2) * 0.6 + 0.2])
        tiles.append((base + 0.02 * torch.randn(3, height, width, generator=g)).clamp(0, 1))
    return torch.stack(tiles)
