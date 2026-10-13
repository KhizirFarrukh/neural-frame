"""Your own models: any image upscaling model file spandrel can load (e.g. from OpenModelDB).

``neuralframe models add FILE`` checks the file (architecture, scale, colour channels), copies it into the
model folder and records it in the per-user registry (``models.yaml`` in the settings folder); from then on
it is listed and usable everywhere (``-m <id>``, the desktop app, the web interface).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.models.download import sha256_of
from neuralframe.models.registry import ModelRegistry, ModelSpec, WeightsSpec, user_registry_path

# Not ".pt": spandrel opens those as TorchScript, which can carry code. ".pth" goes through a restricted
# unpickler (tensors and plain containers only) and ".safetensors" holds tensors only.
MODEL_SUFFIXES = (".pth", ".safetensors", ".ckpt")
# Attention-based architectures: an order of magnitude slower and hungrier than CNNs.
TRANSFORMERS = {"SwinIR", "Swin2SR", "HAT", "DAT", "DRCT", "ATD", "SRFormer", "RGT", "OmniSR", "GRL", "IPT"}
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


def suggest_id(path: Path) -> str:
    """``4x-UltraSharp.pth`` -> ``4x-ultrasharp``."""
    stem = re.sub(r"[^a-z0-9._-]+", "-", path.stem.lower()).strip("-._")
    return (stem or "model")[:60]


def _speed(arch: str, params: int) -> str:
    if arch in TRANSFORMERS:
        return "very-slow"
    if params < 2_000_000:
        return "fast"
    if params < 10_000_000:
        return "medium"
    return "slow"


def _read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = raw.get("models") if isinstance(raw, dict) else None
    return list(models) if isinstance(models, list) else []


def _write_registry(path: Path, models: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Your own models, managed by `neuralframe models add` / `neuralframe models remove`.\n"
    path.write_text(header + yaml.safe_dump({"models": models}, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")


def add_user_model(
    file: Path,
    models_dir: Path,
    *,
    model_id: str | None = None,
    description: str | None = None,
    licence: str | None = None,
    content: list[str] | None = None,
    handles: list[str] | None = None,
    rank: int = 0,
    purpose: str | None = None,
) -> ModelSpec:
    """Validate ``file``, copy it into ``models_dir/<id>/`` and register it. Returns the new model.

    Automatic model choice only considers it with a ``rank`` (compared with the built-in models' ranks) and
    ``content``; otherwise it is used when chosen by name."""
    from neuralframe.inference.archs import load_spandrel

    file = file.expanduser().resolve()
    if not file.is_file():
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"model file not found: {file}")
    if file.suffix.lower() not in MODEL_SUFFIXES:
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"{file.name}: expected one of {', '.join(MODEL_SUFFIXES)}",
                               hint="a .pt file that holds plain weights can be renamed to .pth; TorchScript "
                               "files are not accepted")
    mid = model_id or suggest_id(file)
    if not _ID.match(mid):
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"invalid model id {mid!r}",
                               hint="use lowercase letters, digits, '-', '_' or '.' (2-63 characters)")
    if not 0 <= rank <= 100:
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"rank must be 0-100, got {rank}")
    if mid == "auto" or mid in {m.id for m in ModelRegistry.load().all()}:
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"a model called {mid!r} already exists",
                               hint="choose another name with --id")
    descriptor = load_spandrel(file, model_id=mid)
    if descriptor.scale < 2:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE,
                               f"{file.name} is a {descriptor.scale}x model; only upscaling models (2x or more) "
                               "are supported")
    arch = str(descriptor.architecture.name)
    params = sum(p.numel() for p in descriptor.model.parameters())
    target = models_dir / mid / file.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target != file:
        shutil.copy2(file, target)
    spec = ModelSpec(
        id=mid,
        kind="sr",
        scale=int(descriptor.scale),
        architecture="spandrel",
        description=description
        or f"{arch} {descriptor.scale}x, added from {file.name} ({params / 1e6:.1f} M parameters)",
        recommended_content=[],
        content=[c for c in (content or []) if c in ("live-action", "animation")],
        handles=[h for h in (handles or []) if h in ("noise", "compression", "blur")],
        speed=_speed(arch, params),
        rank=rank,
        purpose=(purpose or "").strip()[:80],
        backends=["torch"],
        fp16=bool(descriptor.supports_half),
        memory_per_pixel=16000 if arch in TRANSFORMERS else 4000,
        pad_multiple=max(1, int(descriptor.size_requirements.multiple_of)),
        bundled=False,
        licence=licence or "unknown (check where you downloaded it)",
        source_url=file.as_uri(),
        weights=WeightsSpec(url=file.as_uri(), sha256=sha256_of(target), filename=file.name,
                            size_bytes=target.stat().st_size),
        origin="user",
    )
    path = user_registry_path()
    entries = _read_registry(path)
    entries.append(spec.model_dump(mode="json", exclude={"origin"}, exclude_defaults=True))
    _write_registry(path, entries)
    return spec


def remove_user_model(model_id: str, models_dir: Path, *, delete_file: bool = True) -> None:
    path = user_registry_path()
    entries = _read_registry(path)
    kept = [e for e in entries if e.get("id") != model_id]
    if len(kept) == len(entries):
        builtin = model_id in {m.id for m in ModelRegistry.load(user=False).all()}
        what = f"{model_id!r} is a built-in model" if builtin else f"no added model called {model_id!r}"
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, what, hint="`neuralframe models list` shows added models")
    _write_registry(path, kept)
    folder = models_dir / model_id
    if delete_file and folder.is_dir():
        shutil.rmtree(folder)
