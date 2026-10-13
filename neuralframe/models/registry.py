"""Model registry: loads ``registry.yaml`` (packaged) plus optional user registries."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from neuralframe.errors import ErrorCode, NeuralFrameError

ModelKind = Literal["classical", "sr", "interpolation"]
InputKind = Literal["single", "window", "recurrent"]
Speed = Literal["fast", "medium", "slow", "very-slow"]
ContentKind = Literal["live-action", "animation"]
Degradation = Literal["noise", "compression", "blur"]
Backend = Literal["tensorrt", "onnxruntime", "torch"]


class WeightsSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    sha256: str
    filename: str
    size_bytes: int | None = None


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: ModelKind
    scale: int = Field(ge=1, le=8)
    description: str = ""
    input: InputKind = "single"
    window: int = Field(default=1, ge=1)
    architecture: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    pad_multiple: int = Field(default=1, ge=1)
    resize_kernel: str = "lanczos3"
    recommended_content: list[str] = Field(default_factory=list)
    weights: WeightsSpec | None = None
    licence: str = "unknown"
    licence_url: str | None = None
    licence_restricted: bool = False
    source_url: str | None = None
    state_dict_key: str | None = None
    state_dict_prefix: str | None = None
    """Keep only keys with this prefix (stripped), e.g. ``generator_ema.`` in mmedit checkpoints."""
    state_dict_exclude: list[str] = Field(default_factory=list)
    """Key prefixes (after stripping) of training-only tensors to drop, e.g. ``teacher.``."""
    backends: list[Backend] | None = None
    """Inference backends this model may run on (None = all). Transformer models built through spandrel
    are PyTorch-only: their window attention is not validated for ONNX/TensorRT export at other sizes."""
    fp16: bool = True
    """False: the network overflows in FP16 and runs in bfloat16 (or FP32 where bfloat16 is missing)."""
    memory_per_pixel: int | None = Field(default=None, ge=1)
    """Planning estimate of activation memory in bytes per input pixel at 16-bit precision."""
    bundled: bool = True
    """Shipped inside the Windows app; False = downloaded on first use."""
    speed: Speed | None = None
    content: list[ContentKind] = Field(default_factory=list)
    """Footage the model is trained for (used by automatic model choice)."""
    handles: list[Degradation] = Field(default_factory=list)
    """Degradations the model removes; empty = expects clean input."""
    rank: int = 0
    """Relative output quality among models for the same content (higher is better)."""
    stills: bool = False
    """Best on still images (less steady between video frames); automatic choice prefers others for video."""
    origin: Literal["builtin", "user"] = "builtin"
    """``user`` for models added with ``neuralframe models add``."""

    @property
    def is_classical(self) -> bool:
        return self.kind == "classical"

    def allows(self, backend: str) -> bool:
        return self.backends is None or backend in self.backends

    @property
    def is_temporal(self) -> bool:
        return self.input != "single"

    @property
    def temporal_radius(self) -> int:
        return (self.window - 1) // 2

    @property
    def default_overlap(self) -> int:
        """Temporal overlap frames between chunks (window radius x 2; 0 for single-frame models)."""
        return 2 * self.temporal_radius if self.is_temporal else 0


class ModelRegistry:
    def __init__(self, models: list[ModelSpec]) -> None:
        self._models: dict[str, ModelSpec] = {}
        for spec in models:
            self._models[spec.id] = spec

    @classmethod
    def load(cls, extra: list[Path] | None = None, *, user: bool = True) -> ModelRegistry:
        """Built-in models, then ``extra`` registries, then the user's own models (``models add``)."""
        text = resources.files("neuralframe.data").joinpath("registry.yaml").read_text(encoding="utf-8")
        specs = _parse(yaml.safe_load(text))
        for path in extra or []:
            specs += _parse(yaml.safe_load(path.read_text(encoding="utf-8")))
        if user:
            specs += _user_models({s.id for s in specs})
        return cls(specs)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError:
            raise NeuralFrameError(
                ErrorCode.MODEL_UNAVAILABLE,
                f"unknown model {model_id!r}",
                hint=f"available: {', '.join(sorted(self._models))}",
            ) from None

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())

    def of_kind(self, kind: ModelKind) -> list[ModelSpec]:
        return [m for m in self._models.values() if m.kind == kind]


def user_registry_path() -> Path:
    """Where ``neuralframe models add`` records the user's own models."""
    from neuralframe.config import user_config_dir

    return user_config_dir() / "models.yaml"


def _user_models(taken: set[str]) -> list[ModelSpec]:
    """Entries of the user registry. A hand-edited, broken file must not stop the program: invalid entries
    and ids that clash with built-in models are skipped with a warning."""
    from neuralframe.log import get_logger

    path = user_registry_path()
    if not path.is_file():
        return []
    log = get_logger(component="registry")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("ignoring unreadable user model registry", path=str(path), error=str(exc))
        return []
    items = raw.get("models") if isinstance(raw, dict) else None
    specs: list[ModelSpec] = []
    for item in items if isinstance(items, list) else []:
        try:
            spec = ModelSpec.model_validate(item).model_copy(update={"origin": "user"})
        except Exception as exc:
            log.warning("ignoring invalid user model entry", path=str(path), error=str(exc))
            continue
        if spec.id in taken:
            log.warning("ignoring user model with a built-in model's id", model=spec.id, path=str(path))
            continue
        taken.add(spec.id)
        specs.append(spec)
    return specs


def _parse(raw: Any) -> list[ModelSpec]:
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise NeuralFrameError(ErrorCode.CONFIG_INVALID, "model registry must contain a 'models' list")
    try:
        return [ModelSpec.model_validate(item) for item in raw["models"]]
    except Exception as exc:
        raise NeuralFrameError(ErrorCode.CONFIG_INVALID, f"invalid model registry entry: {exc}") from exc
