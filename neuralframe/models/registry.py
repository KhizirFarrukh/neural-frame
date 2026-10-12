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

    @property
    def is_classical(self) -> bool:
        return self.kind == "classical"

    @property
    def is_temporal(self) -> bool:
        return self.input != "single"

    @property
    def temporal_radius(self) -> int:
        return (self.window - 1) // 2


class ModelRegistry:
    def __init__(self, models: list[ModelSpec]) -> None:
        self._models: dict[str, ModelSpec] = {}
        for spec in models:
            self._models[spec.id] = spec

    @classmethod
    def load(cls, extra: list[Path] | None = None) -> ModelRegistry:
        text = resources.files("neuralframe.data").joinpath("registry.yaml").read_text(encoding="utf-8")
        specs = _parse(yaml.safe_load(text))
        for path in extra or []:
            specs += _parse(yaml.safe_load(path.read_text(encoding="utf-8")))
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


def _parse(raw: Any) -> list[ModelSpec]:
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise NeuralFrameError(ErrorCode.CONFIG_INVALID, "model registry must contain a 'models' list")
    try:
        return [ModelSpec.model_validate(item) for item in raw["models"]]
    except Exception as exc:
        raise NeuralFrameError(ErrorCode.CONFIG_INVALID, f"invalid model registry entry: {exc}") from exc
