"""What a user sees about an engine model: is its engine installed, are its files downloaded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from neuralframe.engines.catalog import engine
from neuralframe.engines.env import EngineEnv, EnvState, InstallProgress
from neuralframe.engines.process import Cancellable
from neuralframe.models.library import DownloadProgress, ModelLibrary, ModelStatus, ProgressCallback
from neuralframe.models.registry import ModelSpec

if TYPE_CHECKING:
    from neuralframe.config import Settings


@dataclass
class EngineModelStatus:
    model: str
    engine: str
    env_state: EnvState
    env_gb: float
    files: ModelStatus | None
    """None for engines that download their own weights."""

    @property
    def ready(self) -> bool:
        return self.env_state == "installed" and (self.files is None or self.files.ready)

    @property
    def download_bytes(self) -> int:
        files = 0 if self.files is None or self.files.ready else (self.files.size_bytes or 0)
        return files + (0 if self.env_state == "installed" else int(self.env_gb * 1e9))

    def text(self) -> str:
        if self.ready:
            return "Installed and ready"
        parts = []
        if self.env_state == "missing":
            parts.append(f"engine not installed (about {self.env_gb:.0f} GB)")
        elif self.env_state == "outdated":
            parts.append("engine needs an update (reinstalled when used)")
        if self.files is not None and not self.files.ready:
            parts.append(self.files.text().replace("Not downloaded", "model not downloaded"))
        return "; ".join(parts).capitalize()


def engine_env(spec: ModelSpec, settings: Settings) -> EngineEnv:
    assert spec.engine is not None
    return EngineEnv.for_settings(engine(spec.engine), settings)


def engine_model_status(spec: ModelSpec, settings: Settings, library: ModelLibrary) -> EngineModelStatus:
    env = engine_env(spec, settings)
    return EngineModelStatus(model=spec.id, engine=env.engine.id, env_state=env.state(), env_gb=env.engine.disk_gb,
                             files=library.status(spec) if spec.all_weights else None)


def install_model(spec: ModelSpec, settings: Settings, library: ModelLibrary, *,
                  progress: ProgressCallback | None = None, cancel: Cancellable | None = None) -> None:
    """Install the engine (if needed) and download the model's files, reporting both as one progress."""
    env = engine_env(spec, settings)
    if env.state() != "installed":
        def on_install(p: InstallProgress) -> None:
            if progress is not None:
                progress(DownloadProgress(spec.id, int(p.fraction * 1000), 1000, 0.0, None,
                                          message=f"installing the engine - {p.step}: {p.message}"[:160]))

        env.install(on_install, cancel=cancel)
    if spec.all_weights:
        library.ensure(spec, progress=progress, cancel=cancel)


def describe(spec: ModelSpec) -> dict[str, Any]:
    e = engine(spec.engine or "")
    return {"engine": e.id, "engine_name": e.name, "engine_licence": e.licence, "engine_gb": e.disk_gb,
            "engine_notes": e.notes, "homepage": e.homepage}
