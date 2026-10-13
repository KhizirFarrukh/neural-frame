"""Weight files: compatibility helpers over :mod:`neuralframe.models.library` (which handles downloads,
local files and updates). Weights live in the models folder, never in git."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.models.library import DownloadProgress, ModelLibrary, sha256_of
from neuralframe.models.registry import ModelSpec

__all__ = ["DownloadProgressCallback", "ensure_weights", "sha256_of", "verify_weights", "weights_path"]

DownloadProgressCallback = Callable[[int, int | None], None]


def weights_path(spec: ModelSpec, models_dir: Path) -> Path:
    if spec.weights is None:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"model {spec.id!r} has no weight file")
    return ModelLibrary(models_dir).target_path(spec)


def verify_weights(spec: ModelSpec, models_dir: Path) -> bool:
    return ModelLibrary(models_dir).is_ready(spec)


def ensure_weights(spec: ModelSpec, models_dir: Path, *, progress: DownloadProgressCallback | None = None,
                   timeout_s: float = 60.0) -> Path:
    """The verified weight file, downloaded first if needed (see :meth:`ModelLibrary.ensure`)."""

    def adapt(event: DownloadProgress) -> None:
        if progress is not None:
            progress(event.done, event.total)

    return ModelLibrary(models_dir, timeout_s=timeout_s).ensure(spec, progress=adapt if progress else None)
