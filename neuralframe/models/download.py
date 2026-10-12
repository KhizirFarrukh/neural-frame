"""Weight download and verification. Weights live in the cache, never in git."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.log import get_logger
from neuralframe.models.registry import ModelSpec

log = get_logger(component="models")

DownloadProgress = Callable[[int, int | None], None]


def weights_path(spec: ModelSpec, models_dir: Path) -> Path:
    if spec.weights is None:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"model {spec.id!r} has no weight file")
    return models_dir / spec.id / spec.weights.filename


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def verify_weights(spec: ModelSpec, models_dir: Path) -> bool:
    path = weights_path(spec, models_dir)
    return path.is_file() and spec.weights is not None and sha256_of(path) == spec.weights.sha256


def ensure_weights(
    spec: ModelSpec,
    models_dir: Path,
    *,
    progress: DownloadProgress | None = None,
    timeout_s: float = 60.0,
) -> Path:
    """Return the verified weight file, downloading it on first use (streamed, SHA256-checked)."""
    if spec.weights is None:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"model {spec.id!r} has no weight file")
    path = weights_path(spec, models_dir)
    if spec.licence_restricted:
        log.warning("model licence restricts use", model=spec.id, licence=spec.licence, url=spec.licence_url)
    if path.is_file():
        if sha256_of(path) == spec.weights.sha256:
            return path
        log.warning("cached weights failed SHA256 check; downloading again", path=str(path))
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    log.info("downloading weights", model=spec.id, url=spec.weights.url)
    try:
        with httpx.stream("GET", spec.weights.url, follow_redirects=True, timeout=timeout_s) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0)) or spec.weights.size_bytes
            done = 0
            with partial.open("wb") as fh:
                for block in response.iter_bytes(1 << 20):
                    fh.write(block)
                    digest.update(block)
                    done += len(block)
                    if progress is not None:
                        progress(done, total)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise NeuralFrameError(
            ErrorCode.DOWNLOAD_FAILED,
            f"cannot download weights for {spec.id}: {exc}",
            hint=f"download {spec.weights.url} manually to {path}",
        ) from exc

    actual = digest.hexdigest()
    if actual != spec.weights.sha256:
        partial.unlink(missing_ok=True)
        raise NeuralFrameError(
            ErrorCode.DOWNLOAD_FAILED,
            f"SHA256 mismatch for {spec.id}: expected {spec.weights.sha256}, got {actual}",
        )
    partial.replace(path)
    log.info("weights verified", model=spec.id, path=str(path))
    return path
