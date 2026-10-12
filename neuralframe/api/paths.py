"""Path sandboxing for the REST API: every file a client names must resolve inside a configured root.

Paths are resolved strictly (symlinks followed, ``..`` collapsed) before the check, so neither traversal
sequences nor links pointing out of a root get through.
"""

from __future__ import annotations

import re
from pathlib import Path

from neuralframe.errors import ErrorCode, NeuralFrameError


class PathNotAllowed(NeuralFrameError):
    def __init__(self, message: str) -> None:
        hint = "the server only reads and writes inside api.input_roots / api.output_roots"
        super().__init__(ErrorCode.INVALID_INPUT, message, hint=hint)


def _resolve(path: str | Path) -> Path:
    try:
        return Path(path).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:  # RuntimeError: symlink loop
        raise PathNotAllowed(f"cannot resolve path {path!s}: {exc}") from exc


def within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in (_resolve(r) for r in roots))


def check_input(path: str | Path, roots: list[Path]) -> Path:
    """An existing file inside one of ``roots``."""
    resolved = _resolve(path)
    if not within(resolved, roots):
        raise PathNotAllowed(f"input {path!s} is outside the allowed input roots")
    if not resolved.is_file():
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"input file not found: {path!s}")
    return resolved


def check_output(path: str | Path, roots: list[Path]) -> Path:
    """A writable location inside one of ``roots`` (missing parent directories are created)."""
    resolved = _resolve(path)
    if not within(resolved, roots):
        raise PathNotAllowed(f"output {path!s} is outside the allowed output roots")
    if resolved.is_dir():
        raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"output {path!s} is a directory")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


_UNSAFE = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_filename(name: str, default: str = "upload") -> str:
    """Client-supplied file names reduced to one harmless path component."""
    base = Path(name.replace("\\", "/")).name
    cleaned = _UNSAFE.sub("_", base).strip(" .")
    return cleaned[:150] or default
