"""Files shipped next to a packaged (PyInstaller) build: FFmpeg binaries and model weights.

A Windows release is a folder with ``NeuralFrame.exe``, ``neuralframe-cli.exe``, ``ffmpeg/`` and ``models/``.
When running from source these lookups return ``None`` and the normal search applies (configuration,
``PATH``, the per-user model cache).
"""

from __future__ import annotations

import sys
from pathlib import Path


def app_dir() -> Path | None:
    """Folder containing the packaged executables, or None when running from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return None


def bundled(*parts: str) -> Path | None:
    """``<app>/<parts...>`` if this is a packaged build and the path exists."""
    root = app_dir()
    if root is None:
        return None
    path = root.joinpath(*parts)
    return path if path.exists() else None
