"""The Hugging Face access token some engines need (gated weights such as FLUX.1-dev).

Looked up in this order: ``engines.hf_token`` in the settings, the ``HF_TOKEN`` (or
``HUGGING_FACE_HUB_TOKEN``) environment variable, then the token saved from the desktop app in the
settings folder (``hf_token``, readable only by the user where the system supports it).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuralframe.config import Settings


def token_path() -> Path:
    from neuralframe.config import user_config_dir

    return user_config_dir() / "hf_token"


def hf_token(settings: Settings | None = None) -> str | None:
    token = settings.engines.hf_token if settings is not None else None
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        path = token_path()
        try:
            token = path.read_text(encoding="utf-8").strip() if path.is_file() else None
        except OSError:
            token = None
    return token or None


def save_hf_token(token: str | None) -> None:
    """Save (or with an empty token, remove) the token in the settings folder."""
    path = token_path()
    if not token:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token.strip(), encoding="utf-8")
    if sys.platform != "win32":
        path.chmod(0o600)
