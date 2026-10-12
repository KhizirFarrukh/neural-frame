"""Default output file names: ``<stem>_<what was done><ext>``, never overwriting an existing file."""

from __future__ import annotations

from pathlib import Path

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def suggest_output(
    input_path: Path,
    *,
    upscale: bool = True,
    interpolate: bool = False,
    target_fps: str | None = None,
    folder: Path | None = None,
) -> Path:
    """E.g. ``clip_upscaled_60fps.mkv`` next to the input (or in ``folder``). Videos default to Matroska,
    which holds every audio/subtitle codec; images keep their format."""
    image = is_image_path(input_path)
    tags = []
    if upscale:
        tags.append("upscaled")
    if interpolate and not image:
        tags.append(f"{target_fps.replace('/', '-')}fps" if target_fps else "framegen")
    suffix = input_path.suffix.lower() if image else ".mkv"
    stem = f"{input_path.stem}_{'_'.join(tags) or 'out'}"
    folder = folder or input_path.parent
    candidate = folder / f"{stem}{suffix}"
    n = 2
    while candidate.exists():
        candidate = folder / f"{stem}_{n}{suffix}"
        n += 1
    return candidate
