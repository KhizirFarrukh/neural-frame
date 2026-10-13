"""Automatic model choice: look at the video, then pick the model that suits it.

A few frames spread over the video are measured (CPU, a second or two):

* **content** - animation or live action: cartoons have hard steps between flat areas, filmed or rendered
  footage has soft shading (calibrated on excerpts of *Sita Sings the Blues* and *Tears of Steel*);
* **noise** - Immerkaer's estimator on low-gradient pixels (sensor noise, film grain);
* **compression** - 8x8 block-boundary steps relative to the rest of the image, plus bits per pixel;
* **real resolution** - the largest downscale the frame survives unchanged: a "1080p" file that is really
  upscaled 540p has no detail above half resolution, so it needs a 2x stronger model than its size says.

The registry describes every model (``content``, ``handles``, ``speed``, ``rank``); the recommendation is
the best-ranked model for the content whose degradation handling fits the source, at a scale close to the
required factor and within the speed the preset allows.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from neuralframe.io.ffmpeg import FFmpegTools, popen_kwargs
from neuralframe.io.media import MediaInfo
from neuralframe.log import get_logger
from neuralframe.models.registry import ModelRegistry, ModelSpec

log = get_logger(component="advisor")

FloatImage = npt.NDArray[np.float32]

# Speeds each preset may use (videos). Single images may use any speed: it is one frame.
PRESET_SPEEDS = {
    "fast": {"fast"},
    "balanced": {"fast", "medium", "slow"},
    "quality": {"fast", "medium", "slow", "very-slow"},
}
# Bits per pixel below which a codec's output is visibly compressed (rough, per codec efficiency).
_LOW_BPP = {"h264": 0.08, "mpeg4": 0.15, "mpeg2video": 0.2, "hevc": 0.05, "vp9": 0.05, "av1": 0.04}


@dataclass
class ContentProfile:
    width: int
    height: int
    frames: int
    animation_score: float
    content: str
    noise: float
    blockiness: float
    bits_per_pixel: float | None
    effective_height: int
    degradations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("animation_score", "noise", "blockiness"):
            data[key] = round(float(data[key]), 3)
        if self.bits_per_pixel is not None:
            data["bits_per_pixel"] = round(self.bits_per_pixel, 4)
        return data


@dataclass
class Recommendation:
    model: str
    reasons: list[str]
    alternatives: list[str]
    profile: ContentProfile
    factor: float

    def summary(self) -> dict[str, Any]:
        return {"model": self.model, "reasons": self.reasons, "alternatives": self.alternatives,
                "factor": round(self.factor, 2), "profile": self.profile.summary()}


# ---- frame measurements --------------------------------------------------------------------------------


def _grab(tools: FFmpegTools, path: Path, seconds: float, width: int, height: int) -> npt.NDArray[np.uint8] | None:
    args = [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(seconds, 0):.3f}",
            "-i", str(path), "-frames:v", "1", "-an", "-sn", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.run(args, capture_output=True, timeout=60, check=False, **popen_kwargs())
    need = width * height * 3
    if proc.returncode != 0 or len(proc.stdout) < need:
        return None
    return np.frombuffer(proc.stdout[:need], dtype=np.uint8).reshape(height, width, 3)


def _luma(rgb: npt.NDArray[np.uint8]) -> FloatImage:
    f = rgb.astype(np.float32)
    luma: FloatImage = (0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]).astype(np.float32)
    return luma


def estimate_noise(gray: FloatImage) -> float:
    """Noise standard deviation in 8-bit levels (Immerkaer 1996), measured on the flatter half of the image
    so that edges and texture are not mistaken for noise."""
    g = gray
    lap = (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:] - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1]
           - 2 * g[1:-1, 2:] + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
    gx = g[1:-1, 2:] - g[1:-1, :-2]
    gy = g[2:, 1:-1] - g[:-2, 1:-1]
    grad = np.abs(gx) + np.abs(gy)
    flat = grad <= np.percentile(grad, 50)
    if not flat.any():
        return 0.0
    return float(math.sqrt(math.pi / 2) * np.abs(lap[flat]).mean() / 6.0)


def estimate_blockiness(gray: FloatImage) -> float:
    """How much stronger the steps on one 8-pixel grid phase are than on the others (1.0 = no blocking).

    All eight phases are tried, so a cropped or shifted video (grid not starting at pixel 0) is measured
    the same way."""
    ratios = []
    for axis in (1, 0):
        steps = np.abs(np.diff(gray, axis=axis)).mean(axis=1 - axis)
        usable = (steps.size // 8) * 8
        if usable < 16:
            continue
        by_phase = steps[:usable].reshape(-1, 8).mean(axis=0)
        peak, others = float(np.max(by_phase)), float(np.median(by_phase))
        if peak < 0.05:
            continue  # flat along this axis: nothing to measure
        # Flat blocks (others ~0, the worst compression) must read as very blocky, not be skipped.
        ratios.append(min(peak / max(others, 0.05), 10.0))
    return float(np.mean(ratios)) if ratios else 1.0


def cartoon_features(rgb: npt.NDArray[np.uint8]) -> tuple[float, float]:
    """(flat fraction, hard/soft edge ratio) of a 320-pixel-wide thumbnail.

    Cartoons have hard steps between flat areas; filmed and rendered footage (also CG films) has soft
    shading. Measured: cartoon frames 0.41-2.2 hard/soft (one flat-shaded scene 0.23 with 52 % flat
    blocks), live action 0.15-0.32, photo-real CG 0.11-0.16."""
    from PIL import Image

    small = np.asarray(Image.fromarray(rgb).resize((320, max(2, round(320 * rgb.shape[0] / rgb.shape[1]))),
                                                   Image.Resampling.BOX), dtype=np.uint8)
    lum = _luma(small)
    h, w = (lum.shape[0] // 4) * 4, (lum.shape[1] // 4) * 4
    flat = float((lum[:h, :w].reshape(h // 4, 4, w // 4, 4).std(axis=(1, 3)) < 1.5).mean())
    grad = np.maximum(np.abs(np.diff(lum, axis=1))[:-1, :], np.abs(np.diff(lum, axis=0))[:, :-1])
    soft = float(((grad > 2) & (grad <= 12)).mean())
    hard = float((grad > 24).mean())
    return flat, hard / max(soft, 1e-3)


def is_cartoon_frame(rgb: npt.NDArray[np.uint8]) -> bool:
    flat, ratio = cartoon_features(rgb)
    return ratio >= 0.35 or (flat >= 0.5 and ratio >= 0.2)


def effective_height(gray: FloatImage, noise: float) -> int:
    """Height of the detail actually present: the strongest downscale whose re-upscale only loses noise."""
    from PIL import Image

    h, w = gray.shape
    img = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8))
    allowed = 1.6 * noise * noise + 2.0  # residual a noisy but otherwise unchanged frame would show
    best = h
    for factor in (0.75, 0.6, 0.5, 0.4):
        sw, sh = max(8, round(w * factor)), max(8, round(h * factor))
        back = img.resize((sw, sh), Image.Resampling.LANCZOS).resize((w, h), Image.Resampling.LANCZOS)
        residual = float(np.mean((np.asarray(back, dtype=np.float32) - gray) ** 2))
        if residual <= allowed:
            best = round(h * factor)
        else:
            break
    return int(best)


def analyse_content(tools: FFmpegTools, path: Path, media: MediaInfo, samples: int = 8) -> ContentProfile:
    """Measure ``samples`` frames spread over the video (one for images)."""
    video = media.require_video()
    width, height = video.width, video.height
    duration = media.duration or video.duration or 0.0
    times = [0.0] if media.is_image or duration <= 0 else [duration * (0.05 + 0.9 * i / max(samples - 1, 1))
                                                             for i in range(samples)]
    frames = [f for f in (_grab(tools, path, t, width, height) for t in times) if f is not None]
    if not frames:
        frames = [f for f in [_grab(tools, path, 0.0, width, height)] if f is not None]
    if not frames:
        raise RuntimeError(f"no frames could be read from {path.name}")
    grays = [_luma(f) for f in frames]
    noise = float(np.median([estimate_noise(g) for g in grays]))
    blocky = float(np.median([estimate_blockiness(g) for g in grays]))
    anim = float(np.mean([is_cartoon_frame(f) for f in frames]))  # share of frames that look like cartoons
    eff = int(np.median([effective_height(g, noise) for g in grays]))

    bpp = None
    rate = video.bit_rate or media.bit_rate
    if rate and video.fps and float(video.fps) > 0 and not media.is_image:
        audio = sum(a.bit_rate or 0 for a in media.audio) if video.bit_rate is None else 0
        bpp = max(rate - audio, 0) / (width * height * float(video.fps))
    degradations: list[str] = []
    notes: list[str] = []
    if noise > 2.0:
        degradations.append("noise")
        notes.append(f"noise / grain about {noise:.1f} levels")
    low_bpp = _LOW_BPP.get(video.codec, 0.06)
    if blocky > 1.15 or (bpp is not None and bpp < low_bpp):
        degradations.append("compression")
        reason = f"blocking ratio {blocky:.2f}" if blocky > 1.15 else f"low bitrate ({bpp:.3f} bits/pixel)"
        notes.append(f"compression artefacts likely ({reason})")
    if eff < 0.7 * height:
        degradations.append("blur")
        notes.append(f"soft image: real detail only about {eff}p in a {height}p frame")
    content = "animation" if anim >= 0.5 else "live-action"
    return ContentProfile(width=width, height=height, frames=len(frames), animation_score=anim, content=content,
                          noise=noise, blockiness=blocky, bits_per_pixel=bpp, effective_height=eff,
                          degradations=degradations, notes=notes)


# ---- choosing ------------------------------------------------------------------------------------------


def _usable(model: ModelSpec) -> bool:
    if model.kind != "sr" or model.input != "single" or not model.content or model.rank <= 0:
        return False
    return model.architecture != "spandrel" or importlib.util.find_spec("spandrel") is not None


def recommend(
    profile: ContentProfile,
    registry: ModelRegistry,
    *,
    factor: float,
    preset: str = "balanced",
    is_image: bool = False,
    gpu: bool = True,
) -> Recommendation:
    """Best model for ``profile`` at an upscale ``factor`` (output height / source height)."""
    speeds = {"fast", "medium", "slow", "very-slow"} if is_image else PRESET_SPEEDS.get(preset,
                                                                                         PRESET_SPEEDS["balanced"])
    if not gpu:
        speeds = {"fast"} if not is_image else {"fast", "medium"}
    needed = set(profile.degradations)
    # The detail to rebuild is relative to the real resolution, not the file's.
    true_factor = factor * profile.height / max(profile.effective_height, 1)
    scored: list[tuple[float, ModelSpec, list[str]]] = []
    for model in registry.all():
        if not _usable(model) or profile.content not in model.content or model.speed not in speeds:
            continue
        score = float(model.rank)
        why = [f"made for {profile.content}"]
        missing = needed - set(model.handles)
        if missing:
            score -= 15 * len(missing)
            why.append(f"does not remove {', '.join(sorted(missing))}")
        elif needed:
            why.append(f"removes {', '.join(sorted(needed))}")
        if not needed and model.handles:
            score -= 8  # restoration models smooth clean footage
            why.append("clean source: a restoration model would smooth it slightly")
        elif not needed:
            why.append("clean source: keeps all detail")
        if model.stills and not is_image:
            score -= 10
            why.append("made for still images (less steady between frames)")
        if true_factor <= 2.5:
            score += 6 if model.scale == 2 else 0
        elif model.scale == 2:
            score -= 18  # two passes: slower, and restoration artefacts compound
            why.append("needs two 2x passes for this factor")
        scored.append((score, model, why))
    if not scored:
        fallback = "realesr-general-x4v3" if gpu or is_image else "lanczos"
        return Recommendation(model=fallback, reasons=["no specialised model available for these settings"],
                              alternatives=[], profile=profile, factor=factor)
    scored.sort(key=lambda s: (-s[0], s[1].id))
    best_score, best, why = scored[0]
    share = round(profile.animation_score * profile.frames)
    reasons = [f"{profile.content.replace('-', ' ')} ({share} of {profile.frames} sampled frames look like "
               "animation)", *(profile.notes or ["clean source: no noise, compression or blur found"]),
               *why[1:]]
    if true_factor > factor * 1.05:
        reasons.append(f"effective upscale {true_factor:.1f}x (file {factor:.1f}x)")
    log.info("model recommendation", model=best.id, score=round(best_score, 1), content=profile.content,
             degradations=profile.degradations)
    return Recommendation(model=best.id, reasons=reasons, alternatives=[m.id for _, m, _ in scored[1:4]],
                          profile=profile, factor=factor)
