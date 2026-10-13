"""Automatic model choice: frame measurements on synthetic images and the choosing rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageDraw

from neuralframe.config import load_settings
from neuralframe.core.job import JobRunner, JobSpec
from neuralframe.io.media import MediaInfo, PixelFormat, VideoStream
from neuralframe.models.advisor import (
    ContentProfile,
    _luma,
    effective_height,
    estimate_blockiness,
    estimate_noise,
    is_cartoon_frame,
    recommend,
)
from neuralframe.models.registry import ModelRegistry, ModelSpec


def natural(seed: int, h: int = 360, w: int = 640) -> npt.NDArray[np.uint8]:
    """1/f-spectrum image (the statistics of photographs): soft shading at every scale, light grain."""
    rng = np.random.default_rng(seed)
    fy, fx = np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spectrum = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) / f
    spectrum[0, 0] = 0
    img = np.fft.irfft2(spectrum, s=(h, w))
    img = (img - img.mean()) / (img.std() * 10) + 0.5
    rgb = np.stack([img, img * 0.8 + 0.1, 1 - img * 0.6], axis=-1) * 255 + rng.normal(0, 1.5, (h, w, 1))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def cartoon(seed: int, h: int = 360, w: int = 640) -> npt.NDArray[np.uint8]:
    """Flat colour areas with outlines."""
    rng = np.random.default_rng(seed)
    im = Image.new("RGB", (w, h), (120, 180, 230))
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, int(h * 0.7), w, h), fill=(90, 170, 70))
    for _ in range(12):
        x, y, s = int(rng.integers(0, w - 80)), int(rng.integers(0, h - 80)), int(rng.integers(30, 120))
        colour = (int(rng.integers(0, 255)), int(rng.integers(0, 255)), int(rng.integers(0, 255)))
        if rng.random() < 0.5:
            draw.ellipse((x, y, x + s, y + s), fill=colour, outline=(0, 0, 0), width=3)
        else:
            draw.rectangle((x, y, x + s, y + s // 2), fill=colour, outline=(0, 0, 0), width=3)
    return np.asarray(im)


# ---- measurements ------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(6))
def test_cartoons_and_natural_images_are_told_apart(seed: int) -> None:
    assert is_cartoon_frame(cartoon(seed))
    assert not is_cartoon_frame(natural(seed))


def test_noise_estimate_ignores_shading_and_measures_grain() -> None:
    yy, xx = np.mgrid[0:360, 0:640].astype(np.float32)
    shading = (60 + 0.2 * xx + 0.15 * yy).astype(np.float32)
    grain = np.random.default_rng(1).normal(0, 5, shading.shape).astype(np.float32)
    assert estimate_noise(shading) < 0.1
    assert estimate_noise(shading + grain) == pytest.approx(5.0, abs=0.75)


def test_blockiness_finds_8x8_blocks_at_any_grid_offset() -> None:
    gray = _luma(natural(3))
    h, w = (gray.shape[0] // 8) * 8, (gray.shape[1] // 8) * 8
    means = gray[:h, :w].reshape(h // 8, 8, w // 8, 8).mean(axis=(1, 3))
    blocks = np.repeat(np.repeat(means, 8, axis=0), 8, axis=1).astype(np.float32)
    half = (0.5 * blocks + 0.5 * gray[:h, :w]).astype(np.float32)
    assert estimate_blockiness(gray) < 1.1
    assert estimate_blockiness(half) > 1.3
    assert estimate_blockiness(half[3:, 5:]) == pytest.approx(estimate_blockiness(half), rel=0.05)
    assert estimate_blockiness(blocks) > 5  # completely flat blocks: the worst case, not "no data"
    assert estimate_blockiness(np.full((64, 64), 100, np.float32)) == 1.0


def test_effective_height_sees_through_upscaled_footage() -> None:
    sharp = _luma(natural(4))
    small = Image.fromarray(np.clip(sharp, 0, 255).astype(np.uint8)).resize((320, 180), Image.Resampling.BICUBIC)
    soft = np.asarray(small.resize((640, 360), Image.Resampling.BICUBIC), dtype=np.float32)
    assert effective_height(sharp, 0.0) == 360
    assert effective_height(soft, 0.0) <= 216


# ---- choosing ----------------------------------------------------------------------------------------


def model(model_id: str, **fields: Any) -> ModelSpec:
    base: dict[str, Any] = {"kind": "sr", "scale": 4, "architecture": "srvgg", "content": ["live-action"],
                            "speed": "fast", "rank": 50}
    return ModelSpec(id=model_id, **{**base, **fields})


REGISTRY = ModelRegistry([
    model("lanczos", kind="classical", architecture=None, content=[], rank=0),
    model("general", handles=["noise", "compression"], content=["live-action", "animation"], rank=60),
    model("detail", speed="very-slow", rank=95),
    model("restore", speed="very-slow", rank=92, handles=["noise", "compression", "blur"]),
    model("strong", speed="slow", rank=65, handles=["noise", "compression"]),
    model("two-x", scale=2, speed="medium", rank=82, handles=["noise"]),
    model("anime-video", content=["animation"], rank=75),
    model("anime-stills", content=["animation"], speed="medium", rank=80, stills=True),
    model("temporal", input="recurrent", window=5, rank=99),
    model("unranked", rank=0),
])


def profile(content: str = "live-action", degradations: tuple[str, ...] = (), height: int = 540,
            effective: int | None = None) -> ContentProfile:
    return ContentProfile(width=960, height=height, frames=8, animation_score=1.0 if content == "animation" else 0.0,
                          content=content, noise=3.0 if "noise" in degradations else 0.5, blockiness=1.0,
                          bits_per_pixel=None, effective_height=effective or height,
                          degradations=list(degradations), notes=[])


def choose(p: ContentProfile, factor: float = 4.0, **kw: Any) -> str:
    return recommend(p, REGISTRY, factor=factor, **kw).model


def test_clean_footage_gets_the_most_detailed_model_the_preset_allows() -> None:
    assert choose(profile(), preset="quality") == "detail"  # restoration models lose points on clean footage
    assert choose(profile(), preset="balanced") == "strong"  # "detail" and "restore" are too slow for it
    assert choose(profile(), preset="fast") == "general"
    assert "clean source: keeps all detail" in recommend(profile(), REGISTRY, factor=4.0, preset="quality").reasons


def test_degraded_footage_gets_a_model_that_removes_the_damage() -> None:
    noisy = profile(degradations=("noise", "compression"))
    assert choose(noisy, preset="quality") == "restore"
    assert choose(noisy, preset="balanced") == "strong"
    choice = recommend(noisy, REGISTRY, factor=4.0, preset="quality")
    assert any("removes compression, noise" in r for r in choice.reasons)
    assert "detail" in choice.alternatives


def test_animation_prefers_video_models_but_images_may_use_still_models() -> None:
    assert choose(profile("animation"), preset="balanced") == "anime-video"
    assert choose(profile("animation"), preset="balanced", is_image=True) == "anime-stills"


def test_2x_models_only_when_one_pass_is_enough() -> None:
    assert choose(profile(degradations=("noise",)), factor=2.0, preset="balanced") == "two-x"
    assert choose(profile(degradations=("noise",)), factor=4.0, preset="balanced") == "strong"


def test_blurry_footage_counts_as_a_bigger_upscale() -> None:
    blurry = profile(degradations=("noise",), effective=270)  # 540p file with 270p of real detail
    choice = recommend(blurry, REGISTRY, factor=2.0, preset="balanced")
    assert choice.model != "two-x"
    assert any("effective upscale 4.0x" in r for r in choice.reasons)


def test_cpu_only_uses_fast_models_and_temporal_or_unranked_never() -> None:
    assert choose(profile(), preset="quality", gpu=False) == "general"
    picks = {choose(profile(c, d), preset=p) for c in ("live-action", "animation")
             for d in ((), ("noise",)) for p in ("fast", "balanced", "quality")}
    assert not picks & {"temporal", "unranked", "lanczos"}


def test_fallback_when_nothing_fits() -> None:
    empty = ModelRegistry([model("lanczos", kind="classical", architecture=None, content=[], rank=0)])
    assert recommend(profile(), empty, factor=4.0).model == "realesr-general-x4v3"
    assert recommend(profile(), empty, factor=4.0, gpu=False).model == "lanczos"


# ---- built-in registry -------------------------------------------------------------------------------


def test_builtin_models_are_annotated_for_automatic_choice() -> None:
    registry = ModelRegistry.load(user=False)
    ranked = [m for m in registry.all() if m.kind == "sr" and m.rank > 0]
    assert len(ranked) >= 12
    for m in ranked:
        assert m.speed and m.content, m.id
        assert m.weights is not None and len(m.weights.sha256) == 64, m.id
    for m in registry.all():
        if m.architecture == "spandrel":
            assert m.backends == ["torch"] and m.allows("torch") and not m.allows("tensorrt"), m.id
            assert m.memory_per_pixel, m.id
        if not m.fp16 and m.rank > 0:
            assert m.speed == "very-slow", m.id  # the ranked transformers overflow in FP16 (bfloat16 instead)
    assert all(m.content for m in ranked if m.speed == "fast")  # CPU fallback always has something
    assert {"live-action", "animation"} <= {c for m in ranked if m.speed == "fast" for c in m.content}


def test_builtin_choices_for_typical_videos() -> None:
    pytest.importorskip("spandrel")
    registry = ModelRegistry.load(user=False)

    def pick(p: ContentProfile, preset: str) -> str:
        return recommend(p, registry, factor=4.0, preset=preset).model

    assert pick(profile("animation"), "balanced") == "realesr-animevideov3"
    assert pick(profile(), "quality") == "hat-l-x4"
    assert pick(profile(degradations=("noise", "compression")), "quality") == "dat2-real-x4"
    assert registry.get(pick(profile(), "fast")).speed == "fast"


# ---- jobs --------------------------------------------------------------------------------------------


def media_960x540() -> MediaInfo:
    video = VideoStream(index=0, codec="h264", width=960, height=540, pix_fmt="yuv420p",
                        pixel_format=PixelFormat(name="yuv420p", bit_depth=8, log2_chroma_w=1, log2_chroma_h=1),
                        r_frame_rate=24, avg_frame_rate=24, duration=10.0)
    return MediaInfo(path=Path("in.mp4"), format_name="mov,mp4", video=video, duration=10.0)


def test_jobs_resolve_auto_with_the_content_analysis(monkeypatch: pytest.MonkeyPatch, settings: Any,
                                                     tools: Any) -> None:
    settings.inference.device = "cpu"  # CPU: fast models only
    registry = ModelRegistry.load(user=False)
    monkeypatch.setattr("neuralframe.models.advisor.analyse_content", lambda *a, **k: profile("animation"))
    spec = JobSpec(input=Path("in.mp4"), output=Path("out.mkv"))  # no model: the preset's, now "auto"
    runner = JobRunner(settings, registry=registry)
    assert runner.resolve_model(spec, tools, media_960x540()).id == "realesr-animevideov3"
    assert runner.model_choice is not None and runner.model_choice["model"] == "realesr-animevideov3"
    assert runner.model_choice["factor"] == 4.0  # 540p -> 2160p
    assert any("animation" in r for r in runner.model_choice["reasons"])

    explicit = JobRunner(settings, registry=registry)
    assert explicit.resolve_model(spec.model_copy(update={"model": "lanczos"}), tools, media_960x540()).id == "lanczos"
    assert explicit.model_choice is None
    assert explicit.resolve_model(spec.model_copy(update={"upscale": False, "interpolate": True})).id == "lanczos"


def test_auto_never_stops_a_job_when_the_analysis_fails(monkeypatch: pytest.MonkeyPatch, settings: Any,
                                                        tools: Any) -> None:
    def broken(*_a: Any, **_k: Any) -> ContentProfile:
        raise RuntimeError("no frames could be read")

    monkeypatch.setattr("neuralframe.models.advisor.analyse_content", broken)
    spec = JobSpec(input=Path("in.mp4"), output=Path("out.mkv"), model="auto")
    runner = JobRunner(settings, registry=ModelRegistry.load(user=False))
    assert runner.resolve_model(spec, tools, media_960x540()).id == "realesr-general-x4v3"
    assert runner.model_choice is not None and "content analysis failed" in runner.model_choice["reasons"][0]
    assert JobRunner(settings).resolve_model(spec).id == "realesr-general-x4v3"  # nothing to analyse


def test_presets_choose_automatically() -> None:
    settings = load_settings(None)
    for name in ("fast", "balanced", "quality"):
        assert settings.preset(name).model == "auto", name
