"""Your own models (`neuralframe models add`): checking, copying, registering, removing, running."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("spandrel")
pytest.importorskip("torch")

import torch

from neuralframe.errors import NeuralFrameError
from neuralframe.inference.archs import load_network
from neuralframe.models.custom import add_user_model, remove_user_model, suggest_id
from neuralframe.models.registry import ModelRegistry, user_registry_path
from tests.models import tiny_swinir_file as tiny_swinir


def test_suggested_ids_are_command_line_friendly() -> None:
    assert suggest_id(Path("4x-UltraSharp.pth")) == "4x-ultrasharp"
    assert suggest_id(Path("My Model (v2).safetensors")) == "my-model-v2"


def test_add_registers_and_copies_the_model(tmp_path: Path, user_config: Path) -> None:
    src = tiny_swinir(tmp_path / "downloads")
    models_dir = tmp_path / "models"
    spec = add_user_model(src, models_dir, content=["live-action", "cartoons"], handles=["noise"],
                          licence="CC0-1.0")
    assert spec.id == "4x-tiny_swinir" and spec.scale == 2 and spec.architecture == "spandrel"
    assert spec.backends == ["torch"] and not spec.fp16 and spec.speed == "very-slow"
    assert spec.content == ["live-action"] and spec.handles == ["noise"] and spec.origin == "user"
    assert spec.rank == 0  # unknown quality: only used when chosen by name
    copied = models_dir / spec.id / src.name
    assert copied.read_bytes() == src.read_bytes()
    assert user_registry_path() == user_config / "models.yaml"

    registry = ModelRegistry.load()
    added = registry.get(spec.id)
    assert added.origin == "user" and added.weights is not None and added.weights.sha256 == spec.weights.sha256
    assert ModelRegistry.load(user=False).all() == [m for m in registry.all() if m.origin == "builtin"]

    net = load_network(added, copied)
    with torch.no_grad():
        out = net(torch.rand(1, 3, 16, 24))
    assert tuple(out.shape) == (1, 3, 32, 48)


def test_add_refuses_bad_files_and_names(tmp_path: Path) -> None:
    src = tiny_swinir(tmp_path)
    models_dir = tmp_path / "models"
    for bad_id in ("auto", "lanczos", "Bad Name"):
        with pytest.raises(NeuralFrameError):
            add_user_model(src, models_dir, model_id=bad_id)
    add_user_model(src, models_dir, model_id="mine")
    with pytest.raises(NeuralFrameError, match="already exists"):
        add_user_model(src, models_dir, model_id="mine")

    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a model")
    with pytest.raises(NeuralFrameError, match="cannot load"):
        add_user_model(junk, models_dir)
    script = tmp_path / "scripted.pt"
    script.write_bytes(b"PK")
    with pytest.raises(NeuralFrameError, match="expected one of"):
        add_user_model(script, models_dir)
    with pytest.raises(NeuralFrameError, match="only upscaling models"):
        add_user_model(tiny_swinir(tmp_path, "1x-denoise.safetensors", scale=1), models_dir)
    assert [m.id for m in ModelRegistry.load().all() if m.origin == "user"] == ["mine"]


def test_automatic_choice_uses_an_added_model_only_with_a_rank(tmp_path: Path) -> None:
    from neuralframe.models.advisor import ContentProfile, recommend

    clean = ContentProfile(width=1280, height=720, frames=8, animation_score=0.0, content="live-action", noise=0.5,
                           blockiness=1.0, bits_per_pixel=None, effective_height=720, degradations=[], notes=[])
    models_dir = tmp_path / "models"
    add_user_model(tiny_swinir(tmp_path), models_dir, model_id="unranked", content=["live-action"])
    assert recommend(clean, ModelRegistry.load(), factor=2.0, preset="quality").model == "hat-l-x4"
    add_user_model(tiny_swinir(tmp_path), models_dir, model_id="ranked", content=["live-action"], rank=99)
    assert recommend(clean, ModelRegistry.load(), factor=2.0, preset="quality").model == "ranked"  # a 2x model
    assert recommend(clean, ModelRegistry.load(), factor=2.0, preset="balanced").model != "ranked"  # very slow
    with pytest.raises(NeuralFrameError, match="rank"):
        add_user_model(tiny_swinir(tmp_path), models_dir, model_id="too-good", rank=101)


def test_remove_deletes_the_entry_and_the_file(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    spec = add_user_model(tiny_swinir(tmp_path), models_dir, model_id="gone-soon")
    keep = add_user_model(tiny_swinir(tmp_path, "other.safetensors"), models_dir, model_id="kept")
    remove_user_model(spec.id, models_dir)
    assert not (models_dir / spec.id).exists() and (models_dir / keep.id).is_dir()
    ids = {m.id for m in ModelRegistry.load().all()}
    assert "gone-soon" not in ids and "kept" in ids
    remove_user_model(keep.id, models_dir, delete_file=False)
    assert (models_dir / keep.id).is_dir()
    with pytest.raises(NeuralFrameError, match="built-in"):
        remove_user_model("realesrgan-x4plus", models_dir)
    with pytest.raises(NeuralFrameError, match="no added model"):
        remove_user_model("never-added", models_dir)


def test_a_broken_user_registry_never_breaks_the_built_in_models(user_config: Path) -> None:
    path = user_config / "models.yaml"
    path.write_text(yaml.safe_dump({"models": [
        {"id": "ok", "kind": "sr", "scale": 4, "architecture": "spandrel"},
        {"id": "broken", "kind": "sr", "scale": 99},
        {"id": "realesrgan-x4plus", "kind": "classical", "scale": 1},
    ]}), encoding="utf-8")
    registry = ModelRegistry.load()
    assert registry.get("ok").origin == "user"
    assert registry.get("realesrgan-x4plus").origin == "builtin" and registry.get("realesrgan-x4plus").kind == "sr"
    assert "broken" not in {m.id for m in registry.all()}
    path.write_text("models: [unclosed", encoding="utf-8")
    assert len(ModelRegistry.load().all()) == len(ModelRegistry.load(user=False).all())
