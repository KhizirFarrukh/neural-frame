"""Engines (M12): catalog, adapters, environments, the generative model and the face stage, without any real
engine - a tiny Python script stands in for one. Written 2026-10-13; not run yet (see docs/PLAN.md, M12)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralframe.engines import adapters
from neuralframe.engines.adapters import Command, EngineCall, collect, progress_from
from neuralframe.engines.catalog import ENGINES, engine
from neuralframe.engines.env import EngineEnv, filter_requirements
from neuralframe.errors import NeuralFrameError
from neuralframe.models.library import ModelLibrary, git_blob_sha1
from neuralframe.models.registry import ModelRegistry, ModelSpec, WeightsSpec

# ---- catalog and registry -----------------------------------------------------------------------------------


def test_every_engine_model_has_an_engine_and_an_adapter() -> None:
    registry = ModelRegistry.load(user=False)
    engine_models = [m for m in registry.all() if m.engine]
    assert len(engine_models) >= 25
    for m in engine_models:
        assert m.engine in ENGINES and m.engine in adapters.ADAPTERS, m.id
        assert m.kind in ("generative", "faces"), m.id
        if m.kind == "generative":
            assert m.is_temporal and m.category == "generative", m.id  # driven by the temporal stage
        e = engine(m.engine)
        assert e.licence and (not e.restricted or m.licence_restricted), m.id
        for w in m.all_weights:
            assert w.url.startswith(("https://huggingface.co/", "manual:")), (m.id, w.url)
            assert w.sha256 or w.git_sha1 or w.url.startswith("manual:"), (m.id, w.filename)
    faces = registry.of_kind("faces")
    assert {m.id for m in faces} >= {"gfpgan-1.4", "codeformer", "keep"}
    main_list = [m.id for _, ms in registry.grouped() for m in ms]
    assert "seedvr2-3b-q4" in main_list and "gfpgan-1.4" not in main_list  # faces have their own selector


def test_engines_are_pinned() -> None:
    for e in ENGINES.values():
        if e.repo is not None:
            assert e.commit is not None and len(e.commit) == 40, e.id
            assert e.source_url == f"https://github.com/{e.repo}/archive/{e.commit}.zip"
        assert e.python.startswith("3.") and e.torch and e.torch_index.startswith("cu"), e.id


def test_requirements_are_filtered() -> None:
    text = "torch==2.6.0\n# comment\ngradio==4.0  # demo\n-e .\nDiffusers>=0.30\nopen_clip_torch==2\n\nwandb\n"
    assert filter_requirements(text, ("torch", "gradio", "wandb", "open-clip-torch")) == ["Diffusers>=0.30"]


# ---- adapters -----------------------------------------------------------------------------------------------


def _call(tmp_path: Path, model: ModelSpec, **kw: Any) -> EngineCall:
    env = EngineEnv(engine(model.engine or ""), tmp_path / "cache")
    folder = tmp_path / "models" / (model.files_dir or model.id)
    files = {w.filename: folder / w.filename for w in model.all_weights}
    defaults: dict[str, Any] = {"input": tmp_path / "in", "output": tmp_path / "out", "frames": 33,
                                "in_size": (640, 360), "out_size": (3840, 2160), "vram_gb": 8.0}
    return EngineCall(env=env, model=model, files=files, **{**defaults, **kw})


def test_every_adapter_builds_a_command(tmp_path: Path) -> None:
    registry = ModelRegistry.load(user=False)
    for m in [m for m in registry.all() if m.engine]:
        command = adapters.build(_call(tmp_path, m))
        assert command.args[0].endswith(("python.exe", "python")), m.id
        assert command.results == tmp_path / "out", m.id


def test_seedvr2_fits_small_graphics_cards(tmp_path: Path) -> None:
    model = ModelRegistry.load(user=False).get("seedvr2-3b-q4")
    small = adapters.build(_call(tmp_path, model, vram_gb=8.0)).args
    assert "--blocks_to_swap" in small and small[small.index("--blocks_to_swap") + 1] == "32"
    assert "--vae_decode_tiled" in small and small[small.index("--resolution") + 1] == "2160"
    batch = int(small[small.index("--batch_size") + 1])
    assert batch == 5 and (batch - 1) % 4 == 0
    big = adapters.build(_call(tmp_path, model, vram_gb=48.0)).args
    assert "--blocks_to_swap" not in big and int(big[big.index("--batch_size") + 1]) == 33


def test_results_are_found_and_ordered(tmp_path: Path) -> None:
    out = tmp_path / "out"
    for folder in ("cropped_faces", "restored_imgs"):
        (out / folder).mkdir(parents=True)
        for i in (10, 2, 1):
            (out / folder / f"{i:06d}.png").write_bytes(b"x")
    found = collect(Command(["x"], tmp_path, {}, out, prefer=("restored_imgs",)), 3)
    assert [p.parent.name for p in found] == ["restored_imgs"] * 3
    assert [p.stem for p in found] == ["000001", "000002", "000010"]
    assert progress_from("Processing 12/40 frames | 3/40") == (3, 40)
    assert progress_from("no numbers here") is None


# ---- environments -------------------------------------------------------------------------------------------


def test_environment_state_follows_the_recipe(tmp_path: Path) -> None:
    env = EngineEnv(engine("gfpgan"), tmp_path)
    assert env.state() == "missing"
    env.python.parent.mkdir(parents=True)
    env.python.write_bytes(b"")
    env.state_path.write_text(json.dumps({"recipe": env.recipe()}), encoding="utf-8")
    assert env.state() == "installed"
    changed = EngineEnv(engine("gfpgan"), tmp_path, cuda_index="cu128")
    assert changed.state() == "outdated"
    environ = env.environ()
    assert environ["HF_HOME"].startswith(str(env.root)) and environ["PYTHONNOUSERSITE"] == "1"
    assert env.remove() >= 0 and env.state() == "missing"


# ---- a fake engine through the real runner ------------------------------------------------------------------

FAKE = """
import sys
from pathlib import Path
from PIL import Image
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
names = sorted(p.name for p in src.glob("*.png"))
for i, name in enumerate(names):
    img = Image.open(src / name)
    img.resize((img.width * 2, img.height * 2)).save(dst / name)
    print(f"frame {i + 1}/{len(names)}", flush=True)
"""


@pytest.fixture
def fake_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, ModelSpec]:
    from neuralframe.engines.model import EngineRunner

    script = tmp_path / "fake_engine.py"
    script.write_text(FAKE, encoding="utf-8")
    model = ModelSpec(id="fake-gen", kind="generative", scale=2, input="recurrent", window=1, engine="gfpgan")

    def build(call: EngineCall) -> Command:
        return Command([sys.executable, str(script), str(call.input), str(call.output)], tmp_path,
                       dict(os.environ), call.output)

    monkeypatch.setitem(adapters.ADAPTERS, "gfpgan", build)
    env = EngineEnv(engine("gfpgan"), tmp_path / "cache")
    messages: list[str] = []
    from neuralframe.io.ffmpeg import locate_ffmpeg

    runner = EngineRunner(model, env, {}, tools=locate_ffmpeg(), work_dir=tmp_path / "work",
                          on_message=messages.append)
    runner.messages = messages  # type: ignore[attr-defined]
    return runner, model


def test_generative_model_runs_chunks_through_an_engine(fake_engine: tuple[Any, ModelSpec]) -> None:
    torch = pytest.importorskip("torch")
    from neuralframe.engines.model import GenerativeModel

    runner, _ = fake_engine
    model = GenerativeModel(runner, device=torch.device("cpu"), out_size=(64, 48))
    frames = torch.rand(3, 3, 24, 32)
    outputs = list(model.run(frames))
    assert [t for t, _ in outputs] == [0, 1, 2]
    assert all(tuple(o.shape) == (3, 48, 64) for _, o in outputs)
    assert abs(float(outputs[1][1][:, ::2, ::2].mean()) - float(frames[1].mean())) < 0.05
    assert model.frames_processed == 3 and any("3/3" in m for m in runner.messages)
    assert not any((runner.work_dir).glob("*/in"))  # temporary files removed


def test_face_stage_keeps_order_and_timestamps(fake_engine: tuple[Any, ModelSpec]) -> None:
    pytest.importorskip("torch")
    from neuralframe.colorspace import ChromaLocation, ColorPlan, ColorRange, Matrix, Primaries, Transfer
    from neuralframe.core.runtime import CancelToken, Frame, StageQueue
    from neuralframe.engines.faces import FaceStage
    from neuralframe.engines.model import FaceRestorer
    from neuralframe.io.rawvideo import RawVideoFormat

    runner, _ = fake_engine
    fmt = RawVideoFormat.yuv420(32, 16, 8)
    cancel = CancelToken()
    q_in: StageQueue[Any] = StageQueue("in", 16, cancel)
    q_out: StageQueue[Any] = StageQueue("out", 16, cancel)
    bt709 = (Matrix.BT709, ColorRange.LIMITED, Primaries.BT709, Transfer.BT709, ChromaLocation.LEFT)
    stage = FaceStage(cancel, q_in, q_out, restorer=FaceRestorer(runner), fmt=fmt,
                      color=ColorPlan(*bt709, *bt709), batch=4)
    for i in range(6):
        q_in.put(Frame(i, i * 100, 100, np.full(fmt.frame_bytes, 128, np.uint8)))
    q_in.close()
    stage.run()
    got = []
    while (frame := q_out.get()) is not None:
        got.append(frame)
    assert [f.index for f in got] == list(range(6)) and [f.pts for f in got] == [i * 100 for i in range(6)]
    assert all(f.data.size == fmt.frame_bytes for f in got)


# ---- multi-file models in the library -----------------------------------------------------------------------


def test_multi_file_models_share_files_and_verify_git_ids(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "dit.bin").write_bytes(b"d" * 3000)
    (remote / "vae.bin").write_bytes(b"v" * 2000)
    (remote / "config.json").write_text('{"a": 1}', encoding="utf-8")

    def w(name: str, **kw: Any) -> WeightsSpec:
        path = remote / name
        return WeightsSpec(url=path.as_uri(), filename=name, size_bytes=path.stat().st_size, **kw)

    dit = w("dit.bin", sha256=hashlib.sha256((remote / "dit.bin").read_bytes()).hexdigest())
    vae = w("vae.bin", sha256=hashlib.sha256((remote / "vae.bin").read_bytes()).hexdigest(), shared=True)
    cfg = w("sub/config.json", git_sha1=git_blob_sha1(remote / "config.json")).model_copy(
        update={"url": (remote / "config.json").as_uri()})
    a = ModelSpec(id="a", kind="generative", scale=4, engine="seedvr2", files_dir="shared", weights=dit,
                  extra_weights=[vae, cfg])
    lib = ModelLibrary(tmp_path / "models")
    assert lib.status(a).state == "missing" and lib.status(a).size_bytes == 3000 + 2000 + 8
    lib.ensure(a)
    assert lib.status(a).state == "installed"
    assert (tmp_path / "models" / "shared" / "sub" / "config.json").is_file()
    lib.delete(a)
    assert (tmp_path / "models" / "shared" / "vae.bin").is_file()  # shared: kept for the other models
    assert not (tmp_path / "models" / "shared" / "dit.bin").exists()


def test_files_that_cannot_be_downloaded_say_how_to_get_them(tmp_path: Path) -> None:
    spec = ModelSpec(id="supir-test", kind="generative", scale=4, engine="supir", files_dir="supir",
                     weights=WeightsSpec(url="manual:https://example.org/page", filename="SUPIR-v0Q.ckpt"))
    with pytest.raises(NeuralFrameError, match="cannot be downloaded automatically") as err:
        ModelLibrary(tmp_path).ensure(spec)
    assert "Use a file I have" in (err.value.hint or "")


def test_face_models_are_not_upscalers(settings: Any) -> None:
    from neuralframe.core.job import JobRunner, JobSpec

    spec = JobSpec(input=Path("in.mp4"), output=Path("out.mkv"), model="gfpgan-1.4")
    with pytest.raises(NeuralFrameError, match="restores faces"):
        JobRunner(settings).resolve_model(spec)
