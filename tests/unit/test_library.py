"""Model library: downloads on demand (progress, cancel, resume, checksum), local files, search folders,
updates (mocked hosts), and the plain-language model list. No network: "downloads" read file:// URLs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.models.library import DownloadProgress, ModelLibrary
from neuralframe.models.registry import CATEGORY_TITLES, ModelRegistry, ModelSpec, WeightsSpec

MB = 1 << 20


def remote_file(folder: Path, name: str = "model.pth", size: int = 5 * MB, seed: int = 1) -> tuple[Path, str]:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    data = (hashlib.sha256(str(seed).encode()).digest() * (size // 32 + 1))[:size]
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def spec_for(path: Path, sha: str, model_id: str = "remote-model", url: str | None = None, **kw: Any) -> ModelSpec:
    return ModelSpec(id=model_id, kind="sr", scale=2, architecture="srvgg", purpose="Test model",
                     weights=WeightsSpec(url=url or path.as_uri(), sha256=sha, filename=path.name,
                                         size_bytes=path.stat().st_size), **kw)


class CancelAfter:
    """Cancels after ``n`` blocks (the download checks once per MiB)."""

    def __init__(self, n: int) -> None:
        self.n = n

    def is_set(self) -> bool:
        self.n -= 1
        return self.n < 0


# ---- downloads -------------------------------------------------------------------------------------------


def test_download_reports_progress_and_verifies(tmp_path: Path) -> None:
    src, sha = remote_file(tmp_path / "remote")
    spec = spec_for(src, sha)
    lib = ModelLibrary(tmp_path / "models")
    assert lib.status(spec).state == "missing" and lib.status(spec).text() == "Not downloaded (5 MB)"
    events: list[DownloadProgress] = []
    path = lib.ensure(spec, progress=events.append)
    assert path == tmp_path / "models" / "remote-model" / "model.pth" and path.read_bytes() == src.read_bytes()
    assert events and events[-1].done == 5 * MB and events[-1].total == 5 * MB and events[-1].fraction == 1.0
    assert "5.2 of 5.2 MB" in events[-1].text()
    status = lib.status(spec)
    assert status.state == "installed" and status.ready and lib.disk_usage() >= 5 * MB
    assert lib.ensure(spec) == path  # no second download
    assert lib.delete(spec) == 5 * MB and lib.status(spec).state == "missing"


def test_cancelled_download_keeps_the_part_and_resumes(tmp_path: Path) -> None:
    src, sha = remote_file(tmp_path / "remote")
    spec = spec_for(src, sha)
    lib = ModelLibrary(tmp_path / "models")
    with pytest.raises(JobCancelled):
        lib.download(spec, cancel=CancelAfter(2))
    status = lib.status(spec)
    assert status.state == "missing" and status.partial_bytes == 2 * MB
    assert "paused at 2 of 5 MB" in status.text()
    events: list[DownloadProgress] = []
    path = lib.ensure(spec, progress=events.append)
    assert events[0].done == 2 * MB  # resumed, not restarted
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_wrong_file_is_rejected_and_removed(tmp_path: Path) -> None:
    src, _ = remote_file(tmp_path / "remote")
    spec = spec_for(src, "0" * 64)
    lib = ModelLibrary(tmp_path / "models")
    with pytest.raises(NeuralFrameError) as err:
        lib.ensure(spec)
    assert err.value.code == ErrorCode.DOWNLOAD_FAILED and "not the expected one" in err.value.message
    assert not any((tmp_path / "models").rglob("*.part"))


def test_unreachable_host_explains_what_to_do(tmp_path: Path) -> None:
    def offline(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no network", request=request)

    src, sha = remote_file(tmp_path / "remote", size=MB)
    spec = spec_for(src, sha, url="https://example.invalid/model.pth")
    lib = ModelLibrary(tmp_path / "models", transport=httpx.MockTransport(offline))
    with pytest.raises(NeuralFrameError) as err:
        lib.ensure(spec)
    assert "internet connection" in (err.value.hint or "") and "models link" in (err.value.hint or "")


def test_http_download_resumes_with_a_range_request(tmp_path: Path) -> None:
    src, sha = remote_file(tmp_path / "remote", size=3 * MB)
    data = src.read_bytes()
    ranges: list[str | None] = []

    def server(request: httpx.Request) -> httpx.Response:
        ranges.append(request.headers.get("range"))
        start = int(request.headers["range"].split("=")[1].rstrip("-")) if "range" in request.headers else 0
        return httpx.Response(206 if start else 200, content=data[start:],
                              headers={"content-length": str(len(data) - start)})

    spec = spec_for(src, sha, url="https://example.org/model.pth")
    lib = ModelLibrary(tmp_path / "models", transport=httpx.MockTransport(server))
    with pytest.raises(JobCancelled):
        lib.download(spec, cancel=CancelAfter(1))
    lib.ensure(spec)
    assert ranges == [None, f"bytes={MB}-"]
    assert hashlib.sha256((tmp_path / "models" / spec.id / "model.pth").read_bytes()).hexdigest() == sha


# ---- files the user already has --------------------------------------------------------------------------


def test_search_folders_are_used_without_downloading(tmp_path: Path) -> None:
    src, sha = remote_file(tmp_path / "remote", size=MB)
    shared = tmp_path / "my models"
    remote_file(shared, name="model.pth", size=MB)
    spec = spec_for(src, sha, url="https://example.invalid/model.pth")
    lib = ModelLibrary(tmp_path / "models", search_paths=[shared])
    assert lib.status(spec).state == "found"
    assert lib.ensure(spec) == shared / "model.pth"


def test_link_the_same_file_or_another_version(tmp_path: Path) -> None:
    from tests.models import tiny_sr_model

    spec, path = tiny_sr_model(tmp_path / "published")
    lib = ModelLibrary(tmp_path / "models")
    copy = tmp_path / "elsewhere" / "my copy.pth"
    copy.parent.mkdir()
    copy.write_bytes(path.read_bytes())
    assert lib.link(spec, copy).state == "linked"
    assert lib.ensure(spec) == copy.resolve() and lib.effective_spec(spec) == spec
    lib.unlink(spec)
    assert lib.status(spec).state == "missing"

    other_spec, other = tiny_sr_model(tmp_path / "retrained", model_id="tiny-other")  # same network, new weights
    status = lib.link(spec, other)
    assert status.state == "linked-different" and "different version" in status.text()
    effective = lib.effective_spec(spec)
    assert effective.weights is not None and other_spec.weights is not None
    assert effective.weights.sha256 == other_spec.weights.sha256  # engine caches follow the real file

    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a model")
    with pytest.raises(NeuralFrameError, match="not a version"):
        lib.link(spec, junk)
    script = tmp_path / "scripted.pt"
    script.write_bytes(b"PK")
    with pytest.raises(NeuralFrameError, match="TorchScript"):
        lib.link(spec, script)


# ---- updates (mocked hosts) ------------------------------------------------------------------------------


def test_hugging_face_update_is_found_and_installed(tmp_path: Path) -> None:
    old, old_sha = remote_file(tmp_path / "v1", name="model.safetensors", size=MB, seed=1)
    new, new_sha = remote_file(tmp_path / "v2", name="model.safetensors", size=2 * MB, seed=2)
    calls: list[str] = []

    def hub(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/api/models/org/repo":
            return httpx.Response(200, json={"siblings": [
                {"rfilename": "model.safetensors", "lfs": {"sha256": new_sha, "size": 2 * MB}}]})
        if request.url.path.endswith("/resolve/main/model.safetensors"):
            return httpx.Response(200, content=new.read_bytes())
        return httpx.Response(404)

    url = "https://huggingface.co/org/repo/resolve/main/model.safetensors"
    spec = spec_for(old, old_sha, model_id="hub-model", url=url)
    lib = ModelLibrary(tmp_path / "models", transport=httpx.MockTransport(hub))
    target = tmp_path / "models" / "hub-model" / "model.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(old.read_bytes())
    assert lib.update_source(spec) == "huggingface"
    info = lib.check_update(spec)
    assert info is not None and info.available and info.sha256 == new_sha
    assert lib.status(spec).update is not None and "update available" in lib.status(spec).text()
    assert lib.check_update(spec) is not None and calls.count("/api/models/org/repo") == 1  # cached
    lib.apply_update(spec)
    assert lib.weights(spec).sha256 == new_sha and lib.status(spec).state == "installed"
    assert not lib.status(spec).update.available  # type: ignore[union-attr]
    assert lib.effective_spec(spec).weights.sha256 == new_sha  # type: ignore[union-attr]


def test_github_release_asset_digest_is_compared(tmp_path: Path) -> None:
    src, sha = remote_file(tmp_path / "v1", name="4xModel.pth", size=MB)

    def api(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/someone/models/releases/tags/4xModel"
        return httpx.Response(200, json={"assets": [{"name": "4xModel.pth", "size": MB, "digest": f"sha256:{sha}",
                                                     "browser_download_url": "https://x/4xModel.pth"}]})

    url = "https://github.com/someone/models/releases/download/4xModel/4xModel.pth"
    spec = spec_for(src, sha, model_id="gh-model", url=url)
    lib = ModelLibrary(tmp_path / "models", transport=httpx.MockTransport(api))
    info = lib.check_update(spec)
    assert info is not None and info.source == "github" and not info.available
    assert ModelLibrary.update_source(spec_for(src, sha, url="https://example.org/m.pth")) is None


# ---- jobs ------------------------------------------------------------------------------------------------


def test_a_job_downloads_its_model_first_and_reports_it(tmp_path: Path, settings: Any) -> None:
    from neuralframe.core.job import JobRunner, ProgressEvent

    src, sha = remote_file(tmp_path / "remote", size=3 * MB)
    spec = spec_for(src, sha)
    settings.paths.models_dir = tmp_path / "models"
    events: list[ProgressEvent] = []
    runner = JobRunner(settings, registry=ModelRegistry([spec]), progress=events.append)
    path = runner.fetch_model(spec)
    assert path.is_file()
    downloading = [e for e in events if e.state == "downloading"]
    assert downloading and downloading[-1].download is not None
    assert downloading[-1].download["label"] == "Test model (remote-model)"
    assert downloading[-1].as_dict()["download"]["done"] == 3 * MB
    assert downloading[-1].percent == 100.0

    other = spec_for(src, sha, model_id="cancel-me")
    runner.cancel.cancel("test")
    with pytest.raises(JobCancelled):
        runner.fetch_model(other)


# ---- the model list --------------------------------------------------------------------------------------


def test_every_model_says_what_it_is_for() -> None:
    registry = ModelRegistry.load(user=False)
    upscalers = [m for m in registry.all() if m.kind in ("sr", "classical")]
    assert len(upscalers) >= 60
    for m in upscalers:
        assert m.purpose and m.category in CATEGORY_TITLES, m.id
        assert m.label.endswith(f"({m.id})") and m.label.startswith(m.purpose)
        if m.weights is not None:
            assert m.weights.url.startswith("https://") and len(m.weights.sha256) == 64, m.id
            assert m.weights.size_bytes, m.id
        assert ("-NC" in m.licence) == m.licence_restricted, m.id
    titles = [t for t, _ in registry.grouped()]
    assert titles == [t for t in CATEGORY_TITLES.values() if t in titles]  # display order
    assert {t for t in titles} == set(CATEGORY_TITLES.values())
    assert len({m.label for m in upscalers}) == len(upscalers)


def test_models_can_be_named_the_way_people_write_them() -> None:
    registry = ModelRegistry.load(user=False)
    assert registry.get("4x_UltraSharpV2").id == "4x-ultrasharpv2"
    assert registry.get("HAT-L-X4").id == "hat-l-x4"
    with pytest.raises(NeuralFrameError, match="unknown model"):
        registry.get("no-such-model")


def test_nothing_is_bundled_or_downloaded_by_default() -> None:
    registry = ModelRegistry.load(user=False)
    community = [m for m in registry.all() if m.rank == 0 and m.weights is not None and m.kind == "sr"]
    assert len(community) >= 50 and not any(m.bundled for m in community)


# ---- safety nets for community models --------------------------------------------------------------------


def test_size_guard_pads_and_crops() -> None:
    torch = pytest.importorskip("torch")
    from neuralframe.inference.archs import SizeGuard

    class NeedsMultipleOf4(torch.nn.Module):
        def forward(self, x: Any) -> Any:
            assert x.shape[-1] % 4 == 0 and x.shape[-2] % 4 == 0 and x.shape[-2] >= 16
            return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    guard = SizeGuard(NeedsMultipleOf4(), scale=2, multiple=4, minimum=16, square=False)
    out = guard(torch.rand(1, 3, 10, 23))
    assert tuple(out.shape) == (1, 3, 20, 46)


def test_precision_follows_what_the_network_supports() -> None:
    torch = pytest.importorskip("torch")
    from neuralframe.inference.factory import model_precision

    fp32_only = ModelSpec(id="m", kind="sr", scale=4, fp16=False, bf16=False)
    cuda = torch.device("cuda")
    assert model_precision(fp32_only, "fp16", cuda) == "fp32"
    assert model_precision(fp32_only, "bf16", cuda) == "fp32"
    assert model_precision(fp32_only, "fp16", torch.device("cpu")) == "fp32"


def test_patch_trained_models_are_tiled_at_their_patch_size() -> None:
    torch = pytest.importorskip("torch")
    from neuralframe.inference.factory import plan_inference

    aura = ModelRegistry.load(user=False).get("aurasr-v2")
    plan = plan_inference(aura, (360, 640), device=torch.device("cpu"), backend="torch", precision="fp32",
                          budget_bytes=8 << 30, overlap=32)
    assert plan.tile == 64 and plan.overlap == 16


def test_library_state_survives_a_broken_file(tmp_path: Path) -> None:
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / ModelLibrary.STATE_FILE).write_text("{broken", encoding="utf-8")
    src, sha = remote_file(tmp_path / "remote", size=MB)
    lib = ModelLibrary(tmp_path / "models")
    assert lib.status(spec_for(src, sha)).state == "missing"
    lib.ensure(spec_for(src, sha))
    assert json.loads((tmp_path / "models" / ModelLibrary.STATE_FILE).read_text(encoding="utf-8"))["verified"]
    assert (tmp_path / "models" / "remote-model" / "model.pth").stat().st_size == MB
