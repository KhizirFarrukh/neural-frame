"""The job service end to end through FastAPI's test client: real jobs on CPU workers."""

from __future__ import annotations

import io
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from neuralframe.api.app import create_app
from neuralframe.api.manager import JobManager
from neuralframe.api.store import JobStore
from neuralframe.config import Settings
from neuralframe.core.job import JobRunner
from tests.conftest import ClipFactory


@pytest.fixture
def service_settings(settings: Settings, tmp_path: Path, clip: ClipFactory) -> Settings:
    media_dir = clip("api-probe-root", width=96, height=64, duration=0.5).parent
    settings.inference.device = "cpu"
    settings.paths.db_path = tmp_path / "jobs.db"
    settings.api.input_roots = [tmp_path / "in", media_dir]
    settings.api.output_roots = [tmp_path / "out"]
    settings.api.upload_dir = tmp_path / "uploads"
    settings.api.output_dir = tmp_path / "outputs"
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    return settings


def make_client(settings: Settings, *, start: bool = True, **manager_kwargs: Any) -> TestClient:
    manager = JobManager(settings, JobStore(settings.paths.database()), workers=["cpu"], poll_s=0.1,
                         **manager_kwargs)
    return TestClient(create_app(settings, manager=manager, start_workers=start))


@pytest.fixture
def client(service_settings: Settings) -> Iterator[TestClient]:
    with make_client(service_settings) as c:
        yield c


def wait(client: TestClient, job_id: str, states: tuple[str, ...] = ("done", "failed", "cancelled"),
         timeout: float = 180) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["state"] in states:
            return dict(job)
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not reach {states}: {job}")


def test_system_models_and_ui(client: TestClient) -> None:
    system = client.get("/v1/system").json()
    assert system["service"]["workers"] == ["cpu"]
    assert system["defaults"]["target"] == "4k" and "balanced" in system["defaults"]["presets"]
    assert "inference_backend" in system["selected"]
    models = {m["id"]: m for m in client.get("/v1/models").json()}
    assert models["realesrgan-x4plus"]["licence"] and models["lanczos"]["weights_present"]
    page = client.get("/")
    assert page.status_code == 200 and "NeuralFrame" in page.text and "EventSource" in page.text


def test_job_lifecycle_with_events_preview_and_download(
    client: TestClient, clip: ClipFactory, service_settings: Settings,
) -> None:
    src = clip("api-job", width=96, height=64, fps="24", duration=1.0)
    out = service_settings.api.output_roots[0] / "job.mkv"
    created = client.post("/v1/jobs", json={
        "input": str(src), "output": str(out), "target": "x2",
        "options": {"model": "lanczos", "encoder": "libx264", "codec": "h264"},
    })
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]

    events: list[tuple[str, dict[str, Any]]] = []
    with client.stream("GET", f"/v1/jobs/{job_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        name = ""
        for line in response.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line[5:])))
                if name == "end":
                    break
    states = [data.get("state") for kind, data in events if kind == "progress"]
    assert states[-1] == "done" and events[-1] == ("end", {"state": "done"})
    assert any(s == "running" for s in states)

    job = client.get(f"/v1/jobs/{job_id}").json()
    assert job["state"] == "done" and job["percent"] == 100 and job["frames_done"] == 24
    assert Path(job["output"]) == out.resolve() and out.is_file()
    report = client.get(f"/v1/jobs/{job_id}/report").json()
    assert report["sync"]["ok"] and report["output"]["frames"] == 24

    for side, size in (("before", (192, 128)), ("after", (192, 128)), ("both", (384, 128))):
        png = client.get(f"/v1/jobs/{job_id}/preview", params={"frame": 12, "side": side})
        assert png.status_code == 200 and png.headers["content-type"] == "image/png"
        assert Image.open(io.BytesIO(png.content)).size == size
    download = client.get(f"/v1/jobs/{job_id}/output")
    assert download.status_code == 200 and download.content == out.read_bytes()

    listed = client.get("/v1/jobs").json()
    assert [j["id"] for j in listed] == [job_id]
    assert client.delete(f"/v1/jobs/{job_id}").status_code == 204
    assert client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_paths_outside_roots_are_refused(client: TestClient, tmp_path: Path, clip: ClipFactory) -> None:
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(b"not a video")
    r = client.post("/v1/jobs", json={"input": str(outside)})
    assert r.status_code == 403 and r.json()["error"]["code"] == "PATH_NOT_ALLOWED"
    src = clip("api-job", width=96, height=64, fps="24", duration=1.0)
    r = client.post("/v1/jobs", json={"input": str(src), "output": str(tmp_path / "x.mkv")})
    assert r.status_code == 403
    r = client.post("/v1/jobs", json={"input": str(src), "output": str(tmp_path / "out" / ".." / "x.mkv")})
    assert r.status_code == 403
    r = client.post("/v1/probe", json={"path": str(outside)})
    assert r.status_code == 403


def test_invalid_options_are_rejected(client: TestClient, clip: ClipFactory) -> None:
    src = clip("api-job", width=96, height=64, fps="24", duration=1.0)
    r = client.post("/v1/jobs", json={"input": str(src), "options": {"quality": 99}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_OPTIONS"
    r = client.post("/v1/jobs", json={"input": str(src), "options": {"no_such_option": 1}})
    assert r.status_code == 422
    r = client.post("/v1/jobs", json={"input": str(src), "options": {"output": "x.mkv"}})
    assert r.status_code == 400


def test_upload_then_frame_generation_job(client: TestClient, clip: ClipFactory, service_settings: Settings) -> None:
    src = clip("api-up", width=96, height=64, fps="25", duration=1.0)
    with src.open("rb") as fh:
        up = client.post("/v1/uploads", files={"file": ("../../my clip.mp4", fh, "video/mp4")})
    assert up.status_code == 201, up.text
    path = Path(up.json()["path"])
    assert path.name == "my clip.mp4" and path.is_relative_to(service_settings.api.upload_dir)  # type: ignore[arg-type]
    info = client.post("/v1/probe", json={"path": str(path)}).json()
    assert info["is_image"] is False and info["video"]["size"] == "96x64"
    created = client.post("/v1/jobs", json={"input": str(path), "options": {
        "upscale": False, "interpolate": True, "target_fps": "50", "framegen_engine": "minterpolate",
        "encoder": "libx264", "codec": "h264",
    }}).json()
    # Uploads are outside the output roots: the result goes to the outputs folder with a descriptive name.
    assert Path(created["output"]).parent == service_settings.api.output_dir.resolve()  # type: ignore[union-attr]
    assert Path(created["output"]).name == "my clip_50fps.mkv"
    job = wait(client, created["id"])
    assert job["state"] == "done", job
    assert client.get(f"/v1/jobs/{created['id']}/report").json()["framegen"]["engine"] == "minterpolate"


def test_queued_job_cancel_resume_delete(service_settings: Settings, clip: ClipFactory) -> None:
    src = clip("api-job", width=96, height=64, fps="24", duration=1.0)
    with make_client(service_settings, start=False) as client:  # no workers: jobs stay queued
        job = client.post("/v1/jobs", json={"input": str(src), "target": "x2", "options": {"model": "lanczos"}}).json()
        assert job["state"] == "queued"
        assert client.delete(f"/v1/jobs/{job['id']}").status_code == 400  # queued jobs must be cancelled first
        assert client.post(f"/v1/jobs/{job['id']}/cancel").json()["state"] == "cancelled"
        assert client.post(f"/v1/jobs/{job['id']}/cancel").status_code == 400
        resumed = client.post(f"/v1/jobs/{job['id']}/resume").json()
        assert resumed["state"] == "queued" and resumed["spec"]["resume"] is True
        assert client.post("/v1/jobs/nope/cancel").status_code == 404


def test_running_job_cancel_then_resume_reuses_chunks(service_settings: Settings, clip: ClipFactory) -> None:
    src = clip("api-long", width=96, height=64, fps="24", duration=3.0)
    holder: dict[str, Any] = {}

    def factory(settings: Settings, **kwargs: Any) -> JobRunner:
        runner = JobRunner(settings, **kwargs)
        if not holder.get("cancelled"):
            def stop(chunk_id: int, _total: int) -> None:
                if chunk_id == 0:
                    holder["cancelled"] = True
                    holder["client"].post(f"/v1/jobs/{runner.job_id}/cancel")
            runner.on_chunk_done = stop
        return runner

    with make_client(service_settings, runner_factory=factory) as client:
        holder["client"] = client
        job = client.post("/v1/jobs", json={
            "input": str(src), "target": "x2",
            "options": {"model": "lanczos", "chunk_frames": 24, "encoder": "libx264", "codec": "h264"},
        }).json()
        assert wait(client, job["id"])["state"] == "cancelled"
        client.post(f"/v1/jobs/{job['id']}/resume")
        finished = wait(client, job["id"])
        assert finished["state"] == "done", finished
        assert finished["attempts"] == 2
        report = client.get(f"/v1/jobs/{job['id']}/report").json()
        assert report["throughput"]["frames_resumed"] >= 24 and report["output"]["frames"] == 72


def test_restart_resumes_interrupted_job(service_settings: Settings, clip: ClipFactory) -> None:
    src = clip("api-job", width=96, height=64, fps="24", duration=1.0)
    store = JobStore(service_settings.paths.database())
    out = service_settings.api.output_roots[0] / "restart.mkv"
    from neuralframe.core.job import JobSpec
    from neuralframe.core.planning import TargetSpec

    job = store.create(JobSpec(input=src, output=out, model="lanczos", target=TargetSpec.factor(2)))
    store.claim_next("cpu")  # a server picked it up ...
    store.update(job["id"], state="running")  # ... and died mid-job
    store.close()
    with make_client(service_settings) as client:
        finished = wait(client, job["id"])
        assert finished["state"] == "done" and finished["spec"]["resume"] is True
        assert out.is_file()


def test_queue_cli_against_the_service(
    service_settings: Settings, clip: ClipFactory, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typer.testing import CliRunner

    from neuralframe.cli import app as cli

    src = clip("api-job", width=96, height=64, fps="24", duration=1.0)
    cfg = tmp_path / "cli.yaml"
    cfg.write_text("inference:\n  device: cpu\n", encoding="utf-8")
    with make_client(service_settings, start=False) as client:
        monkeypatch.setattr(cli, "_client", lambda _server: client)
        runner = CliRunner()
        out = service_settings.api.output_roots[0] / "cli.mkv"
        added = runner.invoke(cli.app, ["-c", str(cfg), "queue", "add", str(src), "-o", str(out), "-t", "x2",
                                        "-m", "lanczos", "--priority", "3"])
        assert added.exit_code == 0, added.output
        job_id = added.stdout.strip().splitlines()[-1]
        listed = runner.invoke(cli.app, ["-c", str(cfg), "queue", "list", "--json"])
        assert listed.exit_code == 0 and json.loads(listed.stdout)[0]["priority"] == 3
        cancelled = runner.invoke(cli.app, ["-c", str(cfg), "queue", "cancel", job_id])
        assert cancelled.exit_code == 0 and "cancelled" in cancelled.output
        resumed = runner.invoke(cli.app, ["-c", str(cfg), "queue", "resume", job_id])
        assert resumed.exit_code == 0 and "queued" in resumed.output
        missing = runner.invoke(cli.app, ["-c", str(cfg), "queue", "cancel", "nope"])
        assert missing.exit_code == 1 and "NOT_FOUND" in missing.output


def test_queue_cli_without_a_server(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from neuralframe.cli.app import app

    result = CliRunner().invoke(app, ["queue", "list", "--server", "http://127.0.0.1:9"])
    assert result.exit_code == 2 and "neuralframe serve" in result.output
