from __future__ import annotations

from pathlib import Path

import pytest

from neuralframe.api.paths import PathNotAllowed, check_input, check_output, safe_filename, within
from neuralframe.api.store import JobStore
from neuralframe.core.job import JobSpec


def spec(tmp_path: Path, name: str = "a") -> JobSpec:
    return JobSpec(input=tmp_path / f"{name}.mp4", output=tmp_path / f"{name}.mkv")


def test_priority_then_fifo(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    a = store.create(spec(tmp_path, "a"))
    b = store.create(spec(tmp_path, "b"), priority=5)
    c = store.create(spec(tmp_path, "c"))
    order = []
    while (claimed := store.claim_next("w")) is not None:
        order.append(claimed[0])
    assert order == [b["id"], a["id"], c["id"]]
    assert store.get(a["id"])["state"] == "probing"  # type: ignore[index]
    assert store.get(a["id"])["attempts"] == 1  # type: ignore[index]


def test_claim_respects_worker_filter(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    gpu1 = store.create(spec(tmp_path).model_copy(update={"gpus": [1]}))
    anywhere = store.create(spec(tmp_path, "b"))
    claimed = store.claim_next("cuda:0", accepts=lambda s: not s.gpus or s.gpus[0] == 0)
    assert claimed is not None and claimed[0] == anywhere["id"]
    assert store.claim_next("cuda:0", accepts=lambda s: not s.gpus or s.gpus[0] == 0) is None
    assert store.get(gpu1["id"])["state"] == "queued"  # type: ignore[index]


def test_interrupted_jobs_resume_after_restart(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    store = JobStore(db)
    running = store.create(spec(tmp_path, "a"))
    paused = store.create(spec(tmp_path, "b"))
    done = store.create(spec(tmp_path, "c"))
    store.claim_next("w")
    store.update(paused["id"], state="paused")
    store.update(done["id"], state="done")
    store.close()

    reopened = JobStore(db)
    assert sorted(reopened.recover_interrupted()) == sorted([running["id"], paused["id"]])
    for job_id in (running["id"], paused["id"]):
        record = reopened.get(job_id)
        assert record is not None and record["state"] == "queued" and record["spec"]["resume"] is True
    assert reopened.get(done["id"])["state"] == "done"  # type: ignore[index]


def test_cancel_only_affects_queued_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    queued = store.create(spec(tmp_path, "a"))
    assert store.cancel_queued(queued["id"])
    assert not store.cancel_queued(queued["id"])
    assert store.get(queued["id"])["state"] == "cancelled"  # type: ignore[index]


def test_progress_round_trip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    job = store.create(spec(tmp_path))
    store.update(job["id"], state="running", percent=42.5, progress={"stages": [{"name": "decode"}], "gpu": None})
    record = store.get(job["id"])
    assert record is not None and record["percent"] == 42.5 and record["stages"] == [{"name": "decode"}]
    assert store.delete(job["id"]) and store.get(job["id"]) is None


# ---- path sandbox --------------------------------------------------------------------------------------


def test_paths_inside_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    f = root / "sub" / "clip.mp4"
    f.write_bytes(b"x")
    assert check_input(f, [root]) == f.resolve()
    assert check_output(root / "new" / "out.mkv", [root]).parent.is_dir()
    assert within(root.resolve(), [root])


@pytest.mark.parametrize("relative", ["../outside.mp4", "sub/../../outside.mp4"])
def test_traversal_is_rejected(tmp_path: Path, relative: str) -> None:
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "outside.mp4").write_bytes(b"x")
    with pytest.raises(PathNotAllowed):
        check_input(root / relative, [root])
    with pytest.raises(PathNotAllowed):
        check_output(root / relative, [root])


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    secret = tmp_path / "secret.mp4"
    secret.write_bytes(b"x")
    link = root / "link.mp4"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("creating symlinks needs extra privileges on this system")
    with pytest.raises(PathNotAllowed):
        check_input(link, [root])


def test_prefix_sibling_is_not_inside(tmp_path: Path) -> None:
    (tmp_path / "videos-private").mkdir()
    f = tmp_path / "videos-private" / "a.mp4"
    f.write_bytes(b"x")
    with pytest.raises(PathNotAllowed):
        check_input(f, [tmp_path / "videos"])


@pytest.mark.parametrize(("name", "expected"), [
    ("clip.mp4", "clip.mp4"), ("../../evil.mp4", "evil.mp4"), ("C:\\Windows\\x.png", "x.png"),
    ("a<b>c|d.mkv", "a_b_c_d.mkv"), ("..", "upload"), ("", "upload"),
])
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected


def test_colour_match_filter_follows_the_job_colour_plan() -> None:
    from neuralframe.core.preview import colour_match_filter

    sd = {"plan": {"color": {
        "input": {"matrix": "bt601", "range": "tv", "primaries": "smpte170m", "transfer": "bt709"},
        "output": {"colorspace": "bt709", "color_primaries": "bt709", "color_trc": "bt709", "color_range": "tv"},
        "hdr_passthrough": False,
    }}}
    f = colour_match_filter(sd)
    assert f is not None and "ispace=smpte170m" in f and "primaries=bt709" in f and "fast=0" in f
    same = {"plan": {"color": {
        "input": {"matrix": "bt709", "range": "tv", "primaries": "bt709", "transfer": "bt709"},
        "output": {"colorspace": "bt709", "color_primaries": "bt709", "color_trc": "bt709", "color_range": "tv"},
    }}}
    assert colour_match_filter(same) is None
    assert colour_match_filter(None) is None
    hdr = {"plan": {"color": {**sd["plan"]["color"], "hdr_passthrough": True}}}
    assert colour_match_filter(hdr) is None
