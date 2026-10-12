from __future__ import annotations

from pathlib import Path

import pytest

from neuralframe.chunking.planner import ChunkPlan, plan_fixed_chunks
from neuralframe.chunking.store import ChunkStore
from neuralframe.errors import NeuralFrameError


def test_fixed_chunks_cover_all_frames_contiguously() -> None:
    plan = plan_fixed_chunks(1000, 240)
    assert [c.start for c in plan.chunks] == [0, 240, 480, 720]
    assert plan.chunks[-1].end == 1000
    assert sum(c.frames for c in plan.chunks) == 1000


def test_tiny_tail_is_merged() -> None:
    plan = plan_fixed_chunks(250, 240)
    assert len(plan.chunks) == 1 and plan.chunks[0].frames == 250


def test_chunk_of_and_open_ended_last_chunk() -> None:
    plan = plan_fixed_chunks(500, 100)
    assert plan.chunk_of(0).id == 0
    assert plan.chunk_of(99).id == 0
    assert plan.chunk_of(100).id == 1
    assert plan.chunk_of(499).id == 4
    assert plan.chunk_of(10_000).id == 4  # more decoded frames than probed still have a home


def test_plan_rejects_gaps() -> None:
    from neuralframe.chunking.planner import Chunk

    with pytest.raises(ValueError):
        ChunkPlan((Chunk(0, 0, 10), Chunk(1, 11, 20)), 20, 10)


def _segment(store: ChunkStore, cid: int, payload: bytes) -> Path:
    path = store.segment_path(cid)
    path.write_bytes(payload)
    return path


def test_store_commit_resume_and_corruption(tmp_path: Path) -> None:
    root = tmp_path / "job"
    store = ChunkStore.open(root, "key", "fp", resume=False)
    for cid in range(3):
        seg = _segment(store, cid, b"x" * (cid + 1))
        store.commit(cid, cid * 10, cid * 10 + 10, 10, cid * 100, cid * 100 + 90, seg)

    resumed = ChunkStore.open(root, "key", "fp", resume=True)
    assert sorted(resumed.completed()) == [0, 1, 2]
    assert [p.name for p in resumed.ordered_segments()] == [f"chunk_{i:05d}.mp4" for i in range(3)]

    resumed.segment_path(1).write_bytes(b"tampered")
    again = ChunkStore.open(root, "key", "fp", resume=True)
    assert sorted(again.completed()) == [0, 2]


def test_store_refuses_resume_with_different_settings(tmp_path: Path) -> None:
    root = tmp_path / "job"
    ChunkStore.open(root, "key", "fp-1", resume=False)
    with pytest.raises(NeuralFrameError, match="different settings"):
        ChunkStore.open(root, "key", "fp-2", resume=True)


def test_fresh_open_wipes_previous_state(tmp_path: Path) -> None:
    root = tmp_path / "job"
    store = ChunkStore.open(root, "key", "fp", resume=False)
    store.commit(0, 0, 10, 10, 0, 9, _segment(store, 0, b"abc"))
    fresh = ChunkStore.open(root, "key", "fp", resume=False)
    assert fresh.completed() == {}
    assert not fresh.segment_path(0).exists()
