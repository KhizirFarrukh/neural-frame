"""Temporal chunk planning.

A chunk is a contiguous range of *output* frame indices that is encoded into its own segment. Chunks exist
for scheduling (multi-GPU), resumability and bounded memory; for single-frame models they never change
pixels, only where segment boundaries fall.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Chunk:
    id: int
    start: int
    end: int
    """Exclusive. The last chunk is open-ended at runtime if the decoder yields more frames than planned."""
    scene_cut_before: bool = False

    @property
    def frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ChunkPlan:
    chunks: tuple[Chunk, ...]
    total_frames: int
    chunk_frames: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.chunks:
            raise ValueError("a chunk plan needs at least one chunk")
        for prev, cur in zip(self.chunks, self.chunks[1:], strict=False):
            if prev.end != cur.start:
                raise ValueError(f"chunks {prev.id} and {cur.id} are not contiguous")
        object.__setattr__(self, "_starts", tuple(c.start for c in self.chunks))

    def chunk_of(self, index: int) -> Chunk:
        """Chunk containing output frame ``index``; frames past the plan belong to the last chunk."""
        starts: tuple[int, ...] = object.__getattribute__(self, "_starts")
        pos = bisect.bisect_right(starts, index) - 1
        return self.chunks[max(pos, 0)]

    def by_id(self, chunk_id: int) -> Chunk:
        return self.chunks[chunk_id]

    def summary(self) -> dict[str, object]:
        return {
            "chunks": len(self.chunks),
            "chunk_frames": self.chunk_frames,
            "total_frames": self.total_frames,
            "scene_aligned_boundaries": sum(1 for c in self.chunks if c.scene_cut_before),
            "notes": list(self.notes),
        }


def plan_fixed_chunks(total_frames: int, chunk_frames: int) -> ChunkPlan:
    """Evenly sized chunks; the remainder is merged into the last chunk if it would be tiny."""
    total = max(total_frames, 1)
    size = max(chunk_frames, 1)
    bounds = list(range(0, total, size))
    if len(bounds) > 1 and total - bounds[-1] < size // 4:
        bounds.pop()  # avoid a sliver chunk at the end
    chunks = [
        Chunk(i, start, bounds[i + 1] if i + 1 < len(bounds) else total)
        for i, start in enumerate(bounds)
    ]
    return ChunkPlan(tuple(chunks), total, size)


def plan_chunks(
    total_frames: int,
    chunk_frames: int,
    *,
    scene_cuts: list[int] | None = None,
    force_cut_boundaries: bool = False,
    align_window: int | None = None,
) -> ChunkPlan:
    """Chunk plan whose boundaries prefer scene cuts.

    * A boundary near its nominal position (``± align_window``, default a quarter chunk) moves onto the
      closest scene cut — cuts are natural keyframe/segment points and nothing is lost there.
    * ``force_cut_boundaries`` (temporal models) makes *every* cut a boundary, so no temporal window and
      no cross-fade ever spans two shots, even if that creates short chunks.
    """
    total = max(total_frames, 1)
    size = max(chunk_frames, 1)
    window = size // 4 if align_window is None else align_window
    cuts = sorted({c for c in (scene_cuts or []) if 0 < c < total})
    cut_set = set(cuts)
    bounds: list[int] = [0]
    pos = 0
    while pos < total:
        nominal = pos + size
        next_forced = next((c for c in cuts if c > pos), None) if force_cut_boundaries else None
        if next_forced is not None and next_forced <= nominal:
            boundary = next_forced
        elif nominal >= total:
            break
        else:
            near = [c for c in cuts if abs(c - nominal) <= window and c > pos]
            boundary = min(near, key=lambda c: (abs(c - nominal), c)) if near else nominal
        if boundary >= total:
            break
        bounds.append(boundary)
        pos = boundary
    # merge a sliver tail unless its start is a scene cut that must stay a boundary
    if len(bounds) > 1 and total - bounds[-1] < size // 4 and not (force_cut_boundaries and bounds[-1] in cut_set):
        bounds.pop()
    chunks = [
        Chunk(i, start, bounds[i + 1] if i + 1 < len(bounds) else total, scene_cut_before=start in cut_set)
        for i, start in enumerate(bounds)
    ]
    aligned = sum(1 for c in chunks if c.scene_cut_before)
    notes = (f"{aligned} of {max(len(cuts), 0)} scene cuts used as chunk boundaries",) if cuts else ()
    return ChunkPlan(tuple(chunks), total, size, notes)
