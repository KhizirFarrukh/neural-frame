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
