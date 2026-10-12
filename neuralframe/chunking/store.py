"""Chunk store: finished segments plus a manifest, so interrupted jobs resume where they stopped."""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.log import get_logger

log = get_logger(component="chunk-store")

MANIFEST_VERSION = 1


class ChunkRecord(BaseModel):
    id: int
    start: int
    end: int
    frames: int
    first_pts: int
    last_pts: int
    file: str
    size: int
    sha256: str
    completed_at: float = Field(default_factory=time.time)


class Manifest(BaseModel):
    version: int = MANIFEST_VERSION
    job_key: str
    fingerprint: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    time_base: str | None = None
    chunks: dict[int, ChunkRecord] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


def sha256_file(path: Path, bufsize: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(bufsize):
            digest.update(block)
    return digest.hexdigest()


class ChunkStore:
    def __init__(self, root: Path, manifest: Manifest) -> None:
        self.root = root
        self.manifest = manifest

    # ---- lifecycle -----------------------------------------------------------------------------------

    @classmethod
    def open(cls, root: Path, job_key: str, fingerprint: str, *, resume: bool) -> ChunkStore:
        """Open (resume) or create a store. A fresh run wipes stale state for the same job key."""
        manifest_path = root / "manifest.json"
        if resume and manifest_path.is_file():
            try:
                manifest = Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise NeuralFrameError(
                    ErrorCode.RESUME_MISMATCH, f"cannot read resume manifest {manifest_path}: {exc}"
                ) from exc
            if manifest.fingerprint != fingerprint or manifest.version != MANIFEST_VERSION:
                raise NeuralFrameError(
                    ErrorCode.RESUME_MISMATCH,
                    "the saved job state was produced with different settings or input",
                    hint="re-run without --resume to start over",
                )
            store = cls(root, manifest)
            store._drop_invalid()
            return store
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        (root / "segments").mkdir(parents=True, exist_ok=True)
        store = cls(root, Manifest(job_key=job_key, fingerprint=fingerprint))
        store.save()
        return store

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def save(self) -> None:
        self.manifest.updated_at = time.time()
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(self.manifest.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.manifest_path)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # ---- segments ------------------------------------------------------------------------------------

    def segment_path(self, chunk_id: int) -> Path:
        return self.root / "segments" / f"chunk_{chunk_id:05d}.mp4"

    def commit(
        self, chunk_id: int, start: int, end: int, frames: int, first_pts: int, last_pts: int, path: Path
    ) -> ChunkRecord:
        record = ChunkRecord(
            id=chunk_id,
            start=start,
            end=end,
            frames=frames,
            first_pts=first_pts,
            last_pts=last_pts,
            file=path.name,
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        self.manifest.chunks[chunk_id] = record
        self.save()
        log.debug("chunk committed", chunk_id=chunk_id, frames=frames, size=record.size)
        return record

    def completed(self) -> dict[int, ChunkRecord]:
        return dict(self.manifest.chunks)

    def verify(self, record: ChunkRecord) -> bool:
        path = self.root / "segments" / record.file
        if not path.is_file() or path.stat().st_size != record.size:
            return False
        return sha256_file(path) == record.sha256

    def _drop_invalid(self) -> None:
        bad = [cid for cid, rec in self.manifest.chunks.items() if not self.verify(rec)]
        for cid in bad:
            log.warning("discarding corrupt or missing chunk", chunk_id=cid)
            del self.manifest.chunks[cid]
            with contextlib.suppress(OSError):
                self.segment_path(cid).unlink()
        if bad:
            self.save()

    def ordered_segments(self) -> list[Path]:
        return [self.root / "segments" / self.manifest.chunks[cid].file for cid in sorted(self.manifest.chunks)]

    def set_meta(self, **values: Any) -> None:
        self.manifest.meta.update(values)
        self.save()

    def dump(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.manifest.model_dump_json())
        return data
