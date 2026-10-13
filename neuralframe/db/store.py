"""Persistent job queue in SQLite (SQLAlchemy 2.0).

Jobs survive restarts: anything that was active when the server stopped is put back in the queue with
``resume`` set, so finished chunks are reused. Ordering is priority first, then submission order.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Float, Integer, String, Text, create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from neuralframe.core.job import JobSpec

ACTIVE_STATES = ("probing", "planning", "downloading", "running", "muxing", "verifying")
TERMINAL_STATES = ("done", "failed", "cancelled")
RESUMABLE_STATES = ("failed", "cancelled", "paused")


class Base(DeclarativeBase):
    pass


class JobRecord(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, index=True)
    """Submission order (FIFO within a priority)."""
    priority: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[str] = mapped_column(String(16), index=True)
    spec_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[float] = mapped_column(Float)
    started_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    finished_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    worker: Mapped[str | None] = mapped_column(String(32), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="")
    frames_done: Mapped[int] = mapped_column(Integer, default=0)
    frames_total: Mapped[int] = mapped_column(Integer, default=0)
    percent: Mapped[float] = mapped_column(Float, default=0.0)
    fps: Mapped[float] = mapped_column(Float, default=0.0)
    eta_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    progress_json: Mapped[str] = mapped_column(Text, default="{}")
    """Latest per-stage throughput and GPU sample."""
    output: Mapped[str | None] = mapped_column(Text, nullable=True)
    report_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_hint: Mapped[str | None] = mapped_column(Text, nullable=True)

    @property
    def spec(self) -> JobSpec:
        return JobSpec.model_validate_json(self.spec_json)

    def as_dict(self) -> dict[str, Any]:
        spec = json.loads(self.spec_json)
        progress = json.loads(self.progress_json or "{}")
        return {
            "id": self.id,
            "state": self.state,
            "priority": self.priority,
            "input": spec.get("input"),
            "output": self.output or spec.get("output"),
            "spec": spec,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "worker": self.worker,
            "attempts": self.attempts,
            "message": self.message,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "percent": round(self.percent, 2),
            "fps": round(self.fps, 2),
            "eta_s": self.eta_s,
            "stages": progress.get("stages", []),
            "gpu": progress.get("gpu"),
            "report_path": self.report_path,
            "error": (
                {"code": self.error_code, "message": self.error_message, "hint": self.error_hint}
                if self.error_code else None
            ),
        }


class JobStore:
    """Thread-safe access to the jobs table (one writer lock; SQLite in WAL mode)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.engine: Engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def _pragmas(dbapi_conn: Any, _record: Any) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=5000")
            cur.close()

        Base.metadata.create_all(self.engine)
        self._lock = threading.RLock()

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self._lock, Session(self.engine, expire_on_commit=False) as session:
            yield session
            session.commit()

    def create(self, spec: JobSpec, *, priority: int = 0) -> dict[str, Any]:
        now = time.time()
        with self._session() as s:
            last = s.scalar(select(JobRecord.seq).order_by(JobRecord.seq.desc()).limit(1))
            record = JobRecord(
                id=uuid.uuid4().hex[:12], seq=(last or 0) + 1, priority=priority, state="queued",
                spec_json=spec.model_dump_json(), created_at=now, updated_at=now, message="queued",
                frames_done=0, frames_total=0, percent=0.0, fps=0.0, progress_json="{}", attempts=0,
            )
            s.add(record)
            s.flush()
            return record.as_dict()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._session() as s:
            record = s.get(JobRecord, job_id)
            return record.as_dict() if record else None

    def spec(self, job_id: str) -> JobSpec | None:
        with self._session() as s:
            record = s.get(JobRecord, job_id)
            return record.spec if record else None

    def list_jobs(self, *, states: tuple[str, ...] | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._session() as s:
            query = select(JobRecord).order_by(JobRecord.seq.desc()).limit(limit)
            if states:
                query = query.where(JobRecord.state.in_(states))
            return [r.as_dict() for r in s.scalars(query)]

    def claim_next(self, worker: str, accepts: Any = None) -> tuple[str, JobSpec] | None:
        """Atomically move the best queued job this worker accepts to ``probing``."""
        with self._session() as s:
            query = select(JobRecord).where(JobRecord.state == "queued").order_by(
                JobRecord.priority.desc(), JobRecord.seq.asc()
            )
            for record in s.scalars(query):
                spec = record.spec
                if accepts is not None and not accepts(spec):
                    continue
                now = time.time()
                record.state = "probing"
                record.worker = worker
                record.attempts += 1
                record.started_at = now
                record.finished_at = None
                record.updated_at = now
                record.message = "starting"
                record.error_code = record.error_message = record.error_hint = None
                return record.id, spec
        return None

    def update(self, job_id: str, **fields: Any) -> None:
        if "progress" in fields:
            fields["progress_json"] = json.dumps(fields.pop("progress"), default=str)
        with self._session() as s:
            record = s.get(JobRecord, job_id)
            if record is None:
                return
            for key, value in fields.items():
                setattr(record, key, value)
            record.updated_at = time.time()

    def set_spec(self, job_id: str, spec: JobSpec) -> None:
        self.update(job_id, spec_json=spec.model_dump_json())

    def cancel_queued(self, job_id: str) -> bool:
        with self._session() as s:
            record = s.get(JobRecord, job_id)
            if record is None or record.state != "queued":
                return False
            record.state = "cancelled"
            record.message = "cancelled before it started"
            record.finished_at = record.updated_at = time.time()
            return True

    def requeue(self, job_id: str, *, message: str) -> dict[str, Any] | None:
        """Queue a stopped job again with ``resume`` set (finished chunks are reused)."""
        with self._session() as s:
            record = s.get(JobRecord, job_id)
            if record is None:
                return None
            spec = record.spec.model_copy(update={"resume": True, "overwrite": True})
            record.spec_json = spec.model_dump_json()
            record.state = "queued"
            record.message = message
            record.worker = None
            record.finished_at = None
            record.error_code = record.error_message = record.error_hint = None
            record.updated_at = time.time()
            return record.as_dict()

    def recover_interrupted(self) -> list[str]:
        """Jobs that were active (or paused by a shutdown) when the server stopped go back to the queue."""
        with self._session() as s:
            ids = list(s.scalars(select(JobRecord.id).where(JobRecord.state.in_((*ACTIVE_STATES, "paused")))))
        for job_id in ids:
            self.requeue(job_id, message="server restarted; resuming from the last finished chunk")
        return ids

    def delete(self, job_id: str) -> bool:
        with self._session() as s:
            record = s.get(JobRecord, job_id)
            if record is None:
                return False
            s.delete(record)
            return True

    def close(self) -> None:
        self.engine.dispose()
