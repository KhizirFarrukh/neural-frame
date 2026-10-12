"""Runs queued jobs: one worker per GPU by default, progress fanned out to API clients.

Each worker thread claims the best queued job it can run (priority, then FIFO), runs it with the same
:class:`JobRunner` the CLI uses, keeps the latest progress in memory for Server-Sent Events and writes it
to the store at most once a second (and on every state change). Stopping the manager pauses running jobs
at the next frame; they are resumed from their finished chunks when the server starts again.
"""

from __future__ import annotations

import importlib.util
import threading
import time
from collections.abc import Callable
from typing import Any

from neuralframe.config import Settings
from neuralframe.core.job import JobRunner, JobSpec, ProgressEvent
from neuralframe.db.store import RESUMABLE_STATES, JobStore
from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.log import get_logger

log = get_logger(component="manager")


def default_workers(settings: Settings) -> list[str]:
    """One worker per visible CUDA device (one active job per GPU), else a single CPU worker."""
    if settings.inference.device != "cpu" and importlib.util.find_spec("torch") is not None:
        from neuralframe.inference.factory import torch_cuda_device_count

        count = torch_cuda_device_count()
        if count:
            gpus = settings.inference.gpus or list(range(count))
            return [f"cuda:{i}" for i in gpus]
    return ["cpu"]


class JobManager:
    def __init__(
        self,
        settings: Settings,
        store: JobStore,
        *,
        workers: list[str] | None = None,
        runner_factory: Callable[..., JobRunner] = JobRunner,
        runner_kwargs: dict[str, Any] | None = None,
        poll_s: float = 0.5,
    ) -> None:
        self.settings = settings
        self.store = store
        self.workers = workers or default_workers(settings)
        self._runner_factory = runner_factory
        self._runner_kwargs = runner_kwargs or {}
        self._poll_s = poll_s
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._runners: dict[str, JobRunner] = {}
        self._live: dict[str, dict[str, Any]] = {}
        self._versions: dict[str, int] = {}

    # ---- lifecycle -------------------------------------------------------------------------------------

    def start(self) -> list[str]:
        recovered = self.store.recover_interrupted()
        if recovered:
            log.info("re-queued interrupted jobs", jobs=recovered)
        for name in self.workers:
            thread = threading.Thread(target=self._worker, args=(name,), name=f"nf-worker-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        return recovered

    def stop(self, timeout: float = 60.0) -> None:
        """Pause running jobs (they resume on the next start) and stop the workers."""
        self._stop.set()
        for runner in list(self._runners.values()):
            runner.cancel.cancel("server shutting down")
        with self._cond:
            self._cond.notify_all()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        self._threads.clear()

    @property
    def running(self) -> list[str]:
        return list(self._runners)

    # ---- commands --------------------------------------------------------------------------------------

    def submit(self, spec: JobSpec, *, priority: int = 0) -> dict[str, Any]:
        record = self.store.create(spec, priority=priority)
        log.info("job queued", job_id=record["id"], input=str(spec.input), priority=priority)
        self._wake()
        return record

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self._require(job_id)
        runner = self._runners.get(job_id)
        if runner is not None:
            runner.cancel.cancel("cancelled by user")
        elif record["state"] == "queued":
            self.store.cancel_queued(job_id)
            self._publish(job_id, self._require(job_id))
        else:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"job {job_id} is {record['state']}; nothing to cancel")
        return self._require(job_id)

    def resume(self, job_id: str) -> dict[str, Any]:
        record = self._require(job_id)
        if record["state"] not in RESUMABLE_STATES:
            raise NeuralFrameError(
                ErrorCode.INVALID_INPUT, f"job {job_id} is {record['state']}; only {', '.join(RESUMABLE_STATES)} "
                "jobs can be resumed"
            )
        updated = self.store.requeue(job_id, message="queued to resume from the last finished chunk")
        assert updated is not None
        self._publish(job_id, updated)
        self._wake()
        return updated

    def delete(self, job_id: str) -> None:
        record = self._require(job_id)
        if job_id in self._runners or record["state"] == "queued":
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"job {job_id} is {record['state']}; cancel it first")
        self.store.delete(job_id)
        with self._cond:
            self._live.pop(job_id, None)
            self._versions.pop(job_id, None)

    # ---- views -----------------------------------------------------------------------------------------

    def job(self, job_id: str) -> dict[str, Any] | None:
        """Stored record overlaid with live progress while the job runs."""
        record = self.store.get(job_id)
        if record is None:
            return None
        live = self._live.get(job_id)
        if live is not None and job_id in self._runners:
            record.update({k: live[k] for k in ("state", "message", "frames_done", "frames_total", "percent",
                                                 "fps", "eta_s", "stages", "gpu") if k in live})
        return record

    def version(self, job_id: str) -> int:
        return self._versions.get(job_id, 0)

    def _require(self, job_id: str) -> dict[str, Any]:
        record = self.store.get(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    # ---- workers ---------------------------------------------------------------------------------------

    def _wake(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def _publish(self, job_id: str, data: dict[str, Any]) -> None:
        with self._cond:
            self._live[job_id] = data
            self._versions[job_id] = self._versions.get(job_id, 0) + 1
            self._cond.notify_all()

    @staticmethod
    def _accepts(worker: str, spec: JobSpec) -> bool:
        if worker.startswith("cuda:") and spec.gpus:
            return spec.gpus[0] == int(worker.split(":", 1)[1])
        return True

    def _worker(self, name: str) -> None:
        while not self._stop.is_set():
            claimed = self.store.claim_next(name, accepts=lambda spec: self._accepts(name, spec))
            if claimed is None:
                with self._cond:
                    self._cond.wait(self._poll_s)
                continue
            job_id, spec = claimed
            try:
                self._run(job_id, spec, name)
            except Exception as exc:  # a worker must survive anything a job throws
                log.exception("worker error", job_id=job_id, error=str(exc))
                self.store.update(job_id, state="failed", error_code=ErrorCode.INTERNAL.value,
                                  error_message=str(exc), finished_at=time.time())

    def _run(self, job_id: str, spec: JobSpec, worker: str) -> None:
        if worker.startswith("cuda:") and spec.gpus is None:
            spec = spec.model_copy(update={"gpus": [int(worker.split(":", 1)[1])]})
        last_write: dict[str, Any] = {"at": 0.0, "state": ""}

        def on_progress(event: ProgressEvent) -> None:
            data = event.as_dict()
            self._publish(job_id, data)
            now = time.monotonic()
            if event.state in ("done", "failed", "cancelled"):
                return  # the final record is written below with the result
            if event.state != last_write["state"] or now - last_write["at"] >= 1.0:
                last_write.update(at=now, state=event.state)
                self.store.update(
                    job_id, state=event.state, message=event.message, frames_done=event.frames_done,
                    frames_total=event.frames_total, percent=event.percent, fps=event.fps, eta_s=event.eta_s,
                    progress={"stages": data.get("stages", []), "gpu": data.get("gpu")},
                )

        runner = self._runner_factory(self.settings, job_id=job_id, progress=on_progress, **self._runner_kwargs)
        self._runners[job_id] = runner
        try:
            result = runner.run(spec)
            report = result.report
            frames = report.get("output", {}).get("frames") if isinstance(report.get("output"), dict) else None
            self.store.update(
                job_id, state="done", output=str(result.output), report_path=str(result.report_path),
                percent=100.0, frames_done=frames or 0, frames_total=frames or 0, eta_s=0.0,
                message=f"wrote {result.output}", finished_at=time.time(),
            )
            log.info("job done", job_id=job_id, output=str(result.output))
        except JobCancelled:
            paused = self._stop.is_set()
            self.store.update(
                job_id, state="paused" if paused else "cancelled", finished_at=time.time(),
                message=("paused by server shutdown; resumes on restart" if paused
                         else "cancelled; finished chunks are kept for resume"),
            )
        except NeuralFrameError as exc:
            self.store.update(job_id, state="failed", error_code=exc.code.value, error_message=exc.message,
                              error_hint=exc.hint, message=str(exc), finished_at=time.time())
            log.warning("job failed", job_id=job_id, error=str(exc))
        finally:
            self._runners.pop(job_id, None)
            record = self.store.get(job_id)
            if record is not None:
                self._publish(job_id, record)
