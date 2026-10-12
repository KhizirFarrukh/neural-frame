"""Pipeline runtime primitives: cancellation, bounded queues with metrics, stage threads."""

from __future__ import annotations

import contextvars
import queue
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from neuralframe.errors import JobCancelled, NeuralFrameError
from neuralframe.log import get_logger

T = TypeVar("T")
U = TypeVar("U")

_POLL_S = 0.1


class CancelToken:
    """Cooperative cancellation shared by every stage of a job."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self.reason = ""

    def cancel(self, reason: str = "cancelled by user") -> None:
        if not self._event.is_set():
            self.reason = reason
            self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise JobCancelled(self.reason or "cancelled")

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


@dataclass(slots=True)
class Frame(Generic[T]):
    """A frame travelling through the pipeline. ``pts``/``duration`` are in the pipeline time base."""

    index: int
    pts: int
    duration: int
    data: T

    def replace(self, data: U) -> Frame[U]:
        return Frame(self.index, self.pts, self.duration, data)


class _End:
    pass


_END = _End()


class StageQueue(Generic[T]):
    """Bounded FIFO between stages (backpressure) that aborts promptly on cancellation."""

    def __init__(self, name: str, maxsize: int, cancel: CancelToken) -> None:
        self.name = name
        self.maxsize = maxsize
        self._q: queue.Queue[T | _End] = queue.Queue(maxsize=maxsize)
        self._cancel = cancel
        self.put_wait_s = 0.0
        self.get_wait_s = 0.0
        self.max_depth = 0

    def put(self, item: T | _End) -> None:
        start = time.perf_counter()
        while True:
            self._cancel.raise_if_cancelled()
            try:
                self._q.put(item, timeout=_POLL_S)
                break
            except queue.Full:
                continue
        self.put_wait_s += time.perf_counter() - start
        self.max_depth = max(self.max_depth, self._q.qsize())

    def get(self) -> T | None:
        """Next item, or None at end of stream."""
        start = time.perf_counter()
        while True:
            self._cancel.raise_if_cancelled()
            try:
                item = self._q.get(timeout=_POLL_S)
                break
            except queue.Empty:
                continue
        self.get_wait_s += time.perf_counter() - start
        if isinstance(item, _End):
            self._q.put(item)  # let other consumers of this queue see the end too
            return None
        return item

    def close(self) -> None:
        self.put(_END)

    @property
    def depth(self) -> int:
        return self._q.qsize()


@dataclass
class StageMetrics:
    name: str
    frames: int = 0
    busy_s: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None

    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at if self.finished_at is not None else time.perf_counter()
        return max(end - self.started_at, 1e-9)

    @property
    def fps(self) -> float:
        """Frames per wall-clock second since the stage started."""
        return self.frames / self.elapsed() if self.started_at is not None else 0.0

    @property
    def busy_fps(self) -> float:
        """Throughput if the stage never waited — its capacity; the lowest one is the bottleneck."""
        return self.frames / self.busy_s if self.busy_s > 0 else 0.0

    @property
    def utilisation(self) -> float:
        return min(1.0, self.busy_s / self.elapsed()) if self.started_at is not None else 0.0


class Stage(ABC):
    """One pipeline stage running on its own thread."""

    def __init__(self, name: str, cancel: CancelToken) -> None:
        self.name = name
        self.cancel = cancel
        self.metrics = StageMetrics(name)
        self.log = get_logger(stage=name)

    @abstractmethod
    def run(self) -> None:
        """Consume input until end-of-stream; close the output queue only on normal completion."""

    def timed(self) -> _BusyTimer:
        return _BusyTimer(self.metrics)


class _BusyTimer:
    def __init__(self, metrics: StageMetrics) -> None:
        self._metrics = metrics
        self._t0 = 0.0

    def __enter__(self) -> None:
        self._t0 = time.perf_counter()

    def __exit__(self, *_: object) -> None:
        self._metrics.busy_s += time.perf_counter() - self._t0


class StageRunner:
    """Runs stages on threads; the first failure cancels the rest and is re-raised."""

    def __init__(self, stages: list[Stage], cancel: CancelToken) -> None:
        self.stages = stages
        self.cancel = cancel
        self._errors: list[BaseException] = []
        self._lock = threading.Lock()

    def _wrap(self, stage: Stage) -> None:
        stage.metrics.started_at = time.perf_counter()
        try:
            stage.run()
        except JobCancelled:
            pass
        except BaseException as exc:
            stage.log.error("stage failed", error=str(exc))
            with self._lock:
                self._errors.append(exc)
            self.cancel.cancel(f"stage {stage.name} failed")
        finally:
            stage.metrics.finished_at = time.perf_counter()

    def run(self, on_tick: Callable[[], None] | None = None, interval_s: float = 0.5) -> None:
        threads = []
        for stage in self.stages:
            ctx = contextvars.copy_context()  # propagate job_id / chunk_id log context to threads
            t = threading.Thread(target=ctx.run, args=(self._wrap, stage), name=f"stage-{stage.name}", daemon=True)
            threads.append(t)
            t.start()
        last_tick = 0.0
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.05)
                now = time.perf_counter()
                if on_tick is not None and now - last_tick >= interval_s:
                    last_tick = now
                    on_tick()
        except KeyboardInterrupt:
            # Ctrl+C: stop cooperatively so finished chunks stay valid for --resume.
            self.cancel.cancel("interrupted")
            for t in threads:
                t.join()
        if on_tick is not None:
            on_tick()
        if self._errors:
            first = next((e for e in self._errors if isinstance(e, NeuralFrameError)), self._errors[0])
            raise first
        self.cancel.raise_if_cancelled()
