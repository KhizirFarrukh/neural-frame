"""Structured logging (structlog) with job/chunk context and an optional JSON-lines file sink."""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections.abc import Callable, MutableMapping
from pathlib import Path
from typing import Any, TextIO, cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, Processor, WrappedLogger

_LOCK = threading.Lock()
_SINKS: list[Callable[[dict[str, Any]], None]] = []


class _JsonFileSink:
    """Appends every event as one JSON line; thread-safe, line-buffered."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def __call__(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, default=str, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")


def _tee(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """Forward a copy of each event to registered sinks (file, GUI, API event stream)."""
    if _SINKS:
        snapshot = dict(event_dict)
        for sink in list(_SINKS):
            try:
                sink(snapshot)
            except Exception:  # a broken sink must never break the pipeline
                pass
    return event_dict


class _DynamicStderrLogger:
    """Writes to whatever ``sys.stderr`` is *now*, so Rich's live display can redirect log lines."""

    def msg(self, message: str) -> None:
        stream = sys.stderr
        stream.write(message + "\n")
        stream.flush()

    log = debug = info = warning = warn = error = critical = exception = fatal = msg


def add_sink(sink: Callable[[dict[str, Any]], None]) -> None:
    with _LOCK:
        _SINKS.append(sink)


def remove_sink(sink: Callable[[dict[str, Any]], None]) -> None:
    with _LOCK:
        if sink in _SINKS:
            _SINKS.remove(sink)


def configure_logging(
    level: str = "INFO",
    *,
    json_output: bool = False,
    log_file: Path | None = None,
    stream: TextIO | None = None,
) -> None:
    """Configure structlog once per process. Safe to call again (e.g. CLI flags override config)."""
    level_no = logging.getLevelName(level.upper())
    if not isinstance(level_no, int):
        level_no = logging.INFO
    out = stream if stream is not None else sys.stderr
    factory: Callable[..., Any]
    if stream is None:
        factory = lambda *_: _DynamicStderrLogger()  # noqa: E731
    else:
        factory = structlog.PrintLoggerFactory(file=stream)

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _tee,
    ]
    if json_output:
        processors.append(structlog.processors.JSONRenderer(default=str))
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=out.isatty()))

    with _LOCK:
        for sink in [s for s in _SINKS if isinstance(s, _JsonFileSink)]:
            _SINKS.remove(sink)
        if log_file is not None:
            _SINKS.append(_JsonFileSink(log_file))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level_no),
        logger_factory=factory,
        cache_logger_on_first_use=False,
    )


def get_logger(**initial: Any) -> FilteringBoundLogger:
    return cast(FilteringBoundLogger, structlog.get_logger(**initial))


def bind_context(**values: Any) -> None:
    """Bind values (job_id, chunk_id, …) to the current context; inherited by stage threads."""
    structlog.contextvars.bind_contextvars(**values)


def clear_context(*keys: str) -> None:
    if keys:
        structlog.contextvars.unbind_contextvars(*keys)
    else:
        structlog.contextvars.clear_contextvars()


def current_context() -> MutableMapping[str, Any]:
    return structlog.contextvars.get_contextvars()
