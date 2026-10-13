"""Running an engine's (or the installer's) command: live output lines, cancellation, a log file."""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.log import get_logger

log = get_logger(component="engines")

_SPLIT = re.compile(r"[\r\n]+")


class Cancellable(Protocol):
    def is_set(self) -> bool: ...


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def run(
    cmd: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    env: dict[str, str],
    on_line: Callable[[str], None] | None = None,
    cancel: Cancellable | None = None,
    log_path: Path | None = None,
    what: str = "engine",
    timeout_s: float | None = None,
) -> None:
    """Run ``cmd``; every output line (``\\r`` progress bars included) goes to ``on_line`` and the log.
    Raises :class:`JobCancelled` when cancelled (the whole process tree is stopped) and
    ``INFERENCE_FAILED`` with the last output lines when the command fails."""
    args = [str(c) for c in cmd]
    log.info("running", what=what, cmd=" ".join(args), cwd=str(cwd))
    kwargs: dict[str, object] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    tail: deque[str] = deque(maxlen=40)
    log_file = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8", errors="replace")
        log_file.write(f"\n$ {' '.join(args)}\n")
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, **kwargs)  # type: ignore[call-overload]
    stopped = threading.Event()

    def watch() -> None:
        while not stopped.wait(0.25):
            if cancel is not None and cancel.is_set():
                _kill_tree(proc)
                return

    watcher = threading.Thread(target=watch, name=f"nf-{what}-cancel", daemon=True)
    watcher.start()
    buffer = ""
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            parts = _SPLIT.split(buffer)
            buffer = parts.pop()
            for line in parts:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                if log_file is not None:
                    log_file.write(line + "\n")
                if on_line is not None:
                    try:
                        on_line(line)
                    except Exception as exc:  # a broken progress callback must not stop the engine
                        log.warning("output callback failed", error=str(exc))
        if buffer.strip():
            tail.append(buffer.strip())
        code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        raise NeuralFrameError(ErrorCode.INFERENCE_FAILED, f"{what} did not finish in time") from None
    finally:
        stopped.set()
        if log_file is not None:
            log_file.close()
    if cancel is not None and cancel.is_set():
        raise JobCancelled(f"{what} cancelled")
    if code != 0:
        last = "\n".join(list(tail)[-12:])
        raise NeuralFrameError(ErrorCode.INFERENCE_FAILED, f"{what} failed (exit code {code}):\n{last}",
                               hint=f"full output: {log_path}" if log_path else None)
