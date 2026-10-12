"""Background work for the GUI without touching widgets off the GUI thread.

Long work (probing, previews, the job itself) runs in Python threads; results come back through
:class:`Dispatcher`, whose queued connection always runs the callback on the thread that owns the
dispatcher (the GUI thread).
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal, Slot


class Dispatcher(QObject):
    call = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        # Parent it to the window: when the window is destroyed, queued callbacks are dropped with it.
        super().__init__(parent)
        self.call.connect(self._invoke, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def _invoke(self, fn: Callable[[], None]) -> None:
        fn()

    def post(self, fn: Callable[[], None]) -> None:
        """Run ``fn`` on the GUI thread (callable from any thread); ignored once the owner is gone."""
        try:
            self.call.emit(fn)
        except RuntimeError:  # the underlying Qt object was deleted (window closed)
            pass

    def run(
        self,
        work: Callable[[], Any],
        done: Callable[[Any], None],
        failed: Callable[[BaseException], None],
        *,
        name: str = "nf-task",
    ) -> threading.Thread:
        """Run ``work`` in a background thread; ``done(result)`` or ``failed(exc)`` runs on the GUI thread."""

        def target() -> None:
            try:
                result = work()
            except BaseException as exc:  # reported to the UI, never lost in a thread
                # Bind now: Python unbinds ``exc`` when the except block ends.
                self.post(functools.partial(failed, exc))
            else:
                self.post(lambda: done(result))

        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread
