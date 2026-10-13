"""Model manager: every model with what it is for, its size and status; download (with a progress bar,
speed and time left), cancel, delete, use a file you already have, and updates."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from neuralframe.errors import JobCancelled, NeuralFrameError
from neuralframe.gui.threads import Dispatcher
from neuralframe.models.library import DownloadProgress, ModelLibrary, ModelStatus
from neuralframe.models.registry import CATEGORY_TITLES, SPEED_WORDS, ModelRegistry, ModelSpec

SHOW_CHOICES = [("All models", "all"), ("On this computer", "ready"), ("Not downloaded", "missing"),
                ("Updates available", "updates")]


def error_text(exc: BaseException) -> str:
    if isinstance(exc, NeuralFrameError):
        return exc.message + (f"\n\n{exc.hint}" if exc.hint else "")
    return str(exc)


class DownloadController(QObject):
    """Model downloads in background threads (several at once), reported on the GUI thread. Owned by the
    main window so downloads keep running while the model manager is closed."""

    progress = Signal(str, object)
    """model id, :class:`DownloadProgress`"""
    finished = Signal(str, str, str)
    """model id, outcome (``done`` / ``cancelled`` / ``failed``), error text"""

    def __init__(self, library: ModelLibrary, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.library = library
        self._dispatch = Dispatcher(self)
        self._cancel: dict[str, threading.Event] = {}
        self.last: dict[str, DownloadProgress] = {}

    def active(self) -> set[str]:
        return set(self._cancel)

    def is_active(self, model_id: str) -> bool:
        return model_id in self._cancel

    def start(self, spec: ModelSpec, *, update: bool = False) -> bool:
        if spec.id in self._cancel:
            return False
        cancel = threading.Event()
        self._cancel[spec.id] = cancel

        def on_progress(event: DownloadProgress) -> None:
            self._dispatch.post(lambda: self._progress(spec.id, event))

        def work() -> None:
            try:
                if update:
                    self.library.apply_update(spec, progress=on_progress, cancel=cancel)
                else:
                    self.library.ensure(spec, progress=on_progress, cancel=cancel)
            except JobCancelled:
                self._dispatch.post(lambda: self._done(spec.id, "cancelled", ""))
            except BaseException as exc:
                text = error_text(exc)
                self._dispatch.post(lambda: self._done(spec.id, "failed", text))
            else:
                self._dispatch.post(lambda: self._done(spec.id, "done", ""))

        threading.Thread(target=work, name=f"nf-download-{spec.id}", daemon=True).start()
        return True

    def cancel(self, model_id: str) -> None:
        event = self._cancel.get(model_id)
        if event is not None:
            event.set()

    def cancel_all(self) -> None:
        for event in self._cancel.values():
            event.set()

    def _progress(self, model_id: str, event: DownloadProgress) -> None:
        self.last[model_id] = event
        self.progress.emit(model_id, event)

    def _done(self, model_id: str, outcome: str, error: str) -> None:
        self._cancel.pop(model_id, None)
        self.last.pop(model_id, None)
        self.finished.emit(model_id, outcome, error)


class ModelRow(QWidget):
    """Actions and download progress for one model."""

    def __init__(self, dialog: ModelManager, spec: ModelSpec) -> None:
        super().__init__()
        self.dialog, self.spec = dialog, spec
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 1, 2, 1)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setFixedWidth(130)
        self.detail = QLabel("")
        self.download_button = QPushButton("Download")
        self.cancel_button = QPushButton("Cancel")
        self.update_button = QPushButton("Update")
        self.delete_button = QPushButton("Delete")
        self.link_button = QPushButton("Use a file I have...")
        self.unlink_button = QPushButton("Stop using my file")
        self.download_button.clicked.connect(lambda: dialog.download(spec.id))
        self.cancel_button.clicked.connect(lambda: dialog.controller.cancel(spec.id))
        self.update_button.clicked.connect(lambda: dialog.download(spec.id, update=True))
        self.delete_button.clicked.connect(lambda: dialog.delete(spec.id))
        self.link_button.clicked.connect(lambda: dialog.choose_local_file(spec.id))
        self.unlink_button.clicked.connect(lambda: dialog.unlink(spec.id))
        self.download_button.setToolTip("Download this model now")
        self.link_button.setToolTip("You already have this model's file (e.g. downloaded it yourself)? Use it "
                                    "instead of downloading it again")
        for w in (self.bar, self.detail, self.download_button, self.cancel_button, self.update_button,
                  self.delete_button, self.link_button, self.unlink_button):
            layout.addWidget(w)
        layout.addStretch(1)

    def show_status(self, status: ModelStatus, downloading: DownloadProgress | None, active: bool) -> None:
        local = status.state in ("linked", "linked-different")
        has_file = self.spec.weights is not None
        self.bar.setVisible(active)
        self.detail.setVisible(active)
        self.cancel_button.setVisible(active)
        self.download_button.setVisible(has_file and not active and status.state == "missing")
        self.download_button.setText("Resume download" if status.partial_bytes else "Download")
        self.link_button.setVisible(has_file and not active and status.state == "missing")
        self.unlink_button.setVisible(not active and local)
        self.delete_button.setVisible(not active and status.state == "installed")
        self.update_button.setVisible(not active and status.state == "installed" and status.update is not None
                                      and status.update.available)
        if active:
            if downloading is None:
                self.bar.setRange(0, 0)
                self.detail.setText("starting...")
            else:
                fraction = downloading.fraction
                self.bar.setRange(0, 0 if fraction is None else 1000)
                if fraction is not None:
                    self.bar.setValue(int(fraction * 1000))
                self.detail.setText(downloading.text())


class ModelManager(QDialog):
    """Non-modal window listing every model, with downloads on request."""

    changed = Signal(str)
    """A model's status changed (downloaded, deleted, linked, updated)."""
    add_model_requested = Signal()
    updates_checked = Signal(int)
    """Number of updates found (-1: the check failed)."""

    def __init__(self, registry: ModelRegistry, library: ModelLibrary, controller: DownloadController,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.registry, self.library, self.controller = registry, library, controller
        self.confirm = True
        """Ask before restricted-licence downloads and deletions (tests switch it off)."""
        self._dispatch = Dispatcher(self)
        self._rows: dict[str, tuple[QTreeWidgetItem, ModelRow]] = {}
        self._parents: dict[str, QTreeWidgetItem] = {}
        self._status: dict[str, ModelStatus] = {}
        self.setWindowTitle("Models")
        self.resize(1180, 680)
        self._build()
        controller.progress.connect(self._on_progress)
        controller.finished.connect(self._on_finished)
        self.populate()

    # ---- layout ------------------------------------------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel("Each model is made for a kind of picture: choose by what you have. Models are downloaded "
                       "only when you choose them (here, or when a job needs one). Hover over a model for details.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search: anime, VHS, photos, faces, game, 2x, a model name...")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.apply_filter)
        self.category = QComboBox()
        self.category.addItem("All kinds of content", None)
        for key, title in CATEGORY_TITLES.items():
            self.category.addItem(title, key)
        self.category.currentIndexChanged.connect(self.apply_filter)
        self.show_combo = QComboBox()
        for label, value in SHOW_CHOICES:
            self.show_combo.addItem(label, value)
        self.show_combo.currentIndexChanged.connect(self.apply_filter)
        filters.addWidget(self.search, 1)
        filters.addWidget(self.category)
        filters.addWidget(self.show_combo)
        layout.addLayout(filters)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(6)
        self.tree.setHeaderLabels(["Use it for", "Name", "Speed", "Size", "Status", ""])
        self.tree.setRootIsDecorated(True)
        self.tree.setUniformRowHeights(False)
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3, 4, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        layout.addWidget(self.tree, 1)

        footer = QHBoxLayout()
        self.usage_label = QLabel("")
        self.check_button = QPushButton("Check for updates")
        self.check_button.setToolTip("Ask the models' websites whether newer versions were published")
        self.check_button.clicked.connect(self.check_updates)
        self.add_button = QPushButton("Add your own model file...")
        self.add_button.clicked.connect(self.add_model_requested.emit)
        self.folder_button = QPushButton("Open models folder")
        self.folder_button.clicked.connect(self._open_folder)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        footer.addWidget(self.usage_label, 1)
        for b in (self.check_button, self.add_button, self.folder_button, close):
            footer.addWidget(b)
        layout.addLayout(footer)

    def populate(self) -> None:
        """(Re)build the list from the registry (call after models were added or removed)."""
        self.tree.clear()
        self._rows.clear()
        self._parents.clear()
        for title, models in self.registry.grouped():
            parent = QTreeWidgetItem([title])
            self._parents[title] = parent
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            parent.setFirstColumnSpanned(True)
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.tree.addTopLevelItem(parent)
            for m in models:
                item = QTreeWidgetItem([m.purpose or m.label, m.id, SPEED_WORDS.get(m.speed or "", "-"), "", ""])
                item.setData(0, Qt.ItemDataRole.UserRole, m.id)
                tip = _tooltip(m)
                for col in range(5):
                    item.setToolTip(col, tip)
                if m.licence_restricted:
                    item.setText(1, f"{m.id}  (personal use only)")
                parent.addChild(item)
                row = ModelRow(self, m)
                self.tree.setItemWidget(item, 5, row)
                self._rows[m.id] = (item, row)
            parent.setExpanded(True)
        self.refresh()

    def refresh(self, model_id: str | None = None) -> None:
        ids = [model_id] if model_id else list(self._rows)
        for mid in ids:
            if mid not in self._rows:
                continue
            spec = self.registry.get(mid)
            status = self.library.status(spec)
            self._status[mid] = status
            item, row = self._rows[mid]
            size = f"{status.size_bytes / 1e6:.0f} MB" if status.size_bytes else "-"
            item.setText(3, size)
            item.setText(4, _short_status(status))
            item.setToolTip(4, status.text())
            row.show_status(status, self.controller.last.get(mid), self.controller.is_active(mid))
        self.usage_label.setText(f"Models on this computer: {self.library.disk_usage() / 1e9:.2f} GB in "
                                 f"{self.library.models_dir}")
        self.apply_filter()

    def apply_filter(self) -> None:
        words = self.search.text().lower().split()
        category = self.category.currentData()
        show = self.show_combo.currentData()
        visible: dict[str, int] = {}
        for mid, (item, _row) in self._rows.items():
            spec = self.registry.get(mid)
            status = self._status.get(mid)
            hay = " ".join([spec.id, spec.purpose, spec.description, spec.category_title, f"{spec.scale}x"]).lower()
            ok = all(w in hay for w in words) and (category is None or spec.category == category)
            if status is not None and show == "ready":
                ok = ok and status.ready
            elif status is not None and show == "missing":
                ok = ok and not status.ready
            elif status is not None and show == "updates":
                ok = ok and status.update is not None and status.update.available
            item.setHidden(not ok)
            visible[spec.category_title] = visible.get(spec.category_title, 0) + ok
        for title, parent in self._parents.items():
            parent.setHidden(visible.get(title, 0) == 0)

    def focus_model(self, model_id: str) -> None:
        """Clear filters and scroll to a model's row."""
        self.search.clear()
        self.category.setCurrentIndex(0)
        self.show_combo.setCurrentIndex(0)
        if model_id in self._rows:
            item = self._rows[model_id][0]
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)

    # ---- actions -----------------------------------------------------------------------------------------

    def download(self, model_id: str, *, update: bool = False) -> bool:
        spec = self.registry.get(model_id)
        if self.confirm and spec.licence_restricted and not update:
            answer = QMessageBox.question(
                self, "Licence", f"{spec.label}\n\nThis model's licence ({spec.licence}) allows personal, "
                "non-commercial use only. Download it?")
            if answer != QMessageBox.StandardButton.Yes:
                return False
        started = self.controller.start(spec, update=update)
        self.refresh(model_id)
        return started

    def delete(self, model_id: str) -> None:
        spec = self.registry.get(model_id)
        status = self._status.get(model_id)
        size = f" ({status.size_bytes / 1e6:.0f} MB)" if status and status.size_bytes else ""
        if self.confirm:
            answer = QMessageBox.question(self, "Delete model", f"Delete {spec.label}{size} from this computer?\n"
                                          "You can download it again at any time.")
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            self.library.delete(spec)
        except NeuralFrameError as exc:
            QMessageBox.warning(self, "Delete model", error_text(exc))
        self.refresh(model_id)
        self.changed.emit(model_id)

    def choose_local_file(self, model_id: str) -> None:
        spec = self.registry.get(model_id)
        name = spec.weights.filename if spec.weights else ""
        path, _ = QFileDialog.getOpenFileName(self, f"Where is {name}?", "",
                                              "Model files (*.pth *.safetensors *.ckpt *.pt);;All files (*)")
        if path:
            self.link_file(model_id, Path(path))

    def link_file(self, model_id: str, path: Path) -> None:
        """Check the file in the background (it can take a few seconds for large models), then use it."""
        spec = self.registry.get(model_id)
        _, row = self._rows[model_id]
        row.detail.setVisible(True)
        row.detail.setText("checking the file...")

        def done(status: ModelStatus) -> None:
            self.refresh(model_id)
            self.changed.emit(model_id)
            if status.state == "linked-different" and self.confirm:
                QMessageBox.information(self, "Model file", "This file is a different version of the model than "
                                        "the one NeuralFrame knows. It loads correctly and will be used.")

        def failed(exc: BaseException) -> None:
            self.refresh(model_id)
            QMessageBox.warning(self, "Model file", error_text(exc))

        self._dispatch.run(lambda: self.library.link(spec, path), done, failed, name="nf-link")

    def unlink(self, model_id: str) -> None:
        self.library.unlink(self.registry.get(model_id))
        self.refresh(model_id)
        self.changed.emit(model_id)

    def check_updates(self) -> None:
        self.check_button.setEnabled(False)
        self.check_button.setText("Checking...")
        candidates = [m for m in self.registry.all()
                      if self.library.update_source(m) is not None and self.library.status(m).state == "installed"]

        def work() -> tuple[int, list[str]]:
            found, errors = 0, []
            for m in candidates:
                try:
                    info = self.library.check_update(m, force=True)
                except NeuralFrameError as exc:
                    errors.append(f"{m.id}: {exc.message}")
                    continue
                found += bool(info and info.available)
            return found, errors

        def done(result: tuple[int, list[str]]) -> None:
            found, errors = result
            self.check_button.setEnabled(True)
            self.check_button.setText("Check for updates")
            self.refresh()
            if not candidates:
                message = "None of the downloaded models can be checked for updates (their websites do not publish " \
                          "version information)."
            elif found:
                message = f"{found} update(s) available: use the Update button on those rows."
            else:
                message = f"All {len(candidates)} checkable downloaded models are up to date."
            if errors:
                message += "\n\nSome checks failed:\n" + "\n".join(errors[:5])
            self.updates_checked.emit(found if not errors or found else -1)
            if self.confirm:
                QMessageBox.information(self, "Model updates", message)

        def failed(exc: BaseException) -> None:
            self.check_button.setEnabled(True)
            self.check_button.setText("Check for updates")
            self.updates_checked.emit(-1)
            if self.confirm:
                QMessageBox.warning(self, "Model updates", error_text(exc))

        self._dispatch.run(work, done, failed, name="nf-update-check")

    def _open_folder(self) -> None:
        folder = self.library.models_dir
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(folder)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ---- download events ---------------------------------------------------------------------------------

    def _on_progress(self, model_id: str, event: DownloadProgress) -> None:
        if model_id in self._rows:
            status = self._status.get(model_id)
            if status is not None:
                self._rows[model_id][1].show_status(status, event, True)

    def _on_finished(self, model_id: str, outcome: str, error: str) -> None:
        self.refresh(model_id)
        self.changed.emit(model_id)
        if outcome == "failed" and self.confirm and self.isVisible():
            QMessageBox.warning(self, "Download failed", error)


def _short_status(status: ModelStatus) -> str:
    text = {"builtin": "built in", "missing": "not downloaded", "installed": "downloaded",
            "bundled": "included", "found": "found on disk", "linked": "your file",
            "linked-different": "your file (other version)"}[status.state]
    if status.state == "missing" and status.partial_bytes:
        text = "paused"
    if status.update is not None and status.update.available:
        text += " - update available"
    return text


def _tooltip(m: Any) -> str:
    lines = [m.description or m.purpose, f"Scale {m.scale}x - speed: {SPEED_WORDS.get(m.speed or '', 'n/a')} - "
             f"licence: {m.licence}"]
    if m.licence_restricted:
        lines.append("Licence allows personal / non-commercial use only.")
    if m.source_url:
        lines.append(f"Source: {m.source_url}")
    return "\n".join(lines)
