"""Model manager and model status in the desktop app (offscreen Qt; "downloads" read local file:// URLs)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import hashlib
import threading
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("torch")

from neuralframe.config import Settings
from neuralframe.errors import JobCancelled
from neuralframe.gui.window import MainWindow
from neuralframe.io.hwcaps import Capabilities
from neuralframe.models.library import DownloadProgress
from neuralframe.models.registry import ModelRegistry, ModelSpec, WeightsSpec
from tests.models import tiny_sr_model

MB = 1 << 20


@pytest.fixture
def remote(tmp_path: Path) -> ModelSpec:
    """A 3 MB model 'published' as a local file."""
    path = tmp_path / "remote" / "3x-Remote.pth"
    path.parent.mkdir()
    data = (b"neuralframe" * (3 * MB // 11 + 1))[: 3 * MB]
    path.write_bytes(data)
    return ModelSpec(id="remote-model", kind="sr", scale=2, architecture="srvgg", purpose="Remote test model",
                     category="filmed", speed="fast", licence="CC0-1.0",
                     weights=WeightsSpec(url=path.as_uri(), sha256=hashlib.sha256(data).hexdigest(),
                                         filename=path.name, size_bytes=len(data)))


@pytest.fixture
def window(qtbot: Any, settings: Settings, caps: Capabilities, tmp_path: Path, remote: ModelSpec) -> MainWindow:
    settings.inference.device = "cpu"
    settings.paths.models_dir = tmp_path / "models"
    registry = ModelRegistry([*ModelRegistry.load(user=False).all(), remote])
    w = MainWindow(settings, registry=registry, capabilities=caps)
    qtbot.addWidget(w)
    return w


def choose(combo: Any, value: Any) -> None:
    index = combo.findData(value)
    assert index >= 0, value
    combo.setCurrentIndex(index)


def test_models_are_listed_by_what_they_are_for(window: MainWindow) -> None:
    combo = window.model_combo
    texts = [combo.itemText(i) for i in range(combo.count())]
    assert texts[0].startswith("Automatic")
    assert "CARTOONS & ANIME" in texts and "OLD, DAMAGED OR VERY LOW QUALITY" in texts
    assert any(t.strip() == "HD anime to 4K, fastest (2x) - fast (2x-animejanai-hd-v3-superultracompact)  "
               "[personal use only]" for t in texts)
    heading = combo.findText("CARTOONS & ANIME")
    assert combo.itemData(heading) is None  # headings cannot be chosen

    manager = window.open_model_manager()
    manager.confirm = False
    titles = [manager.tree.topLevelItem(i).text(0) for i in range(manager.tree.topLevelItemCount())]
    assert titles[:3] == ["Generative restoration (invents detail, very slow)", "Filmed video & photos",
                          "Old, damaged or very low quality"]
    assert titles[-1] == "Face restoration"
    manager.search.setText("vhs")
    visible = [mid for mid, (item, _row) in manager._rows.items() if not item.isHidden()]
    assert visible == ["2x-vhs2hd"]
    manager.search.clear()
    manager.show_combo.setCurrentIndex(manager.show_combo.findData("ready"))
    assert all(item.isHidden() for mid, (item, _row) in manager._rows.items()
               if mid not in ("lanczos", "bicubic"))  # nothing downloaded or installed in a fresh models folder


def test_download_from_the_manager_then_delete(qtbot: Any, window: MainWindow, remote: ModelSpec) -> None:
    manager = window.open_model_manager(focus=remote.id)
    manager.confirm = False
    item, row = manager._rows[remote.id]
    assert item.text(4) == "not downloaded" and item.text(3) == "3 MB" and row.download_button.isVisibleTo(manager)
    with qtbot.waitSignal(window.downloads.finished, timeout=30_000) as blocker:
        assert manager.download(remote.id)
    assert blocker.args[:2] == [remote.id, "done"]
    assert item.text(4) == "downloaded" and row.delete_button.isVisibleTo(manager)
    assert not row.download_button.isVisibleTo(manager)
    manager.delete(remote.id)
    assert item.text(4) == "not downloaded"


def test_cancel_shows_progress_and_keeps_the_app_usable(qtbot: Any, window: MainWindow, remote: ModelSpec,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()

    def slow_ensure(spec: ModelSpec, *, progress: Any = None, cancel: Any = None) -> Path:
        progress(DownloadProgress(spec.id, MB, 3 * MB, 2_000_000.0, 1.0))
        started.set()
        while not cancel.is_set():
            cancel.wait(0.01) if hasattr(cancel, "wait") else None
        raise JobCancelled("download cancelled")

    monkeypatch.setattr(window.library, "ensure", slow_ensure)
    manager = window.open_model_manager()
    manager.confirm = False
    _item, row = manager._rows[remote.id]
    manager.download(remote.id)
    qtbot.waitUntil(lambda: row.detail.text().startswith("1.0 of 3.1 MB"), timeout=10_000)
    assert row.bar.value() == 333 and row.cancel_button.isVisibleTo(manager)
    with qtbot.waitSignal(window.downloads.finished, timeout=10_000) as blocker:
        row.cancel_button.click()
    assert blocker.args[1] == "cancelled" and not row.cancel_button.isVisibleTo(manager)


def test_the_main_window_offers_to_download_the_chosen_model(qtbot: Any, window: MainWindow,
                                                             remote: ModelSpec) -> None:
    window.show()
    choose(window.model_combo, remote.id)
    assert "not downloaded yet (3 MB)" in window.model_status.text()
    assert window.download_now_button.isVisible()
    with qtbot.waitSignal(window.downloads.finished, timeout=30_000):
        window.download_now_button.click()
    assert "on this computer and ready" in window.model_status.text()
    assert not window.download_now_button.isVisible()
    choose(window.model_combo, "lanczos")
    assert window.model_status.text() == ""  # plain resizing needs no download


def test_use_a_model_file_you_already_have(qtbot: Any, window: MainWindow, tmp_path: Path) -> None:
    spec, path = tiny_sr_model(tmp_path / "published", model_id="tiny-local")
    spec = spec.model_copy(update={"category": "filmed", "purpose": "Tiny local model"})
    window.registry = ModelRegistry([*window.registry.all(), spec])
    manager = window.open_model_manager()
    manager.registry = window.registry
    manager.populate()
    manager.confirm = False
    mine = tmp_path / "downloads" / "my copy.pth"
    mine.parent.mkdir()
    mine.write_bytes(path.read_bytes())
    with qtbot.waitSignal(manager.changed, timeout=30_000):
        manager.link_file(spec.id, mine)
    item, row = manager._rows[spec.id]
    assert item.text(4) == "your file" and row.unlink_button.isVisibleTo(manager)
    manager.unlink(spec.id)
    assert item.text(4) == "not downloaded"
