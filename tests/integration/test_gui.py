"""Desktop GUI driven through pytest-qt on the offscreen Qt platform, running real jobs on the CPU."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("torch")

from neuralframe.config import Settings
from neuralframe.gui.window import MainWindow
from neuralframe.io.hwcaps import Capabilities
from tests.conftest import ClipFactory
from tests.models import registry_with, tiny_rife_model

JOB_TIMEOUT_MS = 180_000


@pytest.fixture
def window(qtbot: Any, settings: Settings, caps: Capabilities) -> MainWindow:
    settings.inference.device = "cpu"
    settings.framegen.model = "tiny-rife"
    rife, _ = tiny_rife_model(settings.paths.models())
    w = MainWindow(settings, registry=registry_with(rife), capabilities=caps)
    qtbot.addWidget(w)
    return w


def load(qtbot: Any, w: MainWindow, path: Path) -> None:
    with qtbot.waitSignal(w.file_loaded, timeout=60_000):
        w.load_file(path)
    assert w.media is not None, w.error_label.text()


def choose(combo: Any, value: Any) -> None:
    index = combo.findData(value)
    assert index >= 0, value
    combo.setCurrentIndex(index)


def run_job(qtbot: Any, w: MainWindow, start: Any) -> str:
    with qtbot.waitSignal(w.job_finished, timeout=JOB_TIMEOUT_MS) as blocker:
        start()
    return str(blocker.args[0])


def test_defaults_before_a_file_is_chosen(window: MainWindow) -> None:
    assert window.target_combo.currentData() == "4k"
    assert window.fps_combo.currentData() == "60"
    assert window.mode == "upscale"
    assert not window.start_button.isEnabled()
    assert not window.framegen_box.isEnabled()


def test_upscale_video_with_progress_and_compare(
    qtbot: Any, window: MainWindow, clip: ClipFactory, tmp_path: Path,
) -> None:
    src = clip("gui-video", width=96, height=64, fps="24", duration=1.0)
    load(qtbot, window, src)
    assert "96x64" in window.info_label.text() and "24.000 fps" in window.info_label.text()
    assert window.output_edit.text() == str(src.with_name(f"{src.stem}_upscaled.mkv"))
    assert window.mode_framegen.isEnabled() and window.scan_button.isEnabled()
    choose(window.target_combo, "x2")
    choose(window.model_combo, "lanczos")
    window.output_edit.setText(str(tmp_path / "gui.mkv"))
    assert run_job(qtbot, window, lambda: window.start(confirm_overwrite=False)) == "done"
    assert window.progress.value() == 1000 and "24 / 24 frames" in window.stats_label.text()
    assert "Done: gui.mkv (192x128, 24 frames)" in window.status_label.text()
    assert window.open_result_button.isEnabled() and window.show_folder_button.isEnabled()
    assert (tmp_path / "gui.mkv").is_file()
    qtbot.waitUntil(lambda: window.compare.has_images, timeout=60_000)


def test_images_can_only_be_upscaled(qtbot: Any, window: MainWindow, tmp_path: Path, ffhelpers: Any) -> None:
    image = tmp_path / "photo.png"
    ffhelpers.text(["-f", "lavfi", "-i", "testsrc2=size=96x64:duration=0.04", "-frames:v", "1", "-update", "1",
                    "-y", str(image)])
    load(qtbot, window, image)
    assert not window.mode_framegen.isEnabled() and not window.mode_both.isEnabled()
    assert not window.scan_button.isEnabled()
    assert window.output_edit.text().endswith("photo_upscaled.png")
    choose(window.target_combo, "x2")
    choose(window.model_combo, "lanczos")
    assert run_job(qtbot, window, lambda: window.start(confirm_overwrite=False)) == "done"
    assert (tmp_path / "photo_upscaled.png").is_file()


def test_scan_then_frame_generation(qtbot: Any, window: MainWindow, clip: ClipFactory, tmp_path: Path) -> None:
    src = clip("gui-dup", width=96, height=64, fps="25", duration=2.0, duplicate_frames=(20,), crf=0)
    load(qtbot, window, src)
    with qtbot.waitSignal(window.scan_finished, timeout=60_000):
        window.scan_button.click()
    assert "1 frozen frame(s)" in window.scan_label.text(), window.scan_label.text()

    window.mode_framegen.setChecked(True)
    assert window.framegen_box.isEnabled() and not window.upscale_box.isEnabled()
    choose(window.fps_combo, "custom")
    window.custom_fps.setText("50")
    assert window.output_edit.text().endswith(f"{src.stem}_50fps.mkv")
    window.output_edit.setText(str(tmp_path / "fg.mkv"))
    assert run_job(qtbot, window, lambda: window.start(confirm_overwrite=False)) == "done"
    text = window.status_label.text()
    assert "100 frames" in text and "1 frozen frame(s) repaired" in text, text


def test_cancel_and_resume(qtbot: Any, window: MainWindow, clip: ClipFactory, tmp_path: Path) -> None:
    window.settings.chunking.chunk_frames = 24
    src = clip("gui-long", width=96, height=64, fps="24", duration=3.0)
    load(qtbot, window, src)
    choose(window.target_combo, "x2")
    choose(window.model_combo, "lanczos")
    window.output_edit.setText(str(tmp_path / "long.mkv"))

    def start_and_stop_after_first_chunk() -> None:
        window.start(confirm_overwrite=False)
        runner = window._runner
        assert runner is not None
        runner.on_chunk_done = lambda chunk, _total: chunk == 0 and runner.cancel.cancel("test")

    assert run_job(qtbot, window, start_and_stop_after_first_chunk) == "cancelled"
    assert window.resume_button.isEnabled() and "Resume" in window.status_label.text()
    assert run_job(qtbot, window, window.resume_button.click) == "done"
    assert window.result is not None
    assert window.result.report["throughput"]["frames_resumed"] >= 24
    assert window.result.report["output"]["frames"] == 72


def test_invalid_custom_values_are_reported(qtbot: Any, window: MainWindow, clip: ClipFactory) -> None:
    src = clip("gui-video", width=96, height=64, fps="24", duration=1.0)
    load(qtbot, window, src)
    choose(window.target_combo, "custom")
    window.custom_target.setText("huge")
    window.start(confirm_overwrite=False)
    assert window.error_label.text() and window._job_thread is None
    choose(window.target_combo, "4k")
    window.mode_framegen.setChecked(True)
    choose(window.fps_combo, "custom")
    window.custom_fps.setText("0")
    window.start(confirm_overwrite=False)
    assert "frame rate" in window.error_label.text() and window._job_thread is None


def test_automatic_model_choice_is_shown(qtbot: Any, window: MainWindow, clip: ClipFactory) -> None:
    assert window.model_combo.currentData() == "auto"
    src = clip("gui-auto", width=160, height=90, fps="24", duration=1.0)
    with qtbot.waitSignal(window.recommendation_ready, timeout=60_000):
        load(qtbot, window, src)
    assert window.profile is not None and window.recommended is not None
    assert window.registry.get(window.recommended).speed == "fast"  # the test window runs on the CPU
    assert f"({window.recommended})</b>" in window.auto_label.text()  # purpose - speed (id)
    choose(window.model_combo, "lanczos")
    assert window.auto_label.text() == ""
    choose(window.model_combo, "auto")
    assert window.recommended in window.auto_label.text()
    assert window.build_spec().model == "auto"


def test_add_your_own_model(qtbot: Any, window: MainWindow, tmp_path: Path) -> None:
    pytest.importorskip("spandrel")
    from tests.models import tiny_swinir_file

    file = tiny_swinir_file(tmp_path, "2x-gui-test.safetensors")
    with qtbot.waitSignal(window.model_added, timeout=60_000) as blocker:
        window.add_model(file)
    assert blocker.args == ["2x-gui-test"], window.error_label.text()
    assert window.model_combo.currentData() == "2x-gui-test"
    assert "added" in window.model_combo.currentText()
    with qtbot.waitSignal(window.model_added, timeout=60_000) as blocker:
        window.add_model(file)  # the same name again
    assert blocker.args == [""] and "already exists" in window.error_label.text()
