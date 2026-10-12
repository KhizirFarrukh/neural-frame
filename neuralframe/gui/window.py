"""NeuralFrame desktop window: pick a file, choose what to do, watch it run, compare the result."""

from __future__ import annotations

import functools
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from neuralframe import __version__
from neuralframe.config import Settings, builtin_presets
from neuralframe.core.job import JobResult, JobRunner, JobSpec, ProgressEvent
from neuralframe.core.naming import suggest_output
from neuralframe.core.planning import TargetSpec
from neuralframe.core.preview import colour_match_filter, frame_png, frame_time
from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.framegen.retime import parse_fps
from neuralframe.gui.compare import CompareView
from neuralframe.gui.threads import Dispatcher
from neuralframe.io.ffmpeg import FFmpegTools, locate_ffmpeg
from neuralframe.io.media import MediaInfo
from neuralframe.io.probe import probe
from neuralframe.models.registry import ModelRegistry

TARGETS = [("4K (2160p)", "4k"), ("8K (4320p)", "8k"), ("1440p", "1440p"), ("1080p", "1080p"), ("720p", "720p"),
           ("2x", "x2"), ("3x", "x3"), ("4x", "x4"), ("Custom...", "custom")]
FPS_CHOICES = [("24 fps", "24"), ("30 fps", "30"), ("48 fps", "48"), ("60 fps", "60"), ("120 fps", "120"),
               ("Custom...", "custom")]
REPAIR = [("Smart (keep intentional holds)", "auto"), ("Repair every repeated frame", "on"), ("Off", "off")]
VIDEO_FILTER = "Videos and images (*.mp4 *.mkv *.mov *.webm *.avi *.m4v *.ts *.mts *.wmv *.flv *.png *.jpg " \
               "*.jpeg *.webp *.bmp *.tif *.tiff);;All files (*)"


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = round(seconds)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class MainWindow(QMainWindow):
    # Emitted for observers (and tests) when background steps finish.
    file_loaded = Signal()
    scan_finished = Signal()
    job_finished = Signal(str)
    preview_loaded = Signal()

    def __init__(self, settings: Settings, *, registry: ModelRegistry | None = None,
                 capabilities: Any = None) -> None:
        super().__init__()
        self.settings = settings
        self.registry = registry or ModelRegistry.load()
        self._caps = capabilities
        self._dispatch = Dispatcher(self)
        self._tools: FFmpegTools | None = None
        self.input_path: Path | None = None
        self.media: MediaInfo | None = None
        self._runner: JobRunner | None = None
        self._job_thread: threading.Thread | None = None
        self._last_spec: JobSpec | None = None
        self.result: JobResult | None = None
        self._output_edited = False

        self.setWindowTitle("NeuralFrame")
        self.setAcceptDrops(True)
        self.resize(980, 860)
        self._build()
        self._set_media_controls(False)
        self._update_mode()
        self._load_system_info()

    # ---- layout ----------------------------------------------------------------------------------------

    def _build(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setSpacing(12)

        # 1. file
        file_box = QGroupBox("1. File")
        fl = QVBoxLayout(file_box)
        self.drop_area = QLabel("Drop a video or image here, or")
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setFrameShape(QFrame.Shape.StyledPanel)
        self.drop_area.setMinimumHeight(60)
        self.open_button = QPushButton("Open file...")
        self.open_button.clicked.connect(self._choose_file)
        row = QHBoxLayout()
        row.addWidget(self.drop_area, 1)
        row.addWidget(self.open_button)
        fl.addLayout(row)
        info_row = QHBoxLayout()
        self.thumbnail = QLabel()
        self.thumbnail.setFixedSize(192, 108)
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setFrameShape(QFrame.Shape.StyledPanel)
        self.info_label = QLabel("No file selected.")
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info_col = QVBoxLayout()
        info_col.addWidget(self.info_label)
        scan_row = QHBoxLayout()
        self.scan_button = QPushButton("Scan for dropped frames")
        self.scan_button.setToolTip("Find frozen (repeated) frames and timestamp gaps that frame generation repairs")
        self.scan_button.clicked.connect(self.scan_drops)
        self.scan_label = QLabel("")
        self.scan_label.setWordWrap(True)
        scan_row.addWidget(self.scan_button)
        scan_row.addWidget(self.scan_label, 1)
        info_col.addLayout(scan_row)
        info_col.addStretch(1)
        info_row.addWidget(self.thumbnail)
        info_row.addLayout(info_col, 1)
        fl.addLayout(info_row)
        layout.addWidget(file_box)

        # 2. what to do
        task_box = QGroupBox("2. What to do")
        tl = QVBoxLayout(task_box)
        mode_row = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_upscale = QRadioButton("Upscale")
        self.mode_framegen = QRadioButton("Frame generation")
        self.mode_both = QRadioButton("Both")
        for i, button in enumerate((self.mode_upscale, self.mode_framegen, self.mode_both)):
            self.mode_group.addButton(button, i)
            mode_row.addWidget(button)
        self.mode_upscale.setChecked(True)
        self.mode_group.idToggled.connect(self._mode_toggled)
        mode_row.addStretch(1)
        self.mode_note = QLabel("")
        mode_row.addWidget(self.mode_note)
        tl.addLayout(mode_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        self.upscale_box = QGroupBox("Upscale")
        uf = QFormLayout(self.upscale_box)
        self.target_combo = QComboBox()
        for label, value in TARGETS:
            self.target_combo.addItem(label, value)
        self.target_combo.currentIndexChanged.connect(self._target_changed)
        self.custom_target = QLineEdit()
        self.custom_target.setPlaceholderText("e.g. 2560x1440, x1.5 or 1200p")
        self.custom_target.setVisible(False)
        self.custom_target.textChanged.connect(self._refresh_output)
        self.model_combo = QComboBox()
        self.model_combo.addItem("Automatic (from preset)", None)
        for m in self.registry.all():
            if m.kind == "interpolation":
                continue
            label = f"{m.id}  (x{m.scale}{', temporal' if m.is_temporal else ''})"
            self.model_combo.addItem(label, m.id)
            self.model_combo.setItemData(self.model_combo.count() - 1,
                                         f"{m.description}\nlicence: {m.licence}", Qt.ItemDataRole.ToolTipRole)
        self.preset_combo = QComboBox()
        for name in sorted({*builtin_presets(), *self.settings.presets}):
            self.preset_combo.addItem(name, name)
        self.preset_combo.setCurrentText("balanced")
        uf.addRow("Output size", self.target_combo)
        uf.addRow("", self.custom_target)
        uf.addRow("Model", self.model_combo)
        uf.addRow("Preset", self.preset_combo)

        self.framegen_box = QGroupBox("Frame generation")
        gf = QFormLayout(self.framegen_box)
        self.fps_combo = QComboBox()
        for label, value in FPS_CHOICES:
            self.fps_combo.addItem(label, value)
        default_fps = self.settings.framegen.fps
        idx = self.fps_combo.findData(default_fps)
        self.fps_combo.setCurrentIndex(idx if idx >= 0 else self.fps_combo.findData("60"))
        self.fps_combo.currentIndexChanged.connect(self._fps_changed)
        self.custom_fps = QLineEdit()
        self.custom_fps.setPlaceholderText("e.g. 50, 59.94 or 24000/1001")
        self.custom_fps.setVisible(False)
        self.custom_fps.textChanged.connect(self._refresh_output)
        self.repair_combo = QComboBox()
        for label, value in REPAIR:
            self.repair_combo.addItem(label, value)
        gf.addRow("Frame rate", self.fps_combo)
        gf.addRow("", self.custom_fps)
        gf.addRow("Dropped frames", self.repair_combo)
        grid.addWidget(self.upscale_box, 0, 0)
        grid.addWidget(self.framegen_box, 0, 1)
        tl.addLayout(grid)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Save as"))
        self.output_edit = QLineEdit()
        self.output_edit.setMinimumWidth(200)
        self.output_edit.textEdited.connect(self._output_typed)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._choose_output)
        out_row.addWidget(self.output_edit, 1)
        out_row.addWidget(self.browse_button)
        tl.addLayout(out_row)
        layout.addWidget(task_box)

        # 3. run
        run_box = QGroupBox("3. Run")
        rl = QVBoxLayout(run_box)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.resume_button = QPushButton("Resume")
        self.resume_button.setToolTip("Continue the stopped job from its last finished chunk")
        self.resume_button.clicked.connect(self.resume)
        self.open_result_button = QPushButton("Open result")
        self.open_result_button.clicked.connect(self._open_result)
        self.show_folder_button = QPushButton("Show in folder")
        self.show_folder_button.clicked.connect(self._show_in_folder)
        for b in (self.start_button, self.cancel_button, self.resume_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.open_result_button)
        buttons.addWidget(self.show_folder_button)
        rl.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFormat("%p%")
        rl.addWidget(self.progress)
        self.status_label = QLabel("Ready.")
        self.stats_label = QLabel("")
        self.stages_label = QLabel("")
        self.stages_label.setWordWrap(True)
        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #d04848;")
        self.error_label.setWordWrap(True)
        self.error_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        for w in (self.status_label, self.stats_label, self.stages_label, self.error_label):
            w.setWordWrap(True)  # long lines must not widen the window
            rl.addWidget(w)
        layout.addWidget(run_box)

        # 4. compare
        self.compare_box = QGroupBox("Before / after")
        cl = QVBoxLayout(self.compare_box)
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Frame"))
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 10_000_000)
        frame_row.addWidget(self.frame_spin)
        self.preview_button = QPushButton("Show")
        self.preview_button.clicked.connect(self.load_preview)
        frame_row.addWidget(self.preview_button)
        hint = QLabel("drag across the picture: left = source (bicubic), right = NeuralFrame")
        hint.setWordWrap(True)
        frame_row.addWidget(hint, 1)
        cl.addLayout(frame_row)
        self.compare = CompareView()
        cl.addWidget(self.compare, 1)
        layout.addWidget(self.compare_box, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(root)
        self.setCentralWidget(scroll)
        self.statusBar().showMessage(f"NeuralFrame {__version__}")
        self._set_running(False)

    # ---- helpers ---------------------------------------------------------------------------------------

    def tools(self) -> FFmpegTools:
        if self._tools is None:
            self._tools = locate_ffmpeg(self.settings.ffmpeg.ffmpeg, self.settings.ffmpeg.ffprobe)
        return self._tools

    def _load_system_info(self) -> None:
        def work() -> str:
            from neuralframe.io.hwcaps import detect_capabilities

            caps = self._caps or detect_capabilities(self.tools(), cache_dir=self.settings.paths.cache_dir)
            self._caps = caps
            sel = caps.selected()
            gpu = ", ".join(g.name for g in caps.gpus) or "no NVIDIA GPU (CPU)"
            return (f"NeuralFrame {__version__}  |  {gpu}  |  inference: {sel['inference_backend']}  |  "
                    f"encoder: {sel['encoder_hevc'] or sel['encoder_h264'] or 'software'}")

        self._dispatch.run(work, lambda text: self.statusBar().showMessage(text),
                           lambda exc: self.statusBar().showMessage(f"system check failed: {exc}"))

    def _set_media_controls(self, enabled: bool) -> None:
        for w in (self.start_button, self.output_edit, self.browse_button):
            w.setEnabled(enabled)
        is_video = enabled and self.media is not None and not self.media.is_image
        self.scan_button.setEnabled(is_video)

    def _set_running(self, running: bool) -> None:
        loaded = self.media is not None
        self.start_button.setEnabled(not running and loaded)
        self.cancel_button.setEnabled(running)
        self.open_button.setEnabled(not running)
        self.resume_button.setEnabled(False)
        done = self.result is not None and not running
        self.open_result_button.setEnabled(done)
        self.show_folder_button.setEnabled(done)
        self.preview_button.setEnabled(done)

    @property
    def mode(self) -> str:
        return {0: "upscale", 1: "framegen", 2: "both"}[self.mode_group.checkedId()]

    def _mode_toggled(self, _id: int, checked: bool) -> None:
        if checked:
            self._update_mode()

    def _update_mode(self) -> None:
        image = self.media is not None and self.media.is_image
        if image and self.mode != "upscale":
            self.mode_upscale.setChecked(True)
        self.mode_framegen.setEnabled(not image)
        self.mode_both.setEnabled(not image)
        self.mode_note.setText("images can only be upscaled" if image else "")
        self.upscale_box.setEnabled(self.mode != "framegen")
        self.framegen_box.setEnabled(self.mode != "upscale")
        self._refresh_output()

    def _target_changed(self) -> None:
        self.custom_target.setVisible(self.target_combo.currentData() == "custom")
        self._refresh_output()

    def _fps_changed(self) -> None:
        self.custom_fps.setVisible(self.fps_combo.currentData() == "custom")
        self._refresh_output()

    def target_spec(self) -> TargetSpec:
        value = self.target_combo.currentData()
        if value == "custom":
            value = self.custom_target.text().strip()
        return TargetSpec.parse(str(value))

    def target_fps(self) -> str:
        value = str(self.fps_combo.currentData())
        if value == "custom":
            value = self.custom_fps.text().strip()
        parse_fps(value)  # validates
        return value

    def _output_typed(self, _text: str) -> None:
        self._output_edited = True

    def _refresh_output(self) -> None:
        if self.input_path is None or self._output_edited:
            return
        fps = None
        if self.mode != "upscale":
            try:
                fps = self.target_fps()
            except ValueError:
                fps = None
        out = suggest_output(self.input_path, upscale=self.mode != "framegen", interpolate=self.mode != "upscale",
                             target_fps=fps)
        self.output_edit.setText(str(out))

    def _show_error(self, exc: BaseException) -> None:
        if isinstance(exc, NeuralFrameError):
            text = f"{exc.message}" + (f"\nHint: {exc.hint}" if exc.hint else "")
        else:
            text = str(exc)
        self.error_label.setText(text)

    # ---- file selection --------------------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and self._job_thread is None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = [u for u in event.mimeData().urls() if u.isLocalFile()]
        if urls:
            self.load_file(Path(urls[0].toLocalFile()))

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a video or image", "", VIDEO_FILTER)
        if path:
            self.load_file(Path(path))

    def _choose_output(self) -> None:
        current = self.output_edit.text() or (str(self.input_path.parent) if self.input_path else "")
        image = self.media is not None and self.media.is_image
        filters = ("Images (*.png *.jpg *.jpeg *.webp *.tif *.tiff *.bmp)" if image
                   else "Videos (*.mkv *.mp4 *.mov *.webm)")
        path, _ = QFileDialog.getSaveFileName(self, "Save result as", current, filters)
        if path:
            self.output_edit.setText(path)
            self._output_edited = True

    def load_file(self, path: Path) -> None:
        """Probe ``path`` in the background, then show its summary and a thumbnail."""
        self.error_label.clear()
        self.scan_label.clear()
        self.info_label.setText(f"Reading {path.name}...")
        self.thumbnail.clear()
        self.result = None
        self.compare.clear()
        self._last_spec = None

        def work() -> tuple[MediaInfo, bytes | None]:
            media = probe(path, self.tools())
            thumb = None
            try:
                at = 0.0 if media.is_image else min(5.0, (media.duration or 0.0) * 0.1)
                thumb = frame_png(self.tools(), path, at, None)
            except NeuralFrameError:
                pass
            return media, thumb

        def done(result: tuple[MediaInfo, bytes | None]) -> None:
            media, thumb = result
            self.input_path, self.media = path, media
            self._output_edited = False
            video = media.require_video()
            parts = [f"<b>{path.name}</b>", f"{video.display_size[0]}x{video.display_size[1]}"]
            if media.is_image:
                parts.append("image")
            else:
                parts.append(f"{float(video.fps):.3f} fps")
                if media.duration:
                    parts.append(_fmt_time(media.duration))
                parts.append(video.codec)
                if media.audio:
                    parts.append(f"audio: {', '.join(a.codec for a in media.audio)}")
                if video.hdr_format:
                    parts.append(f"HDR ({video.hdr_format})")
            self.info_label.setText("  -  ".join(parts))
            if thumb:
                pix = QPixmap()
                pix.loadFromData(thumb)
                self.thumbnail.setPixmap(pix.scaled(self.thumbnail.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                    Qt.TransformationMode.SmoothTransformation))
            self.frame_spin.setValue(0)
            self._set_media_controls(True)
            self._update_mode()
            self._set_running(False)
            self.status_label.setText("Ready.")
            self.file_loaded.emit()

        def failed(exc: BaseException) -> None:
            self.input_path = self.media = None
            self.info_label.setText("No file selected.")
            self._set_media_controls(False)
            self._set_running(False)
            self._show_error(exc)
            self.file_loaded.emit()

        self._dispatch.run(work, done, failed, name="nf-probe")

    def scan_drops(self) -> None:
        if self.input_path is None:
            return
        path = self.input_path
        self.scan_button.setEnabled(False)
        self.scan_label.setText("scanning...")

        def done(info: dict[str, Any]) -> None:
            self.scan_button.setEnabled(True)
            frozen, gaps, missing = info["duplicated_frames"], info["timestamp_gaps"], info["frames_missing_in_gaps"]
            holds, cuts = info["cadence_holds_kept"], len(info["scene_cuts"])
            if frozen or gaps:
                text = (f"{frozen} frozen frame(s) and {missing} missing frame(s) in {gaps} gap(s) found; "
                        "frame generation repairs them")
            else:
                text = "no dropped frames found"
            extras = [f"{cuts} scene cut(s)"]
            if holds:
                extras.append(f"{holds} intentional hold(s) kept")
            self.scan_label.setText(f"{text} ({', '.join(extras)}, {info['frames']} frames)")
            self.scan_finished.emit()

        def failed(exc: BaseException) -> None:
            self.scan_button.setEnabled(True)
            self.scan_label.setText("")
            self._show_error(exc)
            self.scan_finished.emit()

        runner = JobRunner(self.settings, registry=self.registry, capabilities=self._caps)
        self._dispatch.run(lambda: runner.scan_drops(path), done, failed, name="nf-scan")

    # ---- running ---------------------------------------------------------------------------------------

    def build_spec(self, *, resume: bool = False, overwrite: bool = False) -> JobSpec:
        assert self.input_path is not None
        upscale = self.mode != "framegen"
        interpolate = self.mode != "upscale"
        output = Path(self.output_edit.text().strip()).expanduser()
        if not output.name:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, "choose where to save the result")
        return JobSpec(
            input=self.input_path,
            output=output,
            target=self.target_spec() if upscale else TargetSpec.default(),
            model=self.model_combo.currentData() if upscale else None,
            preset=self.preset_combo.currentData() or "balanced",
            upscale=upscale,
            interpolate=interpolate,
            target_fps=self.target_fps() if interpolate else None,
            repair_drops=self.repair_combo.currentData() if interpolate else None,
            resume=resume,
            overwrite=overwrite or resume,
        )

    def start(self, *, confirm_overwrite: bool = True) -> None:
        self.error_label.clear()
        try:
            spec = self.build_spec()
        except (ValueError, NeuralFrameError) as exc:
            self._show_error(exc)
            return
        if spec.output.exists():
            if confirm_overwrite:
                answer = QMessageBox.question(self, "Replace file?", f"{spec.output} exists. Replace it?")
                if answer != QMessageBox.StandardButton.Yes:
                    return
            spec = spec.model_copy(update={"overwrite": True})
        self._run(spec)

    def resume(self) -> None:
        if self._last_spec is None:
            return
        self.error_label.clear()
        self._run(self._last_spec.model_copy(update={"resume": True, "overwrite": True}))

    def cancel(self) -> None:
        if self._runner is not None:
            self.status_label.setText("Cancelling (finishing the current frames)...")
            self.cancel_button.setEnabled(False)
            self._runner.cancel.cancel("cancelled by user")

    def _run(self, spec: JobSpec) -> None:
        self._last_spec = spec
        self.result = None
        self.compare.clear()
        self.progress.setValue(0)
        self.stats_label.clear()
        self.stages_label.clear()
        self.status_label.setText("Starting...")
        runner = JobRunner(self.settings, registry=self.registry, capabilities=self._caps,
                           progress=lambda event: self._dispatch.post(lambda: self._on_progress(event)))
        self._runner = runner

        def work() -> None:
            try:
                result = runner.run(spec)
            except JobCancelled:
                self._dispatch.post(lambda: self._finish("cancelled"))
            except BaseException as exc:
                self._dispatch.post(functools.partial(self._finish, "failed", exc))
            else:
                self._dispatch.post(lambda: self._finish("done", result=result))

        self._set_running(True)
        self._job_thread = threading.Thread(target=work, name="nf-job", daemon=True)
        self._job_thread.start()

    def _on_progress(self, event: ProgressEvent) -> None:
        if self._runner is None or event.job_id != self._runner.job_id:
            return
        self.progress.setValue(int(event.percent * 10))
        labels = {"probing": "Reading the input", "planning": "Planning", "running": "Processing",
                  "muxing": "Writing the output file", "verifying": "Checking the output"}
        text = labels.get(event.state, event.state.capitalize())
        self.status_label.setText(f"{text}{': ' + event.message if event.message else ''}")
        stats = [f"{event.frames_done} / {event.frames_total} frames", f"{event.fps:.1f} fps",
                 f"ETA {_fmt_time(event.eta_s)}", f"elapsed {_fmt_time(event.elapsed_s)}"]
        if event.gpu:
            g = next(iter(event.gpu.values()))
            stats.append(f"VRAM {g['used_mb']} / {g['total_mb']} MB, GPU {g['util_pct']:.0f}%")
        self.stats_label.setText("   ".join(stats))
        if event.stages:
            self.stages_label.setText("  -  ".join(f"{s.name} {s.busy_fps:.1f} fps" for s in event.stages))

    def _finish(self, state: str, exc: BaseException | None = None, result: JobResult | None = None) -> None:
        if self._job_thread is not None:
            self._job_thread.join(5)
        self._job_thread = None
        self._runner = None
        self.result = result
        self._set_running(False)
        if state == "done" and result is not None:
            self.progress.setValue(1000)
            report = result.report
            summary = f"Done: {result.output.name} ({report['output'].get('size', '')}"
            if report.get("kind") != "image":
                summary += f", {report['output'].get('frames')} frames"
            summary += f") in {_fmt_time(report['timings'].get('total_s'))}"
            fg = report.get("framegen")
            if fg:
                runtime = fg.get("runtime") or {}
                summary += (f". Frame generation: {runtime.get('interpolated', 0)} frames synthesised, "
                            f"{fg['drops']['duplicated_frames']} frozen frame(s) repaired")
            self.status_label.setText(summary)
            self.load_preview()
        elif state == "cancelled":
            self.status_label.setText("Cancelled. Finished chunks are kept: Resume continues where it stopped.")
            self.resume_button.setEnabled(True)
        else:
            self.status_label.setText("Failed.")
            if exc is not None:
                self._show_error(exc)
            self.resume_button.setEnabled(self._last_spec is not None)
        self.job_finished.emit(state)

    # ---- results ---------------------------------------------------------------------------------------

    def load_preview(self) -> None:
        if self.result is None or self.input_path is None or self.media is None:
            return
        source, output, media = self.input_path, self.result.output, self.media
        report, frame = self.result.report, self.frame_spin.value()

        def work() -> tuple[bytes, bytes]:
            ff = self.tools()
            out_video = probe(output, ff).require_video()
            seconds = frame_time(media, frame)
            before = frame_png(ff, source, seconds, (out_video.width, out_video.height), colour_match_filter(report))
            return before, frame_png(ff, output, seconds)

        def done(images: tuple[bytes, bytes]) -> None:
            before, after = QPixmap(), QPixmap()
            before.loadFromData(images[0])
            after.loadFromData(images[1])
            self.compare.set_images(before, after)
            self.preview_loaded.emit()

        def failed(exc: BaseException) -> None:
            self._show_error(exc)
            self.preview_loaded.emit()

        self._dispatch.run(work, done, failed, name="nf-preview")

    def _open_result(self) -> None:
        if self.result is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.result.output)))

    def _show_in_folder(self) -> None:
        if self.result is None:
            return
        path = self.result.output
        if sys.platform == "win32":
            subprocess.Popen(["explorer", f"/select,{path}"])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._runner is not None and self._job_thread is not None:
            answer = QMessageBox.question(
                self, "Job running", "A job is running. Stop it and quit? Finished chunks are kept for resuming."
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._runner.cancel.cancel("window closed")
            self._job_thread.join(60)
        event.accept()
