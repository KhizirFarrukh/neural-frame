"""NeuralFrame desktop window: pick a file, choose what to do, watch it run, compare the result."""

from __future__ import annotations

import functools
import html
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QPixmap,
    QStandardItemModel,
)
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
from neuralframe.models.library import DownloadProgress, ModelLibrary
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
    recommendation_ready = Signal()
    model_added = Signal(str)
    model_status_updated = Signal()

    def __init__(self, settings: Settings, *, registry: ModelRegistry | None = None,
                 capabilities: Any = None) -> None:
        super().__init__()
        self.settings = settings
        self.registry = registry or ModelRegistry.load()
        self._caps = capabilities
        self._dispatch = Dispatcher(self)
        from neuralframe.gui.models_dialog import DownloadController

        self.library = ModelLibrary.from_settings(settings)
        self.downloads = DownloadController(self.library, self, settings=settings)
        self.downloads.progress.connect(lambda mid, _event: self._update_model_status())
        self.downloads.finished.connect(self._download_finished)
        self.models_window: Any = None
        self._tools: FFmpegTools | None = None
        self.input_path: Path | None = None
        self.media: MediaInfo | None = None
        self._runner: JobRunner | None = None
        self._job_thread: threading.Thread | None = None
        self._last_spec: JobSpec | None = None
        self.result: JobResult | None = None
        self._output_edited = False
        self.profile: Any = None
        """Content analysis of the loaded file (automatic model choice)."""
        self.recommended: str | None = None

        self.setWindowTitle("NeuralFrame")
        self.setAcceptDrops(True)
        self.resize(980, 860)
        self._build()
        self._build_menus()
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
        self._fill_models()
        self.model_combo.currentIndexChanged.connect(self._update_recommendation)
        self.model_combo.setToolTip("Models are listed by what they are for; the name in brackets is the "
                                    "model's own name. Automatic picks one for your file.")
        self.models_button = QPushButton("Models...")
        self.models_button.setToolTip("See all models, download them ahead of time, use files you already have, "
                                      "check for updates")
        self.models_button.clicked.connect(lambda: self.open_model_manager())
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.models_button)
        self.model_status = QLabel("")
        self.model_status.setWordWrap(True)
        self.download_now_button = QPushButton("Download now")
        self.download_now_button.clicked.connect(self._download_selected)
        self.cancel_download_button = QPushButton("Cancel download")
        self.cancel_download_button.clicked.connect(self._cancel_selected_download)
        status_row = QHBoxLayout()
        status_row.addWidget(self.model_status, 1)
        status_row.addWidget(self.download_now_button)
        status_row.addWidget(self.cancel_download_button)
        self.auto_label = QLabel("")
        self.auto_label.setWordWrap(True)
        self.auto_label.setTextFormat(Qt.TextFormat.RichText)
        self.preset_combo = QComboBox()
        for name in sorted({*builtin_presets(), *self.settings.presets}):
            self.preset_combo.addItem(name, name)
        self.preset_combo.setCurrentText("balanced")
        self.preset_combo.setToolTip("fast: fast models only - balanced: up to slow models - quality: also the "
                                     "very slow transformer models with the most detail")
        self.preset_combo.currentIndexChanged.connect(self._update_recommendation)
        self.custom_target.textChanged.connect(self._update_recommendation)
        uf.addRow("Output size", self.target_combo)
        uf.addRow("", self.custom_target)
        uf.addRow("Model", model_row)
        uf.addRow("", self.auto_label)
        uf.addRow("", status_row)
        uf.addRow("Preset", self.preset_combo)
        self.faces_combo = QComboBox()
        self.faces_combo.addItem("Off", None)
        for m in self.registry.of_kind("faces"):
            self.faces_combo.addItem(m.label + ("  [personal use only]" if m.licence_restricted else ""), m.id)
            self.faces_combo.setItemData(self.faces_combo.count() - 1, m.description, Qt.ItemDataRole.ToolTipRole)
        self.faces_combo.setToolTip("Detects faces in the result, restores them and pastes them back (useful for "
                                    "small or blurry faces). Installed the first time you use it.")
        uf.addRow("Restore faces", self.faces_combo)

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

    def _build_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        open_action = QAction("&Open a video or picture...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._choose_file)
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        models_menu = menu.addMenu("&Models")
        manage = QAction("&Model manager...", self)
        manage.setShortcut(QKeySequence("Ctrl+M"))
        manage.setToolTip("All models: download, delete, use files you have, updates")
        manage.triggered.connect(lambda: self.open_model_manager())
        updates = QAction("Check for model &updates", self)
        updates.triggered.connect(self._check_model_updates)
        add = QAction("&Add your own model file...", self)
        add.triggered.connect(self._choose_model_file)
        token = QAction("Hugging Face &token...", self)
        token.setToolTip("Some engines download gated weights (FLUX.1-dev for LucidFlux) and need your token")
        token.triggered.connect(self._set_hf_token)
        for action in (manage, updates, add, token):
            models_menu.addAction(action)
        help_menu = menu.addMenu("&Help")
        how = QAction("&How to use NeuralFrame", self)
        how.setShortcut(QKeySequence.StandardKey.HelpContents)
        how.triggered.connect(self._show_help)
        about = QAction("&About NeuralFrame", self)
        about.triggered.connect(lambda: QMessageBox.about(
            self, "About NeuralFrame",
            f"NeuralFrame {__version__}\n\nMakes videos and pictures sharper and larger with AI, and videos "
            "smoother with frame generation. Everything runs on this computer.\n\nModel licences: see the Models "
            "window (hover over a model)."))
        help_menu.addAction(how)
        help_menu.addAction(about)

    def _set_hf_token(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        from neuralframe.engines.secrets import hf_token, save_hf_token

        current = hf_token(self.settings) or ""
        text, ok = QInputDialog.getText(
            self, "Hugging Face token", "Some engines download weights that need a Hugging Face account (accept the "
            "model's licence on its page, then create a token in Settings > Access Tokens).\nPaste the token "
            "(leave empty to remove it):", QLineEdit.EchoMode.Password, current)
        if ok:
            save_hf_token(text.strip() or None)
            self.library.hf_token = text.strip() or None
            self.statusBar().showMessage("Hugging Face token saved." if text.strip() else "Token removed.", 6000)

    def _show_help(self) -> None:
        QMessageBox.information(self, "How to use NeuralFrame", (
            "1. Open a video or picture (or drop it on the window).\n"
            "2. Choose Upscale, Frame generation or Both.\n"
            "3. Model: leave it on Automatic, or pick what matches your content - the list says what each model "
            "is for (cartoons, old VHS tapes, phone videos, photos, games...).\n"
            "4. Press Start. If the model is not on this computer yet, it is downloaded first (you see the "
            "progress), then the work starts.\n\n"
            "Models menu: see every model, download some ahead of time, use model files you already have, "
            "delete models to free space, check for updates.\n\n"
            "Presets: fast = quick models only, balanced = the default, quality = the slowest, most detailed "
            "models."))

    # ---- models ------------------------------------------------------------------------------------------

    def selected_model(self) -> Any:
        """The model the next job will use (the automatic choice when Automatic is selected), or None."""
        if self.mode == "framegen":
            return None
        mid = self.model_combo.currentData()
        if mid == "auto":
            mid = self.recommended
        if not mid:
            return None
        try:
            return self.registry.get(mid)
        except NeuralFrameError:
            return None

    def _update_model_status(self) -> None:
        """The line under the model list: ready, being downloaded (with progress), or still to download."""
        spec = self.selected_model()
        self.download_now_button.setVisible(False)
        self.cancel_download_button.setVisible(False)
        if spec is None or spec.weights is None:
            self.model_status.setText("")
            self.model_status_updated.emit()
            return
        prefix = "The automatic choice" if self.model_combo.currentData() == "auto" else "This model"
        if spec.is_engine and not self.downloads.is_active(spec.id):
            from neuralframe.engines.status import engine_model_status

            es = engine_model_status(spec, self.settings, self.library)
            note = ("Generative: it creates new, plausible detail (not the original's) and is very slow. "
                    + ("Personal use only. " if spec.licence_restricted else ""))
            if es.ready:
                self.model_status.setText(note + "Installed and ready.")
            else:
                self.model_status.setText(note + f"{es.text()} - about {es.download_bytes / 1e9:.1f} GB, installed "
                                          "when you press Start, or now:")
                self.download_now_button.setText("Install now")
                self.download_now_button.setVisible(True)
            self.model_status_updated.emit()
            return
        if self.downloads.is_active(spec.id):
            last: DownloadProgress | None = self.downloads.last.get(spec.id)
            self.model_status.setText(f"Downloading {spec.id}: {last.text() if last else 'starting...'}")
            self.cancel_download_button.setVisible(True)
        else:
            status = self.library.status(spec)
            if status.ready:
                self.model_status.setText(f"{prefix} is on this computer and ready.")
            else:
                size = f"{status.size_bytes / 1e6:.0f} MB" if status.size_bytes else "size unknown"
                restricted = " Its licence allows personal use only." if spec.licence_restricted else ""
                self.model_status.setText(f"{prefix} is not downloaded yet ({size}). It is downloaded when you press "
                                          f"Start, or now:{restricted}")
                self.download_now_button.setText("Resume download" if status.partial_bytes else "Download now")
                self.download_now_button.setVisible(True)
        self.model_status_updated.emit()

    def _download_selected(self) -> None:
        spec = self.selected_model()
        if spec is not None:
            self.downloads.start(spec)
            self._update_model_status()

    def _cancel_selected_download(self) -> None:
        spec = self.selected_model()
        if spec is not None:
            self.downloads.cancel(spec.id)

    def _download_finished(self, model_id: str, outcome: str, error: str) -> None:
        self._update_model_status()
        if outcome == "failed":
            self.error_label.setText(f"Could not download {model_id}: {error}")
        elif outcome == "done":
            self.statusBar().showMessage(f"Model {model_id} downloaded.", 8000)

    def open_model_manager(self, focus: str | None = None) -> Any:
        """The Models window (created once; downloads keep running when it is closed)."""
        from neuralframe.gui.models_dialog import ModelManager

        if self.models_window is None:
            self.models_window = ModelManager(self.registry, self.library, self.downloads, self,
                                              settings=self.settings)
            self.models_window.changed.connect(lambda _mid: self._update_model_status())
            self.models_window.add_model_requested.connect(self._choose_model_file)
        else:
            self.models_window.refresh()
        if focus:
            self.models_window.focus_model(focus)
        self.models_window.show()
        self.models_window.raise_()
        return self.models_window

    def _check_model_updates(self) -> None:
        self.open_model_manager().check_updates()

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
        self._update_recommendation()

    # ---- models ----------------------------------------------------------------------------------------

    def _fill_models(self, select: str | None = None) -> None:
        current = select or self.model_combo.currentData() or "auto"
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.setMaxVisibleItems(30)
        self.model_combo.addItem("Automatic: picks the right model for your video (auto)", "auto")
        self.model_combo.setItemData(0, "Looks at the video (cartoon or filmed, noise, compression, real "
                                        "resolution) and picks the best model within the preset's speed",
                                     Qt.ItemDataRole.ToolTipRole)
        items = self.model_combo.model()
        for title, models in self.registry.grouped():
            self.model_combo.insertSeparator(self.model_combo.count())
            self.model_combo.addItem(title.upper(), None)
            if isinstance(items, QStandardItemModel):  # a heading, not a choice
                heading = items.item(self.model_combo.count() - 1)
                heading.setFlags(Qt.ItemFlag.NoItemFlags)
                font = heading.font()
                font.setBold(True)
                heading.setFont(font)
            for m in models:
                marks = [x for x in ("personal use only" if m.licence_restricted else None,
                                     "added" if m.origin == "user" else None) if x]
                self.model_combo.addItem("    " + m.label + "".join(f"  [{x}]" for x in marks), m.id)
                self.model_combo.setItemData(self.model_combo.count() - 1, _model_tooltip(m),
                                             Qt.ItemDataRole.ToolTipRole)
        index = self.model_combo.findData(current)
        self.model_combo.setCurrentIndex(index if index >= 0 else 0)
        self.model_combo.blockSignals(False)

    def _analyse_content(self) -> None:
        """Content analysis for the automatic model choice, in the background after a file is loaded."""
        if self.input_path is None or self.media is None:
            return
        path, media = self.input_path, self.media
        self.auto_label.setText("<i>Automatic: analysing the video...</i>")

        def work() -> Any:
            from neuralframe.models.advisor import analyse_content

            return analyse_content(self.tools(), path, media)

        def done(profile: Any) -> None:
            if path != self.input_path:
                return  # another file was loaded meanwhile
            self.profile = profile
            self._update_recommendation()

        def failed(exc: BaseException) -> None:
            self.profile = None
            self.auto_label.setText(f"<i>Automatic: analysis failed ({exc}); the general model will be used</i>")
            self.recommendation_ready.emit()

        self._dispatch.run(work, done, failed, name="nf-analyse")

    def _update_recommendation(self) -> None:
        self._update_model_status()
        if self.model_combo.currentData() != "auto":
            self.auto_label.setText("")
            return
        if self.profile is None or self.media is None:
            return
        from neuralframe.core.planning import plan_geometry
        from neuralframe.models.advisor import recommend

        try:
            geometry, _ = plan_geometry(self.media.require_video(), self.target_spec(), 4)
        except (ValueError, NeuralFrameError):
            return
        gpu = bool(getattr(self._caps, "torch_cuda_available", False)) and self.settings.inference.device != "cpu"
        choice = recommend(self.profile, self.registry, factor=geometry.out_height / max(geometry.decoded_height, 1),
                           preset=self.preset_combo.currentData() or "balanced", is_image=self.media.is_image,
                           gpu=gpu)
        self.recommended = choice.model
        label = html.escape(self.registry.get(choice.model).label)
        reasons = html.escape("; ".join(choice.reasons[:4]))
        self.auto_label.setText(f"Automatic choice: <b>{label}</b> - {reasons}")
        self._update_model_status()
        self.recommendation_ready.emit()

    def _choose_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose a model file", "",
                                              "Model files (*.pth *.safetensors *.ckpt);;All files (*)")
        if path:
            self.add_model(Path(path))

    def add_model(self, file: Path) -> None:
        """Register a model file (checked and copied in the background) and select it."""
        self.error_label.clear()
        self.status_label.setText(f"Checking {file.name}...")

        def work() -> Any:
            from neuralframe.models.custom import add_user_model

            return add_user_model(file, self.settings.paths.models())

        def done(spec: Any) -> None:
            from neuralframe.models.registry import ModelRegistry

            self.registry = ModelRegistry([*self.registry.all(), spec])
            self._fill_models(select=spec.id)
            if self.models_window is not None:
                self.models_window.registry = self.registry
                self.models_window.populate()
            self.status_label.setText(f"Added model {spec.id}: {spec.description}")
            self.model_added.emit(spec.id)

        def failed(exc: BaseException) -> None:
            self.status_label.setText("Could not add the model.")
            self._show_error(exc)
            self.model_added.emit("")

        self._dispatch.run(work, done, failed, name="nf-add-model")

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
            self.profile = None
            self.recommended = None
            self._set_media_controls(True)
            self._update_mode()
            self._set_running(False)
            self.status_label.setText("Ready.")
            self.file_loaded.emit()
            self._analyse_content()

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
            faces=self.faces_combo.currentData() if upscale else None,
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
        if event.state == "downloading":
            download = event.download or {}
            total = download.get("total")
            self.progress.setRange(0, 1000 if total else 0)
            self.progress.setValue(int(event.percent * 10))
            self.status_label.setText(f"Downloading the model first: {download.get('label', '')}")
            self.stats_label.setText(download.get("text", ""))
            self.stages_label.clear()
            return
        waiting = event.state in ("probing", "planning") and event.frames_done == 0
        # Busy animation while planning (one-time TensorRT engine builds): never look frozen.
        self.progress.setRange(0, 0 if waiting else 1000)
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
        self.progress.setRange(0, 1000)
        if self._job_thread is not None:
            self._job_thread.join(5)
        self._job_thread = None
        self._runner = None
        self.result = result
        self._set_running(False)
        self._update_model_status()  # a job may have downloaded its model
        if self.models_window is not None:
            self.models_window.refresh()
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
            self.status_label.setText("Cancelled. Finished parts are kept (also a partly downloaded model): Resume "
                                      "continues where it stopped.")
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
        if self.downloads.active() and self._runner is None:
            answer = QMessageBox.question(
                self, "Downloads running", "Models are still downloading. Stop the downloads and quit?\n"
                "The parts already downloaded are kept and continue next time.")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.downloads.cancel_all()
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


def _model_tooltip(m: Any) -> str:
    """What a model is, its licence and where its weights come from (model list tooltip)."""
    lines = [m.description or m.purpose, f"Scale: {m.scale}x - speed: {m.speed or 'n/a'} - licence: {m.licence}"]
    if m.licence_restricted:
        lines.append("Licence allows personal / non-commercial use only.")
    if m.is_temporal:
        lines.append("Uses neighbouring frames (video only).")
    if m.weights is not None and not m.bundled and m.origin == "builtin":
        lines.append(f"Downloaded the first time it is used ({(m.weights.size_bytes or 0) / 1e6:.0f} MB).")
    return "\n".join(lines)
