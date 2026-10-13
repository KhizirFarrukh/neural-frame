"""``neuralframe-gui`` entry point (also ``NeuralFrame.exe`` in the Windows release)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from types import TracebackType
from typing import Any


def _windowed_streams(log_dir: Path) -> Path:
    """A windowed executable has no console (``sys.stderr`` is None): send output to a log file instead."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "neuralframe-gui.log"
    stream = path.open("a", encoding="utf-8", buffering=1)
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return path


def _self_test(window: Any, app: Any, sample: Path | None, report: Path) -> int:
    """Headless check of a packaged build: window, bundled FFmpeg, models, native module, GPU stack."""
    import importlib.util
    import time

    from neuralframe.bundle import app_dir
    from neuralframe.models.download import weights_path

    result: dict[str, Any] = {"frozen": app_dir() is not None, "ok": False}
    try:
        tools = window.tools()
        result["ffmpeg"] = tools.describe()
        import neuralframe._native as native

        result["native"] = native.build_info()
        if importlib.util.find_spec("torch") is not None:
            import torch

            result["torch"] = torch.__version__
            result["cuda"] = torch.cuda.is_available()
            if result["cuda"]:
                result["gpu"] = torch.cuda.get_device_name(0)
        result["tensorrt"] = importlib.util.find_spec("tensorrt") is not None
        try:
            import spandrel

            result["spandrel"] = spandrel.__version__
        except ImportError as exc:  # optional when running from source
            result["spandrel"] = None
            result["spandrel_error"] = str(exc)
        models = {}
        for m in window.registry.all():
            if m.weights is not None and m.bundled and m.origin == "builtin":
                from neuralframe.bundle import bundled

                local = weights_path(m, window.settings.paths.models()).is_file()
                models[m.id] = bool(bundled("models", m.id, m.weights.filename)) or local
        result["models_available"] = models
        if sample is not None:
            loaded: list[bool] = []
            window.file_loaded.connect(lambda: loaded.append(True))
            window.load_file(sample)
            deadline = time.monotonic() + 60
            while not loaded and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.02)
            result["sample_loaded"] = window.media is not None
            result["sample_info"] = window.info_label.text()
            ready: list[bool] = []
            window.recommendation_ready.connect(lambda: ready.append(True))
            deadline = time.monotonic() + 60
            while not ready and window.recommended is None and time.monotonic() < deadline:
                app.processEvents()
                time.sleep(0.02)
            result["auto_model"] = window.recommended
        result["ok"] = bool(result.get("ffmpeg")) and (sample is None or result.get("sample_loaded", False))
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["ok"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuralframe-gui", description="NeuralFrame desktop application")
    parser.add_argument("file", nargs="?", type=Path, help="video or image to open")
    parser.add_argument("--config", "-c", type=Path, help="YAML config file")
    parser.add_argument("--self-test", type=Path, metavar="REPORT.json", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
    except ImportError:
        print("The desktop app needs PySide6: pip install neuralframe[gui]", file=sys.stderr)
        return 2

    from neuralframe.config import load_settings
    from neuralframe.errors import NeuralFrameError
    from neuralframe.gui.window import MainWindow
    from neuralframe.log import configure_logging

    try:
        settings = load_settings(args.config)
    except NeuralFrameError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    log_file = settings.logging.file
    if sys.stderr is None or sys.stdout is None:
        log_file = log_file or _windowed_streams(settings.paths.cache_dir / "logs")
    configure_logging("WARNING", json_output=False, log_file=log_file)

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("NeuralFrame")

    def excepthook(kind: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        text = "".join(traceback.format_exception(kind, exc, tb))
        print(text, file=sys.stderr)
        QMessageBox.critical(None, "NeuralFrame", f"Something went wrong:\n\n{exc}\n\nDetails are in the log file.")

    sys.excepthook = excepthook
    window = MainWindow(settings)
    if args.self_test is not None:
        return _self_test(window, app, args.file.resolve() if args.file else None, args.self_test)
    window.show()
    if args.file:
        window.load_file(args.file.resolve())
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
