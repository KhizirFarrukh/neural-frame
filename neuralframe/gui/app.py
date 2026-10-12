"""``neuralframe-gui`` entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuralframe-gui", description="NeuralFrame desktop application")
    parser.add_argument("file", nargs="?", type=Path, help="video or image to open")
    parser.add_argument("--config", "-c", type=Path, help="YAML config file")
    args = parser.parse_args(argv)
    try:
        from PySide6.QtWidgets import QApplication
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
    configure_logging("WARNING", json_output=False, log_file=settings.logging.file)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("NeuralFrame")
    window = MainWindow(settings)
    window.show()
    if args.file:
        window.load_file(args.file.resolve())
    return int(app.exec())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
