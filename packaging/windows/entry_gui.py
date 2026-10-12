"""Frozen entry point: NeuralFrame.exe (desktop app, no console window)."""

import multiprocessing
import sys

from neuralframe.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
