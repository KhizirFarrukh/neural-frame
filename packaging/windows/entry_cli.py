"""Frozen entry point: neuralframe-cli.exe (command line)."""

import multiprocessing

from neuralframe.cli.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
