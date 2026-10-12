# PyInstaller spec for the Windows release: two executables sharing one folder.
#   NeuralFrame.exe  - desktop app (windowed)
#   neuralframe-cli.exe - command line (console). Not "neuralframe.exe": Windows file names ignore case,
#                         so it would overwrite NeuralFrame.exe.
# Driven by packaging/windows/build_release.py, which passes options in NF_BUILD_OPTIONS (JSON).
# -*- mode: python ; coding: utf-8 -*-

import importlib.util
import json
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parents[1]  # noqa: F821 - SPECPATH is provided by PyInstaller
HERE = ROOT / "packaging" / "windows"
opts = json.loads(os.environ["NF_BUILD_OPTIONS"])

datas = collect_data_files("neuralframe")  # registry/presets YAML, web UI
native = importlib.util.find_spec("neuralframe._native")
if native is None or native.origin is None:
    raise SystemExit("neuralframe._native is not built; run: pip install --no-build-isolation -e .")
binaries = [(native.origin, "neuralframe")]
hiddenimports = [
    *collect_submodules("neuralframe", filter=lambda name: not name.startswith("neuralframe.testing.")),
    "neuralframe.testing.media",
    *collect_submodules("uvicorn"),
    "sqlalchemy.dialects.sqlite",
]
excludes = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "pytestqt", "_pytest", "playwright",
    "torchvision", "lpips", "scipy", "PyInstaller", "mypy", "ruff", "setuptools", "pip",
]
if opts["tensorrt"]:
    hiddenimports += ["tensorrt", "tensorrt_libs"]
    binaries += collect_dynamic_libs("tensorrt_libs")
else:
    excludes += ["tensorrt", "tensorrt_libs", "tensorrt_bindings"]
if not opts["onnxruntime"]:
    excludes += ["onnxruntime"]

a = Analysis(
    [str(HERE / "entry_gui.py"), str(HERE / "entry_cli.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

common = {"exclude_binaries": True, "icon": opts["icon"], "version": opts["version_file"], "upx": False}
gui = EXE(pyz, [s for s in a.scripts if s[0] != "entry_cli"], [], name="NeuralFrame", console=False, **common)
cli = EXE(pyz, [s for s in a.scripts if s[0] != "entry_gui"], [], name="neuralframe-cli", console=True, **common)
COLLECT(gui, cli, a.binaries, a.datas, name=opts["name"], upx=False)
