"""Installing an engine into its own environment (on demand), checking and removing it.

Layout under ``<cache>/engines/<engine>/``: ``venv/`` (Python + packages), ``src/`` (the upstream code at the
pinned commit, plus our driver scripts), ``data/`` (Hugging Face / torch caches of engines that download
their own weights), ``install.log`` and ``installed.json`` (what was installed; a changed recipe means the
engine is reinstalled). A pinned ``uv`` (downloaded once into ``<cache>/tools``, SHA256-checked) creates the
environment - including the Python interpreter, so nothing has to be installed on the computer.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

from neuralframe.engines.catalog import EngineDef
from neuralframe.engines.process import Cancellable, run
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.log import get_logger

log = get_logger(component="engines")

UV_VERSION = "0.13.0"
UV_ASSETS = {  # platform -> (asset, sha256)
    ("win32", "AMD64"): ("uv-x86_64-pc-windows-msvc.zip",
                         "088962f9e7b7bd9ea740c04c650b2a21c8928c345bd99ac24350dc924dba656c"),
    ("linux", "x86_64"): ("uv-x86_64-unknown-linux-gnu.tar.gz",
                          "1468ebd5a5541121837c5a2817b9972ba6090fa6caa3d142620850a47fb75154"),
    ("linux", "aarch64"): ("uv-aarch64-unknown-linux-gnu.tar.gz",
                           "3ccfb6af6e242433eb552f7d9676abd5412c6595c497e990d9c8cb7b5bd4d2c3"),
    ("darwin", "arm64"): ("uv-aarch64-apple-darwin.tar.gz",
                          "a9c1b29002cf3c83f07fa9cd8a887a3be0107d90e23189721221e7257db8e3d6"),
    ("darwin", "x86_64"): ("uv-x86_64-apple-darwin.tar.gz",
                           "5f44dcbde809b632f47c36fadb241cb4d6f9af71d0c8f172b5d2026d3dde742c"),
}
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"

EnvState = Literal["missing", "installed", "outdated"]
STEPS = ["installer", "python", "source", "pytorch", "packages", "finishing", "weights"]


@dataclass
class InstallProgress:
    engine: str
    step: str
    """One of :data:`STEPS`."""
    fraction: float
    """Overall, 0..1 (by step; downloads inside a step move it smoothly)."""
    message: str


InstallCallback = Callable[[InstallProgress], None]


def _requirement_name(line: str) -> str:
    name = line.split(";", 1)[0].strip()
    for sep in ("==", ">=", "<=", "~=", "!=", ">", "<", "[", "@", " "):
        name = name.split(sep, 1)[0]
    return name.strip().lower().replace("_", "-")


def filter_requirements(text: str, exclude: tuple[str, ...]) -> list[str]:
    """Upstream requirement lines minus excluded packages, comments and pip options."""
    banned = {e.lower().replace("_", "-") for e in exclude}
    kept = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        if _requirement_name(line) in banned:
            continue
        kept.append(line)
    return kept


class EngineEnv:
    """One engine's environment."""

    def __init__(self, engine: EngineDef, cache_dir: Path, *, cuda_index: str | None = None,
                 hf_token: str | None = None, ffmpeg_dir: Path | None = None) -> None:
        self.engine = engine
        self.cache_dir = Path(cache_dir)
        self.root = self.cache_dir / "engines" / engine.id
        self.venv = self.root / "venv"
        self.src = self.root / "src"
        self.data = self.root / "data"
        self.log_path = self.root / "install.log"
        self.state_path = self.root / "installed.json"
        self.cuda_index = cuda_index or engine.torch_index
        self.hf_token = hf_token
        self.ffmpeg_dir = ffmpeg_dir

    @classmethod
    def for_settings(cls, engine: EngineDef, settings: object) -> EngineEnv:
        from neuralframe.config import Settings, default_cache_dir
        from neuralframe.engines.secrets import hf_token

        assert isinstance(settings, Settings)
        return cls(engine, settings.paths.cache_dir or default_cache_dir(), cuda_index=settings.engines.cuda_index,
                   hf_token=hf_token(settings))

    # ---- status --------------------------------------------------------------------------------------------

    @property
    def python(self) -> Path:
        return self.venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def recipe(self) -> dict[str, object]:
        """Everything that defines the installation (a change means reinstalling)."""
        e = self.engine
        return {"id": e.id, "repo": e.repo, "commit": e.commit, "python": e.python, "torch": list(e.torch),
                "index": self.cuda_index, "requirements": e.requirements, "exclude": sorted(e.exclude),
                "packages": list(e.packages), "editable": e.editable,
                "post_install": [list(c) for c in e.post_install], "resources": [list(r) for r in e.resources],
                "prefetch": list(e.prefetch), "resource_hash": self._resource_hash()}

    def _resource_hash(self) -> str:
        digest = hashlib.sha256()
        for name, _dest in self.engine.resources:
            digest.update(resources.files("neuralframe.engines.resources").joinpath(name).read_bytes())
        return digest.hexdigest()[:16]

    def state(self) -> EnvState:
        if not self.state_path.is_file() or not self.python.is_file():
            return "missing"
        try:
            installed = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "missing"
        return "installed" if installed.get("recipe") == self.recipe() else "outdated"

    def disk_usage(self) -> int:
        if not self.root.is_dir():
            return 0
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())

    def remove(self) -> int:
        freed = self.disk_usage()
        if self.root.is_dir():
            shutil.rmtree(self.root, ignore_errors=True)
        return freed

    # ---- running ---------------------------------------------------------------------------------------------

    def environ(self) -> dict[str, str]:
        """Environment for running the engine (its own caches, our FFmpeg, no user site-packages)."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "CONDA", "VIRTUAL_ENV"))}
        scripts = self.python.parent
        path = [str(scripts)]
        if self.ffmpeg_dir is not None:
            path.append(str(self.ffmpeg_dir))
        env["PATH"] = os.pathsep.join([*path, env.get("PATH", "")])
        env.update({
            "VIRTUAL_ENV": str(self.venv), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1",
            "HF_HOME": str(self.data / "huggingface"), "TORCH_HOME": str(self.data / "torch"),
            "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        })
        if self.hf_token:
            env["HF_TOKEN"] = self.hf_token
        installed = self._installed()
        if installed.get("shims"):
            env["PYTHONPATH"] = str(self.src / "nf_shims")
        return env

    def _installed(self) -> dict[str, object]:
        try:
            data: dict[str, object] = json.loads(self.state_path.read_text(encoding="utf-8"))
            return data
        except (OSError, json.JSONDecodeError):
            return {}

    # ---- installing ------------------------------------------------------------------------------------------

    def install(self, progress: InstallCallback | None = None, cancel: Cancellable | None = None) -> None:
        """Create or refresh the environment. Safe to call again after a failure or cancellation."""
        e = self.engine
        t0 = time.monotonic()
        self.root.mkdir(parents=True, exist_ok=True)

        def report(step: str, within: float, message: str) -> None:
            index = STEPS.index(step)
            if progress is not None:
                progress(InstallProgress(e.id, step, min(1.0, (index + within) / len(STEPS)), message))

        def runner(step: str) -> Callable[[str], None]:
            return lambda line: report(step, 0.5, line[-200:])

        uv = self._ensure_uv(lambda frac, msg: report("installer", frac, msg), cancel)
        uv_env = self._uv_env()

        report("python", 0.0, f"creating a Python {e.python} environment")
        if self.state() == "outdated" and self.venv.is_dir():
            shutil.rmtree(self.venv, ignore_errors=True)
        if not self.python.is_file():
            run([uv, "venv", "--python", e.python, "--seed", self.venv], cwd=self.root, env=uv_env,
                on_line=runner("python"), cancel=cancel, log_path=self.log_path, what="uv venv")

        report("source", 0.0, "downloading the engine's code")
        self._ensure_source(lambda frac, msg: report("source", frac, msg), cancel)

        index = TORCH_INDEX.format(tag=self.cuda_index)
        pip: list[str | Path] = [uv, "pip", "install", "--python", self.python]
        report("pytorch", 0.0, f"installing PyTorch ({self.cuda_index}); this is a large download")
        run([*pip, *e.torch, "--index-url", index, "--extra-index-url", "https://pypi.org/simple",
             "--index-strategy", "unsafe-best-match"],
            cwd=self.root, env=uv_env, on_line=runner("pytorch"), cancel=cancel, log_path=self.log_path,
            what="installing PyTorch")

        report("packages", 0.0, "installing the engine's packages")
        constraints = self.root / "constraints.nf.txt"
        constraints.write_text("\n".join(t for t in e.torch if "==" in t) + "\n", encoding="utf-8")
        lines: list[str] = []
        if e.requirements:
            req = self.src / e.requirements
            if req.is_file():
                lines += filter_requirements(req.read_text(encoding="utf-8", errors="replace"), e.exclude)
        overridden = {_requirement_name(p) for p in e.packages}
        lines = [ln for ln in lines if _requirement_name(ln) not in overridden] + list(e.packages)
        if lines:
            requirements = self.root / "requirements.nf.txt"
            requirements.write_text("\n".join(lines) + "\n", encoding="utf-8")
            run([*pip, "-r", requirements, "--constraint", constraints, "--extra-index-url", index,
                 "--index-strategy", "unsafe-best-match"],
                cwd=self.root, env=uv_env, on_line=runner("packages"), cancel=cancel, log_path=self.log_path,
                what="installing packages")
        if e.editable:
            run([*pip, "--no-deps", "-e", self.src], cwd=self.src, env=uv_env, on_line=runner("packages"),
                cancel=cancel, log_path=self.log_path, what="installing the engine")

        report("finishing", 0.0, "finishing the installation")
        for name, dest in e.resources:
            target = self.src / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(resources.files("neuralframe.engines.resources").joinpath(name).read_bytes())
        shims = self._sparse_attention(uv, uv_env, index, cancel) if e.id == "flashvsr" else False
        env = self.environ() | ({"PYTHONPATH": str(self.src / "nf_shims")} if shims else {})
        for command in e.post_install:
            run([self.python, *command], cwd=self.src, env=env, on_line=runner("finishing"), cancel=cancel,
                log_path=self.log_path, what="setting up the engine")

        if e.prefetch:
            report("weights", 0.0, "downloading the engine's weights (this can take a long time)")
            if e.needs_hf_token and not self.hf_token:
                raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE,
                                       f"{e.name} needs a Hugging Face access token",
                                       hint="create one at huggingface.co (Settings > Access Tokens), accept the "
                                       "model's licence on its page, then add it in Models > Hugging Face token or "
                                       "set HF_TOKEN")
            run([self.python, *e.prefetch], cwd=self.src, env=env, on_line=runner("weights"), cancel=cancel,
                log_path=self.log_path, what="downloading weights")

        self.state_path.write_text(json.dumps({
            "recipe": self.recipe(), "shims": shims, "installed_at": time.time(),
            "seconds": round(time.monotonic() - t0), "platform": [sys.platform, platform.machine()],
        }, indent=1), encoding="utf-8")
        report("weights", 1.0, "installed")
        log.info("engine installed", engine=e.id, seconds=round(time.monotonic() - t0))

    def _uv_env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "CONDA", "VIRTUAL_ENV", "UV_"))}
        tools = self.cache_dir / "tools"
        env.update({"UV_CACHE_DIR": str(tools / "uv-cache"), "UV_PYTHON_INSTALL_DIR": str(tools / "python"),
                    "UV_LINK_MODE": "copy", "PYTHONUTF8": "1"})
        return env

    def _ensure_uv(self, progress: Callable[[float, str], None], cancel: Cancellable | None) -> Path:
        from neuralframe.models.library import DownloadProgress, fetch_file

        key = (sys.platform, platform.machine())
        if key not in UV_ASSETS:
            raise NeuralFrameError(ErrorCode.BACKEND_UNAVAILABLE,
                                   f"engines are not available on this platform ({key[0]} {key[1]})")
        asset, sha = UV_ASSETS[key]
        folder = self.cache_dir / "tools" / f"uv-{UV_VERSION}"
        exe = folder / ("uv.exe" if sys.platform == "win32" else "uv")
        if exe.is_file():
            return exe
        archive = folder / asset

        def on_download(event: DownloadProgress) -> None:
            progress(event.fraction or 0.0, f"downloading the installer: {event.text()}")

        fetch_file(f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}", archive, sha256=sha,
                   label="uv", progress=on_download, cancel=cancel)
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(folder)
        else:
            with tarfile.open(archive) as t:
                t.extractall(folder, filter="data")
        found = next((p for p in folder.rglob(exe.name) if p.is_file()), None)
        if found is None:
            raise NeuralFrameError(ErrorCode.TOOL_MISSING, "the uv archive did not contain uv")
        if found != exe:
            shutil.move(str(found), exe)
        if sys.platform != "win32":
            exe.chmod(0o755)
        archive.unlink(missing_ok=True)
        return exe

    def _ensure_source(self, progress: Callable[[float, str], None], cancel: Cancellable | None) -> None:
        from neuralframe.models.library import DownloadProgress, fetch_file

        e = self.engine
        marker = self.src / ".nf-commit"
        if e.source_url is None:
            self.src.mkdir(parents=True, exist_ok=True)
            return
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == e.commit:
            return
        archive = self.root / f"source-{(e.commit or '')[:12]}.zip"

        def on_download(event: DownloadProgress) -> None:
            progress(event.fraction or 0.0, f"downloading the engine's code: {event.text()}")

        fetch_file(e.source_url, archive, label=f"{e.id} source", progress=on_download, cancel=cancel)
        if self.src.exists():
            shutil.rmtree(self.src, ignore_errors=True)
        staging = self.root / "src.tmp"
        shutil.rmtree(staging, ignore_errors=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(staging)
        tops = [p for p in staging.iterdir() if p.is_dir()]
        (tops[0] if len(tops) == 1 else staging).replace(self.src)
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
        marker.write_text(e.commit or "", encoding="utf-8")

    def _sparse_attention(self, uv: Path, uv_env: dict[str, str], index: str, cancel: Cancellable | None) -> bool:
        """FlashVSR: build Block-Sparse-Attention when a CUDA compiler is available; otherwise (or if the
        build fails) use NeuralFrame's PyTorch implementation of the same function. Returns True for the
        latter."""
        nvcc = shutil.which("nvcc") or (Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"
                                        if os.environ.get("CUDA_HOME") else None)
        if nvcc:
            try:
                run([uv, "pip", "install", "--python", self.python, "--no-build-isolation", "packaging", "ninja",
                     "--extra-index-url", index], cwd=self.root, env=uv_env, cancel=cancel, log_path=self.log_path,
                    what="installing build tools")
                run([uv, "pip", "install", "--python", self.python, "--no-build-isolation",
                     "git+https://github.com/mit-han-lab/Block-Sparse-Attention"], cwd=self.root, env=uv_env,
                    cancel=cancel, log_path=self.log_path, what="building Block-Sparse-Attention")
                return False
            except NeuralFrameError as exc:
                log.warning("Block-Sparse-Attention build failed; using the PyTorch implementation", error=str(exc))
        return True

    def summary(self) -> dict[str, object]:
        return {"engine": self.engine.id, "state": self.state(), "root": str(self.root),
                "disk_bytes": self.disk_usage()}
