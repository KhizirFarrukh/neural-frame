"""Build the Windows release of NeuralFrame for people who never touch code.

Produces build/release/NeuralFrame-<version>-win64/ with
    NeuralFrame.exe      desktop app (double-click; no console window)
    neuralframe-cli.exe  command line (a separate name: Windows file names ignore case)
    ffmpeg/              FFmpeg 9 (checksum-pinned static build with NVENC/NVDEC)
    models/              AI model weights (works offline from the first start; the very large transformer
                         models are downloaded on first use instead, see `bundled` in registry.yaml)
    README.txt, licenses/
smoke-tests that folder as a user would run it (fresh cache, bundled FFmpeg and models only), then packs
it into a per-user installer (Inno Setup, when installed) and/or a portable zip, with SHA256SUMS.txt.

Run from the repository root, in the development environment (CUDA PyTorch, TensorRT, PySide6 and the
native module built), e.g.:
    python packaging/windows/build_release.py               # folder + installer (if Inno Setup is found)
    python packaging/windows/build_release.py --zip         # also a portable zip
    python packaging/windows/build_release.py --no-tensorrt # smaller build: PyTorch CUDA only
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

FFMPEG_URL = ("https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-11-13-04/"
              "ffmpeg-n9.0.2-25-g67b60c310b-win64-gpl-9.0.zip")
FFMPEG_SHA256 = "b1860fbeb14e87d1de44aecc5eb9e25dfecc7ee8efa8d54250995a9acfbe6070"
# TensorRT ships one builder resource per GPU architecture. Consumer cards: Turing sm75 (RTX 20/GTX 16),
# Ampere sm86 (RTX 30), Ada sm89 (RTX 40), Blackwell sm120 (RTX 50); PTX covers future cards. The
# datacenter-only architectures are dropped from the consumer build (~0.9 GB); TensorRT failures fall back
# to PyTorch automatically.
DATACENTER_ARCHS = ("sm80", "sm90", "sm100", "sm103", "sm110", "sm121")
NOTICE_DISTS = ["torch", "tensorrt-cu13", "tensorrt-cu13-libs", "tensorrt-cu13-bindings", "onnxruntime-gpu",
                "pyside6-essentials", "shiboken6", "numpy", "av", "pydantic", "pydantic-settings", "fastapi",
                "starlette", "uvicorn", "sqlalchemy", "typer", "click", "rich", "structlog", "httpx", "pillow",
                "pyyaml", "psutil", "nvidia-ml-py", "onnx", "torchvision", "spandrel", "einops", "safetensors"]


def step(text: str) -> None:
    print(f"\n=== {text}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def folder_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def gb(n: int) -> str:
    return f"{n / 1e9:.2f} GB"


# ---- preparation -----------------------------------------------------------------------------------------


def preflight(args: argparse.Namespace) -> dict[str, str]:
    if sys.platform != "win32":
        raise SystemExit("the Windows release must be built on Windows")
    found: dict[str, str] = {}
    required = ["PyInstaller", "PySide6", "torch", "torchvision", "spandrel", "neuralframe._native"]
    if args.tensorrt:
        required.append("tensorrt")
    if args.onnxruntime:
        required.append("onnxruntime")
    for module in required:
        if importlib.util.find_spec(module) is None:
            hint = {"PyInstaller": "pip install pyinstaller", "PySide6": 'pip install -e ".[gui]"',
                    "torchvision": "pip install torchvision --index-url https://download.pytorch.org/whl/cu130",
                    "spandrel": 'pip install -e ".[models]"',
                    "neuralframe._native": "pip install --no-build-isolation -e ."}.get(module, "")
            raise SystemExit(f"missing {module}. {hint}")
    import torch

    from neuralframe import __version__

    found["version"] = __version__
    found["torch"] = torch.__version__
    found["cuda_build"] = str(torch.version.cuda)
    if torch.version.cuda is None:
        print("WARNING: PyTorch is a CPU build; the release will not use NVIDIA GPUs for PyTorch inference")
    if args.tensorrt:
        import tensorrt

        found["tensorrt"] = tensorrt.__version__
    for key, value in found.items():
        print(f"  {key}: {value}")
    return found


def make_icon(path: Path) -> None:
    """A simple generated application icon (gradient tile, frame mark, 'NF')."""
    from PIL import Image, ImageDraw, ImageFont

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad = Image.new("RGBA", (size, size))
    px = grad.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * size)
            px[x, y] = (int(47 + 80 * t), int(111 - 40 * t), int(222 - 20 * t), 255)  # type: ignore[index]
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle((8, 8, size - 8, size - 8), radius=48, fill=255)
    img.paste(grad, (0, 0), mask)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((40, 52, size - 40, size - 52), radius=14, outline=(255, 255, 255, 230), width=10)
    try:
        font = ImageFont.truetype("segoeuib.ttf", 92)
    except OSError:
        font = ImageFont.load_default()
    box = draw.textbbox((0, 0), "NF", font=font)
    draw.text(((size - box[2]) / 2 - box[0] / 2, (size - box[3]) / 2 - box[1] / 2 - 4), "NF", font=font,
              fill=(255, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def version_file(path: Path, version: str) -> None:
    parts = [int(p) for p in version.split("+")[0].split(".") if p.isdigit()]
    nums = [*parts, 0, 0, 0, 0][:4]
    tup = ", ".join(map(str, nums))
    path.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers=({tup}), prodvers=({tup}), mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1,
                    subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Khizir Farrukh'),
      StringStruct('FileDescription', 'NeuralFrame - AI video upscaling and frame generation'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', 'NeuralFrame'),
      StringStruct('OriginalFilename', 'NeuralFrame.exe'),
      StringStruct('ProductName', 'NeuralFrame'),
      StringStruct('ProductVersion', '{version}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""", encoding="utf-8")


# ---- build -------------------------------------------------------------------------------------------------


def run_pyinstaller(args: argparse.Namespace, name: str, icon: Path, version_txt: Path, work: Path) -> Path:
    options = {"name": name, "icon": str(icon), "version_file": str(version_txt), "tensorrt": args.tensorrt,
               "onnxruntime": args.onnxruntime}
    env = {**os.environ, "NF_BUILD_OPTIONS": json.dumps(options)}
    dist = work / "dist"
    cmd = [sys.executable, "-m", "PyInstaller", str(HERE / "neuralframe.spec"), "--noconfirm", "--clean",
           "--log-level", "WARN", "--distpath", str(dist), "--workpath", str(work / "pyinstaller")]
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)
    app = dist / name
    gui, cli = app / "NeuralFrame.exe", app / "neuralframe-cli.exe"
    if not (gui.is_file() and cli.is_file()) or sha256(gui) == sha256(cli):
        raise SystemExit("expected two different executables, NeuralFrame.exe and neuralframe-cli.exe")
    return app


def add_ffmpeg(app: Path, cache: Path) -> None:
    archive = cache / "ffmpeg-win64.zip"
    if not archive.is_file() or sha256(archive) != FFMPEG_SHA256:
        print(f"  downloading {FFMPEG_URL}")
        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(FFMPEG_URL, archive)  # fixed https URL, verified below
    if sha256(archive) != FFMPEG_SHA256:
        archive.unlink()
        raise SystemExit("FFmpeg download failed its SHA256 check")
    target = app / "ffmpeg"
    target.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            name = member.rsplit("/", 1)[-1]
            if member.endswith(("/bin/ffmpeg.exe", "/bin/ffprobe.exe")) or name == "LICENSE.txt":
                (target / name).write_bytes(z.read(member))
    print(f"  FFmpeg 9.0.2 -> {target}")


def add_models(app: Path) -> None:
    from neuralframe.config import load_settings
    from neuralframe.models.download import ensure_weights
    from neuralframe.models.registry import ModelRegistry

    settings = load_settings(None)
    for spec in ModelRegistry.load(user=False).all():
        if spec.weights is None:
            continue
        if not spec.bundled:
            size = (spec.weights.size_bytes or 0) / 1e6
            print(f"  {spec.id}: downloaded on first use ({size:.0f} MB, {spec.licence})")
            continue
        src = ensure_weights(spec, settings.paths.models())
        dst = app / "models" / spec.id / spec.weights.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {spec.id}: {dst.stat().st_size / 1e6:.1f} MB ({spec.licence})")


def prune(app: Path, args: argparse.Namespace) -> None:
    internal = app / "_internal"
    removed = 0
    if args.tensorrt and args.trt_archs == "consumer":
        for arch in DATACENTER_ARCHS:
            for dll in internal.rglob(f"nvinfer_builder_resource_{arch}_*.dll"):
                removed += dll.stat().st_size
                dll.unlink()
    # Build-time only: C/C++ headers, CMake files and import libraries of PyTorch.
    for rel in ("torch/include", "torch/share", "torch/test"):
        path = internal / rel
        if path.is_dir():
            removed += folder_size(path)
            shutil.rmtree(path)
    for lib in (internal / "torch" / "lib").glob("*.lib"):
        removed += lib.stat().st_size
        lib.unlink()
    print(f"  removed {gb(removed)} not needed at run time")


def model_credits() -> list[str]:
    from neuralframe.models.registry import ModelRegistry

    return [f"  {'*' if not m.bundled else ' '}{m.id} - {m.licence} - {m.source_url}"
            for m in ModelRegistry.load(user=False).all() if m.weights is not None]


def add_docs(app: Path, version: str, found: dict[str, str]) -> None:
    (app / "README.txt").write_text((HERE / "README.txt").read_text(encoding="utf-8").replace("{version}", version),
                                    encoding="utf-8")
    # Portable users have no Start-menu entry for the command line: a double-clickable prompt in this folder.
    script = ['@echo off', 'cd /d "%~dp0"', "echo NeuralFrame command line - try: neuralframe-cli.exe --help", "cmd /k"]
    (app / "Command line.cmd").write_bytes(("\r\n".join(script) + "\r\n").encode("ascii"))
    lic = app / "licenses"
    lic.mkdir(exist_ok=True)
    if (app / "ffmpeg" / "LICENSE.txt").is_file():
        shutil.copy2(app / "ffmpeg" / "LICENSE.txt", lic / "FFmpeg-LICENSE.txt")
    lines = [f"NeuralFrame {version} - third-party components", "=" * 60, "",
             "FFmpeg 9.0.2 (static build by BtbN, GPL v3; https://ffmpeg.org, build scripts and sources: "
             "https://github.com/BtbN/FFmpeg-Builds). Runs as a separate program (ffmpeg/ffmpeg.exe).",
             "AI models (weights from their authors, unmodified; * = downloaded on first use, not included):",
             *model_credits(),
             "NVIDIA CUDA, cuDNN and TensorRT runtime libraries are redistributed under NVIDIA's licence terms "
             "(see the NVIDIA notices in the respective packages).", "",
             "Python packages included:"]
    for dist_name in NOTICE_DISTS:
        try:
            dist = importlib.metadata.distribution(dist_name)
        except importlib.metadata.PackageNotFoundError:
            continue
        meta = dist.metadata
        licence = meta.get("License-Expression") or meta.get("License") or ""
        licence = licence.splitlines()[0][:80] if licence else "see files"
        lines.append(f"  {meta['Name']} {dist.version} - {licence}")
        for f in dist.files or []:
            if f.name.upper().startswith(("LICENSE", "LICENCE", "COPYING", "NOTICE")):
                src = Path(str(dist.locate_file(f)))
                if src.is_file():
                    dst = lic / meta["Name"] / f.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
    lines += ["", f"Built with PyTorch {found['torch']} (CUDA {found['cuda_build']})"
              + (f", TensorRT {found['tensorrt']}" if "tensorrt" in found else "") + "."]
    (lic / "THIRD-PARTY-NOTICES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---- smoke test ----------------------------------------------------------------------------------------------


def smoke_test(app: Path, version: str, args: argparse.Namespace) -> dict[str, object]:
    """Run the packaged executables like a user: fresh cache (so bundled models and a fresh TensorRT engine
    are used), bundled FFmpeg only."""
    results: dict[str, object] = {}
    cli = app / "neuralframe-cli.exe"
    with tempfile.TemporaryDirectory(prefix="nf-release-test-") as tmp_s:
        tmp = Path(tmp_s)
        env = {**os.environ, "NEURALFRAME_PATHS__CACHE_DIR": str(tmp / "cache"), "QT_QPA_PLATFORM": "offscreen",
               "PATH": os.environ.get("SYSTEMROOT", r"C:\Windows") + r"\System32"}  # no developer tools on PATH
        env.pop("NEURALFRAME_FFMPEG", None)
        env.pop("NEURALFRAME_FFPROBE", None)

        def run(*cmd: str, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
            t0 = time.perf_counter()
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout, cwd=tmp)
            print(f"  {' '.join(Path(c).name if i == 0 else c for i, c in enumerate(cmd))} -> exit {proc.returncode}"
                  f" ({time.perf_counter() - t0:.0f} s)")
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-4000:], sep="\n")
                raise SystemExit("smoke test failed")
            return proc

        out = run(str(cli), "version").stdout.strip()
        assert out == version, f"version {out!r} != {version!r}"
        doctor = json.loads(run(str(cli), "doctor", "--json").stdout)
        results["doctor"] = doctor["selected"]
        results["gpu"] = [g["name"] for g in doctor.get("gpus", [])]
        clip = tmp / "clip.mp4"
        subprocess.run([str(app / "ffmpeg" / "ffmpeg.exe"), "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=30:duration=2", "-f", "lavfi", "-i", "sine=f=440:duration=2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(clip)],
                       check=True, env=env, timeout=120, stdin=subprocess.DEVNULL)

        def report(output: Path) -> dict[str, object]:
            data: dict[str, object] = json.loads(output.with_name(output.stem + ".report.json").read_text("utf-8"))
            assert data["sync"]["ok"], data["sync"]  # type: ignore[index]
            return data

        out_l = tmp / "lanczos.mkv"
        run(str(cli), "upscale", str(clip), "-o", str(out_l), "-m", "lanczos", "--target", "x2", "-y")
        results["lanczos"] = report(out_l)["output"]
        cuda = bool(results["gpu"])
        if cuda and not args.no_models:
            out_a = tmp / "anime.mkv"
            run(str(cli), "upscale", str(clip), "-o", str(out_a), "-m", "realesr-animevideov3", "--target", "x2",
                "-y")
            rep = report(out_a)
            runner = (rep.get("processor") or {}).get("runner", {})  # type: ignore[union-attr]
            results["neural"] = {"backend": runner.get("backend"), "engine_built": runner.get("engine_built_now"),
                                 "fps": rep["throughput"]["pipeline_fps"]}  # type: ignore[index]
            if args.tensorrt:
                assert runner.get("backend") == "tensorrt", f"expected TensorRT, got {runner.get('backend')}"
            # A model loaded through spandrel (PyTorch only): checks the frozen spandrel/torchvision.
            out_b = tmp / "bsrgan.mkv"
            run(str(cli), "upscale", str(clip), "-o", str(out_b), "-m", "bsrgan-x4", "--target", "x2", "-y")
            results["spandrel"] = report(out_b)["throughput"]["pipeline_fps"]  # type: ignore[index]
        choice = json.loads(run(str(cli), "recommend", str(clip), "--json").stdout)
        results["recommend"] = {"model": choice["model"], "content": choice["profile"]["content"]}
        if not args.no_models:
            out_f = tmp / "fg.mkv"
            run(str(cli), "interpolate", str(clip), "-o", str(out_f), "--fps", "60", "-y")
            results["framegen"] = report(out_f)["framegen"]["runtime"]  # type: ignore[index]
        gui_report = tmp / "gui.json"
        subprocess.run([str(app / "NeuralFrame.exe"), str(clip), "--self-test", str(gui_report)], env=env,
                       timeout=300, cwd=tmp)
        gui = json.loads(gui_report.read_text(encoding="utf-8"))
        if not gui.get("ok"):
            print(json.dumps(gui, indent=2))
            raise SystemExit("GUI self-test failed")
        results["gui"] = {k: gui.get(k) for k in ("ffmpeg", "torch", "cuda", "gpu", "tensorrt", "spandrel",
                                                  "sample_info", "auto_model")}
        if not gui.get("spandrel"):
            raise SystemExit(f"spandrel is not usable in the package: {gui.get('spandrel_error')}")
        missing = [m for m, ok in gui.get("models_available", {}).items() if not ok]
        if missing and not args.no_models:
            raise SystemExit(f"models missing in the package: {missing}")
    return results


# ---- packaging ------------------------------------------------------------------------------------------------


def find_iscc(explicit: str | None) -> Path | None:
    candidates = [explicit] if explicit else []
    candidates += [shutil.which("iscc") or "", r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
                   r"C:\Program Files\Inno Setup 6\ISCC.exe",
                   str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe")]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def build_installer(iscc: Path, app: Path, out: Path, version: str, icon: Path) -> list[Path]:
    # Windows cannot run setup executables much beyond 2 GB: larger payloads go into Setup-N.bin files that
    # must stay next to the .exe.
    spanning = "yes" if folder_size(app) > 2_000_000_000 else "no"
    cmd = [str(iscc), "/Q", f"/DAppVersion={version}", f"/DSourceDir={app}", f"/DOutputDir={out}",
           f"/DIconFile={icon}", f"/DSpanning={spanning}", str(HERE / "neuralframe.iss")]
    subprocess.run(cmd, check=True)
    stem = f"NeuralFrame-{version}-win64-Setup"
    return sorted(p for p in out.iterdir() if p.name.startswith(stem))


def build_zip(app: Path, out: Path) -> Path:
    target = out / f"{app.name}-portable.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for f in sorted(app.rglob("*")):
            if f.is_file():
                z.write(f, Path(app.name) / f.relative_to(app))
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "release")
    parser.add_argument("--no-tensorrt", dest="tensorrt", action="store_false",
                        help="leave TensorRT out (~2 GB smaller, NVIDIA inference 1.3-2.6x slower)")
    parser.add_argument("--trt-archs", choices=["consumer", "all"], default="consumer",
                        help="TensorRT GPU architectures to ship (consumer = GeForce RTX 20/30/40/50 + PTX)")
    parser.add_argument("--with-onnxruntime", dest="onnxruntime", action="store_true",
                        help="also ship ONNX Runtime (not needed on NVIDIA; TensorRT/PyTorch are used)")
    parser.add_argument("--no-models", action="store_true", help="do not bundle weights (downloaded on first use)")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--zip", action="store_true", help="also create a portable zip")
    parser.add_argument("--no-installer", action="store_true")
    parser.add_argument("--iscc", help="path to Inno Setup's ISCC.exe")
    args = parser.parse_args()

    t0 = time.perf_counter()
    step("checking the build environment")
    found = preflight(args)
    version = found["version"]
    name = f"NeuralFrame-{version}-win64" + ("" if args.tensorrt else "-lite")
    out: Path = args.out
    work = out / "work"
    icon = HERE / "neuralframe.ico"
    if not icon.is_file():
        make_icon(icon)
    work.mkdir(parents=True, exist_ok=True)
    version_txt = work / "version_info.txt"
    version_file(version_txt, version)

    step("freezing the application (PyInstaller; takes a while: PyTorch and TensorRT are large)")
    app = run_pyinstaller(args, name, icon, version_txt, work)
    step("adding FFmpeg")
    add_ffmpeg(app, ROOT / "build" / "release-cache")
    if not args.no_models:
        step("adding model weights")
        add_models(app)
    step("removing build-time files")
    prune(app, args)
    add_docs(app, version, found)
    final = out / name
    if final.exists():
        shutil.rmtree(final)
    shutil.move(str(app), final)
    print(f"  application folder: {final} ({gb(folder_size(final))})")

    if not args.skip_tests:
        step("smoke-testing the packaged application (fresh cache, bundled FFmpeg/models only)")
        results = smoke_test(final, version, args)
        (out / f"{name}-smoke-test.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(json.dumps(results, indent=2, default=str))

    artifacts: list[Path] = []
    if not args.no_installer:
        iscc = find_iscc(args.iscc)
        if iscc is None:
            print("\nInno Setup not found: skipping the installer. Install Inno Setup 6 "
                  "(https://jrsoftware.org/isinfo.php or `winget install JRSoftware.InnoSetup`) and run again, "
                  "or use --zip for a portable package.")
        else:
            step(f"building the installer ({iscc})")
            artifacts += build_installer(iscc, final, out, version, icon)
    if args.zip:
        step("building the portable zip")
        artifacts.append(build_zip(final, out))
    if artifacts:
        sums = out / "SHA256SUMS.txt"
        sums.write_text("".join(f"{sha256(a)}  {a.name}\n" for a in artifacts), encoding="utf-8")
        artifacts.append(sums)
    step(f"done in {(time.perf_counter() - t0) / 60:.1f} min")
    print(f"  {final}  ({gb(folder_size(final))})  <- run NeuralFrame.exe from here")
    for a in artifacts:
        print(f"  {a}  ({gb(a.stat().st_size)})")


if __name__ == "__main__":
    main()
