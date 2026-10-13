"""Each engine's command line for "restore these frames towards this size", and finding its results.

Arguments follow the upstream projects at the pinned commits (see catalog.py); where a project has no usable
command line, NeuralFrame's own driver script (``resources/``) calls its Python API.
"""

from __future__ import annotations

import math
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neuralframe.engines.env import EngineEnv
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.models.registry import ModelSpec

HYPIR_LORA_MODULES = ",".join(["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut",
                               "conv_out", "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"])
_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)")
_IMAGE = (".png", ".jpg", ".jpeg", ".webp")


@dataclass
class EngineCall:
    """One run of an engine over one chunk."""

    env: EngineEnv
    model: ModelSpec
    files: dict[str, Path]
    """The model's files by name (``filename`` in the registry -> path on disk)."""
    input: Path
    """A lossless video file (``io: video``) or a folder of PNG frames named 000000.png, ..."""
    output: Path
    frames: int
    in_size: tuple[int, int]
    out_size: tuple[int, int]
    """What the job needs (width, height); engines aim for it, the job resizes the rest."""
    fps: float = 24.0
    gpu: int = 0
    vram_gb: float | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Command:
    args: list[str]
    cwd: Path
    env: dict[str, str]
    results: Path
    """Folder searched (recursively) for the output frames."""
    prefer: tuple[str, ...] = ()
    """Sub-folders that hold the final frames when an engine writes several kinds of images."""


def _file(call: EngineCall, name: str) -> Path:
    path = call.files.get(name)
    if path is None:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"{call.model.id}: missing file {name}")
    return path


def _model_dir(call: EngineCall, prefix: str = "") -> Path:
    """The folder holding the model's files (with an optional sub-folder, e.g. a diffusers model)."""
    first = next(iter(call.files.values()))
    root = first.parent
    for _ in Path(next(iter(call.files))).parts[:-1]:
        root = root.parent
    return root / prefix if prefix else root


def _scale(call: EngineCall, allowed: tuple[int, ...] = (2, 4)) -> int:
    need = max(call.out_size[1] / max(call.in_size[1], 1), call.out_size[0] / max(call.in_size[0], 1))
    for s in allowed:
        if s >= need - 1e-6:
            return s
    return allowed[-1]


def _python(call: EngineCall) -> str:
    return str(call.env.python)


# ---- video engines --------------------------------------------------------------------------------------


def seedvr2(call: EngineCall) -> Command:
    o = call.model.engine_options
    dit = str(o["dit_model"])
    model_dir = _model_dir(call)
    if call.files[dit].parent != model_dir:  # a linked file elsewhere: SeedVR2 reads one folder
        target = model_dir / dit
        if not target.exists():
            try:
                os.link(call.files[dit], target)
            except OSError as exc:
                raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE,
                                       f"SeedVR2 needs {dit} in {model_dir}", hint="download it from the Models "
                                       "window instead of linking it") from exc
    vram = call.vram_gb or 8.0
    big = "7b" in dit
    short = min(call.out_size)
    batch = 5 if vram < 12 else 9 if vram < 20 else 21 if vram < 32 else 33
    batch = max(1, min(batch, 4 * ((call.frames - 1) // 4) + 1))
    args = [_python(call), str(call.env.src / "inference_cli.py"), str(call.input), "--output", str(call.output),
            "--output_format", "png", "--model_dir", str(model_dir), "--dit_model", dit,
            "--resolution", str(short), "--batch_size", str(batch),
            "--color_correction", str(o.get("color_correction", "lab")), "--cuda_device", str(call.gpu)]
    if vram < 12:
        args += ["--dit_offload_device", "cpu", "--vae_offload_device", "cpu", "--blocks_to_swap",
                 "36" if big else "32", "--swap_io_components", "--vae_encode_tiled", "--vae_decode_tiled"]
    elif vram < 24:
        args += ["--dit_offload_device", "cpu", "--blocks_to_swap", "20" if big else "16", "--vae_decode_tiled"]
    return Command(args, call.env.src, call.env.environ(), call.output)


def flashvsr(call: EngineCall) -> Command:
    o = call.model.engine_options
    args = [_python(call), "nf_flashvsr_driver.py", "--input", str(call.input), "--output", str(call.output),
            "--weights", str(_model_dir(call)), "--variant", str(o.get("variant", "full")), "--scale", "4",
            "--sparse-ratio", str(o.get("sparse_ratio", 2.0)), "--local-range", str(o.get("local_range", 11)),
            "--device", f"cuda:{call.gpu}"]
    if (call.vram_gb or 8.0) < 16:
        args.append("--tiled")
    return Command(args, call.env.src / "examples" / "WanVSR", call.env.environ(), call.output)


def dove(call: EngineCall) -> Command:
    args = [_python(call), "inference_script.py", "--input_dir", str(call.input.parent), "--model_path",
            str(_model_dir(call)), "--output_path", str(call.output), "--is_vae_st", "--png_save",
            "--upscale", str(_scale(call, (4,))), "--dtype", "bfloat16"]
    if (call.vram_gb or 8.0) < 40:
        args.append("--is_cpu_offload")
    return Command(args, call.env.src, call.env.environ(), call.output)


def swiftvr(call: EngineCall) -> Command:
    args = [_python(call), "scripts/inference.py", "--input", str(call.input), "--output", str(call.output),
            "--checkpoint", str(_model_dir(call)), "--upscale", str(_scale(call, (2, 4))), "--png",
            "--clip-len", "24", "--dtype", "bfloat16", "--device", f"cuda:{call.gpu}"]
    return Command(args, call.env.src, call.env.environ(), call.output)


def nvidia_vsr(call: EngineCall) -> Command:
    o = call.model.engine_options
    scale = max(1, min(4, math.ceil(max(call.out_size[1] / max(call.in_size[1], 1), 1.0) - 1e-6)))
    args = [_python(call), "nf_nvidia_vsr_driver.py", "--input", str(call.input), "--output", str(call.output),
            "--scale", str(scale), "--quality", str(o.get("quality", "HIGH")), "--gpu", str(call.gpu)]
    return Command(args, call.env.src, call.env.environ(), call.output)


# ---- picture engines ------------------------------------------------------------------------------------


def osediff(call: EngineCall) -> Command:
    root = _model_dir(call)
    args = [_python(call), "test_osediff.py", "-i", str(call.input), "-o", str(call.output),
            "--osediff_path", "preset/models/osediff.pkl", "--pretrained_model_name_or_path", str(root / "sd21"),
            "--ram_ft_path", "preset/models/DAPE.pth", "--ram_path", str(root / "ram_swin_large_14m.pth"),
            "--upscale", str(_scale(call, (4,))), "--mixed_precision", "fp16"]
    return Command(args, call.env.src, call.env.environ(), call.output)


def hypir(call: EngineCall) -> Command:
    root = _model_dir(call)
    args = [_python(call), "test.py", "--base_model_type", "sd2", "--base_model_path", str(root / "sd21"),
            "--model_t", "200", "--coeff_t", "200", "--lora_rank", "256", "--lora_modules", HYPIR_LORA_MODULES,
            "--weight_path", str(root / "HYPIR_sd2.pth"), "--patch_size", "512", "--stride", "256",
            "--lq_dir", str(call.input), "--scale_by", "factor", "--upscale", str(_scale(call, (1, 2, 4, 8))),
            "--captioner", "empty", "--output_dir", str(call.output), "--device", f"cuda:{call.gpu}"]
    return Command(args, call.env.src, call.env.environ(), call.output)


def invsr(call: EngineCall) -> Command:
    args = [_python(call), "inference_invsr.py", "-i", str(call.input), "-o", str(call.output), "--num_steps",
            str(call.model.engine_options.get("steps", 1)), "--sd_path", str(call.env.data / "sd"),
            "--chopping_size", "128"]
    return Command(args, call.env.src, call.env.environ(), call.output)


def supir(call: EngineCall) -> Command:
    o = call.model.engine_options
    sign = str(o.get("sign", "Q"))
    args = [_python(call), "nf_supir_driver.py", "--img_dir", str(call.input), "--save_dir", str(call.output),
            "--upscale", str(_scale(call, (1, 2, 4))), "--SUPIR_sign", sign,
            "--sdxl", str(_file(call, "sd_xl_base_1.0_0.9vae.safetensors")),
            "--supir", str(_file(call, f"SUPIR-v0{sign}.ckpt")), "--no_llava", "--use_tile_vae",
            "--loading_half_params", "--edm_steps", str(o.get("steps", 50))]
    return Command(args, call.env.src, call.env.environ(), call.output)


def lucidflux(call: EngineCall) -> Command:
    env = call.env.environ()
    exported = call.env.src / "weights" / "env.sh"
    if exported.is_file():  # FLUX_DEV_FLOW / FLUX_DEV_AE written by the download tool
        for line in exported.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*export\s+([A-Z0-9_]+)=(.*)", line)
            if m:
                value = m.group(2).strip().strip('"').strip("'")
                env[m.group(1)] = value if Path(value).is_absolute() else str((call.env.src / value).resolve())
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(call.env.src), env.get("PYTHONPATH")]))
    width, height = call.out_size
    longest = max(width, height)
    if longest > 2048:
        width, height = round(width * 2048 / longest), round(height * 2048 / longest)
    width, height = max(16, width // 16 * 16), max(16, height // 16 * 16)
    args = [_python(call), "scripts/inference.py", "--checkpoint", "weights/lucidflux/lucidflux.pth",
            "--control_image", str(call.input), "--output_dir", str(call.output), "--width", str(width),
            "--height", str(height), "--num_steps", str(call.model.engine_options.get("steps", 20)),
            "--swinir_pretrained", "weights/swinir.pth", "--siglip_ckpt", "weights/siglip", "--offload",
            "--device", f"cuda:{call.gpu}"]
    return Command(args, call.env.src, env, call.output)


# ---- faces ------------------------------------------------------------------------------------------------


def gfpgan(call: EngineCall) -> Command:
    version = str(call.model.engine_options.get("version", "1.4"))
    args = [_python(call), "inference_gfpgan.py", "-i", str(call.input), "-o", str(call.output), "-v", version,
            "-s", "1", "--bg_upsampler", "none", "-w", str(call.model.engine_options.get("weight", 0.5))]
    return Command(args, call.env.src, call.env.environ(), call.output, prefer=("restored_imgs",))


def restoreformerpp(call: EngineCall) -> Command:
    args = [_python(call), "inference.py", "-i", str(call.input), "-o", str(call.output), "-v", "RestoreFormer++",
            "-s", "1", "--bg_upsampler", "none", "--save"]
    return Command(args, call.env.src, call.env.environ(), call.output, prefer=("restored_imgs",))


def codeformer(call: EngineCall) -> Command:
    args = [_python(call), "inference_codeformer.py", "-i", str(call.input), "-o", str(call.output),
            "-w", str(call.model.engine_options.get("fidelity", 0.7)), "-s", "1"]
    return Command(args, call.env.src, call.env.environ(), call.output, prefer=("final_results",))


def keep(call: EngineCall) -> Command:
    args = [_python(call), "inference_keep.py", "-i", str(call.input), "-o", str(call.output), "-s", "1",
            "--max_length", "20"]
    return Command(args, call.env.src, call.env.environ(), call.output, prefer=("final_results",))


ADAPTERS: dict[str, Callable[[EngineCall], Command]] = {
    "seedvr2": seedvr2, "flashvsr": flashvsr, "dove": dove, "swiftvr": swiftvr, "nvidia-vsr": nvidia_vsr,
    "osediff": osediff, "hypir": hypir, "invsr": invsr, "supir": supir, "lucidflux": lucidflux,
    "gfpgan": gfpgan, "restoreformerpp": restoreformerpp, "codeformer": codeformer, "keep": keep,
}


def build(call: EngineCall) -> Command:
    engine_id = call.model.engine or ""
    if engine_id not in ADAPTERS:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"no adapter for engine {engine_id!r}")
    return ADAPTERS[engine_id](call)


def _natural(path: Path) -> list[object]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


def collect(command: Command, expected: int) -> list[Path]:
    """The engine's output frames in order. Engines that keep input names are matched by name; otherwise
    natural order. Several candidate folders: the preferred one, else the folder with the most images."""
    images = [p for p in command.results.rglob("*") if p.suffix.lower() in _IMAGE and p.is_file()]
    if not images:
        return []
    by_folder: dict[Path, list[Path]] = {}
    for p in images:
        by_folder.setdefault(p.parent, []).append(p)
    preferred = [f for f in by_folder if f.name in command.prefer]
    folder = preferred[0] if preferred else max(by_folder, key=lambda f: (len(by_folder[f]) >= expected,
                                                                          len(by_folder[f])))
    return sorted(by_folder[folder], key=_natural)


def progress_from(line: str) -> tuple[int, int] | None:
    """A "done/total" pair from a progress line (tqdm and most engines print one), if any."""
    matches = _FRACTION.findall(line)
    if not matches:
        return None
    done, total = (int(x) for x in matches[-1])
    return (done, total) if 0 <= done <= total and total > 0 else None


__all__ = ["ADAPTERS", "Command", "EngineCall", "build", "collect", "progress_from"]
