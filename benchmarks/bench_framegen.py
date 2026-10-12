"""RIFE frame-interpolation throughput per resolution, precision and flow scale (CUDA if available).

Usage: python benchmarks/bench_framegen.py [--repeat 10] [--out benchmarks/results]
Measures synthesised frames per second for one source pair with 1 and 4 timesteps per call (4 timesteps is
what 24 -> 120 fps needs). Writes a Markdown table and a JSON file (with hardware info).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from neuralframe.config import default_cache_dir
from neuralframe.framegen.interpolate import Interpolator, auto_rife_scale
from neuralframe.models.download import ensure_weights
from neuralframe.models.registry import ModelRegistry
from neuralframe.monitoring import cpu_name

SIZES = {"720p": (720, 1280), "1080p": (1080, 1920), "1440p": (1440, 2560), "4K": (2160, 3840)}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--model", default="rife-4.22")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ModelRegistry.load().get(args.model)
    weights = ensure_weights(spec, default_cache_dir() / "models")
    precisions = ["fp16", "fp32"] if device.type == "cuda" else ["fp32"]
    rows: list[dict[str, Any]] = []
    for precision in precisions:
        for name, (h, w) in SIZES.items():
            if device.type == "cpu" and h > 1080:
                continue
            scales = sorted({auto_rife_scale(h, w), 1.0, 0.5})
            for scale in scales:
                interp = Interpolator(spec, weights, device, precision, scale=scale)
                a = torch.rand(3, h, w, device=device)
                b = torch.rand(3, h, w, device=device)
                for taus in ([0.5], [0.2, 0.4, 0.6, 0.8]):
                    try:
                        interp.interpolate(a, b, taus)
                        sync(device)
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        t0 = time.perf_counter()
                        for _ in range(args.repeat):
                            interp.interpolate(a, b, taus)
                        sync(device)
                        dt = (time.perf_counter() - t0) / args.repeat
                        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
                        rows.append({"size": name, "precision": precision, "scale": scale,
                                     "auto_scale": scale == auto_rife_scale(h, w), "timesteps": len(taus),
                                     "ms_per_call": round(dt * 1000, 2), "frames_per_s": round(len(taus) / dt, 1),
                                     "peak_mb": round(peak)})
                    except Exception as exc:  # report OOM etc. per configuration
                        rows.append({"size": name, "precision": precision, "scale": scale,
                                     "timesteps": len(taus), "error": type(exc).__name__})
                    print(rows[-1], flush=True)
                interp.close()
                del interp, a, b

    hardware = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else cpu_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cpu": cpu_name(), "python": platform.python_version(),
    }
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"framegen-{stamp}.json").write_text(
        json.dumps({"hardware": hardware, "model": args.model, "results": rows}, indent=2), encoding="utf-8"
    )
    lines = [f"# Frame generation ({args.model}) - {hardware['device']}", "",
             f"torch {hardware['torch']}, CUDA {hardware['cuda']}, {args.repeat} calls per row; "
             "`*` = scale chosen automatically for that size.", "",
             "| size | precision | flow scale | timesteps/call | ms/call | frames/s | peak MB |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['size']} | {r['precision']} | {r['scale']} | {r['timesteps']} | {r['error']} | | |")
            continue
        mark = "*" if r["auto_scale"] else ""
        lines.append(f"| {r['size']} | {r['precision']} | {r['scale']}{mark} | {r['timesteps']} | "
                     f"{r['ms_per_call']} | {r['frames_per_s']} | {r['peak_mb']} |")
    (args.out / f"framegen-{stamp}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
