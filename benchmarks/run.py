"""Benchmark matrix: model x backend x input resolution x output resolution -> fps, VRAM, GPU util, stages.

Usage: python benchmarks/run.py [--quick] [--only SUBSTRING] [--out benchmarks/results]
Each configuration is a full job (decode -> inference -> encode, audio/subtitles muxed) on a generated
29.97 fps clip, so the numbers are end-to-end throughput, not model-only timings. One-time TensorRT
engine builds are excluded from fps (reported separately). Results go to benchmarks/results/ as Markdown
and JSON together with hardware information.
"""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neuralframe.config import load_settings
from neuralframe.quality.benchmark import BenchConfig, hardware_info, markdown_table, run_benchmark

MATRIX = [
    # The spec's reference shape: 480p -> 1080p.
    BenchConfig("lanczos", "480p", "1080p", 300),
    BenchConfig("realesr-animevideov3", "480p", "1080p", 300, backend="tensorrt"),
    BenchConfig("realesr-animevideov3", "480p", "1080p", 300, backend="onnxruntime"),
    BenchConfig("realesr-animevideov3", "480p", "1080p", 300, backend="torch"),
    BenchConfig("realesrgan-x2plus", "480p", "1080p", 120, backend="tensorrt"),
    BenchConfig("realesrgan-x2plus", "480p", "1080p", 120, backend="torch"),
    BenchConfig("realesrgan-x4plus", "480p", "1080p", 60, backend="tensorrt"),
    BenchConfig("realesrgan-x4plus", "480p", "1080p", 60, backend="onnxruntime"),
    BenchConfig("realesrgan-x4plus", "480p", "1080p", 60, backend="torch"),
    BenchConfig("realbasicvsr-x4", "480p", "1080p", 120, backend="torch"),
    # Frame generation on top (29.97 -> 59.94 fps).
    BenchConfig("lanczos", "480p", "1080p", 300, fps="60"),
    BenchConfig("realesr-animevideov3", "480p", "1080p", 300, backend="tensorrt", fps="60"),
    # 4K output (the product default).
    BenchConfig("lanczos", "720p", "2160p", 300),
    BenchConfig("realesr-animevideov3", "720p", "2160p", 120, backend="tensorrt"),
    BenchConfig("realesrgan-x4plus", "720p", "2160p", 24, backend="tensorrt"),
    BenchConfig("lanczos", "1080p", "2160p", 300),
    BenchConfig("realesr-animevideov3", "1080p", "2160p", 120, backend="tensorrt"),
    BenchConfig("realesrgan-x2plus", "1080p", "2160p", 24, backend="tensorrt"),
]

QUICK = [MATRIX[0], MATRIX[1], MATRIX[6], MATRIX[11]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="four representative configurations")
    parser.add_argument("--only", help="run configurations whose label contains this text")
    parser.add_argument("--cpu", action="store_true", help="add CPU-only rows (no GPU)")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    configs = QUICK if args.quick else MATRIX
    if args.only:
        configs = [c for c in configs if args.only in c.label]
    settings = load_settings(args.config)
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        print(f"-> {cfg.label} ({cfg.frames} frames)", flush=True)
        try:
            row = run_benchmark(settings, cfg)
        except Exception as exc:  # one failing configuration must not lose the others
            traceback.print_exc()
            row = {"config": cfg.label, "model": cfg.model, "backend": cfg.backend or "auto", "error": str(exc)}
        print(f"   {json.dumps(row)}", flush=True)
        rows.append(row)
    if args.cpu:
        cpu = load_settings(args.config)
        cpu.inference.device = "cpu"
        for cfg in (BenchConfig("lanczos", "480p", "1080p", 120),
                    BenchConfig("realesr-animevideov3", "480p", "1080p", 24, backend="torch")):
            row = run_benchmark(cpu, cfg)
            row["backend"] = f"{row['backend']} (CPU)"
            print(f"   {json.dumps(row)}", flush=True)
            rows.append(row)

    hardware = hardware_info()
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"bench-{stamp}.json").write_text(
        json.dumps({"hardware": hardware, "results": rows}, indent=2, default=str), encoding="utf-8"
    )
    hw = ", ".join(f"{k}: {v}" for k, v in hardware.items() if v)
    text = "\n".join([f"# End-to-end throughput - {hardware.get('gpu', hardware['cpu'])}", "", hw, "",
                      "Source clips: generated, 29.97 fps H.264 + AAC; fps = output frames per second through the "
                      "whole pipeline (engine builds excluded).", "", markdown_table(rows), ""])
    (args.out / f"bench-{stamp}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
