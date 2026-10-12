"""Micro-benchmark of frame ops: NumPy reference vs native C++ vs torch (CPU/CUDA).

Usage: python benchmarks/bench_ops.py [--repeat 5] [--out benchmarks/results]
Writes a Markdown table and a JSON file (with hardware info) to the output directory.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neuralframe.colorspace import ChromaLocation, Matrix, Primaries, gamut_matrix
from neuralframe.monitoring import cpu_name
from neuralframe.ops import NumpyOps, native_available


def best_of(fn: Callable[[], object], repeat: int) -> float:
    fn()  # warm-up (caches, tables, thread pool)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    rgb_480 = rng.random((3, 480, 854), dtype=np.float32)
    rgb_1080 = rng.random((3, 1080, 1920), dtype=np.float32)
    rgb_x4 = rng.random((3, 1920, 3416), dtype=np.float32)  # 480p after a x4 model
    numpy_ops = NumpyOps()
    yuv_1080 = numpy_ops.rgb_to_yuv(rgb_1080, matrix=Matrix.BT709, full_range=False, bit_depth=8, log2_cw=1,
                                    log2_ch=1, chroma_location=ChromaLocation.LEFT)
    gamut = gamut_matrix(Primaries.SMPTE170M, Primaries.BT709)

    impls: dict[str, Any] = {"numpy": numpy_ops}
    if native_available():
        from neuralframe import _native
        from neuralframe.ops.native import NativeOps

        impls["native"] = NativeOps()
        threads = _native.get_threads()
    else:
        threads = 0

    cases: dict[str, Callable[[Any], object]] = {
        "yuv420p->rgb 1080p": lambda o: o.yuv_to_rgb(*yuv_1080, matrix=Matrix.BT709, full_range=False,
                                                     bit_depth=8, chroma_location=ChromaLocation.LEFT),
        "rgb->yuv420p 1080p": lambda o: o.rgb_to_yuv(rgb_1080, matrix=Matrix.BT709, full_range=False, bit_depth=8,
                                                     log2_cw=1, log2_ch=1, chroma_location=ChromaLocation.LEFT),
        "lanczos 480p->1080p": lambda o: o.resize(rgb_480, 1080, 1920, "lanczos3"),
        "lanczos 3416x1920->1080p": lambda o: o.resize(rgb_x4, 1080, 1920, "lanczos3"),
        "gamut 601->709 480p": lambda o: o.apply_gamut(rgb_480, gamut),
    }
    rows = []
    for case, fn in cases.items():
        row: dict[str, Any] = {"op": case}
        for name, impl in impls.items():
            row[name] = best_of(lambda impl=impl, fn=fn: fn(impl), args.repeat)  # type: ignore[misc]
        if "native" in row:
            row["speedup"] = row["numpy"] / row["native"]
        rows.append(row)

    info = {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "cpu": cpu_name(),
        "native_threads": threads,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    lines = [
        f"# Frame ops benchmark ({info['date']})",
        "",
        f"CPU: {info['cpu']} - native threads: {threads} - {info['platform']}",
        "",
        "| op | NumPy reference (ms) | native C++ (ms) | speed-up |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        native = f"{r['native'] * 1000:.1f}" if "native" in r else "-"
        speed = f"{r['speedup']:.1f}x" if "speedup" in r else "-"
        lines.append(f"| {r['op']} | {r['numpy'] * 1000:.1f} | {native} | {speed} |")
    text = "\n".join(lines) + "\n"
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d")
    (args.out / f"ops-{stamp}.md").write_text(text, encoding="utf-8")
    (args.out / f"ops-{stamp}.json").write_text(json.dumps({"info": info, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
