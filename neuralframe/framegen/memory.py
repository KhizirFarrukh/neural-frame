"""VRAM needed by RIFE, so super-resolution planning can leave room for it on a shared GPU."""

from __future__ import annotations

BYTES_PER_PIXEL = {"fp16": 420, "fp32": 760}
"""Peak activation memory per synthesised pixel at flow scale 1.0 (RIFE 4.22, measured with
benchmarks/bench_framegen.py; smaller flow scales need less)."""


def rife_reserve_bytes(height: int, width: int, precision: str) -> int:
    """Memory one RIFE call at ``height`` x ``width`` needs: activations for one timestep, the cached source
    frames and the output as float RGB, and the weights."""
    pixels = (-(-height // 64) * 64) * (-(-width // 64) * 64)
    return pixels * BYTES_PER_PIXEL.get(precision, BYTES_PER_PIXEL["fp32"]) + 3 * 3 * pixels * 4 + 64 * 2**20
