from __future__ import annotations

import math

import numpy as np
import pytest

from neuralframe.quality.metrics import flicker, frame_differences, psnr, seam_ratio, ssim


def test_psnr_known_values() -> None:
    a = np.zeros((8, 8), np.float32)
    assert psnr(a, a) == 100.0
    b = a + 5.0  # MSE 25
    assert psnr(a, b) == pytest.approx(10 * math.log10(255**2 / 25))


def _ssim_torch(a: np.ndarray, b: np.ndarray) -> float:
    torch = pytest.importorskip("torch")
    f = torch.nn.functional
    x = torch.from_numpy(a.astype(np.float64))[None, None]
    y = torch.from_numpy(b.astype(np.float64))[None, None]
    g = torch.exp(-((torch.arange(11, dtype=torch.float64) - 5) ** 2) / (2 * 1.5**2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    mu_x, mu_y = f.conv2d(x, w), f.conv2d(y, w)
    sxx = f.conv2d(x * x, w) - mu_x**2
    syy = f.conv2d(y * y, w) - mu_y**2
    sxy = f.conv2d(x * y, w) - mu_x * mu_y
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return float(m.mean())


def test_ssim_matches_an_independent_implementation() -> None:
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 255, (48, 64)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 12, a.shape), 0, 255).astype(np.float32)
    assert ssim(a, a) == pytest.approx(1.0)
    assert ssim(a, b) == pytest.approx(_ssim_torch(a, b), abs=1e-9)
    c = np.clip(a + rng.normal(0, 60, a.shape), 0, 255).astype(np.float32)
    assert 0.0 < ssim(a, c) < ssim(a, b) < 1.0  # more noise, less similarity


def test_seam_ratio() -> None:
    diffs = [0.0] + [1.0] * 99
    assert seam_ratio(diffs, [24, 48, 72]) == pytest.approx(1.0)
    diffs[48] = 3.0  # a visible jump at one boundary
    assert seam_ratio(diffs, [24, 48, 72]) == pytest.approx(5 / 3)
    diffs[60] = 50.0  # a scene cut is not a seam
    assert seam_ratio(diffs, [24, 48, 72], exclude=[60]) == pytest.approx(5 / 3)
    assert seam_ratio(diffs, []) is None


def test_flicker_measures_noise_in_static_regions() -> None:
    rng = np.random.default_rng(0)
    ref = [np.full((16, 16), 100.0, np.float32) for _ in range(10)]
    steady = [r.copy() for r in ref]
    noisy = [r + rng.normal(0, 2, r.shape).astype(np.float32) for r in ref]
    assert flicker(iter(steady), iter(ref)) == 0.0
    assert flicker(iter(noisy), iter(ref)) > 1.5
    assert frame_differences(iter(steady)) == [0.0] * 10
