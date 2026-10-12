"""Torch device ops must match the NumPy reference (run on CPU everywhere, and on CUDA when present)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from neuralframe.colorspace import ChromaLocation, Matrix, Primaries, gamut_matrix  # noqa: E402
from neuralframe.ops import reference as ref  # noqa: E402
from neuralframe.ops import torch_ops  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _rgb(h: int = 37, w: int = 63) -> np.ndarray:
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([0.5 + 0.4 * np.sin(xx / 7), 0.5 + 0.4 * np.cos(yy / 5), 0.5 + 0.3 * np.sin((xx + yy) / 9)])
    return np.clip(base + rng.normal(0, 0.01, base.shape), 0, 1).astype(np.float32)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("matrix", [Matrix.BT601, Matrix.BT709, Matrix.BT2020NC])
@pytest.mark.parametrize(("bit_depth", "full_range"), [(8, False), (8, True), (10, False)])
@pytest.mark.parametrize(("lcw", "lch"), [(1, 1), (1, 0), (0, 0)])
@pytest.mark.parametrize("loc", [ChromaLocation.LEFT, ChromaLocation.CENTER])
def test_colour_conversion_parity(
    device: str, matrix: Matrix, bit_depth: int, full_range: bool, lcw: int, lch: int, loc: ChromaLocation
) -> None:
    rgb = _rgb()
    planes = ref.rgb_to_yuv(rgb, matrix=matrix, full_range=full_range, bit_depth=bit_depth,
                            log2_cw=lcw, log2_ch=lch, chroma_location=loc)
    t_planes = [torch.from_numpy(p.astype(np.int32))[None].to(device) for p in planes]
    back_ref = ref.yuv_to_rgb(*planes, matrix=matrix, full_range=full_range, bit_depth=bit_depth, chroma_location=loc)
    back_t = torch_ops.yuv_to_rgb(*t_planes, matrix=matrix, full_range=full_range, bit_depth=bit_depth,
                                  chroma_location=loc)
    np.testing.assert_allclose(back_t[0].cpu().numpy(), back_ref, atol=2e-6)

    again = torch_ops.rgb_to_yuv(torch.from_numpy(rgb)[None].to(device), matrix=matrix, full_range=full_range,
                                 bit_depth=bit_depth, log2_cw=lcw, log2_ch=lch, chroma_location=loc)
    for a, b in zip(again, planes, strict=True):
        assert np.max(np.abs(a[0].cpu().numpy() - b.astype(np.int32))) <= 1


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(("out_h", "out_w"), [(80, 140), (20, 30), (37, 63), (148, 252)])
@pytest.mark.parametrize("kernel", ["lanczos3", "bicubic"])
def test_resize_parity(device: str, out_h: int, out_w: int, kernel: str) -> None:
    rgb = _rgb()
    expected = ref.resize(rgb, out_h, out_w, kernel)
    got = torch_ops.resize(torch.from_numpy(rgb)[None].to(device), out_h, out_w, kernel)[0].cpu().numpy()
    np.testing.assert_allclose(got, expected, atol=2e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_gamut_parity(device: str) -> None:
    rgb = _rgb()
    m = gamut_matrix(Primaries.SMPTE170M, Primaries.BT709)
    expected = ref.apply_gamut(rgb, m)
    got = torch_ops.apply_gamut(torch.from_numpy(rgb)[None].to(device), torch.from_numpy(m))[0].cpu().numpy()
    np.testing.assert_allclose(got, expected, atol=2e-6)
