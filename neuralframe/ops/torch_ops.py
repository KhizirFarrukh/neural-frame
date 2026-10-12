"""Torch implementations of the frame ops for the GPU path (also runs on CPU).

They mirror :mod:`neuralframe.ops.reference` operation by operation — same chroma siting, same cached
kernel tables (as dense resampling matrices), same transfer-curve tables — so parity tests can compare
them directly. Tensors are batched: planes ``(N, H, W)``, RGB ``(N, 3, H, W)`` float32.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from neuralframe.colorspace import KR_KB, ChromaLocation, Matrix, chroma_siting
from neuralframe.ops import reference

# ---------------------------------------------------------------------------------------------------------
# Chroma resampling
# ---------------------------------------------------------------------------------------------------------


def _shift(x: torch.Tensor, offset: int, dim: int) -> torch.Tensor:
    """Neighbour along ``dim`` with edge replication (offset ±1)."""
    n = x.shape[dim]
    idx = torch.clamp(torch.arange(n, device=x.device) + offset, 0, n - 1)
    return x.index_select(dim, idx)


def _up2(c: torch.Tensor, n_out: int, dim: int, cosited: bool) -> torch.Tensor:
    dim = dim % c.ndim  # stack/flatten below need a non-negative axis
    nxt = _shift(c, 1, dim)
    if cosited:
        even, odd = c, 0.5 * (c + nxt)
    else:
        prev = _shift(c, -1, dim)
        even, odd = 0.75 * c + 0.25 * prev, 0.75 * c + 0.25 * nxt
    out = torch.stack([even, odd], dim=dim + 1).flatten(dim, dim + 1)
    return out.narrow(dim, 0, n_out)


def _down2(x: torch.Tensor, dim: int, cosited: bool) -> torch.Tensor:
    dim = dim % x.ndim
    n = x.shape[dim]
    m = (n + 1) // 2
    base = torch.arange(m, device=x.device) * 2
    centre = x.index_select(dim, torch.clamp(base, 0, n - 1))
    right = x.index_select(dim, torch.clamp(base + 1, 0, n - 1))
    if cosited:
        left = x.index_select(dim, torch.clamp(base - 1, 0, n - 1))
        return 0.25 * left + 0.5 * centre + 0.25 * right
    return 0.5 * (centre + right)


def upsample_chroma(
    c: torch.Tensor, height: int, width: int, log2_cw: int, log2_ch: int, location: ChromaLocation
) -> torch.Tensor:
    h_cosited, v_cosited = chroma_siting(location)
    out = c
    if log2_cw == 1:
        out = _up2(out, width, dim=-1, cosited=h_cosited)
    if log2_ch == 1:
        out = _up2(out, height, dim=-2, cosited=v_cosited)
    return out


def downsample_chroma(c: torch.Tensor, log2_cw: int, log2_ch: int, location: ChromaLocation) -> torch.Tensor:
    h_cosited, v_cosited = chroma_siting(location)
    out = c
    if log2_cw == 1:
        out = _down2(out, dim=-1, cosited=h_cosited)
    if log2_ch == 1:
        out = _down2(out, dim=-2, cosited=v_cosited)
    return out


# ---------------------------------------------------------------------------------------------------------
# YCbCr <-> RGB
# ---------------------------------------------------------------------------------------------------------


def yuv_to_rgb(
    y: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    matrix: Matrix,
    full_range: bool,
    bit_depth: int,
    chroma_location: ChromaLocation = ChromaLocation.LEFT,
) -> torch.Tensor:
    """Planes ``(N, H, W)`` / ``(N, Hc, Wc)`` (integer) → RGB ``(N, 3, H, W)`` float32 in [0, 1]."""
    height, width = y.shape[-2:]
    if matrix is Matrix.GBR:
        peak = float((1 << bit_depth) - 1)
        rgb = torch.stack([v.float(), y.float(), u.float()], dim=1) / peak
        return rgb.clamp_(0.0, 1.0)
    log2_cw = 1 if u.shape[-1] < width else 0
    log2_ch = 1 if u.shape[-2] < height else 0
    yo, ys, co, cs = reference.range_params(bit_depth, full_range)
    yf = (y.float() - yo) / ys
    cb = upsample_chroma((u.float() - co) / cs, height, width, log2_cw, log2_ch, chroma_location)
    cr = upsample_chroma((v.float() - co) / cs, height, width, log2_cw, log2_ch, chroma_location)
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    r = yf + (2.0 * (1.0 - kr)) * cr
    b = yf + (2.0 * (1.0 - kb)) * cb
    g = yf - (2.0 * kr * (1.0 - kr) / kg) * cr - (2.0 * kb * (1.0 - kb) / kg) * cb
    return torch.stack([r, g, b], dim=1).clamp_(0.0, 1.0)


def rgb_to_yuv(
    rgb: torch.Tensor,
    *,
    matrix: Matrix,
    full_range: bool,
    bit_depth: int,
    log2_cw: int,
    log2_ch: int,
    chroma_location: ChromaLocation = ChromaLocation.LEFT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RGB ``(N, 3, H, W)`` → quantised planes as int32 tensors (cast by the caller for transfer)."""
    x = rgb.float().clamp(0.0, 1.0)
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    peak = (1 << bit_depth) - 1

    def quantise(t: torch.Tensor, offset: float, scale: float) -> torch.Tensor:
        return torch.floor(t * scale + offset + 0.5).clamp_(0, peak).to(torch.int32)

    if matrix is Matrix.GBR:
        return quantise(g, 0.0, peak), quantise(b, 0.0, peak), quantise(r, 0.0, peak)
    yo, ys, co, cs = reference.range_params(bit_depth, full_range)
    kr, kb = KR_KB[matrix]
    kg = 1.0 - kr - kb
    yf = kr * r + kg * g + kb * b
    cb = downsample_chroma((b - yf) / (2.0 * (1.0 - kb)), log2_cw, log2_ch, chroma_location)
    cr = downsample_chroma((r - yf) / (2.0 * (1.0 - kr)), log2_cw, log2_ch, chroma_location)
    return quantise(yf, yo, ys), quantise(cb, co, cs), quantise(cr, co, cs)


# ---------------------------------------------------------------------------------------------------------
# Transfer / gamut
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=8)
def _luts(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    lin, enc = reference.transfer_luts()
    return torch.from_numpy(lin.copy()).to(device), torch.from_numpy(enc.copy()).to(device)


def _apply_lut(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    pos = x.clamp(0.0, 1.0) * float(lut.numel() - 1)
    i0 = torch.clamp(pos.to(torch.int64), max=lut.numel() - 2)
    frac = pos - i0.to(pos.dtype)
    lo = lut[i0]
    return lo + (lut[i0 + 1] - lo) * frac


def apply_gamut(rgb: torch.Tensor, matrix3: torch.Tensor) -> torch.Tensor:
    lin_lut, enc_lut = _luts(rgb.device)
    lin = _apply_lut(rgb, lin_lut)
    m = matrix3.to(device=rgb.device, dtype=torch.float32)
    out = torch.stack(
        [m[c, 0] * lin[:, 0] + m[c, 1] * lin[:, 1] + m[c, 2] * lin[:, 2] for c in range(3)], dim=1
    )
    return _apply_lut(out, enc_lut)


# ---------------------------------------------------------------------------------------------------------
# Resampling as dense matrix products (exactly the reference kernel tables)
# ---------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=32)
def resample_matrix(in_size: int, out_size: int, kernel: str, device: torch.device) -> torch.Tensor:
    """``(out_size, in_size)`` float32 matrix whose rows hold the reference resampling weights."""
    idx, w = reference.resample_weights(in_size, out_size, kernel)
    mat = torch.zeros((out_size, in_size), dtype=torch.float32)
    mat.scatter_add_(1, torch.from_numpy(idx.astype("int64")), torch.from_numpy(w.copy()))
    return mat.to(device)


def resize(img: torch.Tensor, out_h: int, out_w: int, kernel: str = "lanczos3") -> torch.Tensor:
    """Resize ``(N, C, H, W)`` float tensors; horizontal pass first, then vertical (like the reference)."""
    h, w = img.shape[-2:]
    out = img.float()
    if w != out_w:
        out = out @ resample_matrix(w, out_w, kernel, img.device).T
    if h != out_h:
        out = resample_matrix(h, out_h, kernel, img.device) @ out
    return out


def pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    """Replicate-pad H and W up to ``multiple``; returns the padded tensor and the padding added."""
    h, w = x.shape[-2:]
    ph = (-h) % multiple
    pw = (-w) % multiple
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, ph, pw
