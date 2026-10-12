"""Full-reference quality metrics and temporal consistency scores.

PSNR and SSIM are computed on luma in NumPy (SSIM: Wang et al. 2004, 11x11 Gaussian window, sigma 1.5).
VMAF comes from FFmpeg's libvmaf, LPIPS from the optional ``lpips`` package. Frames are read as exact
decoded samples (raw YUV, luma plane), never through an RGB conversion.
"""

from __future__ import annotations

import math
import re
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, popen_kwargs

FloatArray = npt.NDArray[np.float32]


# ---- frame access ----------------------------------------------------------------------------------------


def iter_luma(
    tools: FFmpegTools,
    path: Path,
    *,
    size: tuple[int, int] | None = None,
    scale_flags: str = "bicubic",
    max_frames: int | None = None,
    every: int = 1,
) -> Iterator[FloatArray]:
    """Luma planes (float32, 8-bit scale) of ``path``, optionally scaled to ``size`` = (width, height) first
    (that is how the bicubic/Lanczos baselines are produced). 10-bit sources are scaled to 0..255."""
    from neuralframe.io.probe import probe

    video = probe(path, tools).require_video()
    width, height = size or (video.width, video.height)
    deep = video.bit_depth > 8
    pix_fmt = "yuv420p16le" if deep else "yuv420p"
    filters = [f"scale={width}:{height}:flags={scale_flags}"] if size else []
    args = [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
            "-fps_mode", "passthrough"]
    if filters:
        args += ["-vf", ",".join(filters)]
    if max_frames is not None:
        args += ["-frames:v", str(max_frames)]
    args += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    bps = 2 if deep else 1
    luma_bytes = width * height * bps
    frame_bytes = luma_bytes + 2 * ((width + 1) // 2) * ((height + 1) // 2) * bps
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **popen_kwargs())
    assert proc.stdout is not None
    try:
        index = 0
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if index % every == 0:
                dtype = np.uint16 if deep else np.uint8
                y = np.frombuffer(buf[:luma_bytes], dtype=dtype).reshape(height, width).astype(np.float32)
                yield y / 257.0 if deep else y
            index += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


# ---- per-frame metrics -----------------------------------------------------------------------------------


def psnr(a: FloatArray, b: FloatArray, peak: float = 255.0) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse <= 1e-12 else 10.0 * math.log10(peak * peak / mse)


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> npt.NDArray[np.float64]:
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    return k / k.sum()


def _filter(img: npt.NDArray[np.float64], kernel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Separable 'valid' convolution (no padding), as in the reference SSIM implementation."""
    from numpy.lib.stride_tricks import sliding_window_view

    rows = np.tensordot(sliding_window_view(img, kernel.size, axis=1), kernel, axes=([2], [0]))
    out: npt.NDArray[np.float64] = np.tensordot(sliding_window_view(rows, kernel.size, axis=0), kernel,
                                                axes=([2], [0]))
    return out


def ssim(a: FloatArray, b: FloatArray, peak: float = 255.0) -> float:
    """Mean SSIM of two single-channel images."""
    x, y = a.astype(np.float64), b.astype(np.float64)
    k = _gaussian_kernel()
    c1, c2 = (0.01 * peak) ** 2, (0.03 * peak) ** 2
    mu_x, mu_y = _filter(x, k), _filter(y, k)
    sxx = _filter(x * x, k) - mu_x * mu_x
    syy = _filter(y * y, k) - mu_y * mu_y
    sxy = _filter(x * y, k) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def mean_abs_diff(a: FloatArray, b: FloatArray) -> float:
    return float(np.mean(np.abs(a - b)))


# ---- video-level ---------------------------------------------------------------------------------------


@dataclass
class FidelityScores:
    frames: int
    psnr_y: float
    ssim_y: float
    vmaf: float | None = None
    lpips: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"frames": self.frames, "psnr_y": round(self.psnr_y, 3), "ssim_y": round(self.ssim_y, 5),
                "vmaf": None if self.vmaf is None else round(self.vmaf, 3),
                "lpips": None if self.lpips is None else round(self.lpips, 4)}


def compare_videos(
    tools: FFmpegTools,
    distorted: Path,
    reference: Path,
    *,
    distorted_size: tuple[int, int] | None = None,
    scale_flags: str = "bicubic",
    every: int = 1,
    max_frames: int | None = None,
) -> FidelityScores:
    """PSNR-Y / SSIM-Y of ``distorted`` against ``reference`` frame by frame (both must have the same frame
    count and timing). ``distorted_size`` scales the distorted clip first (for interpolation baselines).

    PSNR is computed from the MSE pooled over all frames (FFmpeg's convention): averaging per-frame PSNR
    lets a few identical frames (black, titles: infinite PSNR) dominate the result."""
    ref_iter = iter_luma(tools, reference, every=every, max_frames=max_frames)
    dist_iter = iter_luma(tools, distorted, size=distorted_size, scale_flags=scale_flags, every=every,
                          max_frames=max_frames)
    mse, s, n = 0.0, 0.0, 0
    for ref, dist in zip(ref_iter, dist_iter, strict=False):
        if ref.shape != dist.shape:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"frame sizes differ: {dist.shape} vs {ref.shape}")
        mse += float(np.mean((dist.astype(np.float64) - ref.astype(np.float64)) ** 2))
        s += ssim(dist, ref)
        n += 1
    if n == 0:
        raise NeuralFrameError(ErrorCode.DECODE_FAILED, "no frames to compare")
    mean_mse = mse / n
    pooled = 100.0 if mean_mse <= 1e-12 else 10.0 * math.log10(255.0**2 / mean_mse)
    return FidelityScores(frames=n, psnr_y=pooled, ssim_y=s / n)


_VMAF = re.compile(r"VMAF score[:=]\s*([0-9.]+)")


def vmaf(
    tools: FFmpegTools,
    distorted: Path,
    reference: Path,
    *,
    distorted_size: tuple[int, int] | None = None,
    scale_flags: str = "bicubic",
    threads: int = 8,
    subsample: int = 1,
) -> float:
    """Mean VMAF (libvmaf default model) of ``distorted`` against ``reference``."""
    scale = f"scale={distorted_size[0]}:{distorted_size[1]}:flags={scale_flags}," if distorted_size else ""
    graph = (f"[0:v]{scale}setpts=PTS-STARTPTS,format=yuv420p[d];[1:v]setpts=PTS-STARTPTS,format=yuv420p[r];"
             f"[d][r]libvmaf=n_threads={threads}:n_subsample={subsample}")
    args = [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info", "-i", str(distorted), "-i",
            str(reference), "-lavfi", graph, "-f", "null", "-"]
    proc = subprocess.run(args, capture_output=True, timeout=3600, check=False, **popen_kwargs())
    text = proc.stderr.decode(errors="replace")
    match = _VMAF.search(text)
    if proc.returncode != 0 or match is None:
        raise NeuralFrameError(ErrorCode.TOOL_MISSING, "libvmaf failed: " + text.strip()[-400:],
                               hint="needs an FFmpeg build with libvmaf")
    return float(match.group(1))


class LpipsScorer:
    """LPIPS (AlexNet) on RGB frames; needs the optional ``lpips`` package (pip install neuralframe[lpips])."""

    def __init__(self, device: str = "cuda") -> None:
        import warnings

        import torch

        with warnings.catch_warnings():  # lpips still uses torchvision's deprecated `pretrained=` argument
            warnings.simplefilter("ignore", UserWarning)
            import lpips  # type: ignore[import-not-found,unused-ignore]

            self.torch = torch
            self.device = torch.device(device if torch.cuda.is_available() else "cpu")
            self.net = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    @staticmethod
    def available() -> bool:
        import importlib.util

        return importlib.util.find_spec("lpips") is not None and importlib.util.find_spec("torch") is not None

    def score(self, a_rgb: npt.NDArray[np.uint8], b_rgb: npt.NDArray[np.uint8]) -> float:
        torch = self.torch

        def tensor(img: npt.NDArray[np.uint8]) -> Any:
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float() / 127.5 - 1.0  # writable copy
            return t.to(self.device)

        with torch.inference_mode():
            return float(self.net(tensor(a_rgb), tensor(b_rgb)).item())


def iter_rgb(tools: FFmpegTools, path: Path, *, size: tuple[int, int] | None = None, scale_flags: str = "bicubic",
             every: int = 1, max_frames: int | None = None) -> Iterator[npt.NDArray[np.uint8]]:
    """RGB24 frames (for perceptual metrics), scaled to ``size`` first when given."""
    from neuralframe.io.probe import probe

    video = probe(path, tools).require_video()
    width, height = size or (video.width, video.height)
    vf = [f"scale={width}:{height}:flags={scale_flags}"] if size else []
    args = [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), "-map", "0:v:0",
            "-fps_mode", "passthrough"]
    if vf:
        args += ["-vf", ",".join(vf)]
    if max_frames is not None:
        args += ["-frames:v", str(max_frames)]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    frame_bytes = width * height * 3
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **popen_kwargs())
    assert proc.stdout is not None
    try:
        index = 0
        while len(buf := proc.stdout.read(frame_bytes)) == frame_bytes:
            if index % every == 0:
                yield np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
            index += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def lpips_videos(tools: FFmpegTools, distorted: Path, reference: Path, *, distorted_size: tuple[int, int] | None
                 = None, scale_flags: str = "bicubic", every: int = 8, scorer: LpipsScorer | None = None) -> float:
    scorer = scorer or LpipsScorer()
    values = [scorer.score(d, r) for d, r in zip(
        iter_rgb(tools, distorted, size=distorted_size, scale_flags=scale_flags, every=every),
        iter_rgb(tools, reference, every=every), strict=False)]
    return float(np.mean(values)) if values else float("nan")


# ---- temporal consistency ------------------------------------------------------------------------------


def frame_differences(frames: Iterator[FloatArray]) -> list[float]:
    """Mean absolute luma difference between each frame and the previous one (index 0 = 0.0)."""
    diffs: list[float] = []
    prev: FloatArray | None = None
    for frame in frames:
        diffs.append(0.0 if prev is None else mean_abs_diff(frame, prev))
        prev = frame
    return diffs


def seam_ratio(diffs: list[float], boundaries: list[int], *, exclude: list[int] | None = None) -> float | None:
    """Mean frame-to-frame change at chunk boundaries divided by the mean change elsewhere.

    ~1.0 means chunk boundaries are invisible; the spec flags > 1.2. Scene cuts are excluded on both sides.
    """
    skip = set(exclude or [])
    at = [diffs[b] for b in boundaries if 0 < b < len(diffs) and b not in skip]
    rest = [d for i, d in enumerate(diffs) if i > 0 and i not in skip and i not in set(boundaries)]
    if not at or not rest or float(np.mean(rest)) <= 1e-9:
        return None
    return float(np.mean(at) / np.mean(rest))


def flicker(output: Iterator[FloatArray], reference: Iterator[FloatArray], *, static_threshold: float = 1.0) -> float:
    """Temporal noise added in static regions: mean |frame-to-frame change| of the output where the reference
    barely changes (< ``static_threshold`` levels). Lower is steadier; compare against a baseline."""
    total, count = 0.0, 0
    prev_out: FloatArray | None = None
    prev_ref: FloatArray | None = None
    for out, ref in zip(output, reference, strict=False):
        if prev_out is not None and prev_ref is not None:
            static = np.abs(ref - prev_ref) < static_threshold
            if static.any():
                total += float(np.abs(out - prev_out)[static].mean())
                count += 1
        prev_out, prev_ref = out, ref
    return total / count if count else 0.0
