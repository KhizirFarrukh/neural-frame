"""Device-resident frame processing: raw YUV in → raw YUV out, everything else on one torch device.

On CUDA this keeps colour conversion, the network, the final resize and RGB→YUV on the GPU; only compact
YUV planes cross PCIe (≈1.5 bytes/pixel each way instead of 12 bytes/pixel of float RGB).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from neuralframe.colorspace import ColorPlan, ColorRange
from neuralframe.inference.runner import ModelRunner
from neuralframe.io.rawvideo import RawArray, RawVideoFormat
from neuralframe.ops import torch_ops


@dataclass
class DeviceTimings:
    upload_s: float = 0.0
    preprocess_s: float = 0.0
    inference_s: float = 0.0
    postprocess_s: float = 0.0
    download_s: float = 0.0
    frames: int = 0
    extra: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "upload_s": round(self.upload_s, 3),
            "preprocess_s": round(self.preprocess_s, 3),
            "inference_s": round(self.inference_s, 3),
            "postprocess_s": round(self.postprocess_s, 3),
            "download_s": round(self.download_s, 3),
            "frames": self.frames,
        }


def _planes_to_tensor(frames: list[RawArray], fmt: RawVideoFormat, device: torch.device) -> list[torch.Tensor]:
    """Stack N flat frame buffers and return device tensors for the three planes (N, h, w)."""
    stacked = np.stack(frames)
    if fmt.bit_depth == 16:
        host = torch.from_numpy(stacked.astype(np.int32))
    elif fmt.bit_depth > 8:
        host = torch.from_numpy(stacked.view(np.int16))  # values < 2**12 are identical as int16
    else:
        host = torch.from_numpy(stacked)
    if device.type == "cuda":
        host = host.pin_memory()
    flat = host.to(device, non_blocking=True)
    (yh, yw), (ch, cw), _ = fmt.plane_shapes
    n = flat.shape[0]
    y_end = yh * yw
    c_size = ch * cw
    return [
        flat[:, :y_end].reshape(n, yh, yw),
        flat[:, y_end : y_end + c_size].reshape(n, ch, cw),
        flat[:, y_end + c_size :].reshape(n, ch, cw),
    ]


class DeviceProcessor:
    """Callable used by the pipeline's inference stage in device mode."""

    def __init__(
        self,
        *,
        device: torch.device,
        in_fmt: RawVideoFormat,
        out_fmt: RawVideoFormat,
        color: ColorPlan,
        runner: ModelRunner | None,
        passes: int,
        kernel: str,
        frames_per_call: int = 1,
    ) -> None:
        self.device = device
        self.in_fmt = in_fmt
        self.out_fmt = out_fmt
        self.color = color
        self.runner = runner
        self.passes = passes if runner is not None else 0
        self.kernel = kernel
        self.batch_size = max(1, frames_per_call)
        self.timings = DeviceTimings()
        gamut = color.gamut()
        self._gamut = torch.from_numpy(gamut).to(device) if gamut is not None else None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def to_rgb(self, frames: list[RawArray]) -> torch.Tensor:
        y, u, v = _planes_to_tensor(frames, self.in_fmt, self.device)
        rgb = torch_ops.yuv_to_rgb(
            y, u, v,
            matrix=self.color.in_matrix,
            full_range=self.color.in_range is ColorRange.FULL,
            bit_depth=self.in_fmt.bit_depth,
            chroma_location=self.color.in_chroma,
        )
        if self._gamut is not None:
            rgb = torch_ops.apply_gamut(rgb, self._gamut)
        return rgb

    def from_rgb(self, rgb: torch.Tensor) -> list[RawArray]:
        fmt = self.out_fmt
        if rgb.shape[-2:] != (fmt.height, fmt.width):
            rgb = torch_ops.resize(rgb, fmt.height, fmt.width, self.kernel)
        y, u, v = torch_ops.rgb_to_yuv(
            rgb,
            matrix=self.color.out_matrix,
            full_range=self.color.out_range is ColorRange.FULL,
            bit_depth=fmt.bit_depth,
            log2_cw=fmt.log2_cw,
            log2_ch=fmt.log2_ch,
            chroma_location=self.color.out_chroma,
        )
        n = rgb.shape[0]
        transfer_dtype = torch.uint8 if fmt.bit_depth == 8 else torch.int16
        packed = torch.cat([p.reshape(n, -1) for p in (y, u, v)], dim=1).to(transfer_dtype)
        host = packed.cpu().numpy()
        if fmt.bit_depth > 8:
            host = host.view(np.uint16)
        return [host[i] for i in range(n)]

    def stats(self) -> dict[str, object]:
        data: dict[str, object] = {"device": str(self.device), "timings": self.timings.as_dict()}
        if self.runner is not None:
            data["runner"] = {
                "frames": self.runner.stats.frames,
                "tiles": self.runner.stats.tiles,
                "infer_calls": self.runner.stats.infer_calls,
                "layouts": self.runner.stats.layouts,
                **self.runner.backend.describe(),
            }
        return data

    def close(self) -> None:
        if self.runner is not None:
            self.runner.backend.close()

    def process(self, frames: list[RawArray]) -> list[RawArray]:
        return self(frames)

    def __call__(self, frames: list[RawArray]) -> list[RawArray]:
        t0 = time.perf_counter()
        rgb = self.to_rgb(frames)
        self._sync()
        t1 = time.perf_counter()
        for _ in range(self.passes):
            assert self.runner is not None
            rgb = self.runner.run(rgb)
        self._sync()
        t2 = time.perf_counter()
        out = self.from_rgb(rgb)
        t3 = time.perf_counter()
        self.timings.preprocess_s += t1 - t0
        self.timings.inference_s += t2 - t1
        self.timings.postprocess_s += t3 - t2
        self.timings.frames += len(frames)
        return out
