"""RIFE interpolation and the retiming pipeline stage."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuralframe.colorspace import ColorPlan, ColorRange
from neuralframe.core.runtime import CancelToken, Frame, Stage, StageQueue
from neuralframe.framegen.memory import BYTES_PER_PIXEL
from neuralframe.framegen.retime import INTERP, RetimePlan
from neuralframe.inference.archs import load_network
from neuralframe.inference.backend import InferenceOOM
from neuralframe.io.rawvideo import RawArray, RawVideoFormat
from neuralframe.log import get_logger
from neuralframe.models.registry import ModelSpec
from neuralframe.ops import torch_ops

log = get_logger(component="framegen")


def auto_rife_scale(height: int, width: int) -> float:
    """Flow-estimation scale recommended by Practical-RIFE: 1.0 up to 1080p, 0.5 for 4K-class frames."""
    edge = max(height, width)
    if edge > 4096:
        return 0.25
    if edge > 2048:
        return 0.5
    return 1.0




class Interpolator:
    """RIFE on one device: ``(rgb0, rgb1, taus)`` → one RGB frame per timestep (float32, [0, 1]).

    Timesteps of one source pair are batched, as many per call as free VRAM allows: overcommitting either
    fails or, on Windows, silently spills into shared system memory and runs several times slower.
    """

    def __init__(self, spec: ModelSpec, weights: Path, device: torch.device, precision: str = "fp16",
                 scale: float | None = None) -> None:
        net = load_network(spec, weights).to(device)
        self.dtype = torch.float16 if precision == "fp16" and device.type == "cuda" else torch.float32
        if self.dtype == torch.float16:
            net = net.half()
        self.net: Any = net
        self.device = device
        self.fixed_scale = scale
        self.frames = 0
        self.seconds = 0.0
        self.batch_cap: int | None = None
        """Largest timestep batch known to fit (lowered on out-of-memory)."""
        self.oom_retries = 0

    def _fits(self, pixels: int) -> int:
        """How many timesteps of ``pixels`` each fit in free VRAM right now (at least 1)."""
        if self.device.type != "cuda":
            return 1 << 30
        free, _ = torch.cuda.mem_get_info(self.device)
        cached = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        per_item = pixels * BYTES_PER_PIXEL["fp16" if self.dtype == torch.float16 else "fp32"]
        return max(1, int(0.85 * (free + cached)) // per_item)

    def interpolate(self, rgb0: torch.Tensor, rgb1: torch.Tensor, taus: list[float]) -> torch.Tensor:
        """``rgb0``/``rgb1`` (3, H, W) → (len(taus), 3, H, W)."""
        t0 = time.perf_counter()
        _, h, w = rgb0.shape
        self.net.scale = self.fixed_scale or auto_rife_scale(h, w)
        m = self.net.multiple()
        a = torch_ops.pad_to_multiple(rgb0[None], m)[0].to(self.dtype)
        b = torch_ops.pad_to_multiple(rgb1[None], m)[0].to(self.dtype)
        outs: list[torch.Tensor] = []
        done = 0
        while done < len(taus):
            size = min(len(taus) - done, self._fits(a.shape[-2] * a.shape[-1]), self.batch_cap or 1 << 30)
            t = torch.tensor(taus[done : done + size], device=self.device, dtype=self.dtype).view(size, 1, 1, 1)
            try:
                with torch.inference_mode():
                    res = self.net(a.expand(size, -1, -1, -1), b.expand(size, -1, -1, -1), t)
                    outs.append(res[:, :, :h, :w].float().clamp_(0.0, 1.0))
            except torch.OutOfMemoryError as exc:
                if size == 1:
                    raise InferenceOOM(str(exc)) from exc
                self.batch_cap = size // 2
                self.oom_retries += 1
                log.warning("frame interpolation out of memory; smaller batches", batch=self.batch_cap)
                torch.cuda.empty_cache()
                continue
            done += size
        self.frames += len(taus)
        self.seconds += time.perf_counter() - t0
        return outs[0] if len(outs) == 1 else torch.cat(outs)

    def close(self) -> None:
        self.net = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class RetimeStage(Stage):
    """Source frames (raw YUV, in order) → constant-rate output frames (raw YUV, output indices).

    Exact-copy frames are passed through untouched (bit-exact); interpolated frames are converted to RGB on
    the device, synthesised by RIFE for all timesteps of a source pair in one batch, and converted back.
    """

    def __init__(
        self,
        cancel: CancelToken,
        inq: StageQueue[Frame[RawArray]],
        out: StageQueue[Frame[RawArray]],
        *,
        plan: RetimePlan,
        interpolator: Interpolator | None,
        fmt: RawVideoFormat,
        color: ColorPlan,
        skip_output: Any = None,
        limiter: Any = None,
    ) -> None:
        super().__init__("framegen", cancel)
        self._limiter = limiter
        self._in, self._out = inq, out
        self.plan = plan
        self.interpolator = interpolator
        self.fmt = fmt
        self.color = color
        self.skip_output = skip_output
        self._buffer: dict[int, Frame[RawArray]] = {}
        self._rgb_cache: dict[int, torch.Tensor] = {}
        self._eos = False
        self._last = -1
        self.interpolated = 0
        self.copied = 0

    def _need(self, index: int) -> Frame[RawArray] | None:
        # Arrival order may differ from index order when several workers upscale in parallel.
        while not self._eos and index not in self._buffer:
            frame = self._in.get()
            if frame is None:
                self._eos = True
                break
            self._buffer[frame.index] = frame
            self._last = max(self._last, frame.index)
        return self._buffer.get(index)

    def _rgb(self, index: int, frame: Frame[RawArray]) -> torch.Tensor:
        if index not in self._rgb_cache:
            assert self.interpolator is not None
            fmt = self.fmt
            host = torch.from_numpy(frame.data.view(np.int16) if fmt.bit_depth > 8 else frame.data)
            flat = host.to(self.interpolator.device)
            if fmt.bit_depth == 16:
                flat = flat.to(torch.int32) & 0xFFFF
            (yh, yw), (ch, cw), _ = fmt.plane_shapes
            y_end, c_size = yh * yw, ch * cw
            self._rgb_cache[index] = torch_ops.yuv_to_rgb(
                flat[:y_end].reshape(1, yh, yw), flat[y_end : y_end + c_size].reshape(1, ch, cw),
                flat[y_end + c_size :].reshape(1, ch, cw), matrix=self.color.out_matrix,
                full_range=self.color.out_range is ColorRange.FULL, bit_depth=fmt.bit_depth,
                chroma_location=self.color.out_chroma,
            )[0]
        return self._rgb_cache[index]

    def _to_raw(self, rgb: torch.Tensor) -> RawArray:
        fmt = self.fmt
        y, u, v = torch_ops.rgb_to_yuv(
            rgb[None], matrix=self.color.out_matrix, full_range=self.color.out_range is ColorRange.FULL,
            bit_depth=fmt.bit_depth, log2_cw=fmt.log2_cw, log2_ch=fmt.log2_ch, chroma_location=self.color.out_chroma,
        )
        dtype = torch.uint8 if fmt.bit_depth == 8 else torch.int16
        packed = torch.cat([p.reshape(-1) for p in (y, u, v)]).to(dtype).cpu().numpy()
        return packed.view(np.uint16) if fmt.bit_depth > 8 else packed

    def _put(self, k: int, data: RawArray) -> None:
        plan = self.plan
        self._out.put(Frame(k, int(plan.pts[k]), plan.step, data))
        self.metrics.frames += 1

    def run(self) -> None:
        plan = self.plan
        k = 0
        total = plan.frames
        while k < total:
            a, b = int(plan.src_a[k]), int(plan.src_b[k])
            group_end = k + 1
            if plan.kind[k] == INTERP:
                while (group_end < total and plan.kind[group_end] == INTERP
                       and plan.src_a[group_end] == a and plan.src_b[group_end] == b):
                    group_end += 1
            todo = [i for i in range(k, group_end) if not (self.skip_output and self.skip_output(i))]
            if todo:
                fa = self._need(a)
                if fa is None:
                    break  # the source ended early: stop at the last frame we can produce
                with self.timed():
                    if plan.kind[k] == INTERP and self.interpolator is not None:
                        fb = self._need(b)
                        if fb is None:
                            break
                        outs = self.interpolator.interpolate(
                            self._rgb(a, fa), self._rgb(b, fb), [float(plan.tau[i]) for i in todo]
                        )
                        for j, i in enumerate(todo):
                            self._put(i, self._to_raw(outs[j]))
                        self.interpolated += len(todo)
                    else:
                        for i in todo:  # exact copy / nearest frame: pass the samples through unchanged
                            self._put(i, fa.data)
                        self.copied += len(todo)
            k = group_end
            # Release sources no later output frame needs.
            keep_from = int(plan.src_a[k]) if k < total else 10**12
            for idx in [i for i in self._buffer if i < keep_from]:
                del self._buffer[idx]
                self._rgb_cache.pop(idx, None)
                if self._limiter is not None:
                    self._limiter.release()
        while self._in.get() is not None:
            pass
        self._out.close()

    def stats(self) -> dict[str, Any]:
        data: dict[str, Any] = {"output_frames": self.metrics.frames, "interpolated": self.interpolated,
                                "copied": self.copied}
        if self.interpolator is not None:
            data["rife_seconds"] = round(self.interpolator.seconds, 3)
            data["oom_retries"] = self.interpolator.oom_retries
        return data
