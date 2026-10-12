"""Three-stage GPU pipeline with dedicated CUDA streams: upload → inference → download.

Each stage owns a thread and a CUDA stream. Batches move between stages with CUDA events, so host↔device
copies and colour/resize work of neighbouring batches run concurrently with the network instead of
serialising around it. Host staging buffers are page-locked slots of the native ring buffer (or pinned
torch tensors when the native module is unavailable); they are allocated once and reused.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from neuralframe.colorspace import ColorPlan, ColorRange
from neuralframe.core.runtime import CancelToken, Frame, Stage, StageQueue
from neuralframe.inference.runner import ModelRunner
from neuralframe.io.rawvideo import RawArray, RawVideoFormat
from neuralframe.log import get_logger
from neuralframe.ops import native_available, torch_ops

log = get_logger(component="gpu-stages")


class PinnedPool:
    """Fixed pool of page-locked host buffers; FIFO reuse with CUDA-event retirement."""

    def __init__(self, slots: int, slot_bytes: int) -> None:
        self.slot_bytes = slot_bytes
        self._ring: Any = None
        self._registered = False
        self._buffers: list[torch.Tensor] = []
        if native_available():
            from neuralframe import _native

            self._ring = _native.RingBuffer(slots, slot_bytes)
            # Page-lock the whole native block once so async copies can DMA straight from it.
            rc = torch.cuda.cudart().cudaHostRegister(self._ring.data_ptr, self._ring.nbytes, 0)
            self._registered = int(rc) == 0
            if not self._registered:
                log.warning("cudaHostRegister failed; host staging is pageable", code=int(rc))
        else:
            self._buffers = [torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(slots)]
            self._free: collections.deque[int] = collections.deque(range(slots))
        self._pending: collections.deque[tuple[int, torch.cuda.Event]] = collections.deque()

    @property
    def native(self) -> bool:
        return self._ring is not None

    def acquire(self) -> tuple[int, np.ndarray]:
        """A free slot (waiting for in-flight copies to finish if necessary) and its uint8 view."""
        while True:
            self.retire(block=False)
            if self._ring is not None:
                slot = self._ring.acquire_write(0)
                if slot >= 0:
                    return slot, self._ring.view(slot)
            elif self._free:
                slot = self._free.popleft()
                return slot, self._buffers[slot].numpy()
            self.retire(block=True)

    def submit(self, slot: int, event: torch.cuda.Event) -> None:
        """The slot is in use by an async copy until ``event`` completes."""
        if self._ring is not None:
            self._ring.commit_write(slot, 0)
        self._pending.append((slot, event))

    def retire(self, *, block: bool) -> None:
        while self._pending:
            slot, event = self._pending[0]
            if not event.query():
                if not block:
                    return
                event.synchronize()
            self._pending.popleft()
            if self._ring is not None:
                read_slot, _ = self._ring.acquire_read(0)
                self._ring.release_read(read_slot)
            else:
                self._free.append(slot)
            block = False

    def close(self) -> None:
        self.retire(block=True)
        if self._ring is not None and self._registered:
            torch.cuda.cudart().cudaHostUnregister(self._ring.data_ptr)
            self._registered = False


@dataclass
class GpuBatch:
    frames: list[Frame[Any]]
    tensor: torch.Tensor
    event: torch.cuda.Event


class _GpuStage(Stage):
    def __init__(self, name: str, cancel: CancelToken, device: torch.device) -> None:
        super().__init__(name, cancel)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def _consume(self, batch: GpuBatch) -> torch.Tensor:
        """Make this stage's stream wait for the producer and keep the tensor alive on it."""
        self.stream.wait_event(batch.event)
        batch.tensor.record_stream(self.stream)
        return batch.tensor

    def _done(self) -> torch.cuda.Event:
        event = torch.cuda.Event()
        event.record(self.stream)
        # Waiting on our own stream keeps busy-time honest (GPU time of this stage) without blocking the
        # other stages' streams, which keep running in parallel.
        event.synchronize()
        return event


class UploadStage(_GpuStage):
    def __init__(
        self, cancel: CancelToken, inq: StageQueue[Frame[RawArray]], out: StageQueue[GpuBatch], *,
        device: torch.device, fmt: RawVideoFormat, color: ColorPlan, frames_per_call: int, suffix: str = "",
    ) -> None:
        super().__init__("upload" + suffix, cancel, device)
        self._in, self._out = inq, out
        self.fmt, self.color = fmt, color
        self.batch = max(1, frames_per_call)
        self.pool = PinnedPool(slots=3, slot_bytes=fmt.frame_bytes * self.batch)
        gamut = color.gamut()
        self._gamut = torch.from_numpy(gamut).to(device) if gamut is not None else None

    def _flush(self, frames: list[Frame[RawArray]]) -> None:
        fmt = self.fmt
        with self.timed():
            slot, view = self.pool.acquire()
            n = len(frames)
            nbytes = fmt.frame_bytes * n
            stacked = np.frombuffer(view, dtype=np.uint8, count=nbytes).view(fmt.dtype).reshape(n, -1)
            for i, f in enumerate(frames):
                stacked[i] = f.data
            with torch.cuda.stream(self.stream):
                host = torch.from_numpy(np.frombuffer(view, dtype=np.uint8, count=nbytes))
                flat8 = host.to(self.device, non_blocking=True)
                copy_done = torch.cuda.Event()
                copy_done.record(self.stream)
                self.pool.submit(slot, copy_done)
                flat = flat8.view(torch.uint8 if fmt.bit_depth == 8 else torch.int16).reshape(n, -1)
                if fmt.bit_depth == 16:
                    flat = flat.to(torch.int32) & 0xFFFF
                (yh, yw), (ch, cw), _ = fmt.plane_shapes
                y_end, c_size = yh * yw, ch * cw
                rgb = torch_ops.yuv_to_rgb(
                    flat[:, :y_end].reshape(n, yh, yw),
                    flat[:, y_end : y_end + c_size].reshape(n, ch, cw),
                    flat[:, y_end + c_size :].reshape(n, ch, cw),
                    matrix=self.color.in_matrix,
                    full_range=self.color.in_range is ColorRange.FULL,
                    bit_depth=fmt.bit_depth,
                    chroma_location=self.color.in_chroma,
                )
                if self._gamut is not None:
                    rgb = torch_ops.apply_gamut(rgb, self._gamut)
            event = self._done()
        meta = [Frame(f.index, f.pts, f.duration, None) for f in frames]
        self._out.put(GpuBatch(meta, rgb, event))
        self.metrics.frames += len(frames)

    def run(self) -> None:
        try:
            batch: list[Frame[RawArray]] = []
            while (frame := self._in.get()) is not None:
                batch.append(frame)
                if len(batch) >= self.batch:
                    self._flush(batch)
                    batch = []
            if batch:
                self._flush(batch)
            self._out.close()
        finally:
            self.pool.close()


class InferStage(_GpuStage):
    def __init__(
        self, cancel: CancelToken, inq: StageQueue[GpuBatch], out: StageQueue[GpuBatch], *,
        device: torch.device, runner: ModelRunner | None, passes: int, suffix: str = "",
    ) -> None:
        super().__init__("inference" + suffix, cancel, device)
        self._in, self._out = inq, out
        self.runner = runner
        self.passes = passes if runner is not None else 0

    def run(self) -> None:
        while (batch := self._in.get()) is not None:
            with self.timed():
                x = self._consume(batch)
                with torch.cuda.stream(self.stream):
                    for _ in range(self.passes):
                        assert self.runner is not None
                        x = self.runner.run(x)
                event = self._done()
            self._out.put(GpuBatch(batch.frames, x, event))
            self.metrics.frames += len(batch.frames)
        self._out.close()


class DownloadStage(_GpuStage):
    def __init__(
        self, cancel: CancelToken, inq: StageQueue[GpuBatch], out: StageQueue[Frame[RawArray]], *,
        device: torch.device, fmt: RawVideoFormat, color: ColorPlan, kernel: str, frames_per_call: int,
        suffix: str = "",
    ) -> None:
        super().__init__("postprocess" + suffix, cancel, device)
        self._in, self._out = inq, out
        self.fmt, self.color, self.kernel = fmt, color, kernel
        self.pool = PinnedPool(slots=2, slot_bytes=fmt.frame_bytes * max(1, frames_per_call))

    def run(self) -> None:
        fmt = self.fmt
        try:
            while (batch := self._in.get()) is not None:
                with self.timed():
                    rgb = self._consume(batch)
                    n = len(batch.frames)
                    slot, view = self.pool.acquire()
                    with torch.cuda.stream(self.stream):
                        if rgb.shape[-2:] != (fmt.height, fmt.width):
                            rgb = torch_ops.resize(rgb, fmt.height, fmt.width, self.kernel)
                        planes = torch_ops.rgb_to_yuv(
                            rgb, matrix=self.color.out_matrix,
                            full_range=self.color.out_range is ColorRange.FULL, bit_depth=fmt.bit_depth,
                            log2_cw=fmt.log2_cw, log2_ch=fmt.log2_ch, chroma_location=self.color.out_chroma,
                        )
                        dtype = torch.uint8 if fmt.bit_depth == 8 else torch.int16
                        packed = torch.cat([p.reshape(n, -1) for p in planes], dim=1).to(dtype)
                        host = torch.from_numpy(
                            np.frombuffer(view, dtype=np.uint8, count=fmt.frame_bytes * n)
                        ).view(dtype).reshape(n, -1)
                        host.copy_(packed, non_blocking=True)
                    event = self._done()
                    self.pool.submit(slot, event)
                    out_dtype = np.uint8 if fmt.bit_depth == 8 else np.uint16
                    frames_out = host.numpy().view(out_dtype).copy()  # leave the pinned slot for reuse
                    self.pool.retire(block=False)
                for meta, data in zip(batch.frames, frames_out, strict=True):
                    self._out.put(meta.replace(data))
                self.metrics.frames += n
            self._out.close()
        finally:
            self.pool.close()

