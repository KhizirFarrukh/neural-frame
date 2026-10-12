"""Concrete pipeline stages: decode → preprocess → process → postprocess → encode."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from neuralframe.chunking.planner import Chunk, ChunkPlan
from neuralframe.chunking.store import ChunkStore
from neuralframe.colorspace import ColorPlan, ColorRange
from neuralframe.core.runtime import CancelToken, Frame, InFlightLimiter, Stage, StageQueue
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.processor import BatchProcessor
from neuralframe.io.decoder import DecodedFrame
from neuralframe.io.encoder import SegmentEncoder
from neuralframe.io.rawvideo import RawArray, RawVideoFormat
from neuralframe.ops import FrameOps

F32 = npt.NDArray[np.float32]


class DecodeStage(Stage):
    """Pulls frames from the decoder subprocess (already started) into the pipeline."""

    def __init__(
        self,
        cancel: CancelToken,
        frames: Iterator[DecodedFrame],
        out: StageQueue[Frame[RawArray]],
        *,
        skip: Callable[[int], bool] | None = None,
        limiter: InFlightLimiter | None = None,
    ) -> None:
        super().__init__("decode", cancel)
        self._frames = frames
        self._out = out
        self._skip = skip
        self._limiter = limiter

    def run(self) -> None:
        it = iter(self._frames)
        while True:
            self.cancel.raise_if_cancelled()
            with self.timed():
                decoded = next(it, None)
            if decoded is None:
                break
            if self._skip is not None and self._skip(decoded.index):
                continue
            if self._limiter is not None:
                self._limiter.acquire()
            self._out.put(Frame(decoded.index, decoded.pts, decoded.duration, decoded.data))
            self.metrics.frames += 1
        self._out.close()


class PreprocessStage(Stage):
    """Raw YUV → planar RGB float32 with the source matrix/range, plus optional gamut conversion."""

    def __init__(
        self,
        cancel: CancelToken,
        inq: StageQueue[Frame[RawArray]],
        out: StageQueue[Frame[F32]],
        *,
        ops: FrameOps,
        fmt: RawVideoFormat,
        color: ColorPlan,
    ) -> None:
        super().__init__("preprocess", cancel)
        self._in, self._out = inq, out
        self._ops, self._fmt, self._color = ops, fmt, color
        self._gamut = color.gamut()

    def convert(self, data: RawArray) -> F32:
        y, u, v = self._fmt.split(data)
        rgb = self._ops.yuv_to_rgb(
            y, u, v,
            matrix=self._color.in_matrix,
            full_range=self._color.in_range is ColorRange.FULL,
            bit_depth=self._fmt.bit_depth,
            chroma_location=self._color.in_chroma,
        )
        if self._gamut is not None:
            rgb = self._ops.apply_gamut(rgb, self._gamut)
        return rgb

    def run(self) -> None:
        while (frame := self._in.get()) is not None:
            with self.timed():
                rgb = self.convert(frame.data)
            self._out.put(frame.replace(rgb))
            self.metrics.frames += 1
        self._out.close()


class ProcessStage(Stage):
    """Runs the frame processor (the super-resolution model) on batches of frames.

    In host mode items are float RGB frames; in device mode they are raw YUV buffers and the processor does
    colour conversion, inference and resizing on the GPU.
    """

    def __init__(
        self,
        cancel: CancelToken,
        inq: StageQueue[Frame[Any]],
        out: StageQueue[Frame[Any]],
        *,
        processor: BatchProcessor,
        name: str = "inference",
    ) -> None:
        super().__init__(name, cancel)
        self._in, self._out = inq, out
        self.processor = processor

    def _flush(self, batch: list[Frame[Any]]) -> None:
        with self.timed():
            outputs = self.processor.process([f.data for f in batch])
        if len(outputs) != len(batch):
            raise NeuralFrameError(
                ErrorCode.INFERENCE_FAILED, f"processor returned {len(outputs)} frames for {len(batch)}"
            )
        for frame, result in zip(batch, outputs, strict=True):
            self._out.put(frame.replace(result))
        self.metrics.frames += len(batch)

    def run(self) -> None:
        batch: list[Frame[Any]] = []
        while (frame := self._in.get()) is not None:
            batch.append(frame)
            if len(batch) >= max(1, self.processor.batch_size):
                self._flush(batch)
                batch = []
        if batch:
            self._flush(batch)
        self._out.close()


class PostprocessStage(Stage):
    """Resize to the exact target size, then RGB → quantised YUV in the output colour space."""

    def __init__(
        self,
        cancel: CancelToken,
        inq: StageQueue[Frame[F32]],
        out: StageQueue[Frame[RawArray]],
        *,
        ops: FrameOps,
        out_fmt: RawVideoFormat,
        color: ColorPlan,
        kernel: str,
    ) -> None:
        super().__init__("postprocess", cancel)
        self._in, self._out = inq, out
        self._ops, self._fmt, self._color, self._kernel = ops, out_fmt, color, kernel

    def convert(self, rgb: F32) -> RawArray:
        if rgb.shape[1:] != (self._fmt.height, self._fmt.width):
            rgb = self._ops.resize(rgb, self._fmt.height, self._fmt.width, self._kernel)
        planes = self._ops.rgb_to_yuv(
            rgb,
            matrix=self._color.out_matrix,
            full_range=self._color.out_range is ColorRange.FULL,
            bit_depth=self._fmt.bit_depth,
            log2_cw=self._fmt.log2_cw,
            log2_ch=self._fmt.log2_ch,
            chroma_location=self._color.out_chroma,
        )
        return self._fmt.join(planes)

    def run(self) -> None:
        while (frame := self._in.get()) is not None:
            with self.timed():
                buf = self.convert(frame.data)
            self._out.put(frame.replace(buf))
            self.metrics.frames += 1
        self._out.close()


class EncodeStage(Stage):
    """Restores frame order, writes each chunk to its own segment and commits it to the chunk store."""

    def __init__(
        self,
        cancel: CancelToken,
        inq: StageQueue[Frame[RawArray]],
        *,
        chunks: ChunkPlan,
        store: ChunkStore,
        make_encoder: Callable[[Path], SegmentEncoder],
        first_index: int = 0,
        done_chunks: set[int] | None = None,
        on_chunk_done: Callable[[Chunk, int], None] | None = None,
        limiter: InFlightLimiter | None = None,
    ) -> None:
        super().__init__("encode", cancel)
        self._in = inq
        self._chunks = chunks
        self._store = store
        self._make_encoder = make_encoder
        self._next = first_index
        self._done = frozenset(done_chunks or ())
        self._on_chunk_done = on_chunk_done
        self._limiter = limiter
        self._encoder: SegmentEncoder | None = None
        self._chunk: Chunk | None = None
        self._chunk_start = 0

    def _finish_chunk(self) -> None:
        enc, chunk = self._encoder, self._chunk
        if enc is None or chunk is None:
            return
        with self.timed():
            enc.close()
        assert enc.first_pts is not None and enc.last_pts is not None
        self._store.commit(
            chunk.id, self._chunk_start, self._chunk_start + enc.frames_written, enc.frames_written,
            enc.first_pts, enc.last_pts, enc.out_path,
        )
        if self._on_chunk_done is not None:
            self._on_chunk_done(chunk, enc.frames_written)
        self._encoder, self._chunk = None, None

    def _write(self, frame: Frame[RawArray]) -> None:
        chunk = self._chunks.chunk_of(frame.index)
        if self._chunk is None or chunk.id != self._chunk.id:
            self._finish_chunk()
            self._chunk = chunk
            self._chunk_start = frame.index
            self._encoder = self._make_encoder(self._store.segment_path(chunk.id))
            self._encoder.start()
        assert self._encoder is not None
        with self.timed():
            self._encoder.write(frame.pts, frame.duration, frame.data)
        self.metrics.frames += 1
        if self._limiter is not None:
            self._limiter.release()

    def _skip_completed(self) -> None:
        """When resuming, jump the expected index over chunks that are already in the store."""
        while self._done:
            chunk = self._chunks.chunk_of(self._next)
            if chunk.id not in self._done:
                return
            if chunk.id == self._chunks.chunks[-1].id:
                self._next = sys.maxsize  # everything from here on is already encoded
                return
            self._next = chunk.end

    def run(self) -> None:
        pending: dict[int, Frame[RawArray]] = {}
        try:
            self._skip_completed()
            while (frame := self._in.get()) is not None:
                pending[frame.index] = frame
                while self._next in pending:
                    self._write(pending.pop(self._next))
                    self._next += 1
                    self._skip_completed()
            if pending:
                raise NeuralFrameError(
                    ErrorCode.INTERNAL, f"frames missing before index {min(pending)} (expected {self._next})"
                )
            self._finish_chunk()
        except BaseException:
            if self._encoder is not None:
                self._encoder.abort()  # never leave a partial segment that could look complete
                self._encoder = None
            raise
