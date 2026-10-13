"""Face restoration stage: restores faces in finished (output-size) frames, in batches, before encoding.

Frames arrive as the encoder's raw YUV frames; a batch is converted to 8-bit RGB with the output colour
settings, sent through the face engine (which detects faces, restores them and pastes them back), and
converted back. Frame indices and timestamps are kept, so ordering, chunking and resume work as before.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neuralframe.colorspace import ColorPlan, ColorRange
from neuralframe.core.runtime import CancelToken, Frame, Stage, StageQueue
from neuralframe.engines.model import FaceRestorer
from neuralframe.io.rawvideo import RawArray, RawVideoFormat


class FaceStage(Stage):
    def __init__(self, cancel: CancelToken, inq: StageQueue[Frame[RawArray]], out: StageQueue[Frame[RawArray]], *,
                 restorer: FaceRestorer, fmt: RawVideoFormat, color: ColorPlan, batch: int = 32) -> None:
        super().__init__("faces", cancel)
        self._in, self._out = inq, out
        self.restorer = restorer
        self.fmt, self.color = fmt, color
        self.batch = max(1, batch)

    def _to_rgb(self, frames: list[Frame[RawArray]]) -> list[np.ndarray]:
        import torch

        from neuralframe.ops import torch_ops

        fmt = self.fmt
        stacked = np.stack([f.data for f in frames])
        flat = torch.from_numpy(stacked.view(np.int16) if fmt.bit_depth > 8 else stacked)
        if fmt.bit_depth == 16:
            flat = flat.to(torch.int32) & 0xFFFF
        n = flat.shape[0]
        (yh, yw), (ch, cw), _ = fmt.plane_shapes
        y_end, c_size = yh * yw, ch * cw
        rgb = torch_ops.yuv_to_rgb(
            flat[:, :y_end].reshape(n, yh, yw), flat[:, y_end:y_end + c_size].reshape(n, ch, cw),
            flat[:, y_end + c_size:].reshape(n, ch, cw), matrix=self.color.out_matrix,
            full_range=self.color.out_range is ColorRange.FULL, bit_depth=fmt.bit_depth,
            chroma_location=self.color.out_chroma,
        )
        u8 = (rgb.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        return list(u8)

    def _to_raw(self, rgb: np.ndarray) -> RawArray:
        import torch

        from neuralframe.ops import torch_ops

        fmt = self.fmt
        t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None] / 255.0
        y, u, v = torch_ops.rgb_to_yuv(
            t, matrix=self.color.out_matrix, full_range=self.color.out_range is ColorRange.FULL,
            bit_depth=fmt.bit_depth, log2_cw=fmt.log2_cw, log2_ch=fmt.log2_ch, chroma_location=self.color.out_chroma,
        )
        dtype = torch.uint8 if fmt.bit_depth == 8 else torch.int16
        packed = torch.cat([p.reshape(-1) for p in (y, u, v)]).to(dtype).numpy()
        data: Any = packed.view(np.uint16) if fmt.bit_depth > 8 else packed
        return data  # type: ignore[no-any-return]

    def _flush(self, pending: list[Frame[RawArray]]) -> None:
        with self.timed():
            restored = self.restorer.restore(self._to_rgb(pending))
        for frame, rgb in zip(pending, restored, strict=True):
            self._out.put(Frame(frame.index, frame.pts, frame.duration, self._to_raw(rgb)))
            self.metrics.frames += 1

    def run(self) -> None:
        pending: list[Frame[RawArray]] = []
        while True:
            frame = self._in.get()
            if frame is None:
                break
            pending.append(frame)
            if len(pending) >= self.batch:
                self._flush(pending)
                pending = []
        if pending:
            self._flush(pending)
        self._out.close()
