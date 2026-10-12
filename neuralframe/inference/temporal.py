"""Temporal (multi-frame) super-resolution: chunked sequences with overlap, cross-fades and scene cuts.

Chunks are processed in order on one device. Each chunk sees ``overlap`` extra frames on both sides
(clipped at scene cuts — never across a shot — or replicated at the edges for windowed models). Around
a chunk boundary that is not a scene cut, the two chunks' outputs are cross-faded over ``2 * half``
frames with linear weights, which removes the jump a recurrent model would otherwise show where its
hidden state restarts. Frames leave the stage strictly in index order, so segments and resume work as
for single-frame models.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuralframe.chunking.planner import ChunkPlan
from neuralframe.colorspace import ColorPlan, ColorRange
from neuralframe.core.runtime import CancelToken, Frame, Stage, StageQueue
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.inference.archs import load_network
from neuralframe.inference.backend import InferenceOOM
from neuralframe.io.rawvideo import RawArray, RawVideoFormat
from neuralframe.log import get_logger
from neuralframe.models.registry import ModelSpec
from neuralframe.ops import torch_ops

log = get_logger(component="temporal")


class TemporalModel:
    """A sequence network on one device: (T, 3, H, W) in → per-frame (3, sH, sW) outputs, in order."""

    def __init__(self, spec: ModelSpec, weights: Path, device: torch.device, precision: str = "fp16",
                 *, strip_rows: int | None = None) -> None:
        net = load_network(spec, weights).to(device)
        self.dtype = torch.float16 if precision == "fp16" and device.type == "cuda" else torch.float32
        if self.dtype == torch.float16:
            net = net.half()
        if not hasattr(net, "iter_outputs"):
            raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"{spec.id} is not a sequence model")
        model: Any = net
        if strip_rows is not None and hasattr(model, "basicvsr"):
            model.basicvsr.strip_rows = strip_rows
        self.net: Any = model
        self.spec = spec
        self.device = device
        self.scale = spec.scale
        self.multiple = max(spec.pad_multiple, 1)
        self.frames_processed = 0
        self.seconds = 0.0

    def run(self, frames: torch.Tensor) -> Iterator[tuple[int, torch.Tensor]]:
        """``frames`` (T, 3, H, W) float in [0, 1] on the device → yields ``(t, (3, H*s, W*s) float32)``."""
        t0 = time.perf_counter()
        _, _, h, w = frames.shape
        padded, _, _ = torch_ops.pad_to_multiple(frames, self.multiple)
        x = padded.to(self.dtype).unsqueeze(0)
        try:
            with torch.inference_mode():
                for t, out in self.net.iter_outputs(x):
                    yield t, out[0, :, : h * self.scale, : w * self.scale].float().clamp_(0.0, 1.0)
        except torch.OutOfMemoryError as exc:
            raise InferenceOOM(str(exc)) from exc
        finally:
            self.frames_processed += frames.shape[0]
            self.seconds += time.perf_counter() - t0

    def close(self) -> None:
        self.net = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class TemporalProcessor:
    """BatchProcessor facade so the job layer can create/close sequence models like other processors."""

    batch_size = 1

    def __init__(self, model: TemporalModel) -> None:
        self.model = model
        self.stage: TemporalStage | None = None

    def process(self, items: list[Any]) -> list[Any]:  # pragma: no cover - the TemporalStage drives it
        raise NotImplementedError("sequence models run through TemporalStage")

    def stats(self) -> dict[str, object]:
        if self.stage is not None:
            return dict(self.stage.stats())
        return {"temporal": {"frames": self.model.frames_processed}}

    def close(self) -> None:
        self.model.close()


def crossfade_weight(position: int, width: int) -> float:
    """Weight of the *earlier* chunk at ``position`` (0..width-1) of a ``width``-frame blend zone."""
    return 1.0 - (position + 0.5) / width


@dataclass
class _Boundary:
    index: int
    half: int
    stored: dict[int, torch.Tensor] = field(default_factory=dict)  # earlier chunk's outputs for the zone


class TemporalStage(Stage):
    """Decode-order raw frames in → upscaled, colour-converted raw frames out (single device)."""

    def __init__(
        self,
        cancel: CancelToken,
        inq: StageQueue[Frame[RawArray]],
        out: StageQueue[Frame[RawArray]],
        *,
        model: TemporalModel,
        chunks: ChunkPlan,
        overlap: int,
        in_fmt: RawVideoFormat,
        out_fmt: RawVideoFormat,
        color: ColorPlan,
        kernel: str,
        done_chunks: set[int] | None = None,
        replicate_edges: bool = False,
    ) -> None:
        super().__init__("inference", cancel)
        self._in, self._out = inq, out
        self.model = model
        self.chunks = chunks
        self.overlap = max(0, overlap)
        self.half = max(1, self.overlap // 2) if self.overlap > 0 else 0
        self.in_fmt, self.out_fmt, self.color, self.kernel = in_fmt, out_fmt, color, kernel
        self.done = frozenset(done_chunks or ())
        self.replicate_edges = replicate_edges
        self.device = model.device
        gamut = color.gamut()
        self._gamut = torch.from_numpy(gamut).to(self.device) if gamut is not None else None
        self._buffer: dict[int, Frame[RawArray]] = {}
        self._eos = False
        self._last_index = -1
        self.blended_frames = 0

    # ---- frame plumbing ------------------------------------------------------------------------------

    def _fill_until(self, index: int) -> None:
        """Buffer incoming frames until ``index`` is available (or the stream ends)."""
        while not self._eos and self._last_index < index:
            frame = self._in.get()
            if frame is None:
                self._eos = True
                break
            self._buffer[frame.index] = frame
            self._last_index = frame.index

    def _segment_bounds(self, chunk_index: int) -> tuple[int, int]:
        """[start, end) of the shot containing the chunk: chunk boundaries at scene cuts delimit shots."""
        chunks = self.chunks.chunks
        start = chunk_index
        while start > 0 and not chunks[start].scene_cut_before:
            start -= 1
        end = chunk_index + 1
        while end < len(chunks) and not chunks[end].scene_cut_before:
            end += 1
        return chunks[start].start, chunks[end - 1].end if end < len(chunks) else 10**12

    def _to_rgb(self, frames: list[Frame[RawArray]]) -> torch.Tensor:
        fmt = self.in_fmt
        stacked = np.stack([f.data for f in frames])
        host = torch.from_numpy(stacked.view(np.int16) if fmt.bit_depth > 8 else stacked)
        flat = host.to(self.device, non_blocking=True)
        if fmt.bit_depth == 16:
            flat = flat.to(torch.int32) & 0xFFFF
        n = flat.shape[0]
        (yh, yw), (ch, cw), _ = fmt.plane_shapes
        y_end, c_size = yh * yw, ch * cw
        rgb = torch_ops.yuv_to_rgb(
            flat[:, :y_end].reshape(n, yh, yw), flat[:, y_end : y_end + c_size].reshape(n, ch, cw),
            flat[:, y_end + c_size :].reshape(n, ch, cw), matrix=self.color.in_matrix,
            full_range=self.color.in_range is ColorRange.FULL, bit_depth=fmt.bit_depth,
            chroma_location=self.color.in_chroma,
        )
        return torch_ops.apply_gamut(rgb, self._gamut) if self._gamut is not None else rgb

    def _finish(self, rgb: torch.Tensor) -> torch.Tensor:
        """Model-resolution RGB (3, H, W) → output-resolution RGB (3, oh, ow)."""
        fmt = self.out_fmt
        if rgb.shape[-2:] != (fmt.height, fmt.width):
            rgb = torch_ops.resize(rgb[None], fmt.height, fmt.width, self.kernel)[0]
        return rgb

    def _emit(self, index: int, rgb: torch.Tensor) -> None:
        fmt = self.out_fmt
        y, u, v = torch_ops.rgb_to_yuv(
            rgb[None], matrix=self.color.out_matrix, full_range=self.color.out_range is ColorRange.FULL,
            bit_depth=fmt.bit_depth, log2_cw=fmt.log2_cw, log2_ch=fmt.log2_ch, chroma_location=self.color.out_chroma,
        )
        dtype = torch.uint8 if fmt.bit_depth == 8 else torch.int16
        packed = torch.cat([p.reshape(-1) for p in (y, u, v)]).to(dtype).cpu().numpy()
        data = packed.view(np.uint16) if fmt.bit_depth > 8 else packed
        src = self._buffer[index]
        self._out.put(Frame(index, src.pts, src.duration, data))
        self.metrics.frames += 1

    # ---- main loop -----------------------------------------------------------------------------------

    def _boundary_after(self, k: int) -> _Boundary | None:
        """Blend zone between chunk ``k`` and ``k+1`` (none across scene cuts)."""
        chunks = self.chunks.chunks
        if k + 1 >= len(chunks) or self.half == 0 or chunks[k + 1].scene_cut_before:
            return None
        half = min(self.half, chunks[k].frames // 2, chunks[k + 1].frames // 2)
        return _Boundary(index=chunks[k].end, half=half) if half > 0 else None

    def _emit_if_needed(self, index: int, rgb: torch.Tensor) -> None:
        if self.chunks.chunk_of(index).id not in self.done:  # finished chunks are already in the store
            self._emit(index, rgb)

    def _flush_zone(self, zone: _Boundary, produced: dict[int, torch.Tensor]) -> None:
        for idx in range(zone.index - zone.half, zone.index + zone.half):
            earlier, later = zone.stored.get(idx), produced.get(idx)
            if earlier is not None and later is not None:
                w = crossfade_weight(idx - (zone.index - zone.half), 2 * zone.half)
                # float32 tensors times Python floats round the weights to float32, like the native blend
                self._emit_if_needed(idx, earlier * float(w) + later * float(1.0 - w))
                self.blended_frames += 1
            elif earlier is not None or later is not None:
                frame = later if later is not None else earlier
                assert frame is not None
                self._emit_if_needed(idx, frame)

    def _process_chunk(self, k: int, incoming: _Boundary | None) -> _Boundary | None:
        """Run the model over chunk ``k`` plus overlap; emit its frames in order; return its outgoing
        blend zone with the stored outputs the next chunk will cross-fade with."""
        chunk = self.chunks.chunks[k]
        seg_start, seg_end = self._segment_bounds(k)
        lo, hi = chunk.start - self.overlap, chunk.end + self.overlap
        self._fill_until(hi - 1)
        available = self._last_index + 1
        seg_hi = min(seg_end, available)
        end = min(chunk.end, available)
        outgoing = self._boundary_after(k) if end == chunk.end and available > chunk.end else None
        in_lo, in_hi = (chunk.start - incoming.half, chunk.start + incoming.half) if incoming else (chunk.start,) * 2
        out_lo, out_hi = (end - outgoing.half, end + outgoing.half) if outgoing else (end, end)

        if self.replicate_edges:  # windowed models: every output sees a full window
            logical = list(range(lo, hi))
            sources = [min(max(i, seg_start), seg_hi - 1) for i in logical]
        else:  # recurrent models: just clip at the shot edges
            logical = list(range(max(lo, seg_start), seg_hi))
            sources = logical
        frames = [self._buffer[i] for i in sources]
        produced: dict[int, torch.Tensor] = {}
        flushed = incoming is None
        with self.timed():
            for t, out in self.model.run(self._to_rgb(frames)):
                idx = logical[t]
                if idx < seg_start or idx >= seg_hi:
                    continue  # replicated padding
                if in_lo <= idx < in_hi:
                    produced[idx] = self._finish(out)
                elif in_hi <= idx < out_lo:
                    if not flushed:
                        assert incoming is not None
                        self._flush_zone(incoming, produced)
                        flushed = True
                    self._emit_if_needed(idx, self._finish(out))
                if outgoing is not None and out_lo <= idx < out_hi:
                    outgoing.stored[idx] = self._finish(out)
            if not flushed and incoming is not None:
                self._flush_zone(incoming, produced)
        keep_from = min(chunk.end - self.overlap, out_lo)
        for i in [i for i in self._buffer if i < keep_from]:
            del self._buffer[i]
        return outgoing

    def run(self) -> None:
        pending: _Boundary | None = None
        chunks = self.chunks.chunks
        for k, chunk in enumerate(chunks):
            if self._eos and self._last_index < chunk.start:
                break
            needed = k not in self.done
            next_needed = k + 1 < len(chunks) and (k + 1) not in self.done
            if needed or next_needed:
                # A finished chunk is re-run (its frames are not emitted again) only to rebuild the blend
                # zone it shares with the next unfinished chunk: same inputs, same outputs.
                pending = self._process_chunk(k, pending)
            else:
                pending = None
                self._fill_until(chunk.end - 1)
                for i in [i for i in self._buffer if i < chunk.end - self.overlap]:
                    del self._buffer[i]
        while self._in.get() is not None:  # drain frames past the planned total
            pass
        self._out.close()

    def stats(self) -> dict[str, Any]:
        return {
            "temporal": {
                "frames": self.model.frames_processed,
                "seconds": round(self.model.seconds, 3),
                "overlap": self.overlap,
                "blended_frames": self.blended_frames,
            }
        }
