"""Job specification, planning and execution — the single entry point used by CLI, API and GUI."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuralframe import __version__
from neuralframe.chunking.analysis import FrameAnalysis, analyse
from neuralframe.chunking.planner import ChunkPlan, plan_chunks
from neuralframe.chunking.store import ChunkStore
from neuralframe.config import Codec, Settings, Speed
from neuralframe.core.planning import (
    Geometry,
    TargetSpec,
    plan_color,
    plan_decode,
    plan_encoder,
    plan_geometry,
)
from neuralframe.core.runtime import CancelToken, InFlightLimiter, StageQueue, StageRunner
from neuralframe.core.stages import DecodeStage, EncodeStage, PostprocessStage, PreprocessStage, ProcessStage
from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.framegen.retime import detect_drops, parse_fps, plan_retime, snap_fps
from neuralframe.inference.processor import BatchProcessor, IdentityProcessor
from neuralframe.io.decoder import DecodePlan, ResilientDecoder
from neuralframe.io.encoder import EncoderPlan, SegmentEncoder, default_quality, preflight
from neuralframe.io.ffmpeg import FFmpegTools, locate_ffmpeg
from neuralframe.io.hwcaps import Capabilities, detect_capabilities
from neuralframe.io.media import FrameTimeline, MediaInfo
from neuralframe.io.muxer import MuxPlan, concat_segments, container_for, mux_final, plan_mux, verify_output
from neuralframe.io.probe import probe, scan_frames
from neuralframe.io.rawvideo import RawVideoFormat
from neuralframe.log import bind_context, clear_context, get_logger
from neuralframe.models.registry import ModelRegistry, ModelSpec
from neuralframe.monitoring import ResourceSampler, gpu_count, host_memory
from neuralframe.ops import FrameOps, get_ops

if TYPE_CHECKING:
    from neuralframe.inference.factory import InferencePlan

log = get_logger(component="job")

JobState = Literal[
    "queued", "probing", "planning", "running", "muxing", "verifying", "done", "failed", "cancelled", "paused"
]


class JobSpec(BaseModel):
    """What the user asked for. ``None`` fields fall back to the preset, then to the configuration."""

    model_config = ConfigDict(extra="forbid")

    input: Path
    output: Path
    model: str | None = None
    target: TargetSpec = Field(default_factory=TargetSpec.default)
    preset: str = "balanced"
    backend: Literal["auto", "tensorrt", "torch", "onnxruntime"] | None = None
    encoder: str | None = None
    codec: Codec | None = None
    quality: int | None = Field(default=None, ge=0, le=63)
    speed: Speed | None = None
    ten_bit: bool | None = None
    deinterlace: Literal["auto", "on", "off"] | None = None
    denoise: Literal["off", "light", "medium", "strong"] | None = None
    max_vram_gb: float | None = Field(default=None, gt=0)
    gpus: list[int] | None = None
    tile_size: int | None = Field(default=None, ge=32)
    tile_overlap: int | None = Field(default=None, ge=0)
    batch_size: int | None = Field(default=None, ge=1)
    chunk_frames: int | None = Field(default=None, ge=8)
    resume: bool = False
    overwrite: bool = False
    keep_work_dir: bool | None = None
    upscale: bool = True
    """False = keep the source size (frame generation only)."""
    interpolate: bool = False
    """Frame generation: constant-rate output at ``target_fps`` with dropped frames re-synthesised."""
    target_fps: str | None = None
    repair_drops: Literal["auto", "on", "off"] | None = None
    framegen_engine: Literal["auto", "rife", "minterpolate"] | None = None
    framegen_order: Literal["auto", "upscale_first", "interpolate_first"] | None = None


@dataclass(frozen=True)
class StageSnapshot:
    name: str
    frames: int
    fps: float
    busy_fps: float
    utilisation: float
    queue_depth: int | None = None
    queue_capacity: int | None = None


@dataclass(frozen=True)
class ProgressEvent:
    job_id: str
    state: JobState
    message: str = ""
    frames_done: int = 0
    frames_total: int = 0
    elapsed_s: float = 0.0
    eta_s: float | None = None
    fps: float = 0.0
    stages: tuple[StageSnapshot, ...] = ()
    gpu: dict[str, Any] | None = None

    @property
    def percent(self) -> float:
        if self.state == "done":
            return 100.0
        if self.frames_total <= 0:
            return 0.0
        return min(100.0, 100.0 * self.frames_done / self.frames_total)

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "message": self.message,
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "percent": round(self.percent, 2),
            "elapsed_s": round(self.elapsed_s, 2),
            "eta_s": None if self.eta_s is None else round(self.eta_s, 1),
            "fps": round(self.fps, 2),
            "stages": [s.__dict__ for s in self.stages],
            "gpu": self.gpu,
        }


ProgressCallback = Callable[[ProgressEvent], None]


@dataclass
class JobPlan:
    job_key: str
    fingerprint: str
    spec: JobSpec
    media: MediaInfo
    timeline: FrameTimeline
    model: ModelSpec
    geometry: Geometry
    decode: DecodePlan
    encoder: EncoderPlan
    out_fmt: RawVideoFormat
    mux: MuxPlan
    chunks: ChunkPlan
    ops: str
    resize_kernel: str
    work_dir: Path
    notes: list[str] = field(default_factory=list)
    mode: Literal["device", "host"] = "host"
    inference: InferencePlan | None = None
    weights: Path | None = None
    temporal_overlap: int = 0
    output_frames: int = 0
    source_chunks: ChunkPlan | None = None
    retime: Any = None
    framegen_order: str = "upscale_first"
    framegen_model: ModelSpec | None = None
    framegen_weights: Path | None = None
    framegen: dict[str, Any] | None = None
    out_fps: Fraction | None = None

    def __post_init__(self) -> None:
        if self.source_chunks is None:
            self.source_chunks = self.chunks

    @property
    def passthrough(self) -> bool:
        """True when frames need no pixel processing (frame generation at the source size and format)."""
        c, g = self.encoder.color, self.geometry
        return (
            self.inference is None and self.model.is_classical
            and (g.out_width, g.out_height) == (g.decoded_width, g.decoded_height)
            and self.decode.fmt == self.out_fmt and not c.convert_gamut
            and (c.in_matrix, c.in_range, c.in_chroma) == (c.out_matrix, c.out_range, c.out_chroma)
        )

    @property
    def total_frames(self) -> int:
        """Output frames (differs from the source count with frame generation)."""
        return self.output_frames or self.timeline.frame_count

    def summary(self) -> dict[str, Any]:
        return {
            "job_key": self.job_key,
            "model": self.model.id,
            "geometry": self.geometry.summary(),
            "color": self.encoder.color.summary(),
            "decode": {
                "hwaccel": self.decode.hwaccel,
                "filters": list(self.decode.filters),
                "pipe_format": self.decode.fmt.pix_fmt,
            },
            "encoder": {
                "name": self.encoder.name,
                "codec": self.encoder.codec,
                "quality": self.encoder.quality,
                "speed": self.encoder.speed,
                "ten_bit": self.encoder.ten_bit,
                "args": self.encoder.codec_args(),
            },
            "mux": self.mux.summary(),
            "chunks": self.chunks.summary(),
            "framegen": self.framegen,
            "mode": self.mode,
            "ops": self.ops if self.mode == "host" else "torch",
            "inference": self.inference.summary() if self.inference else None,
            "resize_kernel": self.resize_kernel,
            "notes": self.notes,
        }


@dataclass
class JobResult:
    job_id: str
    output: Path
    report_path: Path
    report: dict[str, Any]


def media_summary(media: MediaInfo, timeline: FrameTimeline | None = None) -> dict[str, Any]:
    video = media.video
    out: dict[str, Any] = {
        "path": str(media.path),
        "format": media.format_name,
        "duration_s": media.duration,
        "size_bytes": media.size,
        "audio_streams": [a.codec for a in media.audio],
        "subtitle_streams": [s.codec for s in media.subtitles],
        "chapters": len(media.chapters),
    }
    if video is not None:
        out["video"] = {
            "codec": video.codec,
            "size": f"{video.width}x{video.height}",
            "pix_fmt": video.pix_fmt,
            "bit_depth": video.bit_depth,
            "fps": f"{video.fps.numerator}/{video.fps.denominator}",
            "sar": f"{video.sar.numerator}:{video.sar.denominator}",
            "rotation": video.rotation,
            "field_order": video.field_order,
            "color": video.color.model_dump(),
            "hdr": video.hdr_format,
        }
    if timeline is not None:
        out["frames"] = timeline.frame_count
        out["vfr"] = timeline.is_vfr
    return out


class JobRunner:
    """Plans and runs one job. Thread-safe to cancel from another thread via :attr:`cancel`."""

    def __init__(
        self,
        settings: Settings,
        *,
        job_id: str | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
        registry: ModelRegistry | None = None,
        capabilities: Capabilities | None = None,
        on_plan: Callable[[JobPlan], None] | None = None,
        on_chunk_done: Callable[[int, int], None] | None = None,
    ) -> None:
        self.settings = settings
        self.on_plan = on_plan
        self.on_chunk_done = on_chunk_done
        self.job_id = job_id or uuid.uuid4().hex[:12]
        self.progress = progress
        self.cancel = cancel or CancelToken()
        self.registry = registry or ModelRegistry.load()
        self._caps = capabilities
        self._t0 = time.perf_counter()
        self._timings: dict[str, float] = {}
        self._state: JobState = "queued"
        self._frames_total = 0
        self._frames_skipped = 0
        self._encode_stage: EncodeStage | None = None
        self._stages: list[Any] = []
        self._queues: dict[str, StageQueue[Any]] = {}
        self._sampler: ResourceSampler | None = None
        self._pipeline_t0: float | None = None
        self._fallbacks: list[str] = []
        self._decoder: ResilientDecoder | None = None
        self._analysis: FrameAnalysis | None = None
        self._adaptations: list[dict[str, Any]] = []
        self._calibration: dict[str, Any] | None = None
        self._resume_info: dict[str, Any] = {"mode": "fresh"}
        self._vram_reserve = 0
        """VRAM kept free for frame interpolation sharing the super-resolution device."""

    # ---- progress ------------------------------------------------------------------------------------

    def _emit(self, state: JobState, message: str = "") -> None:
        self._state = state
        if self.progress is None:
            return
        done = self._frames_skipped + (self._encode_stage.metrics.frames if self._encode_stage else 0)
        if state == "done":
            done = max(done, self._frames_total)
        elapsed = time.perf_counter() - self._t0
        fps = 0.0
        eta = None
        if self._pipeline_t0 is not None and self._encode_stage is not None:
            run_s = max(time.perf_counter() - self._pipeline_t0, 1e-6)
            fps = self._encode_stage.metrics.frames / run_s
            if fps > 0 and self._frames_total > 0:
                eta = max(self._frames_total - done, 0) / fps
        snaps = []
        for stage in self._stages:
            q = self._queues.get(stage.name)
            snaps.append(
                StageSnapshot(
                    name=stage.name,
                    frames=stage.metrics.frames,
                    fps=round(stage.metrics.fps, 2),
                    busy_fps=round(stage.metrics.busy_fps, 2),
                    utilisation=round(stage.metrics.utilisation, 3),
                    queue_depth=q.depth if q is not None else None,
                    queue_capacity=q.maxsize if q is not None else None,
                )
            )
        gpu = None
        if self._sampler is not None:
            gpu = {
                str(i): {"used_mb": s.used_mb, "total_mb": s.total_mb, "util_pct": s.utilization_pct}
                for i, s in self._sampler.current().items()
            }
        event = ProgressEvent(
            job_id=self.job_id, state=state, message=message, frames_done=done,
            frames_total=self._frames_total, elapsed_s=elapsed, eta_s=eta, fps=fps, stages=tuple(snaps), gpu=gpu,
        )
        try:
            self.progress(event)
        except Exception as exc:  # a broken UI callback must not kill the job
            log.warning("progress callback failed", error=str(exc))

    def _timed(self, name: str, start: float) -> None:
        self._timings[name] = round(time.perf_counter() - start, 3)

    # ---- planning ------------------------------------------------------------------------------------

    def capabilities(self, tools: FFmpegTools) -> Capabilities:
        if self._caps is None:
            self._caps = detect_capabilities(tools, cache_dir=self.settings.paths.cache_dir)
        return self._caps

    def _validate_paths(self, spec: JobSpec) -> None:
        if not spec.input.is_file():
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"input file not found: {spec.input}")
        if spec.output.resolve() == spec.input.resolve():
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, "output must differ from input")
        if spec.output.exists() and not (spec.overwrite or spec.resume):
            raise NeuralFrameError(
                ErrorCode.INVALID_INPUT, f"output already exists: {spec.output}", hint="pass --overwrite"
            )

    def resolve_model(self, spec: JobSpec) -> ModelSpec:
        if not spec.upscale:
            return self.registry.get("lanczos")
        preset = self.settings.preset(spec.preset)
        model_id = spec.model or preset.model or "lanczos"
        return self.registry.get(model_id)

    def plan(self, spec: JobSpec, tools: FFmpegTools, media: MediaInfo, timeline: FrameTimeline) -> JobPlan:
        settings = self.settings
        preset = settings.preset(spec.preset)
        caps = self.capabilities(tools)
        video = media.require_video()
        model = self.resolve_model(spec)
        notes: list[str] = []

        target = spec.target if spec.upscale else TargetSpec.native()
        geometry, geo_notes = plan_geometry(video, target, model.scale)
        notes += geo_notes

        codec = spec.codec or settings.encode.codec or preset.codec or "hevc"
        ten_bit = spec.ten_bit if spec.ten_bit is not None else settings.encode.ten_bit
        color = plan_color(video, geometry.out_height, settings.color, ten_bit=ten_bit)
        notes += list(color.notes)

        decode = plan_decode(
            video, spec.input, tools, caps, settings.decode, geometry,
            deinterlace=spec.deinterlace, denoise=spec.denoise,
        )
        notes += list(decode.notes)

        quality = spec.quality
        if quality is None:
            quality = settings.encode.quality
        if quality is None:
            quality = preset.quality.get(codec, default_quality(codec))
        requested_encoder = spec.encoder or settings.encode.encoder
        excluded: set[str] = set()
        while True:
            encoder = plan_encoder(
                caps,
                codec=codec,
                encoder=requested_encoder,
                quality=quality,
                speed=spec.speed or settings.encode.speed or preset.speed or "medium",
                ten_bit=ten_bit,
                color=color,
                out_width=geometry.out_width,
                out_height=geometry.out_height,
                extra_args=tuple(settings.encode.extra_args),
                exclude=frozenset(excluded),
            )
            out_fmt = RawVideoFormat.yuv420(
                geometry.out_width, geometry.out_height, 10 if encoder.ten_bit else 8
            )
            # Preflight with the real size/settings: hardware encoders can still fail at this point
            # (resolution limits, exhausted NVENC sessions) even though the capability probe passed.
            with tempfile.TemporaryDirectory(prefix="nf-preflight-") as tmp:
                problem = preflight(
                    tools, encoder, out_fmt, timeline.time_base, video.fps, Path(tmp)
                )
            if problem is None:
                break
            if requested_encoder != "auto":
                raise NeuralFrameError(
                    ErrorCode.ENCODE_FAILED, f"encoder {encoder.name} failed its preflight: {problem}"
                )
            log.warning("encoder preflight failed, trying the next one", encoder=encoder.name, error=problem)
            notes.append(f"{encoder.name} failed preflight ({problem[:120]}); falling back")
            excluded.add(encoder.name)
        notes += list(encoder.notes)

        mux = plan_mux(media, spec.output)
        notes += list(mux.warnings)

        ops = get_ops(settings.pipeline.ops).name
        mode = self.choose_mode(model)
        fg_engine, fg_order = self.resolve_framegen(spec, model, notes)
        self._vram_reserve = 0
        if fg_engine == "rife" and mode == "device" and settings.inference.device != "cpu":
            from neuralframe.framegen.memory import rife_reserve_bytes

            on_output = fg_order == "upscale_first"
            self._vram_reserve = rife_reserve_bytes(
                geometry.out_height if on_output else geometry.decoded_height,
                geometry.out_width if on_output else geometry.decoded_width,
                settings.inference.precision,
            )
        inference, weights = self.plan_model(spec, model, geometry, mode, notes)

        chunk_frames = spec.chunk_frames or settings.chunking.chunk_frames
        analysis = None
        if self.needs_analysis(model) or spec.interpolate:
            analysis = self.run_analysis(spec, tools, media, decode)
        total_frames = analysis.frames if analysis is not None else timeline.frame_count
        cuts = analysis.scene_cuts(settings.chunking.scene_threshold) if analysis is not None else []
        temporal_overlap = 0
        if model.is_temporal:
            configured = settings.chunking.overlap_frames
            temporal_overlap = model.default_overlap if configured is None else configured
            # A chunk plus its overlap on both sides must fit on the device at once.
            fit = (inference.temporal_frames or 64) - 2 * temporal_overlap if inference else 48
            chunk_frames = max(8, min(chunk_frames, settings.chunking.temporal_chunk_frames, fit))
            notes.append(f"temporal chunks of {chunk_frames} frames, {temporal_overlap} frames of overlap")
        chunks = plan_chunks(total_frames, chunk_frames, scene_cuts=cuts, force_cut_boundaries=model.is_temporal)
        if cuts:
            notes.append(f"scene cuts at frames {cuts[:12]}{' ...' if len(cuts) > 12 else ''}")
        source_chunks = chunks
        output_frames = total_frames
        retime = None
        framegen: dict[str, Any] | None = None
        fg_model: ModelSpec | None = None
        fg_weights: Path | None = None
        out_fps: Fraction | None = None
        if spec.interpolate:
            assert analysis is not None
            fg = settings.framegen
            engine = fg_engine
            out_fps = snap_fps(parse_fps(spec.target_fps or fg.fps), video.fps, ntsc_snap=fg.ntsc_snap)
            fg_cuts = analysis.scene_cuts(settings.chunking.scene_threshold)
            drops = detect_drops(
                analysis, fg_cuts, repair=spec.repair_drops or fg.repair_drops,
                dup_threshold=fg.dup_threshold, motion_threshold=fg.motion_threshold,
            )
            output_chunk = spec.chunk_frames or settings.chunking.chunk_frames
            if engine == "rife":
                from neuralframe.models.download import ensure_weights

                retime = plan_retime(analysis, fg_cuts, out_fps, drops, max_gap_s=fg.max_gap_s)
                output_frames = retime.frames
                fg_model = self.registry.get(fg.model)
                self._emit("planning", f"preparing frame generation model {fg_model.id}")
                fg_weights = ensure_weights(fg_model, settings.paths.models())
                framegen = {"engine": "rife", "model": fg_model.id, "order": fg_order, **retime.summary()}
                notes += retime.notes
            else:
                # Model-free fallback inside the decoder: drop repeated frames, then motion-compensated
                # interpolation onto the constant-rate grid.
                rate = f"{out_fps.numerator}/{out_fps.denominator}"
                end = float(analysis.times()[-1]) + drops.nominal_interval_s
                # minterpolate stops at the last input frame: clone one frame past it so the grid reaches the
                # end of the source, then trim the grid there.
                extra = (["mpdecimate"] if (spec.repair_drops or fg.repair_drops) != "off" else []) + [
                    "tpad=stop_mode=clone:stop=1",
                    f"minterpolate=fps={rate}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1",
                    f"trim=end={end:.6f}",
                ]
                decode = dataclasses.replace(decode, filters=(*decode.filters, *extra))
                span = end - float(analysis.times()[0])
                output_frames = max(1, round(span * float(out_fps)))
                fg_order = "interpolate_first"
                framegen = {"engine": "minterpolate", "fps": rate, "order": fg_order, "drops": drops.summary(),
                            "output_frames_estimate": output_frames}
                notes.append("frame generation: FFmpeg minterpolate (model-free fallback)")
            chunks = plan_chunks(output_frames, output_chunk)
            if engine != "rife":
                source_chunks = chunks  # the decoder already emits the constant-rate stream
            notes.append(
                f"frame generation to {float(out_fps):.3f} fps ({engine}, {fg_order.replace('_', ' ')}): "
                f"{total_frames} -> {output_frames} frames"
            )

        stat = spec.input.stat()
        essentials = {
            "version": __version__,
            "input": [str(spec.input.resolve()), stat.st_size, stat.st_mtime_ns],
            "model": model.id,
            "model_sha": model.weights.sha256 if model.weights else None,
            "geometry": geometry.summary(),
            "color": encoder.color.summary(),
            "decode": [decode.hwaccel is not None, list(decode.filters), decode.fmt.pix_fmt],
            "encoder": encoder.codec_args(),
            "chunks": [chunks.chunk_frames, chunks.total_frames, temporal_overlap],
            "tiles": [spec.tile_size, spec.tile_overlap],
            # Auto-chosen tile sizes depend on free VRAM at start time; they are pinned in the manifest on
            # first run and reused on resume instead of being part of the fingerprint.
            "inference": [mode, inference.backend, inference.precision, inference.overlap] if inference else None,
            "framegen": [framegen.get("engine"), str(out_fps), fg_order, spec.repair_drops,
                         fg_model.id if fg_model else None, output_frames] if framegen else None,
        }
        fingerprint = hashlib.sha256(json.dumps(essentials, sort_keys=True).encode()).hexdigest()
        job_key = hashlib.sha256(
            f"{spec.input.resolve()}|{spec.output.resolve()}|{fingerprint}".encode()
        ).hexdigest()[:16]
        return JobPlan(
            job_key=job_key, fingerprint=fingerprint, spec=spec, media=media, timeline=timeline, model=model,
            geometry=geometry, decode=decode, encoder=encoder, out_fmt=out_fmt, mux=mux, chunks=chunks,
            ops=ops, resize_kernel=model.resize_kernel if model.is_classical else "lanczos3",
            work_dir=settings.paths.work() / job_key, notes=notes, mode=mode, inference=inference,
            weights=weights, temporal_overlap=temporal_overlap, output_frames=output_frames,
            source_chunks=source_chunks, retime=retime, framegen_order=fg_order, framegen_model=fg_model,
            framegen_weights=fg_weights, framegen=framegen, out_fps=out_fps,
        )

    def needs_analysis(self, model: ModelSpec) -> bool:
        mode = self.settings.chunking.scene_detection
        return mode == "on" or (mode == "auto" and model.is_temporal)

    def run_analysis(
        self, spec: JobSpec, tools: FFmpegTools, media: MediaInfo, decode: DecodePlan
    ) -> FrameAnalysis:
        """Thumbnail analysis pass (scene cuts, motion, timestamps), cached per input + filter chain."""
        stat = spec.input.stat()
        key = hashlib.sha256(json.dumps(
            [str(spec.input.resolve()), stat.st_size, stat.st_mtime_ns, list(decode.filters), __version__]
        ).encode()).hexdigest()[:16]
        cache = self.settings.paths.cache_dir / "analysis" / f"{key}.npz"
        self._emit("planning", "analysing scenes and motion")
        result = analyse(tools, spec.input, media.require_video().index, decode, cache=cache,
                         loglevel=self.settings.ffmpeg.loglevel)
        self._analysis = result
        return result

    def resolve_framegen(self, spec: JobSpec, model: ModelSpec, notes: list[str]) -> tuple[str | None, str]:
        """Frame generation engine (None when off) and whether it runs before or after super-resolution."""
        if not spec.interpolate:
            return None, "upscale_first"
        fg = self.settings.framegen
        engine = spec.framegen_engine or fg.engine
        if engine == "auto":
            engine = "rife" if importlib.util.find_spec("torch") is not None else "minterpolate"
        order = spec.framegen_order or fg.order
        if model.is_temporal and order != "upscale_first":
            if order == "interpolate_first":
                notes.append("frame generation runs after the temporal model (it needs the original frames)")
            order = "upscale_first"
        elif order == "auto":
            # Per synthesised frame, interpolating first costs one more super-resolution pass and saves
            # RIFE(output size) - RIFE(source size). That pays off for classical resizing and compact models
            # (measured 3.3x and 1.4x faster on the reference clip) but not for heavy RRDBNet models.
            light = model.is_classical or model.architecture == "srvgg"
            order = "interpolate_first" if light else "upscale_first"
        return engine, order

    def choose_mode(self, model: ModelSpec) -> Literal["device", "host"]:
        requested = self.settings.pipeline.mode
        torch_ok = importlib.util.find_spec("torch") is not None
        if requested == "device" and not torch_ok:
            raise NeuralFrameError(
                ErrorCode.BACKEND_UNAVAILABLE, "pipeline.mode=device needs PyTorch", hint="pip install torch"
            )
        if not model.is_classical and not torch_ok:
            raise NeuralFrameError(
                ErrorCode.BACKEND_UNAVAILABLE,
                f"model {model.id!r} needs PyTorch for inference",
                hint="pip install neuralframe[torch] (CUDA wheels: https://pytorch.org)",
            )
        if requested == "host":
            return "host"
        if requested == "device":
            return "device"
        if not model.is_classical:
            return "device"
        if torch_ok and self.settings.inference.device != "cpu":
            from neuralframe.inference.factory import torch_cuda_available

            return "device" if torch_cuda_available() else "host"
        return "host"

    def plan_model(
        self,
        spec: JobSpec,
        model: ModelSpec,
        geometry: Geometry,
        mode: Literal["device", "host"],
        notes: list[str],
    ) -> tuple[InferencePlan | None, Path | None]:
        """Resolve weights (download + verify) and choose backend, device, tile and batch."""
        if model.is_classical or geometry.model_passes == 0:
            return None, None
        from neuralframe.inference.factory import (
            memory_budget,
            plan_inference,
            resolve_backend_name,
            resolve_device,
            torch_cuda_available,
            torch_cuda_device_count,
        )
        from neuralframe.models.download import ensure_weights

        settings = self.settings
        self._emit("planning", f"preparing model {model.id}")

        def on_download(done: int, total: int | None) -> None:
            pct = f" {100 * done / total:.0f}%" if total else ""
            self._emit("planning", f"downloading {model.id}{pct}")

        weights = ensure_weights(model, settings.paths.models(), progress=on_download)
        if settings.inference.device == "cuda" and not torch_cuda_available():
            raise NeuralFrameError(ErrorCode.BACKEND_UNAVAILABLE, "inference.device=cuda but CUDA is not available")
        device = resolve_device(spec.gpus or settings.inference.gpus, force_cpu=settings.inference.device == "cpu")
        backend = resolve_backend_name(spec.backend or settings.inference.backend, device)
        budget = memory_budget(
            device,
            safety_margin=settings.vram.safety_margin,
            max_vram_gb=spec.max_vram_gb or settings.vram.max_vram_gb,
        ) - self._vram_reserve
        if self._vram_reserve:
            notes.append(f"{self._vram_reserve // 2**20} MB of VRAM kept for frame interpolation")
        if model.is_temporal:
            from neuralframe.inference.factory import device_capacity, plan_temporal

            capacity = device_capacity(device, safety_margin=settings.vram.safety_margin,
                                       max_vram_gb=spec.max_vram_gb or settings.vram.max_vram_gb)
            capacity = max(capacity - self._vram_reserve, 256 * 2**20)
            inference = plan_temporal(
                model, (geometry.decoded_height, geometry.decoded_width), device=device,
                precision=settings.inference.precision, capacity_bytes=capacity,
                requested_backend=spec.backend or settings.inference.backend,
            )
            notes += list(inference.notes)
            notes.append(f"temporal inference: torch on {device} ({inference.precision}), up to "
                          f"{inference.temporal_frames} frames per sequence")
            return inference, weights
        preset = settings.preset(spec.preset)
        overlap = spec.tile_overlap
        if overlap is None:
            overlap = settings.inference.tile_overlap
        if overlap is None:
            overlap = preset.tile_overlap if preset.tile_overlap is not None else 32
        # Size of the largest pass input (cascaded passes run on already-upscaled frames).
        h = geometry.decoded_height * model.scale ** max(geometry.model_passes - 1, 0)
        w = geometry.decoded_width * model.scale ** max(geometry.model_passes - 1, 0)
        inference = plan_inference(
            model, (h, w), device=device, backend=backend, precision=settings.inference.precision,
            budget_bytes=budget, tile=spec.tile_size or settings.inference.tile_size, overlap=overlap,
            batch=spec.batch_size or settings.inference.batch_size,
        )
        devices: tuple[str, ...] = (str(device),)
        if device.type == "cuda" and not model.is_temporal:
            gpus = spec.gpus or settings.inference.gpus
            if gpus:
                devices = tuple(str(resolve_device([g])) for g in gpus)
            else:
                devices = tuple(f"cuda:{i}" for i in range(torch_cuda_device_count()))
        inference = dataclasses.replace(inference, devices=devices)
        if len(devices) > 1:
            notes.append(f"{len(devices)} inference workers pulling from one queue: {', '.join(devices)}")
        notes += list(inference.notes)
        notes.append(
            f"inference: {backend} on {device} ({inference.precision}), "
            + (f"tile {inference.tile}px overlap {inference.overlap}" if inference.tile else "whole frames")
            + f", batch {inference.batch}"
        )
        return inference, weights

    # ---- execution -----------------------------------------------------------------------------------

    def _model_runner(self, plan: JobPlan, *, calibrate: bool = True) -> Any:
        assert plan.inference is not None and plan.weights is not None
        runner, plan.inference = self._calibrated_runner(
            plan.model, plan.weights, plan.inference, plan.geometry, calibrate=calibrate
        )
        return runner

    def _calibrated_runner(
        self, model: ModelSpec, weights: Path | None, inf: InferencePlan, geometry: Geometry, *, calibrate: bool
    ) -> tuple[Any, InferencePlan]:
        """Runner with OOM recovery; calibrated on a dry run, falling back to other backends/devices if
        even the smallest configuration does not fit (``inference.backend: auto``)."""
        attempts: list[InferencePlan] = [inf]
        if self.settings.inference.backend == "auto":
            if inf.backend != "torch":
                attempts.append(dataclasses.replace(inf, backend="torch"))
            if inf.device != "cpu":
                attempts.append(dataclasses.replace(
                    inf, backend="torch", device="cpu", precision="fp32", batch=1, frames_per_call=1,
                    tile=inf.tile or 256, overlap=inf.overlap or 16,
                ))
        last_error: NeuralFrameError | None = None
        for candidate in attempts:
            runner = self._model_runner_for(model, weights, candidate, geometry)
            vram = runner.enable_recovery(max_levels=3, on_adapt=lambda a: self._adaptations.append(a.as_dict()))
            if not (calibrate and self.settings.vram.calibrate and runner.device.type == "cuda"):
                return runner, candidate
            m = max(model.pad_multiple, 1)
            hw = (-(-geometry.decoded_height // m) * m, -(-geometry.decoded_width // m) * m)
            try:
                self._emit("planning", "calibrating VRAM")
                result = vram.calibrate(
                    hw, frames=candidate.frames_per_call, safety_margin=self.settings.vram.safety_margin,
                    reserve_bytes=self._vram_reserve,
                )
            except NeuralFrameError as exc:
                if exc.code is not ErrorCode.OOM_UNRECOVERABLE:
                    raise
                last_error = exc
                runner.backend.close()
                self._fallbacks.append(f"{candidate.backend} on {candidate.device} ran out of memory; trying next")
                continue
            self._calibration = result
            tuned = dataclasses.replace(candidate, batch=runner.batch, tile=runner.tile, overlap=runner.overlap)
            if runner.tile is not None:
                tuned = dataclasses.replace(tuned, frames_per_call=1)
            else:
                tuned = dataclasses.replace(tuned, frames_per_call=min(candidate.frames_per_call, runner.batch))
            return runner, tuned
        assert last_error is not None
        raise last_error

    def _model_runner_for(
        self, model: ModelSpec, weights: Path | None, inf: InferencePlan, geometry: Geometry
    ) -> Any:
        """Create and load the backend, wrapped in a tiling/batching :class:`ModelRunner`."""
        import torch

        from neuralframe.inference.factory import create_backend
        from neuralframe.inference.runner import ModelRunner

        assert weights is not None
        settings = self.settings
        profile = self._shape_profile(model, inf, geometry)

        def make(name: str) -> Any:
            backend = create_backend(
                name, torch.device(inf.device), inf.precision,
                deterministic=settings.inference.deterministic,
                models_dir=settings.paths.models(), engines_dir=settings.paths.engines(),
                onnx_providers=settings.inference.onnx_providers,
                status=lambda msg: self._emit("planning", msg),
            )
            backend.load(model, weights, profile)
            return backend

        try:
            backend = make(inf.backend)
        except NeuralFrameError as exc:
            requested = self.settings.inference.backend
            if requested != "auto" or inf.backend == "torch" or exc.code not in (
                ErrorCode.ENGINE_BUILD_FAILED, ErrorCode.BACKEND_UNAVAILABLE
            ):
                raise
            log.warning("backend unavailable, falling back to torch", backend=inf.backend, error=str(exc))
            self._fallbacks.append(f"{inf.backend} failed ({exc.message}); fell back to torch")
            backend = make("torch")
        return ModelRunner(backend, model, tile=inf.tile, overlap=inf.overlap, batch=inf.batch)

    @staticmethod
    def _shape_profile(model: ModelSpec, inf: InferencePlan, geometry: Geometry) -> Any:
        """Optimisation profile covering every tile/frame shape the runner will submit (all passes)."""
        from neuralframe.chunking.tiling import TileLayout
        from neuralframe.inference.backend import ShapeProfile

        m = max(model.pad_multiple, 1)
        shapes = []
        h, w = geometry.decoded_height, geometry.decoded_width
        for _ in range(max(geometry.model_passes, 1)):
            hp, wp = -(-h // m) * m, -(-w // m) * m
            tile = None if inf.tile is None else max(m, inf.tile - inf.tile % m)
            layout = TileLayout.plan(hp, wp, tile, inf.overlap)
            shapes.append((layout.tile_h, layout.tile_w))
            h, w = h * model.scale, w * model.scale
        opt = shapes[0]
        max_hw = (max(s[0] for s in shapes), max(s[1] for s in shapes))
        # Allow tiles down to 64 px so the OOM ladder can shrink tiles without rebuilding engines.
        min_hw = (min(64, *(s[0] for s in shapes)), min(64, *(s[1] for s in shapes)))
        return ShapeProfile(min_hw=min_hw, opt_hw=opt, max_hw=max_hw, max_batch=inf.batch, opt_batch=inf.batch)

    def create_processors(self, plan: JobPlan, *, calibrate: bool = True) -> list[BatchProcessor]:
        """One processor per inference worker; the first calibrates, the others reuse its settings."""
        first = self.create_processor(plan, calibrate=calibrate)
        if plan.mode != "device" or plan.inference is None or len(plan.inference.workers) <= 1:
            return [first]
        processors = [first]
        base = plan.inference
        for device in base.workers[1:]:
            worker_plan = dataclasses.replace(plan, inference=dataclasses.replace(base, device=device))
            processors.append(self.create_processor(worker_plan, calibrate=False))
        return processors

    def create_processor(self, plan: JobPlan, *, calibrate: bool = True) -> BatchProcessor:
        """Host mode: FrameProcessor on float RGB; device mode: DeviceProcessor on raw YUV."""
        neural = plan.inference is not None
        if plan.passthrough:
            return IdentityProcessor()
        if plan.model.is_temporal and plan.inference is not None and plan.weights is not None:
            import torch

            from neuralframe.inference.temporal import TemporalModel, TemporalProcessor

            return TemporalProcessor(TemporalModel(
                plan.model, plan.weights, torch.device(plan.inference.device), plan.inference.precision,
                strip_rows=plan.inference.strip_rows,
            ))
        if plan.mode == "host":
            if not neural:
                return IdentityProcessor()
            from neuralframe.inference.host import HostModelProcessor

            assert plan.inference is not None
            runner = self._model_runner(plan, calibrate=calibrate)
            return HostModelProcessor(runner, plan.geometry.model_passes, plan.inference.frames_per_call)
        import torch

        from neuralframe.inference.device import DeviceProcessor
        from neuralframe.inference.factory import resolve_device

        runner = self._model_runner(plan, calibrate=calibrate) if neural else None
        force_cpu = self.settings.inference.device == "cpu"
        device = runner.device if runner is not None else resolve_device(plan.spec.gpus, force_cpu=force_cpu)
        return DeviceProcessor(
            device=torch.device(device),
            in_fmt=plan.decode.fmt,
            out_fmt=plan.out_fmt,
            color=plan.encoder.color,
            runner=runner,
            passes=plan.geometry.model_passes,
            kernel=plan.resize_kernel,
            frames_per_call=plan.inference.frames_per_call if plan.inference else 4,
        )

    def _queue_sizes(self, plan: JobPlan, processor: BatchProcessor) -> dict[str, int]:
        budget = self.settings.pipeline.ram_budget_gb
        if budget is None:
            _, available = host_memory()
            budget_bytes = int(min(available * 0.25, 4 * 2**30))
        else:
            budget_bytes = int(budget * 2**30)
        g = plan.geometry
        scale = g.model_scale**g.model_passes
        if plan.mode == "device":
            item_bytes = {"inference": plan.decode.fmt.frame_bytes, "encode": plan.out_fmt.frame_bytes}
        else:
            item_bytes = {
                "preprocess": plan.decode.fmt.frame_bytes,
                "inference": 3 * g.decoded_width * g.decoded_height * 4,
                "postprocess": 3 * g.decoded_width * scale * g.decoded_height * scale * 4,
                "encode": plan.out_fmt.frame_bytes,
            }
        share = budget_bytes // len(item_bytes)
        sizes = {name: int(max(2, min(32, share // max(b, 1)))) for name, b in item_bytes.items()}
        sizes["inference"] = max(sizes["inference"], 2 * max(1, processor.batch_size))
        return sizes

    def run(self, spec: JobSpec) -> JobResult:
        bind_context(job_id=self.job_id)
        self._t0 = time.perf_counter()
        warnings: list[str] = []
        try:
            return self._run(spec, warnings)
        except JobCancelled:
            self._emit("cancelled", "cancelled; finished chunks are kept for --resume")
            raise
        except NeuralFrameError as exc:
            self._emit("failed", str(exc))
            raise
        except Exception as exc:
            self._emit("failed", str(exc))
            raise NeuralFrameError(ErrorCode.INTERNAL, f"unexpected error: {exc}") from exc
        finally:
            clear_context("job_id")

    def _run(self, spec: JobSpec, warnings: list[str]) -> JobResult:
        settings = self.settings
        self._emit("probing")
        t = time.perf_counter()
        tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
        self._validate_paths(spec)
        media = probe(spec.input, tools)
        video = media.require_video()
        if media.is_image:
            from neuralframe.core.image import run_image_job

            self._timed("probe_s", t)
            return run_image_job(self, spec, tools, media)
        container_for(spec.output)
        timeline = scan_frames(spec.input, tools, video.index)
        self._timed("probe_s", t)

        self._emit("planning")
        t = time.perf_counter()
        plan = self.plan(spec, tools, media, timeline)
        self._frames_total = plan.total_frames
        warnings += plan.notes
        for note in plan.notes:
            log.info("plan", note=note)
        log.info(
            "job planned",
            input=str(spec.input), output=str(spec.output), model=plan.model.id,
            size=f"{plan.geometry.out_width}x{plan.geometry.out_height}", encoder=plan.encoder.name,
            hwaccel=plan.decode.hwaccel, ops=plan.ops, chunks=len(plan.chunks.chunks),
        )
        if self.on_plan is not None:
            self.on_plan(plan)
        self._timed("plan_s", t)

        store = ChunkStore.open(plan.work_dir, plan.job_key, plan.fingerprint, resume=spec.resume)
        pinned = store.manifest.meta.get("inference") if spec.resume else None
        if plan.inference is not None and pinned:
            # Reuse the calibrated tile/batch of the interrupted run so resumed chunks match bit for bit.
            plan.inference = dataclasses.replace(
                plan.inference, backend=pinned["backend"], device=pinned["device"],
                precision=pinned["precision"], tile=pinned["tile"], overlap=pinned["tile_overlap"],
                batch=pinned["batch"], frames_per_call=pinned["frames_per_call"],
            )
        processors = self.create_processors(plan, calibrate=pinned is None)
        if plan.inference is not None and not pinned:
            store.set_meta(inference=plan.inference.summary())
        warnings += self._fallbacks
        done_ids = set(store.completed())
        if done_ids:
            log.info("resuming", completed_chunks=sorted(done_ids))
        self._frames_skipped = sum(r.frames for r in store.completed().values())

        gpus = list(range(gpu_count()))
        self._sampler = ResourceSampler(gpus).start()
        t = time.perf_counter()
        try:
            if len(done_ids) < len(plan.chunks.chunks):
                self._run_pipeline(plan, tools, processors, store, done_ids)
            else:
                self._resume_info = {"mode": "complete"}
                log.info("all chunks already complete; skipping to mux")
        finally:
            resources = self._sampler.stop()
            processor_stats = processors[0].stats()
            if len(processors) > 1:
                processor_stats["workers"] = [p.stats() for p in processors]
            for p in processors:
                p.close()
        self._timed("pipeline_s", t)
        if self._decoder is not None:
            warnings += self._decoder.fallbacks

        self._emit("muxing")
        t = time.perf_counter()
        segments = store.ordered_segments()
        joined = plan.work_dir / "video.mp4"
        concat_segments(segments, joined)
        mux_final(tools, joined, media, plan.mux, spec.output, loglevel=settings.ffmpeg.loglevel)
        self._timed("mux_s", t)

        self._emit("verifying")
        t = time.perf_counter()
        frames_out = sum(r.frames for r in store.completed().values())
        rate = plan.out_fps if plan.out_fps is not None else video.fps
        frame_s = float(1 / rate) if rate > 0 else 0.04
        sync = verify_output(
            tools, media, spec.output, expected_frames=frames_out, frame_duration_s=frame_s,
        )
        self._timed("verify_s", t)
        self._timings["total_s"] = round(time.perf_counter() - self._t0, 3)

        report = self._report(plan, store, resources, processor_stats, sync.as_dict(), warnings, frames_out)
        report_path = spec.output.with_name(spec.output.stem + ".report.json")
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        if not sync.ok and settings.pipeline.verify_sync:
            raise NeuralFrameError(
                ErrorCode.SYNC_CHECK_FAILED,
                "output failed verification: " + "; ".join(sync.problems),
                details={"report": str(report_path)},
            )
        keep = spec.keep_work_dir if spec.keep_work_dir is not None else settings.pipeline.keep_work_dir
        if not keep:
            store.cleanup()
        self._emit("done", f"wrote {spec.output}")
        return JobResult(job_id=self.job_id, output=spec.output, report_path=report_path, report=report)

    def _run_pipeline(
        self,
        plan: JobPlan,
        tools: FFmpegTools,
        processors: list[BatchProcessor],
        store: ChunkStore,
        done_ids: set[int],
    ) -> None:
        """decode → [retime] → super-resolution chain → [retime] → encode.

        Without frame generation source and output indices coincide. With it, the retime stage maps source
        frames onto the constant-rate output grid, before or after upscaling (``framegen.order``); encoding,
        chunking and resume then work on output indices.
        """
        settings = self.settings
        cancel = self.cancel
        processor = processors[0]
        workers = len(processors)
        sizes = self._queue_sizes(plan, processor)
        retime = plan.retime
        temporal = plan.model.is_temporal

        def out_done(index: int) -> bool:
            return plan.chunks.chunk_of(index).id in done_ids

        first_index = 0
        while first_index < plan.total_frames and out_done(first_index):
            first_index = plan.chunks.chunk_of(first_index).end

        # Resume by seeking when output == source indices, finished chunks form a prefix (always true for
        # in-order encoding) and no stateful filter or temporal model needs earlier frames.
        resume_after: Fraction | None = None
        records = store.completed()
        prefix = sorted(records)
        stateful_filters = ("hqdn3d", "minterpolate", "mpdecimate", "tpad", "trim")
        stateful = any(f.startswith(stateful_filters) for f in plan.decode.filters)
        tb_meta = store.manifest.time_base
        if (prefix and prefix == list(range(len(prefix))) and not stateful and tb_meta and not temporal
                and retime is None):
            last = records[prefix[-1]]
            resume_after = Fraction(last.last_pts) * Fraction(tb_meta)
            first_index = last.end
            log.info("resuming by seek", after_s=round(float(resume_after), 3), first_index=first_index)
            self._resume_info = {"mode": "seek", "after_s": round(float(resume_after), 6), "first_index": first_index}
        elif done_ids:
            self._resume_info = {"mode": "decode-skip", "first_index": first_index}

        decoder = ResilientDecoder(
            tools, plan.spec.input, plan.media.require_video().index, plan.decode,
            loglevel=settings.ffmpeg.loglevel, resume_after=resume_after,
            first_index=first_index if resume_after is not None else 0,
        ).start()
        self._decoder = decoder
        if store.manifest.time_base is None:
            store.manifest.time_base = f"{decoder.time_base.numerator}/{decoder.time_base.denominator}"
            store.save()

        out_tb = retime.time_base if retime is not None else decoder.time_base
        out_rate = retime.fps if retime is not None else decoder.frame_rate

        def make_encoder(path: Path) -> SegmentEncoder:
            return SegmentEncoder(tools, path, plan.out_fmt, out_tb, out_rate, plan.encoder,
                                  loglevel=settings.ffmpeg.loglevel)

        # Which source frames can be skipped on a resumed run.
        skip_decoded: Any = None
        if done_ids and resume_after is None and not temporal:
            if retime is None:
                skip_decoded = out_done
            else:
                needed_from = retime.needed_sources_from(first_index)
                skip_decoded = lambda index: index < needed_from  # noqa: E731

        def chunk_done(chunk: Any, frames: int) -> None:
            log.info("chunk committed", chunk_id=chunk.id, frames=frames)
            if self.on_chunk_done is not None:
                self.on_chunk_done(chunk.id, len(plan.chunks.chunks))

        # Several workers finish frames out of order; cap frames in flight so reordering stays bounded.
        limiter = (
            InFlightLimiter(sizes["inference"] + sizes["encode"] + workers * (4 * processor.batch_size + 4), cancel)
            if workers > 1 else None
        )
        self._queues = {}
        q_src: StageQueue[Any] = StageQueue("source", sizes["inference"], cancel)
        stages: list[Any] = [DecodeStage(cancel, decoder.frames(), q_src, skip=skip_decoded, limiter=limiter)]
        current = q_src

        interpolator = self._make_interpolator(plan) if retime is not None else None
        self._interpolator = interpolator
        if retime is not None:
            from neuralframe.framegen.interpolate import RetimeStage
        if retime is not None and plan.framegen_order == "interpolate_first":
            # Frame generation at source resolution; super-resolution then runs on the constant-rate stream.
            source_color = dataclasses.replace(
                plan.encoder.color, out_matrix=plan.encoder.color.in_matrix, out_range=plan.encoder.color.in_range,
                out_chroma=plan.encoder.color.in_chroma,
            )
            q_rt: StageQueue[Any] = StageQueue("retimed", sizes["inference"], cancel)
            stages.append(RetimeStage(cancel, current, q_rt, plan=retime, interpolator=interpolator,
                                      fmt=plan.decode.fmt, color=source_color, skip_output=out_done, limiter=limiter))
            self._queues["framegen"] = current
            current = q_rt
            limiter = None

        if not plan.passthrough:
            sr_out: StageQueue[Any] = StageQueue("sr-out", sizes["encode"], cancel, producers=workers)
            stages += self._sr_chain(plan, processors, current, sr_out, sizes,
                                     done_ids if retime is None else set(), skip_decoded is not None)
            current = sr_out
        if retime is not None and plan.framegen_order != "interpolate_first":
            q_rt2: StageQueue[Any] = StageQueue("retimed", sizes["encode"], cancel)
            stages.append(RetimeStage(cancel, current, q_rt2, plan=retime, interpolator=interpolator,
                                      fmt=plan.out_fmt, color=plan.encoder.color, skip_output=out_done,
                                      limiter=limiter))
            self._queues["framegen"] = current
            current = q_rt2
            limiter = None

        stages.append(EncodeStage(
            cancel, current, chunks=plan.chunks, store=store, make_encoder=make_encoder, first_index=first_index,
            done_chunks=done_ids, on_chunk_done=chunk_done, limiter=limiter,
        ))
        self._queues["encode"] = current
        self._stages = stages
        self._encode_stage = stages[-1]
        self._pipeline_t0 = time.perf_counter()
        self._emit("running")
        try:
            with self._torch_memory_cap(processors, interpolator):
                StageRunner(stages, cancel).run(on_tick=lambda: self._emit("running"))
        finally:
            decoder.close()
            if interpolator is not None:
                interpolator.close()

    def _torch_memory_cap(self, processors: list[BatchProcessor], interpolator: Any) -> Any:
        """Context bounding PyTorch's GPU cache for the run (see :func:`cap_torch_cache`)."""
        devices: list[Any] = []
        for proc in processors:  # device processors, host/temporal processors through their model
            for owner in (proc, getattr(proc, "runner", None), getattr(proc, "model", None)):
                device = getattr(owner, "device", None)
                if device is not None:
                    devices.append(device)
                    break
        if interpolator is not None:
            devices.append(interpolator.device)
        if not any(getattr(d, "type", None) == "cuda" for d in devices):
            return contextlib.nullcontext()
        from neuralframe.inference.vram import cap_torch_cache

        return cap_torch_cache(devices, self.settings.vram.safety_margin)

    def _sr_chain(
        self,
        plan: JobPlan,
        processors: list[BatchProcessor],
        q_in: StageQueue[Any],
        q_out: StageQueue[Any],
        sizes: dict[str, int],
        temporal_done: set[int],
        resumed_with_skip: bool,
    ) -> list[Any]:
        """Stages turning decoded source frames (raw YUV) into upscaled output-format frames."""
        settings, cancel = self.settings, self.cancel
        processor = processors[0]
        workers = len(processors)
        stages: list[Any] = []
        if plan.model.is_temporal and plan.inference is not None:
            from neuralframe.inference.temporal import TemporalProcessor, TemporalStage

            assert isinstance(processor, TemporalProcessor)
            assert plan.source_chunks is not None
            stage = TemporalStage(
                cancel, q_in, q_out, model=processor.model, chunks=plan.source_chunks,
                overlap=plan.temporal_overlap, in_fmt=plan.decode.fmt, out_fmt=plan.out_fmt,
                color=plan.encoder.color, kernel=plan.resize_kernel, done_chunks=temporal_done,
                replicate_edges=plan.model.input == "window",
            )
            processor.stage = stage
            self._queues["inference"] = q_in
            return [stage]
        use_streams = False
        if plan.mode == "device":
            from neuralframe.inference.device import DeviceProcessor

            use_streams = (settings.pipeline.cuda_streams and isinstance(processor, DeviceProcessor)
                           and processor.device.type == "cuda")
        if use_streams:
            # upload (H2D + yuv->rgb) -> inference -> download (resize + rgb->yuv + D2H), each GPU stage on
            # its own CUDA stream; with several GPUs every worker runs its own chain from the shared queue.
            from neuralframe.inference.device import DeviceProcessor
            from neuralframe.inference.gpu_stages import DownloadStage, InferStage, UploadStage

            self._queues["upload"] = q_in
            for w, proc in enumerate(processors):
                assert isinstance(proc, DeviceProcessor)
                sfx = f"@{w}" if workers > 1 else ""
                q_inf: StageQueue[Any] = StageQueue("inference" + sfx, 2, cancel)
                q_dl: StageQueue[Any] = StageQueue("postprocess" + sfx, 2, cancel)
                self._queues["inference" + sfx] = q_inf
                self._queues["postprocess" + sfx] = q_dl
                fpc = proc.batch_size
                stages += [
                    UploadStage(cancel, q_in, q_inf, device=proc.device, fmt=plan.decode.fmt,
                                color=plan.encoder.color, frames_per_call=fpc, suffix=sfx),
                    InferStage(cancel, q_inf, q_dl, device=proc.device, runner=proc.runner,
                               passes=proc.passes, suffix=sfx),
                    DownloadStage(cancel, q_dl, q_out, device=proc.device, fmt=plan.out_fmt,
                                  color=plan.encoder.color, kernel=plan.resize_kernel, frames_per_call=fpc,
                                  suffix=sfx),
                ]
            return stages
        if plan.mode == "device":
            self._queues.update({"inference" + (f"@{w}" if workers > 1 else ""): q_in for w in range(workers)})
            return [
                ProcessStage(cancel, q_in, q_out, processor=proc, name="inference" + (f"@{w}" if workers > 1 else ""))
                for w, proc in enumerate(processors)
            ]
        ops: FrameOps = get_ops(settings.pipeline.ops)
        q_rgb: StageQueue[Any] = StageQueue("inference", sizes["inference"], cancel)
        q_up: StageQueue[Any] = StageQueue("postprocess", sizes["postprocess"], cancel)
        self._queues.update({"preprocess": q_in, "inference": q_rgb, "postprocess": q_up})
        return [
            PreprocessStage(cancel, q_in, q_rgb, ops=ops, fmt=plan.decode.fmt, color=plan.encoder.color),
            ProcessStage(cancel, q_rgb, q_up, processor=processor),
            PostprocessStage(cancel, q_up, q_out, ops=ops, out_fmt=plan.out_fmt, color=plan.encoder.color,
                             kernel=plan.resize_kernel),
        ]

    def _make_interpolator(self, plan: JobPlan) -> Any:
        if plan.framegen_weights is None or plan.framegen_model is None:
            return None  # model-free path (FFmpeg minterpolate runs inside the decoder)
        import torch

        from neuralframe.framegen.interpolate import Interpolator
        from neuralframe.inference.factory import resolve_device

        settings = self.settings
        device = resolve_device(plan.spec.gpus or settings.inference.gpus, force_cpu=settings.inference.device == "cpu")
        self._emit("planning", "loading frame interpolation model")
        return Interpolator(plan.framegen_model, plan.framegen_weights, torch.device(device),
                            settings.inference.precision, scale=settings.framegen.rife_scale)

    def _report(
        self,
        plan: JobPlan,
        store: ChunkStore,
        resources: dict[str, Any],
        processor_stats: dict[str, object],
        sync: dict[str, Any],
        warnings: list[str],
        frames_out: int,
    ) -> dict[str, Any]:
        caps = self._caps
        stages = [
            {
                "name": s.name,
                "frames": s.metrics.frames,
                "fps": round(s.metrics.fps, 2),
                "busy_fps": round(s.metrics.busy_fps, 2),
                "utilisation": round(s.metrics.utilisation, 3),
                "max_queue_depth": self._queues[s.name].max_depth if s.name in self._queues else None,
                "queue_capacity": self._queues[s.name].maxsize if s.name in self._queues else None,
            }
            for s in self._stages
        ]
        pipeline_s = self._timings.get("pipeline_s", 0.0)
        processed = sum(s["frames"] for s in stages if s["name"] == "encode")
        substages: list[dict[str, Any]] = []
        timings = processor_stats.get("timings") if isinstance(processor_stats, dict) else None
        if isinstance(timings, dict) and timings.get("frames"):
            frames = int(timings["frames"])
            for name, key in (("preprocess", "preprocess_s"), ("inference", "inference_s"),
                              ("postprocess", "postprocess_s")):
                busy = float(timings.get(key, 0.0))
                substages.append({"name": name, "busy_s": round(busy, 3),
                                  "busy_fps": round(frames / busy, 2) if busy > 0 else None})
        return {
            "neuralframe": __version__,
            "job_id": self.job_id,
            "status": "done" if sync.get("ok") else "verification_failed",
            "input": media_summary(plan.media, plan.timeline),
            "output": {
                "path": str(plan.spec.output),
                "size": f"{plan.geometry.out_width}x{plan.geometry.out_height}",
                "frames": frames_out,
            },
            "settings": json.loads(plan.spec.model_dump_json()),
            "plan": plan.summary(),
            "timings": self._timings,
            "throughput": {
                "frames_processed_this_run": processed,
                "frames_resumed": self._frames_skipped,
                "resume": self._resume_info,
                "pipeline_fps": round(processed / pipeline_s, 2) if pipeline_s > 0 else None,
                "stages": stages,
                "device_substages": substages,
                "bottleneck": min(stages, key=lambda s: s["busy_fps"] or float("inf"))["name"] if stages else None,
            },
            "processor": processor_stats,
            "vram": {"calibration": self._calibration, "adaptations": self._adaptations},
            "framegen": (
                {**plan.framegen, "runtime": next((s.stats() for s in self._stages if s.name == "framegen"), None)}
                if plan.framegen else None
            ),
            "resources": resources,
            "sync": sync,
            "chunks": {"completed": len(store.completed()), "planned": len(plan.chunks.chunks)},
            "warnings": warnings,
            "hardware": {
                "gpus": [g.model_dump() for g in caps.gpus] if caps else [],
                "nvidia_driver": caps.nvidia_driver if caps else None,
                "cpu": caps.cpu if caps else None,
                "ffmpeg": caps.ffmpeg_version if caps else None,
                "torch": caps.torch if caps else None,
                "tensorrt": caps.tensorrt if caps else None,
            },
        }

