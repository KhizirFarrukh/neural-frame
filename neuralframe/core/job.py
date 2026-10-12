"""Job specification, planning and execution — the single entry point used by CLI, API and GUI."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuralframe import __version__
from neuralframe.chunking.planner import ChunkPlan, plan_fixed_chunks
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
from neuralframe.core.runtime import CancelToken, StageQueue, StageRunner
from neuralframe.core.stages import DecodeStage, EncodeStage, PostprocessStage, PreprocessStage, ProcessStage
from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.inference.processor import BatchProcessor, IdentityProcessor
from neuralframe.io.decoder import DecodePlan, FrameDecoder
from neuralframe.io.encoder import EncoderPlan, SegmentEncoder, default_quality
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

    @property
    def total_frames(self) -> int:
        return self.timeline.frame_count

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
    ) -> None:
        self.settings = settings
        self.on_plan = on_plan
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

        geometry, geo_notes = plan_geometry(video, spec.target, model.scale)
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
        encoder = plan_encoder(
            caps,
            codec=codec,
            encoder=spec.encoder or settings.encode.encoder,
            quality=quality,
            speed=spec.speed or settings.encode.speed or preset.speed or "medium",
            ten_bit=ten_bit,
            color=color,
            out_width=geometry.out_width,
            out_height=geometry.out_height,
            extra_args=tuple(settings.encode.extra_args),
        )
        notes += list(encoder.notes)
        out_fmt = RawVideoFormat.yuv420(geometry.out_width, geometry.out_height, 10 if encoder.ten_bit else 8)

        mux = plan_mux(media, spec.output)
        notes += list(mux.warnings)

        chunk_frames = spec.chunk_frames or settings.chunking.chunk_frames
        chunks = plan_fixed_chunks(timeline.frame_count, chunk_frames)
        ops = get_ops(settings.pipeline.ops).name
        mode = self.choose_mode(model)
        inference, weights = self.plan_model(spec, model, geometry, mode, notes)

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
            "chunks": [chunks.chunk_frames, chunks.total_frames],
            "tiles": [spec.tile_size, spec.tile_overlap],
            # Auto-chosen tile sizes depend on free VRAM at start time; they are pinned in the manifest on
            # first run and reused on resume instead of being part of the fingerprint.
            "inference": [mode, inference.backend, inference.precision, inference.overlap] if inference else None,
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
            weights=weights,
        )

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
        )
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
        notes += list(inference.notes)
        notes.append(
            f"inference: {backend} on {device} ({inference.precision}), "
            + (f"tile {inference.tile}px overlap {inference.overlap}" if inference.tile else "whole frames")
            + f", batch {inference.batch}"
        )
        return inference, weights

    # ---- execution -----------------------------------------------------------------------------------

    def _model_runner(self, plan: JobPlan) -> Any:
        assert plan.inference is not None and plan.weights is not None
        return self._model_runner_for(plan.model, plan.weights, plan.inference, plan.geometry)

    def _model_runner_for(
        self, model: ModelSpec, weights: Path | None, inf: InferencePlan, geometry: Geometry
    ) -> Any:
        """Create and load the backend, wrapped in a tiling/batching :class:`ModelRunner`."""
        import torch

        from neuralframe.inference.backend import ShapeProfile
        from neuralframe.inference.factory import create_backend
        from neuralframe.inference.runner import ModelRunner

        assert weights is not None
        backend = create_backend(
            inf.backend, torch.device(inf.device), inf.precision,
            deterministic=self.settings.inference.deterministic,
        )
        tile_h = inf.tile or geometry.decoded_height
        tile_w = inf.tile or geometry.decoded_width
        backend.load(model, weights, ShapeProfile.fixed(tile_h, tile_w, inf.batch))
        return ModelRunner(backend, model, tile=inf.tile, overlap=inf.overlap, batch=inf.batch)

    def create_processor(self, plan: JobPlan) -> BatchProcessor:
        """Host mode: FrameProcessor on float RGB; device mode: DeviceProcessor on raw YUV."""
        neural = plan.inference is not None
        if plan.mode == "host":
            if not neural:
                return IdentityProcessor()
            from neuralframe.inference.host import HostModelProcessor

            assert plan.inference is not None
            return HostModelProcessor(
                self._model_runner(plan), plan.geometry.model_passes, plan.inference.frames_per_call
            )
        import torch

        from neuralframe.inference.device import DeviceProcessor
        from neuralframe.inference.factory import resolve_device

        runner = self._model_runner(plan) if neural else None
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
        if plan.inference is not None:
            pinned = store.manifest.meta.get("inference")
            if spec.resume and pinned:
                # Reuse the tile layout of the interrupted run so resumed chunks match bit for bit.
                plan.inference = dataclasses.replace(
                    plan.inference, tile=pinned["tile"], overlap=pinned["tile_overlap"]
                )
            else:
                store.set_meta(inference=plan.inference.summary())
        processor = self.create_processor(plan)
        done_ids = set(store.completed())
        if done_ids:
            log.info("resuming", completed_chunks=sorted(done_ids))
        self._frames_skipped = sum(r.frames for r in store.completed().values())

        gpus = list(range(gpu_count()))
        self._sampler = ResourceSampler(gpus).start()
        t = time.perf_counter()
        try:
            if len(done_ids) < len(plan.chunks.chunks):
                self._run_pipeline(plan, tools, processor, store, done_ids)
            else:
                log.info("all chunks already complete; skipping to mux")
        finally:
            resources = self._sampler.stop()
            processor_stats = processor.stats()
            processor.close()
        self._timed("pipeline_s", t)

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
        frame_s = float(1 / video.fps) if video.fps > 0 else 0.04
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
        self, plan: JobPlan, tools: FFmpegTools, processor: BatchProcessor, store: ChunkStore, done_ids: set[int]
    ) -> None:
        settings = self.settings
        cancel = self.cancel
        sizes = self._queue_sizes(plan, processor)

        def skip(index: int) -> bool:
            return plan.chunks.chunk_of(index).id in done_ids

        first_index = 0
        while first_index < plan.total_frames and skip(first_index):
            first_index = plan.chunks.chunk_of(first_index).end

        decoder = FrameDecoder(
            tools, plan.spec.input, plan.media.require_video().index, plan.decode,
            loglevel=settings.ffmpeg.loglevel,
        ).start()

        def make_encoder(path: Path) -> SegmentEncoder:
            return SegmentEncoder(
                tools, path, plan.out_fmt, decoder.time_base, decoder.frame_rate, plan.encoder,
                loglevel=settings.ffmpeg.loglevel,
            )

        def skip_decoded(index: int) -> bool:
            # Resume: frames of verified chunks are decoded but not processed again.
            return bool(done_ids) and skip(index)

        encode_stage_args: dict[str, Any] = {
            "chunks": plan.chunks, "store": store, "make_encoder": make_encoder,
            "first_index": first_index, "done_chunks": done_ids,
        }
        stages: list[Any]
        if plan.mode == "device":
            # decode -> [GPU: yuv->rgb, model, resize, rgb->yuv] -> encode
            q_in: StageQueue[Any] = StageQueue("inference", sizes["inference"], cancel)
            q_done: StageQueue[Any] = StageQueue("encode", sizes["encode"], cancel)
            self._queues = {q.name: q for q in (q_in, q_done)}
            stages = [
                DecodeStage(cancel, decoder.frames(), q_in, skip=skip_decoded),
                ProcessStage(cancel, q_in, q_done, processor=processor),
                EncodeStage(cancel, q_done, **encode_stage_args),
            ]
        else:
            ops: FrameOps = get_ops(settings.pipeline.ops)
            q_raw: StageQueue[Any] = StageQueue("preprocess", sizes["preprocess"], cancel)
            q_rgb: StageQueue[Any] = StageQueue("inference", sizes["inference"], cancel)
            q_up: StageQueue[Any] = StageQueue("postprocess", sizes["postprocess"], cancel)
            q_out: StageQueue[Any] = StageQueue("encode", sizes["encode"], cancel)
            self._queues = {q.name: q for q in (q_raw, q_rgb, q_up, q_out)}
            stages = [
                DecodeStage(cancel, decoder.frames(), q_raw, skip=skip_decoded),
                PreprocessStage(cancel, q_raw, q_rgb, ops=ops, fmt=plan.decode.fmt, color=plan.encoder.color),
                ProcessStage(cancel, q_rgb, q_up, processor=processor),
                PostprocessStage(
                    cancel, q_up, q_out, ops=ops, out_fmt=plan.out_fmt, color=plan.encoder.color,
                    kernel=plan.resize_kernel,
                ),
                EncodeStage(cancel, q_out, **encode_stage_args),
            ]
        self._stages = stages
        self._encode_stage = stages[-1]
        self._pipeline_t0 = time.perf_counter()
        self._emit("running")
        try:
            StageRunner(stages, cancel).run(on_tick=lambda: self._emit("running"))
        finally:
            decoder.close()

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
                "pipeline_fps": round(processed / pipeline_s, 2) if pipeline_s > 0 else None,
                "stages": stages,
                "device_substages": substages,
                "bottleneck": min(stages, key=lambda s: s["busy_fps"] or float("inf"))["name"] if stages else None,
            },
            "processor": processor_stats,
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

