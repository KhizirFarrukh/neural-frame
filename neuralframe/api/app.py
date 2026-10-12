"""FastAPI application: REST API (``/v1``), Server-Sent Events for progress and the single-page web UI."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import subprocess
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fractions import Fraction
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from neuralframe import __version__
from neuralframe.api.manager import JobManager
from neuralframe.api.paths import PathNotAllowed, check_input, check_output, safe_filename, within
from neuralframe.api.store import TERMINAL_STATES, JobStore
from neuralframe.config import Settings, builtin_presets
from neuralframe.core.job import JobSpec, media_summary
from neuralframe.core.planning import TargetSpec
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, locate_ffmpeg, popen_kwargs
from neuralframe.io.probe import probe
from neuralframe.log import get_logger

log = get_logger(component="api")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class JobCreate(BaseModel):
    """``options`` takes any :class:`JobSpec` field (model, preset, interpolate, target_fps, codec, ...)."""

    model_config = ConfigDict(extra="forbid")

    input: str
    output: str | None = None
    target: str | None = Field(default=None, description="4k (default), 1440p, 1080p, x2, 1920x1080, native")
    priority: int = 0
    options: dict[str, Any] = Field(default_factory=dict)


class ProbeRequest(BaseModel):
    path: str


def _error(status: int, code: str, message: str, hint: str | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "hint": hint}}, status_code=status)


def default_output(input_path: Path, spec: JobSpec, roots: list[Path], fallback_dir: Path) -> Path:
    """``<stem>_<what was done><ext>`` next to the input when allowed, else in the outputs folder."""
    image = input_path.suffix.lower() in IMAGE_SUFFIXES
    tags = []
    if spec.upscale:
        tags.append("upscaled")
    if spec.interpolate and not image:
        tags.append(f"{(spec.target_fps or 'fps').replace('/', '-')}fps" if spec.target_fps else "framegen")
    suffix = input_path.suffix.lower() if image else ".mkv"
    folder = input_path.parent if within(input_path.parent, roots) else fallback_dir
    candidate = folder / f"{input_path.stem}_{'_'.join(tags) or 'out'}{suffix}"
    n = 2
    while candidate.exists():
        candidate = folder / f"{input_path.stem}_{'_'.join(tags) or 'out'}_{n}{suffix}"
        n += 1
    return candidate


# Our colour enums -> names of FFmpeg's colorspace filter (BT.601 625/525 share one matrix).
_CS_MATRIX = {"bt601": "smpte170m", "bt709": "bt709", "bt2020nc": "bt2020ncl", "smpte240m": "smpte240m", "fcc": "fcc"}


def colour_match_filter(report: dict[str, Any] | None) -> str | None:
    """FFmpeg filter giving a source frame the job's output colorimetry, so "before" and "after" compare
    like for like (the job may convert matrix and primaries, e.g. SD SMPTE-C to BT.709)."""
    color = (report or {}).get("plan", {}).get("color")
    if not color or color.get("hdr_passthrough"):
        return None
    src, dst = color["input"], color["output"]
    ispace, space = _CS_MATRIX.get(src["matrix"]), _CS_MATRIX.get(dst["colorspace"])
    if ispace is None or space is None:
        return None
    if (ispace, src["primaries"], src["range"]) == (space, dst["color_primaries"], dst["color_range"]):
        return None
    return (f"colorspace=ispace={ispace}:iprimaries={src['primaries']}:itrc={src['transfer']}:"
            f"irange={src['range']}:space={space}:primaries={dst['color_primaries']}:trc={dst['color_trc']}:"
            f"range={dst['color_range']}:fast=0")


def _frame_png(tools: FFmpegTools, path: Path, seconds: float, size: tuple[int, int] | None,
               colour: str | None = None) -> bytes:
    args = [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{max(seconds, 0.0):.6f}",
            "-i", str(path), "-frames:v", "1", "-an", "-sn"]
    filters = [f for f in (colour, f"scale={size[0]}:{size[1]}:flags=bicubic" if size else None) if f]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-f", "image2pipe", "-c:v", "png", "pipe:1"]
    proc = subprocess.run(args, capture_output=True, timeout=120, check=False, **popen_kwargs())
    if proc.returncode != 0 or not proc.stdout:
        raise NeuralFrameError(ErrorCode.DECODE_FAILED, f"cannot extract a preview frame from {path.name}: "
                               + proc.stderr.decode(errors="replace").strip()[-300:])
    return bytes(proc.stdout)


def create_app(settings: Settings, *, manager: JobManager | None = None, start_workers: bool = True) -> FastAPI:
    store = manager.store if manager is not None else JobStore(settings.paths.database())
    jobs = manager or JobManager(settings, store)
    upload_dir = settings.api.upload_dir or settings.paths.cache_dir / "uploads"
    output_dir = settings.api.output_dir or settings.paths.cache_dir / "outputs"
    input_roots = [*settings.api.input_roots, upload_dir, output_dir]
    output_roots = [*settings.api.output_roots, output_dir]

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        if start_workers:
            jobs.start()
        yield
        await asyncio.to_thread(jobs.stop)

    app = FastAPI(title="NeuralFrame", version=__version__, lifespan=lifespan,
                  description="Local AI video/image upscaling and frame generation service.")
    app.state.manager = jobs

    @app.exception_handler(PathNotAllowed)
    async def _forbidden(_request: Request, exc: PathNotAllowed) -> JSONResponse:
        return _error(403, "PATH_NOT_ALLOWED", exc.message, exc.hint)

    @app.exception_handler(NeuralFrameError)
    async def _nf_error(_request: Request, exc: NeuralFrameError) -> JSONResponse:
        status = 400 if exc.code in (ErrorCode.INVALID_INPUT, ErrorCode.CONFIG_INVALID) else 500
        return _error(status, exc.code.value, exc.message, exc.hint)

    @app.exception_handler(KeyError)
    async def _not_found(_request: Request, exc: KeyError) -> JSONResponse:
        return _error(404, "NOT_FOUND", f"no job {exc.args[0] if exc.args else ''}")

    @app.exception_handler(ValidationError)
    async def _invalid(_request: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse({"error": {"code": "INVALID_OPTIONS", "message": str(exc),
                                       "details": json.loads(exc.json())}}, status_code=422)

    def tools() -> FFmpegTools:
        return locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)

    def job_or_404(job_id: str) -> dict[str, Any]:
        record = jobs.job(job_id)
        if record is None:
            raise KeyError(job_id)
        return record

    # ---- UI --------------------------------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        html = resources.files("neuralframe.api").joinpath("static/index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    # ---- system / models -------------------------------------------------------------------------------

    @app.get("/v1/system")
    def system(refresh: bool = False) -> dict[str, Any]:
        from neuralframe.io.hwcaps import detect_capabilities

        caps = detect_capabilities(tools(), cache_dir=settings.paths.cache_dir, refresh=refresh)
        presets = sorted({*builtin_presets(), *settings.presets})
        return {
            **caps.model_dump(mode="json"),
            "selected": caps.selected(),
            "service": {"version": __version__, "workers": jobs.workers, "running": jobs.running,
                        "input_roots": [str(p) for p in input_roots],
                        "output_roots": [str(p) for p in output_roots]},
            "defaults": {"target": "4k", "fps": settings.framegen.fps, "presets": presets,
                         "repair_drops": settings.framegen.repair_drops},
        }

    @app.get("/v1/models")
    def models() -> list[dict[str, Any]]:
        from neuralframe.models.registry import ModelRegistry

        engines_dir = settings.paths.engines()
        engines = sorted(engines_dir.glob("*.engine")) if engines_dir.is_dir() else []
        result = []
        for m in ModelRegistry.load().all():
            present = m.weights is None or (settings.paths.models() / m.id / m.weights.filename).is_file()
            result.append({
                **m.model_dump(mode="json"),
                "weights_present": present,
                "engines": [e.name for e in engines if e.name.startswith(f"{m.id}-")],
            })
        return result

    # ---- files -----------------------------------------------------------------------------------------

    @app.post("/v1/uploads", status_code=201)
    async def upload(file: UploadFile = File(...)) -> dict[str, Any]:
        folder = upload_dir / uuid.uuid4().hex[:12]
        folder.mkdir(parents=True)
        dest = folder / safe_filename(file.filename or "upload")
        limit = int(settings.api.max_upload_gb * 2**30)
        size = 0
        try:
            with dest.open("wb") as fh:
                while chunk := await file.read(4 * 2**20):
                    size += len(chunk)
                    if size > limit:
                        raise NeuralFrameError(ErrorCode.INVALID_INPUT,
                                               f"upload exceeds api.max_upload_gb ({settings.api.max_upload_gb} GB)")
                    fh.write(chunk)
        except BaseException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        return {"path": str(dest), "size": size}

    @app.post("/v1/probe")
    def probe_file(body: ProbeRequest) -> dict[str, Any]:
        path = check_input(body.path, input_roots)
        media = probe(path, tools())
        return {"path": str(path), "is_image": media.is_image, **media_summary(media)}

    # ---- jobs ------------------------------------------------------------------------------------------

    @app.post("/v1/jobs", status_code=201)
    def create_job(body: JobCreate) -> dict[str, Any]:
        source = check_input(body.input, input_roots)
        reserved = {"input", "output", "target"} & set(body.options)
        if reserved:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"set {sorted(reserved)} at the top level, not in options")
        target = TargetSpec.parse(body.target) if body.target else TargetSpec.default()
        draft = JobSpec(input=source, output=source.with_suffix(".out"), target=target, **body.options)
        if body.output:
            output = check_output(body.output, output_roots)
        else:
            output = check_output(default_output(source, draft, output_roots, output_dir), output_roots)
        if output.exists() and not draft.overwrite:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"output {output} exists",
                                   hint='set "overwrite": true in options')
        if output == source:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, "output must differ from the input")
        spec = draft.model_copy(update={"output": output})
        return jobs.submit(spec, priority=body.priority)

    @app.get("/v1/jobs")
    def list_jobs(state: list[str] | None = Query(default=None), limit: int = 200) -> list[dict[str, Any]]:
        records = store.list_jobs(states=tuple(state) if state else None, limit=limit)
        return [jobs.job(r["id"]) or r for r in records]

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        return job_or_404(job_id)

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        jobs.delete(job_id)
        return Response(status_code=204)

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        return jobs.cancel(job_id)

    @app.post("/v1/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> dict[str, Any]:
        return jobs.resume(job_id)

    @app.get("/v1/jobs/{job_id}/events")
    async def job_events(job_id: str, request: Request) -> StreamingResponse:
        job_or_404(job_id)

        async def stream() -> AsyncIterator[str]:
            version = -1
            beat = time.monotonic()
            while True:
                if await request.is_disconnected():
                    return
                current = jobs.version(job_id)
                if current != version:
                    version = current
                    record = jobs.job(job_id)
                    if record is None:
                        yield "event: deleted\ndata: {}\n\n"
                        return
                    yield f"event: progress\ndata: {json.dumps(record, default=str)}\n\n"
                    if record["state"] in (*TERMINAL_STATES, "paused") and job_id not in jobs.running:
                        yield f"event: end\ndata: {json.dumps({'state': record['state']})}\n\n"
                        return
                    beat = time.monotonic()
                elif time.monotonic() - beat > 15:
                    yield ": keep-alive\n\n"
                    beat = time.monotonic()
                await asyncio.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/jobs/{job_id}/report")
    def job_report(job_id: str) -> dict[str, Any]:
        record = job_or_404(job_id)
        path = record.get("report_path")
        if not path or not Path(path).is_file():
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, "the report is written when the job finishes")
        data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return data

    @app.get("/v1/jobs/{job_id}/output")
    def job_output(job_id: str) -> FileResponse:
        record = job_or_404(job_id)
        path = Path(record["output"])
        if record["state"] != "done" or not path.is_file():
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"job {job_id} has no finished output")
        return FileResponse(path, filename=path.name)

    @app.get("/v1/jobs/{job_id}/preview")
    def job_preview(
        job_id: str, frame: int = Query(default=0, ge=0),
        side: Literal["before", "after", "both"] = "both",
    ) -> Response:
        """Frame ``frame`` (source numbering) before and/or after processing, as PNG at the output size.

        "before" is the source frame in the output's colorimetry, scaled with bicubic interpolation (the usual
        baseline)."""
        record = job_or_404(job_id)
        ff = tools()
        source = Path(record["input"])
        media = probe(source, ff)
        video = media.require_video()
        fps = video.fps if video.fps and video.fps > 0 else Fraction(25)
        seconds = 0.0 if media.is_image else float((frame - Fraction(1, 2)) / fps)
        output = Path(record["output"])
        done = record["state"] == "done" and output.is_file()
        if side in ("after", "both") and not done:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, "the processed frame is available when the job is done")
        size = None
        colour = None
        if done:
            out_video = probe(output, ff).require_video()
            size = (out_video.width, out_video.height)
            report_path = Path(record.get("report_path") or "")
            if report_path.is_file():
                colour = colour_match_filter(json.loads(report_path.read_text(encoding="utf-8")))
        before = _frame_png(ff, source, seconds, size, colour) if side in ("before", "both") else b""
        after = _frame_png(ff, output, seconds, None) if side in ("after", "both") else b""
        if side != "both":
            return Response(before or after, media_type="image/png")
        from PIL import Image

        left, right = Image.open(io.BytesIO(before)), Image.open(io.BytesIO(after))
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
        canvas.paste(left.convert("RGB"), (0, 0))
        canvas.paste(right.convert("RGB"), (left.width, 0))
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return Response(buf.getvalue(), media_type="image/png")

    return app
