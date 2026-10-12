"""``neuralframe`` command-line interface (Typer)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

from neuralframe import __version__
from neuralframe.config import Settings, load_settings
from neuralframe.core.job import JobPlan, JobRunner, JobSpec, ProgressEvent
from neuralframe.core.planning import TargetSpec
from neuralframe.errors import JobCancelled, NeuralFrameError
from neuralframe.log import configure_logging

app = typer.Typer(
    name="neuralframe",
    help="NeuralFrame — high-performance local AI video upscaler.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
models_app = typer.Typer(help="List, download, verify and export models.", no_args_is_help=True)
app.add_typer(models_app, name="models")
queue_app = typer.Typer(help="Submit and manage jobs on a running `neuralframe serve`.", no_args_is_help=True)
app.add_typer(queue_app, name="queue")

console = Console(stderr=True)
out = Console()


@dataclass
class AppState:
    settings: Settings
    verbose: bool


def _state(ctx: typer.Context) -> AppState:
    state = ctx.obj
    assert isinstance(state, AppState)
    return state


def _fail(exc: NeuralFrameError) -> None:
    console.print(f"[bold red]error[/] [{exc.code}] {exc.message}")
    if exc.hint:
        console.print(f"[dim]hint: {exc.hint}[/]")
    raise typer.Exit(130 if isinstance(exc, JobCancelled) else 2)


@app.callback()
def _main(
    ctx: typer.Context,
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show info-level logs."),
    log_level: str | None = typer.Option(None, "--log-level", help="Override the log level."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON log lines."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Also write JSON logs to this file."),
) -> None:
    try:
        settings = load_settings(config)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    level = log_level or ("INFO" if verbose else ("WARNING" if not json_logs else settings.logging.level))
    configure_logging(
        level,
        json_output=json_logs or settings.logging.json_output,
        log_file=log_file or settings.logging.file,
    )
    ctx.obj = AppState(settings=settings, verbose=verbose)


@app.command()
def version() -> None:
    """Print the NeuralFrame version."""
    out.print(__version__)


# -------------------------------------------------------------------------------------------------------
# doctor
# -------------------------------------------------------------------------------------------------------


@app.command()
def doctor(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON."),
    refresh: bool = typer.Option(False, "--refresh", help="Re-run codec probes instead of using the cache."),
) -> None:
    """Report GPU, driver, CUDA/TensorRT/ONNX Runtime and hardware codec availability."""
    from neuralframe.io.ffmpeg import locate_ffmpeg
    from neuralframe.io.hwcaps import detect_capabilities

    settings = _state(ctx).settings
    try:
        tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
        with console.status("probing hardware codecs..."):
            caps = detect_capabilities(tools, cache_dir=settings.paths.cache_dir, refresh=refresh)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    choices = caps.selected()
    if as_json:
        out.print_json(json.dumps({**caps.model_dump(mode="json"), "selected": choices}))
        return

    sys_table = Table(title="System", show_header=False, title_justify="left")
    rows: list[tuple[str, Any]] = [
        ("NeuralFrame", caps.neuralframe),
        ("Platform", caps.platform),
        ("Python", caps.python),
        ("CPU", f"{caps.cpu} ({caps.cpu_threads} threads)"),
        ("RAM", f"{caps.ram_total_gb} GB total, {caps.ram_available_gb} GB available"),
        ("FFmpeg", f"{caps.ffmpeg_version} ({caps.ffmpeg_path})"),
        ("NVIDIA driver", caps.nvidia_driver or "not detected"),
        ("CUDA (driver max)", caps.cuda_driver or "-"),
        ("PyTorch", f"{caps.torch or 'not installed'}"
         + (f" (CUDA {caps.torch_cuda}, available={caps.torch_cuda_available})" if caps.torch else "")),
        ("cuDNN", caps.cudnn or "-"),
        ("TensorRT", caps.tensorrt or "not installed"),
        ("ONNX Runtime", f"{caps.onnxruntime or 'not installed'}"
         + (f" {caps.onnxruntime_providers}" if caps.onnxruntime else "")),
        ("Native module", caps.native_module or "not built (NumPy fallback)"),
    ]
    for key, value in rows:
        sys_table.add_row(key, str(value))
    console.print(sys_table)

    gpu_table = Table(title="GPUs", title_justify="left")
    for col in ("#", "Name", "VRAM total", "VRAM free", "Compute"):
        gpu_table.add_column(col)
    for g in caps.gpus:
        gpu_table.add_row(
            str(g.index), g.name, f"{g.total_vram_mb} MB", f"{g.free_vram_mb} MB", g.compute_capability or "-"
        )
    if not caps.gpus:
        gpu_table.add_row("-", "no NVIDIA GPU detected (CPU / ONNX Runtime fallback)", "", "", "")
    console.print(gpu_table)

    codec_table = Table(title="Hardware decode / encode", title_justify="left")
    for col in ("Kind", "Name", "Works", "10-bit", "Note"):
        codec_table.add_column(col)
    for h in caps.hwaccels:
        codec_table.add_row("decode", h.name, _yes(h.functional), "", (h.error or "")[:60])
    for e in caps.encoders:
        codec_table.add_row("encode", e.name, _yes(e.functional), _yes(e.ten_bit) if e.functional else "",
                            (e.error or "")[:60])
    console.print(codec_table)

    pick = Table(title="Will use", show_header=False, title_justify="left")
    for key, value in choices.items():
        pick.add_row(key, str(value))
    console.print(pick)


def _yes(flag: bool) -> str:
    return "[green]yes[/]" if flag else "[red]no[/]"


# -------------------------------------------------------------------------------------------------------
# probe
# -------------------------------------------------------------------------------------------------------


@app.command("probe")
def probe_cmd(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Media file."),
    as_json: bool = typer.Option(False, "--json"),
    frames: bool = typer.Option(False, "--frames", help="Also scan packets (exact frame count, VFR)."),
) -> None:
    """Show what NeuralFrame sees in a media file."""
    from neuralframe.core.job import media_summary
    from neuralframe.io.ffmpeg import locate_ffmpeg
    from neuralframe.io.probe import probe, scan_frames

    settings = _state(ctx).settings
    try:
        tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
        media = probe(input_path, tools)
        timeline = scan_frames(input_path, tools, media.video.index) if frames and media.video else None
    except NeuralFrameError as exc:
        _fail(exc)
        return
    data = media.model_dump(mode="json") if as_json else media_summary(media, timeline)
    if timeline is not None and as_json:
        data["timeline"] = {"frames": timeline.frame_count, "vfr": timeline.is_vfr}
    out.print_json(json.dumps(data, default=str))


# -------------------------------------------------------------------------------------------------------
# upscale
# -------------------------------------------------------------------------------------------------------


def _target(
    target: str | None, scale: float | None, target_height: int | None, size: str | None
) -> TargetSpec:
    chosen = [x for x in (target, scale, target_height, size) if x is not None]
    if len(chosen) > 1:
        raise typer.BadParameter("use only one of --target, --scale, --target-height, --size")
    if scale is not None:
        return TargetSpec.factor(scale)
    if target_height is not None:
        return TargetSpec.for_height(target_height)
    if size is not None:
        return TargetSpec.parse(size)
    if target is not None:
        return TargetSpec.parse(target)
    return TargetSpec.default()


def _print_plan(plan: JobPlan) -> None:
    g, e = plan.geometry, plan.encoder
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("model", f"{plan.model.id} (x{plan.model.scale}, passes {g.model_passes})")
    table.add_row("size", f"{g.decoded_width}x{g.decoded_height} -> {g.out_width}x{g.out_height}")
    table.add_row("decode", plan.decode.describe())
    table.add_row("encode", e.describe())
    tags = e.color.ffmpeg_tags()
    table.add_row("colour", f"{tags['colorspace']}/{tags['color_primaries']}/{tags['color_trc']} {tags['color_range']}")
    where = f"device ({plan.inference.device})" if plan.mode == "device" and plan.inference else plan.mode
    if plan.mode == "host":
        where = f"host (ops={plan.ops})"
    table.add_row("frames", f"{plan.total_frames} in {len(plan.chunks.chunks)} chunk(s), pipeline {where}")
    if plan.framegen:
        fg = plan.framegen
        drops = fg.get("drops", {})
        table.add_row(
            "framegen",
            f"{fg['engine']} -> {fg['fps']} fps ({fg['order'].replace('_', ' ')}); "
            f"{drops.get('duplicated_frames', 0)} frozen frame(s), {drops.get('frames_missing_in_gaps', 0)} "
            "missing in gaps to repair",
        )
    console.print(table)
    for note in plan.notes:
        console.print(f"  [dim]- {note}[/]")


@app.command()
def upscale(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input video or image."),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output file: video (.mkv .mp4 .mov .webm) or image (.png .jpg .webp .tif .bmp)."
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model id (see `neuralframe models list`)."),
    target: str | None = typer.Option(
        None, "--target", "-t", help="4k (default), 1440p, 1080p, 8k, native, 1920x1080, x2"
    ),
    scale: float | None = typer.Option(None, "--scale", help="Scale factor relative to the display size."),
    target_height: int | None = typer.Option(None, "--target-height", help="Exact output height."),
    size: str | None = typer.Option(None, "--size", help="Exact output size WIDTHxHEIGHT."),
    preset: str = typer.Option("balanced", "--preset", "-p", help="quality | balanced | fast"),
    backend: str | None = typer.Option(None, "--backend", help="auto | tensorrt | torch | onnxruntime"),
    encoder: str | None = typer.Option(None, "--encoder", help="auto or an FFmpeg encoder name."),
    codec: str | None = typer.Option(None, "--codec", help="hevc | h264 | av1"),
    quality: int | None = typer.Option(None, "--crf", "--cq", help="Constant-quality value."),
    ten_bit: bool = typer.Option(False, "--10bit", help="Encode 10-bit output (less banding)."),
    deinterlace: str | None = typer.Option(None, "--deinterlace", help="auto | on | off"),
    denoise: str | None = typer.Option(None, "--denoise", help="off | light | medium | strong"),
    max_vram_gb: float | None = typer.Option(None, "--max-vram-gb", help="Cap VRAM used by inference."),
    gpus: str | None = typer.Option(None, "--gpus", help="Comma-separated GPU indices, e.g. 0,1"),
    tile_size: int | None = typer.Option(None, "--tile-size"),
    tile_overlap: int | None = typer.Option(None, "--tile-overlap"),
    batch_size: int | None = typer.Option(None, "--batch-size"),
    chunk_frames: int | None = typer.Option(None, "--chunk-frames"),
    resume: bool = typer.Option(False, "--resume", help="Continue an interrupted job."),
    overwrite: bool = typer.Option(False, "--overwrite", "-y", help="Replace an existing output."),
    keep_work: bool = typer.Option(False, "--keep-work", help="Keep intermediate segments."),
    interpolate: bool = typer.Option(
        False, "--interpolate", "-i", help="Frame generation: constant frame rate (default 60) + drop repair."
    ),
    fps: str | None = typer.Option(None, "--fps", help="Frame generation target, e.g. 60, 59.94, 120 (implies -i)."),
    no_upscale: bool = typer.Option(False, "--no-upscale", help="Keep the source size (frame generation only)."),
    repair_drops: str | None = typer.Option(None, "--repair-drops", help="auto | on | off"),
    framegen_engine: str | None = typer.Option(None, "--framegen-engine", help="auto | rife | minterpolate"),
    framegen_order: str | None = typer.Option(
        None, "--framegen-order", help="auto (default) | upscale_first | interpolate_first"
    ),
) -> None:
    """Upscale a video or image (default target: 4K, aspect ratio preserved); optionally generate frames."""
    state = _state(ctx)
    if no_upscale and not (interpolate or fps):
        raise typer.BadParameter("--no-upscale only makes sense with frame generation (--interpolate / --fps)")
    try:
        spec = JobSpec(
            input=input_path,
            output=output,
            model=model,
            target=_target(target, scale, target_height, size),
            preset=preset,
            backend=backend,
            encoder=encoder,
            codec=codec,
            quality=quality,
            ten_bit=True if ten_bit else None,
            deinterlace=deinterlace,
            denoise=denoise,
            max_vram_gb=max_vram_gb,
            gpus=[int(g) for g in gpus.split(",")] if gpus else None,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            chunk_frames=chunk_frames,
            resume=resume,
            overwrite=overwrite,
            keep_work_dir=True if keep_work else None,
            upscale=not no_upscale,
            interpolate=interpolate or fps is not None,
            target_fps=fps,
            repair_drops=repair_drops,
            framegen_engine=framegen_engine,
            framegen_order=framegen_order,
        )
    except NeuralFrameError as exc:
        _fail(exc)
        return
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _run_job(state, spec)


@app.command("interpolate")
def interpolate_cmd(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input video."),
    output: Path = typer.Option(..., "--output", "-o", help="Output video (.mkv .mp4 .mov .webm)."),
    fps: str | None = typer.Option(None, "--fps", help="Target frame rate (default 60; 59.94, 120, 24000/1001...)."),
    repair_drops: str | None = typer.Option(None, "--repair-drops", help="auto | on | off"),
    engine: str | None = typer.Option(None, "--engine", help="auto | rife | minterpolate"),
    encoder: str | None = typer.Option(None, "--encoder", help="auto or an FFmpeg encoder name."),
    codec: str | None = typer.Option(None, "--codec", help="hevc | h264 | av1"),
    quality: int | None = typer.Option(None, "--crf", "--cq", help="Constant-quality value."),
    chunk_frames: int | None = typer.Option(None, "--chunk-frames"),
    resume: bool = typer.Option(False, "--resume", help="Continue an interrupted job."),
    overwrite: bool = typer.Option(False, "--overwrite", "-y", help="Replace an existing output."),
) -> None:
    """Frame generation only: constant frame rate output with dropped/frozen frames re-synthesised."""
    state = _state(ctx)
    try:
        spec = JobSpec(
            input=input_path, output=output, upscale=False, interpolate=True, target_fps=fps,
            repair_drops=repair_drops, framegen_engine=engine, encoder=encoder, codec=codec, quality=quality,
            chunk_frames=chunk_frames, resume=resume, overwrite=overwrite,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _run_job(state, spec)


def _run_job(state: AppState, spec: JobSpec) -> None:
    progress = Progress(
        TextColumn("[bold]{task.fields[state]:<9}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f} frames"),
        TextColumn("[cyan]{task.fields[fps]:.1f} fps"),
        TimeRemainingColumn(),
        TextColumn("[dim]{task.fields[stages]}"),
        console=console,
        transient=False,
    )
    task = progress.add_task("job", total=1, state="starting", fps=0.0, stages="")

    def on_progress(event: ProgressEvent) -> None:
        stages = " ".join(f"{s.name[:3]}:{s.busy_fps:.0f}" for s in event.stages)
        progress.update(
            task,
            total=max(event.frames_total, 1),
            completed=event.frames_done,
            state=event.state,
            fps=event.fps,
            stages=stages,
        )

    runner = JobRunner(state.settings, progress=on_progress, on_plan=_print_plan)
    try:
        with progress:
            result = runner.run(spec)
    except JobCancelled as exc:
        console.print("[yellow]cancelled[/]: finished chunks were kept; re-run with --resume to continue")
        _fail(exc)
        return
    except NeuralFrameError as exc:
        _fail(exc)
        return

    report = result.report
    if report.get("kind") == "image":
        console.print(
            f"[bold green]done[/] {result.output} ({report['output']['size']}) "
            f"in {report['timings']['total_s']:.1f}s"
        )
    else:
        tp = report["throughput"]
        console.print(
            f"[bold green]done[/] {result.output} ({report['output']['size']}, {report['output']['frames']} "
            f"frames) in {report['timings']['total_s']:.1f}s - pipeline {tp['pipeline_fps']} fps, "
            f"bottleneck: {tp['bottleneck']}"
        )
        if report.get("framegen"):
            fg = report["framegen"]
            runtime = fg.get("runtime") or {}
            console.print(
                f"  frame generation: {fg['fps']} fps, {runtime.get('interpolated', 0)} synthesised, "
                f"{fg['drops']['duplicated_frames']} frozen frame(s) repaired"
            )
    console.print(f"[dim]report: {result.report_path}[/]")


# -------------------------------------------------------------------------------------------------------
# models
# -------------------------------------------------------------------------------------------------------


@models_app.command("list")
def models_list(as_json: bool = typer.Option(False, "--json")) -> None:
    """List registered models."""
    from neuralframe.models.registry import ModelRegistry

    registry = ModelRegistry.load()
    if as_json:
        out.print_json(json.dumps([m.model_dump(mode="json") for m in registry.all()]))
        return
    table = Table(title="Models", title_justify="left")
    for col in ("id", "kind", "scale", "input", "licence", "recommended for"):
        table.add_column(col, no_wrap=col == "id")
    for m in registry.all():
        table.add_row(m.id, m.kind, f"x{m.scale}", m.input, m.licence, ", ".join(m.recommended_content))
    console.print(table)


@models_app.command("download")
def models_download(
    ctx: typer.Context,
    model_ids: list[str] = typer.Argument(None, help="Model ids (default: all neural models)."),
) -> None:
    """Download and SHA256-verify model weights into the cache."""
    from rich.progress import DownloadColumn, TransferSpeedColumn

    from neuralframe.models.download import ensure_weights
    from neuralframe.models.registry import ModelRegistry

    settings = _state(ctx).settings
    registry = ModelRegistry.load()
    specs = [registry.get(m) for m in model_ids] if model_ids else [m for m in registry.all() if m.weights]
    for spec in specs:
        if spec.weights is None:
            console.print(f"{spec.id}: classical model, nothing to download")
            continue
        with Progress(TextColumn(spec.id), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                      console=console) as bar:
            task = bar.add_task("dl", total=spec.weights.size_bytes)

            def update(done: int, total: int | None, _task: Any = task, _bar: Progress = bar) -> None:
                _bar.update(_task, completed=done, total=total)

            try:
                path = ensure_weights(spec, settings.paths.models(), progress=update)
            except NeuralFrameError as exc:
                _fail(exc)
                return
        console.print(f"[green]ok[/] {spec.id} -> {path} ({spec.licence})")


@models_app.command("verify")
def models_verify(ctx: typer.Context) -> None:
    """Check SHA256 of every downloaded weight file."""
    from neuralframe.models.download import verify_weights, weights_path
    from neuralframe.models.registry import ModelRegistry

    settings = _state(ctx).settings
    bad = 0
    for spec in ModelRegistry.load().all():
        if spec.weights is None:
            continue
        path = weights_path(spec, settings.paths.models())
        if not path.is_file():
            console.print(f"[dim]{spec.id}: not downloaded[/]")
        elif verify_weights(spec, settings.paths.models()):
            console.print(f"[green]{spec.id}: ok[/]")
        else:
            bad += 1
            console.print(f"[red]{spec.id}: SHA256 mismatch[/] ({path})")
    if bad:
        raise typer.Exit(1)


@models_app.command("export")
def models_export(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., help="Model id to export."),
    check: bool = typer.Option(True, "--check/--no-check", help="Numeric parity check after export."),
    tensorrt: bool = typer.Option(False, "--tensorrt", help="Also build a TensorRT engine and compare it."),
    size: int = typer.Option(128, "--size", help="Tile edge used for the parity check."),
) -> None:
    """Export a model to ONNX (opset 17, dynamic batch/H/W) and verify it against PyTorch."""
    import torch

    from neuralframe.inference.backend import ShapeProfile
    from neuralframe.inference.factory import create_backend
    from neuralframe.models.download import ensure_weights
    from neuralframe.models.export import compare_outputs, export_onnx, onnx_path, sample_tiles
    from neuralframe.models.registry import ModelRegistry

    settings = _state(ctx).settings
    try:
        spec = ModelRegistry.load().get(model_id)
        weights = ensure_weights(spec, settings.paths.models())
        path = export_onnx(spec, weights, onnx_path(spec, settings.paths.models()))
    except NeuralFrameError as exc:
        _fail(exc)
        return
    console.print(f"[green]exported[/] {path}")
    if not check:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiles = sample_tiles(4, size, size).to(device)
    profile = ShapeProfile.fixed(size, size, 4)
    candidates = ["onnxruntime"] + (["tensorrt"] if tensorrt and device.type == "cuda" else [])
    reference = create_backend("torch", device, "fp32", deterministic=True)
    reference.load(spec, weights, profile)
    ref_out = reference.infer(tiles).float()
    reference.close()
    table = Table(title=f"Parity vs PyTorch FP32 ({size}x{size} tiles, {device})", title_justify="left")
    for col in ("backend", "precision", "max |diff|", "mean |diff|", "PSNR (dB)"):
        table.add_column(col)
    for name in candidates:
        precision = "fp16" if name == "tensorrt" else "fp32"
        backend = create_backend(name, device, precision, deterministic=True, models_dir=settings.paths.models(),
                                 engines_dir=settings.paths.engines(), onnx_providers=settings.inference.onnx_providers)
        try:
            backend.load(spec, weights, profile)
            out = backend.infer(tiles).float()
        except NeuralFrameError as exc:
            table.add_row(name, precision, "-", "-", f"[red]{exc.message[:40]}[/]")
            continue
        finally:
            backend.close()
        mx, mean, psnr = compare_outputs(ref_out, out)
        table.add_row(name, precision, f"{mx:.5f}", f"{mean:.6f}", f"{psnr:.1f}")
    console.print(table)


# -------------------------------------------------------------------------------------------------------
# serve / queue
# -------------------------------------------------------------------------------------------------------


@app.command()
def serve(
    ctx: typer.Context,
    host: str | None = typer.Option(None, "--host", help="Bind address (default from config: 127.0.0.1)."),
    port: int | None = typer.Option(None, "--port", help="Port (default from config: 8765)."),
    workers: str | None = typer.Option(
        None, "--workers", help='Job workers: "cpu" or GPU indices like "0,1" (default: one per GPU).'
    ),
) -> None:
    """Run the local job service: REST API, progress events and the web UI."""
    import uvicorn

    from neuralframe.api.app import create_app
    from neuralframe.api.manager import JobManager
    from neuralframe.api.store import JobStore

    settings = _state(ctx).settings
    host = host or settings.api.host
    port = port or settings.api.port
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(f"[yellow]warning[/]: binding to {host} exposes the API to the network; it has no "
                      "authentication (file access is limited to api.input_roots / api.output_roots)")
    names = None
    if workers:
        names = ["cpu"] if workers.strip() == "cpu" else [f"cuda:{int(i)}" for i in workers.split(",")]
    manager = JobManager(settings, JobStore(settings.paths.database()), workers=names)
    console.print(f"NeuralFrame service on http://{host}:{port}  (workers: {', '.join(manager.workers)}; "
                  f"jobs db: {settings.paths.database()})")
    uvicorn.run(create_app(settings, manager=manager), host=host, port=port, log_level="warning")


def _client(server: str) -> Any:
    """HTTP client for the queue commands (replaced in tests)."""
    import httpx

    return httpx.Client(base_url=server, timeout=60)


def _server(ctx: typer.Context, server: str | None) -> str:
    settings = _state(ctx).settings
    return server or f"http://{settings.api.host}:{settings.api.port}"


def _call(server: str, method: str, path: str, body: Any = None) -> Any:
    import httpx

    try:
        with _client(server) as client:
            response = client.request(method, path, json=body)
    except httpx.TransportError as exc:
        console.print(f"[red]cannot reach {server}[/]: {exc}\n  hint: start the service with `neuralframe serve`")
        raise typer.Exit(2) from exc
    if response.status_code >= 400:
        error = (response.json() if response.content else {}).get("error", {})
        console.print(f"[red]error[/] [{error.get('code', response.status_code)}] {error.get('message', '')}")
        if error.get("hint"):
            console.print(f"  hint: {error['hint']}")
        raise typer.Exit(1)
    return response.json() if response.content else None


@queue_app.command("add")
def queue_add(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Input video or image."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output file (default: next to the input)."),
    target: str | None = typer.Option(None, "--target", "-t", help="4k (default), 1440p, 1080p, x2, 1920x1080"),
    model: str | None = typer.Option(None, "--model", "-m"),
    preset: str = typer.Option("balanced", "--preset", "-p"),
    interpolate: bool = typer.Option(False, "--interpolate", "-i", help="Frame generation (default 60 fps)."),
    fps: str | None = typer.Option(None, "--fps", help="Frame generation target (implies -i)."),
    no_upscale: bool = typer.Option(False, "--no-upscale", help="Frame generation only."),
    repair_drops: str | None = typer.Option(None, "--repair-drops", help="auto | on | off"),
    priority: int = typer.Option(0, "--priority", help="Higher runs first."),
    overwrite: bool = typer.Option(False, "--overwrite", "-y"),
    server: str | None = typer.Option(None, "--server", help="Service URL (default from config)."),
) -> None:
    """Queue a job on the service."""
    options: dict[str, Any] = {"preset": preset, "upscale": not no_upscale, "interpolate": interpolate or bool(fps),
                               "overwrite": overwrite}
    for key, value in (("model", model), ("target_fps", fps), ("repair_drops", repair_drops)):
        if value is not None:
            options[key] = value
    body = {"input": str(input_path.resolve()), "output": str(output.resolve()) if output else None,
            "target": target, "priority": priority, "options": options}
    job = _call(_server(ctx, server), "POST", "/v1/jobs", body)
    console.print(f"queued [bold]{job['id']}[/] -> {job['output']}")
    out.print(job["id"])


@queue_app.command("list")
def queue_list(
    ctx: typer.Context,
    as_json: bool = typer.Option(False, "--json"),
    server: str | None = typer.Option(None, "--server"),
) -> None:
    """Show queued, running and finished jobs."""
    jobs = _call(_server(ctx, server), "GET", "/v1/jobs")
    if as_json:
        out.print_json(json.dumps(jobs))
        return
    table = Table(title="Jobs", title_justify="left")
    for col in ("id", "state", "progress", "fps", "input", "output"):
        table.add_column(col, no_wrap=col in ("id", "state"))
    for j in jobs:
        table.add_row(j["id"], j["state"], f"{j['percent']:.1f}%", f"{j['fps']:.1f}" if j["fps"] else "",
                      Path(j["input"] or "").name, Path(j["output"] or "").name)
    console.print(table)


@queue_app.command("cancel")
def queue_cancel(ctx: typer.Context, job_id: str, server: str | None = typer.Option(None, "--server")) -> None:
    """Cancel a job (finished chunks are kept so it can be resumed)."""
    job = _call(_server(ctx, server), "POST", f"/v1/jobs/{job_id}/cancel")
    console.print(f"{job['id']}: {job['state']}")


@queue_app.command("resume")
def queue_resume(ctx: typer.Context, job_id: str, server: str | None = typer.Option(None, "--server")) -> None:
    """Queue a cancelled, failed or paused job again; it continues from its last finished chunk."""
    job = _call(_server(ctx, server), "POST", f"/v1/jobs/{job_id}/resume")
    console.print(f"{job['id']}: {job['state']}")


@queue_app.command("watch")
def queue_watch(ctx: typer.Context, job_id: str, server: str | None = typer.Option(None, "--server")) -> None:
    """Follow a job's progress until it ends."""
    import httpx

    url = _server(ctx, server)
    progress = Progress(TextColumn("[bold]{task.fields[state]:<9}"), BarColumn(), TaskProgressColumn(),
                        TextColumn("{task.fields[fps]:.1f} fps"), TimeRemainingColumn(), console=console)
    task = progress.add_task("job", total=100, state="", fps=0.0)
    final: dict[str, Any] = {}
    try:
        with _client(url) as client, progress, client.stream("GET", f"/v1/jobs/{job_id}/events",
                                                             timeout=None) as response:
            if response.status_code >= 400:
                response.read()
                _call(url, "GET", f"/v1/jobs/{job_id}")  # prints the error and exits
            event = ""
            for line in response.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:") and event == "progress":
                    final = json.loads(line[5:])
                    progress.update(task, completed=final.get("percent", 0), state=final.get("state", ""),
                                    fps=final.get("fps") or 0.0)
    except httpx.TransportError as exc:
        console.print(f"[red]cannot reach {url}[/]: {exc}")
        raise typer.Exit(2) from exc
    state = final.get("state", "unknown")
    console.print(f"{job_id}: {state} {final.get('output') or ''}")
    if state != "done":
        raise typer.Exit(1)


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        console.print("[yellow]interrupted[/]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
