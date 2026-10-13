"""``neuralframe`` command-line interface (Typer)."""

from __future__ import annotations

import json
import sys
import threading
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
from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.log import configure_logging
from neuralframe.models.library import ModelLibrary
from neuralframe.models.registry import SPEED_WORDS, ModelRegistry

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
    model: str | None = typer.Option(
        None, "--model", "-m", help="Model id (see `neuralframe models list`) or auto (default: analyse the input)."
    ),
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
def models_list(ctx: typer.Context, as_json: bool = typer.Option(False, "--json")) -> None:
    """List the models, grouped by what they are for."""
    registry = ModelRegistry.load()
    if as_json:
        out.print_json(json.dumps([{**m.model_dump(mode="json"), "label": m.label,
                                    "category_title": m.category_title} for m in registry.all()]))
        return
    table = Table(title="Models - use the name in brackets with -m (or -m auto: picked for each video)",
                  title_justify="left", show_lines=False)
    table.add_column("use it for")
    table.add_column("-m", no_wrap=True, style="cyan")
    table.add_column("scale", no_wrap=True)
    table.add_column("speed", no_wrap=True)
    table.add_column("licence")
    table.add_column("on this computer")
    library = ModelLibrary.from_settings(_state(ctx).settings)
    for title, models in registry.grouped():
        table.add_section()
        table.add_row(f"[bold]{title}[/]", "", "", "", "", "")
        for m in models:
            licence = m.licence + (" [yellow](personal use only)[/]" if m.licence_restricted else "")
            name = m.id + (" (added)" if m.origin == "user" else "")
            status = library.status(m)
            here = {"builtin": "built in", "missing": "-", "installed": "[green]yes[/]", "bundled": "[green]yes[/]",
                    "found": "[green]yes[/] (found)", "linked": "[green]your file[/]",
                    "linked-different": "[green]your file[/]"}[status.state]
            if status.state == "missing" and status.size_bytes:
                here = f"- ({status.size_bytes / 1e6:.0f} MB)"
            table.add_row(f"  {m.purpose or m.label}", name, f"x{m.scale}", SPEED_WORDS.get(m.speed or "", "-"),
                          licence, here)
    console.print(table)
    console.print("[dim]Models are downloaded when first used (or `neuralframe models download NAME`). More: "
                  "`neuralframe models info NAME`, `models link NAME FILE` (a file you have), `models check-updates`, "
                  "`models add FILE` (your own model).[/]")


@models_app.command("add")
def models_add(
    ctx: typer.Context,
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="Model file (.pth, .safetensors or .ckpt)."),
    model_id: str | None = typer.Option(None, "--id", help="Name to use with -m (default: from the file name)."),
    description: str | None = typer.Option(None, "--description"),
    licence: str | None = typer.Option(None, "--licence", help="Licence of the model, e.g. CC-BY-4.0."),
    content: list[str] = typer.Option([], "--content", help="What it is made for: live-action and/or animation "
                                      "(repeatable)."),
    handles: list[str] = typer.Option([], "--handles", help="What it removes: noise, compression, blur "
                                      "(repeatable)."),
    purpose: str | None = typer.Option(None, "--purpose", help="What it is for, in plain words (shown in model "
                                       "lists), e.g. \"Old cartoons from VHS\"."),
    rank: int = typer.Option(0, "--rank", min=0, max=100, help="Let automatic choice use it (with --content): "
                             "how good it is next to the built-in models, e.g. hat-l-x4 95, realesrgan-x4plus 65 "
                             "(see `models list --json`). 0 = only when chosen with -m."),
) -> None:
    """Add your own upscaling model file (any architecture spandrel supports, e.g. from OpenModelDB)."""
    from neuralframe.models.custom import add_user_model

    settings = _state(ctx).settings
    try:
        spec = add_user_model(file, settings.paths.models(), model_id=model_id, description=description,
                              licence=licence, content=content, handles=handles, rank=rank, purpose=purpose)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    console.print(f"[green]added[/] {spec.label}: {spec.description}")
    console.print(f"use it with: neuralframe upscale INPUT -o OUTPUT -m {spec.id}")
    out.print(spec.id)


@models_app.command("remove")
def models_remove(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., help="Id of a model added with `models add`."),
    keep_file: bool = typer.Option(False, "--keep-file", help="Keep the copied model file."),
) -> None:
    """Remove a model you added."""
    from neuralframe.models.custom import remove_user_model

    try:
        remove_user_model(model_id, _state(ctx).settings.paths.models(), delete_file=not keep_file)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    console.print(f"removed {model_id}")


def _download_with_bar(library: Any, spec: Any, *, update: bool = False) -> Path:
    """Download (or update) one model with a progress bar; Ctrl+C stops it and keeps the part done."""
    from rich.progress import DownloadColumn, TimeRemainingColumn, TransferSpeedColumn

    from neuralframe.models.library import DownloadProgress

    with Progress(TextColumn(f"{spec.id}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                  TimeRemainingColumn(), console=console) as bar:
        task = bar.add_task("download", total=spec.weights.size_bytes if spec.weights else None)

        def on_progress(event: DownloadProgress) -> None:
            bar.update(task, completed=event.done, total=event.total)

        cancel = threading.Event()
        try:
            if update:
                return Path(library.apply_update(spec, progress=on_progress, cancel=cancel))
            return Path(library.ensure(spec, progress=on_progress, cancel=cancel))
        except KeyboardInterrupt:
            cancel.set()
            raise


@models_app.command("download")
def models_download(
    ctx: typer.Context,
    model_ids: list[str] = typer.Argument(None, help="Models to download (names from `neuralframe models list`)."),
    every: bool = typer.Option(False, "--all", help="Download every model (several GB)."),
) -> None:
    """Download models ahead of time (otherwise each one is downloaded when it is first used)."""
    settings = _state(ctx).settings
    registry = ModelRegistry.load()
    library = ModelLibrary.from_settings(settings)
    if not model_ids and not every:
        missing = [m for m in registry.all() if m.weights and not library.is_ready(m)]
        total = sum((m.weights.size_bytes or 0) for m in missing if m.weights)
        _fail(NeuralFrameError(ErrorCode.INVALID_INPUT, "name the models to download",
                               hint=f"e.g. `neuralframe models download hat-l-x4`; `--all` downloads all "
                               f"{len(missing)} missing models ({total / 1e9:.1f} GB)"))
        return
    try:
        specs = [registry.get(m) for m in model_ids] if model_ids else [m for m in registry.all() if m.weights]
    except NeuralFrameError as exc:
        _fail(exc)
        return
    for spec in specs:
        if spec.weights is None:
            console.print(f"{spec.id}: plain resizing, nothing to download")
            continue
        if library.is_ready(spec):
            console.print(f"[dim]{spec.id}: already on this computer[/]")
            continue
        if spec.licence_restricted:
            console.print(f"[yellow]{spec.id}: licence {spec.licence} - personal, non-commercial use only[/]")
        try:
            path = _download_with_bar(library, spec)
        except KeyboardInterrupt:
            console.print(f"[yellow]stopped[/] {spec.id}: the part already downloaded is kept; run the command again "
                          "to continue")
            raise typer.Exit(130) from None
        except NeuralFrameError as exc:
            _fail(exc)
            return
        console.print(f"[green]ok[/] {spec.id} -> {path} ({spec.licence})")


@models_app.command("delete")
def models_delete(
    ctx: typer.Context,
    model_ids: list[str] = typer.Argument(..., help="Models whose downloaded files to delete."),
) -> None:
    """Delete downloaded model files to free disk space (they are downloaded again when needed)."""
    library = ModelLibrary.from_settings(_state(ctx).settings)
    registry = ModelRegistry.load()
    try:
        for mid in model_ids:
            freed = library.delete(registry.get(mid))
            console.print(f"{mid}: " + (f"deleted ({freed / 1e6:.0f} MB freed)" if freed else "nothing downloaded"))
    except NeuralFrameError as exc:
        _fail(exc)


@models_app.command("link")
def models_link(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., help="Model the file belongs to."),
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="The model file you already have."),
) -> None:
    """Use a model file you already have (no download). The file stays where it is."""
    library = ModelLibrary.from_settings(_state(ctx).settings)
    try:
        status = library.link(ModelRegistry.load().get(model_id), file)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    if status.state == "linked-different":
        console.print("[yellow]note[/] the file differs from the published one but loads as this model; it will "
                      "be used")
    console.print(f"[green]ok[/] {model_id}: {status.text()}")


@models_app.command("unlink")
def models_unlink(ctx: typer.Context, model_id: str = typer.Argument(...)) -> None:
    """Stop using a file linked with `models link` (the file itself is not touched)."""
    try:
        ModelLibrary.from_settings(_state(ctx).settings).unlink(ModelRegistry.load().get(model_id))
    except NeuralFrameError as exc:
        _fail(exc)
        return
    console.print(f"{model_id}: no longer uses your file")


@models_app.command("check-updates")
def models_check_updates(
    ctx: typer.Context,
    model_ids: list[str] = typer.Argument(None, help="Models to check (default: every downloaded model)."),
) -> None:
    """Ask the models' websites (Hugging Face, GitHub releases) whether newer files were published."""
    library = ModelLibrary.from_settings(_state(ctx).settings)
    registry = ModelRegistry.load()
    try:
        specs = [registry.get(m) for m in model_ids] if model_ids else [
            m for m in registry.all() if m.weights and library.status(m).state == "installed"]
    except NeuralFrameError as exc:
        _fail(exc)
        return
    found = 0
    for spec in specs:
        if library.update_source(spec) is None:
            console.print(f"[dim]{spec.id}: its website does not publish version information[/]")
            continue
        try:
            info = library.check_update(spec, force=True)
        except NeuralFrameError as exc:
            console.print(f"[red]{spec.id}: {exc.message}[/]")
            continue
        if info is not None and info.available:
            found += 1
            size = f" ({info.size_bytes / 1e6:.0f} MB)" if info.size_bytes else ""
            console.print(f"[green]{spec.id}: update available{size}[/] - `neuralframe models update {spec.id}`")
        else:
            console.print(f"{spec.id}: up to date" + (f" ({info.note})" if info and info.note else ""))
    if not specs:
        console.print("No downloaded models to check.")
    out.print(found)


@models_app.command("update")
def models_update(
    ctx: typer.Context,
    model_ids: list[str] = typer.Argument(..., help="Models to update (see `models check-updates`)."),
) -> None:
    """Download the newer published file of a model and use it from now on."""
    library = ModelLibrary.from_settings(_state(ctx).settings)
    registry = ModelRegistry.load()
    for mid in model_ids:
        try:
            spec = registry.get(mid)
            path = _download_with_bar(library, spec, update=True)
        except KeyboardInterrupt:
            console.print(f"[yellow]stopped[/] {mid}: run the command again to continue")
            raise typer.Exit(130) from None
        except NeuralFrameError as exc:
            _fail(exc)
            return
        console.print(f"[green]updated[/] {mid} -> {path}")


@models_app.command("info")
def models_info(ctx: typer.Context, model_id: str = typer.Argument(...)) -> None:
    """Everything about one model: what it is for, licence, source, whether it is on this computer."""
    library = ModelLibrary.from_settings(_state(ctx).settings)
    try:
        spec = ModelRegistry.load().get(model_id)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    status = library.status(spec)
    table = Table(show_header=False, box=None, padding=(0, 1))
    rows = [("use it for", spec.purpose or "-"), ("name", spec.id), ("group", spec.category_title),
            ("about", spec.description), ("scale", f"x{spec.scale}"), ("speed", spec.speed or "-"),
            ("removes", ", ".join(spec.handles) or "-"),
            ("licence", spec.licence + (" (personal use only)" if spec.licence_restricted else "")),
            ("source", spec.source_url or "-"), ("status", status.text())]
    if spec.weights is not None:
        rows.append(("file", spec.weights.filename))
        rows.append(("updates", "checked on its website" if library.update_source(spec) else "not available"))
    for key, value in rows:
        table.add_row(f"[bold]{key}[/]", value)
    console.print(table)


@models_app.command("verify")
def models_verify(ctx: typer.Context) -> None:
    """Check the SHA256 of every model file on this computer."""
    library = ModelLibrary.from_settings(_state(ctx).settings)
    bad = 0
    for spec in ModelRegistry.load().all():
        if spec.weights is None:
            continue
        quick, _ = library.locate(spec, quick=True)
        if quick == "missing":
            continue
        state, path = library.locate(spec)
        if state == "missing":
            bad += 1
            console.print(f"[red]{spec.id}: the file is damaged or not the published one[/] - "
                          f"`neuralframe models delete {spec.id}` and download it again")
        else:
            console.print(f"[green]{spec.id}: ok[/] ({path})")
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
    from neuralframe.models.export import compare_outputs, export_onnx, onnx_path, sample_tiles

    settings = _state(ctx).settings
    try:
        spec = ModelRegistry.load().get(model_id)
        if not spec.allows("onnxruntime"):
            raise NeuralFrameError(ErrorCode.BACKEND_UNAVAILABLE,
                                   f"{model_id} runs on PyTorch only; ONNX/TensorRT export is not validated for it")
        library = ModelLibrary.from_settings(settings)
        spec = library.effective_spec(spec)
        weights = _download_with_bar(library, spec)
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
# recommend
# -------------------------------------------------------------------------------------------------------


@app.command()
def recommend(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Video or image to analyse."),
    target: str | None = typer.Option(None, "--target", "-t", help="Output size (default 4k)."),
    preset: str = typer.Option("balanced", "--preset", "-p", help="quality | balanced | fast"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Analyse a video or image and show which model `-m auto` would use, and why."""
    from neuralframe.core.planning import plan_geometry
    from neuralframe.io.ffmpeg import locate_ffmpeg
    from neuralframe.io.probe import probe
    from neuralframe.models.advisor import analyse_content
    from neuralframe.models.advisor import recommend as choose
    from neuralframe.models.registry import ModelRegistry

    settings = _state(ctx).settings
    try:
        tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
        media = probe(input_path, tools)
        geometry, _ = plan_geometry(media.require_video(), _target(target, None, None, None), 4)
        with console.status("analysing..."):
            profile = analyse_content(tools, input_path, media)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    gpu = settings.inference.device != "cpu"
    if gpu:
        try:
            from neuralframe.inference.factory import torch_cuda_available

            gpu = torch_cuda_available()
        except ImportError:
            gpu = False
    choice = choose(profile, ModelRegistry.load(), factor=geometry.out_height / max(geometry.decoded_height, 1),
                    preset=preset, is_image=media.is_image, gpu=gpu)
    if as_json:
        out.print_json(json.dumps(choice.summary()))
        return
    p = profile
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("content", f"{p.content} ({round(p.animation_score * p.frames)} of {p.frames} frames look animated)")
    table.add_row("noise", f"{p.noise:.1f} levels")
    table.add_row("compression", f"blocking {p.blockiness:.2f}"
                  + (f", {p.bits_per_pixel:.3f} bits/pixel" if p.bits_per_pixel is not None else ""))
    table.add_row("real detail", f"about {p.effective_height}p in a {p.height}p frame")
    table.add_row("upscale", f"{choice.factor:.2f}x to {geometry.out_width}x{geometry.out_height}")
    console.print(table)
    out.print(f"[bold]{choice.model}[/]  ({preset} preset)")
    for reason in choice.reasons:
        console.print(f"  - {reason}")
    if choice.alternatives:
        console.print(f"[dim]alternatives: {', '.join(choice.alternatives)}[/]")


# -------------------------------------------------------------------------------------------------------
# benchmark / compare
# -------------------------------------------------------------------------------------------------------


@app.command()
def benchmark(
    ctx: typer.Context,
    model: str = typer.Option(..., "--model", "-m", help="Model id."),
    resolution: str = typer.Option("480p", "--resolution", "-r", help="360p 480p 540p 720p 1080p 1440p 2160p"),
    frames: int = typer.Option(300, "--frames", "-n", min=8),
    target: str | None = typer.Option(None, "--target", "-t", help="Output size (default 4k)."),
    backend: str | None = typer.Option(None, "--backend", help="auto | tensorrt | torch | onnxruntime"),
    fps: str | None = typer.Option(None, "--fps", help="Also run frame generation to this rate."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Measure throughput on a generated clip: fps, peak VRAM, GPU utilisation and per-stage fps."""
    from neuralframe.quality.benchmark import RESOLUTIONS, BenchConfig, markdown_table, run_benchmark

    if resolution not in RESOLUTIONS:
        raise typer.BadParameter(f"resolution must be one of {', '.join(RESOLUTIONS)}")
    cfg = BenchConfig(model=model, resolution=resolution, target=target, frames=frames, backend=backend, fps=fps)
    try:
        with console.status(f"benchmarking {cfg.label} ({frames} frames)..."):
            row = run_benchmark(_state(ctx).settings, cfg)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    if as_json:
        out.print_json(json.dumps(row))
        return
    out.print(markdown_table([row]), soft_wrap=True)
    if row["engine_build_s"]:
        console.print(f"[dim]one-time TensorRT engine build: {row['engine_build_s']:.0f} s (not in the fps)[/]")


@app.command()
def compare(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="Source video or image."),
    output_path: Path = typer.Argument(..., exists=True, dir_okay=False, help="NeuralFrame result."),
    frame: int = typer.Option(0, "--frame", "-f", min=0, help="Source frame number."),
    out_file: Path | None = typer.Option(None, "--out", "-o", help="PNG to write."),
    crop: str | None = typer.Option(None, "--crop", help="x,y,w,h in output pixels (e.g. 800,300,480,270)."),
    zoom: int = typer.Option(1, "--zoom", min=1, max=8, help="Enlarge the crop (nearest neighbour)."),
) -> None:
    """Write a side-by-side PNG: source (bicubic to the output size) | NeuralFrame."""
    from neuralframe.core.preview import colour_match_filter, frame_png, frame_time, labelled_side_by_side
    from neuralframe.io.ffmpeg import locate_ffmpeg
    from neuralframe.io.probe import probe

    settings = _state(ctx).settings
    try:
        tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
        media = probe(input_path, tools)
        out_video = probe(output_path, tools).require_video()
        report_path = output_path.with_name(output_path.stem + ".report.json")
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
        seconds = frame_time(media, frame)
        size = (out_video.width, out_video.height)
        before = frame_png(tools, input_path, seconds, size, colour_match_filter(report))
        after = frame_png(tools, output_path, seconds)
    except NeuralFrameError as exc:
        _fail(exc)
        return
    box = None
    if crop:
        try:
            x, y, w, h = (int(v) for v in crop.split(","))
        except ValueError as exc:
            raise typer.BadParameter("--crop takes x,y,w,h") from exc
        box = (x, y, x + w, y + h)
    target = out_file or output_path.with_name(f"{output_path.stem}_compare_f{frame}.png")
    labelled_side_by_side(before, after, target, box=box, zoom=zoom)
    console.print(f"wrote {target}")


# -------------------------------------------------------------------------------------------------------
# gui / serve / queue
# -------------------------------------------------------------------------------------------------------


@app.command()
def gui(
    input_path: Path | None = typer.Argument(None, help="Video or image to open."),
    config: Path | None = typer.Option(None, "--config", "-c", help="YAML config file."),
) -> None:
    """Open the desktop application (same as `neuralframe-gui`)."""
    from neuralframe.gui.app import main as gui_main

    args = [str(input_path)] if input_path else []
    if config:
        args += ["--config", str(config)]
    raise typer.Exit(gui_main(args))


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
    from neuralframe.db.store import JobStore

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
