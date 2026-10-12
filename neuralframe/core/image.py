"""Still-image upscaling (PNG, JPEG, WebP, BMP, TIFF) through the same model runner as video.

FFmpeg decodes any input to planar float RGB(A) (EXIF orientation applied by autorotate) and encodes the
result, so 16-bit PNG/TIFF keep their precision end to end.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from neuralframe import __version__
from neuralframe.core.planning import plan_geometry
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, popen_kwargs, tail
from neuralframe.io.media import MediaInfo
from neuralframe.log import get_logger
from neuralframe.ops import get_ops

if TYPE_CHECKING:
    from neuralframe.core.job import JobResult, JobRunner, JobSpec

log = get_logger(component="image")

F32 = npt.NDArray[np.float32]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def output_args(path: Path, *, sixteen_bit: bool, alpha: bool) -> list[str]:
    ext = path.suffix.lower()
    if ext == ".png":
        pix = ("rgba64be" if alpha else "rgb48be") if sixteen_bit else ("rgba" if alpha else "rgb24")
        return ["-c:v", "png", "-pix_fmt", pix, "-f", "image2"]
    if ext in (".tif", ".tiff"):
        pix = ("rgba64le" if alpha else "rgb48le") if sixteen_bit else ("rgba" if alpha else "rgb24")
        return ["-c:v", "tiff", "-pix_fmt", pix, "-f", "image2"]
    if ext in (".jpg", ".jpeg"):
        return ["-c:v", "mjpeg", "-q:v", "1", "-pix_fmt", "yuvj444p", "-f", "image2"]
    if ext == ".webp":
        return ["-c:v", "libwebp", "-quality", "95", "-pix_fmt", "yuva420p" if alpha else "yuv420p", "-f", "webp"]
    if ext == ".bmp":
        return ["-c:v", "bmp", "-pix_fmt", "bgra" if alpha else "bgr24", "-f", "image2"]
    raise NeuralFrameError(
        ErrorCode.INVALID_INPUT,
        f"unsupported image output {path.suffix!r}",
        hint=f"use one of {', '.join(sorted(IMAGE_SUFFIXES))}",
    )


def decode_image(tools: FFmpegTools, path: Path, width: int, height: int, alpha: bool) -> tuple[F32, F32 | None]:
    """Decode to planar float32: RGB ``(3, H, W)`` in [0, 1] and optional alpha ``(H, W)``."""
    pix = "gbrapf32le" if alpha else "gbrpf32le"
    proc = subprocess.run(
        [tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", pix, "pipe:1"],
        capture_output=True, timeout=600, **popen_kwargs(),
    )
    planes = 4 if alpha else 3
    expected = planes * width * height * 4
    if proc.returncode != 0 or len(proc.stdout) != expected:
        raise NeuralFrameError(
            ErrorCode.DECODE_FAILED,
            f"cannot decode image {path.name} ({len(proc.stdout)} of {expected} bytes): {tail(proc.stderr)}",
        )
    data = np.frombuffer(proc.stdout, dtype="<f4").reshape(planes, height, width)
    rgb = np.stack([data[2], data[0], data[1]]).astype(np.float32)  # gbrp → RGB
    return np.clip(rgb, 0.0, 1.0), (np.clip(data[3], 0.0, 1.0).astype(np.float32) if alpha else None)


def encode_image(
    tools: FFmpegTools, path: Path, rgb: F32, alpha: F32 | None, *, sixteen_bit: bool
) -> None:
    _, height, width = rgb.shape
    planes = [rgb[1], rgb[2], rgb[0]] + ([alpha] if alpha is not None else [])
    payload = np.ascontiguousarray(np.stack(planes), dtype="<f4").tobytes()
    partial = path.with_name(f"{path.stem}.partial{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "gbrapf32le" if alpha is not None else "gbrpf32le",
        "-s", f"{width}x{height}", "-i", "pipe:0", "-frames:v", "1", "-update", "1",
        *output_args(path, sixteen_bit=sixteen_bit, alpha=alpha is not None), str(partial),
    ]
    proc = subprocess.run(cmd, input=payload, capture_output=True, timeout=600, **popen_kwargs())
    if proc.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise NeuralFrameError(ErrorCode.ENCODE_FAILED, f"cannot write {path.name}: {tail(proc.stderr)}")
    partial.replace(path)


def run_image_job(runner: JobRunner, spec: JobSpec, tools: FFmpegTools, media: MediaInfo) -> JobResult:
    from neuralframe.core.job import JobResult

    t0 = time.perf_counter()
    if not is_image_path(spec.output):
        raise NeuralFrameError(
            ErrorCode.INVALID_INPUT,
            f"image input needs an image output, got {spec.output.suffix!r}",
            hint="use .png, .jpg, .webp, .tif or .bmp",
        )
    video = media.require_video()
    model = runner.resolve_model(spec)
    geometry, notes = plan_geometry(video, spec.target, model.scale, even_dims=False, cascade=True)
    alpha = video.pixel_format.has_alpha
    sixteen = bool(spec.ten_bit) or video.bit_depth > 8
    runner._emit("planning")
    mode = runner.choose_mode(model)
    inference, weights = runner.plan_model(spec, model, geometry, mode, notes)

    rgb, alpha_plane = decode_image(tools, spec.input, geometry.decoded_width, geometry.decoded_height, alpha)
    runner._frames_total = 1
    runner._emit("running", "upscaling image")
    t_infer = time.perf_counter()
    out_hw = (geometry.out_height, geometry.out_width)
    stats: dict[str, Any] = {}
    if inference is not None:
        import torch

        from neuralframe.ops import torch_ops

        model_runner, inference = runner._calibrated_runner(model, weights, inference, geometry, calibrate=False)
        try:
            x = torch.from_numpy(rgb[None]).to(model_runner.device)
            for _ in range(geometry.model_passes):
                x = model_runner.run(x)
            x = torch_ops.resize(x, *out_hw, "lanczos3").clamp_(0.0, 1.0)
            out_rgb = x[0].cpu().numpy().astype(np.float32)
            stats = {"tiles": model_runner.stats.tiles, "infer_calls": model_runner.stats.infer_calls}
        finally:
            model_runner.backend.close()
    else:
        ops = get_ops(runner.settings.pipeline.ops)
        out_rgb = np.clip(ops.resize(rgb, *out_hw, model.resize_kernel), 0.0, 1.0)
    out_alpha = None
    if alpha_plane is not None:
        out_alpha = np.clip(get_ops("numpy").resize(alpha_plane[None], *out_hw, "lanczos3")[0], 0.0, 1.0)
    infer_s = time.perf_counter() - t_infer

    runner._emit("muxing", "writing image")
    encode_image(tools, spec.output, out_rgb, out_alpha, sixteen_bit=sixteen and spec.output.suffix.lower() in (
        ".png", ".tif", ".tiff"))
    total = time.perf_counter() - t0
    report: dict[str, Any] = {
        "neuralframe": __version__,
        "job_id": runner.job_id,
        "status": "done",
        "kind": "image",
        "input": {"path": str(spec.input), "size": f"{video.width}x{video.height}", "pix_fmt": video.pix_fmt,
                  "rotation": video.rotation},
        "output": {"path": str(spec.output), "size": f"{geometry.out_width}x{geometry.out_height}",
                   "alpha": alpha, "sixteen_bit": sixteen},
        "settings": json.loads(spec.model_dump_json()),
        "plan": {"model": model.id, "geometry": geometry.summary(), "mode": mode,
                 "inference": inference.summary() if inference else None, "notes": notes},
        "timings": {"inference_s": round(infer_s, 3), "total_s": round(total, 3)},
        "processor": stats,
        "warnings": notes + runner._fallbacks,
    }
    report_path = spec.output.with_name(spec.output.stem + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    runner._emit("done", f"wrote {spec.output}")
    log.info("image done", output=str(spec.output), size=report["output"]["size"], seconds=round(total, 2))
    return JobResult(job_id=runner.job_id, output=spec.output, report_path=report_path, report=report)
