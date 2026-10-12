"""Throughput benchmarks: one job per configuration on a generated clip, summarised from the job report."""

from __future__ import annotations

import platform
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neuralframe import __version__
from neuralframe.config import Settings
from neuralframe.core.job import JobRunner, JobSpec
from neuralframe.core.planning import TargetSpec
from neuralframe.io.ffmpeg import locate_ffmpeg
from neuralframe.monitoring import cpu_name, driver_versions, host_memory

RESOLUTIONS = {"360p": (640, 360), "480p": (854, 480), "540p": (960, 540), "720p": (1280, 720),
               "1080p": (1920, 1080), "1440p": (2560, 1440), "2160p": (3840, 2160)}


@dataclass
class BenchConfig:
    model: str
    resolution: str = "480p"
    target: str | None = None
    """Output size (TargetSpec syntax); None = the product default (4K)."""
    frames: int = 300
    backend: str | None = None
    fps: str | None = None
    """Frame generation target; None = no frame generation."""
    upscale: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        parts = [self.model, self.backend or "auto", self.resolution, self.target or "4k"]
        if self.fps:
            parts.append(f"{self.fps}fps")
        return " / ".join(parts)


def bench_clip(settings: Settings, resolution: str, frames: int) -> Path:
    """A cached 29.97 fps synthetic clip (moving test pattern + stereo AAC) of ``frames`` frames."""
    from neuralframe.testing.media import ClipSpec, make_clip

    width, height = RESOLUTIONS[resolution]
    path = settings.paths.cache_dir / "bench" / f"clip-{resolution}-{frames}.mp4"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
        make_clip(tools, path, ClipSpec(width=width, height=height, fps="30000/1001",
                                        duration=frames * 1001 / 30000, container="mp4", crf=18))
    return path


def run_benchmark(settings: Settings, cfg: BenchConfig, *, source: Path | None = None) -> dict[str, Any]:
    """Run one configuration; returns a flat row for tables (pipeline fps excludes one-time engine builds)."""
    src = source or bench_clip(settings, cfg.resolution, cfg.frames)
    with tempfile.TemporaryDirectory(prefix="nf-bench-") as tmp:
        spec = JobSpec(
            input=src, output=Path(tmp) / "out.mkv", model=cfg.model,
            target=TargetSpec.parse(cfg.target) if cfg.target else TargetSpec.default(),
            backend=cfg.backend, upscale=cfg.upscale, interpolate=cfg.fps is not None, target_fps=cfg.fps,
            overwrite=True, **cfg.extra,
        )
        report = JobRunner(settings).run(spec).report
    tp, res = report["throughput"], report["resources"]
    gpu: dict[str, Any] = next(iter(res.get("gpus", {}).values()), {})
    runner = (report.get("processor") or {}).get("runner") or {}
    inference = report["plan"].get("inference") or {}
    geometry = report["plan"]["geometry"]
    return {
        "config": cfg.label,
        "model": cfg.model,
        "backend": runner.get("backend") or inference.get("backend") or "-",
        "precision": runner.get("precision") or inference.get("precision") or "-",
        "input": geometry["decoded"],
        "output": geometry["output"],
        "frames": report["output"]["frames"],
        "fps": tp["pipeline_fps"],
        "realtime_x": round(tp["pipeline_fps"] / 29.97, 2),
        "bottleneck": tp["bottleneck"],
        "stages": {s["name"]: s["busy_fps"] for s in tp["stages"]},
        "peak_vram_mb": gpu.get("peak_vram_job_mb"),
        "gpu_util_pct": gpu.get("mean_utilization_pct"),
        "tile": inference.get("tile"),
        "batch": inference.get("batch"),
        "engine_build_s": runner.get("build_seconds") or 0.0,
        "pipeline_s": report["timings"]["pipeline_s"],
        "framegen": (report.get("framegen") or {}).get("order"),
    }


def hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {"neuralframe": __version__, "platform": platform.platform(), "cpu": cpu_name(),
                            "ram_gb": round(host_memory()[0] / 2**30, 1), "python": platform.python_version()}
    driver, cuda = driver_versions()
    info.update(nvidia_driver=driver, cuda_driver=cuda)
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        pass
    try:
        import tensorrt

        info["tensorrt"] = tensorrt.__version__
    except ImportError:
        pass
    return info


def markdown_table(rows: list[dict[str, Any]]) -> str:
    head = ("| model | backend | input -> output | frames | fps | x real time | peak VRAM | GPU util | "
            "bottleneck | per-stage busy fps |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['model']} | {r.get('backend', '-')} | {r.get('config', '')} | | failed: "
                         f"{r['error'][:60]} | | | | | |")
            continue
        stages = ", ".join(f"{k} {v:.0f}" for k, v in r["stages"].items())
        vram = f"{r['peak_vram_mb']} MB" if r["peak_vram_mb"] is not None else "-"
        util = f"{r['gpu_util_pct']:.0f}%" if r["gpu_util_pct"] is not None else "-"
        fg = f" + {r['framegen'].replace('_', ' ')}" if r.get("framegen") else ""
        lines.append(f"| {r['model']}{fg} | {r['backend']} {r['precision']} | {r['input']} -> {r['output']} | "
                     f"{r['frames']} | {r['fps']:.2f} | {r['realtime_x']:.2f} | {vram} | {util} | "
                     f"{r['bottleneck']} | {stages} |")
    return "\n".join(lines)


def as_dict(cfg: BenchConfig) -> dict[str, Any]:
    return asdict(cfg)
