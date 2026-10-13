"""Quality evaluation: downscale a real high-resolution clip, upscale it again, compare with the original.

Usage: python benchmarks/quality.py [--sources sintel,tos] [--models a,b] [--seconds 6] [--out benchmarks/results]
                                    [--gallery docs/gallery]

References (both (c) Blender Foundation, CC BY 3.0):
  sintel  the Sintel trailer (computer animation), downloaded once and SHA256-checked; a 1920x816 segment
          (letterbox removed) is the ground truth.
  tos     Tears of Steel (filmed, live action), 1920x800; only the excerpt is read from the archive.org copy
          (HTTP range requests), so the whole 575 MB file is never downloaded.
Low-resolution inputs are made with bicubic downscaling, x4 losslessly and x4 with x264 CRF 28 (compression
artefacts), and x2. Methods: FFmpeg bicubic and Lanczos (baselines), every NeuralFrame model for that scale,
and the automatic model choice for each preset, all at exactly the reference size and encoded losslessly so
codec loss does not enter the scores. Writes a Markdown + JSON report and gallery crops. The transformer
models (very slow) make a full run take hours; use --models to measure a subset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neuralframe.config import Settings, default_cache_dir, load_settings
from neuralframe.core.job import JobRunner, JobSpec
from neuralframe.core.planning import TargetSpec
from neuralframe.core.preview import frame_png, labelled_side_by_side
from neuralframe.io.ffmpeg import FFmpegTools, locate_ffmpeg, popen_kwargs
from neuralframe.io.probe import probe
from neuralframe.quality.benchmark import hardware_info
from neuralframe.quality.metrics import (
    LpipsScorer,
    compare_videos,
    flicker,
    frame_differences,
    iter_luma,
    lpips_videos,
    psnr,
    seam_ratio,
    vmaf,
)

REFERENCE_URL = "https://download.blender.org/durian/trailer/sintel_trailer-1080p.mp4"
REFERENCE_SHA256 = "34bbd52a4b89fdf63c8ace50b268da26653a59508288100cd3c23de276db7931"
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
LOSSLESS = {"encoder": "libx264", "codec": "h264", "quality": 0}


SOURCES: dict[str, dict[str, Any]] = {
    "sintel": {
        "label": "Sintel trailer (c) Blender Foundation | durian.blender.org, CC BY 3.0",
        "url": REFERENCE_URL,
        "sha256": REFERENCE_SHA256,
        "crop": "crop=1920:816:0:132",  # remove the letterbox (black bars would inflate every score)
        "hr": (1920, 816),
        "start": 32.5,  # detailed shots, no fades or titles
        "box": (840, 120, 1400, 435),  # gallery crop: Sintel's face and hair
        "prefix": "",
    },
    "tos": {
        "label": "Tears of Steel (c) Blender Foundation | mango.blender.org, CC BY 3.0 (filmed)",
        "url": "https://archive.org/download/Tears-of-Steel/tears_of_steel_1080p.mkv",
        "sha256": None,  # streamed excerpt; the size is checked instead
        "crop": None,
        "hr": (1920, 800),
        "start": 240.0,  # three actors indoors: faces, hair, beard, fabric
        "box": (300, 110, 860, 425),
        "prefix": "tos-",
    },
}

X4_MODELS = ["realesr-general-x4v3", "realesrgan-x4plus", "realesr-animevideov3", "realbasicvsr-x4", "bsrgan-x4",
             "swinir-l-x4", "atd-x4", "dat2-real-x4", "drct-l-x4", "hat-l-x4"]
SCENARIOS: dict[str, dict[str, Any]] = {
    "x4": {"factor": 4, "crf": None, "models": [*X4_MODELS, "lanczos"]},
    "x4-compressed": {"factor": 4, "crf": 28, "models": X4_MODELS},
    "x2": {"factor": 2, "crf": None, "models": ["realesrgan-x2plus", "realplksr-x2", "lanczos"]},
}
AUTO_PRESETS = ("fast", "balanced", "quality")


def ffmpeg(tools: FFmpegTools, args: list[str]) -> None:
    subprocess.run([tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args], check=True,
                   timeout=3600, **popen_kwargs())


def fetch_reference(cache: Path, source: dict[str, Any]) -> str:
    """Local path of a downloaded, hash-checked reference; or the URL of a streamed one."""
    if source["sha256"] is None:
        return str(source["url"])
    path = cache / "datasets" / str(source["url"]).rsplit("/", 1)[-1]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {source['url']}")
        urllib.request.urlretrieve(source["url"], path)  # fixed https URL, hash-checked below
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        path.unlink()
        raise SystemExit(f"reference clip checksum mismatch ({digest}); deleted, run again")
    return str(path)


def lr_size(hr: tuple[int, int], factor: int) -> tuple[int, int]:
    return hr[0] // factor, hr[1] // factor


def prepare(tools: FFmpegTools, source: dict[str, Any], location: str, work: Path, seconds: float
            ) -> dict[str, Path]:
    work.mkdir(parents=True, exist_ok=True)
    hr = work / "hr.mkv"
    if not hr.is_file():
        vf = ["-vf", source["crop"]] if source["crop"] else []
        ffmpeg(tools, ["-ss", str(source["start"]), "-t", str(seconds), "-i", location, "-map", "0:v:0", *vf,
                       "-c:v", "libx264", "-qp", "0", "-preset", "fast", *TAGS, str(hr)])
        video = probe(hr, tools).require_video()
        if (video.width, video.height) != tuple(source["hr"]):
            hr.unlink()
            raise SystemExit(f"{location}: expected {source['hr']}, got {video.width}x{video.height}")
    clips = {"hr": hr}
    for name, sc in SCENARIOS.items():
        lr = work / f"lr-{name}.mkv"
        if not lr.is_file():
            w, h = lr_size(source["hr"], sc["factor"])
            rate = ["-crf", str(sc["crf"]), "-preset", "medium"] if sc["crf"] else ["-qp", "0", "-preset", "fast"]
            ffmpeg(tools, ["-i", str(hr), "-vf", f"scale={w}:{h}:flags=bicubic", "-c:v", "libx264", *rate, *TAGS,
                           str(lr)])
        clips[name] = lr
    return clips


def upscale(settings: Settings, model: str, src: Path, out: Path, hr: tuple[int, int]) -> dict[str, Any]:
    spec = JobSpec(input=src, output=out, model=model, target=TargetSpec.exact(*hr), chunk_frames=24,
                   overwrite=True, **LOSSLESS)
    t0 = time.perf_counter()
    report = JobRunner(settings).run(spec).report
    report["_wall_s"] = time.perf_counter() - t0
    return report


def auto_choice(settings: Settings, tools: FFmpegTools, src: Path, hr: tuple[int, int], preset: str
                ) -> tuple[str, list[str]]:
    """The model ``-m auto`` picks for this input and preset (content analysis only, nothing is rendered)."""
    spec = JobSpec(input=src, output=src.with_suffix(".auto.mkv"), model="auto", preset=preset,
                   target=TargetSpec.exact(*hr))
    runner = JobRunner(settings)
    model = runner.resolve_model(spec, tools, probe(src, tools))
    return model.id, list((runner.model_choice or {}).get("reasons", []))


def score(tools: FFmpegTools, distorted: Path, hr: Path, *, size: tuple[int, int] | None, flags: str,
          lpips: LpipsScorer | None) -> dict[str, Any]:
    fid = compare_videos(tools, distorted, hr, distorted_size=size, scale_flags=flags)
    fid.vmaf = vmaf(tools, distorted, hr, distorted_size=size, scale_flags=flags)
    if lpips is not None:
        fid.lpips = lpips_videos(tools, distorted, hr, distorted_size=size, scale_flags=flags, every=4,
                                 scorer=lpips)
    out = fid.as_dict()
    out["flicker"] = round(flicker(iter_luma(tools, distorted, size=size, scale_flags=flags),
                                   iter_luma(tools, hr)), 4)
    return out


def seams(tools: FFmpegTools, output: Path, report: dict[str, Any], hr: Path, baseline: list[float]
          ) -> dict[str, float | None]:
    """Seam ratio at the job's chunk boundaries, raw and divided by the same ratio of the (chunk-free)
    bicubic baseline at the same frames, which cancels what the content itself does there."""
    diffs = frame_differences(iter_luma(tools, output))
    hr_diffs = frame_differences(iter_luma(tools, hr))
    cuts = [i for i, d in enumerate(hr_diffs) if d > 12.0]  # shot changes in the reference
    boundaries = report["plan"]["chunks"].get("boundaries", [])
    raw = seam_ratio(diffs, boundaries, exclude=cuts)
    base = seam_ratio(baseline, boundaries, exclude=cuts)
    return {"seam_ratio": raw, "seam_ratio_vs_baseline": (raw / base) if raw is not None and base else None}


def tile_seam_test(settings: Settings, tools: FFmpegTools, lr: Path, work: Path, hr: tuple[int, int]
                   ) -> dict[str, Any]:
    """Real-ESRGAN x4plus with 128 px tiles vs whole frames (PyTorch, same weights): PSNR between the two."""
    first = work / "tile-src.mkv"
    ffmpeg(tools, ["-i", str(lr), "-frames:v", "8", "-c:v", "libx264", "-qp", "0", *TAGS, str(first)])
    outs = {}
    for name, tile in (("whole", None), ("tiled", 128)):
        out = work / f"tile-{name}.mkv"
        JobRunner(settings).run(JobSpec(input=first, output=out, model="realesrgan-x4plus", backend="torch",
                                        target=TargetSpec.exact(*hr), tile_size=tile, tile_overlap=32,
                                        overwrite=True, **LOSSLESS))
        outs[name] = out
    values = [psnr(a, b) for a, b in zip(iter_luma(tools, outs["tiled"]), iter_luma(tools, outs["whole"]),
                                         strict=False)]
    return {"model": "realesrgan-x4plus", "tile": 128, "overlap": 32, "frames": len(values),
            "psnr_tiled_vs_whole_db": round(min(values), 2)}


def gallery(tools: FFmpegTools, hr: Path, lr: Path, outputs: dict[str, Path], folder: Path, name: str,
            frame_s: float, box: tuple[int, int, int, int], size: tuple[int, int]) -> list[str]:
    """Crops at 2x: bicubic | model for each model, plus the ground truth."""
    names = []
    bicubic = frame_png(tools, lr, frame_s, size)
    for model, out in outputs.items():
        target = folder / f"{name}-{model}.png"
        labelled_side_by_side(bicubic, frame_png(tools, out, frame_s), target, box=box, zoom=2,
                              labels=("bicubic", model))
        names.append(target.name)
    truth = folder / f"{name}-ground-truth.png"
    labelled_side_by_side(bicubic, frame_png(tools, hr, frame_s), truth, box=box, zoom=2,
                          labels=("bicubic", "original"))
    names.append(truth.name)
    return names


def run_source(settings: Settings, tools: FFmpegTools, key: str, cache: Path, args: argparse.Namespace,
               scorer: LpipsScorer | None, models: set[str] | None) -> dict[str, Any]:
    source = SOURCES[key]
    hr_size: tuple[int, int] = source["hr"]
    work = cache / "quality" / key
    clips = prepare(tools, source, fetch_reference(cache, source), work, args.seconds)
    hr = clips["hr"]
    scenarios: dict[str, Any] = {}
    for name, sc in SCENARIOS.items():
        lr = clips[name]
        w, h = lr_size(hr_size, sc["factor"])
        rows: dict[str, Any] = {}
        print(f"== {key} {name}: {w}x{h} -> {hr_size[0]}x{hr_size[1]}")
        for flags in ("bicubic", "lanczos"):
            rows[f"ffmpeg {flags}"] = score(tools, lr, hr, size=hr_size, flags=flags, lpips=scorer)
            print(f"  ffmpeg {flags}: {rows[f'ffmpeg {flags}']}")
        baseline_diffs = frame_differences(iter_luma(tools, lr, size=hr_size, scale_flags="bicubic"))
        outputs: dict[str, Path] = {}

        def measure(model: str, lr: Path = lr, name: str = name, rows: dict[str, Any] = rows,
                    outputs: dict[str, Path] = outputs, baseline: list[float] = baseline_diffs) -> dict[str, Any]:
            if model in outputs:
                return dict(rows[f"NeuralFrame {model}"])
            out = work / f"out-{name}-{model}.mkv"
            report = upscale(settings, model, lr, out, hr_size)
            row = score(tools, out, hr, size=None, flags="bicubic", lpips=scorer)
            if model != "lanczos":
                row.update(seams(tools, out, report, hr, baseline))
            row["fps"] = report["throughput"]["pipeline_fps"]
            row["backend"] = (report["plan"].get("inference") or {}).get("backend", "-")
            row["precision"] = (report["plan"].get("inference") or {}).get("precision", "-")
            rows[f"NeuralFrame {model}"] = row
            outputs[model] = out
            print(f"  {model}: {row}")
            return row

        for model in sc["models"]:
            if models is None or model in models:
                measure(model)
        auto: dict[str, Any] = {}
        for preset in AUTO_PRESETS:
            chosen, reasons = auto_choice(settings, tools, lr, hr_size, preset)
            auto[preset] = {"model": chosen, "reasons": reasons}
            print(f"  auto ({preset}) -> {chosen}: {'; '.join(reasons)}")
            if models is None or chosen in models:
                rows[f"auto ({preset}) = {chosen}"] = measure(chosen)
        crops = gallery(tools, hr, lr, {m: o for m, o in outputs.items() if m != "lanczos"}, args.gallery,
                        f"{source['prefix']}{name}", frame_s=0.5, box=source["box"], size=hr_size)
        scenarios[name] = {"input": f"{w}x{h}", "crf": sc["crf"], "rows": rows, "auto": auto, "gallery": crops}
    result: dict[str, Any] = {"source": source["label"], "url": source["url"], "sha256": source["sha256"],
                              "segment": f"{source['start']}s + {args.seconds}s, {hr_size[0]}x{hr_size[1]}",
                              "scenarios": scenarios}
    if key == "sintel" and (models is None or "realesrgan-x4plus" in models):
        result["tile_seam"] = tile_seam_test(settings, tools, clips["x4"], work, hr_size)
        print("tile seam:", result["tile_seam"])
    return result


def report_lines(results: dict[str, Any]) -> list[str]:
    lines = [f"# Quality evaluation - {results['hardware'].get('gpu', results['hardware']['cpu'])}", "",
             "Higher is better for PSNR / SSIM / VMAF; lower is better for LPIPS (perceptual distance: the best "
             "measure of whether detail looks real rather than smeared) and flicker (temporal noise in static "
             "regions, luma levels). PSNR uses MSE pooled over all frames; it rewards smooth, averaged output, "
             "so a model that invents plausible texture can lose PSNR while looking much closer to the original. "
             "Seam ratio: frame-to-frame change at chunk boundaries / elsewhere, divided by the same ratio of "
             "the chunk-free bicubic baseline at those frames (~1.0 = invisible; the spec flags > 1.2). "
             "`auto (preset) = model` rows show what `-m auto` picked for that input.", ""]
    for ref in results["references"].values():
        lines += [f"## {ref['source']}", "", f"Segment: {ref['segment']}.", ""]
        for name, sc in ref["scenarios"].items():
            hr = ref["segment"].rsplit(" ", 1)[-1]
            lines += [f"### {name}: {sc['input']} -> {hr}" + (f" (input x264 CRF {sc['crf']})" if sc["crf"]
                                                             else ""), "",
                      "| method | PSNR-Y | SSIM-Y | VMAF | LPIPS | flicker | seam ratio | fps |",
                      "|---|---|---|---|---|---|---|---|"]
            for method, r in sc["rows"].items():
                lp = "-" if r["lpips"] is None else f"{r['lpips']:.4f}"
                seam = "-" if r.get("seam_ratio_vs_baseline") is None else f"{r['seam_ratio_vs_baseline']:.3f}"
                fps = "-" if r.get("fps") is None else f"{r['fps']:.1f}"
                lines.append(f"| {method} | {r['psnr_y']:.2f} | {r['ssim_y']:.4f} | {r['vmaf']:.2f} | {lp} | "
                             f"{r['flicker']:.3f} | {seam} | {fps} |")
            lines.append("")
        ts = ref.get("tile_seam")
        if ts:
            lines += ["### Tile seams", "", f"{ts['model']} with {ts['tile']} px tiles ({ts['overlap']} px overlap) "
                      f"vs whole frames: worst frame {ts['psnr_tiled_vs_whole_db']} dB PSNR over {ts['frames']} "
                      "frames (spec threshold: > 45 dB).", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="sintel,tos", help=f"comma-separated: {', '.join(SOURCES)}")
    parser.add_argument("--models", help="comma-separated model ids to measure (default: all for each scale)")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--gallery", type=Path, default=Path(__file__).parents[1] / "docs" / "gallery")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # progress lines appear even when redirected
    keys = [k.strip() for k in args.sources.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SOURCES]
    if unknown:
        parser.error(f"unknown source(s): {', '.join(unknown)}")
    models = {m.strip() for m in args.models.split(",") if m.strip()} if args.models else None

    settings = load_settings(args.config)
    tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
    cache = settings.paths.cache_dir or default_cache_dir()
    scorer = LpipsScorer() if LpipsScorer.available() and not args.no_lpips else None
    results: dict[str, Any] = {"hardware": hardware_info(), "references": {}}
    for key in keys:
        results["references"][key] = run_source(settings, tools, key, cache, args, scorer, models)

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"quality-{stamp}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    lines = report_lines(results)
    (args.out / f"quality-{stamp}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
