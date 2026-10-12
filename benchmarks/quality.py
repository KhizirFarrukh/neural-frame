"""Quality evaluation: downscale a real high-resolution clip, upscale it again, compare with the original.

Usage: python benchmarks/quality.py [--seconds 6] [--out benchmarks/results] [--gallery docs/gallery]

Reference: the Sintel trailer, (c) Blender Foundation | durian.blender.org, CC BY 3.0 (downloaded once,
SHA256-checked). A 1920x816 segment (letterbox removed) is the ground truth; low-resolution inputs are
made with bicubic downscaling, x4 losslessly and x4 with x264 CRF 28 (compression artefacts), and x2.
Methods: FFmpeg bicubic and Lanczos (baselines) and NeuralFrame models, all at exactly the reference size
and encoded losslessly so codec loss does not enter the scores. Writes a Markdown + JSON report and
gallery crops.
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
CROP = "crop=1920:816:0:132"  # remove the letterbox (black bars would inflate every score)
HR = (1920, 816)
TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
LOSSLESS = {"encoder": "libx264", "codec": "h264", "quality": 0}

SCENARIOS = {
    "x4": {"size": (480, 204), "crf": None, "models": ["realesrgan-x4plus", "realesr-animevideov3",
                                                       "realbasicvsr-x4", "lanczos"]},
    "x4-compressed": {"size": (480, 204), "crf": 28, "models": ["realesrgan-x4plus", "realesr-animevideov3",
                                                                "realbasicvsr-x4"]},
    "x2": {"size": (960, 408), "crf": None, "models": ["realesrgan-x2plus", "lanczos"]},
}


def ffmpeg(tools: FFmpegTools, args: list[str]) -> None:
    subprocess.run([tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *args], check=True,
                   timeout=3600, **popen_kwargs())


def fetch_reference(cache: Path) -> Path:
    path = cache / "datasets" / "sintel_trailer-1080p.mp4"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {REFERENCE_URL}")
        urllib.request.urlretrieve(REFERENCE_URL, path)  # fixed https URL, hash-checked below
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != REFERENCE_SHA256:
        path.unlink()
        raise SystemExit(f"reference clip checksum mismatch ({digest}); deleted, run again")
    return path


def prepare(tools: FFmpegTools, source: Path, work: Path, start: float, seconds: float) -> dict[str, Path]:
    work.mkdir(parents=True, exist_ok=True)
    hr = work / "hr.mkv"
    if not hr.is_file():
        ffmpeg(tools, ["-ss", str(start), "-t", str(seconds), "-i", str(source), "-map", "0:v:0", "-vf", CROP,
                       "-c:v", "libx264", "-qp", "0", "-preset", "fast", *TAGS, str(hr)])
    clips = {"hr": hr}
    for name, sc in SCENARIOS.items():
        lr = work / f"lr-{name}.mkv"
        if not lr.is_file():
            w, h = sc["size"]
            rate = ["-crf", str(sc["crf"]), "-preset", "medium"] if sc["crf"] else ["-qp", "0", "-preset", "fast"]
            ffmpeg(tools, ["-i", str(hr), "-vf", f"scale={w}:{h}:flags=bicubic", "-c:v", "libx264", *rate, *TAGS,
                           str(lr)])
        clips[name] = lr
    return clips


def upscale(settings: Settings, model: str, src: Path, out: Path) -> dict[str, Any]:
    spec = JobSpec(input=src, output=out, model=model, target=TargetSpec.exact(*HR), chunk_frames=24,
                   overwrite=True, **LOSSLESS)
    t0 = time.perf_counter()
    report = JobRunner(settings).run(spec).report
    report["_wall_s"] = time.perf_counter() - t0
    return report


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


def tile_seam_test(settings: Settings, tools: FFmpegTools, lr: Path, work: Path) -> dict[str, Any]:
    """Real-ESRGAN x4plus with 128 px tiles vs whole frames (PyTorch, same weights): PSNR between the two."""
    first = work / "tile-src.mkv"
    ffmpeg(tools, ["-i", str(lr), "-frames:v", "8", "-c:v", "libx264", "-qp", "0", *TAGS, str(first)])
    outs = {}
    for name, tile in (("whole", None), ("tiled", 128)):
        out = work / f"tile-{name}.mkv"
        JobRunner(settings).run(JobSpec(input=first, output=out, model="realesrgan-x4plus", backend="torch",
                                        target=TargetSpec.exact(*HR), tile_size=tile, tile_overlap=32,
                                        overwrite=True, **LOSSLESS))
        outs[name] = out
    values = [psnr(a, b) for a, b in zip(iter_luma(tools, outs["tiled"]), iter_luma(tools, outs["whole"]),
                                         strict=False)]
    return {"model": "realesrgan-x4plus", "tile": 128, "overlap": 32, "frames": len(values),
            "psnr_tiled_vs_whole_db": round(min(values), 2)}


def gallery(tools: FFmpegTools, hr: Path, lr: Path, outputs: dict[str, Path], folder: Path, scenario: str,
            frame_s: float, box: tuple[int, int, int, int]) -> list[str]:
    """Crops at 2x: bicubic | model for each model, plus the ground truth."""
    names = []
    bicubic = frame_png(tools, lr, frame_s, HR)
    for model, out in outputs.items():
        target = folder / f"{scenario}-{model}.png"
        labelled_side_by_side(bicubic, frame_png(tools, out, frame_s), target, box=box, zoom=2,
                              labels=("bicubic", model))
        names.append(target.name)
    truth = folder / f"{scenario}-ground-truth.png"
    labelled_side_by_side(bicubic, frame_png(tools, hr, frame_s), truth, box=box, zoom=2,
                          labels=("bicubic", "original"))
    names.append(truth.name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=float, default=32.5)  # detailed shots, no fades or titles
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--gallery", type=Path, default=Path(__file__).parents[1] / "docs" / "gallery")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # progress lines appear even when redirected

    settings = load_settings(args.config)
    tools = locate_ffmpeg(settings.ffmpeg.ffmpeg, settings.ffmpeg.ffprobe)
    cache = settings.paths.cache_dir or default_cache_dir()
    work = cache / "quality"
    clips = prepare(tools, fetch_reference(cache), work, args.start, args.seconds)
    hr = clips["hr"]
    scorer = LpipsScorer() if LpipsScorer.available() and not args.no_lpips else None

    results: dict[str, Any] = {"hardware": hardware_info(), "reference": {
        "source": "Sintel trailer (c) Blender Foundation | durian.blender.org, CC BY 3.0", "sha256": REFERENCE_SHA256,
        "segment": f"{args.start}s + {args.seconds}s, 1920x816 (letterbox removed)"}, "scenarios": {}}
    for name, sc in SCENARIOS.items():
        lr = clips[name]
        rows: dict[str, Any] = {}
        print(f"== {name}: {sc['size'][0]}x{sc['size'][1]} -> {HR[0]}x{HR[1]}")
        for flags in ("bicubic", "lanczos"):
            rows[f"ffmpeg {flags}"] = score(tools, lr, hr, size=HR, flags=flags, lpips=scorer)
            print(f"  ffmpeg {flags}: {rows[f'ffmpeg {flags}']}")
        baseline_diffs = frame_differences(iter_luma(tools, lr, size=HR, scale_flags="bicubic"))
        outputs = {}
        for model in sc["models"]:
            out = work / f"out-{name}-{model}.mkv"
            report = upscale(settings, model, lr, out)
            row = score(tools, out, hr, size=None, flags="bicubic", lpips=scorer)
            if model != "lanczos":
                row.update(seams(tools, out, report, hr, baseline_diffs))
            row["fps"] = report["throughput"]["pipeline_fps"]
            row["backend"] = (report["plan"].get("inference") or {}).get("backend", "-")
            rows[f"NeuralFrame {model}"] = row
            outputs[model] = out
            print(f"  {model}: {row}")
        crops = gallery(tools, hr, lr, {m: o for m, o in outputs.items() if m != "lanczos"}, args.gallery, name,
                        frame_s=0.5, box=(840, 120, 1400, 435))
        results["scenarios"][name] = {"input": f"{sc['size'][0]}x{sc['size'][1]}", "crf": sc["crf"], "rows": rows,
                                      "gallery": crops}
    results["tile_seam"] = tile_seam_test(settings, tools, clips["x4"], work)
    print("tile seam:", results["tile_seam"])

    stamp = datetime.now(UTC).strftime("%Y%m%d")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"quality-{stamp}.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    lines = [f"# Quality evaluation - {results['hardware'].get('gpu', results['hardware']['cpu'])}", "",
             f"Reference: {results['reference']['source']}; {results['reference']['segment']}.",
             "Higher is better for PSNR / SSIM / VMAF; lower is better for LPIPS and flicker (temporal noise in "
             "static regions, luma levels). PSNR uses MSE pooled over all frames. Seam ratio: frame-to-frame "
             "change at chunk boundaries / elsewhere, divided by the same ratio of the chunk-free bicubic "
             "baseline at those frames (~1.0 = invisible; the spec flags > 1.2).", ""]
    for name, sc in results["scenarios"].items():
        lines += [f"## {name}: {sc['input']} -> {HR[0]}x{HR[1]}" + (f" (input x264 CRF {sc['crf']})" if sc["crf"]
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
    ts = results["tile_seam"]
    lines += ["## Tile seams", "", f"{ts['model']} with {ts['tile']} px tiles ({ts['overlap']} px overlap) vs whole "
              f"frames: worst frame {ts['psnr_tiled_vs_whole_db']} dB PSNR over {ts['frames']} frames "
              "(spec threshold: > 45 dB).", ""]
    (args.out / f"quality-{stamp}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
