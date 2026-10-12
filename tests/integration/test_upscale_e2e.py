"""End-to-end jobs on synthetic fixtures (classical model, CPU ops) — runs everywhere, including CI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neuralframe.config import Settings
from neuralframe.core.job import JobRunner, JobSpec, ProgressEvent
from neuralframe.core.planning import TargetSpec
from neuralframe.errors import JobCancelled
from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.io.hwcaps import Capabilities
from neuralframe.io.probe import probe
from tests.conftest import ClipFactory


def run_job(settings: Settings, caps: Capabilities, **fields: Any) -> dict[str, Any]:
    fields.setdefault("model", "lanczos")
    fields.setdefault("overwrite", True)
    runner = JobRunner(settings, capabilities=caps, progress=fields.pop("progress", None))
    result = runner.run(JobSpec(**fields))
    assert result.report_path.is_file()
    return result.report


def test_reference_like_clip_mp4_to_mkv(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path, ffhelpers: Any
) -> None:
    src = clip("e2e-ref", width=214, height=120, duration=2.0, subtitles=True, chapters=True, scene_cut_at=1.0)
    out = tmp_path / "out.mkv"
    report = run_job(settings, caps, input=src, output=out, target=TargetSpec.for_height(360))

    media = probe(out, tools)
    v = media.require_video()
    assert (v.width, v.height) == (640, 360)  # 214x120 ≈ 16:9 → snapped, exact 640x360
    # SD output keeps the (assumed) BT.601 source description instead of converting.
    assert (v.color.matrix, v.color.primaries, v.color.range) == ("smpte170m", "smpte170m", "tv")
    assert report["sync"]["ok"], report["sync"]
    assert report["output"]["frames"] == 60

    # Frame timestamps survive (Matroska stores ms).
    src_t, out_t = ffhelpers.frame_times(src), ffhelpers.frame_times(out)
    assert len(src_t) == len(out_t) == 60
    assert max(abs(a - b) for a, b in zip(src_t, out_t, strict=True)) <= 0.001

    # Audio is a bit-exact stream copy; subtitles converted mov_text → srt with text intact.
    assert ffhelpers.packet_md5s(src, "a:0") == ffhelpers.packet_md5s(out, "a:0")
    assert [s.codec for s in media.subtitles] == ["subrip"]
    srt = ffhelpers.text(["-i", str(out), "-map", "0:s:0", "-f", "srt", "-"])
    assert "First line" in srt and "Second line" in srt
    assert [c.title for c in media.chapters] == ["Part one", "Part two"]

    stage_names = [s["name"] for s in report["throughput"]["stages"]]
    if report["plan"]["mode"] == "device" and "upload" in stage_names:  # CUDA-stream overlap
        assert stage_names == ["decode", "upload", "inference", "postprocess", "encode"]
    elif report["plan"]["mode"] == "device":  # colour conversion + resize run inside the GPU stage
        assert stage_names == ["decode", "inference", "encode"]
        assert [s["name"] for s in report["throughput"]["device_substages"]] == [
            "preprocess", "inference", "postprocess"]
    else:
        assert stage_names == ["decode", "preprocess", "inference", "postprocess", "encode"]
    assert json.loads(out.with_name("out.report.json").read_text())["job_id"] == report["job_id"]


def test_sd_to_hd_converts_to_bt709(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path
) -> None:
    src = clip("e2e-hd", width=320, height=180, duration=0.5)
    out = tmp_path / "hd.mkv"
    report = run_job(settings, caps, input=src, output=out, target=TargetSpec.for_height(720))
    v = probe(out, tools).require_video()
    assert (v.width, v.height) == (1280, 720)
    assert (v.color.matrix, v.color.primaries, v.color.transfer, v.color.range) == ("bt709", "bt709", "bt709", "tv")
    assert report["plan"]["color"]["convert_gamut"] is True


def test_mp4_output_copies_subtitles_bit_exact(
    caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path, ffhelpers: Any
) -> None:
    src = clip("e2e-mp4", duration=1.0, subtitles=True)
    out = tmp_path / "out.mp4"
    report = run_job(settings, caps, input=src, output=out, target=TargetSpec.factor(2))
    assert report["sync"]["ok"]
    assert ffhelpers.packet_md5s(src, "a:0") == ffhelpers.packet_md5s(out, "a:0")
    assert ffhelpers.packet_md5s(src, "s:0") == ffhelpers.packet_md5s(out, "s:0")


def test_vfr_timestamps_are_preserved(
    caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path, ffhelpers: Any
) -> None:
    src = clip("e2e-vfr", duration=1.5, drop_frames=(7, 8, 20))
    out = tmp_path / "out.mp4"
    run_job(settings, caps, input=src, output=out, target=TargetSpec.factor(2))
    src_t, out_t = ffhelpers.frame_times(src), ffhelpers.frame_times(out)
    assert len(src_t) == len(out_t)
    assert max(abs(a - b) for a, b in zip(src_t, out_t, strict=True)) < 1e-4


@pytest.mark.parametrize(
    ("name", "fields", "spec", "expect"),
    [
        ("odd", {"width": 161, "height": 91, "codec": "ffv1", "container": "mkv"}, {"target": TargetSpec.factor(2)},
         {"size": (322, 182), "pix_fmt": "yuv420p"}),
        ("tenbit", {"duration": 1.0}, {"target": TargetSpec.factor(2), "ten_bit": True},
         {"size": (320, 180), "pix_fmt": "yuv420p10le"}),
        ("rotated", {"rotation": 90, "duration": 1.0}, {"target": TargetSpec.factor(2)},
         {"size": (180, 320), "rotation": 0}),
        ("dvd", {"width": 360, "height": 240, "sar": "32:27", "duration": 0.5}, {"target": TargetSpec.for_height(480)},
         {"size": (854, 480), "sar": "1"}),
    ],
)
def test_format_handling(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path,
    name: str, fields: dict[str, Any], spec: dict[str, Any], expect: dict[str, Any],
) -> None:
    src = clip(f"e2e-{name}", **fields)
    out = tmp_path / f"{name}.mkv"
    report = run_job(settings, caps, input=src, output=out, **spec)
    assert report["sync"]["ok"], report["sync"]
    v = probe(out, tools).require_video()
    assert (v.width, v.height) == expect["size"]
    if "pix_fmt" in expect:
        assert v.pix_fmt == expect["pix_fmt"]
    if "rotation" in expect:
        assert v.rotation == expect["rotation"]
    if "sar" in expect:
        assert v.sar == 1


def test_interlaced_input_is_deinterlaced(
    caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path
) -> None:
    src = clip("e2e-interlaced", interlaced=True, fps="25", width=320, height=180, duration=1.0)
    out = tmp_path / "deint.mkv"
    report = run_job(settings, caps, input=src, output=out, target=TargetSpec.factor(2))
    assert any(f.startswith("bwdif") for f in report["plan"]["decode"]["filters"])
    assert report["output"]["frames"] == 25


def test_hdr_is_passed_through(
    tools: FFmpegTools, caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path
) -> None:
    src = clip("e2e-hdr", codec="libx265", pix_fmt="yuv420p10le", hdr=True, container="mkv", duration=0.5)
    out = tmp_path / "hdr.mkv"
    report = run_job(settings, caps, input=src, output=out, target=TargetSpec.factor(2))
    v = probe(out, tools).require_video()
    assert v.bit_depth == 10
    assert (v.color.transfer, v.color.primaries) == ("smpte2084", "bt2020")
    assert any("HDR" in w for w in report["warnings"])


def test_video_without_audio(caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path) -> None:
    src = clip("e2e-silent", duration=0.5, audio=False)
    report = run_job(settings, caps, input=src, output=tmp_path / "silent.mp4", target=TargetSpec.factor(2))
    assert report["sync"]["ok"]


def _kill_and_resume(
    settings: Settings, caps: Capabilities, clip: ClipFactory, tmp_path: Path, ffhelpers: Any, **encode: Any
) -> None:
    src = clip("e2e-resume", duration=2.0)
    common: dict[str, Any] = {"input": src, "target": TargetSpec.factor(2), "chunk_frames": 16, **encode}

    reference = tmp_path / "uninterrupted.mkv"
    run_job(settings, caps, output=reference, **common)

    resumed_out = tmp_path / "resumed.mkv"
    runner = JobRunner(settings, capabilities=caps)

    def stop_after_two_chunks(event: ProgressEvent) -> None:
        if event.state == "running" and event.frames_done >= 32:
            runner.cancel.cancel("simulated kill")

    runner.progress = stop_after_two_chunks
    with pytest.raises(JobCancelled):
        runner.run(JobSpec(output=resumed_out, model="lanczos", overwrite=True, **common))
    assert not resumed_out.exists()

    report = run_job(settings, caps, output=resumed_out, resume=True, **common)
    assert report["throughput"]["frames_resumed"] >= 32
    assert report["throughput"]["frames_processed_this_run"] < 60
    assert ffhelpers.frame_md5s(resumed_out) == ffhelpers.frame_md5s(reference)


def test_kill_and_resume_matches_uninterrupted_run(
    caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path, ffhelpers: Any
) -> None:
    # Lossless x264 (CRF 0): decoded frames are fixed by the pipeline alone. x264's lossy decisions were
    # measured to vary slightly under heavy CPU contention, which would test the encoder, not us.
    _kill_and_resume(settings, caps, clip, tmp_path, ffhelpers, encoder="libx264", codec="h264", quality=0)


@pytest.mark.gpu
def test_kill_and_resume_with_nvenc_is_bit_identical(
    caps: Capabilities, settings: Settings, clip: ClipFactory, tmp_path: Path, ffhelpers: Any
) -> None:
    if caps.encoder("hevc_nvenc") is None or not caps.encoder("hevc_nvenc").functional:  # type: ignore[union-attr]
        pytest.skip("NVENC not available")
    _kill_and_resume(settings, caps, clip, tmp_path, ffhelpers, encoder="hevc_nvenc", codec="hevc")
