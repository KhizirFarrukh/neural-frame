from __future__ import annotations

from fractions import Fraction

from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.io.probe import detect_interlacing, probe, scan_frames
from tests.conftest import ClipFactory


def test_basic_clip(tools: FFmpegTools, clip: ClipFactory) -> None:
    path = clip("basic", subtitles=True, chapters=True)
    media = probe(path, tools)
    v = media.require_video()
    assert (v.width, v.height, v.pix_fmt, v.bit_depth) == (160, 90, "yuv420p", 8)
    assert v.fps == Fraction(30000, 1001)
    assert v.rotation == 0 and v.sar == 1
    assert [a.codec for a in media.audio] == ["aac"]
    assert media.audio[0].channels == 2
    assert [s.codec for s in media.subtitles] == ["mov_text"]
    assert [c.title for c in media.chapters] == ["Part one", "Part two"]
    timeline = scan_frames(path, tools, v.index)
    assert timeline.frame_count == 60
    assert not timeline.is_vfr


def test_vfr_gaps_are_detected(tools: FFmpegTools, clip: ClipFactory) -> None:
    path = clip("vfr", drop_frames=(10, 11, 30), audio=False)
    timeline = scan_frames(path, tools, probe(path, tools).require_video().index)
    assert timeline.frame_count == 57
    assert timeline.is_vfr


def test_rotation_sar_and_odd_dims(tools: FFmpegTools, clip: ClipFactory) -> None:
    rotated = probe(clip("rot", rotation=90, audio=False), tools).require_video()
    assert rotated.rotation == 90
    assert rotated.display_size == (90, 160)

    dvd = probe(clip("sar", width=720, height=480, sar="32:27", duration=0.5, audio=False), tools).require_video()
    assert dvd.sar == Fraction(32, 27)
    assert dvd.display_aspect == Fraction(16, 9)

    odd = probe(clip("odd", width=161, height=91, codec="ffv1", container="mkv", audio=False), tools)
    assert (odd.require_video().width, odd.require_video().height) == (161, 91)


def test_ten_bit_hdr_and_full_range(tools: FFmpegTools, clip: ClipFactory) -> None:
    hdr = probe(
        clip("hdr", codec="libx265", pix_fmt="yuv420p10le", hdr=True, container="mkv", duration=0.5, audio=False),
        tools,
    ).require_video()
    assert hdr.bit_depth == 10
    assert hdr.hdr_format == "hdr10"
    assert hdr.color.primaries == "bt2020"

    full = probe(clip("full", full_range=True, duration=0.5, audio=False), tools).require_video()
    assert full.color.range == "pc"


def test_interlace_tagging_and_idet(tools: FFmpegTools, clip: ClipFactory) -> None:
    path = clip("interlaced", interlaced=True, fps="25", width=320, height=180, audio=False)
    v = probe(path, tools).require_video()
    assert v.is_interlaced_tagged
    assert detect_interlacing(path, tools).is_interlaced
