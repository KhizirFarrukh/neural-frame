"""Decoder → encoder round trips through the NUT pipes must be sample- and timestamp-exact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neuralframe.colorspace import ChromaLocation, ColorPlan, ColorRange, Matrix, Primaries, Transfer
from neuralframe.io.decoder import DecodePlan, FrameDecoder
from neuralframe.io.encoder import EncoderPlan, SegmentEncoder
from neuralframe.io.ffmpeg import FFmpegTools
from neuralframe.io.probe import probe
from neuralframe.io.rawvideo import RawVideoFormat
from tests.conftest import ClipFactory


def _color(full: bool) -> ColorPlan:
    rng = ColorRange.FULL if full else ColorRange.LIMITED
    return ColorPlan(
        Matrix.BT709, rng, Primaries.BT709, Transfer.BT709, ChromaLocation.LEFT,
        Matrix.BT709, rng, Primaries.BT709, Transfer.BT709,
    )


def _decode_all(tools: FFmpegTools, path: Path) -> tuple[list[np.ndarray], list[int], FrameDecoder]:
    media = probe(path, tools)
    v = media.require_video()
    fmt = RawVideoFormat.for_source(v.pixel_format, v.width, v.height)
    frames, pts = [], []
    with FrameDecoder(tools, path, v.index, DecodePlan(fmt)) as dec:
        for f in dec:
            frames.append(f.data)
            pts.append(f.pts)
    return frames, pts, dec


@pytest.mark.parametrize(
    ("name", "fields"),
    [
        ("rt-8bit", {"audio": False}),
        ("rt-10bit", {"codec": "libx265", "pix_fmt": "yuv420p10le", "container": "mkv", "audio": False}),
        ("rt-full", {"full_range": True, "audio": False}),
        ("rt-odd", {"width": 161, "height": 91, "codec": "ffv1", "container": "mkv", "audio": False}),
        ("rt-vfr", {"drop_frames": (5, 6, 20), "audio": False}),
    ],
)
def test_lossless_round_trip(tools: FFmpegTools, clip: ClipFactory, tmp_path: Path, name: str, fields: dict) -> None:
    src = clip(name, duration=1.0, **fields)
    frames, pts, dec = _decode_all(tools, src)
    v = probe(src, tools).require_video()
    fmt = RawVideoFormat.for_source(v.pixel_format, v.width, v.height)
    plan = EncoderPlan(
        name="ffv1", codec="ffv1", family="software", quality=0, speed="medium",
        ten_bit=fmt.bit_depth > 8, color=_color(fields.get("full_range", False)),
    )
    out = tmp_path / f"{name}.mkv"
    with SegmentEncoder(tools, out, fmt, dec.time_base, dec.frame_rate, plan, container="matroska") as enc:
        for i, (data, p) in enumerate(zip(frames, pts, strict=True)):
            enc.write(p, 1, data)
            assert enc.frames_written == i + 1

    again, pts2, dec2 = _decode_all(tools, out)
    assert len(again) == len(frames)
    for a, b in zip(frames, again, strict=True):
        np.testing.assert_array_equal(a, b)
    # Matroska stores milliseconds: compare in seconds with sub-millisecond tolerance.
    t1 = np.array(pts, dtype=np.float64) * float(dec.time_base)
    t2 = np.array(pts2, dtype=np.float64) * float(dec2.time_base)
    np.testing.assert_allclose(t1, t2, atol=6e-4)
