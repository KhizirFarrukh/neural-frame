from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from neuralframe.colorspace import ColorRange, Matrix, Primaries, Transfer
from neuralframe.config import ColorConfig
from neuralframe.core.planning import TargetSpec, plan_color, plan_encoder, plan_geometry, snap_aspect
from neuralframe.errors import NeuralFrameError
from neuralframe.io.hwcaps import Capabilities, CodecCheck
from neuralframe.io.media import ColorInfo, PixelFormat, VideoStream


def video(**overrides: Any) -> VideoStream:
    base: dict[str, Any] = {
        "index": 0,
        "codec": "h264",
        "width": 854,
        "height": 480,
        "pix_fmt": "yuv420p",
        "pixel_format": PixelFormat(name="yuv420p", bit_depth=8, log2_chroma_w=1, log2_chroma_h=1),
        "r_frame_rate": Fraction(30000, 1001),
        "avg_frame_rate": Fraction(30000, 1001),
        "time_base": Fraction(1, 30000),
    }
    base.update(overrides)
    return VideoStream(**base)


# ---- target parsing ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "mode", "w", "h", "scale"),
    [
        ("4k", "box", 3840, 2160, None),
        ("1080p", "box", 1920, 1080, None),
        ("1920x1080", "size", 1920, 1080, None),
        ("x2", "scale", None, None, 2.0),
        ("2.5x", "scale", None, None, 2.5),
        ("h720", "height", None, 720, None),
        ("native", "native", None, None, None),
    ],
)
def test_target_parse(text: str, mode: str, w: int | None, h: int | None, scale: float | None) -> None:
    t = TargetSpec.parse(text)
    assert t.mode == mode
    if mode in ("box", "size"):
        assert (t.width, t.height) == (w, h)
    if mode == "height":
        assert t.height == h
    if mode == "scale":
        assert t.scale == scale


def test_target_parse_rejects_garbage() -> None:
    with pytest.raises(NeuralFrameError):
        TargetSpec.parse("huge")


# ---- geometry ---------------------------------------------------------------------------------------


def test_snap_aspect() -> None:
    assert snap_aspect(Fraction(854, 480)) == Fraction(16, 9)
    assert snap_aspect(Fraction(1000, 333)) == Fraction(1000, 333)


def test_reference_case_geometry_is_exact_1080p() -> None:
    geo, notes = plan_geometry(video(), TargetSpec.for_height(1080), model_scale=4)
    assert (geo.out_width, geo.out_height) == (1920, 1080)
    assert geo.model_passes == 1
    assert (geo.model_out_width, geo.model_out_height) == (3416, 1920)
    assert any("snapped" in n for n in notes)


def test_default_target_is_4k() -> None:
    geo, _ = plan_geometry(video(), TargetSpec.default(), model_scale=4)
    assert (geo.out_width, geo.out_height) == (3840, 2160)


def test_portrait_rotation_fits_4k_box_upright() -> None:
    geo, _ = plan_geometry(video(width=1280, height=720, rotation=90), TargetSpec.default(), model_scale=4)
    assert (geo.decoded_width, geo.decoded_height) == (720, 1280)
    assert (geo.out_width, geo.out_height) == (2160, 3840)


@pytest.mark.parametrize(("sar", "expected"), [(Fraction(32, 27), (3840, 2160)), (Fraction(8, 9), (2880, 2160))])
def test_anamorphic_dvd_becomes_square_pixels(sar: Fraction, expected: tuple[int, int]) -> None:
    geo, _ = plan_geometry(video(width=720, height=480, sar=sar), TargetSpec.default(), model_scale=4)
    assert (geo.out_width, geo.out_height) == expected


def test_cinemascope_fits_width_of_box() -> None:
    geo, _ = plan_geometry(video(width=1920, height=800), TargetSpec.default(), model_scale=2)
    assert geo.out_width == 3840
    assert geo.out_height == 1600


def test_source_larger_than_box_is_not_downscaled() -> None:
    geo, notes = plan_geometry(video(width=4096, height=2160), TargetSpec.default(), model_scale=4)
    assert (geo.out_width, geo.out_height) == (4096, 2160)
    assert geo.model_passes == 0
    assert any("keeping source size" in n for n in notes)


def test_scale_and_exact_size_modes() -> None:
    geo, _ = plan_geometry(video(width=640, height=360), TargetSpec.factor(2), model_scale=4)
    assert (geo.out_width, geo.out_height) == (1280, 720)
    geo, notes = plan_geometry(video(width=640, height=360), TargetSpec.exact(1000, 1000), model_scale=4)
    assert (geo.out_width, geo.out_height) == (1000, 1000)
    assert any("stretch" in n for n in notes)


def test_odd_source_gives_even_output() -> None:
    geo, _ = plan_geometry(video(width=161, height=91), TargetSpec.factor(2), model_scale=4)
    assert geo.out_width % 2 == 0 and geo.out_height % 2 == 0


def test_cascade_for_large_factors() -> None:
    small = video(width=320, height=180)
    geo, _ = plan_geometry(small, TargetSpec.default(), model_scale=4, cascade=True)
    assert geo.model_passes == 2
    geo, notes = plan_geometry(small, TargetSpec.default(), model_scale=4, cascade=False)
    assert geo.model_passes == 1
    assert any("Lanczos" in n for n in notes)


# ---- colour -----------------------------------------------------------------------------------------


def test_untagged_sd_converts_to_bt709_for_hd_output() -> None:
    plan = plan_color(video(), 1080, ColorConfig(), ten_bit=False)
    assert plan.in_matrix is Matrix.BT601
    assert plan.in_primaries is Primaries.SMPTE170M
    assert plan.out_matrix is Matrix.BT709
    assert plan.out_primaries is Primaries.BT709
    assert plan.convert_gamut
    assert plan.ffmpeg_tags()["colorspace"] == "bt709"
    assert plan.out_range is ColorRange.LIMITED


def test_pal_sd_assumes_bt470bg() -> None:
    plan = plan_color(video(width=720, height=576), 1080, ColorConfig(), ten_bit=False)
    assert plan.in_primaries is Primaries.BT470BG


def test_preserve_matrix_keeps_sd_tags() -> None:
    plan = plan_color(video(), 1080, ColorConfig(matrix="preserve"), ten_bit=False)
    assert plan.out_matrix is Matrix.BT601
    assert not plan.convert_gamut
    assert plan.ffmpeg_tags()["colorspace"] == "smpte170m"


def test_hd_source_needs_no_conversion() -> None:
    plan = plan_color(video(width=1920, height=1080), 2160, ColorConfig(), ten_bit=True)
    assert plan.in_matrix is Matrix.BT709 and plan.out_matrix is Matrix.BT709
    assert not plan.convert_gamut
    assert plan.out_bit_depth == 10


def test_full_range_detected_from_yuvj() -> None:
    plan = plan_color(video(pix_fmt="yuvj420p"), 1080, ColorConfig(), ten_bit=False)
    assert plan.in_range is ColorRange.FULL
    assert plan.out_range is ColorRange.LIMITED


def test_hdr_is_passed_through_as_10_bit() -> None:
    hdr = video(
        width=3840, height=2160, pix_fmt="yuv420p10le",
        pixel_format=PixelFormat(name="yuv420p10le", bit_depth=10, log2_chroma_w=1, log2_chroma_h=1),
        color=ColorInfo(matrix="bt2020nc", primaries="bt2020", transfer="smpte2084", range="tv"),
        hdr_format="hdr10",
    )
    plan = plan_color(hdr, 2160, ColorConfig(), ten_bit=False)
    assert plan.hdr
    assert plan.out_bit_depth == 10
    assert plan.out_transfer is Transfer.SMPTE2084
    assert plan.out_primaries is Primaries.BT2020
    assert not plan.convert_gamut


# ---- encoder selection ------------------------------------------------------------------------------


def fake_caps(*checks: CodecCheck) -> Capabilities:
    return Capabilities(
        platform="test", python="3", neuralframe="0", cpu="cpu", cpu_threads=1, ram_total_gb=1,
        ram_available_gb=1, ffmpeg_path="ffmpeg", ffmpeg_version="9", encoders=list(checks),
    )


NVENC = CodecCheck(name="hevc_nvenc", codec="hevc", family="nvenc", listed=True, functional=True, ten_bit=True)
X265 = CodecCheck(name="libx265", codec="hevc", family="software", listed=True, functional=True, ten_bit=True)
H264_NVENC = CodecCheck(name="h264_nvenc", codec="h264", family="nvenc", listed=True, functional=True)
X264 = CodecCheck(name="libx264", codec="h264", family="software", listed=True, functional=True, ten_bit=True)


def encoder_for(caps: Capabilities, codec: str, *, ten_bit: bool = False, w: int = 1920, h: int = 1080,
                encoder: str = "auto") -> Any:
    color = plan_color(video(), h, ColorConfig(), ten_bit=ten_bit)
    return plan_encoder(caps, codec=codec, encoder=encoder, quality=20, speed="medium", ten_bit=ten_bit,
                        color=color, out_width=w, out_height=h)


def test_hardware_encoder_preferred() -> None:
    assert encoder_for(fake_caps(X265, NVENC), "hevc").name == "hevc_nvenc"


def test_ten_bit_falls_through_to_capable_encoder() -> None:
    plan = encoder_for(fake_caps(H264_NVENC, X264), "h264", ten_bit=True)
    assert plan.name == "libx264" and plan.ten_bit
    assert plan.color.out_bit_depth == 10


def test_ten_bit_downgrades_when_impossible() -> None:
    plan = encoder_for(fake_caps(H264_NVENC), "h264", ten_bit=True)
    assert plan.name == "h264_nvenc" and not plan.ten_bit
    assert plan.color.out_bit_depth == 8


def test_h264_nvenc_skipped_beyond_4096() -> None:
    plan = encoder_for(fake_caps(H264_NVENC, X264), "h264", w=7680, h=4320)
    assert plan.name == "libx264"


def test_unavailable_explicit_encoder_is_an_error() -> None:
    with pytest.raises(NeuralFrameError, match="not available"):
        encoder_for(fake_caps(X265), "hevc", encoder="hevc_nvenc")


def test_colour_tags_on_frames_match_encoder_options() -> None:
    plan = encoder_for(fake_caps(NVENC), "hevc")
    chain = plan.filters()
    assert chain[0].startswith("setparams=")
    assert "colorspace=bt709" in chain[0] and "color_primaries=bt709" in chain[0]
    args = plan.codec_args()
    assert "-cq" in args
    # Frame tags and encoder options must agree, or FFmpeg inserts an implicit swscale conversion.
    assert args[args.index("-colorspace") + 1] == "bt709"
    assert args[args.index("-color_range") + 1] == "tv"
