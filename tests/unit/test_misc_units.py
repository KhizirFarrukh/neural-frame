from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neuralframe.config import load_settings
from neuralframe.core.runtime import CancelToken, Stage, StageQueue, StageRunner
from neuralframe.errors import JobCancelled, NeuralFrameError
from neuralframe.io.media import AudioStream, MediaInfo, PixelFormat, SubtitleStream, VideoStream
from neuralframe.io.muxer import plan_mux
from neuralframe.io.rawvideo import RawVideoFormat

# ---- raw video layouts ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "pix_fmt", "chroma"),
    [
        (RawVideoFormat(161, 91, 8, 1, 1), "yuv420p", (46, 81)),
        (RawVideoFormat(161, 91, 10, 1, 0), "yuv422p10le", (91, 81)),
        (RawVideoFormat(160, 90, 12, 0, 0), "yuv444p12le", (90, 160)),
        (RawVideoFormat(64, 32, 16, 0, 0, rgb=True), "gbrp16le", (32, 64)),
    ],
)
def test_raw_formats(fmt: RawVideoFormat, pix_fmt: str, chroma: tuple[int, int]) -> None:
    assert fmt.pix_fmt == pix_fmt
    assert fmt.plane_shapes[1] == chroma
    buf = np.arange(fmt.frame_samples, dtype=fmt.dtype)
    planes = fmt.split(buf)
    assert np.array_equal(fmt.join(planes), buf)


def test_raw_format_from_pixel_format() -> None:
    p010 = PixelFormat(name="p010le", bit_depth=10, log2_chroma_w=1, log2_chroma_h=1)
    assert RawVideoFormat.for_source(p010, 1920, 1080).pix_fmt == "yuv420p10le"
    gray = PixelFormat(name="gray", bit_depth=8, nb_components=1)
    assert RawVideoFormat.for_source(gray, 64, 64).pix_fmt == "yuv420p"
    rgb = PixelFormat(name="rgb24", bit_depth=8, is_rgb=True, is_planar=False)
    assert RawVideoFormat.for_source(rgb, 64, 64).pix_fmt == "gbrp"


# ---- mux planning -----------------------------------------------------------------------------------


def media(audio: list[str], subs: list[str]) -> MediaInfo:
    v = VideoStream(
        index=0, codec="h264", width=64, height=64, pix_fmt="yuv420p",
        pixel_format=PixelFormat(name="yuv420p", bit_depth=8, log2_chroma_w=1, log2_chroma_h=1),
    )
    return MediaInfo(
        path=Path("in.mp4"), format_name="mov,mp4", video=v,
        audio=[AudioStream(index=1 + i, codec=c) for i, c in enumerate(audio)],
        subtitles=[
            SubtitleStream(index=1 + len(audio) + i, codec=c, is_text=c in ("mov_text", "subrip", "ass"))
            for i, c in enumerate(subs)
        ],
    )


def test_mux_plan_converts_text_subtitles_per_container() -> None:
    to_mkv = plan_mux(media(["aac"], ["mov_text"]), Path("out.mkv"))
    assert [(s.kind, s.action, s.target_codec) for s in to_mkv.streams] == [
        ("audio", "copy", None), ("subtitle", "convert", "srt"),
    ]
    to_mp4 = plan_mux(media(["aac"], ["subrip", "mov_text"]), Path("out.mp4"))
    assert [s.action for s in to_mp4.streams] == ["copy", "convert", "copy"]


def test_mux_plan_drops_bitmap_subtitles_in_mp4() -> None:
    plan = plan_mux(media([], ["hdmv_pgs_subtitle"]), Path("out.mp4"))
    assert plan.streams[0].action == "drop" and plan.warnings


def test_mux_plan_refuses_lossy_audio_handling() -> None:
    with pytest.raises(NeuralFrameError, match="cannot be stream-copied"):
        plan_mux(media(["pcm_s16le"], []), Path("out.mp4"))
    assert plan_mux(media(["pcm_s16le"], []), Path("out.mkv")).streams[0].action == "copy"


def test_unknown_output_extension() -> None:
    with pytest.raises(NeuralFrameError, match="unsupported output extension"):
        plan_mux(media([], []), Path("out.xyz"))


# ---- runtime ----------------------------------------------------------------------------------------


def test_queue_end_of_stream_and_cancel() -> None:
    cancel = CancelToken()
    q: StageQueue[int] = StageQueue("q", 2, cancel)
    q.put(1)
    q.close()
    assert q.get() == 1
    assert q.get() is None
    assert q.get() is None  # end-of-stream is sticky for every consumer

    blocked: StageQueue[int] = StageQueue("b", 1, cancel)
    blocked.put(1)
    threading.Timer(0.2, cancel.cancel).start()
    with pytest.raises(JobCancelled):
        blocked.put(2)


class _Producer(Stage):
    def __init__(self, cancel: CancelToken, out: StageQueue[int], fail_at: int | None = None) -> None:
        super().__init__("producer", cancel)
        self.out, self.fail_at = out, fail_at

    def run(self) -> None:
        for i in range(50):
            if i == self.fail_at:
                raise NeuralFrameError("DECODE_FAILED", "boom")  # type: ignore[arg-type]
            self.out.put(i)
            self.metrics.frames += 1
        self.out.close()


class _Consumer(Stage):
    def __init__(self, cancel: CancelToken, inq: StageQueue[int]) -> None:
        super().__init__("consumer", cancel)
        self.inq, self.seen = inq, []  # type: list[int]

    def run(self) -> None:
        while (item := self.inq.get()) is not None:
            time.sleep(0.001)
            self.seen.append(item)


def test_stage_runner_passes_all_items() -> None:
    cancel = CancelToken()
    q: StageQueue[int] = StageQueue("consumer", 4, cancel)
    consumer = _Consumer(cancel, q)
    StageRunner([_Producer(cancel, q), consumer], cancel).run()
    assert consumer.seen == list(range(50))


def test_stage_runner_propagates_first_failure_and_stops_others() -> None:
    cancel = CancelToken()
    q: StageQueue[int] = StageQueue("consumer", 4, cancel)
    consumer = _Consumer(cancel, q)
    with pytest.raises(NeuralFrameError, match="boom"):
        StageRunner([_Producer(cancel, q, fail_at=10), consumer], cancel).run()
    assert cancel.is_set()
    assert len(consumer.seen) <= 10


# ---- config -----------------------------------------------------------------------------------------


def test_settings_yaml_env_and_preset_merge(tmp_path: Path, monkeypatch: Any) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "encode:\n  codec: av1\n  quality: 30\n"
        "presets:\n  quality:\n    speed: medium\n  tiny:\n    model: lanczos\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NEURALFRAME_ENCODE__QUALITY", "25")
    s = load_settings(cfg)
    assert s.encode.codec == "av1"
    assert s.encode.quality == 25  # env beats file
    assert s.preset("quality").speed == "medium"
    assert s.preset("quality").model == "realesrgan-x4plus"  # untouched built-in fields survive
    assert s.preset("tiny").model == "lanczos"
    with pytest.raises(NeuralFrameError):
        s.preset("nope")


def test_settings_reject_unknown_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text("encode:\n  codek: av1\n", encoding="utf-8")
    with pytest.raises(NeuralFrameError):
        load_settings(cfg)


def test_example_config_is_valid() -> None:
    example = Path(__file__).resolve().parents[2] / "config.example.yaml"
    s = load_settings(example)
    assert s.api.host == "127.0.0.1"
    assert s.preset("balanced").speed == "medium"


def test_multi_producer_queue_closes_after_last_producer() -> None:
    cancel = CancelToken()
    q: StageQueue[int] = StageQueue("q", 10, cancel, producers=2)
    q.put(1)
    q.close()
    q.put(2)
    assert q.get() == 1 and q.get() == 2
    q.close()
    assert q.get() is None


def test_in_flight_limiter_blocks_and_cancels() -> None:
    from neuralframe.core.runtime import InFlightLimiter

    cancel = CancelToken()
    limiter = InFlightLimiter(2, cancel)
    limiter.acquire()
    limiter.acquire()
    threading.Timer(0.2, limiter.release).start()
    limiter.acquire()  # unblocked by the release
    threading.Timer(0.2, cancel.cancel).start()
    with pytest.raises(JobCancelled):
        limiter.acquire()
