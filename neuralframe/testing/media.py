"""Synthetic test media generated with FFmpeg (used by tests, benchmarks and the reference case).

All sources are lavfi generators, so fixtures are created at test time and never committed. Recipes avoid
``drawtext`` (needs fontconfig) and ``-shortest`` (can stall with subtitle inputs).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from neuralframe.io.ffmpeg import FFmpegTools, popen_kwargs, tail


@dataclass(frozen=True)
class ClipSpec:
    width: int = 160
    height: int = 90
    fps: str = "30000/1001"
    duration: float = 2.0
    container: str = "mp4"
    codec: str = "libx264"
    pix_fmt: str = "yuv420p"
    crf: int = 18
    audio: bool = True
    audio_codec: str | None = None
    audio_channels: int = 2
    subtitles: bool = False
    chapters: bool = False
    scene_cut_at: float | None = None
    interlaced: bool = False
    drop_frames: tuple[int, ...] = ()
    """Frame numbers removed with ``select`` while keeping timestamps — produces VFR gaps."""
    duplicate_frames: tuple[int, ...] = ()
    """Frame numbers replaced by a copy of the previous frame (same timestamps) — a frozen-frame drop."""
    rotation: int | None = None
    sar: str | None = None
    hdr: bool = False
    full_range: bool = False
    color_tags: dict[str, str] = field(default_factory=dict)
    gop: int | None = None

    @property
    def fps_float(self) -> float:
        num, _, den = self.fps.partition("/")
        return float(num) / float(den or 1)

    @property
    def frame_count(self) -> int:
        return round(self.duration * self.fps_float) - len(self.drop_frames)


def _video_graph(spec: ClipSpec) -> str:
    rate = spec.fps
    size = f"{spec.width}x{spec.height}"
    gen_size = f"{spec.width + spec.width % 2}x{spec.height + spec.height % 2}"
    motion = (
        f"color=c=red:size={max(8, spec.width // 8)}x{max(8, spec.height // 8)}:rate={rate}"
        f":duration={{d}}[box{{i}}];[bars{{i}}][box{{i}}]overlay=x='mod(t*{spec.width // 2}\\,{spec.width})'"
        f":y='{spec.height // 3}+{spec.height // 6}*sin(t*3)'"
    )
    if spec.interlaced:
        # Render at double rate and weave pairs of frames into fields (top field first).
        num, _, den = spec.fps.partition("/")
        double = f"{int(num) * 2}/{den or 1}"
        graph = (
            f"testsrc2=size={gen_size}:rate={double}:duration={spec.duration},"
            "tinterlace=mode=interleave_top,fieldorder=tff"
        )
    elif spec.scene_cut_at is not None:
        first = spec.scene_cut_at
        second = spec.duration - first
        graph = (
            f"testsrc2=size={gen_size}:rate={rate}:duration={first}[a];"
            f"smptehdbars=size={gen_size}:rate={rate}:duration={second}[bars0];"
            + motion.format(d=second, i=0)
            + "[b];[a][b]concat=n=2:v=1:a=0"
        )
    else:
        graph = f"testsrc2=size={gen_size}:rate={rate}:duration={spec.duration}"

    post: list[str] = []
    if gen_size != size:
        post.append(f"scale={spec.width}:{spec.height}")
    if spec.duplicate_frames:
        # Replace listed frames with their predecessor: freeze while timestamps keep advancing.
        cond = "+".join(f"eq(n\\,{n})" for n in spec.duplicate_frames)
        post.append(f"select='not({cond})',fps={spec.fps}:round=down")
    if spec.drop_frames:
        cond = "+".join(f"eq(n\\,{n})" for n in spec.drop_frames)
        post.append(f"select='not({cond})'")
    if spec.sar:
        post.append(f"setsar={spec.sar.replace(':', '/')}")
    params = dict(spec.color_tags)
    if spec.hdr:
        params.update(color_primaries="bt2020", color_trc="smpte2084", colorspace="bt2020nc", range="tv")
    if spec.full_range:
        post.append("scale=out_range=full")
        params["range"] = "pc"
    if params:
        post.append("setparams=" + ":".join(f"{k}={v}" for k, v in params.items()))
    post.append(f"format={spec.pix_fmt}")
    return f"{graph},{','.join(post)}[v]"


def make_clip(tools: FFmpegTools, path: Path, spec: ClipSpec) -> Path:
    """Create ``path`` according to ``spec`` (idempotent if the file already exists)."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    work = path.with_name(path.stem + ".tmp" + path.suffix)
    inputs: list[str] = []
    maps = ["-map", "[v]"]
    next_input = 0
    if spec.audio:
        inputs += [
            "-f", "lavfi", "-i",
            f"sine=frequency=440:sample_rate=48000:duration={spec.duration},"
            f"aformat=channel_layouts={'stereo' if spec.audio_channels == 2 else 'mono'}",
        ]
        maps += ["-map", f"{next_input}:a"]
        next_input += 1
    if spec.subtitles:
        srt = path.with_suffix(".srt")
        half = spec.duration / 2
        srt.write_text(
            "1\n00:00:00,200 --> " + _ts(half - 0.1) + "\nFirst line\n\n"
            "2\n" + _ts(half + 0.1) + " --> " + _ts(spec.duration - 0.1) + "\nSecond line\n",
            encoding="utf-8",
        )
        inputs += ["-i", str(srt)]
        maps += ["-map", f"{next_input}:s"]
        next_input += 1
    if spec.chapters:
        meta = path.with_suffix(".chapters.txt")
        ms = int(spec.duration * 1000)
        meta.write_text(
            ";FFMETADATA1\ntitle=NeuralFrame fixture\n"
            f"[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND={ms // 2}\ntitle=Part one\n"
            f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={ms // 2}\nEND={ms}\ntitle=Part two\n",
            encoding="utf-8",
        )
        inputs += ["-i", str(meta)]
        maps += ["-map_metadata", str(next_input), "-map_chapters", str(next_input)]
        next_input += 1

    codec_args = ["-c:v", spec.codec]
    if spec.codec == "libx264":
        codec_args += ["-crf", str(spec.crf), "-preset", "veryfast"]
        if spec.interlaced:
            codec_args += ["-flags", "+ilme+ildct", "-x264-params", "tff=1"]
    elif spec.codec == "libx265":
        x265 = "log-level=error"
        if spec.hdr:
            x265 += (
                ":hdr10=1:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)"
                "L(10000000,1):max-cll=1000,400"
            )
        codec_args += ["-crf", str(spec.crf), "-preset", "fast", "-x265-params", x265]
    if spec.gop:
        codec_args += ["-g", str(spec.gop)]
    if spec.drop_frames:
        codec_args += ["-fps_mode", "passthrough"]
    if spec.audio:
        audio_codec = spec.audio_codec or ("aac" if spec.container in ("mp4", "mov") else "flac")
        codec_args += ["-c:a", audio_codec]
        if audio_codec == "aac":
            codec_args += ["-b:a", "128k"]
    if spec.subtitles:
        codec_args += ["-c:s", "mov_text" if spec.container in ("mp4", "mov") else "srt"]

    cmd = [
        tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        *inputs, "-filter_complex", _video_graph(spec), *maps, *codec_args,
        "-f", "matroska" if spec.container == "mkv" else spec.container, str(work),
    ]
    _run(cmd)
    if spec.rotation:
        rotated = path.with_name(path.stem + ".rot" + path.suffix)
        _run([
            tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-display_rotation", str(spec.rotation), "-i", str(work), "-map", "0", "-c", "copy", str(rotated),
        ])
        work.unlink()
        work = rotated
    work.replace(path)
    return path


def _ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, timeout=300, **popen_kwargs())
    if proc.returncode != 0:
        raise RuntimeError(f"fixture generation failed: {tail(proc.stderr)}\n{' '.join(cmd)}")


def reference_clip(tools: FFmpegTools, path: Path) -> Path:
    """The spec's reference case: 10 s, 854x480, 29.97 fps H.264, stereo AAC, SRT subs, scene cut at 5.0 s."""
    return make_clip(
        tools,
        path,
        ClipSpec(
            width=854, height=480, fps="30000/1001", duration=10.0, container="mp4", subtitles=True,
            scene_cut_at=5.0, crf=18,
        ),
    )
