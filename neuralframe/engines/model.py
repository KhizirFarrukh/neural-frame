"""Engines inside the job pipeline.

:class:`GenerativeModel` has the interface of :class:`neuralframe.inference.temporal.TemporalModel`, so the
temporal stage drives it: for every chunk of frames (with overlap, never across a scene cut) it writes the
frames out, runs the engine in its environment, and reads the results back. The stage then resizes them to
the exact output size, converts colours, cross-fades chunk boundaries, and the job encodes and muxes as for
any other model. :class:`FaceRestorer` does the same for face restoration on finished frames.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from neuralframe.engines.adapters import EngineCall, build, collect, progress_from
from neuralframe.engines.env import EngineEnv
from neuralframe.engines.process import Cancellable, run
from neuralframe.errors import ErrorCode, NeuralFrameError
from neuralframe.io.ffmpeg import FFmpegTools, popen_kwargs
from neuralframe.log import get_logger
from neuralframe.models.registry import ModelSpec

log = get_logger(component="engines")

RGB8 = npt.NDArray[np.uint8]
MessageCallback = Callable[[str], None]


def write_frames(frames: list[RGB8], folder: Path) -> list[str]:
    """PNG frames 000000.png, 000001.png, ... (lossless, what every engine reads)."""
    from PIL import Image

    folder.mkdir(parents=True, exist_ok=True)
    names = []
    for i, frame in enumerate(frames):
        name = f"{i:06d}.png"
        Image.fromarray(frame).save(folder / name, compress_level=1)
        names.append(name)
    return names


def write_video(frames: list[RGB8], path: Path, tools: FFmpegTools, fps: float) -> None:
    """A lossless H.264 4:4:4 video (for engines that read a video file)."""
    h, w = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [tools.ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "pipe:0", "-c:v", "libx264", "-qp", "0", "-preset",
            "ultrafast", "-pix_fmt", "yuv444p", str(path)]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE, **popen_kwargs())
    assert proc.stdin is not None
    try:
        for frame in frames:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        proc.stdin.close()
    _, err = proc.communicate(timeout=600)
    if proc.returncode != 0:
        raise NeuralFrameError(ErrorCode.ENCODE_FAILED, f"cannot write the engine's input video: {err.decode()[-500:]}")


def read_frames(paths: list[Path]) -> list[RGB8]:
    from PIL import Image

    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]


class EngineRunner:
    """Runs one engine/model on batches of RGB frames (shared by the generative model and faces)."""

    def __init__(self, spec: ModelSpec, env: EngineEnv, files: dict[str, Path], *, tools: FFmpegTools,
                 work_dir: Path, gpu: int = 0, vram_gb: float | None = None, fps: float = 24.0,
                 cancel: Cancellable | None = None, on_message: MessageCallback | None = None,
                 keep_temp: bool = False) -> None:
        self.spec, self.env, self.files, self.tools = spec, env, files, tools
        self.work_dir = work_dir
        self.gpu, self.vram_gb, self.fps = gpu, vram_gb, fps
        self.cancel, self.on_message, self.keep_temp = cancel, on_message, keep_temp
        self.calls = 0
        self.frames_processed = 0
        self.seconds = 0.0

    def process(self, frames: list[RGB8], out_size: tuple[int, int], *, label: str = "") -> list[RGB8]:
        from neuralframe.engines.catalog import engine

        if not frames:
            return []
        t0 = time.perf_counter()
        self.calls += 1
        call_dir = self.work_dir / f"{self.spec.engine}-{self.calls:05d}"
        shutil.rmtree(call_dir, ignore_errors=True)
        inputs, outputs = call_dir / "in", call_dir / "out"
        outputs.mkdir(parents=True, exist_ok=True)
        h, w = frames[0].shape[:2]
        if engine(self.spec.engine or "").io == "video":
            source = inputs / "input.mp4"
            write_video(frames, source, self.tools, self.fps)
        else:
            write_frames(frames, inputs)
            source = inputs
        call = EngineCall(env=self.env, model=self.spec, files=self.files, input=source, output=outputs,
                          frames=len(frames), in_size=(w, h), out_size=out_size, fps=self.fps, gpu=self.gpu,
                          vram_gb=self.vram_gb, options=dict(self.spec.engine_options))
        command = build(call)
        start = time.monotonic()

        def on_line(line: str) -> None:
            if self.on_message is None:
                return
            pair = progress_from(line)
            if pair is not None:
                done, total = pair
                elapsed = time.monotonic() - start
                eta = f", about {round(elapsed / done * (total - done))} s left" if done else ""
                self.on_message(f"{label}{done}/{total}{eta}")

        if self.on_message is not None:
            self.on_message(f"{label}starting {self.spec.id} on {len(frames)} frame(s)")
        run(command.args, cwd=command.cwd, env=command.env, on_line=on_line, cancel=self.cancel,
            log_path=self.env.root / "runs.log", what=self.spec.id)
        results = collect(command, len(frames))
        if not results:
            raise NeuralFrameError(ErrorCode.INFERENCE_FAILED, f"{self.spec.id} produced no frames",
                                   hint=f"see {self.env.root / 'runs.log'}")
        if len(results) < len(frames):
            log.warning("engine returned fewer frames; repeating the last", model=self.spec.id,
                        got=len(results), expected=len(frames))
            results = results + [results[-1]] * (len(frames) - len(results))
        out = read_frames(results[: len(frames)])
        if not self.keep_temp:
            shutil.rmtree(call_dir, ignore_errors=True)
        self.frames_processed += len(frames)
        self.seconds += time.perf_counter() - t0
        return out


class GenerativeModel:
    """A generative engine as a sequence model: (T, 3, H, W) float frames in, restored frames out."""

    def __init__(self, runner: EngineRunner, *, device: Any, out_size: tuple[int, int]) -> None:
        self.runner = runner
        self.spec = runner.spec
        self.device = device
        self.scale = runner.spec.scale
        self.out_size = out_size
        self.multiple = 1

    @property
    def frames_processed(self) -> int:
        return self.runner.frames_processed

    @property
    def seconds(self) -> float:
        return self.runner.seconds

    def run(self, frames: Any) -> Iterator[tuple[int, Any]]:
        import torch

        rgb = (frames.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        restored = self.runner.process(list(rgb), self.out_size, label=f"{self.spec.id}: ")
        for t, arr in enumerate(restored):
            tensor = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device).permute(2, 0, 1).float() / 255.0
            yield t, tensor

    def close(self) -> None:
        return None


class FaceRestorer:
    """Face restoration on finished frames (same size in and out)."""

    def __init__(self, runner: EngineRunner) -> None:
        self.runner = runner

    def restore(self, frames: list[RGB8]) -> list[RGB8]:
        h, w = frames[0].shape[:2]
        restored = self.runner.process(frames, (w, h), label="faces: ")
        return [r if r.shape[:2] == (h, w) else _resize(r, w, h) for r in restored]


def _resize(frame: RGB8, w: int, h: int) -> RGB8:
    from PIL import Image

    return np.asarray(Image.fromarray(frame).resize((w, h), Image.Resampling.LANCZOS), dtype=np.uint8)


def model_files(library: Any, spec: ModelSpec) -> dict[str, Path]:
    """Paths of a model's files (the main one may be a linked file elsewhere)."""
    files: dict[str, Path] = {}
    if spec.weights is not None:
        _state, main = library.locate(spec, quick=True)
        files[library.weights(spec).filename] = main or library.target_path(spec)
    for w in spec.extra_weights:
        files[w.filename] = library.folder(spec) / w.filename
    return files  # empty for engines that manage their own weights
