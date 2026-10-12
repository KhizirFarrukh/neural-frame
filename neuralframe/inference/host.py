"""Host-mode neural processor: float RGB frames from the CPU stages in, upscaled float RGB frames out."""

from __future__ import annotations

import time

import numpy as np
import torch

from neuralframe.inference.processor import F32, FrameProcessor
from neuralframe.inference.runner import ModelRunner


class HostModelProcessor(FrameProcessor):
    name = "model"

    def __init__(self, runner: ModelRunner, passes: int, frames_per_call: int) -> None:
        self.runner = runner
        self.passes = passes
        self.scale = runner.scale**passes
        self.batch_size = max(1, frames_per_call)
        self._infer_s = 0.0

    def process(self, frames: list[F32]) -> list[F32]:
        t0 = time.perf_counter()
        x = torch.from_numpy(np.stack(frames)).to(self.runner.device)
        for _ in range(self.passes):
            x = self.runner.run(x)
        out = x.cpu().numpy().astype(np.float32, copy=False)
        self._infer_s += time.perf_counter() - t0
        return [out[i] for i in range(out.shape[0])]

    def stats(self) -> dict[str, object]:
        return {
            "inference_s": round(self._infer_s, 3),
            "runner": {
                "frames": self.runner.stats.frames,
                "tiles": self.runner.stats.tiles,
                "infer_calls": self.runner.stats.infer_calls,
                "layouts": self.runner.stats.layouts,
                **self.runner.backend.describe(),
            },
        }

    def close(self) -> None:
        self.runner.backend.close()
