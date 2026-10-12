"""SRVGGNetCompact (Real-ESRGAN "compact" / AnimeVideo models), re-implemented without basicsr."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ) -> None:
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), self._act(act_type, num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act(act_type, num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _act(kind: str, num_feat: int) -> nn.Module:
        if kind == "relu":
            return nn.ReLU(inplace=True)
        if kind == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        if kind == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        raise ValueError(f"unknown activation {kind!r}")

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # The network predicts a residual on top of nearest-neighbour upsampling.
        return out + F.interpolate(x, scale_factor=float(self.upscale), mode="nearest")
