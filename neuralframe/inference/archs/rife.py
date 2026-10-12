"""RIFE IFNet (Practical-RIFE HD v3 family, v4.22 layout) for arbitrary-timestep frame interpolation.

Re-implemented in plain PyTorch with parameter names matching the released ``flownet.pkl`` files
(MIT licence). Four coarse-to-fine IFBlocks estimate bidirectional flow to the intermediate time ``t``
plus a fusion mask; a light encoder supplies 8-channel features that are warped alongside the images.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=True), nn.LeakyReLU(0.2, True))


def warp(x: Tensor, flow: Tensor) -> Tensor:
    """Backward-warp ``x`` by ``flow`` (N, 2, H, W) in pixels; border padding, align_corners=True."""
    _, _, h, w = x.shape
    gy, gx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=x.device, dtype=flow.dtype),
        torch.linspace(-1.0, 1.0, w, device=x.device, dtype=flow.dtype),
        indexing="ij",
    )
    norm = torch.cat([flow[:, 0:1] / ((w - 1.0) / 2.0), flow[:, 1:2] / ((h - 1.0) / 2.0)], 1)
    grid = (torch.stack([gx, gy], 0).unsqueeze(0) + norm).permute(0, 2, 3, 1)
    return F.grid_sample(x, grid.to(x.dtype), mode="bilinear", padding_mode="border", align_corners=True)


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class ResConv(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)))
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    def __init__(self, in_planes: int, c: int = 64) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(_conv(in_planes, c // 2, 2), _conv(c // 2, c, 2))
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1), nn.PixelShuffle(2))

    def forward(self, x: Tensor, flow: Tensor | None, scale: float) -> tuple[Tensor, Tensor, Tensor]:
        x = F.interpolate(x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False) / scale
            x = torch.cat((x, flow), 1)
        feat = self.convblock(self.conv0(x))
        tmp = F.interpolate(self.lastconv(feat), scale_factor=scale, mode="bilinear", align_corners=False)
        return tmp[:, :4] * scale, tmp[:, 4:5], tmp[:, 5:]


class IFNet(nn.Module):
    """``forward(img0, img1, timestep)`` → intermediate frame. Inputs (N, 3, H, W) in [0, 1] with H and W
    multiples of :meth:`multiple`; ``timestep`` is a float or an (N, 1, 1, 1) tensor in (0, 1)."""

    def __init__(self, scale: float = 1.0, channels: tuple[int, int, int, int] = (256, 192, 96, 48)) -> None:
        super().__init__()
        self.block0 = IFBlock(7 + 16, c=channels[0])
        self.block1 = IFBlock(8 + 4 + 16 + 8, c=channels[1])
        self.block2 = IFBlock(8 + 4 + 16 + 8, c=channels[2])
        self.block3 = IFBlock(8 + 4 + 16 + 8, c=channels[3])
        self.encode = Head()
        self.scale = scale

    @property
    def scale_list(self) -> list[float]:
        return [8.0 / self.scale, 4.0 / self.scale, 2.0 / self.scale, 1.0 / self.scale]

    def multiple(self) -> int:
        """Input alignment: the coarsest level works at 1/(8/scale) and its encoder adds another /4."""
        return max(32, int(32 / self.scale))

    def forward(self, img0: Tensor, img1: Tensor, timestep: Tensor | float = 0.5) -> Tensor:
        img0 = img0.clamp(0.0, 1.0)
        img1 = img1.clamp(0.0, 1.0)
        n, _, h, w = img0.shape
        t = (timestep if isinstance(timestep, Tensor) else img0.new_full((n, 1, 1, 1), float(timestep)))
        t = t.to(img0.dtype).reshape(n, 1, 1, 1).expand(n, 1, h, w)
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        blocks = [self.block0, self.block1, self.block2, self.block3]
        flow: Tensor | None = None
        mask = feat = img0.new_empty(0)
        warped0, warped1 = img0, img1
        for i, block in enumerate(blocks):
            if flow is None:
                flow, mask, feat = block(torch.cat((img0, img1, f0, f1, t), 1), None, scale=self.scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, mask, feat = block(
                    torch.cat((warped0, warped1, wf0, wf1, t, mask, feat), 1), flow, scale=self.scale_list[i]
                )
                flow = flow + fd
            warped0 = warp(img0, flow[:, :2])
            warped1 = warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return warped0 * m + warped1 * (1 - m)
