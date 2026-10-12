"""RealBasicVSR (Chan et al., CVPR 2022): image cleaning + BasicVSR bidirectional recurrent propagation.

Re-implemented without mmcv/mmagic; parameter names match the official checkpoint
(``generator_ema.*`` in the OpenMMLab release) so ``load_state_dict`` works directly.

Unlike the reference ``forward``, :meth:`RealBasicVSRNet.iter_outputs` yields one upscaled frame at a
time, so a chunk never holds every 4x output in VRAM at once.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def flow_warp(x: Tensor, flow: Tensor, padding_mode: str = "zeros") -> Tensor:
    """Warp ``x`` (N, C, H, W) with ``flow`` (N, H, W, 2) in pixels (bilinear, align_corners=True)."""
    _, _, h, w = x.shape
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(x, torch.stack((vgrid_x, vgrid_y), dim=3), mode="bilinear",
                         padding_mode=padding_mode, align_corners=True)


class ResidualBlockNoBN(nn.Module):
    def __init__(self, mid_channels: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.relu(self.conv1(x)))


class ResidualBlocksWithInputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Sequential(*[ResidualBlockNoBN(out_channels) for _ in range(num_blocks)]),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class PixelShufflePack(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int, kernel: int = 3) -> None:
        super().__init__()
        self.scale = scale
        self.upsample_conv = nn.Conv2d(in_channels, out_channels * scale * scale, kernel, 1, (kernel - 1) // 2)

    def forward(self, x: Tensor) -> Tensor:
        return F.pixel_shuffle(self.upsample_conv(x), self.scale)


class _ConvModule(nn.Module):
    """mmcv ConvModule subset: conv (+ optional ReLU), weights under ``.conv``."""

    def __init__(self, cin: int, cout: int, relu: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 7, 1, 3)
        self.act: nn.Module = nn.ReLU(inplace=True) if relu else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.conv(x))


class SPyNetBasicModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.Sequential(
            _ConvModule(8, 32, True), _ConvModule(32, 64, True), _ConvModule(64, 32, True),
            _ConvModule(32, 16, True), _ConvModule(16, 2, False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.basic_module(x)


class SPyNet(nn.Module):
    """Spatial-pyramid optical flow (6 levels, coarse to fine)."""

    mean: Tensor
    std: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref: Tensor, supp: Tensor) -> Tensor:
        n, _, h, w = ref.shape
        refs = [(ref - self.mean) / self.std]
        supps = [(supp - self.mean) / self.std]
        for _ in range(5):
            refs.append(F.avg_pool2d(refs[-1], kernel_size=2, stride=2, count_include_pad=False))
            supps.append(F.avg_pool2d(supps[-1], kernel_size=2, stride=2, count_include_pad=False))
        refs, supps = refs[::-1], supps[::-1]
        flow = refs[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(refs)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=True) * 2.0
            warped = flow_warp(supps[level], flow_up.permute(0, 2, 3, 1), padding_mode="border")
            flow = flow_up + self.basic_module[level](torch.cat([refs[level], warped, flow_up], 1))
        return flow

    def forward(self, ref: Tensor, supp: Tensor) -> Tensor:
        h, w = ref.shape[2:4]
        w_up = w if w % 32 == 0 else 32 * (w // 32 + 1)
        h_up = h if h % 32 == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(ref, size=(h_up, w_up), mode="bilinear", align_corners=False)
        supp = F.interpolate(supp, size=(h_up, w_up), mode="bilinear", align_corners=False)
        flow = F.interpolate(self.compute_flow(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        scale = torch.tensor([w / w_up, h / h_up], dtype=flow.dtype, device=flow.device).view(1, 2, 1, 1)
        return flow * scale


# Frames per call of the per-frame parts (optical flow, image cleaning). They have no coupling between
# frames, so processing a chunk in groups gives the same result while peak memory stops growing with the
# chunk length (measured: a 2.7 GB single allocation for a 480p chunk when run on all frames at once).
FRAME_GROUP = 4


class BasicVSRNet(nn.Module):
    def __init__(self, mid_channels: int = 64, num_blocks: int = 30) -> None:
        super().__init__()
        self.mid_channels = mid_channels
        self.spynet = SPyNet()
        self.backward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.forward_resblocks = ResidualBlocksWithInputConv(mid_channels + 3, mid_channels, num_blocks)
        self.fusion = nn.Conv2d(mid_channels * 2, mid_channels, 1, 1, 0)
        self.upsample1 = PixelShufflePack(mid_channels, mid_channels, 2)
        self.upsample2 = PixelShufflePack(mid_channels, 64, 2)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.strip_rows: int | None = None
        """Low-res rows per strip for the 4x upsampling head (None = whole frame)."""

    _HALO = 4  # low-res rows of context (receptive field ~2 rows): strips match the whole frame up to float rounding

    def _head(self, fused: Tensor) -> Tensor:
        out = self.lrelu(self.upsample1(fused))
        out = self.lrelu(self.upsample2(out))
        return self.conv_last(self.lrelu(self.conv_hr(out)))

    def upsample_head(self, fused: Tensor) -> Tensor:
        """Run the 4x head in horizontal strips (with halo) to bound its large activation memory."""
        h = fused.shape[-2]
        rows = self.strip_rows
        if rows is None or rows >= h:
            return self._head(fused)
        parts = []
        for y0 in range(0, h, rows):
            y1 = min(h, y0 + rows)
            a, b = max(0, y0 - self._HALO), min(h, y1 + self._HALO)
            out = self._head(fused[..., a:b, :])
            parts.append(out[..., (y0 - a) * 4 : (y0 - a) * 4 + (y1 - y0) * 4, :])
        return torch.cat(parts, dim=-2)

    def _flows(self, a: Tensor, b: Tensor, dtype: torch.dtype) -> Tensor:
        """SPyNet flow ``a -> b`` for frame pairs, a few pairs per call (bounded memory)."""
        return torch.cat([self.spynet(a[s : s + FRAME_GROUP].to(dtype), b[s : s + FRAME_GROUP].to(dtype))
                          for s in range(0, a.shape[0], FRAME_GROUP)])

    def iter_outputs(self, lrs: Tensor) -> Iterator[tuple[int, Tensor]]:
        """Yield ``(t, (N, 3, 4H, 4W))`` frame by frame for ``lrs`` (N, T, 3, H, W)."""
        n, t, c, h, w = lrs.shape
        if t > 1:
            # Optical flow is computed in the precision of the SPyNet weights (kept FP32 under FP16
            # inference: small flow errors misalign whole features; measured +PSNR vs all-FP16).
            flow_dtype = self.spynet.mean.dtype
            lrs_1 = lrs[:, :-1].reshape(-1, c, h, w)
            lrs_2 = lrs[:, 1:].reshape(-1, c, h, w)
            flows_backward = self._flows(lrs_1, lrs_2, flow_dtype).view(n, t - 1, 2, h, w).to(lrs.dtype)
            flows_forward = self._flows(lrs_2, lrs_1, flow_dtype).view(n, t - 1, 2, h, w).to(lrs.dtype)
        # backward-time propagation (features kept at low resolution)
        backward: list[Tensor] = [lrs.new_empty(0)] * t
        feat = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            if i < t - 1:
                feat = flow_warp(feat, flows_backward[:, i].permute(0, 2, 3, 1))
            feat = self.backward_resblocks(torch.cat([lrs[:, i], feat], dim=1))
            backward[i] = feat
        # forward-time propagation, fusion and upsampling, one frame at a time
        feat = torch.zeros_like(feat)
        for i in range(t):
            lr = lrs[:, i]
            if i > 0:
                feat = flow_warp(feat, flows_forward[:, i - 1].permute(0, 2, 3, 1))
            feat = self.forward_resblocks(torch.cat([lr, feat], dim=1))
            out = self.lrelu(self.fusion(torch.cat([backward[i], feat], dim=1)))
            backward[i] = lrs.new_empty(0)  # release the backward feature as soon as it is used
            out = self.upsample_head(out)
            out = out + F.interpolate(lr, scale_factor=4.0, mode="bilinear", align_corners=False)
            yield i, out

    def forward(self, lrs: Tensor) -> Tensor:
        return torch.stack([out for _, out in self.iter_outputs(lrs)], dim=1)


class RealBasicVSRNet(nn.Module):
    """Image cleaning (residual) followed by BasicVSR. Input/outputs: (N, T, 3, H, W) → (N, T, 3, 4H, 4W)."""

    is_sequence_model = True

    def half(self) -> RealBasicVSRNet:
        """FP16 everywhere except the optical-flow network."""
        super().half()
        self.basicvsr.spynet.float()
        return self

    def __init__(
        self,
        mid_channels: int = 64,
        num_propagation_blocks: int = 20,
        num_cleaning_blocks: int = 20,
        dynamic_refine_thres: float = 255.0,
    ) -> None:
        super().__init__()
        self.dynamic_refine_thres = dynamic_refine_thres / 255.0
        self.image_cleaning = nn.Sequential(
            ResidualBlocksWithInputConv(3, mid_channels, num_cleaning_blocks),
            nn.Conv2d(mid_channels, 3, 3, 1, 1),
        )
        self.basicvsr = BasicVSRNet(mid_channels, num_propagation_blocks)

    def clean(self, lqs: Tensor) -> Tensor:
        n, t, c, h, w = lqs.shape
        flat = lqs.reshape(-1, c, h, w)
        for _ in range(3):  # at most three cleaning passes, as in the paper
            cleaned = []
            residue_sum = flat.new_zeros((), dtype=torch.float32)
            for s in range(0, flat.shape[0], FRAME_GROUP):  # per-frame network: a few frames per call
                part = flat[s : s + FRAME_GROUP]
                residues = self.image_cleaning(part)
                residue_sum += residues.abs().sum(dtype=torch.float32)
                cleaned.append(part + residues)
            flat = torch.cat(cleaned)
            if residue_sum / flat.numel() < self.dynamic_refine_thres:  # mean over all frames, as before
                break
        return flat.view(n, t, c, h, w)

    def iter_outputs(self, lqs: Tensor) -> Iterator[tuple[int, Tensor]]:
        yield from self.basicvsr.iter_outputs(self.clean(lqs))

    def forward(self, lqs: Tensor) -> Tensor:
        return self.basicvsr(self.clean(lqs))
