"""NeuralFrame driver for NVIDIA RTX Video Super Resolution (nvidia-vfx), following NVIDIA's
video_super_resolution.py sample: PNG frames in, PNG frames out (8-bit RGB)."""

import argparse
import os
import sys

import numpy as np
import torch
from nvvfx import VideoSuperRes
from PIL import Image


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--scale", type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument("--quality", default="HIGH")
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    names = sorted(n for n in os.listdir(args.input) if n.lower().endswith(".png"))
    if not names:
        sys.exit("no input frames")
    w, h = Image.open(os.path.join(args.input, names[0])).size
    sr = VideoSuperRes(device=args.gpu, quality=VideoSuperRes.QualityLevel[args.quality], strength=args.strength,
                       image_encoding=VideoSuperRes.ImageEncoding.RGB8)
    sr.input_width, sr.input_height = w, h
    sr.output_width, sr.output_height = w * args.scale, h * args.scale
    sr.load()
    os.makedirs(args.output, exist_ok=True)
    device = f"cuda:{args.gpu}"
    for i, name in enumerate(names):
        arr = np.asarray(Image.open(os.path.join(args.input, name)).convert("RGB"), dtype=np.uint8)
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(device).permute(2, 0, 1).float().div(255.0).contiguous()
        out = torch.from_dlpack(sr.run(x).image).clone()
        img = (out.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(img).save(os.path.join(args.output, name))
        print(f"frame {i + 1}/{len(names)}", flush=True)
    sr.close()


if __name__ == "__main__":
    main()
