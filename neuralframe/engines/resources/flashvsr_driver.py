"""NeuralFrame driver for FlashVSR v1.1 (runs inside the FlashVSR environment, in examples/WanVSR).

Based on the project's infer_flashvsr_v1.1_full.py / _tiny.py, with paths and settings as arguments and one
change: frames are padded (edge pixels repeated) to the 128-pixel multiple the model needs instead of
centre-cropped, and the padding is cut off again, so the output covers the whole picture at exactly 4x.
Input: a folder of PNG frames; output: PNG frames with the same names.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--variant", choices=["full", "tiny"], default="full")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--sparse-ratio", type=float, default=2.0)
    p.add_argument("--local-range", type=int, default=11)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_pipeline(weights: Path, variant: str, device: str):
    from diffsynth import FlashVSRFullPipeline, FlashVSRTinyPipeline, ModelManager
    from utils.utils import Causal_LQ4x_Proj

    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    files = [str(weights / "diffusion_pytorch_model_streaming_dmd.safetensors")]
    if variant == "full":
        files.append(str(weights / "Wan2.1_VAE.pth"))
    mm.load_models(files)
    cls = FlashVSRFullPipeline if variant == "full" else FlashVSRTinyPipeline
    pipe = cls.from_model_manager(mm, device=device)
    proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=torch.bfloat16)
    proj.load_state_dict(torch.load(weights / "LQ_proj_in.ckpt", map_location="cpu"), strict=True)
    pipe.denoising_model().LQ_proj_in = proj
    if variant == "full":
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None
    else:
        from utils.TCDecoder import build_tcdecoder

        pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
        print(pipe.TCDecoder.load_state_dict(torch.load(weights / "TCDecoder.ckpt", map_location="cpu"),
                                             strict=False))
    pipe.to(device)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(["dit", "vae"])
    return pipe


def main() -> None:
    args = parse()
    names = sorted(p for p in os.listdir(args.input) if p.lower().endswith(".png"))
    if not names:
        sys.exit("no input frames")
    first = Image.open(os.path.join(args.input, names[0]))
    w0, h0 = first.size
    sw, sh = w0 * args.scale, h0 * args.scale
    tw, th = -(-sw // 128) * 128, -(-sh // 128) * 128
    count = len(names)
    padded = names + [names[-1]] * 4
    total = ((len(padded) - 1) // 8) * 8 + 1
    while total < count + 4:  # always cover every real frame (8n+1 frames incl. 4 of look-ahead)
        padded.append(names[-1])
        total = ((len(padded) - 1) // 8) * 8 + 1
    padded = padded[:total]
    print(f"input {w0}x{h0}, {count} frames -> {sw}x{sh} (model {tw}x{th}, {total} frames)", flush=True)

    frames = []
    for i, name in enumerate(padded):
        img = Image.open(os.path.join(args.input, name)).convert("RGB").resize((sw, sh), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.uint8)
        arr = np.pad(arr, ((0, th - sh), (0, tw - sw), (0, 0)), mode="edge")
        t = torch.from_numpy(arr).to(args.device, torch.float32).permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frames.append(t.to(torch.bfloat16))
        print(f"loading {i + 1}/{total}", flush=True)
    lq = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)

    pipe = load_pipeline(Path(args.weights), args.variant, args.device)
    kwargs = {"tiled": args.tiled} if args.variant == "full" else {}
    video = pipe(prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=args.seed,
                 LQ_video=lq, num_frames=total, height=th, width=tw, is_full_block=False, if_buffer=True,
                 topk_ratio=args.sparse_ratio * 768 * 1280 / (th * tw), kv_ratio=3.0,
                 local_range=args.local_range, color_fix=True, **kwargs)
    video = rearrange(video, "C T H W -> T H W C")
    out = ((video.float() + 1) * 127.5).clip(0, 255).to(torch.uint8).cpu().numpy()
    os.makedirs(args.output, exist_ok=True)
    produced = out.shape[0]
    for i, name in enumerate(names):
        frame = out[min(i, produced - 1)][:sh, :sw]
        Image.fromarray(frame).save(os.path.join(args.output, name))
        print(f"saving {i + 1}/{count}", flush=True)
    if produced < count:
        print(f"warning: the model produced {produced} of {count} frames; the last one was repeated", flush=True)


if __name__ == "__main__":
    main()
