# Model cards

NeuralFrame never ships weights: they are downloaded on first use from the URLs below, verified against
the pinned SHA256 and cached (`<cache>/models/<id>/`). `neuralframe models list` shows the registry,
`neuralframe models download <id>` fetches ahead of time, `neuralframe models export <id>` builds the ONNX
file and checks TensorRT/ONNX Runtime parity against PyTorch. The registry lives in
`neuralframe/data/registry.yaml`.

Speeds are end-to-end (decode -> model -> encode) on an RTX 4060 Laptop GPU (8 GB) from
`docs/BENCHMARKS.md`; quality numbers come from `docs/BENCHMARKS.md#quality` (Sintel trailer, CC BY 3.0).

## Super-resolution

### realesrgan-x4plus

| | |
|---|---|
| Purpose | General-purpose 4x upscaling of real-world, compressed live-action footage and photos |
| Architecture | RRDBNet (23 RRDB blocks, 64 features), ~16.7 M parameters |
| Source | Real-ESRGAN, Wang et al., ICCVW 2021 - https://github.com/xinntao/Real-ESRGAN |
| Licence | BSD-3-Clause (code and released weights) |
| Weights | `RealESRGAN_x4plus.pth`, 67.0 MB, SHA256 `4fa0d389...d682f1` |
| Backends | TensorRT FP16, ONNX Runtime, PyTorch (FP16 on CUDA, FP32 on CPU) |
| VRAM | ~8.8 KB per input pixel (FP16, PyTorch); whole 480p frames fit in 8 GB, larger inputs are tiled |
| Quality | 4x Sintel test: VMAF 86.7 vs 69.8 bicubic (clean input), 65.5 vs 44.5 (compressed input); LPIPS 0.100 / 0.131 vs 0.103 / 0.174; PSNR within 0.7 dB of bicubic |
| Limitations | GAN model: invents plausible detail rather than recovering it, and that detail can shimmer between frames (flicker 0.79 vs 0.32 for bicubic); slowest single-frame model here |

### realesrgan-x2plus

| | |
|---|---|
| Purpose | 2x upscaling (e.g. 1080p -> 4K) with the same network as x4plus |
| Architecture | RRDBNet on pixel-unshuffled input, ~16.7 M parameters |
| Source / licence | Real-ESRGAN (as above), BSD-3-Clause |
| Weights | `RealESRGAN_x2plus.pth`, 67.1 MB, SHA256 `49fafd45...fa266abb` |
| Backends | TensorRT FP16, ONNX Runtime, PyTorch |
| VRAM | ~2.2 KB per input pixel (FP16) |
| Limitations | Trained on heavy degradations: on clean input it smooths fine texture (2x Sintel test: PSNR 42.9 vs 45.9 dB and LPIPS 0.079 vs 0.032 for Lanczos, VMAF about equal). Use it for degraded sources |

### realesr-animevideov3

| | |
|---|---|
| Purpose | Fast 4x upscaling of animation, anime and cartoons; also a fast general option |
| Architecture | SRVGGNetCompact (16 conv layers, 64 features, PReLU), ~0.6 M parameters |
| Source / licence | Real-ESRGAN (as above), BSD-3-Clause |
| Weights | `realesr-animevideov3.pth`, 2.5 MB, SHA256 `b8a83768...0a18b75d` |
| Backends | TensorRT FP16, ONNX Runtime, PyTorch |
| VRAM | ~0.55 KB per input pixel (FP16) |
| Speed | Faster than real time for 480p -> 1080p with TensorRT |
| Quality | Best LPIPS in the 4x Sintel test (0.088 clean, 0.122 compressed vs 0.103 / 0.174 bicubic); VMAF 81.8 / 62.9 vs 69.8 / 44.5 |
| Limitations | Tuned for flat shading and line art; live-action skin and foliage can look painted |

### realbasicvsr-x4 (temporal)

| | |
|---|---|
| Purpose | 4x upscaling of real-world degraded video with temporal consistency |
| Architecture | RealBasicVSR: image cleaning module + bidirectional recurrent propagation with SPyNet optical flow, 64 channels, 20+20 blocks |
| Source | Chan et al., CVPR 2022 - https://github.com/ckkelvinchan/RealBasicVSR |
| Licence | Apache-2.0 (code); weights released with the repository (trained on REDS: check its terms for your use) |
| Weights | `realbasicvsr_c64b20_1x30x8_lr5e-5_150k_reds_20211104-52f77c2c.pth`, 148 MB, SHA256 `52f77c2c...55a4e376` |
| Backends | PyTorch only (recurrent across a whole chunk); FP16 with SPyNet in FP32 |
| Chunking | Chunks of up to 64 frames with overlap and cross-fades; scene cuts are hard boundaries |
| Quality | Highest VMAF in the 4x Sintel test (92.4 clean, 76.2 compressed) but the worst LPIPS (0.25 / 0.31) and PSNR: it sharpens strongly and invents line texture, which VMAF rewards. On soft CG footage the result looks painted; it suits noisy real-world video better |
| Limitations | Most temporal noise in static regions of the evaluated models; memory grows with chunk length (bounded automatically) |

### lanczos / bicubic (classical)

Resampling only (Lanczos-3, Catmull-Rom bicubic), implemented in NumPy, C++ and PyTorch. Baselines for
the evaluation, the "keep the size" path for frame generation, and a fast option for already-clean
content. No licence restrictions.

## Frame interpolation

### rife-4.22

| | |
|---|---|
| Purpose | Frame generation: constant output frame rate (default 60 fps) and re-synthesis of dropped/frozen frames |
| Architecture | IFNet (Practical-RIFE v4.22): four coarse-to-fine IFBlocks estimating bidirectional flow and a fusion mask at an arbitrary timestep |
| Source | Huang et al., ECCV 2022 - https://github.com/hzwer/Practical-RIFE (weights packaged by https://github.com/HolyWu/vs-rife) |
| Licence | MIT |
| Weights | `flownet_v4.22.pkl`, 41.0 MB, SHA256 `549df85a...8c3d4af5` |
| Backends | PyTorch FP16 (ONNX/TensorRT export not implemented yet) |
| Speed (alone) | 720p 45 fps, 1080p 19.5 fps, 1440p 15.6 fps, 4K 6.8 fps (flow scale 1.0 up to 1080p, 0.5 above) |
| Quality | 36.6 dB vs 26.6 dB for a cross-fade when regenerating every other frame of a 50 fps pan |
| Limitations | Large or non-rigid motion, occlusions and on-screen text can warp; across scene cuts and long gaps NeuralFrame holds the nearest frame instead of interpolating |
