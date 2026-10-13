# Benchmarks

All numbers were measured on one machine; raw results (with hardware details) are in
`benchmarks/results/`. Re-run with `python benchmarks/run.py`, `python benchmarks/quality.py`,
`python benchmarks/bench_framegen.py` and `python benchmarks/bench_ops.py`.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, driver 617.42 |
| CPU / RAM | AMD Ryzen 5 7640HS (6 cores / 12 threads), 15 GB |
| Software | Windows 11, Python 3.12, PyTorch 2.14.1 (CUDA 13.0), TensorRT 10.16.1, ONNX Runtime 1.31, FFmpeg 9.0.2 |

A laptop GPU at ~85-90 C throttles under sustained load, so long runs are a few percent slower than short
ones. fps always means output frames per second through the whole pipeline (decode -> inference ->
encode with audio and subtitles muxed), not model-only throughput; one-time TensorRT engine builds are
excluded and listed separately.

## End-to-end throughput

Raw table from `benchmarks/run.py` ([bench-20261012.md](../benchmarks/results/bench-20261012.md)); generated
29.97 fps H.264 + AAC clips. "x real time" compares output fps with the source rate; per-stage values are
busy fps of each pipeline stage (the lowest one is the bottleneck).

| model | backend | input -> output | frames | fps | x real time | peak VRAM | GPU util | bottleneck | per-stage busy fps |
|---|---|---|---|---|---|---|---|---|---|
| lanczos | - - | 854x480 -> 1920x1080 | 300 | 101.66 | 3.39 | 1275 MB | 53% | postprocess | decode 698, upload 144, inference 7363, postprocess 134, encode 179 |
| realesr-animevideov3 | tensorrt fp16 | 854x480 -> 1920x1080 | 300 | 35.16 | 1.17 | 5235 MB | 82% | upload | decode 1148, upload 41, inference 54, postprocess 41, encode 97 |
| realesr-animevideov3 | onnxruntime fp32 | 854x480 -> 1920x1080 | 300 | 3.79 | 0.13 | 4436 MB | 74% | inference | decode 458, upload 6, inference 4, postprocess 6, encode 19 |
| realesr-animevideov3 | torch fp16 | 854x480 -> 1920x1080 | 300 | 15.51 | 0.52 | 5564 MB | 100% | inference | decode 1069, upload 16, inference 16, postprocess 19, encode 129 |
| realesrgan-x2plus | tensorrt fp16 | 854x480 -> 1920x1080 | 120 | 6.27 | 0.21 | 1474 MB | 98% | inference | decode 854, upload 61, inference 6, postprocess 57, encode 164 |
| realesrgan-x2plus | torch fp16 | 854x480 -> 1920x1080 | 120 | 2.39 | 0.08 | 5760 MB | 99% | inference | decode 502, upload 53, inference 2, postprocess 47, encode 118 |
| realesrgan-x4plus | tensorrt fp16 | 854x480 -> 1920x1080 | 60 | 1.63 | 0.05 | 928 MB | 98% | inference | decode 1076, upload 22, inference 2, postprocess 20, encode 195 |
| realesrgan-x4plus | onnxruntime fp32 | 854x480 -> 1920x1080 | 60 | 0.28 | 0.01 | 2216 MB | 99% | inference | decode 791, upload 6, inference 0, postprocess 6, encode 113 |
| realesrgan-x4plus | torch fp16 | 854x480 -> 1920x1080 | 60 | 0.59 | 0.02 | 5253 MB | 100% | inference | decode 943, upload 18, inference 1, postprocess 15, encode 142 |
| realbasicvsr-x4 | torch fp16 | 854x480 -> 1920x1080 | 120 | 1.28 | 0.04 | 6613 MB | 98% | inference | decode 395, inference 1, encode 26 |
| lanczos + interpolate first | - - | 854x480 -> 1920x1080 | 600 | 65.53 | 2.19 | 1149 MB | 79% | framegen | decode 939, framegen 69, upload 313, inference 16768, postprocess 155, encode 206 |
| realesr-animevideov3 + interpolate first | tensorrt fp16 | 854x480 -> 1920x1080 | 600 | 29.39 | 0.98 | 5267 MB | 86% | framegen | decode 1104, framegen 31, upload 343, inference 61, postprocess 133, encode 108 |
| lanczos | - - | 1280x720 -> 3840x2160 | 300 | 42.48 | 1.42 | 3199 MB | 73% | postprocess | decode 279, upload 125, inference 3079, postprocess 52, encode 70 |
| realesr-animevideov3 | tensorrt fp16 | 1280x720 -> 3840x2160 | 120 | 10.54 | 0.35 | 4087 MB | 81% | postprocess | decode 196, upload 17, inference 15, postprocess 12, encode 35 |
| realesrgan-x4plus | tensorrt fp16 | 1280x720 -> 3840x2160 | 24 | 0.60 | 0.02 | 1988 MB | 98% | inference | decode 352, upload 25, inference 1, postprocess 10, encode 29 |
| lanczos | - - | 1920x1080 -> 3840x2160 | 300 | 32.81 | 1.09 | 3841 MB | 57% | postprocess | decode 141, upload 93, inference 1390, postprocess 48, encode 49 |
| realesr-animevideov3 | tensorrt fp16 | 1920x1080 -> 3840x2160 | 120 | 8.63 | 0.29 | 3847 MB | 89% | postprocess | decode 163, upload 11, inference 11, postprocess 9, encode 38 |
| realesrgan-x2plus | tensorrt fp16 | 1920x1080 -> 3840x2160 | 24 | 1.25 | 0.04 | 1504 MB | 96% | inference | decode 236, upload 25, inference 1, postprocess 19, encode 29 |
| lanczos | - (CPU) - | 854x480 -> 1920x1080 | 120 | 58.82 | 1.96 | 299 MB | 4% | postprocess | decode 614, preprocess 187, inference 457666, postprocess 72, encode 146 |
| realesr-animevideov3 | torch (CPU) fp32 | 854x480 -> 1920x1080 | 24 | 0.55 | 0.02 | 253 MB | 0% | inference | decode 392, inference 1, encode 32 |

Summary:

| | 480p -> 1080p | 720p -> 4K | 1080p -> 4K |
|---|---|---|---|
| Lanczos (GPU resize) | 101.7 fps | 42.5 fps | 32.8 fps |
| realesr-animevideov3, TensorRT | **35.2 fps (1.17x real time)** | 10.5 fps | 8.6 fps |
| realesrgan-x2plus, TensorRT | 6.3 fps | - | 1.25 fps |
| realesrgan-x4plus, TensorRT | 1.63 fps | 0.60 fps | 0.18 fps (acceptance run) |
| realbasicvsr-x4, PyTorch | 1.28 fps | - | - |

* **TensorRT vs PyTorch FP16, end to end**: animevideov3 2.3x (35.2 vs 15.5 fps), x2plus 2.6x (6.27 vs
  2.39), x4plus 2.8x (1.63 vs 0.59). ONNX Runtime runs the exported FP32 graph (3.8 / 0.28 fps): it is the
  portable path for non-NVIDIA GPUs, not the fast one on NVIDIA.
* **The GPU is the bottleneck** for every neural model (82-100 % utilisation, inference or the GPU-side
  upload/postprocess stages lowest); decode and encode run 3-50x faster than needed.
* **Frame generation** to 59.94 fps on top of animevideov3 keeps 29.4 output fps (600 frames from 300).
* x4 models at 2x targets compute the full 4x output and downscale it (1080p input -> 7680x4320 ->
  3840x2160), so x2plus or a smaller target is the efficient choice for 1080p sources.
* CPU only (no GPU): Lanczos still runs at 58.8 fps with the C++ ops; neural models are ~0.5 fps.

## Acceptance runs

| Run | Result |
|---|---|
| Reference case: `neuralframe upscale ref_480p.mp4 -o ref_1080p.mkv --model realesrgan-x4plus --target-height 1080 --preset quality` (10 s, 854x480, 29.97 fps, AAC + SRT, scene cut at 5 s) | 300 frames in 186 s of pipeline time (1.61 fps), GPU 99.3 % busy, inference-bound; sync verified, audio stream-copied, subtitles converted to SRT; one-time engine build 218 s |
| 4K target on the 8 GB GPU: Real-ESRGAN x4plus, 1080p -> 2160p | completes; the planner tiles automatically (768 px tiles, 32 px overlap, budget 5976 MB), peak VRAM 6.6 GB of 8 GB, 0.18 fps (x4plus on 1080p input is ~8x the work of 480p) |
| 480p -> 1080p + frame generation to 59.94 fps (animevideov3, TensorRT) | 600 frames, 29.8 fps output, same speed from CLI, service and desktop app |
| Docker image (`docker run --gpus all ... upscale`, Docker Desktop on Windows, WSL2 GPU) | `doctor` finds TensorRT, NVDEC and NVENC (H.264/HEVC/AV1); animevideov3 480p -> 1080p completes with TensorRT + NVENC and verified sync at 11.9 fps (native: 35.2 fps; the WSL2 GPU path is encode-bound) |
| Windows release (`packaging/windows/build_release.py`) | packaged CLI and desktop app pass the smoke test from a fresh cache with only the bundled FFmpeg and models: Lanczos, TensorRT (engine built inside the package) and frame generation jobs, desktop app self-test |

## Quality

The tables below were measured before the model update (M10). `benchmarks/quality.py` now also covers the
new models (BSRGAN, RealPLKSR, the HAT-L / DRCT-L / DAT2 / ATD / SwinIR-L transformers, realesr-general),
the automatic model choice for each preset, and a filmed reference (*Tears of Steel*, CC BY 3.0, 1920x800);
those results will be added here once measured.

Sintel trailer ((c) Blender Foundation | durian.blender.org, CC BY 3.0), 6 s from 32.5 s, 1920x816 with
the letterbox removed, downscaled with bicubic filtering. PSNR uses MSE pooled over all frames; SSIM and
PSNR are on luma. Flicker = temporal change in regions that are static in the original (lower is
steadier). Seam ratio = frame-to-frame change at chunk boundaries relative to elsewhere, normalised by the
chunk-free bicubic baseline (1.0 = invisible; the spec flags > 1.2). Crops: [GALLERY.md](GALLERY.md).

### 4x, clean input (480x204 -> 1920x816)

| method | PSNR-Y | SSIM-Y | VMAF | LPIPS | flicker | seam ratio | fps |
|---|---|---|---|---|---|---|---|
| FFmpeg bicubic | 39.95 | 0.9828 | 69.84 | 0.1029 | 0.319 | - | - |
| FFmpeg Lanczos | 40.18 | 0.9830 | 73.01 | 0.1018 | 0.328 | - | - |
| NeuralFrame realesrgan-x4plus | 39.24 | 0.9761 | 86.73 | 0.0998 | 0.787 | 1.041 | 6.8 |
| NeuralFrame realesr-animevideov3 | 37.51 | 0.9797 | 81.83 | **0.0877** | 0.616 | 1.012 | 29.1 |
| NeuralFrame realbasicvsr-x4 | 34.96 | 0.9303 | **92.37** | 0.2526 | 1.675 | 1.066 | 4.2 |
| NeuralFrame lanczos | 40.20 | 0.9831 | 72.79 | 0.1002 | 0.319 | - | 35.1 |

### 4x, compressed input (x264 CRF 28)

| method | PSNR-Y | SSIM-Y | VMAF | LPIPS | flicker | seam ratio | fps |
|---|---|---|---|---|---|---|---|
| FFmpeg bicubic | 37.25 | 0.9714 | 44.48 | 0.1739 | 0.724 | - | - |
| FFmpeg Lanczos | 37.32 | 0.9712 | 46.08 | 0.1733 | 0.731 | - | - |
| NeuralFrame realesrgan-x4plus | 37.24 | 0.9690 | 65.52 | 0.1307 | 0.900 | 1.039 | 6.6 |
| NeuralFrame realesr-animevideov3 | 35.86 | 0.9720 | 62.94 | **0.1223** | 0.725 | 1.015 | 31.4 |
| NeuralFrame realbasicvsr-x4 | 33.02 | 0.9103 | **76.24** | 0.3068 | 1.863 | 1.050 | 5.0 |

### 2x, clean input (960x408 -> 1920x816)

| method | PSNR-Y | SSIM-Y | VMAF | LPIPS | flicker | seam ratio | fps |
|---|---|---|---|---|---|---|---|
| FFmpeg bicubic | 45.47 | 0.9942 | 91.34 | 0.0324 | 0.166 | - | - |
| FFmpeg Lanczos | 45.91 | 0.9944 | 92.77 | 0.0319 | 0.171 | - | - |
| NeuralFrame realesrgan-x2plus | 42.89 | 0.9841 | 92.22 | 0.0789 | 0.748 | 1.027 | 6.6 |
| NeuralFrame lanczos | 45.95 | 0.9945 | 92.68 | 0.0256 | 0.165 | - | 41.0 |

Reading the numbers:

* The neural models do what they are trained for on degraded input: on the compressed clip they add
  18-32 VMAF points and cut LPIPS by 25-30 % over bicubic. PSNR stays at or below bicubic because GAN
  models invent plausible detail instead of averaging (PSNR rewards blur).
* RealBasicVSR scores the highest VMAF but the worst LPIPS and flicker: it sharpens strongly and invents
  line texture, which VMAF rewards; on this soft CG footage that looks painted (see the gallery).
* On clean 2x input Real-ESRGAN x2plus (trained for heavy degradations) does not beat Lanczos.
* **Chunk seams are invisible** (normalised ratio 1.01-1.07 for every model, below the 1.2 threshold);
  **tile seams**: Real-ESRGAN x4plus with 128 px tiles vs whole frames differs by 55.9 dB PSNR at worst
  (threshold 45 dB).
* The absolute PSNR values are high because Sintel is soft CG with depth of field; compare methods within
  a table, not with photo benchmarks.

## Frame generation (RIFE 4.22, PyTorch FP16)

Interpolation alone (one source pair, synthetic frames, `benchmarks/bench_framegen.py`):

| size | flow scale | frames/s (1 timestep/call) | peak VRAM |
|---|---|---|---|
| 720p | 1.0 | 45.1 | 390 MB |
| 1080p | 1.0 | 19.5 | 845 MB |
| 1440p | 0.5 | 15.6 | 1045 MB |
| 4K | 0.5 | 6.8 | 2294 MB |

FP16 is about 2x FP32; batching several timesteps gives no speed-up but multiplies VRAM, so batches are
sized from free VRAM. Quality: regenerating every other frame of a 50 fps pan, RIFE reaches 36.6 dB PSNR
against 26.6 dB for a cross-fade; copied frames are bit-exact.

Order of upscaling and interpolation (reference clip, 300 -> 600 frames at 59.94 fps; measured in M6,
before the cache-cap fix below, so absolute values are lower than the throughput table):

| SR model | upscale first | interpolate first | `auto` picks |
|---|---|---|---|
| realesr-animevideov3 -> 1080p | 21.9 fps | 29.8 fps | interpolate first |
| Lanczos -> 4K | 10.4 fps | 34.2 fps | interpolate first |
| realesrgan-x4plus | (SR ~1.7 s/frame dominates) | 2x the SR work | upscale first |

## Native C++ ops vs the NumPy reference

Bit-identical output; Ryzen 5 7640HS, 12 threads (`benchmarks/bench_ops.py`):

| op | NumPy (ms) | C++ (ms) | speed-up |
|---|---:|---:|---:|
| YUV 4:2:0 -> RGB, 1080p | 56.7 | 2.8 | 20.1x |
| RGB -> YUV 4:2:0, 1080p | 61.4 | 3.3 | 18.8x |
| Lanczos 480p -> 1080p | 174.5 | 9.0 | 19.3x |
| Lanczos 3416x1920 -> 1080p | 774.5 | 23.1 | 33.5x |
| Gamut BT.601 -> BT.709, 480p | 48.6 | 2.1 | 23.0x |

## Optimisation log (measured before / after)

| Change | Before | After |
|---|---|---|
| C++ ops instead of NumPy (Lanczos pipeline, 1080p output, host mode) | 2.9 fps | 45 fps |
| TensorRT FP16 instead of PyTorch eager FP16 (480p whole frames, model only) | animevideov3 52.8 ms, x2plus 344 ms, x4plus 1550 ms | 39.8 / 159 / 590 ms (1.3x / 2.2x / 2.6x) |
| Banded instead of dense resize matrices on the GPU (8 frames 3416x1920 -> 1080p) | 87.8 ms | 18.3 ms |
| CUDA-stream overlap of upload / inference / download (animevideov3 480p -> 1080p) | 34.5 fps | 35.5 fps (GPU already ~82 % busy) |
| RealBasicVSR: SPyNet in FP32 instead of all-FP16 (vs FP32 reference) | 44.7 dB | 49.8 dB, same speed |
| RIFE timestep batches sized from free VRAM (1440p FP32, 4 timesteps) | 3.16 s/call (shared-memory spill), 4K OOM | 0.73 s/call, no OOM |
| PyTorch cache cap + RIFE VRAM reservation (480p -> 1080p + 59.94 fps) | 11.2 fps, 8.2 GB used | 21.1 fps, 1.2 GB kept free |
| Frame generation order chosen by model cost (Lanczos -> 4K + 60 fps) | 10.4 fps | 34.2 fps |
| Cache cap keeps 5 % headroom instead of a second safety margin (animevideov3 TensorRT 480p -> 1080p; three configurations that crashed now run) | 17.4 fps, OOM crashes | 35.2 fps |
| RealBasicVSR flow + cleaning in 4-frame groups (480p -> 1080p) | out of memory (2.7 GB allocation) | 1.28 fps in 6.6 GB |
