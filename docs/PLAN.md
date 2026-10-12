# NeuralFrame — Development Plan

Living roadmap for building NeuralFrame. Each milestone ends with green tests and a commit on
`develop`; checkboxes are updated as work lands.

## 1. What we are building

An offline, GPU-accelerated video super-resolution pipeline:

```
probe → decode (hw) → preprocess → chunk plan → inference (tiles, batches, TRT) → stitch
      → postprocess (resize, colour) → encode (hw) → mux (stream-copy audio/subs/chapters)
```

Exposed through a Typer CLI, a FastAPI job service (SQLite queue, SSE progress), a minimal web UI and a
desktop GUI application.

### 1.1 Scope additions (requested 2026-10-12, extend the original spec)

1. **Desktop GUI application** (PySide6/Qt): pick an image or video file (dialog or drag-and-drop), choose
   what to do, watch a live progress bar (percent, ETA, fps, current stage, cancel), get the output file
   (open / show in folder).
   - Video → *Upscale*, *Frame generation*, or *Both*. Image → *Upscale* only.
2. **Upscale target defaults to 4K** (2160p, aspect ratio preserved; portrait fits 2160×3840). Custom
   targets: preset heights (720p/1080p/1440p/4K/8K), scale factor (×2, ×3, ×4, any float), or exact
   width×height. Factors larger than the model's native scale use cascaded model passes when worthwhile.
3. **Image upscaling** (PNG/JPEG/WebP/BMP/TIFF in and out; 16-bit PNG preserved) through the same
   tiled inference core.
4. **Frame generation** (interpolation) — moves out of the original non-goals:
   - Output is constant frame rate, **default 60 fps**, user-selectable (snaps to 60000/1001 when the
     source is on an NTSC clock).
   - **Smart drop repair:** an analysis pass finds (a) timestamp gaps (delta > 1.5× the median frame
     interval) and (b) duplicated frames inside moving content (near-zero difference between neighbours
     whose surrounding motion is non-zero — the signature of a dropped frame filled by a repeat). Duplicates
     become holes; holes and gaps are re-synthesised at their correct times, so motion is smooth.
   - Retiming maps each output timestamp to its two surrounding unique source frames and an interpolation
     time `t`; exact matches are copied, never interpolated. No interpolation across scene cuts (nearest
     frame is used instead).
   - Model: RIFE v4.x family (arbitrary-timestep IFNet, MIT licence) in PyTorch with ONNX/TensorRT export;
     FFmpeg `minterpolate` as a model-free fallback.
   - Default order when both are selected: **upscale → interpolate** (the SR model is the expensive part, so
     it runs on source frames only; interpolated frames inherit consistent SR detail). Configurable.
   - Frame/drop statistics go into the job report.

## 2. Development machine (capability check, 2026-10-12)

| Item | Detected |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU / RAM | AMD Ryzen 5 7640HS (12 threads) / 15 GB |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB, SM 8.9, driver 617.42 (CUDA 13.4 capable) |
| Python | 3.12.10 (uv-managed `.venv`); 3.11 also available |
| C++ toolchain | MSVC 14.51 (Visual Studio 2026 Community), CMake + Ninja via pip |
| CUDA toolkit (nvcc) | **not installed** → optional CUDA kernels cannot be compiled locally |
| FFmpeg | 9.0.2 full build (gyan.dev): NVENC/NVDEC/CUVID, AMF, QSV, VAAPI, libx264/x265/svtav1, libvmaf, scdet, bwdif |
| Torch | 2.14.1+cu130 (CUDA available) |
| TensorRT | 10.16.1 (`tensorrt-cu13`) |
| ONNX Runtime | 1.31.0 GPU (TensorRT / CUDA / CPU EPs) |
| Docker | 29.7 with `nvidia` runtime |

## 3. Key design decisions

1. **Video I/O = FFmpeg subprocesses + NUT pipes.** The decoder runs `ffmpeg … -c:v rawvideo -f nut pipe:1`;
   PyAV *demuxes* (not decodes) the NUT stream so each raw frame arrives with its exact PTS. The encoder is
   fed the same way in reverse (PyAV muxes raw packets with original PTS → `ffmpeg -f nut -i pipe:0`).
   Verified experimentally: PTS round-trips exactly, including VFR jitter and 10-bit formats.
   *Why:* hardware codecs available to the system FFmpeg (what `doctor` reports) are exactly what we use;
   codec work runs in separate processes (no GIL contention); timestamps are carried in-band so VFR is
   handled without assuming constant FPS.
2. **Colour conversion is ours, not swscale's.** Decoder outputs native planar YUV (8/10-bit); our ops
   convert YUV→RGB with the correct matrix/range (BT.601/709/2020, limited/full) exactly once, optional
   gamut conversion SD→BT.709 at input resolution, and RGB→YUV + chroma downsampling on output.
3. **One op, three implementations.** Every hot-path op (YUV↔RGB, Lanczos resize, tile feather-merge,
   temporal cross-fade) exists as: NumPy reference (`neuralframe/ops/reference.py`, ground truth in tests),
   C++ (`neuralframe._native`, multi-threaded, releases the GIL) and Torch device ops (GPU path; stands in
   for hand-written CUDA kernels, which are optional and only built when `nvcc` exists).
4. **Fixed-size spatial tiles.** Tiles keep a constant shape (edge tiles shift inward instead of
   shrinking) so batches are uniform and TensorRT runs one optimisation profile. Merge uses feathered linear
   ramps only on interior edges.
5. **Chunks are pts ranges, outputs are segments.** Each chunk is encoded into its own segment in a chunk
   store with a manifest (frame count + checksum). Resume skips verified chunks; the final output is a
   concat of segments + one stream-copy mux. Temporal models add *boundary segments* where adjacent chunks'
   outputs are cross-faded, so chunks stay independent (multi-GPU friendly) and blending never crosses a
   scene cut.
6. **Inference backends behind one ABC** (`load / infer / max_batch_for / close`): Torch (reference),
   ONNX Runtime (CPU/CUDA/DirectML/ROCm), TensorRT (FP16, dynamic profiles, on-disk engine cache keyed by
   model hash + GPU + driver + TRT version + precision + profile).
7. **"Classical" pseudo-models** (`lanczos`, `bicubic`) share the pipeline; they give the Milestone 1
   no-op upscale and the quality baselines.
8. **Subtitle compatibility.** Audio is always stream-copied bit-exact. Subtitles are copied when the
   output container supports the codec; otherwise converted (e.g. MP4 `mov_text` → MKV `srt`, verified
   necessary: Matroska rejects `mov_text`) with a logged warning.
9. **Frame count is not assumed constant.** Stages pass `(index, pts)` pairs; a retime stage
   (frame generation) may emit a different number of frames on a new CFR clock. Chunks therefore carry
   both source ranges and output ranges, and the muxer only relies on the PTS carried in-band.
10. **One job layer, three front ends.** CLI, REST/web UI and the desktop GUI all build the same `JobSpec`
   and run the same pipeline, receiving progress through one callback/event interface.
11. **CUDA 13 stack locally** (torch cu130 + TRT 10.16 cu13 + ORT 1.31); nothing in the code is tied to a
   CUDA major version. Docker image pins the same stack.

## 4. Assumptions & limitations (initial)

- No CUDA toolkit locally: the optional `.cu` kernels are compiled only when CMake finds CUDA; GPU-side
  ops are verified through the Torch implementations.
- Only one GPU is available, so multi-GPU scheduling is tested with simulated devices / CPU workers.
- AMD/Intel inference paths (ROCm/DirectML) are implemented through ONNX Runtime providers but cannot be
  exercised on this machine; AMF/QSV/VAAPI encoders are capability-probed at runtime.
- NVDEC→CUDA zero-copy interop is out of scope for v1; decoded frames travel through pinned host memory
  (≈0.6 MB/frame at 480p, negligible next to inference).
- Weights are downloaded on first use; SHA256 values in the registry are pinned from the official release
  files.

## 5. Milestones

### M1 — Skeleton + I/O
- [ ] `pyproject.toml` (scikit-build-core + pybind11), CMake skeleton with a minimal `_native` module
      (proves the MSVC toolchain early), `.gitattributes`, ruff/mypy/pytest config
- [ ] `config.py` (pydantic-settings + YAML presets), `errors.py` (error taxonomy), `logging.py` (structlog)
- [ ] `io/ffmpeg.py` (binary discovery, runners), `io/probe.py` → typed `MediaInfo`
      (fps, time base, VFR, colour, bit depth, rotation, SAR/DAR, HDR, interlacing, frame count)
- [ ] `io/hwcaps.py` + `neuralframe doctor` (GPU, driver, CUDA/TRT/ORT, functional encoder/decoder probes)
- [ ] Output planning: target geometry (4K default; preset height / scale factor / exact W×H; DAR →
      square pixels, standard-ratio snapping, even dims), colour plan, encoder selection
- [ ] NumPy reference ops: YUV↔RGB (all matrices/ranges/subsamplings), separable Lanczos-3 resize
- [ ] Decoder (NUT pipe), encoder (NUT pipe, per-encoder rate-control mapping), muxer (copy + subtitle
      compat, `-copyts`, atomic rename)
- [ ] Threaded pipeline with bounded queues + per-stage metrics; `lanczos` classical model
- [ ] `neuralframe upscale` end-to-end; job report JSON
- [ ] Test fixtures generator (testsrc2 + tone + subtitles + chapters, scene cut, VFR, odd dims, 10-bit,
      interlaced, rotation, SAR, HDR tags)
- [ ] Tests: probe parsing, colour round-trips, resize vs reference, E2E (frame count, timestamps,
      bit-exact audio packets, A/V sync within one frame, colour tags)

### M2 — Inference core
- [ ] `models/registry.yaml` + loader, download with SHA256 verification, licence warnings
- [ ] Architectures in plain PyTorch: RRDBNet (x4plus, x2plus), SRVGGNetCompact (animevideov3)
- [ ] `InferenceBackend` ABC + TorchBackend (FP16 on CUDA, FP32 on CPU)
- [ ] Spatial tiling: fixed-size tile layout, feathered weights, NumPy reference split/merge
- [ ] `FrameUpscaler`: tiling + cross-frame batching + backend + padding to model multiples
- [ ] Pipeline integration (GPU device ops for colour/resize when on CUDA)
- [ ] Image upscaling path (`neuralframe upscale photo.png -o photo_4k.png`), 8/16-bit, alpha kept via
      Lanczos-resized alpha plane; cascaded passes for factors beyond the model's native scale
- [ ] Tests: tile layout/merge (no-seam test vs whole-frame), tiny random model E2E on CPU, image E2E

### M3 — Native module
- [ ] C++17 ops: YUV↔RGB, Lanczos resize, feather tile merge, temporal blend, pinned ring buffer
      (blocking acquire/release, GIL released while waiting); OpenMP/std::thread parallelism
- [ ] Optional CUDA kernels (built only when nvcc is found)
- [ ] GoogleTest C++ tests via CTest
- [ ] Parity tests native vs NumPy (≤ 1 LSB at 8-bit); pytest-benchmark speed-up numbers

### M4 — TensorRT + hardware I/O
- [ ] ONNX export (opset ≥ 17, dynamic batch/H/W) + parity check (max abs diff, PSNR)
- [ ] OnnxRuntimeBackend (CUDA/TRT/DirectML/ROCm/CPU EPs)
- [ ] TensorRTBackend: FP16 builder, dynamic profiles, engine cache + invalidation, timing cache
- [ ] NVDEC/NVENC (and AMF/QSV/VAAPI) paths with automatic software fallback
- [ ] CUDA-stream overlap (upload / compute / download) with measured before/after

### M5 — Chunking, VRAM manager, resume
- [ ] Scene detection (scdet / histogram) + chunk planner (pts ranges, scene alignment, overlap)
- [ ] Chunk store + manifest + resume verification; cancellation leaves resumable state
- [ ] VRAM calibration (batch × tile search, safety margin, `--max-vram-gb`, cached per model/GPU/res)
- [ ] OOM recovery ladder (halve batch → halve tile → next backend) without losing finished work
- [ ] Temporal model path: padded chunks, edge replication, boundary cross-fade segments
- [ ] Stretch: one temporal SR model (RealBasicVSR- or AnimeSR-family)
- [ ] Multi-GPU chunk scheduling (queue pull)
- [ ] Tests: planner, simulated OOM, kill-and-resume identical frame checksums

### M6 — Frame generation
- [ ] Analysis pass shared with scene detection: per-pair timestamp delta, downscaled-luma difference,
      scene score → drop map (gaps, duplicate runs) + unique-frame timeline
- [ ] Retime planner: CFR target grid (default 60, NTSC snapping), (frame0, frame1, t) per output frame,
      exact-match copies, scene-cut nearest-frame rule, source-faster-than-target decimation with warning
- [ ] `InterpolationBackend` + RIFE v4.x IFNet (Torch), ONNX/TensorRT export with timestep input,
      `minterpolate` fallback
- [ ] Pipeline integration: retime stage after upscaling (configurable order), chunk overlap of one
      frame on the right, resume-safe; job report gets drop/interpolation stats
- [ ] Tests: synthetic clips with injected drops (removed frames, duplicated frames, VFR gaps) →
      detected exactly; output frame count/timestamps CFR; A/V sync; no ghosting across scene cuts

### M7 — Job service + web UI
- [ ] SQLite job store (SQLAlchemy 2.0), priorities, restart-safe queue, one active job per GPU
- [ ] FastAPI: jobs CRUD, SSE events, cancel/resume, preview, models, system; path-root sandboxing
- [ ] `neuralframe queue add|list|cancel|resume`, `neuralframe serve`
- [ ] Single-page web UI (drop file, preset, live per-stage throughput, before/after slider)

### M8 — Desktop GUI application
- [ ] PySide6 app (`neuralframe-gui`): file picker + drag-and-drop for images and videos, input summary
      (resolution, fps, duration, detected drops after analysis)
- [ ] Mode: Upscale / Frame generation / Both (video); Upscale only (image)
- [ ] Upscale options: 4K default, preset heights, scale factor, exact W×H, model (auto by content),
      quality preset; frame-gen options: target fps (60 default, custom), drop repair toggle
- [ ] Runs the same pipeline in a worker thread; live progress bar, ETA, fps, stage, cancel/resume
- [ ] Output path chooser, before/after preview, open file / show in folder on completion
- [ ] GUI smoke tests (pytest-qt, offscreen)

### M9 — Quality, benchmarks, docs
- [ ] PSNR/SSIM (NumPy), LPIPS (optional), VMAF (libvmaf) vs bicubic/Lanczos baselines
- [ ] Seam test (chunk-boundary MAD ratio), tile-seam test (> 45 dB), flicker metric
- [ ] `benchmarks/run.py` → Markdown + JSON results with hardware info
- [ ] README (Mermaid diagram, quickstart pip + Docker, CLI/config reference, hardware matrix,
      troubleshooting, assumptions & limitations), ARCHITECTURE, BENCHMARKS, GALLERY, MODEL_CARDS
- [ ] Dockerfile + docker-compose verified with the NVIDIA runtime; CI workflow (CPU path)
- [ ] Reference case run + final acceptance checklist

## 6. Working agreement

- All work happens on `develop` (one or more commits per milestone); merging to `main` only on request.
- Gate for every milestone: `ruff check`, `mypy --strict` on core packages, `pytest -m "not gpu"`,
  then GPU tests locally, C++ tests once M3 lands.
- Simple + correct first (software decode, Torch, NumPy), keep it as the reference, then optimise and
  prove parity; every optimisation is measured before/after and recorded in `benchmarks/results/`.
- No hard-coded GPU names, paths or codec availability; every automatic choice is logged.
