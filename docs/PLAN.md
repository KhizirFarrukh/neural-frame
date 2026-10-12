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
   - Model: RIFE v4.22 (arbitrary-timestep IFNet, MIT licence) in PyTorch (FP16); FFmpeg `minterpolate`
     as a model-free fallback.
   - Order when both are selected (`framegen.order`, default `auto`): heavy SR models run **upscale →
     interpolate** (SR sees each source frame once); classical resizing and compact models run
     **interpolate → upscale**, which measured 3.3× / 1.4× faster (see M6 findings). Configurable.
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
- Drop detection works on 64x36 thumbnails: a frozen frame is found when its neighbourhood moves. A drop
  inside a nearly static shot (only a small object moving) is indistinguishable from a still frame and is
  left alone; `framegen.repair_drops: on` repairs every repeat instead.

### Findings from Milestone 1 (measured, not assumed)

- **Colour tags go on the frames (`setparams`) and, identically, on the encoder options.** If frame
  properties and encoder options disagree, FFmpeg 9 inserts an implicit swscale pass during format
  negotiation (measured: full-range FFV1 squeezed to limited range; untagged frames + tagged options changed
  x264 output). Our samples are final, so nothing may be converted.
- **Segments must be encoded with `-copyts`** so each keeps absolute PTS; concatenation is then a plain
  packet copy (MP4 segments keep exact DTS; Matroska segments lose DTS and cannot be packet-concatenated).
- **Matroska cannot hold `mov_text`**: MP4 → MKV jobs convert text subtitles (warning logged); audio is
  always stream-copied bit-exact or the job is refused.
- **NumPy SIMD transcendental functions are alignment-dependent** (an ULP between runs). Kernel weights and
  transfer curves are built with scalar `math` into cached tables shared with the native module.
- **x264 lossy decisions vary slightly under heavy CPU contention** even with identical input and
  `-threads 1` (NVENC does not). Kill-and-resume bit-identity therefore holds for NVENC and for lossless
  encodes; with x264/x265 the resumed output is valid and in sync but may differ by a few bytes.
- `idet` misreports synthetic progressive content as interlaced, so `deinterlace=auto` trusts a
  `progressive` container tag and only runs `idet` when the field order is unknown.
- YAML 1.1 reads unquoted `on`/`off` as booleans; option validators map them back.

### Findings from Milestone 2

- PyTorch eager FP16 baselines on the RTX 4060 Laptop (8 GB), deterministic cuDNN: Real-ESRGAN x4plus
  0.25 Mpx/s (≈0.6 fps at 480p, 8.8 KB VRAM per input pixel), x2plus 1.2 Mpx/s (2.2 KB/px),
  animevideov3 7.7 Mpx/s (≈19 fps at 480p, 0.55 KB/px). Reference case 480p→1080p with x4plus:
  300 frames in 505 s, GPU utilisation 99.5 %, peak VRAM 5.4 GB → inference-bound as intended.
- Device mode (torch ops for colour/resize on the GPU) is the default whenever torch is installed and a
  neural model or CUDA is used; host mode (CPU stages) is kept for CPU-only/classical runs and is
  pixel-equivalent within 1 LSB.
- FFmpeg applies JPEG EXIF orientation through *frame* side data, invisible in stream-level probe output;
  the probe reads the first frame's display matrix for still images.

### Findings from Milestone 4

- TensorRT 10.16 FP16 vs PyTorch eager FP16 (cuDNN benchmark), 480×856 whole frames, RTX 4060 Laptop:
  animevideov3 52.8 → 39.8 ms (1.3×), x2plus 344 → 159 ms (2.2×), x4plus 1550 → 590 ms (2.6×);
  parity vs Torch 58–70 dB PSNR. One-time engine builds: 71 s / 122 s / 203 s (cached afterwards).
- Dense-matrix resizing wasted ~7× FLOPs on banded Lanczos weights; banded block matmuls are 4.8×
  faster (87.8 → 18.3 ms per 8 frames at 3416×1920 → 1080p) with identical output.
- CUDA-stream overlap (upload / inference / download stages, pinned ring-buffer staging): 25.5 → 28.7 fps
  before the resize fix; after it 34.5 → 35.5 fps (GPU already ~82 % busy). animevideov3 480p → 1080p now
  runs faster than real time (29.97 fps source).
- Hardware encoders can pass a 320×240 capability probe yet fail on the real job (size/session limits),
  so planning runs a 2-frame preflight with the real settings before the fingerprint is fixed.

### Findings from Milestone 5

- Thumbnail analysis (64x36 through the same filter chain) finds scene cuts with a histogram + motion
  test: the reference clip's cut at frame 150 scores 0.51 against <= 0.06 for motion-only frames.
- Resume seeks to just before the last committed frame and skips by exact PTS (frames are delivered in
  index order, so finished chunks always form a prefix); temporal denoise filters fall back to decoding
  from the start because their state would differ after a seek.
- OOM ladder: batch halvings (always converge) do not count against the three tile-reduction levels;
  calibration pins batch/tile in the manifest so resumed chunks are bit-identical.
- RealBasicVSR in FP16 with SPyNet kept in FP32: 49.8 dB vs FP32 (44.7 dB all-FP16) at the same speed;
  the 4x head runs in strips with a 4-row halo to bound memory (matches whole-frame within 1e-4).
- Temporal chunks cross-fade over `overlap` frames at non-cut boundaries; a resumed job re-runs the last
  finished chunk without emitting it to rebuild the shared blend zone, keeping output bit-identical.

### Findings from Milestone 6

- RIFE 4.22 re-implemented in plain PyTorch loads the released `flownet.pkl` unchanged. On a panning test
  (50 fps truth, every other frame regenerated from a 25 fps copy) RIFE reaches 36.6 dB PSNR against
  26.6 dB for a cross-fade; copied frames are bit-exact. A zero-initialised narrow IFNet is an exact
  midpoint blend, which makes CI tests deterministic without downloading weights.
- RIFE FP16 throughput on the RTX 4060 Laptop (`benchmarks/results/framegen-*.md`): 720p 45 fps, 1080p
  19.5 fps, 1440p 15.6 fps (flow scale 0.5), 4K 6.8 fps (0.5); FP16 is ~2× FP32. Batching several
  timesteps of one pair gives no speed-up (the GPU is already saturated) but multiplies VRAM: 4 timesteps
  at 4K ran out of memory, and at 1440p FP32 the Windows driver spilled into shared memory (3.2 s instead
  of 0.73 s per call). Timestep batches are now sized from free VRAM and halved on OOM.
- PyTorch's caching allocator keeps a separate cache per CUDA stream; with upload / inference / download /
  RIFE on their own streams it grew to fill the 8 GB card (6.8 GB reserved for a 4.1 GB working set) and
  the driver spilled to system memory. The job now caps the cache at the memory free when the pipeline
  starts minus the safety margin (lifted afterwards), and SR planning/calibration leaves RIFE's measured
  footprint free: 480p → 1080p + 59.94 fps went from 53.6 s (11.2 fps) to 28.4 s (21.1 fps), no OOMs.
- Order, reference clip (300 → 600 frames, 59.94 fps): animevideov3 → 1080p upscale-first 21.9 fps vs
  interpolate-first 29.8 fps; Lanczos → 4K 10.4 vs 34.2 fps. Interpolating first adds one SR pass per
  synthesised frame and saves RIFE(output) − RIFE(source), so `auto` interpolates first only for classical
  and compact models; RRDBNet models (≈1.7 s/frame for x4plus) upscale first.
- FFmpeg `minterpolate` stops at the last input frame; the fallback clones one frame past the end
  (`tpad`) and trims the grid there, so both engines give the same frame count.
- Matroska stores no per-stream duration and its container duration includes audio padding (21 ms for
  AAC, more than one 60 fps frame); verification measures the video span from the packet scan.

## 5. Milestones

### M1 — Skeleton + I/O
- [x] `pyproject.toml` (scikit-build-core + pybind11), CMake skeleton with a minimal `_native` module
      (proves the MSVC toolchain early), `.gitattributes`, ruff/mypy/pytest config
- [x] `config.py` (pydantic-settings + YAML presets), `errors.py` (error taxonomy), `logging.py` (structlog)
- [x] `io/ffmpeg.py` (binary discovery, runners), `io/probe.py` → typed `MediaInfo`
      (fps, time base, VFR, colour, bit depth, rotation, SAR/DAR, HDR, interlacing, frame count)
- [x] `io/hwcaps.py` + `neuralframe doctor` (GPU, driver, CUDA/TRT/ORT, functional encoder/decoder probes)
- [x] Output planning: target geometry (4K default; preset height / scale factor / exact W×H; DAR →
      square pixels, standard-ratio snapping, even dims), colour plan, encoder selection
- [x] NumPy reference ops: YUV↔RGB (all matrices/ranges/subsamplings), separable Lanczos-3 resize
- [x] Decoder (NUT pipe), encoder (NUT pipe, per-encoder rate-control mapping), muxer (copy + subtitle
      compat, `-copyts`, atomic rename)
- [x] Threaded pipeline with bounded queues + per-stage metrics; `lanczos` classical model
- [x] `neuralframe upscale` end-to-end; job report JSON
- [x] Test fixtures generator (testsrc2 + tone + subtitles + chapters, scene cut, VFR, odd dims, 10-bit,
      interlaced, rotation, SAR, HDR tags)
- [x] Tests: probe parsing, colour round-trips, resize vs reference, E2E (frame count, timestamps,
      bit-exact audio packets, A/V sync within one frame, colour tags)

### M2 — Inference core
- [x] `models/registry.yaml` + loader, download with SHA256 verification, licence warnings
- [x] Architectures in plain PyTorch: RRDBNet (x4plus, x2plus), SRVGGNetCompact (animevideov3)
- [x] `InferenceBackend` ABC + TorchBackend (FP16 on CUDA, FP32 on CPU)
- [x] Spatial tiling: fixed-size tile layout, feathered weights, NumPy reference split/merge
- [x] `FrameUpscaler`: tiling + cross-frame batching + backend + padding to model multiples
- [x] Pipeline integration (GPU device ops for colour/resize when on CUDA)
- [x] Image upscaling path (`neuralframe upscale photo.png -o photo_4k.png`), 8/16-bit, alpha kept via
      Lanczos-resized alpha plane; cascaded passes for factors beyond the model's native scale
- [x] Tests: tile layout/merge (no-seam test vs whole-frame), tiny random model E2E on CPU, image E2E

### M3 — Native module
- [x] C++17 ops: YUV↔RGB, Lanczos resize, feather tile merge, temporal blend, pinned ring buffer
      (blocking acquire/release, GIL released while waiting); persistent thread pool
- [x] ~~Optional CUDA kernels~~ → GPU-side ops are PyTorch tensor ops (`ops/torch_ops.py`), parity-tested
      on CUDA; no CUDA toolkit is needed to build (none is installed here to verify hand-written kernels)
- [x] GoogleTest C++ tests via CTest (14 tests)
- [x] Parity tests native vs NumPy: **bit-identical** (stricter than the ≤ 1 LSB requirement); speed-up
      19–34× per op, Lanczos 1080p pipeline 2.9 → 45 fps in host mode (`benchmarks/results/ops-*.md`)
- [x] Ring buffer wired into the GPU upload/download stages as page-locked staging (M4)

### M4 — TensorRT + hardware I/O
- [x] ONNX export (opset ≥ 17, dynamic batch/H/W) + parity check (max abs diff, PSNR)
- [x] OnnxRuntimeBackend (CUDA/TRT/DirectML/ROCm/CPU EPs)
- [x] TensorRTBackend: FP16 builder, dynamic profiles, engine cache + invalidation, timing cache
- [x] NVDEC/NVENC (and AMF/QSV/VAAPI) paths with automatic software fallback
- [x] CUDA-stream overlap (upload / compute / download) with measured before/after

### M5 — Chunking, VRAM manager, resume
- [x] Scene detection (scdet / histogram) + chunk planner (pts ranges, scene alignment, overlap)
- [x] Chunk store + manifest + resume verification; cancellation leaves resumable state
- [x] VRAM calibration (batch × tile search, safety margin, `--max-vram-gb`, cached per model/GPU/res)
- [x] OOM recovery ladder (halve batch → halve tile → next backend) without losing finished work
- [x] Temporal model path: padded chunks, edge replication, boundary cross-fade segments
- [x] Stretch: one temporal SR model (RealBasicVSR- or AnimeSR-family)
- [x] Multi-GPU chunk scheduling (queue pull)
- [x] Tests: planner, simulated OOM, kill-and-resume identical frame checksums

### M6 — Frame generation
- [x] Analysis pass shared with scene detection: per-pair timestamp delta, downscaled-luma difference,
      scene score → drop map (gaps, sporadic repeats in moving content; regular holds such as animation
      on twos are kept in `auto` mode)
- [x] Retime planner: CFR target grid (default 60, NTSC snapping), (frame0, frame1, t) per output frame,
      exact-match copies, scene-cut and long-gap nearest-frame rule, source-faster-than-target selection
      with a note
- [x] RIFE 4.22 IFNet (Torch, FP16, VRAM-sized timestep batches, OOM split) + `minterpolate` fallback;
      ~~ONNX/TensorRT export~~ deferred: RIFE's per-call shapes change with the flow scale and the stage is
      GPU-bound alongside SR, so a TRT engine needs its own profile set and benchmark (listed in M9)
- [x] Pipeline integration: retime stage before or after super-resolution (`auto` order by model cost),
      framegen-only passthrough (no pixel processing, copies bit-exact), output-indexed chunks, resume by
      decode-skip, report section with drop/interpolation stats; CLI `--interpolate/--fps/--no-upscale`
      and `neuralframe interpolate`
- [x] Tests: synthetic clips with removed frames, frozen frames and VFR gaps → detected exactly; output
      frame count and timestamps CFR; A/V sync; nearest frame (no blend) across scene cuts; both orders;
      kill-and-resume identical; real RIFE beats blending on CUDA

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
- [ ] Optional: RIFE ONNX/TensorRT export (timestep input) if it beats PyTorch FP16 inside the pipeline
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
