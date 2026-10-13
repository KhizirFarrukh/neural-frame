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

### 1.2 Scope additions (requested 2026-10-13)

1. **More and stronger models**: the large transformer models (HAT-L, DRCT-L, DAT2, ATD, SwinIR-L) and
   strong CNNs (BSRGAN, RealPLKSR, Real-ESRGAN general v3, x4plus anime), plus any model file a user adds
   (OpenModelDB), built through spandrel.
2. **Quality complaint**: output looked "smoothed and smudged", not like a real high-resolution recording.
   Cause: every preset used Real-ESRGAN x4plus/x2plus, which are trained on heavy synthetic degradations
   and smooth clean footage. Goal: detail that looks recorded (plausible synthesis; nothing can recover
   detail that was never captured).
3. **Automatic model choice**: analyse the video and pick the model that suits it; the presets default to
   it.

### 1.3 Scope additions (requested 2026-10-13, second round)

Requests, in the user's order (kept here so none is lost):

1. Commit the model update (M10) - done as `57e8afa` on `develop` (not pushed).
2. Research more AI upscalers: models that turn very low-quality video and images (144p, 240p, 360p, 480p)
   into convincing 4K, and "hyper-realistic" generative models (findings: section 4, M11 research).
3. **Plain-language model names**: every list shows what the model is for, then its name in brackets
   (`HD anime to 4K, fastest - fast (2x-animejanai-hd-v3-superultracompact)`), so people who know nothing
   about models can choose by content and people who do can choose by name.
4. **A massive variety of models**, grouped by purpose: filmed video & photos; old, damaged or very low
   quality; cartoons & anime; AI images & art; games, textures & pixel art; text, documents & comics.
5. **No model is downloaded up front** - not by the developer, not by the release build. A model is
   downloaded when the user chooses (or automatic choice picks) it, before the job runs, with a progress
   bar showing details (MB done / total, speed, time left) and a way to cancel.
6. **Building the Windows app downloads no models by default** (opt-in flag to bundle some).
7. **Model manager** in the app (menu): every model with what it is for, size and status; a Download button
   on rows that are not downloaded; a progress bar on that row while it downloads; Cancel; Delete to free
   space.
8. **Use a locally stored model file**: a model the user already downloaded and keeps somewhere else can
   be linked instead of downloaded again (checked before use); folders to search can be configured.
9. **Updates**: models whose source supports it (Hugging Face files, GitHub release assets) are checked
   for a newer version; the manager shows "update available" and installs it on request.
10. **Make the app very user friendly** throughout.
11. Add all of this to the plan before working on it (this section and M11).

Standing constraints from the user: no test renders / GPU work while their own jobs run (ask first); do
not touch the `build/` folder while they use the app from it; delete the Docker image after testing that
it builds; never add AI attribution anywhere; push only when asked.

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
- The REST API has no authentication: it binds to 127.0.0.1 by default and only touches files inside the
  configured input/output roots (plus its upload/output folders). Exposing it on a network is the
  operator's decision (a warning is printed).
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

### Findings from Milestone 7

- A before/after comparison is only fair in one colorimetry: the job converts untagged SD (BT.601 matrix,
  SMPTE-C primaries) to BT.709, so the raw source frame looked colour-shifted next to the output. The
  preview converts the "before" frame with FFmpeg's `colorspace` filter (accurate mode) using the job's
  colour plan; red bar 252,0,0 vs output 247,18,0 became 245,18,0.
- SSE streams poll an in-memory version counter per job (0.2 s) instead of parking a thread per client;
  the store is written at most once a second per job plus on state changes.
- Through the service the reference job (480p -> 1080p + 59.94 fps, animevideov3) runs at 29.7 fps, the
  same as the CLI.

### Findings from Milestone 8

- Results cross from worker threads to Qt through a dispatcher `QObject` with a queued connection,
  parented to the window, so callbacks for a closed window are dropped (pytest-qt caught a preview that
  finished after teardown touching a deleted widget).
- Python unbinds `except ... as exc` names when the block ends, so error callbacks posted to the GUI
  thread bind the exception eagerly (`functools.partial`); a lambda would raise `NameError` instead.
- The reference job (480p -> 1080p + 59.94 fps, animevideov3) takes 21 s from the GUI, the same throughput
  as the CLI and the service (all three run the same `JobRunner`).

### Findings from Milestone 9

- Averaging per-frame PSNR lets identical frames dominate: a segment with a fade to black (22 pure black
  frames at a capped 100 dB) reported 49.9 dB for bicubic 4x; MSE pooled over all frames gives 36.2 dB.
  The evaluation pools MSE, and uses a fade-free segment with the letterbox cropped away.
- A raw chunk-boundary seam ratio is noisy with few boundaries and depends on what the content does at
  those frames; normalising by the same ratio of the chunk-free bicubic baseline gives 1.01-1.07 for every
  model (spec threshold 1.2).
- GAN models trade PSNR for perceptual quality: on compressed input they add 18-32 VMAF points and cut
  LPIPS by 25-30 % while PSNR stays at bicubic level. RealBasicVSR has the highest VMAF but the worst
  LPIPS and flicker (strong sharpening, invented texture) - VMAF alone would have ranked it first.
- The PyTorch cache cap from M6 subtracted the safety margin a second time and turned would-be spills into
  crashes (ONNX Runtime, RealBasicVSR and frame generation rows failed; TensorRT throughput halved from
  allocator thrash). The cap now keeps only 5 % (<= 512 MB) headroom; PyTorch OOMs around the backend
  call enter the recovery ladder, ONNX Runtime's arena grows by request instead of powers of two, and an
  unrecoverable OOM is reported as `OOM_UNRECOVERABLE` with a hint instead of an internal error.
- RealBasicVSR ran SPyNet and the cleaning network over the whole chunk at once (a single 2.7 GB
  allocation at 480p). Both are per-frame, so they now run four frames per call: identical output, peak
  memory independent of the chunk length; 480p -> 1080p now runs at 1.3 fps in 6.9 GB.
- Windows file names ignore case: `neuralframe.exe` silently overwrote `NeuralFrame.exe` in the release
  folder (the GUI self-test caught it); the command line is `neuralframe-cli.exe`.
- The Windows release is 5.3 GB unpacked (PyTorch 2.8 GB, TensorRT 2.2 GB); dropping TensorRT builder
  resources for datacenter GPUs and build-time files saves 0.9 GB; the portable zip is 3.7 GB. GitHub
  release assets are limited to 2 GB per file, so publishing there needs a split archive, the installer's
  disk spanning (Inno Setup) or the `--no-tensorrt` build.
- Ubuntu 24.04 images already have a `ubuntu` user with UID 1000; the container runs as that user.
- The Docker image works under Docker Desktop on Windows (WSL2 GPU paravirtualisation: TensorRT, NVDEC and
  NVENC all functional) but ran animevideov3 480p -> 1080p at 11.9 fps vs 35.2 fps natively, encode-bound;
  a native Linux host with the NVIDIA Container Toolkit could not be measured here. Building the 18.8 GB
  image next to a GPU job exhausted the 15 GB of RAM and crashed the Docker engine: build images alone.

### Findings from Milestone 10 (code; GPU measurements pending)

- spandrel 0.4 builds all the new architectures from their checkpoints and reproduces the project's own
  SRVGG and RRDBNet modules exactly (max difference 0.0 on the same weights), so only the new networks
  depend on it. It reports `supports_half = False` for HAT, DRCT, DAT, ATD and SwinIR: those run in BF16
  on GPUs with native bfloat16 and FP32 elsewhere.
- Cartoon detection by flat-area share alone called the Sintel trailer (CG animation, film-like shading)
  71 % animated. The ratio of hard (> 24 levels) to soft (2-12 levels) steps on a 320 px thumbnail
  separates them: 2D cartoon frames 0.41-2.2 (one flat-shaded scene 0.23 with 52 % flat blocks), live
  action 0.15-0.32, photo-real CG 0.11-0.16. CG films are therefore treated as filmed footage, which is
  also what the live-action models suit.
- 8x8 blocking measured at grid phase 0 only missed shifted or cropped videos; the strongest of the eight
  phases against their median is shift-invariant. Completely flat blocks (the worst case) left every
  other phase at zero and were skipped as "no data" until the ratio was floored.
- The real resolution matters more than the file's: 1080p upscaled from 480p needs a 4.5x model to reach
  4K, so the choice works on file factor x height / effective height. 2x models need two passes beyond
  about 2.5x and lose to a 4x model there.
- Still-image models (x4plus anime) won anime videos on rank alone; they are flagged `stills` and
  penalised for video, where AnimeVideo v3 is steadier.
- Users' model files come from the internet: only `.pth` (spandrel's restricted unpickler),
  `.safetensors` and `.ckpt` are accepted, never `.pt` (spandrel opens those as TorchScript, which can
  carry code). A broken user `models.yaml` is skipped entry by entry instead of breaking every command.

### Findings from Milestone 11 (code; no downloads, no GPU)

- OpenModelDB publishes licence, architecture, scale, size and SHA256 for 671 models; 212 are 2x+ colour
  upscalers spandrel can load with a direct download link. Spandrel 0.4.2 has no SRFormer, so SRFormer
  models were left out; one 2x ESRGAN variant (pixel-unshuffle) needs inputs that are multiples of 4 and
  DRCT needs 16 - the size guard pads any spandrel network to what it reports instead of trusting the
  registry.
- RealPLKSR, RCAN and the DAT/HAT/ATD/SwinIR/RGT families report no FP16 support in spandrel (BF16 is
  fine); AuraSR supports neither FP16 nor BF16, so the registry gained `bf16` and the backends fall back to
  FP32 for it. AuraSR is trained on 64 px patches: `max_tile` caps its tiles.
- `neuralframe models download` with no names used to fetch every model; with 65 models that is 5.2 GB,
  so it now needs names or an explicit `--all` (the error states the total).
- Hashing a 2.5 GB file takes seconds: status displays (desktop app, web, `models list`) trust a file of
  the right size until it is used; jobs always verify (cached per path, size and modification time).
- Linking an identical copy of a model must not change its identity (an earlier version gave it a new URL
  and file name, which would have rebuilt TensorRT engines); only a different file changes the weights in
  the job's model spec.

### Research: upscalers for very low-quality sources and generative restoration (2026-10-13)

What exists (sources checked 2026-10-13; licences from the projects' own pages where stated):

- **No model recovers detail that was never recorded.** Going from 144p-480p to 4K (4.5x-15x) means the
  model *invents* plausible detail. CNN/transformer upscalers (everything in M10) do this conservatively;
  generative (diffusion) restorers invent far more - faces, textures, text - which looks "real" but can
  change what is in the picture.
- **Video, one-step diffusion** (the current state of the art for badly degraded video):
  - SeedVR2 (ByteDance, ICLR 2026): Apache-2.0 code; 3B and 7B models. Official code is multi-GPU
    (one H100-80G for 100 frames of 720p; four for 1080p/2K) and needs flash-attn and apex. The community
    ComfyUI/standalone runner (Apache-2.0) adds GGUF Q4/Q8 and FP8 weights, block swapping to the CPU and
    VAE tiling, and documents runs on 8 GB GPUs (slow). An independent study (arXiv 2605.25940, 360p/720p
    -> 2160p) ranked SeedVR2 (7B), DOVE and Topaz Starlight Mini best, with no clear winner; SeedVR2 took
    about 11 s per 4K frame on a datacenter GPU and ghosted with small batches.
  - FlashVSR (CVPR 2026): Apache-2.0, Wan 2.1-based, streaming one-step 4x VSR, ~17 fps at 768x1408 on an
    A100; its sparse-attention kernel is tuned for A100/A800 and untested on RTX 40/50.
  - DOVE (NeurIPS 2025): Apache-2.0 code, CogVideoX 1.5-5B (also a 2B version); strong temporal
    consistency, ~18 s per 4K frame in the study above.
  - SwiftVR (June 2026): Wan 2.2-TI2V-5B based, 26 fps at 1080p on an RTX 5090, 38 GB peak at 1440p on an
    H100; code Apache-2.0 but **weights under a research licence** (commercial use needs permission).
  - Topaz Starlight (commercial, closed): Mini runs locally with >= 10 GB VRAM; reference point only.
- **Images, generative**: SUPIR (SDXL + 13B LLaVA, ~30 GB VRAM x2, non-commercial terms), HYPIR (SD 2.1
  LoRA, SIGGRAPH 2025, non-commercial), LucidFlux (Flux.1-dev based: Flux-dev non-commercial licence),
  OSEDiff (one-step, SD 2.1, Apache-2.0), InvSR (S-Lab non-commercial), AuraSR v2 (GigaGAN-style 4x GAN,
  600 M parameters, Apache-2.0, loadable by spandrel, FP32 only, trained on 64 px patches).
- **NVIDIA RTX Video Super Resolution for developers**: `nvidia-vfx` (Python, DLPack/PyTorch interop,
  Turing+ Tensor Cores, driver >= 570.65) exposes `VideoSuperRes` with 1-4x upscaling, quality levels,
  denoise/deblur and high-bitrate modes - the "DLSS-like" real-time option. Proprietary licence, terms of
  redistribution not stated on PyPI; would have to be an optional, user-installed component.
- **Faces**: GFPGAN-class restoration helps tiny faces most; CodeFormer and the video face model KEEP are
  S-Lab non-commercial. Needs face detection, alignment and paste-back - a separate pipeline stage.
- **Community CNN/transformer models (OpenModelDB, 671 entries with licences and SHA256)**: the best fits
  for low-quality sources are trained on realistic degradations (RealWebPhoto v4 DAT2/DRCT-L, Nomos web
  photo, VHS2HD, LiveAction SPAN, AniScale/Ani4K/AnimeJaNai for anime). These run in the existing pipeline
  through spandrel and are the M11 "variety" set.

Recommendation: M11 adds the community models and AuraSR v2 within the current pipeline. A generative
engine (M12) is worth it only as an optional, separately downloaded component: SeedVR2 3B (Apache-2.0,
quantised + block swap) is the most practical for an 8 GB card, FlashVSR the fastest on large GPUs;
SwiftVR and the image models above are excluded or optional because of their licences. Expect minutes per
4K frame on an 8 GB laptop GPU.

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
- [x] SQLite job store (SQLAlchemy 2.0, WAL), priority + FIFO, restart-safe queue (interrupted jobs are
      re-queued with resume), one worker / active job per GPU, throttled progress persistence
- [x] FastAPI (`neuralframe/api/`): jobs create/list/get/delete, SSE events, cancel/resume, preview
      (before/after PNG at output size and colorimetry), report, output download, uploads, probe, models,
      system; input/output roots enforced after resolving symlinks and `..`; binds 127.0.0.1 by default
- [x] `neuralframe serve`, `neuralframe queue add|list|cancel|resume|watch`
- [x] Single-page web UI: drop/upload or path, Upscale / Frame generation / Both (images: upscale only),
      4K default + custom targets, 60 fps default + custom, drop repair, live progress with per-stage
      throughput and VRAM, cancel/resume, download, before/after slider; driven in headless Edge by a
      Playwright test

### M8 — Desktop GUI application
- [x] PySide6 app (`neuralframe-gui`, also `neuralframe gui`; extra `[gui]`): file picker + drag-and-drop
      for images and videos, input summary with thumbnail, "Scan for dropped frames" (the job's cached
      analysis pass: frozen frames, gaps, scene cuts, intentional holds)
- [x] Mode: Upscale / Frame generation / Both (video); Upscale only (image)
- [x] Upscale options: 4K default, 8K/1440p/1080p/720p, 2x/3x/4x, custom (W×H, xF, NNNp), model (automatic
      from preset or any registered model; M10: automatic = content analysis), quality preset; frame generation: 60 fps default + presets +
      custom (59.94, 24000/1001 ...), drop repair (smart / every repeat / off)
- [x] Runs the same `JobRunner` in a worker thread; live progress bar, frames, fps, ETA, VRAM/GPU load,
      per-stage throughput, cancel (keeps finished chunks) and resume
- [x] Output name suggested from the task (`clip_upscaled_60fps.mkv`), save-as dialog, replace prompt,
      open result / show in folder, before/after compare view with frame selector
- [x] GUI tests (pytest-qt, offscreen): defaults, video and image flows, scan + frame generation,
      cancel/resume with chunk reuse, input validation

### M9 — Quality, benchmarks, docs, release
- [x] PSNR/SSIM (NumPy, SSIM cross-checked against an independent implementation), LPIPS (optional),
      VMAF (libvmaf) vs bicubic/Lanczos baselines on a real CC BY clip (`benchmarks/quality.py`)
- [x] Seam test (chunk-boundary change ratio normalised by the chunk-free baseline), tile-seam test
      (> 45 dB), flicker metric (temporal noise in static regions)
- [x] `benchmarks/run.py` + `neuralframe benchmark` → Markdown + JSON results with hardware info;
      `neuralframe compare` side-by-side crops
- [ ] Optional: RIFE ONNX/TensorRT export (timestep input) - not done; documented as a limitation
- [x] README (Mermaid diagram, quickstart Windows app / Docker / pip, CLI/config reference, hardware
      matrix, troubleshooting, assumptions & limitations), ARCHITECTURE, BENCHMARKS, GALLERY, MODEL_CARDS
- [x] Dockerfile + docker-compose; CI workflow (CPU path + C++ tests on Linux and Windows)
- [x] Windows release for non-technical users (`packaging/windows/build_release.py`): PyInstaller folder
      with `NeuralFrame.exe` + `neuralframe-cli.exe`, pinned FFmpeg, bundled models, smoke-tested as a user
      runs it, portable zip (+ Inno Setup installer when available)
- [x] Reference case run + final acceptance checklist

### M10 — More models, automatic model choice
- [x] spandrel loader (`architecture: spandrel`), registry fields `backends`, `fp16`, `memory_per_pixel`,
      `bundled`, `speed`, `content`, `handles`, `rank`, `stills`; BF16/FP32 for FP16-unsafe models;
      PyTorch-only models never sent to TensorRT/ONNX
- [x] 9 new checksum-pinned models: HAT-L, DRCT-L, DAT2 (real), ATD, SwinIR-L (real GAN), BSRGAN,
      RealPLKSR 2x, Real-ESRGAN general v3, x4plus anime 6B
- [x] Content analysis + recommendation (`models/advisor.py`), `-m auto` as every preset's model, reasons in
      the job report, `neuralframe recommend`, the choice and reasons shown live in the desktop app,
      "Automatic" in the web UI
- [x] Your own models: `neuralframe models add/remove`, "Add model..." in the desktop app, per-user
      registry (`models.yaml` in the settings folder), optional `--rank` for automatic choice
- [x] Release (spandrel/torchvision collected, large models download on first use, smoke test covers a
      spandrel model and `recommend`), Docker and CI install the `models` extra
- [x] Unit tests (detectors on synthetic images, choosing rules, user models, job integration) and
      integration tests (CLI, desktop app, CPU job with an added transformer, GPU job per family)
- [x] Quality evaluation extended: all models, automatic choice per preset, filmed reference (Tears of
      Steel)
- [ ] Measure speed, VRAM per pixel and quality of the new models on the GPU; calibrate `rank`,
      `memory_per_pixel` and the preset speed sets from the results
- [ ] Full gate (ruff, mypy, pytest incl. GPU, CTest); release build and Docker image build re-tested

### M11 — Model library: plain-language names, many more models, downloads on demand
Planned 2026-10-13 from section 1.3. Order of work: plan -> registry/naming -> download manager ->
model manager UI -> CLI/web -> build changes -> tests -> docs.
- [x] Registry: `purpose` (plain words) and `category` for every model; one label format everywhere:
      `purpose - speed (id)`; lists grouped by category; ids accepted case-insensitively (`4x_UltraSharpV2`)
- [x] 52 more models (65 AI upscalers in total) from OpenModelDB and Hugging Face, checksum-pinned, licences recorded; non-commercial
      ones marked "personal use only"; none chosen automatically until measured (rank 0)
- [x] Any model loaded through spandrel pads its input to the size multiple / minimum the network needs
      (some 2x ESRGAN variants need multiples of 4, DRCT of 16), so registry entries cannot get it wrong
- [x] Download manager (`models/library.py`): status per model (not downloaded / downloaded / linked local
      file / update available), streamed download with progress events (bytes, total, speed, ETA),
      cancellation, partial-file cleanup, SHA256 check, delete, disk usage
- [x] Jobs download a missing model first (also when automatic choice picked it) and report it as a
      "downloading model" stage with bytes / speed / time left; cancelling the job cancels the download
- [x] Local files: link a file the user already has (`models link ID FILE`, "Use a file I have..." in the
      manager); exact checksum = verified copy, otherwise accepted only if it loads as the same kind of
      network and scale ("local file, different version"); `models.search_paths` setting for folders
      to look in
- [x] Updates: for models hosted as Hugging Face files or GitHub release assets, compare the published
      SHA256 with the installed one; "Check for updates" / "Update" in the manager, `models check-updates`
      / `models update` on the command line; updated pins stored per user, never edited into the registry
- [x] Desktop app: Models menu -> Model manager window (search box, category filter, per-row status, size,
      Download / Cancel / Delete / Use a file I have... / Update, per-row progress bar with details, total
      disk use, "Check for updates"); the main window says when the selected model still has to be
      downloaded (size) and offers "Download now"; the job progress shows the download first
- [x] User friendliness: plain-language labels and tooltips everywhere, licence notices, clear errors with
      a retry (offline, disk full, checksum), sizes before downloading, nothing technical required
- [x] Command line: `models list` grouped by purpose with status; `models download/delete/link/unlink/
      check-updates/update`; rich progress bars
- [x] Web UI: grouped plain-language model list; job progress shows the model download
- [x] Windows release: no models unless `--with-models`; the smoke test uses the developer's model folder
      read-only through `models.search_paths` (skips neural checks when a model is missing) instead of
      downloading
- [x] Tests (CPU only, local `file://` "downloads" of tiny models): progress, cancel, checksum failure,
      link/unlink, search paths, update check against mocked APIs, manager window, job download stage
- [x] Docs: README (model manager, downloads, local files, updates, model list by purpose), model cards
      for every new model, packaged README.txt
- [x] Research write-up (generative upscalers for very low-quality sources) and a recommendation for M12
- [ ] Later, with the user's OK for GPU work: measure the new models, calibrate ranks so automatic choice
      can use them; full gate; release and Docker build tests (delete the Docker image afterwards)

### M12 — Generative restoration (proposed, needs the user's decision)
- [ ] One-step diffusion video restoration engine for very low-quality sources (candidates in section 4:
      SeedVR2 3B, FlashVSR, SwiftVR), optional download, quantised weights + block swapping for 8 GB GPUs
- [ ] Optional face restoration pass for small faces (GFPGAN-class, face detection + paste-back)

## 6. Working agreement

- All work happens on `develop` (one or more commits per milestone); merging to `main` only on request.
- Gate for every milestone: `ruff check`, `mypy --strict` on core packages, `pytest -m "not gpu"`,
  then GPU tests locally, C++ tests once M3 lands.
- Simple + correct first (software decode, Torch, NumPy), keep it as the reference, then optimise and
  prove parity; every optimisation is measured before/after and recorded in `benchmarks/results/`.
- No hard-coded GPU names, paths or codec availability; every automatic choice is logged.
