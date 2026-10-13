# Model cards

Model files are never in the source code or, by default, in the Windows app: each one is downloaded
from its authors' URL when it is first chosen (or from the app's Models window / `neuralframe models
download`), checked against the pinned SHA256 and kept in `<cache>/models/<id>/`. A Windows build made
with `--with-models` includes the models marked *bundled*. Files you already have can be used instead
(`neuralframe models link`, or a folder in `paths.model_search_paths`), and models hosted as Hugging Face
files or GitHub release assets can be checked for updates. The registry lives in
`neuralframe/data/registry.yaml`; models you add yourself (`neuralframe models add`) are recorded in
`models.yaml` in your settings folder. `neuralframe models export <id>` builds the ONNX file and checks
TensorRT/ONNX Runtime parity against PyTorch (not for models loaded through spandrel, which run in PyTorch
only).

Speeds are end-to-end (decode -> model -> encode) on an RTX 4060 Laptop GPU (8 GB) from
`docs/BENCHMARKS.md`; quality numbers come from `docs/BENCHMARKS.md#quality` (Sintel trailer, CC BY 3.0).
Models without numbers have not been measured yet.

## Automatic model choice

`-m auto` (the default in every preset) analyses the input - cartoon or filmed, noise, compression
(blocking and bits per pixel), real detail versus file size - and scores every single-frame model with a
`rank`, keeping only those whose `speed` the preset allows (`fast`: fast; `balanced`: up to slow;
`quality`: all) and whose `content` matches. Points are subtracted for each problem found that the model
does not remove (`handles`), for restoration models on clean input (they smooth it), for still-image models
on video, and for 2x models when the job needs more than about 2.5x. The registry fields, per model:

| Field | Meaning |
|---|---|
| `speed` | `fast`, `medium`, `slow`, `very-slow` |
| `content` | `live-action` (filmed and computer-animated footage) and/or `animation` (2D cartoons, anime) |
| `handles` | degradations it removes: `noise`, `compression`, `blur` (empty: expects a clean source) |
| `rank` | relative output quality for its content, 1-100 (0 = never chosen automatically) |
| `stills` | made for still images (less stable between video frames) |
| `backends`, `fp16`, `bundled` | allowed inference backends, whether FP16 is safe, included in the Windows app |

The ranks are a first calibration from published results and spot checks; `benchmarks/quality.py` measures
every model on animated (Sintel) and filmed (Tears of Steel) references to refine them.

## All models by purpose

What each model is for, its name (use it with `-m`), size and licence. *Rank* > 0: used by automatic choice (measured models); the others are chosen by you. **NC** = non-commercial licence (personal use only). Detailed cards for the core models follow below.

### Filmed video & photos

| Use it for | Name | Scale | Speed | Size | Licence | Rank | Source |
|---|---|---|---|---|---|---|---|
| Any video, quick all-rounder | `realesr-general-x4v3` | 4x | fast | 5 MB | BSD-3-Clause | 60 | [link](https://github.com/xinntao/Real-ESRGAN) |
| Good-quality video to 2x, e.g. 1080p to 4K | `realplksr-x2` | 2x | medium | 30 MB | Apache-2.0 | 82 | [link](https://github.com/Phhofm/models/releases/tag/2xPublic_realplksr_dysample_layernorm_real) |
| Clean filmed video & photos, most detail | `hat-l-x4` | 4x | very slow | 165 MB | CC-BY-4.0 | 95 | [link](https://github.com/Phhofm/models/releases/tag/4xBHI_small_hat-l) |
| Clean filmed video & photos, largest model | `drct-l-x4` | 4x | very slow | 243 MB | CC-BY-4.0 | 94 | [link](https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_drct-l) |
| Clean filmed video & photos, high detail | `atd-x4` | 4x | very slow | 82 MB | CC-BY-4.0 | 90 | [link](https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_atd) |
| Clean photos & video, very high detail | `4x-nomos2-hq-dat2` | 4x | very slow | 140 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Nomos2-hq-dat2) |
| Clean photos & video | `4x-nomos2-hq-mosr` | 4x | medium | 17 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Nomos2-hq-mosr) |
| Clean photos, natural look | `4x-lsdir` | 4x | slow | 67 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-LSDIR) |
| Everyday photos (people, pets, events) | `4x-purephoto-span` | 4x | fast | 9 MB | CC-BY-SA-4.0 | - | [link](https://openmodeldb.info/models/4x-PurePhoto-span) |
| Everyday photos (people, pets, events), more detail | `4x-purephoto-realplksr` | 4x | medium | 30 MB | CC-BY-SA-4.0 | - | [link](https://openmodeldb.info/models/4x-PurePhoto-RealPLSKR) |

### Old, damaged or very low quality

| Use it for | Name | Scale | Speed | Size | Licence | Rank | Source |
|---|---|---|---|---|---|---|---|
| Damaged filmed footage & photos | `realesrgan-x4plus` | 4x | slow | 67 MB | BSD-3-Clause | 65 | [link](https://github.com/xinntao/Real-ESRGAN) |
| Damaged filmed footage & photos (2x) | `realesrgan-x2plus` | 2x | slow | 67 MB | BSD-3-Clause | 55 | [link](https://github.com/xinntao/Real-ESRGAN) |
| Badly degraded, blurry or noisy footage | `bsrgan-x4` | 4x | slow | 67 MB | Apache-2.0 | 70 | [link](https://github.com/cszn/BSRGAN) |
| Real-world footage with noise & compression, best repair | `dat2-real-x4` | 4x | very slow | 140 MB | CC-BY-4.0 | 92 | [link](https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real) |
| Heavily degraded footage, strong repair | `swinir-l-x4` | 4x | very slow | 142 MB | Apache-2.0 | 85 | [link](https://github.com/JingyunLiang/SwinIR) |
| Noisy real-world video, steady between frames | `realbasicvsr-x4` | 4x | slow | 148 MB | Apache-2.0 | 50 | [link](https://github.com/ckkelvinchan/RealBasicVSR) |
| Low-quality web, phone & social-media photos and video | `4x-realwebphoto-v4-dat2` | 4x | very slow | 140 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-RealWebPhoto-v4-dat2) |
| Very low-quality web & phone footage, strongest repair | `4x-realwebphoto-v4-drct-l` | 4x | very slow | 243 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-RealWebPhoto-v4-drct-l) |
| Web & phone photos and video, high detail | `4x-nomoswebphoto-atd` | 4x | very slow | 82 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-NomosWebPhoto-atd) |
| Web & phone photos and video | `4x-nomoswebphoto-realplksr` | 4x | medium | 30 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-NomosWebPhoto-RealPLKSR) |
| Damaged photos & video (blur, noise, JPEG) | `4x-nomos2-otf-esrgan` | 4x | slow | 34 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Nomos2-otf-esrgan) |
| Compressed photos & video | `4x-nomos2-realplksr-dysample` | 4x | medium | 30 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Nomos2-realplksr-dysample) |
| JPEG-damaged photos, high detail | `4x-nomos8k-atd-jpg` | 4x | very slow | 82 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Nomos8k-atd-jpg) |
| Blurry or JPEG photos (HAT) | `4x-nomos8kschat-s` | 4x | very slow | 81 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Nomos8kSCHAT-S) |
| Blurry or JPEG photos (ESRGAN) | `4x-nomos8ksc` | 4x | slow | 67 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Nomos8kSC) |
| Photos with noise, blur & JPEG | `4x-lsdirplusr` | 4x | slow | 67 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-LSDIRplusR) |
| JPEG-compressed photos | `4x-lsdirplusc` | 4x | slow | 67 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-LSDIRplusC) |
| Compressed TV, film & streaming video (2x) | `2x-liveactionv1-span` | 2x | fast | 9 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-LiveActionV1-SPAN) |
| VHS tapes & old TV recordings (2x) | `2x-vhs2hd` | 2x | medium | 30 MB | CC-BY-SA-4.0 | - | [link](https://openmodeldb.info/models/2x-VHS2HD) |
| Degraded real-world photos (SwinIR-M) | `swinir-m-x4` | 4x | very slow | 67 MB | Apache-2.0 | - | [link](https://openmodeldb.info/models/4x-realSR-BSRGAN-DFO-s64w8-SwinIR-M-x4-GAN) |
| Degraded real-world photos, 2x (SwinIR-M) | `swinir-m-x2` | 2x | very slow | 67 MB | Apache-2.0 | - | [link](https://openmodeldb.info/models/2x-realSR-BSRGAN-DFO-s64w8-SwinIR-M-x2-GAN) |
| Badly degraded footage, 2x (BSRGAN) | `bsrgan-x2` | 2x | slow | 67 MB | Apache-2.0 | - | [link](https://openmodeldb.info/models/2x-BSRGAN) |
| Tiny clean images, 8x in one step (smooth, no repair) | `swinir-m-classical-x8` | 8x | very slow | 68 MB | Apache-2.0 | - | [link](https://openmodeldb.info/models/8x-classicalSR-DF2K-s64w8-SwinIR-M) |

### Cartoons & anime

| Use it for | Name | Scale | Speed | Size | Licence | Rank | Source |
|---|---|---|---|---|---|---|---|
| Cartoons & anime video | `realesr-animevideov3` | 4x | fast | 3 MB | BSD-3-Clause | 75 | [link](https://github.com/xinntao/Real-ESRGAN) |
| Anime pictures & illustrations | `realesrgan-x4plus-anime` | 4x | medium | 18 MB | BSD-3-Clause | 80 | [link](https://github.com/xinntao/Real-ESRGAN) |
| Anime, sharpest lines (2x) | `2x-animesharpv4` | 2x | slow | 31 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-AnimeSharpV4) |
| Anime, sharp lines, faster (2x) | `2x-animesharpv4-fast` | 2x | slow | 31 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-AnimeSharpV4-Fast-RCAN-PU) |
| HD anime to 4K, real-time capable (2x) | `2x-animejanai-hd-v3-compact` | 2x | fast | 5 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-Compact) |
| HD anime to 4K, faster (2x) | `2x-animejanai-hd-v3-ultracompact` | 2x | fast | 2 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-UltraCompact) |
| HD anime to 4K, fastest (2x) | `2x-animejanai-hd-v3-superultracompact` | 2x | fast | 0 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-AnimeJaNai-HD-V3-SuperUltraCompact) |
| Anime from DVD & web releases (2x) | `2x-aniscale-2-esrgan` | 2x | slow | 67 MB | CC-BY-NC-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-AniScale-2-ESRGAN) |
| Anime from DVD & web releases (2x) | `2x-aniscale-2-omnisr` | 2x | fast | 3 MB | CC-BY-NC-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-AniScale-2-OmniSR) |
| Modern anime, Blu-ray to poor web rips (2x) | `2x-ani4vk-v2-compact` | 2x | fast | 2 MB | CC-BY-NC-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-Ani4VK-v2-Compact) |
| Anime with compression, noise & blur (4x) | `4x-hfa2k-ludvae-realplksr` | 4x | medium | 30 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-HFA2k-ludvae-realplksr-dysample) |
| Anime with compression, noise & blur, small transformer (4x) | `4x-hfa2k-ludvae-swinir-light` | 4x | slow | 17 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-HFA2kLUDVAESwinIR-light) |
| Anime video with H.264 compression (2x) | `2x-hfa2k-avc-compact` | 2x | fast | 5 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/2x-HFA2kAVCCompact) |
| Modern western cartoons & animation (2x) | `2x-modernspanimation-v1` | 2x | fast | 16 MB | MIT | - | [link](https://openmodeldb.info/models/2x-ModernSpanimationV1) |
| Flat-colour 2D cartoons (2x) | `2x-starsample-v2` | 2x | slow | 67 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-StarSample-V2) |
| Flat-colour 2D cartoons (2x) | `2x-starsample-v2-lite` | 2x | fast | 9 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/2x-StarSample-V2-Lite) |
| Anime from DVD, soft look (4x) | `4x-drawimation-compact` | 4x | fast | 3 MB | CC-BY-SA-4.0 | - | [link](https://openmodeldb.info/models/4x-Drawimation-compact) |
| Anime films (4x) | `4x-anifilm-compact` | 4x | fast | 3 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/4x-anifilm-compact) |
| Grainy hand-drawn animation from old DVDs (2x) | `2x-digitalfilm-v5-lite` | 2x | medium | 8 MB | WTFPL | - | [link](https://openmodeldb.info/models/2x-DigitalFilmV5-Lite) |

### AI images & digital art

| Use it for | Name | Scale | Speed | Size | Licence | Rank | Source |
|---|---|---|---|---|---|---|---|
| Pictures of any kind, very sharp (photos, art, anime) | `4x-ultrasharpv2` | 4x | very slow | 140 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/4x-UltraSharpV2) |
| AI-generated images (Stable Diffusion, Midjourney...) | `4x-lexicahat` | 4x | very slow | 85 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-LexicaHAT) |
| AI-generated images, 2x | `2x-lexicaswinir` | 2x | very slow | 59 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/2x-LexicaSwinIR) |
| Faces in paintings, drawings & AI art | `4x-artfaces-realplksr` | 4x | medium | 30 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-ArtFaces-realplksr-dysample) |
| AI images & photos: invents fine, realistic detail | `aurasr-v2` | 4x | very slow | 2470 MB | Apache-2.0 | - | [link](https://huggingface.co/fal/AuraSR-v2) |

### Games, textures & pixel art

| Use it for | Name | Scale | Speed | Size | Licence | Rank | Source |
|---|---|---|---|---|---|---|---|
| Old game textures, best quality | `4x-pbrify-v4` | 4x | very slow | 140 MB | CC0-1.0 | - | [link](https://openmodeldb.info/models/4x-PBRify-UpscalerV4) |
| Old game textures | `4x-pbrify-span-v4` | 4x | fast | 9 MB | CC0-1.0 | - | [link](https://openmodeldb.info/models/4x-PBRify-UpscalerSPANV4) |
| Realistic game textures | `4x-textures-gtav-rgt-s` | 4x | very slow | 136 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-Textures-GTAV-rgt-s) |
| Game textures & renders (2x) | `2x-aomr-mosr` | 2x | medium | 17 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/2x-AoMR-mosr) |
| Pixel art & MS Paint drawings (8x) | `8x-ms-unpainter` | 8x | slow | 67 MB | CC-BY-NC-SA-4.0 (NC) | - | [link](https://openmodeldb.info/models/8x-MS-Unpainter) |

### Text, documents & comics

| Use it for | Name | Scale | Speed | Size | Licence | Rank | Source |
|---|---|---|---|---|---|---|---|
| Low-quality text, screenshots & scans (2x) | `2x-text2hd` | 2x | medium | 30 MB | CC-BY-SA-4.0 | - | [link](https://openmodeldb.info/models/2x-Text2HD-v-1) |
| Printed photos & comics: removes halftone dots | `4x-dwtp-ds-dat2-v3` | 4x | very slow | 140 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-DWTP-DS-dat2-v3) |
| Manga & comic scans | `4x-wtp-uds-esrgan` | 4x | slow | 67 MB | CC-BY-4.0 | - | [link](https://openmodeldb.info/models/4x-WTP-UDS-Esrgan) |

### Plain resizing (no AI)

| Use it for | Name | Scale | Speed | Size | Licence | Rank | Source |
|---|---|---|---|---|---|---|---|
| Plain resize, sharp edges | `lanczos` | 1x | fast | - | n/a | - | - |
| Plain resize, soft | `bicubic` | 1x | fast | - | n/a | - | - |

## Super-resolution: core models

### hat-l-x4

| | |
|---|---|
| Purpose | The most detail when upscaling clean, good-quality filmed footage and photos 4x |
| Architecture | HAT-L (Hybrid Attention Transformer, large), 40.8 M parameters |
| Source | Architecture: Chen et al., "Activating More Pixels in Image Super-Resolution Transformer", CVPR 2023 - https://github.com/XPixelGroup/HAT. Weights `4xBHI_small_hat-l` trained by Philip Hofmann - https://github.com/Phhofm/models/releases/tag/4xBHI_small_hat-l |
| Licence | CC BY 4.0 (weights; credit the author when redistributing them) |
| Weights | `4xBHI_small_hat-l.safetensors`, 165.3 MB, SHA256 `054c060b...4e72eb53`; downloaded on first use |
| Backends | PyTorch only, through spandrel; BF16 on GPUs with native bfloat16 (RTX 30-series and newer), FP32 otherwise (FP16 overflows) |
| VRAM | planning estimate 16 KB per input pixel; frames are tiled to fit |
| Automatic choice | live-action, very slow, rank 95, removes nothing: chosen for clean filmed sources with the quality preset |
| Limitations | Expects clean input: it sharpens noise and compression blocks along with real detail. Very slow |

### drct-l-x4

| | |
|---|---|
| Purpose | Top detail for clean, photography-like footage (the largest model here) |
| Architecture | DRCT-L (Dense Residual Connected Transformer, large), 27.6 M parameters; input padded to a multiple of 16 |
| Source | Hsu et al., "DRCT: Saving Image Super-Resolution away from Information Bottleneck", CVPR Workshops 2024 - https://github.com/ming053l/DRCT. Weights `4xNomos2_hq_drct-l` by Philip Hofmann - https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_drct-l |
| Licence | CC BY 4.0 (weights) |
| Weights | `4xNomos2_hq_drct-l.safetensors`, 242.5 MB, SHA256 `1a106321...43477797`; downloaded on first use |
| Backends | PyTorch only (spandrel); BF16 / FP32 as for HAT-L |
| VRAM | planning estimate 16 KB per input pixel |
| Automatic choice | live-action, very slow, rank 94, removes nothing |
| Limitations | No degradation removal; very slow |

### atd-x4

| | |
|---|---|
| Purpose | High detail for clean footage at about half the size of HAT-L / DRCT-L |
| Architecture | ATD (Adaptive Token Dictionary transformer), 20.3 M parameters |
| Source | Zhang et al., "Transcending the Limit of Local Window: Advanced Super-Resolution Transformer with Adaptive Token Dictionary", CVPR 2024 - https://github.com/LabShuHangGU/Adaptive-Token-Dictionary. Weights `4xNomos2_hq_atd` by Philip Hofmann - https://github.com/Phhofm/models/releases/tag/4xNomos2_hq_atd |
| Licence | CC BY 4.0 (weights) |
| Weights | `4xNomos2_hq_atd.safetensors`, 81.7 MB, SHA256 `0fc5d114...93df39a6`; downloaded on first use |
| Backends | PyTorch only (spandrel); BF16 / FP32 |
| VRAM | planning estimate 14 KB per input pixel |
| Automatic choice | live-action, very slow, rank 90, removes nothing |
| Limitations | No degradation removal; very slow |

### dat2-real-x4

| | |
|---|---|
| Purpose | Restoring real-world filmed footage while upscaling 4x: removes realistic noise, some blur and JPEG/WebP-style compression, rebuilding natural texture |
| Architecture | DAT2 (Dual Aggregation Transformer), 11.2 M parameters |
| Source | Chen et al., "Dual Aggregation Transformer for Image Super-Resolution", ICCV 2023 - https://github.com/zhengchen1999/DAT. Weights `4xBHI_dat2_real` by Philip Hofmann - https://github.com/Phhofm/models/releases/tag/4xBHI_dat2_real |
| Licence | CC BY 4.0 (weights) |
| Weights | `4xBHI_dat2_real.safetensors`, 139.8 MB, SHA256 `1f651fe8...2d5b38f6`; downloaded on first use |
| Backends | PyTorch only (spandrel); BF16 / FP32 |
| VRAM | planning estimate 16 KB per input pixel |
| Automatic choice | live-action, very slow, rank 92, removes noise, compression, blur: chosen for damaged filmed sources with the quality preset |
| Limitations | Very slow; on clean sources HAT-L keeps more fine texture |

### swinir-l-x4

| | |
|---|---|
| Purpose | Heavily degraded filmed footage |
| Architecture | SwinIR-L (Swin Transformer), 28.0 M parameters |
| Source | Liang et al., "SwinIR: Image Restoration Using Swin Transformer", ICCV Workshops 2021 - https://github.com/JingyunLiang/SwinIR (official real-world GAN weights, trained on the BSRGAN degradation model) |
| Licence | Apache-2.0 |
| Weights | `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`, 142.5 MB, SHA256 `99adfa91...16328c0a`; downloaded on first use |
| Backends | PyTorch only (spandrel); BF16 / FP32 |
| VRAM | planning estimate 16 KB per input pixel |
| Automatic choice | live-action, very slow, rank 85, removes noise, compression, blur |
| Limitations | Very slow; strong restoration also flattens some genuine texture |

### realplksr-x2

| | |
|---|---|
| Purpose | Natural 2x upscaling (e.g. 1080p -> 4K) with light handling of blur, noise, JPEG and sharpening artefacts |
| Architecture | RealPLKSR (partial large-kernel CNN) with DySample upsampling and layer norm, 7.4 M parameters |
| Source | PLKSR: Lee et al., "Partial Large Kernel CNNs for Efficient Super-Resolution", 2024; weights `2xPublic_realplksr_dysample_layernorm_real` by Philip Hofmann - https://github.com/Phhofm/models/releases/tag/2xPublic_realplksr_dysample_layernorm_real |
| Licence | Apache-2.0 (as published with the weights) |
| Weights | `2xPublic_realplksr_dysample_layernorm_real.safetensors`, 29.5 MB, SHA256 `5eb81e8c...8471cd44`; downloaded on first use |
| Backends | PyTorch only (spandrel); FP16 on CUDA |
| VRAM | planning estimate 3 KB per input pixel |
| Automatic choice | live-action, medium, rank 82; preferred when the job needs at most about 2.5x |
| Limitations | 2x only: larger factors need a second pass, which automatic choice avoids |

### bsrgan-x4

| | |
|---|---|
| Purpose | Badly degraded, blurry and noisy filmed footage |
| Architecture | ESRGAN / RRDBNet (23 blocks), 16.7 M parameters |
| Source | Zhang et al., "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution", ICCV 2021 - https://github.com/cszn/BSRGAN |
| Licence | Apache-2.0 |
| Weights | `BSRGAN.pth`, 67.0 MB, SHA256 `5d505a07...988b1c54`; bundled |
| Backends | PyTorch only (spandrel); FP16 on CUDA |
| VRAM | planning estimate 9 KB per input pixel |
| Automatic choice | live-action, slow, rank 70, removes noise, compression, blur |
| Limitations | Smooths clean sources like realesrgan-x4plus |

### realesr-general-x4v3

| | |
|---|---|
| Purpose | Fast general-purpose 4x model for filmed and animated footage with mild noise and compression; keeps more texture than x4plus |
| Architecture | SRVGGNetCompact (32 conv layers, 64 features, PReLU), 1.2 M parameters |
| Source / licence | Real-ESRGAN (as below), BSD-3-Clause |
| Weights | `realesr-general-x4v3.pth`, 4.9 MB, SHA256 `8dc7edb9...ece96292`; bundled |
| Backends | TensorRT FP16, ONNX Runtime, PyTorch |
| Automatic choice | live-action and animation, fast, rank 60, removes noise and compression; also the fallback when content analysis fails |
| Limitations | Less detail than the larger models |

### realesrgan-x4plus-anime

| | |
|---|---|
| Purpose | Sharp lines for anime pictures and illustrations |
| Architecture | RRDBNet with 6 blocks, 4.5 M parameters |
| Source / licence | Real-ESRGAN (as below), BSD-3-Clause |
| Weights | `RealESRGAN_x4plus_anime_6B.pth`, 17.9 MB, SHA256 `f872d837...155e99da`; bundled |
| Backends | TensorRT FP16, ONNX Runtime, PyTorch |
| Automatic choice | animation, medium, rank 80, made for stills: preferred for pictures, penalised for video |
| Limitations | Less stable between video frames than realesr-animevideov3 |

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
| Limitations | Trained on heavy synthetic degradations: on clean sources it smooths fine texture (the "smeared" look); automatic choice uses it for degraded filmed footage. GAN model: invents plausible detail rather than recovering it, and that detail can shimmer between frames (flicker 0.79 vs 0.32 for bicubic) |

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
