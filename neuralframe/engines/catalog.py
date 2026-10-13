"""What each engine needs: pinned upstream code, Python, PyTorch build and packages.

Upstream requirement files are used where they exist, minus packages that are only needed for training,
demos or metrics, or that do not install on Windows (``exclude``); ``packages`` adds or replaces pins.
Commits were pinned on 2026-10-13. None of these recipes has been run yet (see docs/PLAN.md, M12).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from neuralframe.errors import ErrorCode, NeuralFrameError

EngineKind = Literal["video", "image", "faces"]

# Packages that upstream requirement files list for training, demos, logging or evaluation, or that have no
# Windows wheels; none of them is needed to run inference.
NOT_FOR_INFERENCE = (
    "gradio", "gradio_client", "gradio_imageslider", "wandb", "tensorboard", "tb-nightly", "openai", "fastapi",
    "uvicorn", "deepspeed", "triton", "cv_bridge", "external", "polars", "tenacity", "vision-aided-loss",
    "pyiqa", "boto3", "botocore", "datasets", "pytorch-lightning", "torchmetrics", "torchsde", "webdataset",
    "SwissArmyTransformer", "moviepy", "scikit-video", "lpips", "gdown", "liger_kernel", "ipython",
)


@dataclass(frozen=True)
class EngineDef:
    id: str
    name: str
    homepage: str
    licence: str
    restricted: bool
    """Licence allows personal / non-commercial use only."""
    kind: EngineKind
    io: Literal["video", "frames"]
    """What the engine reads per chunk: a lossless video file, or a folder of PNG frames."""
    python: str
    torch: tuple[str, ...]
    torch_index: str
    """PyTorch wheel index (cu118, cu124, cu128, cu130)."""
    repo: str | None = None
    commit: str | None = None
    requirements: str | None = None
    """Requirement file in the upstream source tree."""
    exclude: tuple[str, ...] = NOT_FOR_INFERENCE
    packages: tuple[str, ...] = ()
    editable: bool = False
    """``pip install -e`` the upstream source (projects with a setup.py that must be importable)."""
    post_install: tuple[tuple[str, ...], ...] = ()
    """Commands run with the environment's Python in the source folder after the packages."""
    resources: tuple[tuple[str, str], ...] = ()
    """(file in neuralframe/engines/resources, destination relative to the source folder)."""
    weights: Literal["library", "engine"] = "library"
    """``library``: files listed with the models, downloaded by NeuralFrame; ``engine``: the engine
    downloads its own during installation or on first use."""
    prefetch: tuple[str, ...] = ()
    """For ``engine`` weights: a command (relative to the source folder) that downloads them at install."""
    needs_hf_token: bool = False
    sequence_frames: int = 48
    """Frames per engine call (one chunk of the job)."""
    overlap: int = 8
    """Frames shared by neighbouring chunks (cross-faded) so chunk boundaries do not show."""
    disk_gb: float = 0.0
    """Rough size of the environment (packages), shown before installing."""
    notes: str = ""

    @property
    def source_url(self) -> str | None:
        if self.repo is None or self.commit is None:
            return None
        return f"https://github.com/{self.repo}/archive/{self.commit}.zip"


ENGINES: dict[str, EngineDef] = {e.id: e for e in [
    # ---- video ---------------------------------------------------------------------------------------------
    EngineDef(
        id="seedvr2", name="SeedVR2 (one-step diffusion video restoration, ByteDance)",
        homepage="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler", licence="Apache-2.0", restricted=False,
        kind="video", io="video", python="3.12", torch=("torch", "torchvision"), torch_index="cu130",
        repo="numz/ComfyUI-SeedVR2_VideoUpscaler", commit="4490bd1f482e026674543386bb2a4d176da245b9",
        requirements="requirements.txt", sequence_frames=81, overlap=8, disk_gb=7.0,
        notes="Standalone runner of SeedVR2 with GGUF/FP8 weights, block swapping and VAE tiling for small GPUs.",
    ),
    EngineDef(
        id="flashvsr", name="FlashVSR v1.1 (streaming one-step diffusion video super-resolution)",
        homepage="https://github.com/OpenImagingLab/FlashVSR", licence="Apache-2.0", restricted=False,
        kind="video", io="frames", python="3.11", torch=("torch==2.6.0", "torchvision==0.21.0"),
        torch_index="cu124", repo="OpenImagingLab/FlashVSR", commit="cf910c61a60733e610e9c6e8b607f80c3a6c202b",
        requirements="requirements.txt",
        exclude=(*NOT_FOR_INFERENCE, "torch", "torchaudio", "torchvision", "opencv-python-headless", "pandas"),
        editable=True,
        resources=(("flashvsr_driver.py", "examples/WanVSR/nf_flashvsr_driver.py"),
                   ("block_sparse_attn.py", "nf_shims/block_sparse_attn.py")),
        sequence_frames=77, overlap=8, disk_gb=8.0,
        notes="Needs the Block-Sparse-Attention CUDA kernels for full speed and quality; built when a CUDA "
              "compiler is available, otherwise a slower PyTorch implementation of the same attention is used.",
    ),
    EngineDef(
        id="dove", name="DOVE (one-step diffusion video super-resolution on CogVideoX)",
        homepage="https://github.com/zhengchen1999/DOVE", licence="Apache-2.0", restricted=False,
        kind="video", io="video", python="3.11", torch=("torch==2.5.1", "torchvision==0.20.1"), torch_index="cu124",
        repo="zhengchen1999/DOVE", commit="0cd4240442cb5d122839c279977142cb6d648987",
        requirements="requirements.txt", exclude=(*NOT_FOR_INFERENCE, "torch", "torchvision", "decord"),
        packages=("diffusers==0.32.2",), sequence_frames=49, overlap=8, disk_gb=7.0,
    ),
    EngineDef(
        id="swiftvr", name="SwiftVR (real-time one-step generative video restoration)",
        homepage="https://github.com/H-oliday/SwiftVR", licence="Apache-2.0 code, SwiftVR Research License weights",
        restricted=True, kind="video", io="frames", python="3.10",
        torch=("torch==2.10.0", "torchvision==0.25.0"), torch_index="cu128",
        repo="H-oliday/SwiftVR", commit="4fb82651efee06a90ac47fd2d9af2f632e84ab45",
        requirements="requirements.txt", exclude=(*NOT_FOR_INFERENCE, "torch", "torchvision", "decord"),
        editable=True, sequence_frames=96, overlap=8, disk_gb=7.0,
        notes="Weights: research licence, commercial use needs the authors' written permission.",
    ),
    EngineDef(
        id="nvidia-vsr", name="NVIDIA RTX Video Super Resolution (nvidia-vfx)",
        homepage="https://github.com/NVIDIA-Maxine/nvidia-vfx-python-samples",
        licence="NVIDIA proprietary licence (nvidia-vfx)", restricted=False, kind="video", io="frames",
        python="3.12", torch=("torch==2.11.0",), torch_index="cu128",
        packages=("nvidia-vfx==0.2.0.0", "numpy", "pillow"),
        resources=(("nvidia_vsr_driver.py", "nf_nvidia_vsr_driver.py"),), weights="engine",
        sequence_frames=120, overlap=0, disk_gb=4.0,
        notes="Needs an NVIDIA RTX GPU (Turing or newer) and driver 570.65 or newer. Proprietary: read NVIDIA's "
              "licence before commercial use.",
    ),
    # ---- pictures (also usable on video frame by frame) ------------------------------------------------------
    EngineDef(
        id="osediff", name="OSEDiff (one-step diffusion image super-resolution)",
        homepage="https://github.com/cswry/OSEDiff", licence="Apache-2.0", restricted=False, kind="image",
        io="frames", python="3.10", torch=("torch==2.0.1", "torchvision==0.15.2"), torch_index="cu118",
        repo="cswry/OSEDiff", commit="cdd3b7b04850fa9bb7faf8f2187edd279cc96507", requirements="requirements.txt",
        exclude=(*NOT_FOR_INFERENCE, "torch", "xformers"), packages=("numpy<2",), sequence_frames=16, overlap=0,
        disk_gb=5.0,
    ),
    EngineDef(
        id="hypir", name="HYPIR (diffusion-prior image restoration, SIGGRAPH 2025)",
        homepage="https://github.com/XPixelGroup/HYPIR", licence="HYPIR licence (non-commercial)", restricted=True,
        kind="image", io="frames", python="3.10", torch=("torch==2.6.0", "torchvision==0.21.0"),
        torch_index="cu124", repo="XPixelGroup/HYPIR", commit="b61d107c6cef38f01a93c7833558869731cfa8c1",
        requirements="requirements.txt", exclude=(*NOT_FOR_INFERENCE, "torch", "torchvision", "open_clip_torch"),
        sequence_frames=16, overlap=0, disk_gb=5.0,
    ),
    EngineDef(
        id="invsr", name="InvSR (diffusion inversion super-resolution, CVPR 2025)",
        homepage="https://github.com/zsyOAOA/InvSR", licence="S-Lab License 1.0 (non-commercial)", restricted=True,
        kind="image", io="frames", python="3.10", torch=("torch==2.4.0", "torchvision==0.19.0"),
        torch_index="cu121", repo="zsyOAOA/InvSR", commit="609788cd076f1990e38d3912092ca67286e1e0bc",
        requirements="requirements.txt", exclude=(*NOT_FOR_INFERENCE, "bitsandbytes"),
        packages=("accelerate==0.34.2", "huggingface-hub==0.25.1", "numpy==1.26.3", "safetensors==0.4.5",
                  "scipy==1.14.1", "timm==1.0.9", "transformers==4.46.0", "omegaconf==2.3.0", "einops==0.8.0"),
        editable=True, weights="engine", sequence_frames=16, overlap=0, disk_gb=5.0,
        notes="Downloads SD-Turbo and its noise predictor itself on first use.",
    ),
    EngineDef(
        id="supir", name="SUPIR (SDXL-based photo-realistic restoration, CVPR 2024)",
        homepage="https://github.com/Fanghua-Yu/SUPIR", licence="SUPIR licence (non-commercial)", restricted=True,
        kind="image", io="frames", python="3.10", torch=("torch==2.1.2", "torchvision==0.16.2"),
        torch_index="cu118", repo="Fanghua-Yu/SUPIR", commit="bda91af2000042f8bedfec8897d92917e67c1d88",
        requirements="requirements.txt",
        exclude=(*NOT_FOR_INFERENCE, "torch", "torchvision", "xformers", "ninja", "scikit-learn", "Markdown",
                 "requests", "httpx", "urllib3"),
        packages=("xformers==0.0.23.post1",), resources=(("supir_driver.py", "nf_supir_driver.py"),),
        sequence_frames=8, overlap=0, disk_gb=6.0,
        notes="The SUPIR-v0Q checkpoint is published only on Google Drive and Baidu: download it yourself and "
              "use it with \"Use a file I have\". Needs a lot of graphics and system memory.",
    ),
    EngineDef(
        id="lucidflux", name="LucidFlux (caption-free restoration on FLUX.1, ICLR 2026)",
        homepage="https://github.com/W2GenAI-Lab/LucidFlux", licence="FLUX.1-dev Non-Commercial License",
        restricted=True, kind="image", io="frames", python="3.11", torch=("torch==2.5.1", "torchvision==0.20.1"),
        torch_index="cu124", repo="W2GenAI-Lab/LucidFlux", commit="3e688aca5a6199e4757c97462ea0ec282d4cafad",
        requirements="requirements.txt", exclude=(*NOT_FOR_INFERENCE, "attr"), weights="engine",
        prefetch=("-m", "tools.download_weights", "--dest", "weights"), needs_hf_token=True,
        sequence_frames=4, overlap=0, disk_gb=6.0,
        notes="Downloads FLUX.1-dev (about 33 GB, gated: accept its licence on huggingface.co and add your "
              "Hugging Face token), SigLIP, SwinIR and the LucidFlux weights during installation.",
    ),
    # ---- faces -------------------------------------------------------------------------------------------------
    EngineDef(
        id="gfpgan", name="GFPGAN / RestoreFormer (face restoration)", homepage="https://github.com/TencentARC/GFPGAN",
        licence="Apache-2.0", restricted=False, kind="faces", io="frames", python="3.10",
        torch=("torch==2.0.1", "torchvision==0.15.2"), torch_index="cu118", repo="TencentARC/GFPGAN",
        commit="7552a7791caad982045a7bbe5634bbf1cd5c8679",
        packages=("basicsr==1.4.2", "facexlib==0.3.0", "numpy<2", "opencv-python", "pyyaml", "scipy", "tqdm", "lmdb",
                  "yapf"),
        weights="engine", sequence_frames=64, overlap=0, disk_gb=4.0,
        notes="Downloads its face and face-detection weights from the GFPGAN and facexlib releases on first use.",
    ),
    EngineDef(
        id="restoreformerpp", name="RestoreFormer++ (face restoration)",
        homepage="https://github.com/wzhouxiff/RestoreFormerPlusPlus", licence="Apache-2.0", restricted=False,
        kind="faces", io="frames", python="3.10", torch=("torch==2.0.1", "torchvision==0.15.2"), torch_index="cu118",
        repo="wzhouxiff/RestoreFormerPlusPlus", commit="59d250f45d6b11a3e620d62e6e067ca1288f93a9",
        packages=("basicsr==1.4.2", "facexlib==0.3.0", "numpy<2", "opencv-python", "pyyaml", "scipy", "tqdm",
                  "omegaconf", "einops", "pytorch-lightning==1.9.5"),
        weights="engine", sequence_frames=64, overlap=0, disk_gb=4.0,
    ),
    EngineDef(
        id="codeformer", name="CodeFormer (face restoration)", homepage="https://github.com/sczhou/CodeFormer",
        licence="S-Lab License 1.0 (non-commercial)", restricted=True, kind="faces", io="frames", python="3.10",
        torch=("torch==2.0.1", "torchvision==0.15.2"), torch_index="cu118", repo="sczhou/CodeFormer",
        commit="b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b", requirements="requirements.txt",
        exclude=(*NOT_FOR_INFERENCE, "torch", "torchvision"), packages=("numpy<2",),
        post_install=(("basicsr/setup.py", "develop"),), weights="engine", sequence_frames=64, overlap=0,
        disk_gb=4.0,
    ),
    EngineDef(
        id="keep", name="KEEP (video face restoration, ECCV 2024)", homepage="https://github.com/jnjaby/KEEP",
        licence="S-Lab License 1.0 (non-commercial)", restricted=True, kind="faces", io="frames", python="3.10",
        torch=("torch==2.0.1", "torchvision==0.15.2"), torch_index="cu118", repo="jnjaby/KEEP",
        commit="f382730e329d5924fcaf5344b392a6036ba59a4d", requirements="requirements.txt",
        exclude=(*NOT_FOR_INFERENCE, "torch", "torchvision"), packages=("numpy<2",),
        post_install=(("basicsr/setup.py", "develop"),), weights="engine", sequence_frames=40, overlap=0,
        disk_gb=4.0,
    ),
]}


def engine(engine_id: str) -> EngineDef:
    try:
        return ENGINES[engine_id]
    except KeyError:
        raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"unknown engine {engine_id!r}",
                               hint=f"known engines: {', '.join(sorted(ENGINES))}") from None
