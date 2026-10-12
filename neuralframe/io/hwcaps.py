"""Runtime capability detection: GPUs, ML runtimes, and *functional* FFmpeg hardware codecs.

"Listed by ``ffmpeg -encoders``" is not the same as "works on this machine" (NVENC is compiled into most
builds but needs an NVIDIA GPU and driver), so every hardware codec is probed by actually encoding /
decoding a few frames. Results are cached on disk, keyed by FFmpeg version + GPU/driver identity.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from neuralframe import __version__
from neuralframe.errors import ErrorCode
from neuralframe.io.ffmpeg import FFmpegTools, popen_kwargs, run_tool, tail
from neuralframe.log import get_logger
from neuralframe.monitoring import cpu_name, driver_versions, gpu_count, gpu_status, host_memory

log = get_logger(component="hwcaps")

CodecFamily = Literal["nvenc", "amf", "qsv", "vaapi", "videotoolbox", "software"]

# Preference order within each codec: hardware first, software last.
ENCODER_CANDIDATES: list[tuple[str, str, CodecFamily]] = [
    ("hevc_nvenc", "hevc", "nvenc"),
    ("h264_nvenc", "h264", "nvenc"),
    ("av1_nvenc", "av1", "nvenc"),
    ("hevc_amf", "hevc", "amf"),
    ("h264_amf", "h264", "amf"),
    ("av1_amf", "av1", "amf"),
    ("hevc_qsv", "hevc", "qsv"),
    ("h264_qsv", "h264", "qsv"),
    ("av1_qsv", "av1", "qsv"),
    ("hevc_vaapi", "hevc", "vaapi"),
    ("h264_vaapi", "h264", "vaapi"),
    ("av1_vaapi", "av1", "vaapi"),
    ("hevc_videotoolbox", "hevc", "videotoolbox"),
    ("h264_videotoolbox", "h264", "videotoolbox"),
    ("libx265", "hevc", "software"),
    ("libx264", "h264", "software"),
    ("libsvtav1", "av1", "software"),
    ("libaom-av1", "av1", "software"),
]
FAMILY_ORDER: list[CodecFamily] = ["nvenc", "amf", "qsv", "vaapi", "videotoolbox", "software"]

HWACCEL_CANDIDATES: dict[str, list[str]] = {
    "win32": ["cuda", "d3d11va", "qsv", "dxva2"],
    "linux": ["cuda", "vaapi", "qsv"],
    "darwin": ["videotoolbox"],
}

# Codecs NVDEC can decode (subject to profile limits; FFmpeg falls back to software automatically).
NVDEC_CODECS = {"h264", "hevc", "av1", "vp8", "vp9", "mpeg1video", "mpeg2video", "mpeg4", "vc1", "mjpeg"}

_VAAPI_DEVICE = "/dev/dri/renderD128"
_HW_FAILURE_MARKERS = (
    "failed setup",
    "hwaccel initialisation returned error",
    "device creation failed",
    "no device available",
    "error creating",
    "failed to initialise",
    "cannot load",
    "could not",
    "not supported",
)


class GpuInfo(BaseModel):
    index: int
    name: str
    vendor: str = "nvidia"
    total_vram_mb: int
    free_vram_mb: int
    compute_capability: str | None = None


class CodecCheck(BaseModel):
    name: str
    codec: str
    family: CodecFamily
    listed: bool
    functional: bool = False
    ten_bit: bool = False
    error: str | None = None


class HwaccelCheck(BaseModel):
    name: str
    listed: bool
    functional: bool = False
    error: str | None = None


class Capabilities(BaseModel):
    generated_at: float = Field(default_factory=time.time)
    platform: str
    python: str
    neuralframe: str
    cpu: str
    cpu_threads: int
    ram_total_gb: float
    ram_available_gb: float
    ffmpeg_path: str
    ffmpeg_version: str
    nvidia_driver: str | None = None
    cuda_driver: str | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    torch: str | None = None
    torch_cuda: str | None = None
    torch_cuda_available: bool = False
    cudnn: str | None = None
    tensorrt: str | None = None
    onnxruntime: str | None = None
    onnxruntime_providers: list[str] = Field(default_factory=list)
    native_module: str | None = None
    hwaccels: list[HwaccelCheck] = Field(default_factory=list)
    encoders: list[CodecCheck] = Field(default_factory=list)
    probed_codecs: bool = True

    # ---- selection helpers -------------------------------------------------------------------------

    def functional_encoders(self) -> list[CodecCheck]:
        return [e for e in self.encoders if e.functional]

    def select_encoder(self, codec: str, *, ten_bit: bool) -> CodecCheck | None:
        """Best functional encoder for ``codec`` (hardware first); honours 10-bit when requested."""
        candidates = [e for e in self.encoders if e.codec == codec and e.functional]
        candidates.sort(key=lambda e: FAMILY_ORDER.index(e.family))
        if ten_bit:
            ten = [e for e in candidates if e.ten_bit]
            if ten:
                return ten[0]
        return candidates[0] if candidates else None

    def encoder(self, name: str) -> CodecCheck | None:
        return next((e for e in self.encoders if e.name == name), None)

    def select_hwaccel(self, codec: str) -> str | None:
        functional = [h.name for h in self.hwaccels if h.functional]
        for name in functional:
            if name == "cuda" and codec not in NVDEC_CODECS:
                continue
            return name
        return None

    def best_inference_backend(self) -> str:
        if self.tensorrt and self.torch_cuda_available:
            return "tensorrt"
        if self.torch_cuda_available:
            return "torch"
        if self.onnxruntime:
            return "onnxruntime"
        if self.torch:
            return "torch"
        return "none"


# ---------------------------------------------------------------------------------------------------------
# Library detection
# ---------------------------------------------------------------------------------------------------------


def _version_of(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _detect_ml_runtimes(caps: dict[str, Any]) -> None:
    with contextlib.suppress(Exception):
        torch = importlib.import_module("torch")
        caps["torch"] = str(torch.__version__)
        caps["torch_cuda"] = torch.version.cuda
        caps["torch_cuda_available"] = bool(torch.cuda.is_available())
        if caps["torch_cuda_available"]:
            caps["cudnn"] = str(torch.backends.cudnn.version())
    with contextlib.suppress(Exception):
        trt = importlib.import_module("tensorrt")
        caps["tensorrt"] = str(trt.__version__)
    with contextlib.suppress(Exception):
        ort = importlib.import_module("onnxruntime")
        caps["onnxruntime"] = str(ort.__version__)
        caps["onnxruntime_providers"] = list(ort.get_available_providers())
    with contextlib.suppress(Exception):
        native = importlib.import_module("neuralframe._native")
        caps["native_module"] = f"{native.__version__} ({native.build_info()})"


# ---------------------------------------------------------------------------------------------------------
# FFmpeg codec probing
# ---------------------------------------------------------------------------------------------------------


def _listed(tools: FFmpegTools, kind: Literal["encoders", "hwaccels"]) -> set[str]:
    proc = run_tool(
        [tools.ffmpeg, "-hide_banner", f"-{kind}"], error_code=ErrorCode.TOOL_MISSING, what=f"ffmpeg -{kind}"
    )
    text = proc.stdout.decode("utf-8", "replace")
    names: set[str] = set()
    for line in text.splitlines():
        parts = line.split()
        if kind == "encoders" and len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
        elif kind == "hwaccels" and len(parts) == 1 and not line.endswith(":"):
            names.add(parts[0])
    return names


def _encode_probe(tools: FFmpegTools, name: str, family: CodecFamily, pix_fmt: str) -> str | None:
    """Encode a few synthetic frames; return None on success or the error text."""
    pre: list[str] = []
    vf: list[str] = []
    if family == "vaapi":
        pre = ["-vaapi_device", _VAAPI_DEVICE]
        vf = ["-vf", f"format={'p010le' if '10' in pix_fmt else 'nv12'},hwupload"]
    cmd = [
        tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *pre,
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-frames:v", "5",
        *vf, "-pix_fmt", pix_fmt, "-c:v", name, "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60, **popen_kwargs())
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)
    if proc.returncode != 0:
        return tail(proc.stderr, 3)
    return None


def _probe_encoder(tools: FFmpegTools, name: str, codec: str, family: CodecFamily) -> CodecCheck:
    eight = "yuv420p" if family in ("software", "vaapi") else "nv12"
    ten = "yuv420p10le" if family in ("software", "vaapi") else "p010le"
    error = _encode_probe(tools, name, family, eight)
    if error is not None:
        return CodecCheck(name=name, codec=codec, family=family, listed=True, functional=False, error=error)
    ten_ok = _encode_probe(tools, name, family, ten) is None
    return CodecCheck(name=name, codec=codec, family=family, listed=True, functional=True, ten_bit=ten_ok)


def _make_decode_sample(tools: FFmpegTools, directory: Path) -> Path | None:
    sample = directory / "hwaccel_probe.mp4"
    for codec in ("libx264", "mpeg4"):
        proc = subprocess.run(
            [
                tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25", "-frames:v", "10",
                "-pix_fmt", "yuv420p", "-c:v", codec, str(sample),
            ],
            capture_output=True, timeout=60, **popen_kwargs(),
        )
        if proc.returncode == 0:
            return sample
    return None


def _probe_hwaccel(tools: FFmpegTools, name: str, sample: Path | None) -> HwaccelCheck:
    if sample is None:
        return HwaccelCheck(name=name, listed=True, functional=False, error="no sample to decode")
    pre = ["-hwaccel", name]
    if name == "vaapi":
        pre += ["-hwaccel_device", _VAAPI_DEVICE]
    cmd = [
        tools.ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "warning", *pre,
        "-i", str(sample), "-f", "null", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60, **popen_kwargs())
    except (OSError, subprocess.SubprocessError) as exc:
        return HwaccelCheck(name=name, listed=True, functional=False, error=str(exc))
    stderr = proc.stderr.decode("utf-8", "replace").lower()
    failed = proc.returncode != 0 or any(marker in stderr for marker in _HW_FAILURE_MARKERS)
    return HwaccelCheck(
        name=name, listed=True, functional=not failed, error=tail(proc.stderr, 2) if failed else None
    )


# ---------------------------------------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------------------------------------


def _cache_key(tools: FFmpegTools, driver: str | None, gpu_names: list[str]) -> str:
    raw = json.dumps([tools.ffmpeg, tools.version, driver, gpu_names, sys.platform, __version__])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_capabilities(
    tools: FFmpegTools,
    *,
    cache_dir: Path | None = None,
    probe_codecs: bool = True,
    refresh: bool = False,
    max_age_s: float = 7 * 24 * 3600,
) -> Capabilities:
    driver, cuda_driver = driver_versions()
    gpus: list[GpuInfo] = []
    for idx in range(gpu_count()):
        status = gpu_status(idx)
        if status is None:
            continue
        cc = status.compute_capability
        gpus.append(
            GpuInfo(
                index=idx,
                name=status.name,
                total_vram_mb=status.total_mb,
                free_vram_mb=status.free_mb,
                compute_capability=f"{cc[0]}.{cc[1]}" if cc else None,
            )
        )
    total, available = host_memory()
    base: dict[str, Any] = {
        "platform": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "neuralframe": __version__,
        "cpu": cpu_name(),
        "cpu_threads": _cpu_threads(),
        "ram_total_gb": round(total / 2**30, 1),
        "ram_available_gb": round(available / 2**30, 1),
        "ffmpeg_path": tools.ffmpeg,
        "ffmpeg_version": tools.version,
        "nvidia_driver": driver,
        "cuda_driver": cuda_driver,
        "gpus": gpus,
    }
    _detect_ml_runtimes(base)

    cache_file = None
    if cache_dir is not None:
        cache_file = cache_dir / f"capabilities-{_cache_key(tools, driver, [g.name for g in gpus])}.json"
        if probe_codecs and not refresh and cache_file.is_file():
            with contextlib.suppress(Exception):
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if time.time() - float(cached.get("generated_at", 0)) < max_age_s:
                    base["hwaccels"] = cached["hwaccels"]
                    base["encoders"] = cached["encoders"]
                    log.debug("capabilities loaded from cache", path=str(cache_file))
                    return Capabilities.model_validate(base)

    listed_encoders = _listed(tools, "encoders")
    listed_hwaccels = _listed(tools, "hwaccels")
    platform_key = "win32" if sys.platform == "win32" else ("darwin" if sys.platform == "darwin" else "linux")
    hw_candidates = [h for h in HWACCEL_CANDIDATES[platform_key] if h in listed_hwaccels]

    encoders: list[CodecCheck] = []
    hwaccels: list[HwaccelCheck] = []
    if probe_codecs:
        candidates = [c for c in ENCODER_CANDIDATES if c[0] in listed_encoders]
        # VAAPI exists only on Linux and VideoToolbox only on macOS; never probe them elsewhere.
        if platform_key != "linux":
            candidates = [c for c in candidates if c[2] != "vaapi"]
        if platform_key != "darwin":
            candidates = [c for c in candidates if c[2] != "videotoolbox"]
        if not gpus:
            # Skip NVIDIA-only probes without an NVIDIA GPU; they would fail anyway (and slowly).
            candidates = [c for c in candidates if c[2] != "nvenc"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            encoders = list(pool.map(lambda c: _probe_encoder(tools, *c), candidates))
            with tempfile.TemporaryDirectory(prefix="nf-hwprobe-") as tmp:
                sample = _make_decode_sample(tools, Path(tmp))
                hw_list = [h for h in hw_candidates if h != "cuda" or gpus]
                hwaccels = list(pool.map(lambda h: _probe_hwaccel(tools, h, sample), hw_list))
    else:
        encoders = [
            CodecCheck(name=n, codec=c, family=f, listed=True, functional=f == "software")
            for n, c, f in ENCODER_CANDIDATES
            if n in listed_encoders
        ]
        hwaccels = [HwaccelCheck(name=h, listed=True, functional=False) for h in hw_candidates]

    caps = Capabilities.model_validate(
        {**base, "hwaccels": hwaccels, "encoders": encoders, "probed_codecs": probe_codecs}
    )
    if cache_file is not None and probe_codecs:
        with contextlib.suppress(OSError):
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(caps.model_dump_json(indent=2), encoding="utf-8")
    return caps


def _cpu_threads() -> int:
    import os

    return os.cpu_count() or 1
