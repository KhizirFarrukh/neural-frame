"""Application configuration: pydantic-settings with YAML file + ``NEURALFRAME_*`` environment overrides.

Precedence (highest first): explicit keyword arguments → environment variables
(``NEURALFRAME_ENCODE__CODEC=h264``) → YAML config file → built-in defaults.
"""

from __future__ import annotations

import os
import sys
from contextvars import ContextVar
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from neuralframe.errors import ErrorCode, NeuralFrameError

Codec = Literal["hevc", "h264", "av1"]
Speed = Literal["slow", "medium", "fast"]

_CONFIG_PATH: ContextVar[Path | None] = ContextVar("neuralframe_config_path", default=None)


def default_cache_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "neuralframe"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "neuralframe"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "neuralframe"


def user_config_dir() -> Path:
    """Per-user settings folder: %APPDATA%\\neuralframe (Windows) or ~/.config/neuralframe."""
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home())) / "neuralframe"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "neuralframe"


def default_config_path() -> Path | None:
    """First existing of: $NEURALFRAME_CONFIG, ./config.yaml, user config dir."""
    env = os.environ.get("NEURALFRAME_CONFIG")
    if env:
        return Path(env)
    local = Path.cwd() / "config.yaml"
    if local.is_file():
        return local
    user = user_config_dir() / "config.yaml"
    return user if user.is_file() else None


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Section):
    cache_dir: Path = Field(default_factory=default_cache_dir)
    models_dir: Path | None = None
    model_search_paths: list[Path] = Field(default_factory=list)
    """Extra folders with model files you already have (looked up by file name, checked by SHA256)."""
    engines_dir: Path | None = None
    work_dir: Path | None = None
    db_path: Path | None = None

    def models(self) -> Path:
        return self.models_dir or self.cache_dir / "models"

    def engines(self) -> Path:
        return self.engines_dir or self.cache_dir / "engines"

    def work(self) -> Path:
        return self.work_dir or self.cache_dir / "work"

    def database(self) -> Path:
        return self.db_path or self.cache_dir / "jobs.db"


class FFmpegConfig(_Section):
    ffmpeg: str | None = None
    ffprobe: str | None = None
    loglevel: str = "error"


def _yaml_switch(value: Any) -> Any:
    """YAML 1.1 reads unquoted on/off as booleans; map them back to the option names."""
    if value is True:
        return "on"
    if value is False:
        return "off"
    return value


class DecodeConfig(_Section):
    hwaccel: Literal["auto", "none", "cuda", "d3d11va", "dxva2", "qsv", "vaapi", "videotoolbox"] = "auto"
    deinterlace: Literal["auto", "on", "off"] = "auto"
    deinterlace_filter: Literal["bwdif", "yadif"] = "bwdif"
    denoise: Literal["off", "light", "medium", "strong"] = "off"
    deblock: bool = False

    _switches = field_validator("deinterlace", "denoise", mode="before")(_yaml_switch)


class EncodeConfig(_Section):
    encoder: str = "auto"
    codec: Codec | None = None
    quality: int | None = Field(default=None, ge=0, le=63)
    speed: Speed | None = None
    ten_bit: bool = False
    extra_args: list[str] = Field(default_factory=list)


class ColorConfig(_Section):
    matrix: Literal["auto", "preserve", "bt709"] = "auto"
    convert_primaries: bool = True
    output_range: Literal["limited", "full", "preserve"] = "limited"


class ChunkingConfig(_Section):
    chunk_frames: int = Field(default=240, ge=8)
    overlap_frames: int | None = Field(default=None, ge=0)
    scene_detection: Literal["auto", "on", "off"] = "auto"

    _switches = field_validator("scene_detection", mode="before")(_yaml_switch)
    scene_threshold: float = Field(default=0.35, gt=0, lt=1)
    temporal_chunk_frames: int = Field(default=48, ge=8)
    chunk_format: Literal["segment", "lossless"] = "segment"


class PipelineConfig(_Section):
    # device: colour conversion, model and resize run in torch on the GPU (or CPU) inside the inference
    # stage; host: separate CPU stages using the native/NumPy ops. auto = device when torch is installed
    # and a neural model is used or a CUDA GPU is available.
    mode: Literal["auto", "device", "host"] = "auto"
    # Device mode on CUDA: split the GPU work into upload / inference / download stages, each with its own
    # thread and CUDA stream, so transfers and colour/resize work overlap with the network.
    cuda_streams: bool = True
    ops: Literal["auto", "native", "numpy"] = "auto"
    ram_budget_gb: float | None = Field(default=None, gt=0)
    keep_work_dir: bool = False
    verify_sync: bool = True


class InferenceConfig(_Section):
    backend: Literal["auto", "tensorrt", "torch", "onnxruntime"] = "auto"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    precision: Literal["fp16", "fp32"] = "fp16"
    tile_size: int | None = Field(default=None, ge=32)
    tile_overlap: int | None = Field(default=None, ge=0)
    batch_size: int | None = Field(default=None, ge=1)
    gpus: list[int] | None = None
    onnx_providers: list[str] | None = None
    deterministic: bool = True


class VramConfig(_Section):
    safety_margin: float = Field(default=0.15, ge=0.0, lt=0.9)
    max_vram_gb: float | None = Field(default=None, gt=0)
    calibrate: bool = True


class FrameGenConfig(_Section):
    fps: str = "60"
    """Default target frame rate for frame generation ("60", "59.94", "60000/1001", ...)."""
    ntsc_snap: bool = True
    """Use 60000/1001 instead of 60 when the source runs on an NTSC clock (exact 2x for 29.97)."""
    engine: Literal["auto", "rife", "minterpolate"] = "auto"
    model: str = "rife-4.22"
    repair_drops: Literal["auto", "on", "off"] = "auto"
    order: Literal["auto", "upscale_first", "interpolate_first"] = "auto"
    """auto: interpolate first for classical and compact models (cheaper), upscale first for heavy models."""
    rife_scale: float | None = Field(default=None, gt=0, le=1)
    """RIFE flow scale (1.0, 0.5, 0.25); None = automatic by frame size."""
    max_gap_s: float = Field(default=0.5, gt=0)
    """Longer gaps are held (nearest frame) instead of morphed across."""
    dup_threshold: float = Field(default=0.0025, gt=0)
    motion_threshold: float = Field(default=0.008, gt=0)

    _switches = field_validator("repair_drops", mode="before")(_yaml_switch)


class ApiConfig(_Section):
    host: str = "127.0.0.1"
    port: int = 8765
    input_roots: list[Path] = Field(default_factory=lambda: [Path.home()])
    output_roots: list[Path] = Field(default_factory=lambda: [Path.home()])
    upload_dir: Path | None = None
    """Where uploaded files are stored (default: <cache_dir>/uploads); always readable by jobs."""
    output_dir: Path | None = None
    """Outputs of uploaded files and inputs outside the output roots (default: <cache_dir>/outputs)."""
    max_upload_gb: float = 20.0


class LoggingConfig(_Section):
    level: str = "INFO"
    json_output: bool = False
    file: Path | None = None


class PresetConfig(_Section):
    description: str = ""
    model: str | None = None
    tile_overlap: int | None = None
    codec: Codec | None = None
    speed: Speed | None = None
    quality: dict[str, int] = Field(default_factory=dict)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NEURALFRAME_",
        env_nested_delimiter="__",
        extra="forbid",
        validate_default=True,
    )

    paths: PathsConfig = Field(default_factory=PathsConfig)
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    encode: EncodeConfig = Field(default_factory=EncodeConfig)
    color: ColorConfig = Field(default_factory=ColorConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    vram: VramConfig = Field(default_factory=VramConfig)
    framegen: FrameGenConfig = Field(default_factory=FrameGenConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    presets: dict[str, PresetConfig] = Field(default_factory=dict)

    @field_validator("presets", mode="after")
    @classmethod
    def _merge_builtin_presets(cls, value: dict[str, PresetConfig]) -> dict[str, PresetConfig]:
        merged = dict(builtin_presets())
        for name, preset in value.items():
            base = merged.get(name)
            if base is None:
                merged[name] = preset
            else:
                update = preset.model_dump(exclude_unset=True)
                quality = {**base.quality, **update.pop("quality", {})}
                merged[name] = base.model_copy(update={**update, "quality": quality})
        return merged

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        path = _CONFIG_PATH.get()
        if path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path, yaml_file_encoding="utf-8"))
        return tuple(sources)

    def preset(self, name: str) -> PresetConfig:
        try:
            return self.presets[name]
        except KeyError:
            raise NeuralFrameError(
                ErrorCode.CONFIG_INVALID,
                f"unknown preset {name!r}",
                hint=f"available presets: {', '.join(sorted(self.presets))}",
            ) from None


def builtin_presets() -> dict[str, PresetConfig]:
    text = resources.files("neuralframe.data").joinpath("presets.yaml").read_text(encoding="utf-8")
    raw: dict[str, Any] = yaml.safe_load(text) or {}
    return {name: PresetConfig.model_validate(body) for name, body in raw.items()}


def load_settings(config_path: Path | None = None, **overrides: Any) -> Settings:
    """Load settings from ``config_path`` (or the default location) plus environment and overrides."""
    path = config_path if config_path is not None else default_config_path()
    if path is not None and not path.is_file():
        raise NeuralFrameError(ErrorCode.CONFIG_INVALID, f"config file not found: {path}")
    token = _CONFIG_PATH.set(path)
    try:
        return Settings(**overrides)
    except NeuralFrameError:
        raise
    except Exception as exc:
        raise NeuralFrameError(ErrorCode.CONFIG_INVALID, f"invalid configuration: {exc}") from exc
    finally:
        _CONFIG_PATH.reset(token)
