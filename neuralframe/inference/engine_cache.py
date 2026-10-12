"""On-disk TensorRT engine cache.

Engines are only valid for the exact GPU, driver and TensorRT build they were made with, so the key
covers model weights, ONNX opset, GPU name + compute capability, driver version, TensorRT version,
precision and the optimisation profile. Engines whose environment no longer matches are pruned.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from neuralframe.log import get_logger

log = get_logger(component="engine-cache")

Shape = tuple[int, int, int, int]


@dataclass(frozen=True)
class EngineKey:
    model_id: str
    weights_sha256: str
    opset: int
    gpu_name: str
    compute_capability: str
    driver: str
    trt_version: str
    precision: str
    min_shape: Shape
    opt_shape: Shape
    max_shape: Shape

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]

    def environment(self) -> dict[str, str]:
        return {
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
            "driver": self.driver,
            "trt_version": self.trt_version,
        }

    @property
    def filename(self) -> str:
        h, w = self.opt_shape[2], self.opt_shape[3]
        return f"{self.model_id}-{self.precision}-{h}x{w}b{self.opt_shape[0]}-{self.digest()}.engine"


class EngineCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: EngineKey) -> Path:
        return self.root / key.filename

    def load(self, key: EngineKey) -> bytes | None:
        path = self.path_for(key)
        meta_path = path.with_suffix(".json")
        if not path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if meta.get("key") != json.loads(json.dumps(asdict(key))):
            return None
        return path.read_bytes()

    def store(self, key: EngineKey, data: bytes, build_seconds: float) -> Path:
        path = self.path_for(key)
        tmp = path.with_suffix(".engine.tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        meta = {"key": asdict(key), "size_bytes": len(data), "build_seconds": round(build_seconds, 1),
                "created_at": time.time()}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path

    def entries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for meta_path in sorted(self.root.glob("*.json")):
            if meta_path.name.startswith("timing"):
                continue
            with contextlib.suppress(OSError, json.JSONDecodeError):
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["path"] = str(meta_path.with_suffix(".engine"))
                out.append(meta)
        return out

    def prune_stale(self, environment: dict[str, str]) -> list[Path]:
        """Delete engines built for another GPU, driver or TensorRT version."""
        removed: list[Path] = []
        for meta in self.entries():
            key = meta.get("key", {})
            if any(key.get(k) != v for k, v in environment.items()):
                engine = Path(meta["path"])
                for p in (engine, engine.with_suffix(".json")):
                    with contextlib.suppress(OSError):
                        p.unlink()
                removed.append(engine)
        if removed:
            log.info("pruned stale TensorRT engines", count=len(removed))
        return removed

    def timing_cache_path(self, environment: dict[str, str]) -> Path:
        tag = hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()[:12]
        return self.root / f"timing-{tag}.cache"
