"""GPU / host monitoring via NVML (pynvml) and psutil. Every call degrades gracefully without NVIDIA GPUs."""

from __future__ import annotations

import contextlib
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import psutil

try:  # nvidia-ml-py
    import pynvml

    _HAVE_NVML = True
except ImportError:  # pragma: no cover - dependency is declared, but keep monitoring optional
    _HAVE_NVML = False

_NVML_LOCK = threading.Lock()
_NVML_READY: bool | None = None


def nvml_available() -> bool:
    global _NVML_READY
    with _NVML_LOCK:
        if _NVML_READY is None:
            if not _HAVE_NVML:
                _NVML_READY = False
            else:
                try:
                    pynvml.nvmlInit()
                    _NVML_READY = True
                except Exception:
                    _NVML_READY = False
        return _NVML_READY


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


@dataclass(frozen=True)
class GpuStatus:
    index: int
    name: str
    uuid: str
    total_mb: int
    used_mb: int
    free_mb: int
    utilization_pct: int | None
    temperature_c: int | None
    compute_capability: tuple[int, int] | None


def gpu_count() -> int:
    if not nvml_available():
        return 0
    try:
        return int(pynvml.nvmlDeviceGetCount())
    except Exception:
        return 0


def gpu_status(index: int) -> GpuStatus | None:
    if not nvml_available():
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util: int | None
        temp: int | None
        cc: tuple[int, int] | None
        try:
            util = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        except Exception:
            util = None
        try:
            temp = int(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            temp = None
        try:
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            cc = (int(major), int(minor))
        except Exception:
            cc = None
        return GpuStatus(
            index=index,
            name=_text(pynvml.nvmlDeviceGetName(handle)),
            uuid=_text(pynvml.nvmlDeviceGetUUID(handle)),
            total_mb=int(mem.total) // 2**20,
            used_mb=int(mem.used) // 2**20,
            free_mb=int(mem.free) // 2**20,
            utilization_pct=util,
            temperature_c=temp,
            compute_capability=cc,
        )
    except Exception:
        return None


def driver_versions() -> tuple[str | None, str | None]:
    """(NVIDIA driver version, max CUDA version supported by the driver)."""
    if not nvml_available():
        return None, None
    try:
        driver = _text(pynvml.nvmlSystemGetDriverVersion())
    except Exception:
        driver = None
    try:
        cuda = int(pynvml.nvmlSystemGetCudaDriverVersion())
        cuda_text: str | None = f"{cuda // 1000}.{(cuda % 1000) // 10}"
    except Exception:
        cuda_text = None
    return driver, cuda_text


def cpu_name() -> str:
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    elif sys.platform.startswith("linux"):
        with contextlib.suppress(OSError), open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def host_memory() -> tuple[int, int]:
    """(total, available) host RAM in bytes."""
    vm = psutil.virtual_memory()
    return int(vm.total), int(vm.available)


@dataclass
class _GpuPeak:
    peak_used_mb: int = 0
    util_sum: float = 0.0
    util_samples: int = 0
    max_temp_c: int = 0


@dataclass
class ResourceSampler:
    """Background sampler recording VRAM peak, mean GPU utilisation and host RSS during a job."""

    gpu_indices: list[int]
    interval_s: float = 0.5
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _peaks: dict[int, _GpuPeak] = field(default_factory=dict)
    _peak_rss: int = 0
    _baseline_used_mb: dict[int, int] = field(default_factory=dict)

    def start(self) -> ResourceSampler:
        for idx in self.gpu_indices:
            status = gpu_status(idx)
            self._baseline_used_mb[idx] = status.used_mb if status else 0
            self._peaks[idx] = _GpuPeak()
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            self.sample_once(proc)
            self._stop.wait(self.interval_s)

    def sample_once(self, proc: psutil.Process | None = None) -> None:
        with contextlib.suppress(Exception):
            self._peak_rss = max(self._peak_rss, int((proc or psutil.Process()).memory_info().rss))
        for idx in self.gpu_indices:
            status = gpu_status(idx)
            if status is None:
                continue
            peak = self._peaks.setdefault(idx, _GpuPeak())
            peak.peak_used_mb = max(peak.peak_used_mb, status.used_mb)
            if status.utilization_pct is not None:
                peak.util_sum += status.utilization_pct
                peak.util_samples += 1
            if status.temperature_c is not None:
                peak.max_temp_c = max(peak.max_temp_c, status.temperature_c)

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.sample_once()
        gpus = {
            str(idx): {
                "peak_vram_used_mb": p.peak_used_mb,
                "peak_vram_job_mb": max(0, p.peak_used_mb - self._baseline_used_mb.get(idx, 0)),
                "mean_utilization_pct": round(p.util_sum / p.util_samples, 1) if p.util_samples else None,
                "max_temperature_c": p.max_temp_c or None,
            }
            for idx, p in self._peaks.items()
        }
        return {"gpus": gpus, "peak_host_rss_mb": self._peak_rss // 2**20}

    def current(self) -> dict[int, GpuStatus]:
        out: dict[int, GpuStatus] = {}
        for idx in self.gpu_indices:
            status = gpu_status(idx)
            if status is not None:
                out[idx] = status
        return out


def wall_clock() -> float:
    return time.perf_counter()
