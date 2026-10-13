"""Model library: which models are on this computer, downloads on demand, local files and updates.

Nothing is downloaded until a model is needed by a job or the user asks for it (model manager, ``neuralframe
models download``). A model's file can be:

* downloaded into the model folder (``<models>/<id>/<file>``), checked against the registry's SHA256;
* included with a Windows build made with ``--with-models`` (``<app>/models/<id>/<file>``);
* found in a folder from ``paths.model_search_paths`` (same file name, same SHA256);
* a file the user already has elsewhere and *linked* (``neuralframe models link``): the same SHA256 means
  the same file; a different file is accepted only if it loads as the same kind of network and scale;
* an *update*: for weights published as Hugging Face files or GitHub release assets, the published SHA256
  is compared with the installed one, and a newer file can be installed. Updated pins are kept per user in
  ``library.json`` (the registry itself is never edited).

Downloads stream to ``<file>.part`` (resumed when the server supports it), report progress (bytes, speed,
time left), can be cancelled, and are verified before they are used.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import httpx

from neuralframe.errors import ErrorCode, JobCancelled, NeuralFrameError
from neuralframe.log import get_logger
from neuralframe.models.registry import ModelSpec, WeightsSpec

if TYPE_CHECKING:
    from neuralframe.config import Settings

log = get_logger(component="models")

State = Literal["builtin", "missing", "installed", "bundled", "found", "linked", "linked-different"]
LINKABLE_SUFFIXES = (".pth", ".safetensors", ".ckpt")
CHECK_MAX_AGE_S = 6 * 3600
_HF = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/([^?]+)")
_GH = re.compile(r"^https://github\.com/([^/]+/[^/]+)/releases/download/([^/]+)/([^/?]+)")


class Cancellable(Protocol):
    def is_set(self) -> bool: ...


@dataclass(frozen=True)
class DownloadProgress:
    model: str
    done: int
    total: int | None
    speed_bps: float
    eta_s: float | None

    @property
    def fraction(self) -> float | None:
        return min(1.0, self.done / self.total) if self.total else None

    def text(self) -> str:
        """``45.2 of 140.0 MB - 5.1 MB/s - about 19 s left``"""
        done = f"{self.done / 1e6:.1f}" + (f" of {self.total / 1e6:.1f} MB" if self.total else " MB")
        parts = [done]
        if self.speed_bps > 0:
            parts.append(f"{self.speed_bps / 1e6:.1f} MB/s")
        if self.eta_s is not None:
            parts.append("about " + _duration(self.eta_s) + " left")
        return " - ".join(parts)


ProgressCallback = Callable[[DownloadProgress], None]


@dataclass
class UpdateInfo:
    model: str
    source: str
    """``huggingface`` or ``github``."""
    url: str
    sha256: str | None
    size_bytes: int | None
    checked_at: float
    available: bool
    note: str = ""


@dataclass
class ModelStatus:
    model: str
    state: State
    path: Path | None
    size_bytes: int | None
    """On disk when present, else the download size."""
    partial_bytes: int = 0
    update: UpdateInfo | None = None

    @property
    def ready(self) -> bool:
        return self.state != "missing"

    def text(self) -> str:
        size = f"{self.size_bytes / 1e6:.0f} MB" if self.size_bytes else "size unknown"
        if self.state == "missing":
            if self.partial_bytes:
                return f"Not downloaded - paused at {self.partial_bytes / 1e6:.0f} of {size}"
            return f"Not downloaded ({size})"
        text = {
            "builtin": "Built in",
            "installed": f"Downloaded ({size})",
            "bundled": "Included with the app",
            "found": f"Found in {self.path.parent if self.path else '?'}",
            "linked": f"Using your file: {self.path}",
            "linked-different": f"Using your file (a different version): {self.path}",
        }[self.state]
        if self.update is not None and self.update.available:
            text += " - update available"
        return text


def _duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    if seconds < 90:
        return f"{seconds} s"
    if seconds < 5400:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class ModelLibrary:
    """Model files for one models folder (thread-safe; one instance can serve several downloads)."""

    STATE_FILE = "library.json"

    def __init__(self, models_dir: Path, *, search_paths: list[Path] | tuple[Path, ...] = (),
                 timeout_s: float = 60.0, transport: httpx.BaseTransport | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.search_paths = [Path(p) for p in search_paths]
        self.timeout_s = timeout_s
        self.transport = transport
        """HTTP transport override (tests); None = the network."""
        self._lock = threading.RLock()
        self._busy: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> ModelLibrary:
        return cls(settings.paths.models(), search_paths=list(settings.paths.model_search_paths))

    # ---- persistent state --------------------------------------------------------------------------------

    def _state_path(self) -> Path:
        return self.models_dir / self.STATE_FILE

    def _state(self) -> dict[str, Any]:
        path = self._state_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("ignoring unreadable model library state", path=str(path), error=str(exc))
            data = {}
        for key in ("links", "updates", "checks", "verified"):
            data.setdefault(key, {})
        return data

    def _save(self, data: dict[str, Any]) -> None:
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _matches(self, path: Path, sha256: str, size: int | None = None, *, quick: bool = False) -> bool:
        """SHA256 check, remembered per (path, size, mtime) so large files are hashed once. ``quick`` (status
        displays) trusts a file of the expected size that was not hashed yet; using a model always checks."""
        try:
            st = path.stat()
        except OSError:
            return False
        key = str(path.resolve())
        stamp = [st.st_size, st.st_mtime_ns]
        with self._lock:
            data = self._state()
            cached = data["verified"].get(key)
            if cached and cached[:2] == stamp:
                return bool(cached[2] == sha256)
        if quick:
            return size is None or st.st_size == size
        actual = sha256_of(path)
        with self._lock:
            data = self._state()
            data["verified"][key] = [*stamp, actual]
            self._save(data)
        return actual == sha256

    # ---- what is installed -------------------------------------------------------------------------------

    def weights(self, spec: ModelSpec) -> WeightsSpec:
        """The weights to use: the registry's pin, or an installed update."""
        if spec.weights is None:
            raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"model {spec.id!r} has no weight file")
        update = self._state()["updates"].get(spec.id)
        if update and update.get("base_sha256") == spec.weights.sha256:
            return WeightsSpec(url=update["url"], sha256=update["sha256"], filename=update["filename"],
                               size_bytes=update.get("size_bytes"))
        return spec.weights

    def effective_spec(self, spec: ModelSpec) -> ModelSpec:
        """``spec`` with the weights that will actually be used (update or a linked different version), so
        engine caches and resume keys follow the real file."""
        if spec.weights is None:
            return spec
        link = self._state()["links"].get(spec.id)
        pinned = self.weights(spec)
        if link and link.get("sha256") not in (None, pinned.sha256) and Path(link["path"]).is_file():
            path = Path(link["path"])
            weights = WeightsSpec(url=path.as_uri(), sha256=link["sha256"], filename=path.name,
                                  size_bytes=link.get("size_bytes"))
        else:
            weights = pinned
        return spec if weights == spec.weights else spec.model_copy(update={"weights": weights})

    def target_path(self, spec: ModelSpec) -> Path:
        return self.models_dir / spec.id / self.weights(spec).filename

    def locate(self, spec: ModelSpec, *, quick: bool = False) -> tuple[State, Path | None]:
        """Where the model's file is (``quick``: without hashing files not checked before)."""
        if spec.weights is None:
            return "builtin", None
        weights = self.weights(spec)
        size = weights.size_bytes
        link = self._state()["links"].get(spec.id)
        if link and Path(link["path"]).is_file():
            return ("linked" if link.get("sha256") == weights.sha256 else "linked-different"), Path(link["path"])
        target = self.target_path(spec)
        if target.is_file() and self._matches(target, weights.sha256, size, quick=quick):
            return "installed", target
        from neuralframe.bundle import bundled

        shipped = bundled("models", spec.id, weights.filename)
        if shipped is not None and self._matches(shipped, weights.sha256, size, quick=quick):
            return "bundled", shipped
        for folder in self.search_paths:
            for candidate in (folder / spec.id / weights.filename, folder / weights.filename):
                if candidate.is_file() and self._matches(candidate, weights.sha256, size, quick=quick):
                    return "found", candidate
        return "missing", None

    def status(self, spec: ModelSpec) -> ModelStatus:
        """What the user sees (fast: files of the right size count as present until they are used)."""
        state, path = self.locate(spec, quick=True)
        size = None
        partial = 0
        if path is not None:
            size = path.stat().st_size
        elif spec.weights is not None:
            size = self.weights(spec).size_bytes
            part = self._partial_path(spec)
            partial = part.stat().st_size if part.is_file() else 0
        check = self._state()["checks"].get(spec.id)
        update = UpdateInfo(**check) if check else None
        if update is not None and spec.weights is not None and state != "builtin":
            current = self.weights(spec)
            update.available = _is_newer(update, current)
        return ModelStatus(model=spec.id, state=state, path=path, size_bytes=size, partial_bytes=partial,
                           update=update)

    def is_ready(self, spec: ModelSpec) -> bool:
        return self.locate(spec, quick=True)[0] != "missing"

    def busy(self, spec: ModelSpec) -> bool:
        with self._lock:
            return spec.id in self._busy

    def disk_usage(self) -> int:
        if not self.models_dir.is_dir():
            return 0
        return sum(f.stat().st_size for f in self.models_dir.rglob("*") if f.is_file())

    # ---- downloads ----------------------------------------------------------------------------------------

    def ensure(self, spec: ModelSpec, *, progress: ProgressCallback | None = None,
               cancel: Cancellable | None = None) -> Path:
        """The model's file, downloading it first if it is not on this computer yet."""
        if spec.weights is None:
            raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"model {spec.id!r} has no weight file")
        if spec.licence_restricted:
            log.warning("model licence restricts use", model=spec.id, licence=spec.licence, url=spec.licence_url)
        _state, path = self.locate(spec)
        if path is not None:
            return path
        return self.download(spec, progress=progress, cancel=cancel)

    def _partial_path(self, spec: ModelSpec) -> Path:
        target = self.target_path(spec)
        return target.with_name(target.name + ".part")

    def download(self, spec: ModelSpec, *, progress: ProgressCallback | None = None,
                 cancel: Cancellable | None = None, weights: WeightsSpec | None = None) -> Path:
        """Download (or resume) the model's file into the models folder and verify it."""
        weights = weights or self.weights(spec)
        with self._lock:
            if spec.id in self._busy:
                raise NeuralFrameError(ErrorCode.DOWNLOAD_FAILED, f"{spec.id} is already being downloaded")
            self._busy.add(spec.id)
        try:
            return self._download(spec, weights, progress, cancel)
        finally:
            with self._lock:
                self._busy.discard(spec.id)

    def _download(self, spec: ModelSpec, weights: WeightsSpec, progress: ProgressCallback | None,
                  cancel: Cancellable | None) -> Path:
        target = self.models_dir / spec.id / weights.filename
        partial = target.with_name(target.name + ".part")
        target.parent.mkdir(parents=True, exist_ok=True)
        meter = _Meter(spec.id, weights.size_bytes, progress)
        digest = hashlib.sha256()
        offset = partial.stat().st_size if partial.is_file() else 0
        log.info("downloading model", model=spec.id, url=weights.url, resume_from=offset)
        try:
            with _Stream(weights.url, offset, self.timeout_s, self.transport) as (stream, resumed, total):
                if resumed:
                    with partial.open("rb") as fh:
                        while block := fh.read(1 << 20):
                            digest.update(block)
                else:
                    offset = 0
                meter.start(offset, total if total is not None else weights.size_bytes)
                with partial.open("ab" if resumed else "wb") as fh:
                    for block in stream:
                        if cancel is not None and cancel.is_set():
                            raise JobCancelled(f"download of {spec.id} cancelled")
                        fh.write(block)
                        digest.update(block)
                        meter.add(len(block))
        except JobCancelled:
            meter.finish()
            log.info("download cancelled; the partial file is kept to resume later", model=spec.id)
            raise
        except (httpx.HTTPError, OSError) as exc:
            raise NeuralFrameError(
                ErrorCode.DOWNLOAD_FAILED, f"cannot download {spec.id}: {exc}",
                hint="check the internet connection and free disk space, then try again (it continues where it "
                f"stopped); or download {weights.url} yourself and use it with `neuralframe models link "
                f"{spec.id} FILE`",
            ) from exc
        meter.finish()
        actual = digest.hexdigest()
        if actual != weights.sha256:
            partial.unlink(missing_ok=True)
            raise NeuralFrameError(ErrorCode.DOWNLOAD_FAILED,
                                   f"{spec.id}: the downloaded file is not the expected one (SHA256 {actual[:12]}..., "
                                   f"expected {weights.sha256[:12]}...)",
                                   hint="try again; if it keeps failing the file was changed at its source - use "
                                   "\"Check for updates\"")
        partial.replace(target)
        with self._lock:
            data = self._state()
            st = target.stat()
            data["verified"][str(target.resolve())] = [st.st_size, st.st_mtime_ns, actual]
            self._save(data)
        log.info("model downloaded and verified", model=spec.id, path=str(target))
        return target

    def delete(self, spec: ModelSpec) -> int:
        """Remove the downloaded copy (and any partial download). Never touches linked or bundled files."""
        if self.busy(spec):
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"{spec.id} is being downloaded; cancel it first")
        folder = self.models_dir / spec.id
        freed = 0
        if folder.is_dir():
            for f in folder.iterdir():
                if f.is_file():
                    freed += f.stat().st_size
                    f.unlink()
            if not any(folder.iterdir()):
                folder.rmdir()
        return freed

    # ---- local files -------------------------------------------------------------------------------------

    def link(self, spec: ModelSpec, file: Path) -> ModelStatus:
        """Use a file the user already has instead of downloading it."""
        if spec.weights is None:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"{spec.id} needs no model file")
        file = Path(file).expanduser().resolve()
        if not file.is_file():
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"file not found: {file}")
        weights = self.weights(spec)
        sha = sha256_of(file)
        if sha != weights.sha256:
            if file.suffix.lower() not in LINKABLE_SUFFIXES:
                raise NeuralFrameError(ErrorCode.INVALID_INPUT,
                                       f"{file.name} is not the expected file and is not a .pth, .safetensors or "
                                       ".ckpt file",
                                       hint="TorchScript (.pt) files can contain code and are only accepted when "
                                       "they are exactly the published file")
            self._check_loads(spec, file)
        with self._lock:
            data = self._state()
            data["links"][spec.id] = {"path": str(file), "sha256": sha, "size_bytes": file.stat().st_size}
            self._save(data)
        return self.status(spec)

    @staticmethod
    def _check_loads(spec: ModelSpec, file: Path) -> None:
        from neuralframe.inference.archs import load_network

        try:
            load_network(spec, file)
        except NeuralFrameError as exc:
            raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE,
                                   f"{file.name} is not a version of {spec.id}: {exc.message}",
                                   hint="choose the file published for this model (see `neuralframe models info "
                                   f"{spec.id}`), or add it as a new model with `neuralframe models add`") from exc
        except Exception as exc:
            raise NeuralFrameError(ErrorCode.MODEL_UNAVAILABLE, f"{file.name} is not a version of {spec.id}: {exc}",
                                   hint="add it as a new model with `neuralframe models add` instead") from exc

    def unlink(self, spec: ModelSpec) -> None:
        with self._lock:
            data = self._state()
            data["links"].pop(spec.id, None)
            self._save(data)

    # ---- updates -----------------------------------------------------------------------------------------

    @staticmethod
    def update_source(spec: ModelSpec) -> str | None:
        """``huggingface`` / ``github`` when the weights' host publishes checksums we can compare."""
        if spec.weights is None:
            return None
        if _HF.match(spec.weights.url):
            return "huggingface"
        if _GH.match(spec.weights.url):
            return "github"
        return None

    def check_update(self, spec: ModelSpec, *, force: bool = False, client: httpx.Client | None = None
                     ) -> UpdateInfo | None:
        """Ask the host for the current published file; None when the model's host has no update support."""
        source = self.update_source(spec)
        if source is None or spec.weights is None:
            return None
        cached = self._state()["checks"].get(spec.id)
        if cached and not force and time.time() - cached["checked_at"] < CHECK_MAX_AGE_S:
            info = UpdateInfo(**cached)
            info.available = _is_newer(info, self.weights(spec))
            return info
        own = client is None
        http = client or httpx.Client(timeout=30.0, follow_redirects=True, transport=self.transport,
                                      headers={"User-Agent": "neuralframe-model-updates"})
        try:
            info = (_check_huggingface if source == "huggingface" else _check_github)(spec, http)
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise NeuralFrameError(ErrorCode.DOWNLOAD_FAILED, f"cannot check {spec.id} for updates: {exc}",
                                   hint="check the internet connection and try again later") from exc
        finally:
            if own:
                http.close()
        info.available = _is_newer(info, self.weights(spec))
        with self._lock:
            data = self._state()
            data["checks"][spec.id] = asdict(info)
            self._save(data)
        return info

    def apply_update(self, spec: ModelSpec, *, progress: ProgressCallback | None = None,
                     cancel: Cancellable | None = None) -> Path:
        info = self.check_update(spec)
        if info is None or not info.available:
            raise NeuralFrameError(ErrorCode.INVALID_INPUT, f"{spec.id} has no update available")
        if info.sha256 is None:
            raise NeuralFrameError(ErrorCode.DOWNLOAD_FAILED,
                                   f"{spec.id}: the host does not publish a checksum for the new file",
                                   hint="download it yourself and use it with `neuralframe models link`")
        assert spec.weights is not None
        filename = urllib.parse.unquote(info.url.rsplit("/", 1)[-1].split("?", 1)[0]) or spec.weights.filename
        new = WeightsSpec(url=info.url, sha256=info.sha256, filename=filename, size_bytes=info.size_bytes)
        old_path = self.target_path(spec)
        path = self.download(spec, progress=progress, cancel=cancel, weights=new)
        with self._lock:
            data = self._state()
            data["updates"][spec.id] = {**new.model_dump(), "base_sha256": spec.weights.sha256,
                                        "installed_at": time.time()}
            self._save(data)
        if old_path.is_file() and old_path != path:
            old_path.unlink()
        return path


def _is_newer(info: UpdateInfo, current: WeightsSpec) -> bool:
    if info.sha256:
        return info.sha256 != current.sha256
    return bool(info.size_bytes and current.size_bytes and info.size_bytes != current.size_bytes)


def _check_huggingface(spec: ModelSpec, http: httpx.Client) -> UpdateInfo:
    assert spec.weights is not None
    match = _HF.match(spec.weights.url)
    assert match is not None
    repo, _rev, file = match.group(1), match.group(2), urllib.parse.unquote(match.group(3))
    data = http.get(f"https://huggingface.co/api/models/{repo}", params={"blobs": "true"}).raise_for_status().json()
    sibling = next((s for s in data.get("siblings", []) if s.get("rfilename") == file), None)
    if sibling is None:
        return UpdateInfo(model=spec.id, source="huggingface", url=spec.weights.url, sha256=None, size_bytes=None,
                          checked_at=time.time(), available=False, note="file no longer published")
    lfs = sibling.get("lfs") or {}
    return UpdateInfo(model=spec.id, source="huggingface",
                      url=f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(file)}",
                      sha256=lfs.get("sha256"), size_bytes=lfs.get("size") or sibling.get("size"),
                      checked_at=time.time(), available=False)


def _check_github(spec: ModelSpec, http: httpx.Client) -> UpdateInfo:
    assert spec.weights is not None
    match = _GH.match(spec.weights.url)
    assert match is not None
    repo, tag, asset = match.group(1), match.group(2), urllib.parse.unquote(match.group(3))
    release = http.get(f"https://api.github.com/repos/{repo}/releases/tags/{tag}",
                       headers={"Accept": "application/vnd.github+json"}).raise_for_status().json()
    item = next((a for a in release.get("assets", []) if a.get("name") == asset), None)
    if item is None:
        return UpdateInfo(model=spec.id, source="github", url=spec.weights.url, sha256=None, size_bytes=None,
                          checked_at=time.time(), available=False, note="file no longer published")
    digest = item.get("digest") or ""
    return UpdateInfo(model=spec.id, source="github", url=item.get("browser_download_url") or spec.weights.url,
                      sha256=digest.split(":", 1)[1] if digest.startswith("sha256:") else None,
                      size_bytes=item.get("size"), checked_at=time.time(), available=False)


class _Meter:
    """Throttled progress with a sliding-window speed (about 5 updates per second)."""

    def __init__(self, model: str, total: int | None, callback: ProgressCallback | None) -> None:
        self.model, self.total, self.callback = model, total, callback
        self.done = 0
        self._window: list[tuple[float, int]] = []
        self._last = 0.0

    def start(self, done: int, total: int | None) -> None:
        self.done, self.total = done, total
        self._window = [(time.monotonic(), done)]
        self._emit(force=True)

    def add(self, n: int) -> None:
        self.done += n
        self._emit()

    def finish(self) -> None:
        self._emit(force=True)

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if self.callback is None or (not force and now - self._last < 0.2):
            return
        self._last = now
        self._window.append((now, self.done))
        while len(self._window) > 2 and now - self._window[0][0] > 5.0:
            self._window.pop(0)
        t0, d0 = self._window[0]
        speed = (self.done - d0) / (now - t0) if now - t0 > 0.05 else 0.0
        eta = (self.total - self.done) / speed if self.total and speed > 0 else None
        self.callback(DownloadProgress(self.model, self.done, self.total, speed, eta))


class _Stream:
    """Stream ``url`` from ``offset`` (HTTP range when supported; ``file://`` for local mirrors and tests).
    Yields (chunks, resumed, total size)."""

    def __init__(self, url: str, offset: int, timeout_s: float, transport: httpx.BaseTransport | None) -> None:
        self.url, self.offset, self.timeout_s, self.transport = url, offset, timeout_s, transport
        self._stack: list[Any] = []

    def __enter__(self) -> tuple[Iterator[bytes], bool, int | None]:
        if self.url.startswith("file:"):
            path = Path(urllib.request.url2pathname(urllib.parse.urlparse(self.url).path))
            size = path.stat().st_size
            fh = path.open("rb")
            self._stack.append(fh)
            resumed = 0 < self.offset < size
            if resumed:
                fh.seek(self.offset)
            return iter(lambda: fh.read(1 << 20), b""), resumed, size
        client = httpx.Client(follow_redirects=True, timeout=self.timeout_s, transport=self.transport,
                              headers={"User-Agent": "neuralframe-model-download"})
        self._stack.append(client)
        headers = {"Range": f"bytes={self.offset}-"} if self.offset else {}
        response = client.send(client.build_request("GET", self.url, headers=headers), stream=True)
        self._stack.append(response)
        response.raise_for_status()
        resumed = response.status_code == 206
        length = int(response.headers.get("content-length", 0)) or None
        total = (length + self.offset) if resumed and length else length
        return response.iter_bytes(1 << 20), resumed, total

    def __exit__(self, *exc: object) -> None:
        for item in reversed(self._stack):
            item.close()
