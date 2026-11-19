"""The recordings (HDF5 files, archive stores and videos): catalog, frame thumbnails, and frames read and flat-fielded.

The Recordings screen and Labeling's New queue need, for every recording
(``.h5``, ``.avi`` or ``.mp4`` file, or an archive's ``dataset.zarr`` store,
:mod:`.recording_files`) of a setup (in its recording folders or
registered to it), its frame count and image size and whether it can
actually be read (some recordings hold chunks compressed with an HDF5
filter plugin that is not installed, and a few files are not HDF5 at all).
Opening every file to read its shape is slow over network storage (and a
video is indexed when first opened, :mod:`.frame_reader`), so the per-file
facts are cached in one JSON index keyed by path; an entry is reused while
the file's size and modification time (a store's root ``zarr.json``) are
unchanged and refreshed otherwise.  Thumbnails are flat-fielded when the field cache holds the
recording's field and are otherwise the raw frame.

``RecordingSource`` reads one recording's frames (an HDF5 recording, an
archive store or a video, through :func:`.frame_reader.open_frames`) and flat-fields
them with a per-recording correction fitted once from frames spread over
the recording and cached on disk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .flat_field import FlatField, apply_flat_field, estimate_flat_field
from .frame_reader import DATASET_PATH, open_frames
from .recording_files import (
    ARCHIVE_NAME, HDF5_SUFFIXES, VIDEO_SUFFIXES, is_recording, open_metadata, recording_kind, recording_name, stamp_path,
)
from .settings import CACHE_DIR


CACHE_VERSION = 1
FLAT_FIELD_SAMPLE_COUNT = 64
PENDING_FACTS = {"frames": None, "height": None, "width": None, "readable": False, "error": None}
# One app process writes the index from several threads (listings and the prober).
_INDEX_LOCK = threading.Lock()


@dataclass
class RecordingInfo:
    """One recording as the browser shows it."""

    name: str
    path: str
    frames: int | None
    height: int | None
    width: int | None
    size_bytes: int
    modified_at: str
    readable: bool
    error: str | None
    # The dataset holding the frames, under the h5's names (ignored for a video).
    dataset: str = DATASET_PATH
    # Not opened yet: frames and readability are unknown until a prober has looked.
    pending: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def list_directory(path: Path, *, all_files: bool = False) -> dict[str, Any]:
    """Directories, HDF5 files (kind "h5"), archive stores (kind "archive") and videos (kind "video") directly under ``path``, for the file explorer.

    Hidden entries are skipped; unreadable subdirectories are listed but
    flagged.  An archive store is a directory but a recording, not a folder
    to open.  With ``all_files`` every other regular file is listed too
    (kind "file"), for videos stored under other names.
    """

    directory = Path(path).expanduser()
    if not directory.exists():
        raise FileNotFoundError(f"{directory} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    directory = directory.resolve()
    entries: list[dict[str, Any]] = []
    try:
        children = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as error:
        raise PermissionError(f"{directory}: permission denied") from error
    for child in children:
        if child.name.startswith("."):
            continue
        try:
            if child.is_dir() and recording_kind(child) == "archive":
                entries.append({"name": child.name, "path": str(child), "kind": "archive", "size_bytes": None, "modified_at": _iso_utc(child.stat().st_mtime), "readable": os.access(child, os.R_OK | os.X_OK)})
            elif child.is_dir():
                entries.append({"name": child.name, "path": str(child), "kind": "dir", "size_bytes": None, "modified_at": _iso_utc(child.stat().st_mtime), "readable": os.access(child, os.R_OK | os.X_OK)})
            elif child.is_file():
                kind = recording_kind(child) or ("file" if all_files else None)
                if kind is None:
                    continue
                stat = child.stat()
                entries.append({"name": child.name, "path": str(child), "kind": kind, "size_bytes": int(stat.st_size), "modified_at": _iso_utc(stat.st_mtime), "readable": os.access(child, os.R_OK)})
        except OSError:
            continue
    parent = None if directory.parent == directory else str(directory.parent)
    return {"path": str(directory), "parent": parent, "entries": entries, "all_files": all_files, "suffixes": [*HDF5_SUFFIXES, *VIDEO_SUFFIXES], "archive_name": ARCHIVE_NAME}


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def probe_frames(frame_count: int) -> tuple[int, ...]:
    """First, middle and last frame: one recording reads its first chunk but not the later ones."""

    if frame_count <= 0:
        return ()
    return tuple(sorted({0, frame_count // 2, frame_count - 1}))


def probe_recording(path: Path, dataset: str = DATASET_PATH) -> dict[str, Any]:
    """Shape and readability of one file: open it (indexing a video), read its shape, then a few frames.

    Reading frames is what tells an installed-filter recording from one whose
    chunks need a plugin we do not have, and a damaged video from a whole
    one; the shape alone reads fine for both.
    """

    facts: dict[str, Any] = {"frames": None, "height": None, "width": None, "readable": False, "error": None}
    try:
        with open_frames(path, dataset) as frames:
            facts["frames"] = frames.frame_count
            facts["height"], facts["width"] = frames.shape
            for frame in probe_frames(frames.frame_count):
                try:
                    frames[frame]
                except (OSError, ValueError) as error:
                    raise OSError(f"frame {frame} not readable: {error}") from error
            facts["readable"] = True
    except Exception as error:  # noqa: BLE001 - any failure makes the file unreadable, and we want its message
        facts["error"] = f"{type(error).__name__}: {error}".strip()
    return facts


def _file_stamp(path: Path) -> dict[str, int]:
    stat = stamp_path(path).stat()
    return {"mtime_ns": int(stat.st_mtime_ns), "size": int(stat.st_size)}


class RecordingIndex:
    """The JSON cache of per-file facts; ``cache is None`` keeps it in memory only."""

    def __init__(self, cache: Path | None) -> None:
        self.cache = None if cache is None else Path(cache)
        self.entries: dict[str, dict[str, Any]] = {}
        # Facts probed by this instance, which ``save`` adds to the file.
        self._new: dict[str, dict[str, Any]] = {}
        self._dirty = False
        if self.cache is not None and self.cache.exists():
            try:
                data = json.loads(self.cache.read_text())
                if data.get("version") == CACHE_VERSION:
                    self.entries = dict(data.get("entries") or {})
            except (OSError, ValueError):
                self.entries = {}

    @staticmethod
    def _key(path: Path, dataset: str) -> str:
        return str(path) if dataset == DATASET_PATH else f"{path}#{dataset}"

    def cached(self, path: Path, dataset: str = DATASET_PATH) -> dict[str, Any] | None:
        """The facts of ``path`` while the file is unchanged since they were probed, else ``None``."""

        stamp = _file_stamp(path)
        entry = self.entries.get(self._key(path, dataset))
        if entry is None or entry.get("mtime_ns") != stamp["mtime_ns"] or entry.get("size") != stamp["size"]:
            return None
        return entry

    def facts(self, path: Path, dataset: str = DATASET_PATH) -> dict[str, Any]:
        entry = self.cached(path, dataset)
        if entry is None:
            entry = {**_file_stamp(path), **probe_recording(path, dataset)}
            self.entries[self._key(path, dataset)] = entry
            self._new[self._key(path, dataset)] = entry
            self._dirty = True
        return entry

    def save(self) -> None:
        """Add the facts probed here to the file, keeping what other listings saved since this one read it."""

        if self.cache is None or not self._dirty:
            return
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        with _INDEX_LOCK:
            on_disk = RecordingIndex(self.cache).entries
            self.entries = {**on_disk, **self._new}
            # A unique temporary: a reader must never see a half-written file.
            with tempfile.NamedTemporaryFile("w", dir=self.cache.parent, prefix=self.cache.name + ".", suffix=".tmp", delete=False) as handle:
                json.dump({"version": CACHE_VERSION, "entries": self.entries}, handle, indent=1)
            os.replace(handle.name, self.cache)
        self._new = {}
        self._dirty = False


class RecordingProber:
    """Probes recordings in a background thread, so a listing does not wait for them.

    Probing opens each file and reads three frames, which over network
    storage takes about a second per recording (longer for a damaged one,
    and for a video, which is indexed on its first opening), so
    a folder of a hundred recordings would hold the Recordings screen for
    minutes.  The screen lists the files at once (:func:`list_recordings`
    with ``probe=False``), hands the unprobed ones here and polls; each
    probed file is saved to the index as it is done.
    """

    def __init__(self, cache: Path | None) -> None:
        self.cache = cache
        self._queue: list[tuple[Path, str]] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(self, items: Iterable[tuple[Path, str]]) -> None:
        with self._lock:
            self._queue.extend(item for item in items if item not in self._queue)
            if self._queue and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._run, name="recording-prober", daemon=True)
                self._thread.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    return
                path, dataset = self._queue[0]
            index = RecordingIndex(self.cache)
            try:
                index.facts(path, dataset)
                index.save()
            except OSError:  # vanished since it was listed: the next listing drops it
                pass
            with self._lock:
                self._queue.remove((path, dataset))


# How many folders below a recording folder are searched: ``<folder>/<day>/<recording>.h5`` is two, as is
# ``<folder>/<dataset>/dataset.zarr``.
FOLDER_DEPTH = 2


def find_recordings(roots: Iterable[Path], depth: int = FOLDER_DEPTH) -> list[Path]:
    """Every recording (HDF5 or video file, archive store) given as a root or under a root at most ``depth`` folders down (hidden ones skipped), resolved, sorted by name then path.

    Resolving makes a recording reached through two roots (one a symlink to the other) one recording.
    An archive store is not searched inside.
    """

    found: set[Path] = set()
    for root in roots:
        root = Path(root)
        if is_recording(root):
            found.add(root.resolve())
            continue
        if not root.is_dir():
            continue
        folders = [root]
        for _ in range(depth + 1):
            below = []
            for folder in folders:
                try:
                    children = list(folder.iterdir())
                except OSError:
                    continue
                for path in children:
                    if path.name.startswith("."):
                        continue
                    if is_recording(path):
                        found.add(path.resolve())
                    elif path.is_dir():
                        below.append(path)
            folders = below
    return sorted(found, key=lambda p: (recording_name(p), str(p)))


def list_recordings(roots: Iterable[Path], *, cache: Path | None = None, dataset: str = DATASET_PATH, probe: bool = True) -> list[RecordingInfo]:
    """Catalog the recordings under ``roots`` (directories, or recording files themselves), reusing ``cache`` for unchanged files.

    ``dataset`` is the HDF5 dataset holding the frames (a setup's video dataset).
    With ``probe=False`` a file the cache has no current facts for is not
    opened but listed as ``pending`` (for :class:`RecordingProber`).
    """

    index = RecordingIndex(cache)
    infos = []
    for path in find_recordings(roots):
        try:
            stamp = _file_stamp(path)
            entry = index.facts(path, dataset) if probe else (index.cached(path, dataset) or PENDING_FACTS)
        except OSError as error:
            # Vanished or unstat-able between listing and probing: report it rather than drop it.
            entry = {"frames": None, "height": None, "width": None, "readable": False, "error": f"{type(error).__name__}: {error}"}
            stamp = {"mtime_ns": 0, "size": 0}
        infos.append(
            RecordingInfo(
                name=recording_name(path), path=str(path), frames=entry["frames"], height=entry["height"], width=entry["width"],
                size_bytes=int(stamp["size"]), modified_at=_iso_utc(stamp["mtime_ns"] / 1e9), readable=bool(entry["readable"]),
                error=entry["error"], dataset=dataset, pending=entry is PENDING_FACTS,
            )
        )
    index.save()
    return infos


def _read_frame(path: Path, frame: int, dataset: str = DATASET_PATH) -> NDArray[np.uint8]:
    if recording_kind(path) not in ("h5", "archive"):
        with open_frames(path) as frames:
            return frames[frame]
    # An HDF5 or archive frame is read raw (open_frames casts to uint8), so a 16-bit camera's can be rescaled below.
    with open_metadata(path) as handle:
        if dataset not in handle:
            raise KeyError(f"no dataset {dataset}")
        data = handle[dataset]
        if not 0 <= frame < data.shape[0]:
            raise IndexError(f"frame {frame} out of range for {data.shape[0]} frames")
        values = np.asarray(data[int(frame)])
        if values.dtype != np.uint8:
            # 16-bit cameras: scale by the frame's own range so the thumbnail is visible.
            low, high = float(values.min()), float(values.max())
            values = ((values - low) / max(high - low, 1.0) * 255.0).astype(np.uint8)
        return values


def thumbnail_frame(path: Path, frame: int, dataset_root: Path = CACHE_DIR, dataset: str = DATASET_PATH) -> NDArray[np.uint8]:
    """The frame flat-fielded through the cached field when one exists, else raw."""

    path = Path(path)
    field_cache = Path(dataset_root) / "flat_fields"
    if (field_cache / f"{recording_name(path)}.npz").exists():
        source = RecordingSource(path, field_cache, dataset=dataset)
        try:
            _, corrected = source.corrected(int(frame))
        finally:
            source.close()
        return corrected
    return _read_frame(path, int(frame), dataset)


def thumbnail_png(
    path: Path, frame: int, scale: float = 0.25, dataset_root: Path = CACHE_DIR, dataset: str = DATASET_PATH
) -> bytes:
    """PNG bytes of one frame scaled by ``scale`` (at least 1x1 pixel)."""

    if not scale > 0:
        raise ValueError("scale must be positive")
    image = Image.fromarray(thumbnail_frame(path, frame, dataset_root, dataset), mode="L")
    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    if size != image.size:
        image = image.resize(size, Image.Resampling.BOX if scale < 1 else Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class RecordingSource:
    """One read-only recording with a lazily fitted, disk-cached flat field."""

    def __init__(self, path: Path, cache_dir: Path, dataset: str = DATASET_PATH) -> None:
        self.path = Path(path)
        self.name = recording_name(self.path)
        self.cache_dir = Path(cache_dir)
        self._field_lock = threading.Lock()
        self._field: FlatField | None = None
        self.frames = open_frames(self.path, dataset)
        self.frame_count = self.frames.frame_count
        self.shape = self.frames.shape

    def read(self, frame_index: int) -> NDArray[np.uint8]:
        return self.frames[frame_index]

    def flat_field(self) -> FlatField:
        # Light/full previews and label proposals can request the same field concurrently.
        with self._field_lock:
            return self._prepare_flat_field()

    def _prepare_flat_field(self) -> FlatField:
        if self._field is not None:
            return self._field
        cache = self.cache_dir / f"{self.name}.npz"
        if cache.exists():
            with np.load(cache) as archive:
                self._field = FlatField(
                    illumination=np.asarray(archive["illumination"], dtype=np.float64),
                    dark_level=float(archive["dark_level"]),
                    reference_level=float(archive["reference_level"]),
                    gain=np.asarray(archive["gain"], dtype=np.float64),
                )
            return self._field
        indices = np.linspace(0, self.frame_count - 1, min(FLAT_FIELD_SAMPLE_COUNT, self.frame_count), dtype=np.int64)
        frames = []
        for i in indices:
            try:
                frames.append(self.frames[int(i)])
            except OSError:
                # Some recordings hold chunks compressed with an HDF5 filter plugin that is not installed.
                continue
        if len(frames) < min(8, len(indices)):
            raise OSError(f"only {len(frames)} of {len(indices)} calibration frames of {self.name} are readable")
        calibration = np.stack(frames)
        field = estimate_flat_field(
            calibration, temporal_quantile=0.8, spatial_radius=31, smoothing_passes=2, min_gain=0.5, max_gain=2.5
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Every writer owns its temporary file, including separate server/job processes.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, prefix=f"{cache.name}.", suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle, illumination=field.illumination, dark_level=field.dark_level,
                    reference_level=field.reference_level, gain=field.gain,
                )
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        self._field = field
        return field

    def corrected(self, frame_index: int) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        raw = self.read(frame_index)
        corrected = apply_flat_field(raw, self.flat_field(), clip=(0.0, 255.0))
        return raw, np.clip(np.rint(corrected), 0, 255).astype(np.uint8)

    def close(self) -> None:
        self.frames.close()

    def __enter__(self) -> RecordingSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
