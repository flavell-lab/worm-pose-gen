"""On-disk workspace for one recording range.

A workspace is the place the pipeline stages and the user's fixes read from
and write back to; the app keeps one per
recording, always the whole recording (``docs/APP_SIMPLIFICATION.md``)::

    <workspaces>/<name>/
      workspace.json         recording path, frame range, settings (setup, models)
      state.npz              current per-frame arrays (the poses.npz layout)
      hypotheses.npz         candidates per frame (hypotheses_*, path_*, prediction_*)
      provenance.npz         per frame: algorithm id, job or edit id, unix time
      summary.json           the stages' run summary (fit configuration, prior, propagation)
      masks/chunk_NNNNN.npz  bitpacked cleaned masks, 1024 rows per chunk
      fields/chunk_NNNNN.npz the body-field network's outputs the fits read (``field_store``)
      overrides/masks/       sparse: frames whose mask the user edited
      edits.jsonl            append-only log of every intervention
      fixes/                 Refit and Relabel previews waiting for Keep or Discard
      exports/               the per-recording tables the Export button wrote

Rows are positions in ``frame_index = range(first, last + 1, step)``; every
per-frame array has one entry per row.  Every file write goes through a
temporary file and ``os.replace`` so a crash mid-write leaves the previous
version intact.  Masks are kept (refits and mask edits need them) as
``np.packbits`` of the flattened boolean image, which compresses to a few
kilobytes per frame; probability maps are not kept, but what the fits read
of the body-field network's outputs is (:mod:`field_store`).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import threading
import time as time_module
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .frame_reader import DATASET_PATH, open_frames
from .settings import WORKSPACES_DIR


DEFAULT_WORKSPACES_ROOT = WORKSPACES_DIR
MASK_CHUNK_ROWS = 1024
HYPOTHESIS_PREFIXES = ("hypotheses_", "path_", "prediction_")
ALGORITHM_DTYPE = "<U32"
JOB_DTYPE = "<U64"
_CHUNK_CACHE_SIZE = 2

BoolArray = NDArray[np.bool_]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    # The temporary is dot-prefixed so no listing glob (``chunk_*``, the
    # override row names) can match one left behind by a crash, and it is
    # written through an open handle so numpy does not append ``.npz``.
    tmp = path.with_name("." + path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez_compressed(handle, **{k: np.asarray(v) for k, v in arrays.items()})
    os.replace(tmp, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def read_recording_shape(recording: Path, dataset: str = DATASET_PATH) -> tuple[int, int, int] | None:
    """``(T, H, W)`` of a recording, ``None`` when it cannot be read."""

    try:
        with open_frames(recording, dataset) as frames:
            return frames.frame_count, *frames.shape
    except (OSError, KeyError, ValueError):
        return None


def read_image_shape(recording: Path, dataset: str = DATASET_PATH) -> tuple[int, int] | None:
    """``(H, W)`` of a recording's frames, ``None`` when it cannot be read."""

    shape = read_recording_shape(recording, dataset)
    return None if shape is None else (shape[1], shape[2])


def pack_mask(mask: NDArray[np.generic]) -> NDArray[np.uint8]:
    return np.packbits(np.asarray(mask, dtype=bool).ravel())


def unpack_mask(packed: NDArray[np.uint8], shape: tuple[int, int]) -> BoolArray:
    height, width = int(shape[0]), int(shape[1])
    return np.unpackbits(packed, count=height * width).astype(bool).reshape(height, width)


def is_hypothesis_key(key: str) -> bool:
    return key.startswith(HYPOTHESIS_PREFIXES)


def split_arrays(arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """A ``poses.npz`` dictionary as ``(state, hypotheses)``."""

    state = {k: v for k, v in arrays.items() if not is_hypothesis_key(k)}
    hypotheses = {k: v for k, v in arrays.items() if is_hypothesis_key(k)}
    return state, hypotheses


@dataclass
class WorkspaceInfo:
    """The contents of ``workspace.json``."""

    name: str
    path: str
    recording: str
    frames: list[int]
    step: int
    created_at: str
    settings: dict[str, Any] = field(default_factory=dict)
    frame_count: int = 0
    image_shape: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkspaceInfo":
        known = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in data.items() if k in known}
        values["frames"] = [int(v) for v in values["frames"]]
        return cls(**values)


def _frame_range(first: int, last: int, step: int) -> NDArray[np.int64]:
    if step < 1:
        raise ValueError("step must be at least 1")
    if last < first:
        raise ValueError(f"empty frame range {first}..{last}")
    return np.arange(int(first), int(last) + 1, int(step), dtype=np.int64)


def _validate_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid workspace name {name!r}")
    return name


class Workspace:
    """One workspace directory; see the module docstring for the layout."""

    def __init__(self, path: Path, info: WorkspaceInfo) -> None:
        self.path = Path(path)
        self.info = info
        self.frame_index = _frame_range(info.frames[0], info.frames[1], info.step)
        self.n = int(len(self.frame_index))
        self._rows = {int(f): r for r, f in enumerate(self.frame_index.tolist())}
        self._lock = threading.RLock()
        self._chunk_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._edit_counter: int | None = None

    # ----------------------------------------------------------------- files

    @property
    def info_path(self) -> Path:
        return self.path / "workspace.json"

    @property
    def state_path(self) -> Path:
        return self.path / "state.npz"

    @property
    def hypotheses_path(self) -> Path:
        return self.path / "hypotheses.npz"

    @property
    def provenance_path(self) -> Path:
        return self.path / "provenance.npz"

    @property
    def masks_dir(self) -> Path:
        return self.path / "masks"

    @property
    def overrides_dir(self) -> Path:
        return self.path / "overrides" / "masks"

    @property
    def edits_path(self) -> Path:
        return self.path / "edits.jsonl"

    @property
    def recording(self) -> Path:
        return Path(self.info.recording)

    def save_info(self) -> None:
        with self._lock:
            _write_json_atomic(self.info_path, self.info.to_dict())

    # ------------------------------------------------------------ construction

    @classmethod
    def create(
        cls,
        root: Path,
        name: str,
        recording: Path,
        first: int,
        last: int,
        step: int = 1,
        settings: dict[str, Any] | None = None,
    ) -> "Workspace":
        """Make ``root/name`` for ``recording`` frames ``first..last`` by ``step``."""

        _validate_name(name)
        path = Path(root) / name
        if path.exists():
            raise FileExistsError(f"workspace {path} already exists")
        frame_index = _frame_range(first, last, step)
        recording_shape = read_recording_shape(Path(recording), str((settings or {}).get("dataset") or DATASET_PATH))
        if recording_shape is not None and int(frame_index[-1]) >= recording_shape[0]:
            raise ValueError(f"frame {int(frame_index[-1])} is beyond the {recording_shape[0]} frames of {recording}")
        shape = None if recording_shape is None else recording_shape[1:]
        info = WorkspaceInfo(
            name=name,
            path=str(path),
            recording=str(recording),
            frames=[int(frame_index[0]), int(frame_index[-1])],
            step=int(step),
            created_at=utc_now(),
            settings=dict(settings or {}),
            frame_count=int(len(frame_index)),
            image_shape=None if shape is None else [shape[0], shape[1]],
        )
        path.mkdir(parents=True)
        workspace = cls(path, info)
        workspace.save_info()
        return workspace

    @classmethod
    def open(cls, path: Path) -> "Workspace":
        path = Path(path)
        info_path = path / "workspace.json"
        if not info_path.exists():
            raise FileNotFoundError(f"{path} is not a workspace (no workspace.json)")
        info = WorkspaceInfo.from_dict(json.loads(info_path.read_text()))
        info.path = str(path)
        info.name = path.name
        return cls(path, info)

    # ------------------------------------------------------------------ rows

    def row_of(self, frame: int) -> int:
        try:
            return self._rows[int(frame)]
        except KeyError as error:
            raise ValueError(f"frame {frame} is not in workspace {self.info.name}") from error

    @property
    def image_shape(self) -> tuple[int, int] | None:
        """``(H, W)`` of the recording's frames, read once and cached in ``workspace.json``."""

        if self.info.image_shape is None:
            shape = read_image_shape(self.recording, str(self.info.settings.get("dataset") or DATASET_PATH))
            if shape is None:
                return None
            self.info.image_shape = [shape[0], shape[1]]
            self.save_info()
        return int(self.info.image_shape[0]), int(self.info.image_shape[1])

    # ----------------------------------------------------------------- arrays

    def load_state(self) -> dict[str, np.ndarray]:
        with self._lock:
            return _load_npz(self.state_path)

    def save_state(self, arrays: dict[str, np.ndarray]) -> None:
        with self._lock:
            _write_npz_atomic(self.state_path, arrays)

    def load_hypotheses(self) -> dict[str, np.ndarray]:
        with self._lock:
            return _load_npz(self.hypotheses_path)

    def save_hypotheses(self, arrays: dict[str, np.ndarray]) -> None:
        with self._lock:
            _write_npz_atomic(self.hypotheses_path, arrays)

    def load_arrays(self) -> dict[str, np.ndarray]:
        """State and hypotheses merged: the old ``poses.npz`` dictionary."""

        with self._lock:
            arrays = self.load_state()
            arrays.update(self.load_hypotheses())
            return arrays

    # ------------------------------------------------------------- provenance

    def _empty_provenance(self) -> dict[str, np.ndarray]:
        return {
            "algorithm": np.full(self.n, "", dtype=ALGORITHM_DTYPE),
            "job": np.full(self.n, "", dtype=JOB_DTYPE),
            "time": np.full(self.n, np.nan, dtype=np.float64),
        }

    def load_provenance(self) -> dict[str, np.ndarray]:
        with self._lock:
            stored = _load_npz(self.provenance_path)
            provenance = self._empty_provenance()
            for key, empty in provenance.items():
                if key in stored and len(stored[key]) == self.n:
                    provenance[key] = np.asarray(stored[key]).astype(empty.dtype)
            return provenance

    def set_provenance(self, rows: Sequence[int] | np.ndarray, algorithm: str, job: str, time: float | None = None) -> None:
        """Record that ``algorithm`` produced the poses of ``rows`` in ``job`` at ``time`` (now by default)."""

        index = np.asarray(rows, dtype=np.int64)
        if len(index) == 0:
            return
        if index.min() < 0 or index.max() >= self.n:
            raise ValueError("provenance rows out of range")
        with self._lock:
            provenance = self.load_provenance()
            provenance["algorithm"][index] = algorithm
            provenance["job"][index] = job
            provenance["time"][index] = time_module.time() if time is None else float(time)
            _write_npz_atomic(self.provenance_path, provenance)

    def provenance_counts(self) -> dict[str, int]:
        algorithms = self.load_provenance()["algorithm"]
        names, counts = np.unique(algorithms[algorithms != ""], return_counts=True)
        return {str(name): int(count) for name, count in zip(names, counts)}

    # ------------------------------------------------------------------ masks

    def _chunk_path(self, chunk: int) -> Path:
        return self.masks_dir / f"chunk_{chunk:05d}.npz"

    def _chunk_files(self) -> list[Path]:
        if not self.masks_dir.exists():
            return []
        return sorted(self.masks_dir.glob("chunk_[0-9][0-9][0-9][0-9][0-9].npz"))

    def _load_chunk(self, chunk: int) -> dict[str, np.ndarray] | None:
        with self._lock:
            if chunk in self._chunk_cache:
                self._chunk_cache.move_to_end(chunk)
                return self._chunk_cache[chunk]
            path = self._chunk_path(chunk)
            if not path.exists():
                return None
            loaded = _load_npz(path)
            self._chunk_cache[chunk] = loaded
            while len(self._chunk_cache) > _CHUNK_CACHE_SIZE:
                self._chunk_cache.popitem(last=False)
            return loaded

    def clear_mask_cache(self) -> None:
        """Refresh chunk reads after another workspace instance may have written masks."""
        with self._lock:
            self._chunk_cache.clear()

    def has_masks(self) -> bool:
        return len(self.mask_rows()) > 0

    def mask_rows(self) -> NDArray[np.int64]:
        """Rows with a stored or an override mask, ascending."""

        rows: list[np.ndarray] = []
        for path in self._chunk_files():
            with np.load(path, allow_pickle=False) as archive:
                rows.append(np.asarray(archive["rows"], dtype=np.int64))
        rows.append(np.asarray(self.override_rows(), dtype=np.int64))
        if not rows:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(rows))

    def stored_mask_rows(self) -> NDArray[np.int64]:
        """Rows with a mask in the chunk files (overrides excluded)."""

        rows = [np.asarray(_load_npz(p)["rows"], dtype=np.int64) for p in self._chunk_files()]
        return np.unique(np.concatenate(rows)) if rows else np.zeros(0, dtype=np.int64)

    def get_mask(self, row: int) -> BoolArray | None:
        """The stored (not override) mask of ``row``, ``None`` when there is none."""

        return self.get_masks([row]).get(int(row))

    def get_masks(self, rows: Sequence[int] | np.ndarray) -> dict[int, BoolArray]:
        out: dict[int, BoolArray] = {}
        for row in (int(r) for r in rows):
            chunk = self._load_chunk(row // MASK_CHUNK_ROWS)
            if chunk is None:
                continue
            hits = np.nonzero(chunk["rows"] == row)[0]
            if len(hits):
                out[row] = unpack_mask(chunk["packed"][hits[0]], tuple(chunk["shape"]))
        return out

    def set_masks(self, rows: Sequence[int] | np.ndarray, masks: Sequence[NDArray[np.generic]]) -> None:
        """Store ``masks`` for ``rows``, rewriting only the chunk files they land in."""

        index = [int(r) for r in rows]
        if len(index) != len(masks):
            raise ValueError("rows and masks differ in length")
        if not index:
            return
        if min(index) < 0 or max(index) >= self.n:
            raise ValueError("mask rows out of range")
        shape = tuple(int(v) for v in np.asarray(masks[0]).shape)
        if len(shape) != 2 or any(tuple(np.asarray(m).shape) != shape for m in masks):
            raise ValueError("masks must all be [H, W] arrays of one shape")
        self.set_packed_masks(index, [pack_mask(mask) for mask in masks], (shape[0], shape[1]))

    def set_packed_masks(self, rows: Sequence[int] | np.ndarray, packed: Sequence[NDArray[np.uint8]], shape: tuple[int, int]) -> None:
        """``set_masks`` for masks already packed with ``pack_mask``, so a caller can buffer a chunk's worth cheaply."""

        index = [int(r) for r in rows]
        if len(index) != len(packed):
            raise ValueError("rows and masks differ in length")
        if not index:
            return
        if min(index) < 0 or max(index) >= self.n:
            raise ValueError("mask rows out of range")
        shape = (int(shape[0]), int(shape[1]))
        by_chunk: dict[int, dict[int, np.ndarray]] = {}
        for row, bits in zip(index, packed, strict=True):
            by_chunk.setdefault(row // MASK_CHUNK_ROWS, {})[row] = np.asarray(bits, dtype=np.uint8)
        with self._lock:
            for chunk, new in by_chunk.items():
                self._write_chunk(chunk, new, shape)

    def _write_chunk(self, chunk: int, new: dict[int, np.ndarray], shape: tuple[int, int]) -> None:
        existing = self._load_chunk(chunk)
        merged: dict[int, np.ndarray] = {}
        if existing is not None:
            if tuple(int(v) for v in existing["shape"]) != shape:
                raise ValueError(f"mask shape {shape} differs from the stored shape {tuple(existing['shape'])}")
            merged.update({int(r): p for r, p in zip(existing["rows"].tolist(), existing["packed"])})
        merged.update(new)
        rows = np.array(sorted(merged), dtype=np.int64)
        packed = np.stack([merged[int(r)] for r in rows]).astype(np.uint8)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        payload = {"rows": rows, "packed": packed, "shape": np.array(shape, dtype=np.int64)}
        _write_npz_atomic(self._chunk_path(chunk), payload)
        self._chunk_cache[chunk] = payload
        self._chunk_cache.move_to_end(chunk)

    def _override_path(self, row: int) -> Path:
        return self.overrides_dir / f"{int(row):07d}.npz"

    def override_rows(self) -> list[int]:
        if not self.overrides_dir.exists():
            return []
        return sorted(int(p.stem) for p in self.overrides_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9][0-9].npz"))

    def set_override_mask(self, row: int, mask: NDArray[np.generic]) -> None:
        if not 0 <= int(row) < self.n:
            raise ValueError("override row out of range")
        labels = np.asarray(mask)
        if labels.ndim != 2 or (self.image_shape is not None and labels.shape != self.image_shape):
            raise ValueError(f"override mask shape {labels.shape} does not match frame {self.image_shape}")
        if not np.isin(labels, [0, 1, 255]).all():
            raise ValueError("override labels must be 0 background, 1 worm, or 255 ignore")
        with self._lock:
            self.overrides_dir.mkdir(parents=True, exist_ok=True)
            _write_npz_atomic(self._override_path(row), {"labels": labels.astype(np.uint8)})

    def get_override_mask(self, row: int) -> NDArray[np.uint8] | None:
        """Full labels (0 background, 1 worm, 255 ignore), including legacy binary overrides."""
        path = self._override_path(row)
        if not path.exists():
            return None
        stored = _load_npz(path)
        if "labels" in stored:
            return np.asarray(stored["labels"], dtype=np.uint8)
        return unpack_mask(stored["packed"], tuple(stored["shape"])).astype(np.uint8)

    def mask_revision(self, row: int) -> str:
        """Content fingerprint of the effective input, including override labels and source."""
        override = self.get_override_mask(row)
        mask = override if override is not None else self.get_mask(row)
        digest = hashlib.sha256(b"override" if override is not None else b"stored")
        if mask is not None:
            digest.update(str(mask.shape).encode())
            digest.update(np.asarray(mask, dtype=np.uint8).tobytes())
        return digest.hexdigest()

    def clear_override_mask(self, row: int) -> bool:
        """Remove the override of ``row``; whether there was one."""

        path = self._override_path(row)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True

    def effective_mask(self, row: int) -> BoolArray | None:
        override = self.get_override_mask(row)
        return (override == 1) if override is not None else self.get_mask(row)

    # ------------------------------------------------------------------ edits

    def _next_edit_id(self) -> str:
        if self._edit_counter is None:
            self._edit_counter = 0
            if self.edits_path.exists():
                for line in self.edits_path.read_text().splitlines():
                    if line.strip():
                        self._edit_counter = max(self._edit_counter, int(json.loads(line)["id"][1:]))
        self._edit_counter += 1
        return f"e{self._edit_counter:06d}"

    def append_edit(self, kind: str, payload: dict[str, Any]) -> str:
        """Log one intervention; returns its id."""

        with self._lock:
            edit_id = self._next_edit_id()
            record = {"id": edit_id, "kind": str(kind), "time": utc_now(), "payload": payload}
            line = json.dumps(record)
            with open(self.edits_path, "a") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return edit_id

    def edits(self) -> list[dict[str, Any]]:
        if not self.edits_path.exists():
            return []
        return [json.loads(line) for line in self.edits_path.read_text().splitlines() if line.strip()]

    # ---------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        state = self.load_state()
        fitted = np.asarray(state["fitted"], dtype=bool) if "fitted" in state else np.zeros(self.n, dtype=bool)
        out: dict[str, Any] = {
            "frame_count": self.n,
            "fitted": int(fitted.sum()),
            "has_masks": self.has_masks(),
            "mask_rows": int(len(self.mask_rows())),
            "override_rows": len(self.override_rows()),
            "has_hypotheses": self.hypotheses_path.exists(),
            "provenance": self.provenance_counts(),
            "edits": len(self.edits()),
            "has_prior": (self.path / "recording_prior.json").exists(),
        }
        if "iou" in state and fitted.any():
            iou = np.asarray(state["iou"], dtype=np.float64)[fitted]
            iou = iou[np.isfinite(iou)]
            if len(iou):
                out["iou"] = {
                    "median": float(np.median(iou)),
                    "p10": float(np.percentile(iou, 10)),
                    "min": float(iou.min()),
                    "frames_below_0.9": int((iou < 0.9).sum()),
                }
        return out


def list_workspaces(root: Path = DEFAULT_WORKSPACES_ROOT) -> list[WorkspaceInfo]:
    """Every workspace under ``root``, newest first; unreadable directories are skipped."""

    root = Path(root)
    if not root.exists():
        return []
    infos: list[WorkspaceInfo] = []
    for path in sorted(root.iterdir()):
        info_path = path / "workspace.json"
        if not info_path.exists():
            continue
        try:
            info = WorkspaceInfo.from_dict(json.loads(info_path.read_text()))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        info.path = str(path)
        info.name = path.name
        infos.append(info)
    infos.sort(key=lambda i: i.created_at, reverse=True)
    return infos
