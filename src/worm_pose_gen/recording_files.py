"""What a recording is on disk, what it is called, and how its acquisition metadata is read.

A recording is one of three things:

- an HDF5 file (``.h5``, ``.hdf5``) written by the acquisition: the frames in
  ``/img_nir`` and the metadata beside them (``/pos_feature``, ``/pos_stage``,
  ``/img_metadata/...``);
- a video (``.avi``, ``.mp4``), which carries frames only;
- a dataset of the lab's raw-data archive: a Zarr v3 store
  ``<dataset>/dataset.zarr`` (a directory; flv-c-transfer
  ``docs/archive_spec.md``) whose ``behavior`` group holds the NIR h5's
  datasets under the same names, except ``img_nir``, which is ``nir``.

:func:`open_metadata` opens an HDF5 recording or an archive store as one
mapping under the h5's names, so a reader of ``/pos_stage`` or
``/img_metadata/img_timestamp`` is one code path for both.

A recording is known by :func:`recording_name`: the file name without its
extension, or for an archive store, which is always ``dataset.zarr``, the
name of the folder holding it (the dataset's uuid, or its legacy name under
the archive's ``by-prj/<project>/<uid>`` links).  Caches, workspaces and
exports are named by it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import zarr


HDF5_SUFFIXES = (".h5", ".hdf5")
VIDEO_SUFFIXES = (".avi", ".mp4")
ARCHIVE_NAME = "dataset.zarr"
ARCHIVE_GROUP = "behavior"
# The NIR h5's names that the archive's behavior group renames.
ARCHIVE_NAMES = {"img_nir": "nir"}

# The Rust codec pipeline reads the archive's sharded Blosc chunks several times faster than zarr's own.
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})


def recording_kind(path: Path | str) -> str | None:
    """``"h5"``, ``"video"`` or ``"archive"`` by the path's name (the path need not exist), else ``None``."""

    path = Path(path)
    if path.name == ARCHIVE_NAME:
        return "archive"
    suffix = path.suffix.lower()
    return "h5" if suffix in HDF5_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES else None


def is_recording(path: Path | str) -> bool:
    """Whether ``path`` exists as a recording: an HDF5 or video file, or an archive store's directory."""

    path = Path(path)
    kind = recording_kind(path)
    return kind is not None and (path.is_dir() if kind == "archive" else path.is_file())


def recording_name(path: Path | str) -> str:
    """The recording's name: the folder holding an archive store, else the file name without its extension."""

    path = Path(path)
    return path.parent.name if recording_kind(path) == "archive" else path.stem


def stamp_path(path: Path | str) -> Path:
    """The file whose size and modification time change when the recording does: an archive's root metadata, else the file."""

    path = Path(path)
    return path / "zarr.json" if recording_kind(path) == "archive" else path


def open_metadata(path: Path | str) -> h5py.File | ArchiveBehavior:
    """An HDF5 recording or an archive store as a mapping of the NIR h5's names (``/img_nir``, ``/pos_stage``, ...); close it after use.

    ``KeyError`` names a missing dataset; ``ValueError`` a path that is
    neither (a video carries no metadata).
    """

    kind = recording_kind(path)
    if kind == "h5":
        return h5py.File(path, "r")
    if kind == "archive":
        return ArchiveBehavior(Path(path))
    raise ValueError(f"{path}: not an HDF5 recording or a {ARCHIVE_NAME} store")


class ArchiveBehavior:
    """An archive store's ``behavior`` group, read only, under the NIR h5's names.

    Indexing takes an h5 path (a leading ``/`` is optional) and returns the
    zarr array or group, which slice, index by integer arrays and convert
    with ``np.asarray`` as h5py datasets do.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not (self.path / "zarr.json").is_file():
            raise FileNotFoundError(f"{self.path}: not a Zarr store (no zarr.json)")
        root = zarr.open_group(self.path, mode="r")
        try:
            self._group = root[ARCHIVE_GROUP]
        except KeyError as error:
            raise KeyError(f"{self.path}: no {ARCHIVE_GROUP} group") from error

    @staticmethod
    def _key(name: str) -> str:
        parts = name.strip("/").split("/")
        parts[0] = ARCHIVE_NAMES.get(parts[0], parts[0])
        return "/".join(parts)

    def __getitem__(self, name: str) -> Any:
        try:
            return self._group[self._key(name)]
        except KeyError as error:
            raise KeyError(f"{self.path}: no {name}") from error

    def __contains__(self, name: str) -> bool:
        try:
            self[name]
        except KeyError:
            return False
        return True

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default

    def close(self) -> None:
        """Nothing to release: zarr opens a shard file only while reading it."""

    def __enter__(self) -> ArchiveBehavior:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
