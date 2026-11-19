"""Tiny raw-data archive stores (``<dataset>/dataset.zarr``) written as the lab's converter writes them.

Zarr v3, Blosc zstd with bit-shuffle, ``behavior/nir`` one frame per chunk
in shards, consolidated metadata (flv-c-transfer ``docs/archive_spec.md``).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import zarr
from zarr.codecs import BloscCodec

from worm_pose_gen.recording_files import ARCHIVE_GROUP, ARCHIVE_NAME, ARCHIVE_NAMES


def write_archive(directory: Path, datasets: dict[str, np.ndarray], frames_per_shard: int = 4) -> Path:
    """The store ``directory/dataset.zarr`` whose behavior group holds ``datasets`` (keyed by the NIR h5's names)."""

    store = Path(directory) / ARCHIVE_NAME
    root = zarr.open_group(store, mode="w-")
    root.create_group("neural")
    behavior = root.create_group(ARCHIVE_GROUP)
    for name, data in datasets.items():
        parts = name.strip("/").split("/")
        parts[0] = ARCHIVE_NAMES.get(parts[0], parts[0])
        group = behavior
        for part in parts[:-1]:
            group = group.require_group(part)
        data = np.asarray(data)
        if parts == ["nir"]:
            chunks, shards = (1, *data.shape[1:]), (frames_per_shard, *data.shape[1:])
        else:
            chunks, shards = data.shape, None
        group.create_array(
            parts[-1], data=data, chunks=chunks, shards=shards,
            compressors=BloscCodec(cname="zstd", clevel=1, shuffle="bitshuffle"),
        )
    zarr.consolidate_metadata(store)
    return store


def archive_from_h5(h5_path: Path, directory: Path, frames_per_shard: int = 4) -> Path:
    """The h5 recording's every dataset, in a store under ``directory``."""

    datasets: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as handle:
        handle.visititems(lambda name, item: datasets.__setitem__(name, item[()]) if isinstance(item, h5py.Dataset) else None)
    return write_archive(directory, datasets, frames_per_shard)
