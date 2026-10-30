"""Videos in other containers (``.avi``), converted once to an HDF5 recording.

Every reader of a recording seeks frames at random, which an HDF5 dataset
chunked by frame serves directly and a compressed video does not, so a
video is decoded once (through imageio-ffmpeg's bundled ffmpeg) into the
recordings' layout rather than read in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import imageio_ffmpeg
import numpy as np

from .frame_reader import VIDEO_SUFFIXES


def is_video(path: str | Path) -> bool:
    """Whether ``path`` is a video ``convert_video`` reads rather than an HDF5 recording."""

    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def convert_video(source: Path, output: Path, dataset: str) -> dict[str, Any]:
    """Decode the video ``source`` to grayscale and write it to ``output`` as a ``[T,H,W]`` uint8 ``dataset``.

    The layout is the recordings': one uncompressed chunk per frame
    (compression saves little on camera noise and slows every random read).
    The source path and frame rate are kept as attributes of the dataset.
    The file is written beside ``output`` and renamed when complete, so a
    failed or interrupted conversion leaves no recording behind.
    """

    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = imageio_ffmpeg.read_frames(str(source), pix_fmt="gray", bpp=1)
    temporary = output.with_name(f"{output.name}.partial")
    try:
        meta = next(frames)
        width, height = (int(v) for v in meta["size"])
        with h5py.File(temporary, "w") as handle:
            data = handle.create_dataset(dataset, shape=(0, height, width), maxshape=(None, height, width), chunks=(1, height, width), dtype=np.uint8)
            data.attrs["source_video"] = str(source)
            data.attrs["fps"] = float(meta["fps"])
            count = 0
            for frame in frames:
                if count == data.shape[0]:
                    data.resize(count + 256, axis=0)
                data[count] = np.frombuffer(frame, dtype=np.uint8).reshape(height, width)
                count += 1
            if count == 0:
                raise ValueError(f"{source} holds no frames")
            data.resize(count, axis=0)
        temporary.replace(output)
    finally:
        frames.close()
        temporary.unlink(missing_ok=True)
    return {"frames": count, "height": height, "width": width, "fps": float(meta["fps"])}
