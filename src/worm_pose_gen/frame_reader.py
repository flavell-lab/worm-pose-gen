"""Frames of a recording read at random, from an HDF5 dataset or a video in place.

Every reader of a recording asks for frames out of order: the timeline
scrubs, Labeling shows context frames, the body network reads time-lagged
neighbours, refits read ranges, and the flat field samples frames spread
over the recording.  :func:`open_frames` gives each the same reader for an
HDF5 recording (a ``[T,H,W]`` dataset, one frame per chunk) and for a video
(``.avi``, ``.mp4``) decoded through PyAV.

A video is indexed once: its packets are demuxed without decoding, and
frame ``k`` is the ``k``-th presentation timestamp in display order, so a
frame is the same image whatever order frames are read in, with B-frames
and variable frame rate too; the timestamps are also the frames' times
(``times``).  The index is cached under the data directory's cache and
rebuilt when the file's size or modification time changes.  A frame is
decoded by seeking to the keyframe at or before it and decoding forward to
its timestamp; a read of the frame after the last decoded one, or of a
later frame of its group of pictures, decodes forward without seeking, so
consecutive reads cost one decode each.  Decoded frames are kept in a small
LRU cache.

Frames are decoded to 8-bit gray through swscale, pixel for pixel what
``ffmpeg -pix_fmt gray`` (the old conversion to HDF5) wrote.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import hashlib
import operator
from pathlib import Path
import tempfile
import threading
from typing import Iterator, Sequence

import av
import h5py
import numpy as np
from numpy.typing import NDArray

from .settings import CACHE_DIR


DATASET_PATH = "/img_nir"
HDF5_SUFFIXES = (".h5", ".hdf5")
VIDEO_SUFFIXES = (".avi", ".mp4")
INDEX_DIR = CACHE_DIR / "video_index"
INDEX_VERSION = 2
# Decoded frames kept per video reader (about 90 frames of 968x732).
CACHE_BYTES = 64 * 2**20
# Decoder threads per video: frame threads speed H.264 up about as many times (130 frames/s single-threaded).
DECODE_THREADS = 4


def open_frames(path: Path | str, dataset: str = DATASET_PATH) -> FrameReader:
    """The reader of the recording at ``path``, chosen by suffix; ``dataset`` names the HDF5 dataset and is ignored for videos."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in HDF5_SUFFIXES:
        return Hdf5Frames(path, dataset)
    if suffix in VIDEO_SUFFIXES:
        return VideoFrames(path)
    raise ValueError(f"{path}: not a recording ({', '.join((*HDF5_SUFFIXES, *VIDEO_SUFFIXES))})")


class FrameReader:
    """One recording's frames as uint8 ``[H, W]`` images, safe to share between threads.

    ``reader[k]`` is frame ``k``, ``reader[a:b]`` the ``[b - a, H, W]`` stack
    of a range (step 1), and :meth:`read` any list of frames.
    """

    path: Path
    frame_count: int
    shape: tuple[int, int]
    # Frames per second, when the file records it.
    fps: float | None
    # Each frame's time in seconds from the first frame, when the frames carry it: a video's presentation
    # timestamps.  None for HDF5, whose camera timestamps live in the acquisition metadata (export_table).
    times: NDArray[np.float64] | None

    def __getitem__(self, key: int | slice) -> NDArray[np.uint8]:
        if isinstance(key, slice):
            start, stop, step = key.indices(self.frame_count)
            if step != 1:
                raise ValueError("frame ranges have step 1")
            return self._range(start, max(start, stop))
        index = operator.index(key)
        if not 0 <= index < self.frame_count:
            raise IndexError(f"frame {index} out of range for {self.frame_count} frames")
        return self._frame(index)

    def read(self, indices: Sequence[int]) -> NDArray[np.uint8]:
        """Frames ``indices`` (any order, repeats allowed) as ``[N, H, W]``; one range read when they are consecutive."""

        indices = [int(i) for i in indices]
        if not indices:
            return np.empty((0, *self.shape), dtype=np.uint8)
        if indices == list(range(indices[0], indices[0] + len(indices))) and 0 <= indices[0] and indices[-1] < self.frame_count:
            return self._range(indices[0], indices[-1] + 1)
        # Ascending, so a video decodes forward through nearby frames.
        frames = {i: self[i] for i in sorted(set(indices))}
        return np.stack([frames[i] for i in indices])

    def _frame(self, index: int) -> NDArray[np.uint8]:
        raise NotImplementedError

    def _range(self, start: int, stop: int) -> NDArray[np.uint8]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> FrameReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class Hdf5Frames(FrameReader):
    """A ``[T, H, W]`` HDF5 dataset; frames are cast to uint8 as read."""

    def __init__(self, path: Path, dataset: str = DATASET_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._handle = h5py.File(self.path, "r")
        try:
            if dataset not in self._handle:
                raise KeyError(f"{self.path}: no dataset {dataset}")
            self._data = self._handle[dataset]
            if self._data.ndim != 3:
                raise ValueError(f"{self.path}: expected a [T,H,W] dataset, got shape {tuple(self._data.shape)}")
        except Exception:
            self._handle.close()
            raise
        self.frame_count = int(self._data.shape[0])
        self.shape = (int(self._data.shape[1]), int(self._data.shape[2]))
        fps = self._data.attrs.get("fps")
        self.fps = None if fps is None else float(fps)
        self.times = None

    def _frame(self, index: int) -> NDArray[np.uint8]:
        with self._lock:
            return np.asarray(self._data[index], dtype=np.uint8)

    def _range(self, start: int, stop: int) -> NDArray[np.uint8]:
        with self._lock:
            return np.asarray(self._data[start:stop], dtype=np.uint8)

    def close(self) -> None:
        with self._lock:
            self._handle.close()


class VideoIndex:
    """A video's frames in display order: presentation timestamps (in ``time_base`` seconds) and keyframe flags, plus its size and rate."""

    def __init__(
        self, pts: NDArray[np.int64], keyframe: NDArray[np.bool_], time_base: Fraction, shape: tuple[int, int], fps: float | None
    ) -> None:
        self.pts = pts
        self.keyframe = keyframe
        self.time_base = time_base
        self.shape = shape
        self.fps = fps

    @classmethod
    def build(cls, path: Path) -> VideoIndex:
        """Demux every packet of the first video stream without decoding it."""

        timestamps, keyframes = [], []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                if packet.size == 0:  # the flush packet at the end, or an empty (dropped) frame
                    continue
                timestamp = packet.pts if packet.pts is not None else packet.dts
                if timestamp is None:
                    raise ValueError(f"{path}: a packet has no timestamp")
                timestamps.append(timestamp)
                keyframes.append(packet.is_keyframe)
            rate = stream.average_rate or stream.guessed_rate
            time_base = Fraction(stream.time_base)
            shape = (int(stream.codec_context.height), int(stream.codec_context.width))
        if not timestamps:
            raise ValueError(f"{path} holds no frames")
        order = np.argsort(np.asarray(timestamps, dtype=np.int64), kind="stable")
        pts = np.asarray(timestamps, dtype=np.int64)[order]
        if np.any(np.diff(pts) == 0):
            raise ValueError(f"{path}: two frames share a timestamp")
        return cls(pts, np.asarray(keyframes, dtype=bool)[order], time_base, shape, None if rate is None else float(rate))

    @classmethod
    def load(cls, path: Path) -> VideoIndex:
        """The cached index of ``path`` while the file is unchanged, built and cached otherwise."""

        stat = path.stat()
        cache = INDEX_DIR / f"{path.stem}-{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]}.npz"
        stamp = np.array([INDEX_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)
        try:
            with np.load(cache) as archive:
                if np.array_equal(archive["stamp"], stamp):
                    fps = float(archive["fps"])
                    numerator, denominator = (int(v) for v in archive["time_base"])
                    return cls(
                        archive["pts"], archive["keyframe"], Fraction(numerator, denominator),
                        tuple(int(v) for v in archive["shape"]), None if np.isnan(fps) else fps,
                    )
        except (OSError, KeyError, ValueError):
            pass
        index = cls.build(path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Every writer owns its temporary file; a reader never sees a half-written index.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=f"{cache.name}.", suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(
                    handle, stamp=stamp, pts=index.pts, keyframe=index.keyframe,
                    time_base=np.array([index.time_base.numerator, index.time_base.denominator]), shape=np.array(index.shape),
                    fps=np.nan if index.fps is None else index.fps,
                )
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return index


class VideoFrames(FrameReader):
    """A video decoded through PyAV: frame ``k`` is the ``k``-th frame of :class:`VideoIndex`, decoded to 8-bit gray."""

    def __init__(self, path: Path, cache_bytes: int = CACHE_BYTES) -> None:
        self.path = Path(path)
        self.index = VideoIndex.load(self.path)
        self.frame_count = len(self.index.pts)
        self.shape = self.index.shape
        self.fps = self.index.fps
        self.times = (self.index.pts - self.index.pts[0]).astype(np.float64) * float(self.index.time_base)
        positions = np.arange(self.frame_count)
        # The keyframe at or before each frame (the first frame when none is).
        self._key_before = np.maximum.accumulate(np.where(self.index.keyframe, positions, 0))
        self._lock = threading.Lock()
        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._stream.codec_context.thread_count = DECODE_THREADS
        self._decoded: Iterator[av.VideoFrame] | None = None
        # Display index of the last decoded frame (-1 before the first), the keyframe sought last,
        # and whether that keyframe has been decoded since: frames before it are leading frames
        # whose references precede the seek, and a later first frame means the seek landed late.
        self._position = -1
        self._key = 0
        self._key_seen = False
        self._cache: OrderedDict[int, NDArray[np.uint8]] = OrderedDict()
        self._cache_frames = max(1, cache_bytes // (self.shape[0] * self.shape[1]))

    def _frame(self, index: int) -> NDArray[np.uint8]:
        with self._lock:
            return self._decode(index).copy()

    def _range(self, start: int, stop: int) -> NDArray[np.uint8]:
        out = np.empty((stop - start, *self.shape), dtype=np.uint8)
        with self._lock:
            for row, index in enumerate(range(start, stop)):
                out[row] = self._decode(index)
        return out

    def _seek(self, key: int) -> None:
        self._container.seek(int(self.index.pts[key]), stream=self._stream, backward=True, any_frame=False)
        self._decoded = self._frames()
        self._position, self._key, self._key_seen = -1, key, False

    def _frames(self) -> Iterator[av.VideoFrame]:
        for packet in self._container.demux(self._stream):
            # Empty packets mark dropped frames, except the last, which flushes the decoder.
            if packet.size or packet.dts is None:
                yield from packet.decode()

    def _decode(self, index: int) -> NDArray[np.uint8]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        # Forward to the next frame or within the group of pictures; a seek otherwise (for intra-only
        # video, where every frame is a keyframe, a seek costs about what decoding one frame does).
        ahead = self._decoded is not None and self._position < index
        if not (ahead and (index == self._position + 1 or self._key_before[index] <= self._position)):
            self._seek(int(self._key_before[index]))
        try:
            return self._decode_forward(index)
        except Exception:
            # The decoder is left mid-stream: the next read seeks.
            self._decoded = None
            raise

    def _decode_forward(self, index: int) -> NDArray[np.uint8]:
        while True:
            frame = next(self._decoded, None)
            if frame is None:
                raise OSError(f"{self.path}: frame {index} not decoded before the end of the video")
            timestamp = frame.pts if frame.pts is not None else frame.dts
            position = int(np.searchsorted(self.index.pts, timestamp)) if timestamp is not None else self.frame_count
            if position == self.frame_count or self.index.pts[position] != timestamp:
                continue
            if not self._key_seen and position >= self._key:
                if position > self._key and self._key > 0:
                    # The seek landed past its keyframe: seek to the one before.
                    self._seek(int(self._key_before[self._key - 1]))
                    continue
                self._key_seen = True
            if not self._key_seen:
                continue
            self._position = position
            image = frame.to_ndarray(format="gray")
            self._cache[position] = image
            self._cache.move_to_end(position)
            while len(self._cache) > self._cache_frames:
                self._cache.popitem(last=False)
            if position == index:
                return image
            if position > index:
                raise OSError(f"{self.path}: frame {index} not decoded (decoding went from frame {self._key} to {position})")

    def close(self) -> None:
        with self._lock:
            self._decoded = None
            self._cache.clear()
            self._container.close()
