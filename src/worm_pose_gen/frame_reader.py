"""Frames of a recording read at random, from an HDF5 dataset or a video in place.

Every reader of a recording asks for frames out of order: the timeline
scrubs, Labeling shows context frames, the body network reads time-lagged
neighbours, refits read ranges, and the flat field samples frames spread
over the recording.  :func:`open_frames` gives each the same reader for an
HDF5 recording (a ``[T,H,W]`` dataset, one frame per chunk) and for a video
(``.avi``, ``.mp4``) decoded through PyAV.

A video is indexed once: frame ``k`` is the ``k``-th frame the decoder
outputs, identified by its presentation timestamp, so a frame is the same
image whatever order frames are read in, with B-frames and variable frame
rate too.  Where packets carry presentation timestamps (MP4 and MOV) or
every packet is a keyframe (the lab's MJPEG), the packets are demuxed
without decoding and their timestamps, sorted, are the frames in display
order; an AVI's packets carry their decode order instead, which B-frames
reorder, so such a video is decoded once to number its frames.  An edit
list's pre-roll packets, which the decoder never outputs, are not frames,
and a truncated file's last packets, which do not decode, are dropped (an
interrupted acquisition stays readable up to its last whole frame).  The
sorted timestamps are also the frames' times (``times``).  The index is
cached under the data directory's cache and rebuilt when the file's size or
modification time changes.  A frame is decoded by seeking to the keyframe
at or before it and decoding forward to its timestamp; a read of the frame
after the last decoded one, or of a later frame of its group of pictures,
decodes forward without seeking, so consecutive reads cost one decode
each.  Decoded frames are kept in a small LRU cache.  A frame that does not
decode raises ``OSError``, as an unreadable HDF5 chunk does.

Frames are decoded to 8-bit gray through swscale, pixel for pixel what
``ffmpeg -pix_fmt gray`` (the old conversion to HDF5) wrote.
"""

from __future__ import annotations

from collections import OrderedDict
from fractions import Fraction
import hashlib
import operator
import os
from pathlib import Path
import tempfile
import threading
from typing import Iterator, Sequence
import warnings

import av
import h5py
import numpy as np
from numpy.typing import NDArray

from .settings import CACHE_DIR


DATASET_PATH = "/img_nir"
HDF5_SUFFIXES = (".h5", ".hdf5")
VIDEO_SUFFIXES = (".avi", ".mp4")
INDEX_DIR = CACHE_DIR / "video_index"
INDEX_VERSION = 3
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
    """A video's frames in display order: the timestamp each decodes with (in ``time_base`` seconds) and keyframe flags, plus its size and rate."""

    def __init__(
        self, pts: NDArray[np.int64], keyframe: NDArray[np.bool_], time_base: Fraction, shape: tuple[int, int], fps: float | None
    ) -> None:
        self.pts = pts
        self.keyframe = keyframe
        self.time_base = time_base
        self.shape = shape
        self.fps = fps

    @property
    def times(self) -> NDArray[np.float64]:
        """Each frame's time in seconds from the first: the ``k``-th smallest timestamp (a decoder-numbered AVI's are packet numbers)."""

        pts = np.sort(self.pts)
        return (pts - pts[0]).astype(np.float64) * float(self.time_base)

    @classmethod
    def build(cls, path: Path) -> VideoIndex:
        """Demux every packet of the first video stream; decode the whole stream only where packet timestamps are in decode order, else its end."""

        timestamps, keyframes = [], []
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError(f"{path}: no video stream")
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                # The flush packet at the end, an empty (dropped) frame, and an edit list's pre-roll, which the decoder never outputs.
                if packet.size == 0 or packet.is_discard:
                    continue
                timestamp = packet.pts if packet.pts is not None else packet.dts
                if timestamp is None:
                    raise ValueError(f"{path}: a packet has no timestamp")
                timestamps.append(timestamp)
                keyframes.append(packet.is_keyframe)
            if not timestamps:
                raise ValueError(f"{path} holds no frames")
            rate = stream.average_rate or stream.guessed_rate
            time_base = Fraction(stream.time_base)
            shape = (int(stream.codec_context.height), int(stream.codec_context.width))
            packet_pts, packet_key = np.asarray(timestamps, dtype=np.int64), np.asarray(keyframes, dtype=bool)
            if packet_key.all() or "mp4" in container.format.name.split(","):
                # Presentation timestamps: sorted, they are the frames in display order.
                order = np.argsort(packet_pts, kind="stable")
                pts, keyframe = packet_pts[order], packet_key[order]
                # Decode the end, from the last keyframe that yields a frame, and drop the frames after the last that decodes.
                for start in np.union1d(0, np.flatnonzero(keyframe))[::-1]:
                    decoded = _decoded_timestamps(container, stream, int(pts[start]))
                    if decoded:
                        last = int(np.searchsorted(pts, max(decoded)))
                        pts, keyframe = pts[: last + 1], keyframe[: last + 1]
                        break
                else:
                    raise ValueError(f"{path}: no frame decodes")
            else:
                # An AVI's packet timestamps count packets in decode order, which B-frames reorder: number the
                # frames as the decoder outputs them, each identified by the timestamp of the packet it came from.
                stream.thread_type = "AUTO"
                pts = np.asarray(_decoded_timestamps(container, stream, int(packet_pts.min())), dtype=np.int64)
                if not len(pts):
                    raise ValueError(f"{path}: no frame decodes")
                keyframe = np.isin(pts, packet_pts[packet_key])
        if np.any(np.diff(np.sort(pts)) == 0):
            raise ValueError(f"{path}: two frames share a timestamp")
        return cls(pts, keyframe, time_base, shape, None if rate is None else float(rate))

    @classmethod
    def load(cls, path: Path) -> VideoIndex:
        """The cached index of ``path`` while the file is unchanged, built and cached otherwise."""

        stat = path.stat()
        cache = INDEX_DIR / f"{path.stem}-{hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:16]}.npz"
        stamp = np.array([INDEX_VERSION, stat.st_size, stat.st_mtime_ns], dtype=np.int64)
        try:
            with open(cache, "rb") as handle, np.load(handle) as archive:
                if np.array_equal(archive["stamp"], stamp):
                    fps = float(archive["fps"])
                    numerator, denominator = (int(v) for v in archive["time_base"])
                    return cls(
                        archive["pts"], archive["keyframe"], Fraction(numerator, denominator),
                        tuple(int(v) for v in archive["shape"]), None if np.isnan(fps) else fps,
                    )
        except Exception:  # noqa: BLE001 - a missing, empty or damaged cache file is stale
            pass
        index = cls.build(path)
        try:
            index._save(cache, stamp)
        except OSError as error:
            # An unwritable cache directory: the video opens all the same, indexed again next time.
            warnings.warn(f"{path}: video index not cached ({error})", stacklevel=2)
        return index

    def _save(self, cache: Path, stamp: NDArray[np.int64]) -> None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Every writer owns its temporary file; a reader never sees a half-written index, even after a crash.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=f"{cache.name}.", suffix=".partial", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(
                    handle, stamp=stamp, pts=self.pts, keyframe=self.keyframe,
                    time_base=np.array([self.time_base.numerator, self.time_base.denominator]), shape=np.array(self.shape),
                    fps=np.nan if self.fps is None else self.fps,
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(cache)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _decoded_timestamps(container: av.container.InputContainer, stream: av.VideoStream, start: int) -> list[int]:
    """Timestamps of the frames decoded from the keyframe at or before ``start`` to the end; a packet that does not decode is skipped."""

    container.seek(start, stream=stream, backward=True, any_frame=False)
    timestamps = []
    for packet in container.demux(stream):
        if packet.size == 0 and packet.dts is not None:  # a dropped frame (the flush packet has no timestamp)
            continue
        try:
            frames = packet.decode()
        except av.FFmpegError:
            continue
        timestamps.extend(frame.pts if frame.pts is not None else frame.dts for frame in frames)
    return timestamps


def _decoded_frames(container: av.container.InputContainer, stream: av.VideoStream) -> Iterator[av.VideoFrame]:
    """The frames decoded from the container's position on (a function, not a method: the reader holds the generator)."""

    for packet in container.demux(stream):
        # Empty packets mark dropped frames, except the last, which flushes the decoder.
        if packet.size or packet.dts is None:
            yield from packet.decode()


class VideoFrames(FrameReader):
    """A video decoded through PyAV: frame ``k`` is the ``k``-th frame of :class:`VideoIndex`, decoded to 8-bit gray."""

    def __init__(self, path: Path, cache_bytes: int = CACHE_BYTES) -> None:
        self.path = Path(path)
        self.index = VideoIndex.load(self.path)
        self.frame_count = len(self.index.pts)
        self.shape = self.index.shape
        self.fps = self.index.fps
        self.times = self.index.times
        positions = np.arange(self.frame_count)
        # The keyframe at or before each frame (the first frame when none is).
        self._key_before = np.maximum.accumulate(np.where(self.index.keyframe, positions, 0))
        # Timestamps sorted, and their frames: a decoded frame's position (a decoder-numbered AVI's are not in order).
        self._by_pts = np.argsort(self.index.pts, kind="stable")
        self._sorted_pts = self.index.pts[self._by_pts]
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
        self._decoded = _decoded_frames(self._container, self._stream)
        self._position, self._key, self._key_seen = -1, key, False

    def _decode(self, index: int) -> NDArray[np.uint8]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        # Forward to the next frame or within the group of pictures; a seek otherwise (for intra-only
        # video, where every frame is a keyframe, a seek costs about what decoding one frame does).
        ahead = self._decoded is not None and self._position < index
        try:
            if not (ahead and (index == self._position + 1 or self._key_before[index] <= self._position)):
                self._seek(int(self._key_before[index]))
            return self._decode_forward(index)
        except Exception as error:
            # The decoder is left mid-stream: the next read seeks.
            self._decoded = None
            if isinstance(error, av.FFmpegError):
                raise OSError(f"{self.path}: frame {index} not decoded: {error}") from error
            raise

    def _decode_forward(self, index: int) -> NDArray[np.uint8]:
        while True:
            frame = next(self._decoded, None)
            if frame is None:
                raise OSError(f"{self.path}: frame {index} not decoded before the end of the video")
            timestamp = frame.pts if frame.pts is not None else frame.dts
            found = int(np.searchsorted(self._sorted_pts, timestamp)) if timestamp is not None else self.frame_count
            if found == self.frame_count or self._sorted_pts[found] != timestamp:
                continue
            position = int(self._by_pts[found])
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
