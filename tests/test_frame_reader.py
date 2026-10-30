"""``frame_reader``: HDF5 recordings and videos read in place, frame ``k`` the same image whatever the access order."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import gc
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import weakref

import av
import h5py
import numpy as np

from worm_pose_gen import frame_reader
from worm_pose_gen.frame_reader import Hdf5Frames, VideoFrames, VideoIndex, open_frames


HEIGHT, WIDTH, FRAMES = 48, 64, 60
BLOCK = 16
BITS = 6


def _coded_frame(index: int) -> np.ndarray:
    """Twelve flat 16x16 blocks: the index's six bits, then their complements, dark for 1 and bright for 0."""

    bits = [(index >> b) & 1 for b in range(BITS)]
    image = np.empty((HEIGHT, WIDTH), dtype=np.uint8)
    for block, bit in enumerate(bits + [1 - b for b in bits]):
        row, column = divmod(block, WIDTH // BLOCK)
        image[row * BLOCK : (row + 1) * BLOCK, column * BLOCK : (column + 1) * BLOCK] = 40 if bit else 215
    return image


def _decoded_index(image: np.ndarray) -> int:
    """The index a frame codes, checked against the complement blocks."""

    means = image.reshape(HEIGHT // BLOCK, BLOCK, WIDTH // BLOCK, BLOCK).mean(axis=(1, 3)).ravel()
    bits = means < 128
    if not np.all(bits[:BITS] != bits[BITS:]):
        raise ValueError("frame code and complement disagree")
    return int(sum(int(bit) << b for b, bit in enumerate(bits[:BITS])))


def _write_video(path: Path, codec: str, frames: int = FRAMES, milliseconds: list[int] | None = None) -> None:
    """MJPEG (every frame a keyframe), or H.264 with B-frames and an open group of 12 pictures, in the container of ``path``'s suffix.

    ``milliseconds`` gives each frame's presentation time (a variable frame rate); 20 frames/s otherwise.
    """

    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=20)
        stream.width, stream.height = WIDTH, HEIGHT
        if milliseconds is not None:
            stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        if codec == "mjpeg":
            stream.pix_fmt = "yuvj420p"
        else:
            stream.pix_fmt = "yuv420p"
            stream.options = {"x264-params": "keyint=12:min-keyint=12:scenecut=0:bframes=3:open-gop=1"}
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(_coded_frame(index), format="gray")
            if milliseconds is not None:
                frame.pts, frame.time_base = milliseconds[index], Fraction(1, 1000)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


class VideoFramesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.patch = mock.patch.object(frame_reader, "INDEX_DIR", cls.root / "video_index")
        cls.patch.start()
        cls.videos = {"avi": cls.root / "mjpeg.avi", "mp4": cls.root / "h264.mp4", "h264 avi": cls.root / "h264.avi"}
        _write_video(cls.videos["avi"], "mjpeg")
        _write_video(cls.videos["mp4"], "libx264")
        _write_video(cls.videos["h264 avi"], "libx264")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.patch.stop()
        cls.directory.cleanup()

    def _reader(self, kind: str) -> VideoFrames:
        # A one-frame cache, so every read below decodes.
        return self._reader_of(self.videos[kind])

    @staticmethod
    def _reader_of(path: Path) -> VideoFrames:
        return VideoFrames(path, cache_frames=1)

    def test_the_test_videos_are_what_they_claim(self) -> None:
        for kind in ("mp4", "h264 avi"):
            with open_frames(self.videos[kind]) as reader:
                self.assertEqual(int(reader.index.keyframe.sum()), FRAMES // 12)
        with av.open(str(self.videos["mp4"])) as container:
            decode_order = [p.pts for p in container.demux() if p.size]
        self.assertNotEqual(decode_order, sorted(decode_order), "no B-frames")
        # The AVI's packets number the decode order, so their timestamps say nothing of the B-frames' display order.
        with av.open(str(self.videos["h264 avi"])) as container:
            packets = [p.pts for p in container.demux() if p.size]
        with av.open(str(self.videos["h264 avi"])) as container:
            decoded = [f.pts for f in container.decode(video=0)]
        self.assertEqual(packets, sorted(packets))
        self.assertNotEqual(decoded, sorted(decoded), "no B-frames")
        with open_frames(self.videos["avi"]) as reader:
            self.assertTrue(reader.index.keyframe.all())

    def test_frame_count_shape_and_rate(self) -> None:
        for kind, path in self.videos.items():
            with self.subTest(kind), open_frames(path) as reader, av.open(str(path)) as container:
                decoded = sum(1 for _ in container.decode(video=0))
                self.assertEqual((reader.frame_count, decoded), (FRAMES, FRAMES))
                self.assertEqual(reader.shape, (HEIGHT, WIDTH))
                self.assertEqual(reader.fps, 20.0)

    def test_every_frame_decodes_its_own_index(self) -> None:
        for kind in self.videos:
            with self.subTest(kind), self._reader(kind) as reader:
                stack = reader[0:FRAMES]
                self.assertEqual((stack.dtype, stack.shape), (np.uint8, (FRAMES, HEIGHT, WIDTH)))
                self.assertEqual([_decoded_index(image) for image in stack], list(range(FRAMES)))

    def test_any_access_order_reads_the_sequential_frames(self) -> None:
        rng = np.random.default_rng(0)
        for kind in self.videos:
            with self.subTest(kind):
                with self._reader(kind) as reader:
                    sequential = np.stack([reader[k] for k in range(FRAMES)])
                self.assertEqual([_decoded_index(image) for image in sequential], list(range(FRAMES)))
                with self._reader(kind) as reader:
                    for k in rng.permutation(FRAMES):
                        np.testing.assert_array_equal(reader[int(k)], sequential[k])
                with self._reader(kind) as reader:
                    for k in reversed(range(FRAMES)):
                        np.testing.assert_array_equal(reader[k], sequential[k])
                with self._reader(kind) as reader:
                    np.testing.assert_array_equal(reader[17:41], sequential[17:41])
                    np.testing.assert_array_equal(reader[50:], sequential[50:])
                    indices = [33, 5, 5, 59, 0, 12, 11]
                    np.testing.assert_array_equal(reader.read(indices), sequential[indices])
                    np.testing.assert_array_equal(reader.read(range(20, 30)), sequential[20:30])
                    self.assertEqual(reader.read([]).shape, (0, HEIGHT, WIDTH))
                    with self.assertRaises(IndexError):
                        reader[FRAMES]
                # One reader shared by threads, as the app's server shares it.
                with VideoFrames(self.videos[kind], cache_frames=1) as reader:
                    order = [int(k) for k in rng.permutation(FRAMES)]
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        shared = list(pool.map(reader.__getitem__, order))
                    for k, image in zip(order, shared, strict=True):
                        np.testing.assert_array_equal(image, sequential[k])

    def test_a_seek_that_lands_past_its_keyframe_steps_back(self) -> None:
        with self._reader("mp4") as reader:
            expected = reader[0:FRAMES]
        with self._reader("mp4") as reader:
            keyframes = [int(p) for p in reader.index.pts[reader.index.keyframe]]
            container, sought = reader._container, []

            class LandsLateOnce:
                """The container, but its first seek lands on the next keyframe."""

                def seek(self, offset: int, **kwargs) -> None:
                    later = [p for p in keyframes if p > offset]
                    sought.append(offset)
                    container.seek(later[0] if len(sought) == 1 and later else offset, **kwargs)

                def demux(self, *args):
                    return container.demux(*args)

            reader._container = LandsLateOnce()
            np.testing.assert_array_equal(reader[30], expected[30])
            self.assertEqual(sought, [keyframes[2], keyframes[1]])
            reader._container = container

    def test_times_are_the_presentation_timestamps_from_the_first_frame(self) -> None:
        with open_frames(self.videos["avi"]) as reader:
            np.testing.assert_allclose(reader.times, np.arange(FRAMES) / 20.0)
        # A variable frame rate (a dropped frame, uneven gaps) keeps its times, whatever time base the muxer picks.
        milliseconds = [0, 50, 100, 200, 230, 300, 410, 450, 500, 650, 700, 760, 800, 850, 900, 1000]
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            path = Path(directory) / "vfr.mp4"
            _write_video(path, "libx264", frames=len(milliseconds), milliseconds=milliseconds)
            for _ in range(2):  # built, then loaded from the cache
                with open_frames(path) as reader:
                    np.testing.assert_allclose(reader.times, np.array(milliseconds) / 1000.0, atol=1e-9)
                    self.assertEqual([_decoded_index(reader[k]) for k in (3, 9, 15)], [3, 9, 15])

    def test_an_edit_list_pre_roll_is_not_frames(self) -> None:
        # An MP4 cut by stream copy at frame 27, mid group of pictures: frames 24-26 are decoded as references, not shown.
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            cut = Path(directory) / "cut.mp4"
            with av.open(str(self.videos["mp4"])) as source, av.open(str(cut), "w") as target:
                stream = source.streams.video[0]
                copy = target.add_stream_from_template(stream)
                packets = [p for p in source.demux(stream) if p.size]
                pts = sorted(p.pts for p in packets)
                key, shift = pts[24], pts[27]
                started = False
                for packet in packets:
                    started = started or (packet.is_keyframe and packet.pts == key)
                    if started:
                        packet.pts, packet.dts, packet.stream = packet.pts - shift, packet.dts - shift, copy
                        target.mux(packet)
            with av.open(str(cut)) as container:
                self.assertEqual(sum(p.is_discard for p in container.demux() if p.size), 3)
            with self._reader_of(cut) as reader:
                self.assertEqual(reader.frame_count, FRAMES - 27)
                np.testing.assert_allclose(reader.times, np.arange(FRAMES - 27) / 20.0)
                for k in [*np.random.default_rng(2).permutation(FRAMES - 27), *reversed(range(FRAMES - 27))]:
                    self.assertEqual(_decoded_index(reader[int(k)]), 27 + k)

    def test_a_frame_that_does_not_decode_is_an_os_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            path = Path(directory) / "damaged.avi"
            _write_damaged_mjpeg(path, damaged=17)
            with self._reader_of(path) as reader:
                self.assertEqual(reader.frame_count, FRAMES)
                for k in (16, 17, 18, 17, 16, 17):
                    if k == 17:
                        with self.assertRaisesRegex(OSError, "frame 17 not decoded") as raised:
                            reader[k]
                        self.assertIsInstance(raised.exception.__cause__, av.FFmpegError)
                    else:
                        self.assertEqual(_decoded_index(reader[k]), k)
                with self.assertRaises(OSError):
                    reader[10:20]

    def test_a_truncated_video_reads_up_to_its_last_whole_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            whole = Path(directory) / "whole.avi"
            _write_video(whole, "mjpeg")
            for fraction in (0.5, 0.7, 0.93):
                with self.subTest(fraction):
                    path = Path(directory) / f"cut{fraction}.avi"
                    data = whole.read_bytes()
                    path.write_bytes(data[: int(len(data) * fraction)])
                    with self._reader_of(path) as reader:
                        self.assertEqual(reader.frame_count, _frames_that_decode(path))
                        self.assertGreater(reader.frame_count, FRAMES * fraction - 5)
                        for k in reversed(range(reader.frame_count)):
                            self.assertEqual(_decoded_index(reader[k]), k)

    def test_a_reader_is_freed_by_reference_counting(self) -> None:
        # Readers that are never closed (one per segment in a process pool) must not wait for the cycle collector.
        enabled = gc.isenabled()
        gc.disable()
        try:
            for kind in self.videos:
                reader = VideoFrames(self.videos[kind])
                reader[5]
                reference = weakref.ref(reader)
                del reader
                self.assertIsNone(reference(), kind)
        finally:
            if enabled:
                gc.enable()

    def test_a_damaged_index_cache_is_rebuilt(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            path = Path(directory) / "clip.avi"
            _write_video(path, "mjpeg", frames=10)
            open_frames(path).close()
            (cache,) = (Path(directory) / "index").glob("clip-*.npz")
            whole = cache.read_bytes()
            for damaged in (b"", whole[: len(whole) // 2]):
                cache.write_bytes(damaged)
                with open_frames(path) as reader:
                    self.assertEqual(_decoded_index(reader[9]), 9)
                self.assertEqual(cache.read_bytes(), whole)

    def test_an_unwritable_index_directory_keeps_the_index_in_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "index"
            index.mkdir(mode=0o500)
            path = Path(directory) / "clip.avi"
            _write_video(path, "mjpeg", frames=10)
            try:
                with mock.patch.object(frame_reader, "INDEX_DIR", index), self.assertWarnsRegex(UserWarning, "video index not cached"):
                    with open_frames(path) as reader:
                        self.assertEqual((reader.frame_count, _decoded_index(reader[9])), (10, 9))
                self.assertEqual(list(index.iterdir()), [])
            finally:
                index.chmod(0o700)

    def test_index_cache_is_reused_and_rebuilt_when_the_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            path = Path(directory) / "clip.mp4"
            _write_video(path, "libx264", frames=24)
            build = VideoIndex.build
            with mock.patch.object(VideoIndex, "build", side_effect=build) as built:
                open_frames(path).close()
                with open_frames(path) as reader:
                    self.assertEqual(reader.frame_count, 24)
                self.assertEqual(built.call_count, 1)
                self.assertEqual(len(list((Path(directory) / "index").glob("clip-*.npz"))), 1)
                _write_video(path, "libx264", frames=36)
                with open_frames(path) as reader:
                    self.assertEqual(reader.frame_count, 36)
                    self.assertEqual(_decoded_index(reader[35]), 35)
                self.assertEqual(built.call_count, 2)
                # Same size, new modification time: rebuilt too.
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
                open_frames(path).close()
                self.assertEqual(built.call_count, 3)
                open_frames(path).close()
                self.assertEqual(built.call_count, 3)

    def test_a_file_without_a_video_stream_is_a_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            path = Path(directory) / "audio.avi"
            with av.open(str(path), "w") as container:
                stream = container.add_stream("pcm_s16le", rate=8000, layout="mono")
                packet = av.Packet(bytes(1600))
                packet.stream, packet.pts, packet.dts = stream, 0, 0
                container.mux(packet)
            with self.assertRaisesRegex(ValueError, "no video stream"):
                open_frames(path)


def _write_damaged_mjpeg(path: Path, damaged: int) -> None:
    """An MJPEG AVI of ``FRAMES`` coded frames whose frame ``damaged`` keeps only the first half of its JPEG."""

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mjpeg", rate=20)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuvj420p"
        for index in range(FRAMES):
            for packet in stream.encode(av.VideoFrame.from_ndarray(_coded_frame(index), format="gray")):
                if index == damaged:
                    data = bytes(packet)
                    half = av.Packet(data[: len(data) // 2])
                    half.stream, half.pts, half.dts, half.is_keyframe = stream, packet.pts, packet.dts, True
                    packet = half
                container.mux(packet)
        container.mux(stream.encode(None))


def _frames_that_decode(path: Path) -> int:
    """Frames a plain decode of the whole file yields, skipping the packets that fail."""

    count = 0
    with av.open(str(path)) as container:
        for packet in container.demux(video=0):
            try:
                count += len(packet.decode())
            except av.FFmpegError:
                pass
    return count


class Hdf5FramesTests(unittest.TestCase):
    def test_reads_match_h5py(self) -> None:
        rng = np.random.default_rng(1)
        stack = rng.integers(0, 256, (9, 7, 5), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rec.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset("/img_nir", data=stack, chunks=(1, 7, 5))
                handle.create_dataset("/camera/frames", data=stack.astype(np.uint16) * 3)
                handle["/img_nir"].attrs["fps"] = 20.0
            with open_frames(path) as reader, h5py.File(path, "r") as handle:
                self.assertIsInstance(reader, Hdf5Frames)
                data = handle["/img_nir"]
                self.assertEqual((reader.frame_count, reader.shape, reader.fps, reader.times), (9, (7, 5), 20.0, None))
                for k in range(9):
                    np.testing.assert_array_equal(reader[k], data[k])
                np.testing.assert_array_equal(reader[2:6], data[2:6])
                np.testing.assert_array_equal(reader.read([4, 1, 1, 8]), stack[[4, 1, 1, 8]])
                np.testing.assert_array_equal(reader.read([3, 4, 5]), data[3:6])
                with self.assertRaises(IndexError):
                    reader[9]
                with self.assertRaises(IndexError):
                    reader[-1]
            # Other dtypes are cast as NumPy casts them, as the readers always did.
            with open_frames(path, "/camera/frames") as reader, h5py.File(path, "r") as handle:
                self.assertIsNone(reader.fps)
                expected = np.asarray(handle["/camera/frames"][3], dtype=np.uint8)
                np.testing.assert_array_equal(reader[3], expected)
                self.assertEqual(reader[3].dtype, np.uint8)
            with self.assertRaisesRegex(KeyError, "no dataset /missing"):
                open_frames(path, "/missing")
            with self.assertRaisesRegex(ValueError, "not a recording"):
                open_frames(Path(directory) / "rec.tif")


if __name__ == "__main__":
    unittest.main()
