"""``frame_reader``: HDF5 recordings and videos read in place, frame ``k`` the same image whatever the access order."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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


def _write_video(path: Path, codec: str, frames: int = FRAMES) -> None:
    """MJPEG (every frame a keyframe), or H.264 with B-frames and an open group of 12 pictures."""

    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=20)
        stream.width, stream.height = WIDTH, HEIGHT
        if codec == "mjpeg":
            stream.pix_fmt = "yuvj420p"
        else:
            stream.pix_fmt = "yuv420p"
            stream.options = {"x264-params": "keyint=12:min-keyint=12:scenecut=0:bframes=3:open-gop=1"}
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(_coded_frame(index), format="gray")
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


class VideoFramesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.patch = mock.patch.object(frame_reader, "INDEX_DIR", cls.root / "video_index")
        cls.patch.start()
        cls.videos = {"avi": cls.root / "mjpeg.avi", "mp4": cls.root / "h264.mp4"}
        _write_video(cls.videos["avi"], "mjpeg")
        _write_video(cls.videos["mp4"], "libx264")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.patch.stop()
        cls.directory.cleanup()

    def _reader(self, kind: str) -> VideoFrames:
        # A one-frame cache, so every read below decodes.
        return VideoFrames(self.videos[kind], cache_bytes=1)

    def test_the_test_videos_are_what_they_claim(self) -> None:
        with open_frames(self.videos["mp4"]) as reader:
            self.assertEqual(int(reader.index.keyframe.sum()), FRAMES // 12)
        with av.open(str(self.videos["mp4"])) as container:
            decode_order = [p.pts for p in container.demux() if p.size]
        self.assertNotEqual(decode_order, sorted(decode_order), "no B-frames")
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
                with VideoFrames(self.videos[kind], cache_bytes=1) as reader:
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
                self.assertEqual((reader.frame_count, reader.shape, reader.fps), (9, (7, 5), 20.0))
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
