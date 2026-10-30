from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import h5py
import numpy as np
from PIL import Image

from tests.test_frame_reader import (
    FRAMES as VIDEO_FRAMES, HEIGHT as VIDEO_HEIGHT, WIDTH as VIDEO_WIDTH, _decoded_index, _write_damaged_mjpeg, _write_video,
)
from worm_pose_gen import frame_reader
from worm_pose_gen.frame_reader import open_frames
from worm_pose_gen.recordings import (
    RecordingInfo,
    RecordingProber,
    RecordingSource,
    find_recordings,
    list_recordings,
    probe_frames,
    probe_recording,
    thumbnail_png,
)


HEIGHT, WIDTH, FRAMES = 96, 128, 6


def _write_recording(path: Path, frames: int = FRAMES, height: int = HEIGHT, width: int = WIDTH) -> None:
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:height, :width]
    stack = np.empty((frames, height, width), dtype=np.uint8)
    for index in range(frames):
        background = np.full((height, width), 190.0)
        body = np.abs(yy - (48 + 10 * np.sin(xx / 20 + index))) < 6
        body &= (xx > 12) & (xx < 116)
        image = background - 80 * body + rng.normal(0, 3, (height, width))
        stack[index] = np.clip(image, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("/img_nir", data=stack)


def _write_garbage(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not an HDF5 file" * 64)


class RecordingsFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        index_dir = mock.patch.object(frame_reader, "INDEX_DIR", self.root / "video_index")
        index_dir.start()
        self.addCleanup(index_dir.stop)
        self.data = self.root / "data"
        self.rec_a = self.data / "2024-01-31" / "2024-01-31-02.h5"
        self.rec_b = self.data / "2024-05-28" / "2024-05-28-02.h5"
        self.bad = self.data / "2023-03-30" / "2023-03-30-01.h5"
        _write_recording(self.rec_a)
        _write_recording(self.rec_b, frames=4, height=64, width=80)
        _write_garbage(self.bad)
        (self.data / "2024-01-31" / "2024-01-31-02.nd2").write_bytes(b"ignored")

    def tearDown(self) -> None:
        self._directory.cleanup()

    def listing(self, **kwargs) -> list[RecordingInfo]:
        kwargs.setdefault("cache", None)
        return list_recordings([self.data], **kwargs)


class ListRecordingsTests(RecordingsFixture):
    def test_finds_h5_files_recursively_and_ignores_other_suffixes(self) -> None:
        found = find_recordings([self.data, self.root / "missing"])
        self.assertEqual([p.name for p in found], ["2023-03-30-01.h5", "2024-01-31-02.h5", "2024-05-28-02.h5"])

    def test_searches_two_folders_deep_skipping_hidden_folders(self) -> None:
        deep = self.data / "2024-01-31" / "day" / "2024-01-31-07.h5"
        too_deep = self.data / "a" / "b" / "c" / "too-deep.h5"
        hidden = self.data / ".trash" / "old.h5"
        for path in (deep, too_deep, hidden):
            _write_garbage(path)
        found = [p.name for p in find_recordings([self.data])]
        self.assertIn("2024-01-31-07.h5", found)
        self.assertNotIn("too-deep.h5", found)
        self.assertNotIn("old.h5", found)
        self.assertIn("too-deep.h5", [p.name for p in find_recordings([self.data / "a"])])

    def test_videos_are_probed_like_hdf5_recordings(self) -> None:
        _write_video(self.data / "2024-01-31" / "worm004.avi", "mjpeg")
        _write_video(self.data / "2024-01-31" / "worm005.MP4", "libx264")
        (self.data / "2024-01-31" / "damaged.avi").write_bytes(b"not a video" * 64)
        cache = self.root / "index.json"
        pending = {i.name: i for i in list_recordings([self.data], cache=cache, probe=False)}
        self.assertTrue(pending["worm005"].pending)
        prober = RecordingProber(cache)
        prober.submit((Path(info.path), info.dataset) for info in pending.values() if info.pending)
        prober._thread.join(timeout=30)
        listed = {i.name: i for i in list_recordings([self.data], cache=cache, probe=False)}
        for name in ("worm004", "worm005"):
            info = listed[name]
            self.assertEqual((info.frames, info.height, info.width, info.readable, info.pending), (VIDEO_FRAMES, VIDEO_HEIGHT, VIDEO_WIDTH, True, False))
        self.assertFalse(listed["damaged"].readable)
        self.assertTrue(listed["damaged"].error)
        video = self.data / "2024-01-31" / "worm005.MP4"
        thumbnail = np.asarray(Image.open(io.BytesIO(thumbnail_png(video, 7, scale=1.0, dataset_root=self.root / "none"))))
        self.assertEqual(_decoded_index(thumbnail), 7)

    def test_a_damaged_video_keeps_its_readable_frames(self) -> None:
        # An interrupted acquisition ends in part of a frame; a damaged frame mid-file is sampled by the flat field (all 60 are).
        truncated, damaged = self.data / "truncated.avi", self.data / "damaged.avi"
        _write_video(truncated, "mjpeg")
        truncated.write_bytes(truncated.read_bytes()[: int(truncated.stat().st_size * 0.93)])
        bad = 20  # not one the probe reads (first, middle, last)
        _write_damaged_mjpeg(damaged, damaged=bad)
        for path in (truncated, damaged):
            with self.subTest(path.name):
                facts = probe_recording(path)
                self.assertTrue(facts["readable"], facts["error"])
                source = RecordingSource(path, self.root / "fields")
                try:
                    self.assertEqual(source.frame_count, facts["frames"])
                    self.assertEqual(_decoded_index(source.read(source.frame_count - 1)), source.frame_count - 1)
                    self.assertIsNotNone(source.flat_field())
                finally:
                    source.close()
        self.assertLess(probe_recording(truncated)["frames"], VIDEO_FRAMES)
        with open_frames(damaged) as frames:
            with self.assertRaises(OSError):
                frames[bad]

    def test_without_probing_new_files_are_pending_until_a_prober_saves_them(self) -> None:
        cache = self.root / "index.json"
        pending = list_recordings([self.data], cache=cache, probe=False)
        self.assertTrue(all(info.pending and info.frames is None for info in pending))
        prober = RecordingProber(cache)
        prober.submit((Path(info.path), info.dataset) for info in pending)
        prober._thread.join(timeout=30)
        listed = {info.name: info for info in list_recordings([self.data], cache=cache, probe=False)}
        self.assertFalse(any(info.pending for info in listed.values()))
        self.assertEqual(listed["2024-05-28-02"].frames, 4)
        self.assertFalse(listed["2023-03-30-01"].readable)

    def test_a_root_that_links_to_another_lists_each_recording_once(self) -> None:
        link = self.root / "link"
        link.symlink_to(self.data)
        found = find_recordings([link, self.data, self.rec_a])
        self.assertEqual(found, find_recordings([self.data]))
        self.assertEqual(len(found), 3)

    def test_readable_recordings_report_shape_and_garbage_reports_error(self) -> None:
        infos = self.listing()
        by_name = {info.name: info for info in infos}
        self.assertEqual(sorted(by_name), ["2023-03-30-01", "2024-01-31-02", "2024-05-28-02"])
        a = by_name["2024-01-31-02"]
        self.assertTrue(a.readable)
        self.assertIsNone(a.error)
        self.assertEqual((a.frames, a.height, a.width), (FRAMES, HEIGHT, WIDTH))
        self.assertEqual(a.path, str(self.rec_a))
        self.assertEqual(a.size_bytes, self.rec_a.stat().st_size)
        self.assertTrue(a.modified_at.endswith("+00:00"))
        self.assertEqual(a.dataset, "/img_nir")
        b = by_name["2024-05-28-02"]
        self.assertEqual((b.frames, b.height, b.width), (4, 64, 80))
        bad = by_name["2023-03-30-01"]
        self.assertFalse(bad.readable)
        self.assertIsNotNone(bad.error)
        self.assertEqual((bad.frames, bad.height, bad.width), (None, None, None))
        self.assertEqual(bad.size_bytes, self.bad.stat().st_size)
        json.dumps([info.to_dict() for info in infos])  # the payload is JSON-serialisable

    def test_probe_rejects_files_without_the_dataset(self) -> None:
        other = self.data / "other.h5"
        with h5py.File(other, "w") as handle:
            handle.create_dataset("/something", data=np.zeros((2, 2), dtype=np.uint8))
        facts = probe_recording(other)
        self.assertFalse(facts["readable"])
        self.assertIn("img_nir", facts["error"])

    def test_cache_is_reused_until_the_file_changes(self) -> None:
        cache = self.root / "index" / "recordings.json"
        self.listing(cache=cache)
        self.assertTrue(cache.exists())
        data = json.loads(cache.read_text())
        self.assertEqual(data["version"], 1)
        self.assertEqual(set(data["entries"]), {str(self.rec_a), str(self.rec_b), str(self.bad)})
        # Plant a sentinel in the cached entry: an unchanged file must come back from the cache verbatim.
        data["entries"][str(self.rec_a)]["frames"] = 999
        cache.write_text(json.dumps(data))
        infos = {info.name: info for info in self.listing(cache=cache)}
        self.assertEqual(infos["2024-01-31-02"].frames, 999)
        self.assertEqual(infos["2024-05-28-02"].frames, 4)
        # A rewritten file (new size and mtime) is probed again and the cache refreshed.
        _write_recording(self.rec_a, frames=7)
        os.utime(self.rec_a, ns=(self.rec_a.stat().st_atime_ns, self.rec_a.stat().st_mtime_ns + 5_000_000_000))
        infos = {info.name: info for info in self.listing(cache=cache)}
        self.assertEqual(infos["2024-01-31-02"].frames, 7)
        data = json.loads(cache.read_text())
        self.assertEqual(data["entries"][str(self.rec_a)]["frames"], 7)
        # Garbage entries are cached too, so the failing open is not repeated every scan.
        self.assertFalse(data["entries"][str(self.bad)]["readable"])

    def test_concurrent_listings_share_one_cache_file(self) -> None:
        # The app serves listings from a thread pool; each writes the cache when it probed something.
        cache = self.root / "index" / "recordings.json"
        errors: list[BaseException] = []
        counts: list[int] = []

        def listing() -> None:
            try:
                counts.append(len(self.listing(cache=cache)))
            except BaseException as error:  # noqa: BLE001 - collected for the assertion
                errors.append(error)

        for _ in range(3):
            if cache.exists():
                cache.unlink()
            threads = [threading.Thread(target=listing) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(counts, [3] * 18)
        self.assertEqual(json.loads(cache.read_text())["version"], 1)
        self.assertEqual([p.name for p in cache.parent.iterdir()], ["recordings.json"])

    def test_cache_survives_a_corrupt_index_file(self) -> None:
        cache = self.root / "recordings.json"
        cache.write_text("{not json")
        infos = self.listing(cache=cache)
        self.assertEqual(len(infos), 3)
        self.assertEqual(json.loads(cache.read_text())["version"], 1)

    def test_probe_samples_first_middle_and_last_frames(self) -> None:
        self.assertEqual(probe_frames(0), ())
        self.assertEqual(probe_frames(1), (0,))
        self.assertEqual(probe_frames(2), (0, 1))
        self.assertEqual(probe_frames(20481), (0, 10240, 20480))

    def test_same_stem_in_two_directories_lists_both_paths(self) -> None:
        copy = self.data / "2024-05-28" / "2024-01-31-02.h5"
        copy.write_bytes(self.rec_a.read_bytes())
        infos = [info for info in self.listing() if info.name == "2024-01-31-02"]
        self.assertEqual([info.path for info in infos], [str(self.rec_a), str(copy)])
        self.assertTrue(all(info.readable for info in infos))

    def test_missing_roots_yield_an_empty_catalog(self) -> None:
        self.assertEqual(list_recordings([self.root / "nowhere"]), [])


class ThumbnailTests(RecordingsFixture):
    def test_directory_listing(self) -> None:
        from worm_pose_gen.recordings import list_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conventional = root / "a" / "conv.h5"
            _write_recording(conventional)
            for name in ("other.h5", "two.h5"):
                with h5py.File(root / "a" / name, "w") as handle:
                    handle.create_dataset("/x", data=np.zeros((3, 32, 40), dtype=np.uint8))
            (root / "a" / ".hidden").mkdir()
            (root / "a" / "notes.txt").write_text("x")
            (root / "a" / "movie.AVI").write_bytes(b"")
            (root / "a" / "sub").mkdir()
            listing = list_directory(root / "a")
            self.assertEqual(listing["parent"], str(root.resolve()))
            self.assertEqual([(e["name"], e["kind"]) for e in listing["entries"]], [("sub", "dir"), ("conv.h5", "h5"), ("movie.AVI", "video"), ("other.h5", "h5"), ("two.h5", "h5")])
            self.assertEqual(listing["entries"][1]["size_bytes"], conventional.stat().st_size)
            self.assertIsNone(list_directory(Path("/"))["parent"])
            everything = list_directory(root / "a", all_files=True)
            self.assertEqual([(e["name"], e["kind"]) for e in everything["entries"]], [("sub", "dir"), ("conv.h5", "h5"), ("movie.AVI", "video"), ("notes.txt", "file"), ("other.h5", "h5"), ("two.h5", "h5")])
            with self.assertRaises(FileNotFoundError):
                list_directory(root / "missing")
            with self.assertRaises(NotADirectoryError):
                list_directory(conventional)

    def test_a_setup_dataset_other_than_img_nir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "roots" / "cam.h5"
            _write_recording(root / "plain.h5")
            camera.parent.mkdir()
            with h5py.File(root / "plain.h5", "r") as source, h5py.File(camera, "w") as handle:
                handle.create_dataset("/camera/frames", data=source["/img_nir"][...])
            infos = list_recordings([root / "roots"], cache=root / "index.json", dataset="/camera/frames")
            self.assertEqual([(i.name, i.dataset, i.frames, i.readable) for i in infos], [("cam", "/camera/frames", FRAMES, True)])
            png = thumbnail_png(camera, 1, 0.5, dataset_root=root / "dataset", dataset="/camera/frames")
            self.assertEqual(Image.open(io.BytesIO(png)).size, (WIDTH // 2, HEIGHT // 2))
            # Under the conventional dataset name the same file probes as unreadable.
            plain = list_recordings([root / "roots"], cache=None)
            self.assertFalse(plain[0].readable)
            self.assertIn("no dataset /img_nir", plain[0].error)

    def test_raw_thumbnail_is_a_scaled_png(self) -> None:
        png = thumbnail_png(self.rec_a, 2, scale=0.25, dataset_root=self.root / "no-dataset")
        self.assertTrue(png.startswith(b"\x89PNG"))
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.size, (WIDTH // 4, HEIGHT // 4))
        self.assertEqual(image.mode, "L")
        full = np.asarray(Image.open(io.BytesIO(thumbnail_png(self.rec_a, 2, scale=1.0, dataset_root=self.root / "no-dataset"))))
        with h5py.File(self.rec_a, "r") as handle:
            self.assertTrue(np.array_equal(full, handle["/img_nir"][2]))
        # The dark worm body stays darker than the background after downscaling.
        small = np.asarray(image, dtype=np.float64)
        self.assertLess(small[:, 16].min(), small[0, 16] - 30)

    def test_flat_fielded_thumbnail_uses_the_cached_field(self) -> None:
        dataset_root = self.root / "dataset"
        source = RecordingSource(self.rec_a, dataset_root / "flat_fields")
        source.flat_field()
        source.close()
        self.assertTrue((dataset_root / "flat_fields" / "2024-01-31-02.npz").exists())
        png = thumbnail_png(self.rec_a, 1, scale=0.5, dataset_root=dataset_root)
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.size, (WIDTH // 2, HEIGHT // 2))
        raw = np.asarray(Image.open(io.BytesIO(thumbnail_png(self.rec_a, 1, scale=0.5, dataset_root=self.root / "none"))))
        self.assertFalse(np.array_equal(np.asarray(image), raw))

    def test_thumbnail_errors_are_explicit(self) -> None:
        with self.assertRaises(IndexError):
            thumbnail_png(self.rec_a, FRAMES, dataset_root=self.root / "none")
        with self.assertRaises(OSError):
            thumbnail_png(self.bad, 0, dataset_root=self.root / "none")
        with self.assertRaises(ValueError):
            thumbnail_png(self.rec_a, 0, scale=0.0, dataset_root=self.root / "none")


if __name__ == "__main__":
    unittest.main()
