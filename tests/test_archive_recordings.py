"""Raw-data archive stores (``<dataset>/dataset.zarr``) as recordings.

Each reader of an HDF5 recording reads the same values from the store that
holds that recording's datasets (``tests.archive_fixtures``): frames, camera
timestamps and stage (the export), nose landmarks, thumbnails.  A store is
found, listed and named as one recording, by the folder that holds it.
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from PIL import Image

from tests import test_export
from tests.archive_fixtures import archive_from_h5, write_archive
from tests.test_head_tracking import recording as head_tracking_recording
from tests.test_recordings import FRAMES, HEIGHT, WIDTH, _write_recording
from worm_pose_gen import library
from worm_pose_gen.frame_reader import ArrayFrames, open_frames
from worm_pose_gen.head_tracking import read_head_tracking
from worm_pose_gen.recording_files import ArchiveBehavior, is_recording, open_metadata, recording_kind, recording_name, stamp_path
from worm_pose_gen.recordings import RecordingSource, find_recordings, list_directory, list_recordings, thumbnail_png
from worm_pose_gen.stage_alignment import read_stage, recording_calibration

UUID = "01a1ad1a-b949-7ead-9ea9-58a31ee7e026"


class ArchiveFixture(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.h5 = self.root / "2022-06-22-01.h5"
        _write_recording(self.h5)
        with h5py.File(self.h5, "a") as handle:
            handle["/pos_stage"] = np.stack([np.arange(FRAMES) * 10.0, np.full(FRAMES, 5.0)], axis=1)
        self.store = archive_from_h5(self.h5, self.root / "raw" / UUID)


class NamingTests(unittest.TestCase):
    def test_a_store_is_named_by_its_folder_and_files_by_their_stem(self) -> None:
        store = Path("/archive/raw/by-uuid") / UUID / "dataset.zarr"
        self.assertEqual((recording_kind(store), recording_name(store)), ("archive", UUID))
        self.assertEqual(recording_name(Path("/archive/raw/by-prj/prj_x/2022-06-22-01/dataset.zarr")), "2022-06-22-01")
        self.assertEqual(stamp_path(store), store / "zarr.json")
        for path, kind in (("/d/2022-06-22-01.h5", "h5"), ("/d/worm.MP4", "video"), ("/d/other.zarr", None), ("/d/notes.txt", None)):
            self.assertEqual(recording_kind(path), kind)
        self.assertEqual(recording_name("/d/2022-06-22-01.h5"), "2022-06-22-01")


class ArchiveReadTests(ArchiveFixture):
    def test_frames_match_the_h5(self) -> None:
        with open_frames(self.store) as reader, open_frames(self.h5) as expected:
            self.assertIsInstance(reader, ArrayFrames)
            self.assertEqual((reader.frame_count, reader.shape, reader.fps, reader.times), (FRAMES, (HEIGHT, WIDTH), None, None))
            for k in range(FRAMES):
                np.testing.assert_array_equal(reader[k], expected[k])
            np.testing.assert_array_equal(reader[1:5], expected[1:5])
            np.testing.assert_array_equal(reader.read([5, 0, 0, 3]), expected.read([5, 0, 0, 3]))
            with self.assertRaises(IndexError):
                reader[FRAMES]
        with self.assertRaisesRegex(KeyError, "no dataset /camera/frames"):
            open_frames(self.store, "/camera/frames")

    def test_metadata_under_the_h5_names(self) -> None:
        with open_metadata(self.store) as archive, h5py.File(self.h5, "r") as handle:
            self.assertIsInstance(archive, ArchiveBehavior)
            for name in ("/img_nir", "img_nir", "/pos_stage"):
                self.assertIn(name, archive)
                np.testing.assert_array_equal(np.asarray(archive[name]), handle[name][()])
            self.assertNotIn("/pos_feature", archive)
            self.assertIsNone(archive.get("/img_metadata"))
            # Integer-array rows, as head tracking reads them.
            np.testing.assert_array_equal(archive["/img_nir"][np.array([0, 2, 5])], handle["/img_nir"][[0, 2, 5]])
        with self.assertRaisesRegex(FileNotFoundError, "not a Zarr store"):
            open_metadata(self.root / "missing" / "dataset.zarr")
        with self.assertRaisesRegex(ValueError, "not an HDF5 recording"):
            open_metadata(self.root / "worm.avi")

    def test_stage_matches_the_h5_and_calibration_is_cached_by_name(self) -> None:
        np.testing.assert_array_equal(read_stage(self.store).raw, read_stage(self.h5).raw)
        cache = self.root / "calibrations"
        cache.mkdir()
        (cache / f"{UUID}.json").write_text(
            '{"matrix": [[0.1, 0.0], [0.0, 0.1]], "offset_frames": 0.0, "residual_px": 1.0, "pairs": 10, "outliers": 0}'
        )
        self.assertEqual(recording_calibration(self.store, cache_dir=cache).matrix, ((0.1, 0.0), (0.0, 0.1)))

    def test_thumbnail_and_flat_field_source(self) -> None:
        png = thumbnail_png(self.store, 2, 0.5, dataset_root=self.root / "cache")
        self.assertEqual(Image.open(io.BytesIO(png)).size, (WIDTH // 2, HEIGHT // 2))
        np.testing.assert_array_equal(
            np.asarray(Image.open(io.BytesIO(png))), np.asarray(Image.open(io.BytesIO(thumbnail_png(self.h5, 2, 0.5, dataset_root=self.root / "cache"))))
        )
        source = RecordingSource(self.store, self.root / "fields")
        try:
            self.assertEqual(source.name, UUID)
            source.flat_field()
        finally:
            source.close()
        self.assertTrue((self.root / "fields" / f"{UUID}.npz").is_file())


class ArchiveHeadTrackingTests(unittest.TestCase):
    def test_nose_landmarks_and_camera_alignment_match_the_h5(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for metadata in (True, False):
                root = Path(directory) / str(metadata)
                root.mkdir()
                h5 = head_tracking_recording(root, metadata=metadata)
                store = archive_from_h5(h5, root / UUID)
                expected = read_head_tracking(h5, [3, 0, 3, 2], (20, 30))
                result = read_head_tracking(store, [3, 0, 3, 2], (20, 30))
                self.assertEqual(result.provenance["status"], "available")
                self.assertEqual(result.provenance["camera_alignment"], expected.provenance["camera_alignment"])
                for field in ("xy", "valid", "confidence", "source_frame_ids", "source_timestamps"):
                    np.testing.assert_array_equal(getattr(result, field), getattr(expected, field))
            store = write_archive(Path(directory) / "bare", {"img_nir": np.zeros((2, 20, 30), np.uint8)})
            bare = read_head_tracking(store, [0], (20, 30))
            self.assertEqual(bare.provenance["reason"], "missing_img_nir_or_pos_feature")


class ArchiveRecordingMotionTests(test_export.RecordingMotionTests):
    """The export's camera times and stage, from the store of the same acquisition file."""

    def setUp(self) -> None:
        super().setUp()
        self.path = archive_from_h5(self.path, Path(self.tmp.name) / UUID)

    def test_times_come_from_the_camera_metadata(self) -> None:
        times, _, notes = test_export.read_recording_motion(self.path, "/img_nir", np.arange(self.FRAMES))
        self.assertEqual(notes["time_source"]["source"], "camera timestamps")
        np.testing.assert_allclose(times, self.seconds, atol=1e-9)


class ArchiveDiscoveryTests(ArchiveFixture):
    def test_a_store_is_found_listed_and_probed_as_one_recording(self) -> None:
        found = find_recordings([self.root])
        self.assertEqual(found, sorted([self.h5.resolve(), self.store.resolve()], key=lambda p: (recording_name(p), str(p))))
        self.assertEqual(find_recordings([self.store]), [self.store.resolve()])
        # By-project links to the store resolve to it: one recording.
        (self.root / "by-prj" / "prj_x").mkdir(parents=True)
        (self.root / "by-prj" / "prj_x" / "2022-06-22-01").symlink_to(self.store.parent)
        self.assertEqual(find_recordings([self.root / "by-prj"]), [self.store.resolve()])
        infos = list_recordings([self.root / "raw"], cache=self.root / "index.json")
        self.assertEqual([(i.name, i.frames, i.height, i.width, i.readable) for i in infos], [(UUID, FRAMES, HEIGHT, WIDTH, True)])
        # The cache entry is stamped by the store's root metadata and reused.
        again = list_recordings([self.root / "raw"], cache=self.root / "index.json", probe=False)
        self.assertFalse(again[0].pending)

    def test_the_browser_shows_a_store_as_a_recording(self) -> None:
        listing = list_directory(self.store.parent)
        self.assertEqual([(e["name"], e["kind"]) for e in listing["entries"]], [("dataset.zarr", "archive")])
        self.assertTrue(is_recording(self.store))
        self.assertFalse(is_recording(self.store.parent))

    def test_registration_names_the_store_by_its_folder(self) -> None:
        from tests.test_library import make_libraries

        libraries = make_libraries(self.root / "libraries")
        library.create_setup(libraries, "rig2", name="Rig 2")
        entry = library.register_recording(libraries, self.store, "mine:rig2")
        self.assertEqual(entry["id"], UUID)
        self.assertEqual(library.setup_for_recording(libraries, self.store), "mine:rig2")
        with self.assertRaises(FileNotFoundError):
            library.register_recording(libraries, self.root / "missing" / "dataset.zarr", "mine:rig2")


if __name__ == "__main__":
    unittest.main()
