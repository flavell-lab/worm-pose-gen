"""The library: references, setups and defaults, label collections and revisions, datasets' per-recording splits, benchmarks, model cards."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from tests.test_frame_reader import FRAMES as VIDEO_FRAMES, HEIGHT as VIDEO_HEIGHT, WIDTH as VIDEO_WIDTH, _write_video
from worm_pose_gen import frame_reader, library
from worm_pose_gen.frame_reader import open_frames
from worm_pose_gen.library import Libraries
from worm_pose_gen.library import Collection, roots
from worm_pose_gen.library.datasets import Dataset
from worm_pose_gen.library.roots import LAB_LIBRARY_ENGAGING, LAB_LIBRARY_FLV, default_lab_root, parse_ref, ref_filename, write_json
from worm_pose_gen.recordings import find_recordings
from worm_pose_gen.settings import LIBRARY_DIR


SHAPE = (24, 32)


def label_inputs(seed=0, *, max_lag=2):
    rng = np.random.default_rng(seed)
    context = rng.integers(0, 255, (2 * max_lag + 1, *SHAPE), dtype=np.uint8)
    mask = np.zeros(SHAPE, np.uint8)
    mask[8:14, 4:28] = 1
    return {
        "image": context[max_lag], "image_raw": context[max_lag] // 2 + 1, "mask": mask, "context": context,
        "context_valid": np.ones(2 * max_lag + 1, bool), "source_path": "/data/rec.h5",
    }


def make_libraries(root: Path) -> Libraries:
    """A lab library with setup ``nir``, two lab labels of ``rec-a`` and dataset ``base`` putting it in train; an empty personal one."""

    lab, personal = root / "lab", root / "mine"
    library.write_setup(lab, "nir", name="NIR", fps=20.0, pixel_size_um=2.5)
    libraries = Libraries(lab=lab, personal=personal)
    (root / "recordings").mkdir()
    library.add_recording_folder(libraries, root / "recordings", "lab:nir")
    collection = Collection(libraries, "lab:nir")
    collection.save(recording="rec-a", frame=10, origin="spread", scope="lab", saved_at="2026-01-01T00:00:00+00:00", **label_inputs(1))
    collection.save(recording="rec-a", frame=20, origin="spread", orientation="manual", head_xy=[4.0, 11.0], scope="lab",
                    saved_at="2026-01-01T00:00:00+00:00", **label_inputs(2))
    library.create_dataset(libraries, "base", setup="lab:nir", splits={"rec-a": "train"}, scope="lab")
    return libraries


class RootTests(unittest.TestCase):
    def test_refs_and_host_defaults(self):
        self.assertEqual(parse_ref("lab:nir-labels"), ("lab", "nir-labels"))
        self.assertEqual(ref_filename("mine:copper_b1"), "mine.copper_b1")
        for bad in ("nir", "other:x", "lab:", "lab:a/b", "mine:a.b", "lab:../x"):
            with self.assertRaises(ValueError):
                parse_ref(bad)
        self.assertEqual(default_lab_root("flv-c3"), LAB_LIBRARY_FLV)
        for engaging in ("orcd-login001", "node2433"):
            self.assertEqual(default_lab_root(engaging), LAB_LIBRARY_ENGAGING)
        self.assertIsNone(default_lab_root("laptop"))
        self.assertEqual(Libraries.for_host().personal, LIBRARY_DIR)

    def test_json_files_take_numpy_scalars(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a" / "meta.json"
            write_json(path, {"cut": np.bool_(True), "iou": np.float32(0.5)})
            self.assertEqual(json.loads(path.read_text()), {"cut": True, "iou": 0.5})
            self.assertEqual(path.stat().st_mode & 0o777, 0o666 & ~roots.UMASK)

    def test_missing_lab_library_lists_personal_items_only(self):
        with tempfile.TemporaryDirectory() as directory:
            libraries = Libraries(lab=Path(directory) / "absent", personal=Path(directory) / "mine")
            self.assertEqual(libraries.scopes(), ("mine",))
            library.create_setup(libraries, "rig", name="Rig")
            self.assertEqual([s.ref for s in library.list_setups(libraries)], ["mine:rig"])
            self.assertEqual(Libraries(lab=None, personal=libraries.personal).scopes(), ("mine",))


class SetupTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)
        weights = self.root / "w.ckpt"
        weights.write_bytes(b"weights")
        inputs = library.make_inputs([], fps=20.0, pixel_size_um=2.5)
        library.write_model(self.libraries.lab, "seg", {"name": "seg", "kind": "segmenter", "setup": "lab:nir", "inputs": inputs, "outputs": ["mask"]}, weights)
        library.create_model(self.libraries, "body", {"name": "body", "kind": "body_net", "setup": "lab:nir",
                                                      "inputs": library.make_inputs([1, 4], fps=20.0, pixel_size_um=2.5),
                                                      "outputs": ["mask", "ap", "head", "tail", "overlap"]}, weights)

    def test_personal_override_of_a_lab_default_is_logged(self):
        setup = library.get_setup(self.libraries, "lab:nir")
        self.assertEqual((setup.fps, setup.pixel_size_um, setup.video["dataset_path"]), (20.0, 2.5, "/img_nir"))
        with self.assertRaises(ValueError):  # a mask-only model cannot drive the body
            library.set_default(self.libraries, "lab:nir", "body", "lab:seg", reason="try")
        with self.assertRaises(ValueError):
            library.set_default(self.libraries, "lab:nir", "body", "mine:body", reason=" ")
        updated = library.set_default(self.libraries, "lab:nir", "body", "mine:body", reason="better heads", who="kim")
        self.assertEqual(updated.defaults, {"body": "mine:body"})
        self.assertEqual(updated.overridden, ("body",))
        self.assertFalse((self.libraries.lab / "setups" / "nir.override.json").exists())
        updated = library.set_default(self.libraries, "lab:nir", "mask", "lab:seg", reason="lab mask")
        self.assertEqual(updated.defaults, {"body": "mine:body", "mask": "lab:seg"})
        log = library.defaults_log(self.libraries, "lab:nir")
        self.assertEqual([(e["role"], e["model"], e["previous"]) for e in log], [("body", "mine:body", None), ("mask", "lab:seg", None)])
        self.assertEqual(log[0]["who"], "kim")
        self.assertEqual(log[0]["reason"], "better heads")

    def test_recording_belongs_to_its_registration_else_its_folder(self):
        recordings = self.root / "recordings" / "day"
        recordings.mkdir(parents=True)
        inside = recordings / "2024-01-01-01.h5"
        inside.write_bytes(b"")
        outside = self.root / "elsewhere" / "2024-02-02-02.h5"
        outside.parent.mkdir()
        outside.write_bytes(b"")
        self.assertEqual(library.setup_for_recording(self.libraries, inside), "lab:nir")
        self.assertIsNone(library.setup_for_recording(self.libraries, outside))
        library.create_setup(self.libraries, "rig2", name="Rig 2")
        entry = library.register_recording(self.libraries, outside, "mine:rig2")
        self.assertEqual(entry["id"], "2024-02-02-02")
        self.assertEqual(library.setup_for_recording(self.libraries, outside), "mine:rig2")
        self.assertEqual(library.recording_sources(self.libraries, "mine:rig2"), [outside.resolve()])
        with self.assertRaises(LookupError):
            library.register_recording(self.libraries, outside, "mine:nope")

    def test_recording_folders_belong_to_one_setup_and_can_be_removed(self):
        folder = self.root / "plates"
        (folder / "day").mkdir(parents=True)
        recording = folder / "day" / "2024-03-03-03.h5"
        recording.write_bytes(b"")
        library.create_setup(self.libraries, "rig2", name="Rig 2")
        self.assertEqual(library.add_recording_folder(self.libraries, folder, "mine:rig2"), {"path": str(folder.resolve()), "setup": "mine:rig2"})
        self.assertEqual(library.setup_for_recording(self.libraries, recording), "mine:rig2")
        self.assertEqual(library.recording_sources(self.libraries, "mine:rig2"), [folder.resolve()])
        with self.assertRaises(NotADirectoryError):
            library.add_recording_folder(self.libraries, recording, "mine:rig2")
        with self.assertRaises(LookupError):
            library.add_recording_folder(self.libraries, folder, "mine:nope")
        library.remove_recording_folder(self.libraries, folder)
        self.assertIsNone(library.setup_for_recording(self.libraries, recording))
        with self.assertRaises(LookupError):
            library.remove_recording_folder(self.libraries, folder)

    def test_recording_folders_give_their_setup_and_can_be_removed(self):
        library.create_setup(self.libraries, "rig2", name="Rig 2")
        other = self.root / "other"
        (other / "day").mkdir(parents=True)
        recording = other / "day" / "2024-03-03-03.h5"
        recording.write_bytes(b"")
        self.assertIsNone(library.setup_for_recording(self.libraries, recording))
        library.add_recording_folder(self.libraries, other, "mine:rig2")
        self.assertEqual(library.setup_for_recording(self.libraries, recording), "mine:rig2")
        self.assertEqual(library.recording_sources(self.libraries, "mine:rig2"), [other.resolve()])
        self.assertEqual(sorted(library.recording_folders(self.libraries)), sorted([str(other.resolve()), str((self.root / "recordings").resolve())]))
        library.remove_recording_folder(self.libraries, other)
        self.assertIsNone(library.setup_for_recording(self.libraries, recording))
        with self.assertRaises(LookupError):
            library.remove_recording_folder(self.libraries, other)
        with self.assertRaises(NotADirectoryError):
            library.add_recording_folder(self.libraries, recording, "mine:rig2")

    def test_a_recording_reached_through_a_link_in_the_folder_belongs_to_it(self):
        elsewhere = self.root / "store" / "prj_x" / "data_raw" / "2025-03-18"
        elsewhere.mkdir(parents=True)
        recording = elsewhere / "2025-03-18-16.h5"
        recording.write_bytes(b"")
        (self.root / "recordings" / "prj_x").symlink_to(elsewhere.parent)
        self.assertEqual(library.setup_for_recording(self.libraries, recording), "lab:nir")
        self.assertEqual(library.setup_for_recording(self.libraries, self.root / "recordings" / "prj_x" / "2025-03-18" / recording.name), "lab:nir")
        stray = self.root / "store" / "other.h5"
        stray.write_bytes(b"")
        self.assertIsNone(library.setup_for_recording(self.libraries, stray))

    def test_a_video_is_registered_as_itself_and_opens(self):
        video = self.root / "elsewhere" / "worm004.avi"
        video.parent.mkdir(parents=True)
        _write_video(video, "mjpeg")
        entry = library.register_recording(self.libraries, video, "lab:nir")
        self.assertEqual(entry, {"id": "worm004", "path": str(video.resolve()), "setup": "lab:nir"})
        self.assertEqual(library.setup_for_recording(self.libraries, video), "lab:nir")
        self.assertEqual(find_recordings(library.recording_sources(self.libraries, "lab:nir")), [video.resolve()])
        with mock.patch.object(frame_reader, "INDEX_DIR", self.root / "video_index"), open_frames(video) as frames:
            self.assertEqual((frames.frame_count, frames.shape), (VIDEO_FRAMES, (VIDEO_HEIGHT, VIDEO_WIDTH)))


class CollectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)
        self.collection = Collection(self.libraries, "lab:nir")

    def test_revisions_are_immutable_and_concurrent_edits_refused(self):
        collection = self.collection
        first = collection.save(recording="rec-b", frame=3, origin="spread", expected_revision=0, **label_inputs(3))
        with self.assertRaisesRegex(ValueError, "changed since it was opened"):
            collection.save(recording="rec-b", frame=3, origin="fix", expected_revision=0, **label_inputs(3))
        inputs = label_inputs(3)
        inputs["mask"] = np.zeros(SHAPE, np.uint8)
        second = collection.save(recording="rec-b", frame=3, origin="fix", expected_revision=1, mask_only=True, **inputs)
        self.assertEqual((first.revision, second.revision), (1, 2))
        self.assertEqual((first.scope, first.setup, first.split), ("mine", "lab:nir", None))
        self.assertNotEqual(first.sha256, second.sha256)
        self.assertEqual([r.revision for r in collection.revisions("rec-b", 3)], [1, 2])
        self.assertTrue(first.load().mask.any())  # the old revision still reads as it was
        self.assertEqual(second.status, "mask_only")
        self.assertEqual(collection.get("rec-b", 3).revision, 2)
        self.assertEqual(collection.get("rec-b", 3, revision=1, scope="mine").sha256, first.sha256)
        self.assertEqual(collection.own_revision("rec-b", 3), 2)
        self.assertEqual(collection.recordings(), {"rec-a": 2, "rec-b": 1})

    def test_label_validation(self):
        collection = self.collection
        inputs = label_inputs(4)
        bad = [
            dict(inputs, mask=np.full(SHAPE, 2, np.uint8)),
            dict(inputs, image=inputs["image"] + 1),  # not the centre context frame
            dict(inputs, context_valid=np.zeros(5, bool)),
        ]
        for case in bad:
            with self.assertRaises(ValueError):
                collection.save(recording="rec-c", frame=1, origin="spread", **case)
        with self.assertRaises(ValueError):
            collection.save(recording="rec-c", frame=1, origin="painted", **inputs)
        with self.assertRaises(ValueError):  # manual without a head or trace
            collection.save(recording="rec-c", frame=1, origin="spread", orientation="manual", **inputs)
        outside = np.zeros(SHAPE, np.uint8)
        outside[inputs["mask"] != 1] = 1
        for overlap in (outside, np.full(SHAPE, 2, np.uint8) * (inputs["mask"] == 1)):  # off the worm; not 0/1
            with self.assertRaises(ValueError):
                collection.save(recording="rec-c", frame=1, origin="spread", overlap=overlap, **inputs)
        traced = collection.save(recording="rec-c", frame=1, origin="spread", trace_xy=[[4, 11], [16, 11], [27, 11]], **inputs)
        self.assertIsNone(traced.overlap_px)
        self.assertIsNone(traced.load().overlap)
        reviewed = collection.save(recording="rec-c", frame=1, origin="spread", overlap=np.zeros(SHAPE, bool), **inputs)
        self.assertEqual(reviewed.overlap_px, 0)  # reviewed: no overlap
        self.assertEqual(collection.get("rec-c", 1).overlap_px, 0)  # read back from the index
        self.assertFalse(reviewed.load().overlap.any())
        self.assertEqual((traced.orientation, traced.has_trace, traced.status), ("manual", True, "complete"))
        loaded = traced.load()
        self.assertIsNone(loaded.head_xy)
        np.testing.assert_array_equal(loaded.trace_xy[0], [4, 11])
        self.assertEqual(loaded.max_lag, 2)
        self.assertFalse(loaded.nose_valid.any())

    def test_a_personal_edit_of_a_lab_label_is_its_newest_revision(self):
        collection = self.collection
        edited = collection.save(recording="rec-a", frame=10, origin="fix", expected_revision=0, **label_inputs(9))
        records = collection.labels()
        self.assertEqual([(r.scope, r.key, r.revision) for r in records], [("mine", "rec-a/000010", 1), ("lab", "rec-a/000020", 1)])
        self.assertEqual(records[0], edited)
        self.assertEqual([r.scope for r in collection.revisions("rec-a", 10)], ["lab", "mine"])
        self.assertEqual(collection.get("rec-a", 10, revision=1, scope="lab").origin, "spread")  # the lab label is untouched
        with self.assertRaises(LookupError):
            collection.get("rec-a", 10, revision=2, scope="mine")
        statuses = {r.key: r.status for r in records}
        self.assertEqual(statuses, {"rec-a/000010": "auto", "rec-a/000020": "complete"})
        # A lab copy with the same saved_at (a published label) is current from then on.
        lab = collection.save(recording="rec-a", frame=10, origin="fix", scope="lab", saved_at=edited.saved_at, **label_inputs(9))
        self.assertEqual(collection.get("rec-a", 10), lab)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)
        self.collection = Collection(self.libraries, "lab:nir")

    def test_a_new_dataset_includes_nothing_until_splits_are_chosen(self):
        mine = library.create_dataset(self.libraries, "copper", setup="lab:nir")
        self.assertEqual(mine.labels(), [])
        summary = mine.summary()
        self.assertEqual(summary["by_split"], {"train": 0, "val": 0, "test": 0})
        self.assertEqual(summary["not_included"], {"recordings": 1, "labels": 2})
        self.assertEqual([(r["recording"], r["split"], r["labels"]) for r in summary["recordings"]], [("rec-a", None, 2)])
        self.assertEqual(summary["readiness"], ["no training labels", "no validation recording", "no test recording"])

        mine.set_splits({"rec-a": "val"})
        self.assertEqual([(r.key, r.split) for r in mine.labels()], [("rec-a/000010", "val"), ("rec-a/000020", "val")])
        # A recording labeled for the first time joins the collection, and every dataset lists it as not included.
        self.collection.save(recording="rec-new", frame=5, origin="spread", **label_inputs(10))
        for dataset in (mine, Dataset(self.libraries, "lab:base")):
            rows = {r["recording"]: r["split"] for r in dataset.summary()["recordings"]}
            self.assertIsNone(rows["rec-new"])
        self.assertEqual(len(mine.labels()), 2)
        # Once it has a split, its later labels follow it.
        mine.set_splits({"rec-new": "test"})
        self.collection.save(recording="rec-new", frame=6, origin="spread", **label_inputs(11))
        self.assertEqual([r.frame for r in library.labels(self.libraries, "mine:copper", "test")], [5, 6])
        self.assertEqual(mine.summary()["by_split"], {"train": 0, "val": 2, "test": 2})
        self.assertEqual(len(library.labels(self.libraries, "mine:copper", recording="rec-a", status="complete")), 1)
        mine.set_splits({"rec-a": None})
        self.assertEqual(mine.splits(), {"rec-new": "test"})
        self.assertEqual(Dataset(self.libraries, "lab:base").splits(), {"rec-a": "train"})  # datasets are independent
        with self.assertRaises(ValueError):
            mine.set_splits({"rec-a": "holdout"})
        with self.assertRaises(ValueError):
            library.labels(self.libraries, "mine:copper", "holdout")
        with self.assertRaises(PermissionError):
            Dataset(self.libraries, "lab:base").set_splits({"rec-a": "test"})
        with self.assertRaises(FileExistsError):
            library.create_dataset(self.libraries, "copper", setup="lab:nir")

    def test_a_dataset_can_start_from_given_splits(self):
        copy = library.create_dataset(self.libraries, "copy", setup="lab:nir", splits=Dataset(self.libraries, "lab:base").splits())
        self.assertEqual({r.split for r in copy.labels()}, {"train"})
        with self.assertRaises(ValueError):
            library.create_dataset(self.libraries, "bad", setup="lab:nir", splits={"rec-a": "everything"})

    def test_trained_on_names_exact_revisions_and_splits(self):
        records = library.labels(self.libraries, "lab:base")
        summary = library.trained_on("lab:base", records)
        self.assertEqual(summary[0]["dataset"], "lab:base")
        self.assertEqual(summary[0]["counts"], {"train": 2, "val": 0, "test": 0})
        self.assertEqual(library.fingerprint(records), library.fingerprint(records[::-1]))
        self.assertNotEqual(library.fingerprint(records), library.fingerprint(records[:1]))
        moved = library.create_dataset(self.libraries, "moved", setup="lab:nir", splits={"rec-a": "val"})
        self.assertNotEqual(library.fingerprint(moved.labels()), library.fingerprint(records))  # same revisions, other split


class BenchmarkAndModelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.libraries = make_libraries(self.root)

    def test_freeze_takes_spread_and_migrated_test_labels_once(self):
        collection = Collection(self.libraries, "lab:nir")
        library.create_dataset(self.libraries, "copper", setup="lab:nir", splits={"held": "test"})
        spread = collection.save(recording="held", frame=1, origin="spread", **label_inputs(1))
        collection.save(recording="held", frame=2, origin="fix", **label_inputs(2))
        frozen = library.freeze_benchmark(self.libraries, "mine:copper")
        self.assertEqual((frozen.ref, frozen.dataset), ("mine:copper-b1", "mine:copper"))
        self.assertEqual([e["frame"] for e in frozen.entries], [1])
        self.assertEqual(library.benchmark_labels(self.libraries, frozen.ref), [replace(spread, split="test")])
        # Frozen means frozen: a later label does not join it, and the next freeze is b2.
        collection.save(recording="held", frame=3, origin="spread", **label_inputs(3))
        self.assertEqual(len(library.get_benchmark(self.libraries, frozen.ref).entries), 1)
        self.assertEqual(library.freeze_benchmark(self.libraries, "mine:copper").ref, "mine:copper-b2")
        with self.assertRaises(FileExistsError):
            library.freeze_benchmark(self.libraries, "mine:copper", "copper-b1")
        self.assertEqual([b.ref for b in library.list_benchmarks(self.libraries, "lab:nir")], ["mine:copper-b1", "mine:copper-b2"])
        with self.assertRaises(ValueError):  # no test labels
            library.freeze_benchmark(self.libraries, "lab:base")

    def test_model_cards_weights_and_where_evaluations_go(self):
        weights = self.root / "best.ckpt"
        weights.write_bytes(b"lab weights")
        card = {"name": "nir-hand2", "kind": "segmenter", "setup": "lab:nir", "outputs": ["mask"],
                "inputs": library.make_inputs([], fps=20.0, pixel_size_um=2.5)}
        library.write_model(self.libraries.lab, "nir-hand2", card, weights, training_records={"labels.json": {"n": 2}})
        with self.assertRaises(FileExistsError):
            library.write_model(self.libraries.lab, "nir-hand2", card, weights)
        with self.assertRaises(ValueError):
            library.create_model(self.libraries, "bad", {**card, "outputs": ["mask", "colour"]}, weights)
        body = library.create_model(self.libraries, "body", {**card, "name": "body", "kind": "body_net", "parent": "lab:nir-hand2",
                                                             "inputs": library.make_inputs([1, 4, 16], fps=20.0, pixel_size_um=None),
                                                             "outputs": ["mask", "ap", "head", "tail", "overlap"]}, weights)
        self.assertEqual(body.inputs["lags_s"], [0.05, 0.2, 0.8])
        self.assertEqual(library.weights_path(self.libraries, "lab:nir-hand2").read_bytes(), b"lab weights")
        self.assertEqual(json.loads((library.training_dir(self.libraries, "lab:nir-hand2") / "labels.json").read_text()), {"n": 2})
        self.assertEqual([c.ref for c in library.list_models(self.libraries, "lab:nir")], ["lab:nir-hand2", "mine:body"])
        lab_eval = library.evaluation_path(self.libraries, "lab:nir-hand2", "mine:copper-b1")
        self.assertEqual(lab_eval, self.libraries.personal / "evaluations" / "lab.nir-hand2" / "mine.copper-b1.json")
        self.assertEqual(library.evaluation_path(self.libraries, "mine:body", "lab:v1"),
                         self.libraries.personal / "models" / "body" / "evaluations" / "lab.v1.json")
        # A lab evaluation published with the model, then a personal rescoring on the same benchmark wins.
        published = self.libraries.lab / "models" / "nir-hand2" / "evaluations" / "lab.v1.json"
        published.parent.mkdir()
        published.write_text(json.dumps({"benchmark": "lab:v1", "iou_mean": 0.9}))
        self.assertEqual(library.evaluations(self.libraries, "lab:nir-hand2")["lab:v1"]["iou_mean"], 0.9)
        library.save_evaluation(self.libraries, "lab:nir-hand2", "lab:v1", {"iou_mean": 0.95})
        library.save_evaluation(self.libraries, "lab:nir-hand2", "mine:copper-b1", {"iou_mean": 0.8})
        scores = library.evaluations(self.libraries, "lab:nir-hand2")
        self.assertEqual({k: v["iou_mean"] for k, v in scores.items()}, {"lab:v1": 0.95, "mine:copper-b1": 0.8})
        self.assertFalse((self.libraries.lab / "models" / "nir-hand2" / "evaluations" / "mine.copper-b1.json").exists())


if __name__ == "__main__":
    unittest.main()
