"""The per-recording export (``worm_pose_gen.export_table``, ``pipeline.run_export``, ``app/exporting.py``).

Pure parts on synthetic arrays (geometry, status, the feature list), the
recording reader on a synthetic acquisition file with camera timestamps and
a tracking stage, and whole exports of a synthetic fitted workspace through
the pipeline stage and the API.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient
import h5py
import numpy as np
import pyarrow.parquet as pq

from worm_pose_gen import export_table, frame_reader, pipeline
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.app.exporting import export_workspace, exported_file, list_exports
from worm_pose_gen.export_table import MIDLINE_POINTS, Meta, body_geometry, build_table, frame_status, read_recording_motion
from worm_pose_gen.stage_alignment import StageCalibration, read_stage, stage_shift_px
from tests.test_frame_reader import _write_video
from tests.test_frame_view import FRAMES, _write_recording, _write_workspace


def _arc(radius: float, center: tuple[float, float], start: float, span: float, points: int = 100) -> np.ndarray:
    angle = start + np.linspace(0.0, span, points)
    return np.stack((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)), axis=1)


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat(timespec="seconds")


class GeometryAndStatusTests(unittest.TestCase):
    def test_resampled_midline_width_and_curvature_in_micrometres(self) -> None:
        straight = np.stack((np.linspace(10, 109, 100), np.full(100, 40.0)), axis=1)
        # Counter-clockwise in (x, y) maths terms is clockwise on screen (y down): positive curvature.
        arc = _arc(50.0, (60.0, 60.0), 0.0, 1.5)
        curves = np.stack((straight, arc, straight))
        profile = np.tile(np.linspace(2.0, 4.0, 100), (3, 1))
        midline, width = body_geometry(curves, profile, np.array([True, True, False]), scale=2.0)
        self.assertEqual(midline.shape, (3, MIDLINE_POINTS, 2))
        np.testing.assert_allclose(midline[0, 0], (20.0, 80.0))
        np.testing.assert_allclose(midline[0, -1], (218.0, 80.0))
        np.testing.assert_allclose(width[0], np.linspace(4.0, 8.0, MIDLINE_POINTS))
        self.assertTrue(np.isnan(midline[2]).all() and np.isnan(width[2]).all())
        columns = export_table.curvature(midline, width, np.zeros(3), Meta("um", None))["curvature"]
        self.assertEqual(columns.unit, "rad/um")
        np.testing.assert_allclose(columns.values[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(columns.values[1, 2:-2], 1.0 / 100.0, rtol=1e-3)  # radius 50 px * 2 µm/px
        self.assertTrue(np.isnan(columns.values[2]).all())

    def test_status_precedence(self) -> None:
        fitted = np.array([True, False, True, True, True, True])
        flagged = np.array([False, False, True, True, True, False])
        reviewed = np.array([False, False, False, True, True, True])
        fixed = np.array([False, False, False, False, True, True])
        stale = np.array([False, False, False, False, False, True])
        self.assertEqual(
            frame_status(fitted, flagged, stale, fixed, reviewed).tolist(),
            ["auto", "unresolved", "unresolved", "reviewed", "fixed", "unresolved"],
        )

    def test_table_columns_units_and_nulls(self) -> None:
        curves = np.stack([np.stack((np.linspace(0, 99, 100) + 3 * k, np.full(100, 40.0)), axis=1) for k in range(3)])
        midline, width = body_geometry(curves, np.full((3, 100), 5.0), np.array([True, False, True]), scale=1.0)
        table, columns = build_table(np.array([4, 5, 6]), np.array([0.0, 0.05, 0.1]), np.array(["auto", "unresolved", "fixed"], dtype=object), midline, width, Meta("px", None))
        self.assertEqual(table.column_names[:3], ["frame", "time_s", "status"])
        self.assertEqual(list(columns), table.column_names)
        for name in ("midline_x", "midline_y", "curvature", "width", "head_x", "head_y", "tail_x", "tail_y", "centroid_x", "centroid_y",
                     "centroid_world_x", "centroid_world_y", "velocity_x", "velocity_y", "speed"):
            self.assertIn(name, columns)
        self.assertEqual(columns["velocity_x"]["unit"], "px/s")
        self.assertEqual(columns["midline_x"]["dtype"], "list<item: float>")
        self.assertEqual(table.schema.field("speed").metadata[b"unit"], b"px/s")
        with tempfile.TemporaryDirectory() as directory:
            # Null list rows (frames without a pose) survive the Parquet round trip.
            pq.write_table(table, Path(directory) / "t.parquet")
            rows = pq.read_table(Path(directory) / "t.parquet").to_pylist()
        self.assertEqual(len(rows[0]["midline_x"]), MIDLINE_POINTS)
        self.assertIsNone(rows[1]["midline_x"])
        self.assertIsNone(rows[1]["head_x"])
        self.assertEqual(rows[2]["status"], "fixed")
        self.assertAlmostEqual(rows[0]["head_x"], 0.0)
        self.assertAlmostEqual(rows[2]["tail_x"], 105.0, places=4)
        # Rows 0 and 2 have no neighbour with a pose on the other side, so the central difference is undefined.
        self.assertIsNone(rows[0]["velocity_x"])

    def test_a_feature_is_one_function(self) -> None:
        def body_length(midline, width, time, meta):
            length = np.linalg.norm(np.diff(midline, axis=1), axis=2).sum(axis=1)
            return {"body_length": export_table.Column(length, meta.length_unit, "arc length of the midline")}

        curves = np.stack((np.linspace(0, 99, 100), np.zeros(100)), axis=1)[None]
        midline, width = body_geometry(curves, np.full((1, 100), 5.0), np.array([True]), scale=1.0)
        with mock.patch.object(export_table, "FEATURES", export_table.FEATURES + (body_length,)):
            table, columns = build_table(np.array([0]), np.array([0.0]), np.array(["auto"], dtype=object), midline, width, Meta("px", None))
        self.assertAlmostEqual(table.column("body_length")[0].as_py(), 99.0, places=6)
        self.assertEqual(columns["body_length"]["unit"], "px")


class RecordingMotionTests(unittest.TestCase):
    """A worm crawling at a constant world velocity while the stage tracks it, in a ConfocalTrackerControl-shaped file.

    The stage's calibration against the image (``stage_alignment``) is given:
    0.1 µm stage units, ``PIXEL_UM`` µm pixels, readings at the exposure; the
    offsets are the stage's alone (the blank frames cannot be registered to
    fill the failed read, ``tests.test_stage_alignment`` covers that).
    """

    FRAMES = 40
    PIXEL_UM = 1.25
    WORLD_UM_PER_S = np.array([100.0, -40.0])
    CALIBRATION = StageCalibration(matrix=((0.1 / PIXEL_UM, 0.0), (0.0, 0.1 / PIXEL_UM)), offset_frames=0.0, residual_px=1.0, pairs=100, outliers=0)

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "acq.h5"
        n = self.FRAMES
        # Saved frames at 20 fps with one dropped saved frame after frame 19 (the camera runs at 40 Hz).
        seconds = np.arange(n) * 0.05
        seconds[20:] += 0.05
        camera = np.repeat(seconds, 2) - np.tile([0.025, 0.0], n)
        save = np.tile([0, 1], n).astype(np.uint8)
        self.seconds = seconds
        world = seconds[:, None] * self.WORLD_UM_PER_S + np.array([500.0, 300.0])
        # The stage keeps the worm near the image centre: image = world + stage (µm), stage lags by a little.
        stage_um = np.round(world - 0.3 * np.sin(np.arange(n))[:, None] * 20.0, 1)
        self.image_um = world + stage_um
        self.origin_um = stage_um[0]  # the world frame's origin: the image position of the first stage reading
        raw = stage_um / 0.1
        raw[7] = np.nan  # a failed serial read
        self.missing = 7
        calibrated = mock.patch.object(export_table, "recording_calibration", return_value=self.CALIBRATION)
        calibrated.start()
        self.addCleanup(calibrated.stop)
        stage_only = mock.patch.object(export_table, "substrate_shift_px", lambda recording, frames, calibration: stage_shift_px(read_stage(recording), frames, calibration))
        stage_only.start()
        self.addCleanup(stage_only.stop)
        with h5py.File(self.path, "w") as handle:
            handle.create_dataset("/img_nir", data=np.zeros((n, 4, 4), dtype=np.uint8))
            handle.create_dataset("/pos_stage", data=raw)
            group = handle.create_group("img_metadata")
            group.create_dataset("q_iter_save", data=save)
            group.create_dataset("q_recording", data=np.ones(2 * n, dtype=np.uint8))
            group.create_dataset("img_id", data=np.arange(2 * n))
            group.create_dataset("img_timestamp", data=(1_317_481_283_558_544 + np.round(camera * 1e9)).astype(np.int64))

    def test_timestamps_and_stage(self) -> None:
        frames = np.arange(3, self.FRAMES)
        times, stage, notes = read_recording_motion(self.path, "/img_nir", frames)
        np.testing.assert_allclose(times, self.seconds[frames], atol=1e-9)
        self.assertEqual((notes["stage_unknown_frames"], notes["calibration"]["offset_frames"]), (0, 0.0))
        world = self.image_um[frames] - stage * self.PIXEL_UM
        good = frames != self.missing
        expected = self.seconds[frames, None] * self.WORLD_UM_PER_S + np.array([500.0, 300.0]) + self.origin_um
        np.testing.assert_allclose(world[good], expected[good], atol=1e-6)

    def test_world_velocity_of_the_exported_centroid(self) -> None:
        frames = np.arange(self.FRAMES)
        times, stage, _ = read_recording_motion(self.path, "/img_nir", frames)
        # A straight body centred on the image position, given in pixels.
        offsets = np.stack((np.linspace(-40, 40, 100), np.zeros(100)), axis=1)
        curves = self.image_um[:, None, :] / self.PIXEL_UM + offsets[None]
        midline, width = body_geometry(curves, np.full((self.FRAMES, 100), 6.0), np.ones(self.FRAMES, dtype=bool), self.PIXEL_UM)
        columns = export_table.motion(midline, width, times, Meta("um", stage * self.PIXEL_UM))
        velocity = np.stack((columns["velocity_x"].values, columns["velocity_y"].values), axis=1)
        away = np.abs(frames[:, None] - np.array([self.missing - 1, self.missing, self.missing + 1, self.missing + 2])).min(axis=1) > 0
        np.testing.assert_allclose(velocity[away], np.tile(self.WORLD_UM_PER_S, (away.sum(), 1)), rtol=1e-6, atol=1e-6)
        self.assertEqual(columns["speed"].unit, "um/s")

    def test_files_without_metadata_or_unreadable(self) -> None:
        bare = Path(self.tmp.name) / "bare.h5"
        with h5py.File(bare, "w") as handle:
            handle.create_dataset("/img_nir", data=np.zeros((3, 4, 4), dtype=np.uint8))
        times, stage, notes = read_recording_motion(bare, "/img_nir", np.arange(3))
        self.assertIsNone(times)
        self.assertIsNone(stage)
        self.assertIn("img_metadata", notes["time"])
        times, stage, notes = read_recording_motion(Path(self.tmp.name) / "missing.h5", "/img_nir", np.arange(3))
        self.assertIsNone(times)
        self.assertIn("unreadable", notes["error"])


class VideoExportTests(unittest.TestCase):
    """A video recording: its presentation timestamps are the export's times, and it has no stage."""

    MILLISECONDS = [0, 50, 100, 200, 250, 330, 400, 450]

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        indexed = mock.patch.object(frame_reader, "INDEX_DIR", Path(self.tmp.name) / "index")
        indexed.start()
        self.addCleanup(indexed.stop)
        self.path = Path(self.tmp.name) / "video_export_test.mp4"
        _write_video(self.path, "libx264", frames=len(self.MILLISECONDS), milliseconds=self.MILLISECONDS)

    def test_variable_frame_rate_times_and_no_stage(self) -> None:
        seconds = np.array(self.MILLISECONDS) / 1000.0
        frames = np.array([1, 3, 4, 7])
        times, stage, notes = read_recording_motion(self.path, "/img_nir", frames)
        np.testing.assert_allclose(times, seconds[frames], atol=1e-9)
        self.assertIsNone(stage)
        self.assertEqual((notes["time_source"]["source"], notes["stage"]), ("video presentation timestamps", "no /pos_stage"))
        n = len(self.MILLISECONDS)
        curves = np.stack([np.stack((np.linspace(5, 40, 100) + t, np.full(100, 20.0)), axis=1) for t in range(n)])
        result = export_table.export(
            Path(self.tmp.name) / "exports", recording=self.path, dataset="/img_nir", frames=np.arange(n),
            centerline_xy=curves, width_profile_px=np.full((n, 100), 4.0), fitted=np.ones(n, dtype=bool),
            status=np.full(n, "auto", dtype=object), pixel_size_um=None, fps=20.0, about={},
        )
        self.assertEqual(result["time"]["source"], "video presentation timestamps")
        self.assertEqual(result["velocity"]["frame"], "image")
        table = pq.read_table(result["path"])
        np.testing.assert_allclose(table.column("time_s").to_numpy(), seconds, atol=1e-9)
        # The body moves one pixel a frame: its velocity follows the uneven frame times.
        np.testing.assert_allclose(table.column("velocity_x").to_numpy(), np.gradient(np.arange(n, dtype=float), seconds), rtol=1e-6)


class WorkspaceExportTests(unittest.TestCase):
    """Whole exports of a synthetic fitted workspace (no camera metadata, no stage)."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        recording = root / "rec.h5"
        _write_recording(recording)
        self.workspace = _write_workspace(root / "workspaces", "test", recording)
        self.app = mock.Mock(workspace=lambda name: self.workspace, device="cpu")

    def test_stage_writes_documented_table_and_metadata(self) -> None:
        summary = (self.workspace.path / "summary.json").read_text()
        result = pipeline.run_stage(self.workspace, "export", {"pixel_size_um": 2.0, "fps": 20.0, "setup": "lab:nir-flv"}, device="cpu")
        directory = Path(result["path"]).parent
        self.assertEqual(directory.parent, self.workspace.path / "exports")
        self.assertRegex(directory.name, r"^rec_\d{4}-\d\d-\d\dT\d\d-\d\d-\d\dZ$")
        self.assertEqual(result["table"], directory.name + ".parquet")
        metadata = json.loads((directory / "export.json").read_text())
        self.assertEqual(metadata, {k: v for k, v in result.items() if k != "path"})
        self.assertEqual((metadata["setup"], metadata["pixel_size_um"], metadata["fps"], metadata["length_unit"]), ("lab:nir-flv", 2.0, 20.0, "um"))
        self.assertEqual(metadata["time"]["source"], "frame rate")
        self.assertEqual(metadata["velocity"]["frame"], "image")
        self.assertIn("img_metadata", metadata["velocity"]["reason"] + metadata["time"]["reason"])
        self.assertIn("commit", metadata["app_revision"])
        self.assertIn("mask", metadata["models"])
        self.assertEqual(metadata["frames"], {"first": 0, "last": FRAMES - 1, "count": FRAMES})
        table = pq.read_table(result["path"])
        self.assertEqual(table.num_rows, FRAMES)
        self.assertEqual(list(metadata["columns"]), table.column_names)
        self.assertEqual(json.loads(table.schema.metadata[b"export.json"]), metadata)
        self.assertEqual(metadata["columns"]["head_x"]["unit"], "um")
        rows = table.to_pylist()
        np.testing.assert_allclose([r["time_s"] for r in rows], np.arange(FRAMES) / 20.0)
        self.assertEqual(len(rows[0]["midline_x"]), MIDLINE_POINTS)
        state = self.workspace.load_state()
        np.testing.assert_allclose(rows[0]["head_x"], 2.0 * state["centerline_xy"][0, 0, 0], rtol=1e-5)
        # The export reads the workspace and writes only under exports/.
        self.assertEqual((self.workspace.path / "summary.json").read_text(), summary)

    def test_status_from_reviews_fixes_and_flags(self) -> None:
        n = self.workspace.n
        state = self.workspace.load_state()
        state["flag_low_iou"] = np.zeros(n, dtype=bool)
        state["flag_low_iou"][[2, 3, 4]] = True
        state["mask_stale"] = np.zeros(n, dtype=bool)
        state["mask_stale"][5] = True
        self.workspace.save_state(state)
        now = time.time()
        self.workspace.set_provenance([1], "manual:flip", "edit:e000001", now - 100)
        # Rows 2 and 3 were reviewed; row 3's pose changed after the review, row 2's did not.
        review = {"rows": [2, 3], "reviewed_at": _iso(now - 50), "row_fingerprints": {}}
        (self.workspace.path / "human_review.json").write_text(json.dumps(review))
        self.workspace.set_provenance([3], "independent_fit", "stage:fit", now - 10)
        status = pq.read_table(pipeline.run_stage(self.workspace, "export", {}, device="cpu")["path"]).column("status").to_pylist()
        self.assertEqual(status[:6], ["auto", "fixed", "reviewed", "unresolved", "unresolved", "unresolved"])
        # An undo logged after the review (it restores older provenance) also ends the review of its rows.
        self.workspace.append_edit("undo", {"rows": [2], "undoes": "e000001"})
        result = pipeline.run_stage(self.workspace, "export", {}, device="cpu")
        self.assertEqual(pq.read_table(result["path"]).column("status").to_pylist()[2], "unresolved")
        self.assertEqual(result["time"]["source"], "none")
        self.assertEqual(result["length_unit"], "px")
        self.assertTrue(np.isnan(pq.read_table(result["path"]).column("velocity_x").to_numpy(zero_copy_only=False).astype(float)).all())
        self.assertEqual(sum(result["status"]["counts"].values()), n)

    def test_app_export_lock_names_listing_and_download_paths(self) -> None:
        with pipeline.workspace_lock(self.workspace):
            with self.assertRaises(pipeline.WorkspaceBusy):
                export_workspace(self.app, "test")
        self.assertFalse((self.workspace.path / "exports").exists())
        with mock.patch.object(export_table, "timestamp_slug", return_value="2026-10-07T12-00-00Z"):
            first = export_workspace(self.app, "test")
            second = export_workspace(self.app, "test", {"fps": 20})
        self.assertEqual((first["name"], second["name"]), ("rec_2026-10-07T12-00-00Z", "rec_2026-10-07T12-00-00Z-2"))
        self.assertEqual({e["name"] for e in list_exports(self.workspace)}, {first["name"], second["name"]})
        self.assertEqual(exported_file(self.workspace, second["name"], second["table"]), Path(second["path"]))
        self.assertTrue(exported_file(self.workspace, first["name"], "export.json").is_file())
        for export, filename in (("..", "export.json"), (first["name"], "../../workspace.json"), (first["name"], "state.npz"), ("missing", "export.json")):
            with self.assertRaises(FileNotFoundError):
                exported_file(self.workspace, export, filename)

    def test_failed_write_leaves_nothing(self) -> None:
        with mock.patch.object(export_table.pq, "write_table", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                export_workspace(self.app, "test")
        self.assertEqual(list((self.workspace.path / "exports").iterdir()), [])
        self.assertEqual(list_exports(self.workspace), [])


class ExportApiTests(unittest.TestCase):
    def test_export_list_and_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recording = root / "recordings" / "rec-a.h5"
            recording.parent.mkdir()
            _write_recording(recording)
            name = _write_workspace(root / "workspaces", "demo", recording).info.name
            config = AppConfig(
                workspaces_root=root / "workspaces", dataset_root=root / "dataset", gpus=(0,), device="cpu",
                lab_library=root / "lab", library=root / "mine",
            )
            with TestClient(create_app(config), raise_server_exceptions=False) as client:
                response = client.post(f"/api/workspaces/{name}/export", json={"pixel_size_um": 1.25})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(result["pixel_size_um"], 1.25)
                listed = client.get(f"/api/workspaces/{name}/exports").json()
                self.assertEqual([e["name"] for e in listed], [result["name"]])
                table = client.get(result["download_url"])
                self.assertEqual(table.status_code, 200)
                self.assertEqual(table.content[:4], b"PAR1")
                self.assertEqual(client.get(result["metadata_url"]).json()["name"], result["name"])
                self.assertEqual(client.get(f"/api/workspaces/{name}/exports/{result['name']}/state.npz").status_code, 404)
                self.assertEqual(client.post(f"/api/workspaces/{name}/export").status_code, 200)


if __name__ == "__main__":
    unittest.main()
