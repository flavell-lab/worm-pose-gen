"""The FastAPI pose app over a synthetic recording of a library setup, a fitted workspace, and a job runner on a fake GPU pool.

Jobs run real subprocesses: a python one-liner for the ``command`` kind, and
the pipeline's segment stage on the CPU for the ``stage`` kind (with
``pipeline.stage_command`` patched so the child never touches a GPU).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import gc
import time
import unittest
from unittest import mock
import weakref

from PIL import Image
from fastapi.testclient import TestClient

from worm_pose_gen import library, pipeline
from worm_pose_gen.app import AppConfig, create_app
from worm_pose_gen.app import state as app_state
from worm_pose_gen.workspace import Workspace

from tests.test_frame_view import FRAMES, HEIGHT, WIDTH, _write_recording, _write_workspace

SRC = str(Path(__file__).resolve().parents[1] / "src")
SEGMENT_PARAMS = {"checkpoint": None, "flat_field": False, "min_worm_pixels": 200, "slab": 4}


def _cpu_stage_command(workspace_path: Path | str, stage: str, params: dict | None) -> list[str]:
    """``pipeline.stage_command`` for tests: the same CLI, this interpreter, the CPU."""

    argv = ["--workspace", str(workspace_path), "--stage", stage, "--params", json.dumps(params or {}), "--device", "cpu"]
    code = f"import sys; sys.path.insert(0, {SRC!r}); from worm_pose_gen.pipeline import main; sys.exit(main({argv!r}))"
    return [sys.executable, "-c", code]


def _reporting_command(message: str, result: dict) -> list[str]:
    code = (
        f"import sys, os; sys.path.insert(0, {SRC!r}); from worm_pose_gen.jobs import report_progress\n"
        "report_progress(0.5, 'half way')\nprint('hello from the job')\n"
        f"report_progress(1.0, {message!r}, {{**{result!r}, 'gpu': os.environ.get('CUDA_VISIBLE_DEVICES')}})\n"
    )
    return [sys.executable, "-c", code]


class AppTests(unittest.TestCase):
    DEV = False

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.root = root
        cls.recording = root / "recordings" / "rec-a.h5"
        cls.recording.parent.mkdir()
        _write_recording(cls.recording)
        # The setup whose recordings the app serves.
        library.write_setup(root / "lab", "rig", name="Rig", fps=20.0)
        library.add_recording_folder(library.Libraries(root / "lab", root / "mine"), root / "recordings", "lab:rig")
        cls.fitted = _write_workspace(root / "workspaces", "fitted", cls.recording)
        config = AppConfig(
            workspaces_root=root / "workspaces", dataset_root=root / "dataset", gpus=(0,), device="cpu", job_interval=0.1,
            lab_library=root / "lab", library=root / "mine", dev=cls.DEV,
        )
        cls.app = create_app(config)
        cls.client = TestClient(cls.app, raise_server_exceptions=False)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.__exit__(None, None, None)
        cls._directory.cleanup()

    # ----- helpers

    def get(self, path: str, status: int = 200) -> dict | list:
        response = self.client.get(path)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def post(self, path: str, payload: dict, status: int = 200) -> dict | list:
        response = self.client.post(path, json=payload)
        self.assertEqual(response.status_code, status, response.text)
        return response.json()

    def wait_for_job(self, job_id: str, timeout: float = 120.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(f"/api/jobs/{job_id}")
            if record["state"] in ("done", "failed", "cancelled"):
                return record
            time.sleep(0.1)
        self.fail(f"job {job_id} did not finish: {self.get(f'/api/jobs/{job_id}')}")

    # ----- tests

    def test_static_files_config_and_errors(self) -> None:
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("text/html", page.headers["content-type"])
        self.assertIn("Worm pose", page.text)
        script = self.client.get("/static/shell.js")
        self.assertIn("javascript", script.headers["content-type"])
        self.assertIn("text/css", self.client.get("/static/style.css").headers["content-type"])
        config = self.get("/api/config")
        self.assertEqual(config["dev"], self.DEV)
        self.assertFalse(config["gpu"])
        self.assertIn("personal", config["libraries"])
        self.assertEqual(self.client.get("/static/nope.js").status_code, 404)
        self.assertEqual(self.client.get("/static/nope.js").json(), {"error": "no static file 'nope.js'"})
        self.assertEqual(self.client.get("/nowhere").json(), {"error": "Not Found"})
        # The old viewer, run catalog, review notes and region endpoints are gone.
        for path in ("/api/state", "/api/run?name=x", "/api/notes", "/api/outcomes", "/api/stages", "/api/recordings", "/api/corpus"):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_thumbnails(self) -> None:
        response = self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=1&scale=0.5")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertTrue(response.content.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=99").status_code, 400)
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={self.root}/recordings/none.h5&frame=0").status_code, 404)
        # Only recordings of a setup are served, and never larger than the frame.
        outside = self.root / "elsewhere.h5"
        shutil.copyfile(self.recording, outside)
        self.assertIn("does not belong to a setup", self.client.get(f"/api/recordings/thumbnail?path={outside}&frame=0").json()["error"])
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={outside}&frame=0").status_code, 404)
        self.assertEqual(self.client.get("/api/recordings/thumbnail?path=/etc/passwd&frame=0").status_code, 404)
        huge = self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=0&scale=40")
        self.assertEqual(huge.status_code, 200)
        self.assertEqual(Image.open(io.BytesIO(huge.content)).size, (WIDTH, HEIGHT))
        self.assertEqual(self.client.get(f"/api/recordings/thumbnail?path={self.recording}&frame=0&scale=0").status_code, 400)
        garbage = self.root / "recordings" / "garbage.h5"
        garbage.write_bytes(b"not hdf5" * 32)
        try:
            response = self.client.get(f"/api/recordings/thumbnail?path={garbage}&frame=0")
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["error"], "garbage.h5 cannot be read as a recording")
        finally:
            garbage.unlink()

    def test_workspace_payloads(self) -> None:
        payload = self.get("/api/workspaces/fitted")
        self.assertEqual((payload["name"], payload["frames"]), ("fitted", [0, FRAMES - 1]))
        self.assertEqual(payload["summary"]["fitted"], FRAMES)
        self.assertEqual(payload["provenance_counts"], {"chain_forward": 1, "independent_fit": FRAMES - 1})
        self.assertEqual(payload["provenance"]["algorithms"], ["chain_forward", "independent_fit"])
        self.assertEqual(payload["provenance"]["edited"], [0] * FRAMES)
        self.assertEqual(payload["image_shape"], [HEIGHT, WIDTH])
        self.assertEqual(payload["series"]["classification"], ["clean"] * (FRAMES - 1) + ["ambiguous"])
        self.assertEqual(self.get("/api/workspaces/missing", 404), {"error": "unknown workspace 'missing'"})
        self.assertEqual(self.get("/api/workspaces/a%5Cb", 400)["error"], "invalid workspace name " + repr("a\\b"))

        frame = self.get(f"/api/workspaces/fitted/frame?frame={FRAMES - 1}&raw=1")
        self.assertEqual((frame["height"], frame["width"]), (HEIGHT, WIDTH))
        self.assertTrue(frame["image_raw"].startswith("data:image/jpeg"))
        self.assertIn("tube_independent", frame["layers"])
        self.assertFalse(frame["has_stored_mask"])
        self.assertEqual(frame["provenance"]["algorithm"], "chain_forward")
        self.assertNotIn("hypotheses", frame["pose"])
        # The fixture's summary names a checkpoint that does not exist: only the developer's layers ask for it.
        self.assertNotIn("probability", frame["layers"])
        self.assertEqual(any("checkpoint" in e for e in frame["errors"]), self.DEV)
        light = self.get("/api/workspaces/fitted/frame?frame=2&detail=light")
        self.assertEqual(sorted(light["layers"]), ["image"])
        self.assertEqual(self.get("/api/workspaces/fitted/frame?frame=2&detail=medium", 400)["error"], "detail must be 'full' or 'light'")
        self.assertEqual(self.get("/api/workspaces/fitted/frame?frame=99", 400)["error"], "frame 99 is not in this workspace")
        self.assertIn("frame", self.get("/api/workspaces/fitted/frame?frame=x", 400)["error"])
        # Read-only runs, snapshots, starts, poses and raw edits are not served any more.
        self.assertEqual(self.client.post("/api/workspaces/import", json={"run": "x"}).status_code, 405)
        for path in ("snapshot", "edits"):
            self.assertEqual(self.client.post(f"/api/workspaces/fitted/{path}", json={}).status_code, 404, path)
        for path in ("starts?frame=0", "pose?frame=0", "segment?frame=0", "inspection", "candidates", "region?frame=0"):
            self.assertEqual(self.client.get(f"/api/workspaces/fitted/{path}").status_code, 404, path)

    def test_only_the_last_workspaces_opened_stay_open(self) -> None:
        state = self.app.state.app_state
        _write_workspace(self.root / "workspaces", "second", self.recording)
        self.addCleanup(shutil.rmtree, self.root / "workspaces" / "second")
        enabled = gc.isenabled()
        gc.disable()
        self.addCleanup(lambda: gc.enable() if enabled else None)
        with mock.patch.object(app_state, "OPEN_WORKSPACES", 1):
            held = state.view("fitted")
            reader = weakref.ref(held.source.frames)
            self.assertIs(state.view("fitted"), held)
            state.view("second")
            # Evicted, the view still serves the request that holds it, and is freed with it (no cycle collection).
            self.assertEqual(held.source.read(0).shape, (HEIGHT, WIDTH))
            self.assertIsNot(state.view("fitted"), held)
            del held
            self.assertIsNone(reader())
            # The Recordings screen's status does not open the workspace or its recording.
            with mock.patch.object(state, "view", side_effect=AssertionError("opened")):
                self.assertEqual(self.get("/api/workspaces/second/status")["name"], "second")

    def test_stage_job_on_a_new_workspace(self) -> None:
        Workspace.create(self.root / "workspaces", "fresh", self.recording, 0, FRAMES - 1)
        empty = self.get("/api/workspaces/fresh")
        self.assertEqual(empty["series"]["fitted"], [0] * FRAMES)
        self.assertEqual(empty["series"]["classification"], ["unfitted"] * FRAMES)
        frame = self.get("/api/workspaces/fresh/frame?frame=2")
        self.assertEqual(sorted(frame["layers"]), ["image"])
        self.assertIsNone(frame["pose"])

        self.assertIn("unknown stage", self.post("/api/jobs", {"kind": "stage", "workspace": "fresh", "stage": "nope"}, 400)["error"])
        self.assertEqual(self.post("/api/jobs", {"kind": "stage", "workspace": "missing", "stage": "segment"}, 404)["error"], "unknown workspace 'missing'")
        self.assertIn("unknown job kind", self.post("/api/jobs", {"kind": "region"}, 400)["error"])
        self.assertEqual(self.get("/api/jobs/j00000099", 404)["error"], "unknown job 'j00000099'")
        self.assertIn("unknown job state", self.get("/api/jobs?state=sleeping", 400)["error"])

        with mock.patch.object(pipeline, "stage_command", _cpu_stage_command):
            submitted = self.post("/api/jobs", {"kind": "stage", "workspace": "fresh", "stage": "segment", "params": SEGMENT_PARAMS, "label": "seg"})
        self.assertEqual(submitted["state"], "queued")
        self.assertEqual(submitted["spec"], {"kind": "stage", "params": {"stage": "segment", "params": {**SEGMENT_PARAMS, "mask_source": "segmenter", "body_net": None, "dataset_root": str(self.root / "dataset")}}, "workspace": "fresh", "frames": [0, FRAMES - 1], "gpus": 1, "gpu": None, "label": "seg", "run_on": "local", "slurm": None})
        self.assertEqual(submitted["command"][0], sys.executable)
        self.assertIn("--stage', 'segment'", submitted["command"][-1])
        record = self.wait_for_job(submitted["id"])
        self.assertEqual(record["state"], "done", record)
        self.assertEqual(record["gpu"], 0)
        self.assertEqual(record["progress"], 1.0)
        self.assertEqual(record["result"]["frames"], FRAMES)
        self.assertEqual(record["result"]["frames_with_worm"], FRAMES)
        self.assertEqual(self.get(f"/api/jobs/{record['id']}/log?tail=5")["id"], record["id"])
        self.assertIn(record["id"], [j["id"] for j in self.get("/api/jobs?state=done")])
        self.assertEqual(self.get("/api/jobs?state=running"), [])

        # The workspace now has masks; the page serves the stored one without a segmenter.
        workspace = Workspace.open(self.root / "workspaces" / "fresh")
        self.assertEqual(workspace.mask_rows().tolist(), list(range(FRAMES)))
        opened = self.get("/api/workspaces/fresh")
        self.assertTrue(opened["summary"]["has_masks"])
        self.assertEqual(opened["mask_rows"], list(range(FRAMES)))
        self.assertGreater(opened["series"]["worm_pixels"][2], 200)
        frame = self.get("/api/workspaces/fresh/frame?frame=2")
        self.assertEqual(sorted(frame["layers"]), ["image", "mask_final"])
        self.assertEqual(frame["mask_final_source"], "stored")
        self.assertTrue(frame["has_stored_mask"])
        self.assertEqual(frame["errors"], [])
        self.assertEqual(frame["mask_stats_stored"]["worm_pixels"], int(workspace.get_mask(2).sum()))

    def test_file_explorer(self) -> None:
        listing = self.get(f"/api/files?path={self.root}")
        self.assertEqual(listing["path"], str(self.root.resolve()))
        self.assertIn("recordings", [e["name"] for e in listing["entries"] if e["kind"] == "dir"])
        # The setups' recording roots are the shortcuts, and where the explorer starts.
        self.assertEqual([s["path"] for s in listing["shortcuts"]][0], str(self.root / "recordings"))
        inside = self.get(f"/api/files?path={self.root / 'recordings'}")
        self.assertEqual([(e["name"], e["kind"]) for e in inside["entries"]], [("rec-a.h5", "h5")])
        self.assertEqual(self.get("/api/files")["path"], str((self.root / "recordings").resolve()))
        (self.root / "recordings" / "notes.txt").write_text("x")
        self.addCleanup(lambda: (self.root / "recordings" / "notes.txt").unlink())
        self.assertEqual([e["name"] for e in self.get(f"/api/files?path={self.root / 'recordings'}")["entries"]], ["rec-a.h5"])
        everything = self.get(f"/api/files?path={self.root / 'recordings'}&all=1")
        self.assertEqual([(e["name"], e["kind"]) for e in everything["entries"]], [("notes.txt", "file"), ("rec-a.h5", "h5")])
        self.assertTrue(everything["all_files"])
        self.assertEqual(self.get(f"/api/files?path={self.root}/nope", 404)["error"], f"{self.root}/nope does not exist")
        self.assertIn("not a directory", self.get(f"/api/files?path={self.recording}", 400)["error"])

    def test_command_jobs_and_cancel(self) -> None:
        submitted = self.post("/api/jobs", {"kind": "command", "command": _reporting_command("all done", {"answer": 42}), "label": "one-liner"})
        self.assertEqual((submitted["spec"]["kind"], submitted["spec"]["label"]), ("command", "one-liner"))
        record = self.wait_for_job(submitted["id"], timeout=30.0)
        self.assertEqual(record["state"], "done", record)
        self.assertEqual(record["message"], "all done")
        self.assertEqual(record["result"], {"answer": 42, "gpu": "0"})
        self.assertIn("hello from the job", self.get(f"/api/jobs/{record['id']}/log")["log"])
        self.assertIn("a command job needs", self.post("/api/jobs", {"kind": "command", "command": "ls"}, 400)["error"])

        failing = self.post("/api/jobs", {"kind": "command", "command": [sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"]})
        record = self.wait_for_job(failing["id"], timeout=30.0)
        self.assertEqual(record["state"], "failed")
        self.assertIn("boom", record["error"])

        waiting = self.post("/api/jobs", {"kind": "command", "command": [sys.executable, "-c", "import time; time.sleep(30)"]})
        deadline = time.monotonic() + 30
        while self.get(f"/api/jobs/{waiting['id']}")["state"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.1)
        cancelled = self.post(f"/api/jobs/{waiting['id']}/cancel", {})
        self.assertEqual(cancelled["state"], "cancelled")
        self.assertEqual(self.post("/api/jobs/j00000099/cancel", {}, 404)["error"], "unknown job 'j00000099'")
        newest_first = [j["id"] for j in self.get("/api/jobs")]
        self.assertEqual(newest_first, sorted(newest_first, reverse=True))


class DevAppTests(AppTests):
    """The same app with ``--dev``: a rested frame also asks the workspace's segmenter for the developer's layers."""

    DEV = True
    # Only the payloads differ in developer mode.
    test_thumbnails = test_stage_job_on_a_new_workspace = test_file_explorer = test_command_jobs_and_cancel = None
    test_only_the_last_workspaces_opened_stay_open = None


if __name__ == "__main__":
    unittest.main()


class StartupTests(unittest.TestCase):
    def test_a_taken_port_moves_to_the_next_free_one(self) -> None:
        import socket

        from worm_pose_gen.app import free_port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            self.assertGreater(free_port("127.0.0.1", start=port), port)

    def test_the_tunnel_reaches_the_node_through_the_login_node_on_engaging(self) -> None:
        from worm_pose_gen.app import tunnel_command

        self.assertEqual(tunnel_command(8770, "flv-c3", "kim"), "ssh -N -L 8770:localhost:8770 kim@flv-c3")
        self.assertEqual(tunnel_command(8768, "node2433.inband", "kim"), "ssh -N -L 8768:localhost:8768 -J kim@orcd-login.mit.edu kim@node2433")
        self.assertEqual(tunnel_command(8768, "orcd-login001", "kim"), "ssh -N -L 8768:localhost:8768 kim@orcd-login001.mit.edu")
