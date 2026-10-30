"""Labeling queues: ordered frames to label, and the Relabel round trip back to the workspace.

A queue is one JSON file, ``<workspaces_root>/queues/<id>.json``, so it
survives restarts::

    {"id", "kind", "name", "setup", "origin", "author", "created_at",
     "workspace": name (relabel), "job": id (spread), "state": "finding" | "ready" | "failed", "error",
     "types": [frame type, ...] (spread: what the search was limited to; empty for all),
     "entries": [{"path", "recording", "frame", "uncertainty"?, "types"?, "saved": label identity | null}]}

There are three kinds (``docs/APP_SIMPLIFICATION.md``, section 3):

``relabel``
    keyframes of a workspace stretch, made by the Workspace's Relabel fix
    (:func:`relabel_queue`).  Its labels have origin ``fix``.  When every
    keyframe is saved the page offers *Back to workspace*, and the
    Workspace calls :func:`stitch`.
``spread``
    frames a *Find frames* job picked over recordings of the setup,
    optionally only frames of some types (``contact``, ``edge``, ...;
    :data:`frame_search.TYPES`) (:func:`spread_queue`,
    :mod:`frame_search`); each entry records its ``types``; the queue stays
    ``finding`` until the job is done and then takes the job's frames.
    Its labels have origin ``spread``, so their test labels can go into a
    benchmark.
``manifest``
    the frames of a labeling manifest (``docs/labeling_*/manifest.json``),
    made from the command line by ``worm-pose-app --queue`` for the
    developer (:func:`manifest_queue`); targeted frames, origin ``fix``.

An entry is done when it was saved through the queue: the save records the
exact label revision (``saved``), which is what a stitch reads.
"""

from __future__ import annotations

import getpass
import json
from pathlib import Path
import sys
import uuid
from typing import Any, Callable, Sequence

import numpy as np

from .. import edits, frame_search, library
from ..frame_reader import open_frames
from ..jobs import JobSpec
from ..library.roots import locked, read_json, write_json
from ..workspace import utc_now
from .state import NotFound

KINDS = ("relabel", "spread", "manifest")
ORIGIN = {"relabel": "fix", "spread": "spread", "manifest": "fix"}
FIND_JOB_KIND = "find_frames"
# A New queue looks at no more frames than this.
MAX_FRAMES = 500


class QueueStore:
    """The queue files under ``root``; reads finish a ``finding`` queue whose job is done."""

    def __init__(self, root: Path, runner: Any = None) -> None:
        self.root = Path(root)
        self.runner = runner

    def path(self, queue_id: str) -> Path:
        if not queue_id or not queue_id.replace("-", "").isalnum():
            raise NotFound(f"unknown queue {queue_id!r}")
        return self.root / f"{queue_id}.json"

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        queue = {
            "id": uuid.uuid4().hex[:10], "author": getpass.getuser(), "created_at": utc_now(), "state": "ready", "error": None,
            "workspace": None, "job": None, "origin": ORIGIN[fields["kind"]], **fields,
        }
        queue["entries"] = [{"saved": None, **entry} for entry in fields.get("entries", [])]
        with locked(self.root):
            write_json(self.path(queue["id"]), queue)
        return queue

    def get(self, queue_id: str) -> dict[str, Any]:
        queue = read_json(self.path(queue_id))
        if queue is None:
            raise NotFound(f"unknown queue {queue_id!r}")
        if queue["state"] == "finding":
            queue = self._finish(queue)
        return queue

    def list(self, setup: str | None = None) -> list[dict[str, Any]]:
        queues = []
        for path in sorted(self.root.glob("*.json")) if self.root.is_dir() else ():
            queue = self.get(path.stem)
            if setup is None or queue["setup"] == setup:
                queues.append(queue)
        return sorted(queues, key=lambda q: q["created_at"], reverse=True)

    def update(self, queue_id: str, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        with locked(self.root):
            queue = read_json(self.path(queue_id))
            if queue is None:
                raise NotFound(f"unknown queue {queue_id!r}")
            change(queue)
            write_json(self.path(queue_id), queue)
        return queue

    def delete(self, queue_id: str) -> None:
        with locked(self.root):
            path = self.path(queue_id)
            if not path.exists():
                raise NotFound(f"unknown queue {queue_id!r}")
            path.unlink()

    def mark_saved(self, queue_id: str, entry: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
        """Record the label revision saved for a queue entry; returns the queue's summary."""

        def change(queue: dict[str, Any]) -> None:
            for item in queue["entries"]:
                if item["recording"] == entry["recording"] and int(item["frame"]) == int(entry["frame"]):
                    item["saved"] = {k: identity[k] for k in ("scope", "setup", "recording", "frame", "revision", "sha256")}
                    return
            raise ValueError(f"{entry['recording']} frame {entry['frame']} is not in queue {queue['id']}")

        return summary(self.update(queue_id, change))

    def _finish(self, queue: dict[str, Any]) -> dict[str, Any]:
        """A finding queue whose job finished: its frames, or the job's error."""

        if self.runner is None or not queue.get("job"):
            return queue
        try:
            job = self.runner.get(queue["job"])
        except KeyError:
            job = None
        if job is not None and not job.finished:
            return queue

        def change(stored: dict[str, Any]) -> None:
            if stored["state"] != "finding":
                return
            if job is not None and job.state == "done" and job.result and "entries" in job.result:
                stored["entries"] = [{"saved": None, **entry} for entry in job.result["entries"]]
                stored["state"] = "ready"
            else:
                stored["state"] = "failed"
                stored["error"] = "the search job is gone" if job is None else (job.error or f"the search job was {job.state}")

        return self.update(queue["id"], change)


def summary(queue: dict[str, Any]) -> dict[str, Any]:
    """A queue without its entries, with its progress and the first entry still to label (``None`` when done)."""

    entries = queue.get("entries") or []
    saved = sum(entry.get("saved") is not None for entry in entries)
    first = next((k for k, entry in enumerate(entries) if entry.get("saved") is None), None)
    return {**{k: v for k, v in queue.items() if k != "entries"},
            "progress": {"total": len(entries), "saved": saved, "remaining": len(entries) - saved},
            "first_unsaved": first, "complete": bool(entries) and first is None}


def detail(queue: dict[str, Any]) -> dict[str, Any]:
    return {**summary(queue), "entries": queue.get("entries") or []}


# --------------------------------------------------------------------------- making queues


def _setup_of(app: Any, path: Path) -> str:
    setup = library.setup_for_recording(app.libraries, path)
    if setup is None:
        raise ValueError(f"{path.name} belongs to no setup; add it to a setup on the Workspace page first")
    return setup


def relabel_queue(app: Any, workspace_name: str, frames: Sequence[Any]) -> dict[str, Any]:
    """The keyframes of a workspace stretch as a queue (Relabel)."""

    if not isinstance(frames, list) or not frames:
        raise ValueError("'frames' must be a non-empty list of frame numbers")
    workspace = app.workspace(workspace_name)
    try:
        ordered = sorted({int(frame) for frame in frames})
    except (TypeError, ValueError) as error:
        raise ValueError("'frames' must be frame numbers") from error
    for frame in ordered:
        workspace.row_of(frame)
    recording = workspace.recording.resolve()
    setup = _setup_of(app, recording)
    entries = [{"path": str(recording), "recording": library.recording_id(recording), "frame": frame} for frame in ordered]
    name = f"Relabel {workspace_name} frames {ordered[0]}–{ordered[-1]}"
    return app.queues.create({"kind": "relabel", "name": name, "setup": setup, "workspace": workspace_name, "entries": entries})


TYPE_NAMES = {"contact": "self-contact", "edge": "edge", "pieces": "in pieces", "empty": "no worm", "clear": "clear"}


def spread_queue(app: Any, setup_ref: str, paths: Sequence[Any], frames: Any, types: Sequence[Any] | None = None) -> dict[str, Any]:
    """A New queue: a Find-frames job over recordings of the setup, of any of ``types`` (all by default); the queue fills when the job is done."""

    from .routers.jobs import place

    library.get_setup(app.libraries, setup_ref)
    if not isinstance(paths, list) or not paths:
        raise ValueError("choose at least one recording")
    try:
        count = int(frames)
    except (TypeError, ValueError) as error:
        raise ValueError("'frames' must be a number") from error
    if not 1 <= count <= MAX_FRAMES:
        raise ValueError(f"'frames' must be between 1 and {MAX_FRAMES}")
    if types is not None and not isinstance(types, list):
        raise ValueError("'types' must be a list of frame types")
    requested = set(map(str, types or ()))
    chosen = [t for t in frame_search.TYPES if t in requested]
    unknown = sorted(requested - set(frame_search.TYPES))
    if unknown:
        raise ValueError(f"unknown frame types {unknown}; choose from {list(frame_search.TYPES)}")
    labeled: dict[str, set[int]] = {}
    for record in library.Collection(app.libraries, setup_ref).labels():
        labeled.setdefault(record.recording, set()).add(record.frame)
    setup = library.get_setup(app.libraries, setup_ref)
    recordings = []
    for value in paths:
        path = Path(str(value)).expanduser().resolve()
        if _setup_of(app, path) != setup_ref:
            raise ValueError(f"{path.name} does not belong to {setup_ref}")
        count_frames = _frame_count(path, str(setup.video["dataset_path"]))
        name = library.recording_id(path)
        recordings.append({"path": str(path), "id": name, "frames": count_frames, "exclude": sorted(labeled.get(name, ()))})
    model = setup.defaults.get("mask")
    if chosen and not model:
        raise ValueError(f"{setup_ref} has no mask model yet, so frames cannot be chosen by type")
    spec_json = {
        "recordings": recordings, "frames": count, "types": chosen, "model": model,
        "libraries": {"lab": None if app.libraries.lab is None else str(app.libraries.lab), "personal": str(app.libraries.personal)},
        "video": setup.video, "fps": setup.fps, "dataset_root": str(app.config.dataset_root),
    }
    names = ", ".join(r["id"] for r in recordings[:3]) + ("…" if len(recordings) > 3 else "")
    kinds = f"{', '.join(TYPE_NAMES[t] for t in chosen)} " if chosen else ""
    spec = JobSpec(kind=FIND_JOB_KIND, params={"recordings": [r["id"] for r in recordings], "frames": count, "types": chosen, "model": model},
                   gpus=1 if app.config.gpus and model else 0, label=f"Find {count} {kinds}frames in {names}")
    job = app.runner.submit(place(app, spec, {}), find_command(spec_json))
    return app.queues.create({
        "kind": "spread", "name": f"{count} {kinds}frames from {names}", "setup": setup_ref, "state": "finding", "job": job.id,
        "model": model, "types": chosen, "entries": [],
    })


def find_command(spec: dict[str, Any]) -> list[str]:
    """The argv of a Find-frames job (:func:`frame_search.main`)."""

    return [sys.executable, "-m", "worm_pose_gen.frame_search", "--spec", json.dumps(spec)]


def _frame_count(path: Path, dataset: str) -> int:
    try:
        with open_frames(path, dataset) as frames:
            return frames.frame_count
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(f"{path.name} cannot be read: {error}") from error


def manifest_queue(app: Any, manifest: Path) -> dict[str, Any]:
    """The frames of a labeling manifest (``{"name", "recordings": {alias: {"path"}}, "frames": [{"recording", "frame_index"}]}``)."""

    path = Path(manifest).expanduser().resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("recordings"), dict) or not isinstance(data.get("frames"), list):
        raise ValueError("a manifest needs a recordings object and a frames list")
    files: dict[str, Path] = {}
    for alias, record in data["recordings"].items():
        file = Path(record["path"]).expanduser()
        files[alias] = (file if file.is_absolute() else path.parent / file).resolve()
    setups = {_setup_of(app, file) for file in files.values()}
    if len(setups) != 1:
        raise ValueError(f"a manifest's recordings must belong to one setup, not {sorted(setups)}")
    entries = []
    for item in data["frames"]:
        file = files[item["recording"]]
        entries.append({"path": str(file), "recording": library.recording_id(file), "frame": int(item["frame_index"])})
    return app.queues.create({"kind": "manifest", "name": str(data.get("name") or path.parent.name), "setup": setups.pop(), "entries": entries})


# --------------------------------------------------------------------------- the Relabel round trip


def stitch(app: Any, queue_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write a finished Relabel queue's keyframe labels into its workspace and start the stitch between them.

    Each keyframe label's mask becomes the workspace's mask override of that
    frame (an ordinary mask edit, so the Fixes list can undo it; a frame
    whose override already equals the label is left alone).  Each keyframe
    that is not mask-only gives the stitch its pose: the head-first
    centerline and width profile of the label's body targets, built here
    when the background job has not built them yet.  Answers like ``POST
    /api/workspaces/<ws>/fixes/stitch``: ``{job, preview, plan}``.
    """

    from . import fixes as fix_ops
    from .routers.jobs import place

    queue = app.queues.get(queue_id)
    if queue["kind"] != "relabel":
        raise ValueError(f"queue {queue_id} is not a Relabel queue")
    missing = [e["frame"] for e in queue["entries"] if e.get("saved") is None]
    if missing:
        raise ValueError(f"{len(missing)} keyframe(s) are not labeled yet: frames {', '.join(map(str, missing))}")
    name = queue["workspace"]
    view = app.view(name)
    app.check_writable(name)
    view.refresh()
    workspace = view.workspace
    keyframes = []
    for entry in queue["entries"]:
        saved = entry["saved"]
        record = library.Collection(app.libraries, saved["setup"]).get(
            saved["recording"], int(saved["frame"]), int(saved["revision"]), saved["scope"])
        label = record.load()
        row = workspace.row_of(int(entry["frame"]))
        current = workspace.get_override_mask(row)
        if current is None or not np.array_equal(current, label.mask):
            edits.set_mask(workspace, row, label.mask, note=f"Relabel keyframe frame {entry['frame']}")
        if record.mask_only or not (label.mask == 1).any():
            continue
        with app.labeling.fit_lock:
            library.build_targets(app.libraries, record, device=app.device)
        _, arrays = library.load_targets(app.libraries, record)
        keyframes.append({"frame": int(entry["frame"]), "centerline_xy": arrays["centerline_xy"].tolist(),
                          "width_profile": arrays["width_profile"].tolist()})
    view.invalidate()
    if not keyframes:
        raise ValueError("every keyframe is mask-only; label the body of at least one to stitch")
    spec, command, answer = fix_ops.stitch_job(view, {"keyframes": keyframes, "params": params or {}})
    job = app.runner.submit(place(app, spec, {}), command)
    return {"job": job.to_dict(), **answer}
