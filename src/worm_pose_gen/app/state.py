"""The application's shared state: workspaces, recordings, models, the libraries and the job queue.

One instance per server, attached to the FastAPI app.  It owns the
segmenters and body-field networks loaded once on the app's device, a
``WorkspaceView`` for each of the last few workspaces opened (each with its
recording opened read-only, a ``RecordingSource``), the libraries, the
Labeling page's services and queues, and the ``JobRunner`` whose background
thread starts job processes on the local GPUs or submits them to SLURM.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import threading
from typing import Any

import torch

from ..compute import detect_compute, gpu_memory_used
from ..jobs import JobRunner, LocalGPUBackend, SlurmBackend
from .. import library
from ..library import Libraries
from ..pipeline import WorkspaceBusy, workspace_dataset
from ..recordings import DATASET_PATH, RecordingProber, RecordingSource, list_directory, thumbnail_png
from ..workspace import Workspace, list_workspaces, read_recording_shape
from .config import AppConfig
from .frame_view import Segmenters
from .workspace_view import WorkspaceView


# Workspace views kept open, each with its recording's reader (a video's holds up to frame_reader.CACHE_FRAMES decoded frames).
OPEN_WORKSPACES = 4


class NotFound(LookupError):
    """A workspace, job, queue or file the request named does not exist."""


def _validate_name(name: str) -> str:
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"invalid workspace name {name!r}")
    return name


def _integer(payload: dict[str, Any], key: str, default: int | None = None) -> int:
    """An integer field of a request body; a missing or malformed value is a bad request, not a server fault."""

    value = payload.get(key)
    if value is None or value == "":
        if default is None:
            raise ValueError(f"'{key}' is required")
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"'{key}' must be an integer, got {value!r}") from error


class AppState:
    """Everything the endpoints share; see the module docstring."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        config.workspaces_root.mkdir(parents=True, exist_ok=True)
        # Where jobs can run, found once at startup (``GET /api/compute``).
        self.compute = detect_compute()
        memory_used = lambda: gpu_memory_used(self.compute.gpus)  # noqa: E731 - other users' load, read when a GPU is chosen
        device = config.server_device(memory_used() if config.device is None and len(config.gpus) > 1 else None)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.segmenters = Segmenters(self.device)
        slurm = SlurmBackend(self.compute.slurm.defaults) if self.compute.slurm.available else None
        self.runner = JobRunner(
            config.jobs_root, LocalGPUBackend(list(config.gpus)), slurm, max_concurrent=config.max_concurrent, memory_used=memory_used,
        )
        self._views: OrderedDict[str, WorkspaceView] = OrderedDict()
        self._lock = threading.Lock()
        self.libraries = Libraries(config.lab_library, config.library)
        # Probes the Recordings screen's new files in the background (``<workspaces_root>/recordings_index.json``).
        self.prober = RecordingProber(config.recordings_cache)
        # The Labeling page: its frame services and the queues it walks (``<workspaces_root>/queues``).
        from .labeling import Labeling
        from .queues import QueueStore
        self.labeling = Labeling(self)
        self.queues = QueueStore(config.workspaces_root / "queues", self.runner)
        self._body_nets: dict[Path, Any] = {}
        self._body_net_lock = threading.Lock()
        from .network_fields import NetworkFields
        self.network_fields = NetworkFields(self)

    # ----------------------------------------------------------------- lifecycle

    def start(self) -> None:
        self.runner.recover()
        self.runner.start(interval=self.config.job_interval)

    def close(self) -> None:
        self.runner.stop()
        self.network_fields.close()
        with self._lock:
            views = list(self._views.values())
            self._views.clear()
        for view in views:
            if view.source is not None:
                view.source.close()

    def body_net(self, path: Path) -> Any:
        """The body-field network at ``path`` (a workspace's body model), loaded on the app's device at first use."""

        path = Path(path)
        with self._body_net_lock:
            if path not in self._body_nets:
                if not path.is_file():
                    raise ValueError(f"no body-field network checkpoint at {path}")
                from ..body_net import load_body_net

                self._body_nets[path] = load_body_net(path, self.device)
            return self._body_nets[path]

    # ---------------------------------------------------------------- workspaces

    def workspace_path(self, name: str) -> Path:
        return self.config.workspaces_root / _validate_name(name)

    def workspace(self, name: str) -> Workspace:
        view = self.view(name)
        view.refresh()
        return view.workspace

    def view(self, name: str) -> WorkspaceView:
        """The (cached) view of a workspace; ``NotFound`` when there is no such directory."""

        path = self.workspace_path(name)
        with self._lock:
            view = self._views.get(name)
            if view is not None:
                self._views.move_to_end(name)
        if view is None or not path.exists():
            if not (path / "workspace.json").exists():
                with self._lock:
                    self._views.pop(name, None)
                raise NotFound(f"unknown workspace {name!r}")
            workspace = Workspace.open(path)
            # Opened outside the lock: indexing a new video takes seconds, and other workspaces' requests go on meanwhile.
            try:
                source = RecordingSource(workspace.recording, self.config.dataset_root / "flat_fields", dataset=workspace_dataset(workspace))
                error = None
            except Exception as failure:  # noqa: BLE001 - any failure makes the recording unreadable, and the page says why
                source, error = None, f"{type(failure).__name__}: {failure}"
            opened = WorkspaceView(workspace, source, error)
            with self._lock:
                view = self._views.setdefault(name, opened)
                self._views.move_to_end(name)
                # Dropped, not closed: a request may still be reading the view; its reader is freed with the last reference.
                while len(self._views) > OPEN_WORKSPACES:
                    self._views.popitem(last=False)
            if view is not opened and source is not None:
                source.close()
        return view

    def has_workspace(self, name: str) -> bool:
        try:
            return (self.workspace_path(name) / "workspace.json").exists()
        except ValueError:
            return False

    def create_workspace(self, payload: dict[str, Any]) -> WorkspaceView:
        """A new workspace on ``recording``: the whole recording, named by the recording's id.

        ``first``, ``last`` and ``step`` narrow it (a developer's range; the
        Workspace page always takes the whole recording).  An existing name
        gets a numbered suffix.
        """

        recording = self.recording_path(str(payload["recording"]))
        if not recording.is_file():
            raise ValueError(f"recording {recording} does not exist")
        settings: dict[str, Any] = {}
        # The recording's own dataset (its setup's); /img_nir needs no setting.
        dataset = self.recording_dataset(recording)
        if dataset != DATASET_PATH:
            settings["dataset"] = dataset
        shape = read_recording_shape(recording, dataset)
        if shape is None:
            raise ValueError(f"{recording} cannot be read as a recording with dataset {dataset}")
        first, last = _integer(payload, "first", 0), _integer(payload, "last", shape[0] - 1)
        step = _integer(payload, "step", 1)
        base = recording.stem if (first, last, step) == (0, shape[0] - 1, 1) else f"{recording.stem}_f{first}-{last}"
        name = next(candidate for candidate in (base, *(f"{base}-{k}" for k in range(2, 1000))) if not self.workspace_path(candidate).exists())
        Workspace.create(self.config.workspaces_root, _validate_name(name), recording, first, last, step, settings=settings or None)
        return self.view(name)

    def workspace_of_recording(self, recording: Path) -> str | None:
        """The newest workspace on ``recording`` (one workspace per recording), or ``None``."""

        resolved = Path(recording).resolve()
        for info in list_workspaces(self.config.workspaces_root):
            if Path(info.recording).resolve() == resolved:
                return info.name
        return None

    # ---------------------------------------------------------------- recordings

    def recording_path(self, path: str) -> Path:
        """``path`` as a recording of a setup (in its folders or registered to it); ``NotFound`` for anything else."""

        recording = Path(path).expanduser()
        try:
            resolved = recording.resolve()
        except OSError as error:
            raise NotFound(f"no recording at {path}") from error
        if library.setup_for_recording(self.libraries, resolved) is None:
            raise NotFound(f"{path} does not belong to a setup; add it to one first")
        return resolved

    def recording_dataset(self, path: Path) -> str:
        """The HDF5 dataset of a recording's frames: its setup's video dataset (ignored for a video)."""

        setup = library.setup_for_recording(self.libraries, path)
        return str(library.get_setup(self.libraries, setup).video["dataset_path"]) if setup is not None else DATASET_PATH

    def browse(self, path: str | None, all_files: bool = False) -> dict[str, Any]:
        """Directories, HDF5 files and videos under ``path`` (the first recording folder when none is given), with the recording folders as shortcuts."""

        roots = [root for root in map(Path, library.recording_folders(self.libraries)) if root.is_dir()]
        start = Path(path).expanduser() if path else (roots[0] if roots else Path.home())
        try:
            listing = list_directory(start, all_files=all_files)
        except FileNotFoundError as error:
            raise NotFound(str(error)) from error
        except (NotADirectoryError, PermissionError) as error:
            raise ValueError(str(error)) from error
        listing["shortcuts"] = [{"name": p.name or str(p), "path": str(p)} for p in (*roots, Path.home())]
        return listing

    def thumbnail(self, path: str, frame: int, scale: float) -> bytes:
        recording = self.recording_path(path)
        if not recording.is_file():
            raise NotFound(f"no recording at {path}")
        try:
            # Thumbnails never exceed the frame: ``scale`` above 1 would build a gigapixel image in the server.
            return thumbnail_png(
                recording, frame, min(float(scale), 1.0), dataset_root=self.config.dataset_root, dataset=self.recording_dataset(recording)
            )
        except OSError as error:
            raise ValueError(f"{recording.name} cannot be read as a recording") from error

    # --------------------------------------------------------------------- edits

    def check_writable(self, name: str) -> None:
        """``WorkspaceBusy`` (a 409) when a job is running on workspace ``name``: an edit made now would land on arrays the job is rewriting.

        The job's process holds the workspace lock for its whole run, so
        without this check an edit would wait for it (minutes to hours) and
        then apply a stale intention; the lock's own timeout covers writers
        this server does not know about.
        """

        running = [r for r in self.runner.list("running") if r.spec.workspace == name]
        if running:
            job = running[0]
            raise WorkspaceBusy(f"job {job.id} ({job.spec.label or job.spec.kind}) is writing workspace {name}; wait for it to finish or cancel it")
