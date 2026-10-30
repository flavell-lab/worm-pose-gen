"""Analyse: one workspace per recording, run with its setup's models (``docs/APP_SIMPLIFICATION.md`` section 2).

A recording belongs to a setup (:func:`library.setup_for_recording`), and
the setup names the default model of each role (``mask``: the segmenter
whose masks the pipeline fits; ``body``: the body-field network that scores
the fits and gives the A-P field).  **Opening** a recording makes its
workspace if it has none (always the whole recording; a developer may give
a range) and starts nothing; the analysis starts from the workspace.
**Analyse** takes the setup's models, or the ones the user picked instead,
and where the masks come from (``mask_source``: the mask model, or the body
model's own mask output, which then needs no mask model; the default is the
body model when the setup's mask default is a body-field network),
resolves each reference to its weights, takes the pipeline settings the
user changed (``PIPELINE_SETTINGS``), records the setup, the model
references, the mask source and the pipeline settings in the workspace
settings (the header, the export and the next Analyse read them back), and
submits one job of kind ``analyse`` that
runs the default stages in one process (``pipeline.stages_command``), so its
progress spans the whole analysis and it survives the browser closing.
Analysing again replaces the poses; the user's mask edits stay.  The
analysis is cancelled like any job (``POST /api/jobs/{id}/cancel``).

**Status** answers what the Recordings screen and the Workspace header show
without loading a workspace's arrays: the models, the analysis job and its
progress, and the state (``not_analysed``, ``queued``, ``analysing``,
``failed``, ``cancelled``, ``analysed``, ``issues``, ``reviewed``,
``exported``).  ``failed`` and ``cancelled`` are only for a workspace that
has never been analysed; a re-analysis that fails or is cancelled leaves
the state of the previous analysis.  The issue
counts come from the summary the Issues panel last wrote
(:func:`inspection.cached_issue_summary`); until the workspace is opened
after an analysis they are unknown and the state is ``analysed``.

The **kymograph** is the curvature along the body over time: one column per
workspace row, one row per midline point head to tail, the signed curvature
times the body length (dimensionless, so worms of any size and pixel size
compare) clipped to ``±KYMOGRAPH_RANGE``.  It goes to the browser as one
8-bit PNG: 0 where the frame has no pose, else ``1 + round(254 * (kL +
R) / 2R)`` (128 is straight).  Beside it goes a second 8-bit PNG of the
same size, 255 where the midline point lies outside the frame (the part of
the body the fit completed off camera), else 0; ``null`` when the
recording's frame size is unknown.
"""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import library, pipeline
from ..batch_fit import PRESETS
from ..jobs import FINISHED_STATES, JobRecord, JobSpec
from .images import data_url
from ..library.setups import ROLE_OUTPUTS
from ..recordings import RecordingInfo, list_recordings
from ..workspace import Workspace, _write_json_atomic, list_workspaces, utc_now
from .exporting import list_exports
from .inspection import cached_issue_summary
from .state import AppState, NotFound

ANALYSE_JOB_KIND = "analyse"
# The model kind that can fill each role.  Masks come from the segmenter
# unless the analysis asks for the body model's (``pipeline.MASK_SOURCES``).
ROLE_KINDS = {"mask": "segmenter", "body": "body_net"}
# The pipeline settings Analyse offers (the rest keep the stages' defaults).  Each is a
# parameter of the ``stages`` that read it (``pipeline.STAGE_PARAMS``), all from the one
# parameter dict of the analysis, so a key means the same in every one of them.  Its type,
# default and help come from the first stage's field; flat_field's default is the setup's.
PIPELINE_SETTINGS: tuple[dict[str, Any], ...] = (
    {"key": "preset", "stages": ("fit",), "label": "Fitting schedule", "choices": tuple(PRESETS)},
    {"key": "threshold", "stages": ("segment", "prior"), "label": "Mask threshold", "min": 0.05, "max": 0.95},
    {"key": "flat_field", "stages": ("segment", "prior"), "label": "Flat field"},
    {"key": "prior", "stages": ("prior", "fit"), "label": "Recording prior", "choices": ("bootstrap", "none")},
    {"key": "rebootstrap", "stages": ("prior",), "label": "Bootstrap the prior again"},
    {"key": "min_score", "stages": ("propagate",), "label": "Track from ambiguity score", "min": 1, "max": 8},
    {"key": "propagate_preset", "stages": ("propagate",), "label": "Tracking schedule", "choices": tuple(PRESETS)},
    {"key": "track_refit", "stages": ("track",), "label": "Length refit", "choices": ("clipped-deviating", "clipped", "deviating", "all")},
)
LAST_OPENED_FILE = "last_opened.json"
KYMOGRAPH_RANGE = 15.0


# ---------------------------------------------------------------------------
# Models and the analysis job


def default_mask_source(libraries: library.Libraries, setup: library.Setup) -> str:
    """Where a setup's masks come from unless the user says: its body model when its mask default is a body-field network, else its segmenter."""

    ref = setup.defaults.get("mask")
    try:
        return "body_net" if ref and library.get_card(libraries, ref).kind == "body_net" else "segmenter"
    except LookupError:
        return "segmenter"


def resolve_models(
    libraries: library.Libraries, setup: library.Setup, requested: dict[str, Any] | None = None, mask_source: str = "segmenter",
) -> dict[str, dict[str, Any] | None]:
    """The model of each role, ``{ref, name, weights}`` or ``None``: a role named in ``requested`` takes that reference (``None``: no model), else the setup's default.

    With masks from the segmenter a mask model is required (a body model is
    not: without one the fit orients by the body's taper).  With masks from
    the body model (``mask_source="body_net"``) the body model is required,
    needs a mask output, and there is no mask model.  A model of the
    wrong kind or without the role's outputs is refused.
    """

    if mask_source not in pipeline.MASK_SOURCES:
        raise ValueError(f"unknown mask source {mask_source!r}; expected one of {pipeline.MASK_SOURCES}")
    requested = dict(requested or {})
    unknown = sorted(set(requested) - set(ROLE_KINDS))
    if unknown:
        raise ValueError(f"unknown model roles {unknown}; expected {sorted(ROLE_KINDS)}")
    if mask_source == "body_net":
        requested["mask"] = None
    models: dict[str, dict[str, Any] | None] = {}
    for role, kind in ROLE_KINDS.items():
        ref = requested[role] if role in requested else setup.defaults.get(role)
        if not ref:
            models[role] = None
            continue
        card = library.get_card(libraries, str(ref))
        if card.kind != kind:
            raise ValueError(f"{ref} is a {card.kind} model; the {role} model must be a {kind}")
        # Masks from the body model need its mask output too.
        wanted = ROLE_OUTPUTS[role] + (("mask",) if role == "body" and mask_source == "body_net" else ())
        missing = [output for output in wanted if output not in card.outputs]
        if missing:
            raise ValueError(f"{ref} cannot be the {role} model: it has no {', '.join(missing)} output")
        models[role] = {"ref": card.ref, "name": card.name, "weights": str(library.weights_path(libraries, card.ref))}
    if mask_source == "body_net" and models["body"] is None:
        raise ValueError("masks from the body model need a body model; choose one")
    if mask_source == "segmenter" and models["mask"] is None:
        raise ValueError(f"no mask model: setup {setup.ref} has no default one; choose one")
    return models


def analysis_params(app: AppState, setup: library.Setup, models: dict[str, dict[str, Any] | None], mask_source: str = "segmenter") -> dict[str, Any]:
    """The parameters every stage of an analysis reads (``run_all``'s ``'*'``): the mask source, the models' weights and the setup's video settings."""

    mask, body = models.get("mask"), models.get("body")
    return {
        "mask_source": mask_source,
        "checkpoint": None if mask is None else mask["weights"],
        "body_net": None if body is None else body["weights"],
        "dataset_root": str(app.config.dataset_root),
        "flat_field": bool(setup.video.get("flat_field", True)),
    }


def pipeline_settings(setup: library.Setup) -> list[dict[str, Any]]:
    """``PIPELINE_SETTINGS`` with each one's ``type`` (``bool``, ``int``, ``float`` or ``str``), ``default`` and ``help``."""

    out = []
    for spec in PIPELINE_SETTINGS:
        field = next(f for f in dataclass_fields(pipeline.STAGE_PARAMS[spec["stages"][0]]) if f.name == spec["key"])
        default = bool(setup.video.get("flat_field", True)) if spec["key"] == "flat_field" else field.default
        setting = {**spec, "stages": list(spec["stages"]), "type": type(default).__name__, "default": default, "help": field.metadata["help"]}
        if "choices" in spec:
            setting["choices"] = list(spec["choices"])
        out.append(setting)
    return out


def pipeline_values(setup: library.Setup, values: dict[str, Any] | None) -> dict[str, Any]:
    """The pipeline settings of an Analyse request, checked against ``pipeline_settings``; a setting left out keeps its default."""

    if values is not None and not isinstance(values, dict):
        raise ValueError("'pipeline' must be an object of pipeline settings")
    specs = {spec["key"]: spec for spec in pipeline_settings(setup)}
    unknown = sorted(set(values or {}) - set(specs))
    if unknown:
        raise ValueError(f"unknown pipeline settings {unknown}; expected some of {sorted(specs)}")
    out: dict[str, Any] = {}
    for key, value in (values or {}).items():
        spec, kind = specs[key], specs[key]["type"]
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        valid = {"bool": isinstance(value, bool), "int": number and float(value).is_integer(), "float": number, "str": isinstance(value, str)}[kind]
        if not valid:
            raise ValueError(f"pipeline setting {key!r} must be a {kind}, not {value!r}")
        value = {"int": int, "float": float}.get(kind, lambda v: v)(value)
        if "choices" in spec and value not in spec["choices"]:
            raise ValueError(f"pipeline setting {key!r} must be one of {spec['choices']}, not {value!r}")
        if "min" in spec and not spec["min"] <= value <= spec["max"]:
            raise ValueError(f"pipeline setting {key!r} must be between {spec['min']} and {spec['max']}, not {value!r}")
        out[key] = value
    return out


def workspace_setup(app: AppState, workspace: Workspace) -> library.Setup:
    """The setup of the workspace's recording; ``ValueError`` when it belongs to none."""

    setup_ref = library.setup_for_recording(app.libraries, workspace.recording)
    if setup_ref is None:
        raise ValueError(f"{workspace.recording} does not belong to a setup; add it to one first")
    return library.get_setup(app.libraries, setup_ref)


def analysis_command(workspace_path: Path, stages: list[str], params: dict[str, Any]) -> list[str]:
    return pipeline.stages_command(workspace_path, stages, params)


def _stages(payload: dict[str, Any]) -> list[str]:
    stages = payload.get("stages") or list(pipeline.DEFAULT_STAGES)
    if not isinstance(stages, list) or not all(isinstance(stage, str) for stage in stages):
        raise ValueError("'stages' must be a list of stage names")
    unknown = [stage for stage in stages if stage not in pipeline.STAGES or stage == "export"]
    if unknown:
        raise ValueError(f"cannot analyse with stages {unknown}; choose from {[s for s in pipeline.STAGES if s != 'export']}")
    return [stage for stage in pipeline.STAGES if stage in stages]


def active_jobs(app: AppState, name: str) -> list[JobRecord]:
    return [record for record in app.runner.list() if record.spec.workspace == name and record.state not in FINISHED_STATES]


def open_recording(app: AppState, payload: dict[str, Any]) -> str:
    """The workspace of the recording at ``payload["path"]``, made when it has none (``first``, ``last`` and ``step`` are a developer's range for a new one)."""

    recording = app.recording_path(str(payload.get("path") or ""))
    name = app.workspace_of_recording(recording)
    if name is None:
        name = app.create_workspace({key: payload.get(key) for key in ("first", "last", "step")} | {"recording": str(recording)}).name
    return name


def analysis_job(app: AppState, name: str, payload: dict[str, Any]) -> tuple[JobSpec, list[str]]:
    """Analyse workspace ``name``: its settings and the job; returns ``(spec, argv)``.

    ``models`` (``{role: ref}``) replaces the setup's defaults,
    ``mask_source`` (``segmenter``, the default, or ``body_net``) says where
    the masks come from and ``pipeline`` (``{key: value}``) changes the
    pipeline settings; ``stages`` are the developer's.  A workspace with a
    queued or running job is busy.
    """

    libraries = app.libraries
    if not app.has_workspace(name):
        raise NotFound(f"unknown workspace {name!r}")
    workspace = Workspace.open(app.workspace_path(name))
    recording = workspace.recording
    setup = workspace_setup(app, workspace)
    setup_ref = setup.ref
    mask_source = str(payload.get("mask_source") or default_mask_source(libraries, setup))
    models = resolve_models(libraries, setup, payload.get("models"), mask_source)
    values = pipeline_values(setup, payload.get("pipeline"))
    stages = _stages(payload)
    busy = active_jobs(app, name)
    if busy:
        raise pipeline.WorkspaceBusy(f"{busy[0].spec.label or busy[0].spec.kind} is {busy[0].state} on {name}; wait for it or cancel it")
    settings = workspace.info.settings
    settings.update({
        "setup": setup_ref,
        "models": {role: None if model is None else model["ref"] for role, model in models.items()},
        "mask_source": mask_source,
        "checkpoint": None if models["mask"] is None else models["mask"]["weights"],
        "body_net": None if models["body"] is None else models["body"]["weights"],
        "pipeline": values,
    })
    with pipeline.workspace_lock(workspace, timeout=0):
        workspace.save_info()
    app.view(name).invalidate()
    params = {**analysis_params(app, setup, models, mask_source), **values}
    spec = JobSpec(
        kind=ANALYSE_JOB_KIND,
        params={"stages": stages, "params": params, "setup": setup_ref, "models": settings["models"], "mask_source": mask_source, "pipeline": values},
        workspace=name, frames=[int(v) for v in workspace.info.frames], label=f"Analyse {recording.stem}",
    )
    return spec, analysis_command(workspace.path, stages, params)


# ---------------------------------------------------------------------------
# Status


def _model_names(libraries: library.Libraries, refs: dict[str, Any]) -> dict[str, dict[str, Any] | None]:
    out: dict[str, dict[str, Any] | None] = {}
    for role in ROLE_KINDS:
        ref = refs.get(role)
        if not ref:
            out[role] = None
            continue
        try:
            card = library.get_card(libraries, str(ref))
            out[role] = {"ref": card.ref, "name": card.name, "outputs": list(card.outputs)}
        except (LookupError, ValueError):
            out[role] = {"ref": str(ref), "name": str(ref), "outputs": [], "missing": True}
    return out


def workspace_body_net(workspace: Any) -> Path | None:
    """The body-field network the workspace was analysed with (its settings, else its fit's), when the file exists."""

    path = workspace.info.settings.get("body_net") or (pipeline.read_summary(workspace).get("fit_params") or {}).get("body_net")
    return Path(path) if path and Path(path).is_file() else None


def workspace_mask_net(workspace: Any) -> Path | None:
    """The segmenter the workspace was analysed with (its settings, else its summary's), when the file exists; ``None`` when its masks came from the body model."""

    summary = pipeline.read_summary(workspace)
    if (workspace.info.settings.get("mask_source") or summary.get("mask_source")) == "body_net":
        return None
    path = workspace.info.settings.get("checkpoint") or (summary.get("checkpoint") or {}).get("path")
    return Path(path) if path and Path(path).is_file() else None


def stage_progress(record: JobRecord) -> list[dict[str, Any]]:
    """Each stage of an analysis job, ``{stage, progress, state}`` (``waiting``, ``running``, ``done``, or the job's ``failed`` or ``cancelled``), from its overall progress.

    ``pipeline.run_all`` reports ``(k + fraction) / n`` with the running
    stage named before the message's colon, so the stages before it are
    done and its own fraction is what is left over.
    """

    stages = list(record.spec.params.get("stages") or [])
    if not stages:
        return []
    n, current = len(stages), record.message.split(":")[0]
    if record.state == "done":
        return [{"stage": stage, "progress": 1.0, "state": "done"} for stage in stages]
    k = stages.index(current) if current in stages else None
    out = []
    for i, stage in enumerate(stages):
        if k is None:
            fraction = min(max(record.progress * n - i, 0.0), 1.0)
            state = "done" if fraction >= 1.0 else "waiting"
        elif i != k:
            fraction, state = (1.0, "done") if i < k else (0.0, "waiting")
        else:
            fraction = min(max(record.progress * n - k, 0.0), 1.0)
            state = record.state if record.state in FINISHED_STATES else "running"
        out.append({"stage": stage, "progress": fraction, "state": state})
    return out


def _job(record: JobRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    job = {
        "id": record.id, "kind": record.spec.kind, "label": record.spec.label, "state": record.state, "progress": record.progress,
        "message": record.message, "error": record.error, "run_on": record.spec.run_on, "slurm_state": record.slurm_state,
        "finished_at": record.finished_at,
    }
    if record.spec.kind == ANALYSE_JOB_KIND:
        job["stages"] = stage_progress(record)
    return job


def last_opened(app: AppState) -> dict[str, str]:
    path = app.config.workspaces_root / LAST_OPENED_FILE
    try:
        return dict(json.loads(path.read_text()))
    except (OSError, ValueError):
        return {}


def mark_opened(app: AppState, name: str) -> str:
    app.workspace_path(name)  # validates the name
    if not app.has_workspace(name):
        raise NotFound(f"unknown workspace {name!r}")
    opened = last_opened(app)
    opened[name] = utc_now()
    _write_json_atomic(app.config.workspaces_root / LAST_OPENED_FILE, opened)
    return opened[name]


def status(app: AppState, name: str, jobs: list[JobRecord] | None = None, opened: dict[str, str] | None = None) -> dict[str, Any]:
    """What the Recordings screen and the Workspace header show of workspace ``name`` (see the module docstring)."""

    if not app.has_workspace(name):
        raise NotFound(f"unknown workspace {name!r}")
    workspace = Workspace.open(app.workspace_path(name))
    settings = workspace.info.settings
    libraries = app.libraries
    setup_ref = settings.get("setup") or library.setup_for_recording(libraries, workspace.recording)
    setup, mask_source = None, settings.get("mask_source", "segmenter")
    if setup_ref:
        try:
            found = library.get_setup(libraries, setup_ref)
            setup = {"ref": found.ref, "name": found.name, "pixel_size_um": found.pixel_size_um, "fps": found.fps}
            mask_source = settings.get("mask_source") or default_mask_source(libraries, found)
        except (LookupError, ValueError):
            setup = {"ref": setup_ref, "name": setup_ref, "pixel_size_um": None, "fps": None, "missing": True}
    records = [r for r in (app.runner.list() if jobs is None else jobs) if r.spec.workspace == name]
    active = [r for r in records if r.state not in FINISHED_STATES]
    analysis = next((r for r in records if r.spec.kind == ANALYSE_JOB_KIND), None)
    analysed = bool(pipeline.read_summary(workspace).get("fit_config"))
    issues = cached_issue_summary(app.view(name)) if analysed and not active else None
    exports = list_exports(workspace)
    if analysis is not None and not analysis.finished:
        state = "queued" if analysis.state == "queued" or analysis.slurm_state == "PENDING" else "analysing"
    elif not analysed:
        state = analysis.state if analysis is not None and analysis.state in ("failed", "cancelled") else "not_analysed"
    elif issues is None:
        state = "analysed"
    elif issues["unreviewed"]:
        state = "issues"
    else:
        state = "exported" if exports else "reviewed"
    return {
        "name": name, "recording": str(workspace.recording), "recording_id": workspace.recording.stem,
        "frames": [int(v) for v in workspace.info.frames], "step": int(workspace.info.step), "frame_count": workspace.n,
        "image_shape": workspace.info.image_shape, "setup": setup,
        "models": _model_names(libraries, settings.get("models") or {}),
        "mask_source": mask_source, "pipeline": settings.get("pipeline") or {},
        "has_body_model": workspace_body_net(workspace) is not None,
        "has_mask_model": workspace_mask_net(workspace) is not None,
        "state": state, "analysed": analysed, "issues": issues,
        "analysis": _job(analysis), "active_jobs": [_job(r) for r in active],
        "exports": len(exports), "last_export": exports[0]["created_at"] if exports else None,
        "last_opened": (last_opened(app) if opened is None else opened).get(name),
    }


def home(app: AppState, setup_ref: str | None) -> dict[str, Any]:
    """The Recordings screen: the setups, the chosen setup's folders, and its recordings with their workspace's status (newest workspace per recording)."""

    libraries = app.libraries
    setups = library.list_setups(libraries)
    payload: dict[str, Any] = {"setups": [s.to_dict() for s in setups], "setup": None, "folders": [], "recordings": []}
    if not setups:
        return payload
    setup = library.get_setup(libraries, setup_ref) if setup_ref else setups[0]
    payload["setup"] = setup.to_dict()
    payload["folders"] = [path for path, entry in sorted(library.recording_folders(libraries).items()) if entry["setup"] == setup.ref]
    # The files are listed at once; the ones not probed yet are probed in the background while the screen polls.
    infos = library_recordings(app, setup, probe=False)
    app.prober.submit((Path(info.path), info.dataset) for info in infos if info.pending)
    workspaces: dict[str, str] = {}
    for info in reversed(list_workspaces(app.config.workspaces_root)):  # oldest first, so the newest wins
        workspaces[str(Path(info.recording).resolve())] = info.name
    jobs, opened = app.runner.list(), last_opened(app)
    rows = []
    for info in infos:
        name = workspaces.get(str(Path(info.path).resolve()))
        rows.append({
            "id": library.recording_id(info.path), "path": info.path, "frames": info.frames, "readable": info.readable, "error": info.error,
            "pending": info.pending,
            "workspace": name, "status": None if name is None else status(app, name, jobs, opened),
        })
    payload["recordings"] = rows
    return payload


def library_recordings(app: AppState, setup: library.Setup, probe: bool = True) -> list[RecordingInfo]:
    """The setup's recordings (in its folders or registered to it, not registered to another setup).

    With ``probe=False`` files not probed yet are listed as pending
    (:func:`recordings.list_recordings`).
    """

    libraries = app.libraries
    infos = list_recordings(
        library.recording_sources(libraries, setup.ref), cache=app.config.recordings_cache, dataset=str(setup.video["dataset_path"]), probe=probe,
    )
    registered = library.setups.registered_recordings(libraries)
    return [info for info in infos if registered.get(info.path, {}).get("setup") in (None, setup.ref)]


# ---------------------------------------------------------------------------
# The kymograph


def curvature_kymograph(centerline_xy: np.ndarray, body_length_px: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """Signed curvature times body length, ``(points, rows)``, NaN where a row has no pose."""

    curves = np.asarray(centerline_xy, dtype=np.float64)
    d1 = np.gradient(curves, axis=1)
    d2 = np.gradient(d1, axis=1)
    speed = np.linalg.norm(d1, axis=2)
    cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = cross / np.where(speed > 0, speed**3, np.nan) * np.asarray(body_length_px, dtype=np.float64)[:, None]
    curvature[~np.asarray(fitted, dtype=bool)] = np.nan
    return curvature.T


def encode_kymograph(values: np.ndarray, limit: float = KYMOGRAPH_RANGE) -> np.ndarray:
    scaled = 1 + np.round(254 * (np.clip(np.nan_to_num(values), -limit, limit) + limit) / (2 * limit))
    return np.where(np.isfinite(values), scaled, 0).astype(np.uint8)


def off_screen_kymograph(centerline_xy: np.ndarray, fitted: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """``(points, rows)``, true where a fitted row's midline point lies outside the ``(H, W)`` frame."""

    curves = np.asarray(centerline_xy, dtype=np.float64)
    height, width = image_shape
    x, y = curves[..., 0], curves[..., 1]
    outside = (x < 0) | (x >= width) | (y < 0) | (y >= height)
    outside[~np.asarray(fitted, dtype=bool)] = False
    return outside.T


def kymograph(app: AppState, name: str) -> dict[str, Any]:
    view = app.view(name)
    run, image_shape = view.run, view.workspace.image_shape
    arrays = run.arrays
    values = curvature_kymograph(arrays["centerline_xy"], arrays["body_length_px"], arrays["fitted"])
    off_screen = None if image_shape is None else off_screen_kymograph(arrays["centerline_xy"], arrays["fitted"], image_shape)
    return {
        "rows": int(values.shape[1]), "points": int(values.shape[0]), "range": KYMOGRAPH_RANGE,
        "frames": [int(run.frame_index[0]), int(run.frame_index[-1])], "step": int(run.frame_index[1] - run.frame_index[0]) if len(run.frame_index) > 1 else 1,
        "image": data_url(encode_kymograph(values)),
        "off_screen": None if off_screen is None else data_url(off_screen.astype(np.uint8) * 255),
    }
