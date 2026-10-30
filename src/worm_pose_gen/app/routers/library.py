"""The library for the Workspace, Labeling and Training pages: setups, recordings, labels, datasets, benchmarks, pseudo-label sets, models.

Everything is read from both libraries (:mod:`worm_pose_gen.library`) and
named by reference (``lab:<id>``, ``mine:<id>``); references go in paths as
they are (``/api/library/datasets/lab:nir-labels``).  A setup's labels are
its collection (``/api/library/setups/<ref>/labels``); a dataset chooses the
split of each of the collection's recordings.  The writes are the ones the
pages need: create a personal setup or dataset, set a dataset's splits,
add or remove a recording folder, register a recording to a setup, save a label revision, freeze a benchmark,
and set a default model.  A write that names a lab item is refused with 403, since
the app never writes the lab library.

Images go to the browser as PNG data URLs: a mask as 0 background, 255
worm, 128 excluded from the loss (:func:`app.images.mask_to_png_values`), a
drawn overlap as 0 or 255, and the A-P field as 0 (undefined) or
``1 + round(254 * ap)``.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import binascii
from typing import Any, Iterator

import numpy as np
from fastapi import APIRouter, Body, Depends, HTTPException

from . import get_state
from ... import library
from ...model_eval import DEFAULT_MIN_FIT_IOU, OVERLAP_STATES, overlap_state
from ..images import data_url, decode_mask_data_url, mask_to_png_values
from ...recordings import list_recordings
from ..state import AppState, NotFound, _integer

router = APIRouter(prefix="/api/library")

SORTS = ("", "fit_iou")


@contextmanager
def _found() -> Iterator[None]:
    """A missing library item is a 404 and a write to the lab library a 403."""

    try:
        yield
    except NotFound:
        raise
    except PermissionError as error:
        raise HTTPException(403, str(error)) from error
    except LookupError as error:
        if isinstance(error, KeyError):
            raise
        raise NotFound(str(error)) from error


def _libraries(app: AppState) -> library.Libraries:
    return app.libraries


@router.get("")
def roots(app: AppState = Depends(get_state)) -> dict[str, Any]:
    libraries = _libraries(app)
    return {"lab": None if libraries.lab is None else str(libraries.lab), "lab_available": "lab" in libraries.scopes(),
            "personal": str(libraries.personal)}


# --------------------------------------------------------------------------- setups


@router.get("/setups")
def setups(app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"setups": [setup.to_dict() for setup in library.list_setups(_libraries(app))]}


@router.post("/setups")
def create_setup(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    fields = {key: payload[key] for key in ("name", "description", "video", "pixel_size_um", "fps") if key in payload}
    if not fields.get("name"):
        raise ValueError("'name' is required")
    with _found():
        return library.create_setup(_libraries(app), str(payload.get("id") or ""), **fields).to_dict()


@router.get("/setups/{ref}")
def setup(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {**library.get_setup(_libraries(app), ref).to_dict(), "defaults_log": library.defaults_log(_libraries(app), ref)}


@router.post("/setups/{ref}/defaults")
def set_default(ref: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        updated = library.set_default(
            _libraries(app), ref, str(payload.get("role") or ""), str(payload.get("model") or ""), reason=str(payload.get("reason") or ""),
        )
        return {**updated.to_dict(), "defaults_log": library.defaults_log(_libraries(app), ref)}


@router.get("/setups/{ref}/recordings")
def setup_recordings(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The recordings in the setup's folders and those registered to it, with frame counts and readability."""

    libraries = _libraries(app)
    with _found():
        setup = library.get_setup(libraries, ref)
        sources = library.recording_sources(libraries, ref)
    infos = list_recordings(sources, cache=app.config.recordings_cache, dataset=str(setup.video["dataset_path"]))
    registered = library.setups.registered_recordings(libraries)
    rows = []
    for info in infos:
        owner = registered.get(info.path, {}).get("setup")
        if owner is None or owner == ref:  # a file in the folders registered to another setup belongs there
            rows.append({**info.to_dict(), "id": library.recording_id(info.path), "registered": info.path in registered})
    return {"setup": ref, "recordings": rows}


@router.get("/recording-setup")
def recording_setup(path: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    return {"path": path, "id": library.recording_id(path), "setup": library.setup_for_recording(_libraries(app), path)}


@router.post("/recordings")
def register_recording(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return library.register_recording(_libraries(app), str(payload.get("path") or ""), str(payload.get("setup") or ""))


@router.get("/recording-folders")
def recording_folders(app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The folders searched for recordings, each with its setup."""

    return {"folders": [{"path": path, **entry} for path, entry in sorted(library.recording_folders(_libraries(app)).items())]}


@router.post("/recording-folders")
def add_recording_folder(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        try:
            return library.add_recording_folder(_libraries(app), str(payload.get("path") or ""), str(payload.get("setup") or ""))
        except NotADirectoryError as error:
            raise ValueError(str(error)) from error


@router.post("/recording-folders/remove")
def remove_recording_folder(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        library.remove_recording_folder(_libraries(app), str(payload.get("path") or ""))
    return recording_folders(app)


# --------------------------------------------------------------------------- datasets


def _dataset(app: AppState, ref: str) -> library.Dataset:
    with _found():
        return library.Dataset(_libraries(app), ref)


@router.get("/datasets")
def datasets(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {"datasets": [d.summary() for d in library.list_datasets(_libraries(app), setup or None)]}


@router.post("/datasets")
def create_dataset(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """A new personal dataset of a setup: every recording not included, unless ``splits`` (recording -> split) says otherwise."""

    splits = payload.get("splits") or {}
    if not isinstance(splits, dict):
        raise ValueError("'splits' must map recordings to a split")
    with _found():
        dataset = library.create_dataset(
            _libraries(app), str(payload.get("id") or ""), setup=str(payload.get("setup") or ""),
            name=str(payload.get("name") or ""), description=str(payload.get("description") or ""),
            splits={str(k): str(v) for k, v in splits.items()},
        )
        return dataset.summary()


@router.get("/datasets/{ref}")
def dataset(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return _dataset(app, ref).summary()


@router.post("/datasets/{ref}/splits")
def set_splits(ref: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """``{"splits": {recording: "train" | "val" | "test" | null}}``: put recordings in a split or (null) out of the dataset."""

    changes = payload.get("splits")
    if not isinstance(changes, dict):
        raise ValueError("'splits' must map recordings to a split or null")
    with _found():
        dataset = _dataset(app, ref)
        dataset.set_splits({str(k): None if v is None else str(v) for k, v in changes.items()})
        return dataset.summary()


# --------------------------------------------------------------------------- labels


def _collection(app: AppState, setup: str) -> library.Collection:
    with _found():
        return library.Collection(_libraries(app), setup)


def label_row(app: AppState, record: library.LabelRecord, builder: str | None = library.SETUP_DEFAULT) -> dict[str, Any]:
    """A label as listings show it, with the state of its body targets (built by ``builder``, by default the setup's)
    and what its overlap trains (:func:`model_eval.overlap_state`; ``tube_overlap_px``: where the fitted tube crosses)."""

    meta = library.cached_meta(_libraries(app), record, builder)
    return {**record.to_dict(), "targets": "built" if meta is not None else "missing",
            "fit_iou": None if meta is None else meta.get("fit_iou"),
            "self_contact": None if meta is None else meta.get("self_contact"),
            "overlap": overlap_state(record, meta, DEFAULT_MIN_FIT_IOU),
            "tube_overlap_px": None if meta is None else meta.get("overlap_px")}


@router.get("/setups/{ref}/collection")
def collection(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The setup's label collection: each recording with a label and its number of labeled frames."""

    with _found():
        recordings = _collection(app, ref).recordings()
    return {"setup": ref, "labels": sum(recordings.values()), "recordings": [{"recording": k, "labels": n} for k, n in recordings.items()]}


@router.get("/setups/{ref}/labels")
def setup_labels(ref: str, recording: str = "", dataset: str = "", split: str = "", status: str = "", contact: str = "",
                 overlap: str = "", sort: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The setup's labels, filtered; with ``dataset`` each carries its recording's split there, which ``split`` filters on
    (``none`` for the recordings it does not include).  ``overlap`` filters on the overlap state
    (``review``: the tube crosses and nobody has drawn the overlap yet).  ``sort=fit_iou`` puts the lowest body fit IoU first."""

    if status and status not in library.STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {library.STATUSES}")
    if overlap and overlap not in OVERLAP_STATES:
        raise ValueError(f"unknown overlap state {overlap!r}; expected one of {OVERLAP_STATES}")
    if contact not in ("", "yes", "no"):
        raise ValueError("contact must be yes or no")
    if sort not in SORTS:
        raise ValueError(f"unknown sort {sort!r}; expected one of {SORTS}")
    if split and split != "none":
        library.datasets.check_split(split)
    if split and not dataset:
        raise ValueError("filtering by split needs a dataset")
    with _found():
        records = _collection(app, ref).labels()
        if dataset:
            found = _dataset(app, dataset)
            if found.setup != ref:
                raise ValueError(f"{dataset} is a dataset of {found.setup}, not of {ref}")
            splits = found.splits()
            records = [replace(r, split=splits.get(r.recording)) for r in records]
        builder = library.target_builder(_libraries(app), ref)
    records = [
        r for r in records
        if (not recording or r.recording == recording) and (not status or r.status == status)
        and (not split or r.split == (None if split == "none" else split))
    ]
    rows = [label_row(app, record, builder) for record in records]
    if contact:
        rows = [row for row in rows if row["self_contact"] is (contact == "yes")]
    if overlap:
        rows = [row for row in rows if row["overlap"] == overlap]
    if sort == "fit_iou":
        rows.sort(key=lambda row: (row["fit_iou"] is None, row["fit_iou"] if row["fit_iou"] is not None else 0.0))
    return {"setup": ref, "dataset": dataset or None, "total": len(rows), "labels": rows}


def _point(xy: np.ndarray) -> list[float] | None:
    return xy.tolist() if np.all(np.isfinite(xy)) else None


def targets_layers(app: AppState, record: library.LabelRecord) -> dict[str, Any] | None:
    """A label's built targets as drawable layers (``None`` until they are built with the setup's builder)."""

    built = library.load_targets(_libraries(app), record)
    if built is None:
        return None
    meta, arrays = built
    result: dict[str, Any] = {"meta": meta}
    if meta.get("has_body"):
        ap = arrays["ap"].astype(np.float32)
        encoded = np.where(np.isfinite(ap), 1 + np.round(254 * np.clip(np.nan_to_num(ap), 0, 1)), 0).astype(np.uint8)
        result.update(
            ap=data_url(encoded), overlap=data_url(arrays["overlap"].astype(np.uint8) * 255),
            centerline_xy=arrays["centerline_xy"].tolist(), width_profile=arrays["width_profile"].tolist(),
            head_xy=_point(arrays["head_xy"]), tail_xy=_point(arrays["tail_xy"]), diameter_px=float(arrays["diameter_px"]),
        )
    return result


def _label(app: AppState, ref: str, recording: str, frame: int, revision: int | None, scope: str) -> library.LabelRecord:
    if revision is not None and not scope:
        raise ValueError("a revision needs its scope (lab or mine)")
    with _found():
        return _collection(app, ref).get(recording, frame, revision, scope or None)


@router.get("/setups/{ref}/labels/{recording}/{frame}")
def label(ref: str, recording: str, frame: int, revision: int | None = None, scope: str = "",
          app: AppState = Depends(get_state)) -> dict[str, Any]:
    """A frame's current label (or one revision), with its images, human fields, built targets and every revision."""

    record = _label(app, ref, recording, frame, revision, scope)
    loaded = record.load()
    return {
        "label": label_row(app, record), "revisions": [r.to_dict() for r in _collection(app, ref).revisions(recording, frame)],
        "width": record.width, "height": record.height, "meta": loaded.meta,
        "image": data_url(loaded.image), "image_raw": data_url(loaded.image_raw), "mask": data_url(mask_to_png_values(loaded.mask)),
        "orientation": record.orientation, "mask_only": record.mask_only,
        "head_xy": None if loaded.head_xy is None else loaded.head_xy.tolist(),
        "trace_xy": None if loaded.trace_xy is None else loaded.trace_xy.tolist(),
        "overlap": None if loaded.overlap is None else data_url(loaded.overlap * np.uint8(255)),
        "max_lag": loaded.max_lag, "context_valid": loaded.context_valid.tolist(),
        "nose_xy": [xy.tolist() if ok else None for xy, ok in zip(loaded.nose_xy, loaded.nose_valid)],
        "targets": targets_layers(app, record),
    }


@router.get("/setups/{ref}/labels/{recording}/{frame}/context")
def label_context(ref: str, recording: str, frame: int, revision: int | None = None, scope: str = "",
                  app: AppState = Depends(get_state)) -> dict[str, Any]:
    loaded = _label(app, ref, recording, frame, revision, scope).load()
    return {"max_lag": loaded.max_lag, "valid": loaded.context_valid.tolist(), "frames": [data_url(f) for f in loaded.context]}


@router.post("/setups/{ref}/labels")
def save_label(ref: str, payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    """Save a new revision of a frame's label into the setup's collection (the personal library's part).

    With ``path`` the frame, its context and its nose landmarks are read
    from that recording (which must belong to the setup); without it they
    are taken from the frame's current label, so an edit never needs the
    recording.  The body
    fields are ``orientation`` (``auto`` or ``manual`` with ``head_xy``),
    ``trace_xy`` (head first) and ``mask_only``; ``overlap`` (a PNG like the
    mask, 255 where the body lies over itself) when it was drawn or
    reviewed; ``origin`` is ``spread`` or
    ``fix``; ``expected_revision`` is the personal revision the editor
    started from (0 for none).
    """

    libraries = _libraries(app)
    frame = _integer(payload, "frame")
    origin = str(payload.get("origin") or "")
    if origin not in ("spread", "fix"):
        raise ValueError("'origin' must be spread or fix")
    with _found():
        collection = _collection(app, ref)
        if payload.get("path"):
            path = str(payload["path"])
            owner = library.setup_for_recording(libraries, path)
            if owner != collection.setup:
                raise ValueError(f"{path} belongs to {owner or 'no setup'}, not to {collection.setup}; register it first")
            from ...library.capture import read_label_inputs

            setup = library.get_setup(libraries, collection.setup)
            inputs = read_label_inputs(path, frame, video=setup.video, flat_field_cache=app.config.dataset_root / "flat_fields")
            recording = library.recording_id(path)
        else:
            recording = str(payload.get("recording") or "")
            existing = collection.get(recording, frame).load()
            inputs = {
                "image": existing.image, "image_raw": existing.image_raw, "context": existing.context,
                "context_valid": existing.context_valid, "nose_xy": existing.nose_xy, "nose_valid": existing.nose_valid,
                "source_path": existing.record.source_path, "dataset_path": existing.record.dataset_path,
            }
        try:
            mask = decode_mask_data_url(str(payload.get("mask") or ""), inputs["image"].shape)
            overlap = None if payload.get("overlap") is None else decode_mask_data_url(str(payload["overlap"]), mask.shape) == 1
        except (OSError, binascii.Error) as error:
            raise ValueError("mask and overlap must contain readable base64 PNG images") from error
        expected = payload.get("expected_revision")
        record = collection.save(
            recording=recording, frame=frame, mask=mask, origin=origin,
            orientation=str(payload.get("orientation") or "auto"), head_xy=payload.get("head_xy"),
            trace_xy=payload.get("trace_xy"), mask_only=bool(payload.get("mask_only")), overlap=overlap,
            expected_revision=None if expected is None else int(expected), **inputs,
        )
    return {"label": label_row(app, record)}


# --------------------------------------------------------------------------- benchmarks and models


@router.get("/pseudo-labels")
def pseudo_labels(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    """The pseudo-label sets a body-field run can mix in (:mod:`library.pseudo_sets`)."""

    with _found():
        return {"pseudo_labels": [p.summary() for p in library.list_pseudo_labels(_libraries(app), setup or None)]}


@router.get("/benchmarks")
def benchmarks(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {"benchmarks": [b.summary() for b in library.list_benchmarks(_libraries(app), setup or None)]}


@router.post("/benchmarks")
def freeze_benchmark(payload: dict[str, Any] = Body(...), app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        frozen = library.freeze_benchmark(
            _libraries(app), str(payload.get("dataset") or ""), payload.get("id") or None,
            description=str(payload.get("description") or ""),
        )
        return frozen.summary()


@router.get("/benchmarks/{ref}")
def benchmark(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        found = library.get_benchmark(_libraries(app), ref)
    return {**found.summary(), "entries": list(found.entries)}


def _model(app: AppState, card: library.ModelCard) -> dict[str, Any]:
    return {**card.to_dict(), "evaluations": library.evaluations(_libraries(app), card.ref)}


@router.get("/models")
def models(setup: str = "", app: AppState = Depends(get_state)) -> dict[str, Any]:
    with _found():
        return {"models": [_model(app, card) for card in library.list_models(_libraries(app), setup or None)]}


@router.get("/models/{ref}")
def model(ref: str, app: AppState = Depends(get_state)) -> dict[str, Any]:
    libraries = _libraries(app)
    with _found():
        card = library.get_card(libraries, ref)
        training = library.training_dir(libraries, ref)
    return {**_model(app, card), "training_files": sorted(p.name for p in training.iterdir()) if training.is_dir() else []}
