"""Human review is separate from automatic quality flags and bound to inputs.

A reviewed row is stored with a fingerprint of its content and its
neighbours' (``row_fingerprints``), so any change near it (a fix, a mask
edit) drops the review.  ``issues`` reads it for the Issues panel
(``worm_pose_gen.fixes.issue_report``: stretches with plain-language
reasons, each unreviewed, reviewed or fixed), and ``review_issue`` (Looks
OK) adds an issue's rows to it.

Every computation of the issues also writes their counts with the review
token to ``issues_summary.json`` (``ISSUES_SUMMARY``), so the Recordings
screen can show "N issues to review" without loading the arrays of every
workspace; ``cached_issue_summary`` returns them while the token still
holds.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict
from weakref import WeakKeyDictionary

import numpy as np
from typing import Any

from fastapi import HTTPException

from .workspace_view import WorkspaceView, STAMPED_FILES
from .. import fixes
from ..pipeline import placed_rows, workspace_lock
from ..workspace import Workspace, _write_json_atomic, utc_now

ISSUES_SUMMARY = "issues_summary.json"
_REVIEW_LOCK = threading.RLock()
_FINGERPRINT_CACHE: WeakKeyDictionary = WeakKeyDictionary()


def revision(workspace: Workspace) -> str:
    paths = [workspace.path / name for name in STAMPED_FILES]
    paths.append(workspace.recording)
    for directory in ("masks", "overrides/masks"):
        paths.extend(sorted((workspace.path / directory).rglob("*")))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}".encode())
    return digest.hexdigest()


def row_fingerprints(view: WorkspaceView, rows: list[int]) -> dict[str, str]:
    """Content identities for reviewed rows plus their immediate quality context.

    File revision remains the request concurrency token. These durable hashes
    deliberately exclude edit-log timestamps and unrelated rows, so reviewing
    one segment survives correcting another. Neighbours cover pairwise motion,
    length and orientation signals even before derived flags are recomputed.
    """
    token = revision(view.workspace)
    cached_token, cached = _FINGERPRINT_CACHE.get(view, (None, {}))
    if cached_token != token:
        cached = {}
    missing = [row for row in rows if str(row) not in cached]
    if not missing:
        return {str(row): cached[str(row)] for row in rows}
    run, workspace = view.run, view.workspace
    recording = workspace.recording
    stat = recording.stat() if recording.exists() else None
    context = dict(
        recording=str(recording),
        recording_revision=None if stat is None else [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns],
        settings=workspace.info.settings,
        frames=workspace.info.frames, step=workspace.info.step,
        config=run.summary.get("fit_config"), checkpoint=run.summary.get("checkpoint"),
        prior=run.prior, thresholds=asdict(run.thresholds),
        cleanup=run.cleanup, threshold=run.threshold,
        image_shape=run.summary.get("image_shape"),
    )
    base = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode())
    arrays = {**run.arrays, **{"provenance:" + k: v for k, v in workspace.load_provenance().items()}}
    masks: dict[int, str] = {}
    result = {}
    for row in missing:
        digest = base.copy()
        first, last = max(0, row - 1), min(workspace.n, row + 2)
        digest.update(f"{row}:{first}:{last}".encode())
        for key, values in sorted(arrays.items()):
            values = np.asarray(values)
            value = values[first:last] if values.ndim and values.shape[0] == workspace.n else values
            digest.update(f"{key}:{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
        for neighbour in range(first, last):
            if neighbour not in masks:
                masks[neighbour] = workspace.mask_revision(neighbour)
            digest.update(masks[neighbour].encode())
        result[str(row)] = digest.hexdigest()
    cached.update(result)
    _FINGERPRINT_CACHE[view] = (token, cached)
    return {str(row): cached[str(row)] for row in rows}


def _reviewed(view: WorkspaceView) -> set[int]:
    """The rows whose review still holds: stored with a fingerprint that matches the row now."""

    path = view.workspace.path / "human_review.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved_rows = [r for r in saved.get("rows", []) if isinstance(r, int) and 0 <= r < view.workspace.n]
    fingerprints = row_fingerprints(view, saved_rows)
    # Legacy global-token records cannot prove an individual row unchanged.
    return {r for r in saved_rows if saved.get("row_fingerprints", {}).get(str(r)) == fingerprints[str(r)]}


def issues(view: WorkspaceView) -> dict[str, Any]:
    """The Issues panel: ``fixes.issue_report`` over the current arrays and review, with the review token ``revision``."""

    with _REVIEW_LOCK, workspace_lock(view.workspace, timeout=0):
        return _issues(view)


def _issues(view: WorkspaceView) -> dict[str, Any]:
    run = view.run
    token = revision(view.workspace)
    n = view.workspace.n
    reviewed = np.zeros(n, dtype=bool)
    reviewed[sorted(_reviewed(view))] = True
    provenance = view.workspace.load_provenance()
    placed = placed_rows(run.arrays, provenance["algorithm"], provenance["job"])
    report = fixes.issue_report(run.arrays, placed, reviewed)
    path = view.workspace.path / ISSUES_SUMMARY
    cached = json.loads(path.read_text()) if path.exists() else None
    if cached != {"revision": token, "summary": report["summary"]}:
        _write_json_atomic(path, {"revision": token, "summary": report["summary"]})
    return {"revision": token, **report, "min_issue_frames": fixes.MIN_ISSUE_FRAMES}


def cached_issue_summary(workspace: Workspace) -> dict[str, Any] | None:
    """The issue counts last computed for the workspace (``fixes.issue_report``'s summary), or ``None`` when anything changed since."""

    path = workspace.path / ISSUES_SUMMARY
    try:
        cached = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return cached.get("summary") if cached.get("revision") == revision(workspace) else None


def review_issue(view: WorkspaceView, first_frame: int, last_frame: int, expected_revision: str) -> dict[str, Any]:
    """Looks OK: mark the frames ``first_frame..last_frame`` reviewed (an issue's frames) and answer with the issues."""

    workspace = view.workspace
    first, last = workspace.row_of(int(first_frame)), workspace.row_of(int(last_frame))
    with _REVIEW_LOCK, workspace_lock(workspace, timeout=0):
        _mark(view, first, last, expected_revision)
        return _issues(view)


def _mark(view: WorkspaceView, first: int, last: int, expected_revision: str) -> None:
    """Add rows ``first..last`` to the stored review (caller holds the locks); 409 when the workspace changed since ``expected_revision``."""

    token = revision(view.workspace)
    if expected_revision != token:
        raise HTTPException(409, "Workspace changed. Refresh the issues before marking them reviewed.")
    if first < 0 or last < first or last >= view.workspace.n:
        raise ValueError("Review bounds must be valid inclusive workspace rows.")
    rows = sorted(_reviewed(view).union(range(first, last + 1)))
    _write_json_atomic(view.workspace.path / "human_review.json", dict(revision=token, rows=rows, row_fingerprints=row_fingerprints(view, rows), reviewed_at=utc_now()))
