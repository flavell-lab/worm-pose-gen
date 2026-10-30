"""Pseudo-labels for the body-field network from the pose pipeline's own confident fits (self-training).

The pipeline turns network output into whole-recording poses with
propagation and linking.  Where those poses are confident they are better
targets than the per-frame network that produced their masks, above all
inside coils and contacts that propagation bridged from clear frames on both
sides.  This module selects such frames from an analysed workspace, renders
body-field targets from the fitted tube, stores them with their temporal
context, and serves them to training beside the hand labels
(:mod:`model_training`, ``--pseudo-labels``); a set of them is a library item
(:mod:`library.pseudo_sets`).

**Selection** (:func:`select_rows`, :class:`SelectionParams`).  A row is

``clear``    an independent fit (never replaced by propagation) with IoU
             against the cleaned mask at least ``clear_iou`` and none of the
             failure flags (:data:`FAILURE_FLAGS`; self-contact, holes and
             the area deficit of an overlapping tube describe the pose, not
             a failed fit, and are allowed);
``bridged``  a row of a propagation stretch of at most ``max_stretch`` rows,
             fitted by a chain or a refit, with IoU at least ``bridged_iou``,
             no pose jump, length or edge flag, and fitted anchors on both
             sides of the stretch, one with IoU at least ``anchor_iou`` and
             neither below ``bridged_iou`` (chains run from both ends, and a
             long chain from one good anchor is as sound as two short ones).

Both kinds must also be **consistent** over ``±window`` rows: every row
fitted, no orientation flip between consecutive rows (the head end of one
pose closer to the tail end of the next), no pose jump of a body width and
no body-length change over ``length_jump`` between consecutive rows.  Clear
rows are kept on every ``clear_stride``-th frame, bridged rows on every
``bridged_stride``-th, and no frame that :func:`temporal_eval.excluded`
rejects (with its context) is ever kept.

**Targets** (:func:`mask_target`, :func:`target_rows`).  The mask target is
the fitted tube, not the cleaned network mask: the mask came from the
network being taught, while the tube is what the pipeline concluded, and at
contacts the two differ by whole blobs (area excess, fragments).  Where tube
and cleaned mask disagree within ``band_px`` of the tube's edge the pixel is
ignored, since there the tube's width model, not the image, decided.  A-P,
overlap and the head come from the tube by :func:`body_targets.render_body_targets`
as for a hand label; the tail heatmap is NaN (no loss), since the tail is the
end the pipeline is least sure of and confirmation of its errors is the main
risk of self-training.

**Items** are one ``.npz`` per frame in an item directory with an
``index.json``: the flat-fielded frames at the offsets the lags need
(:data:`CONTEXT_OFFSETS`, ±16 frames), their validity, the mask target
(0, 1, 255 ignored), A-P (NaN off the body), overlap, head, and the pose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


# Frames stored around an item: every offset lags 1, 4 and 16 need.
CONTEXT_OFFSETS = (-16, -4, -1, 0, 1, 4, 16)
IGNORE = 255
# Flags that say the fit failed rather than describing the pose.
FAILURE_FLAGS = ("low_iou", "area_excess", "fragments", "length_deviation", "pose_jump", "edge_inside")
# Flags a bridged row may carry: a coil or contact is why it was ambiguous.
BRIDGED_FAILURE_FLAGS = ("length_deviation", "pose_jump", "edge_inside")
CHAIN_ALGORITHMS = ("chain_forward", "chain_backward", "independent_refit")
INDEX = "index.json"


@dataclass(frozen=True)
class SelectionParams:
    clear_iou: float = 0.95
    bridged_iou: float = 0.90
    anchor_iou: float = 0.93
    max_stretch: int = 400
    window: int = 30  # rows each side (1.5 s at 20 Hz)
    length_jump: float = 0.03
    clear_stride: int = 8
    bridged_stride: int = 3
    band_px: float = 3.0


# --------------------------------------------------------------------------- selection


def _inside(points: NDArray[np.float64], shape: tuple[int, int] | None) -> NDArray[np.bool_]:
    if shape is None:
        return np.ones(points.shape[:-1], dtype=bool)
    height, width = shape
    return (points[..., 0] >= 0) & (points[..., 0] <= width - 1) & (points[..., 1] >= 0) & (points[..., 1] <= height - 1)


def orientation_flips(centerlines: NDArray[np.generic], image_shape: tuple[int, int] | None = None) -> NDArray[np.bool_]:
    """``[n-1]``: whether pose ``t+1`` matches pose ``t`` better reversed than as is (the ends swapped).

    Only points inside the image on both poses count, as in :func:`ambiguity.pose_jump_px`.
    """

    curves = np.asarray(centerlines, dtype=np.float64)
    a, b = curves[:-1], curves[1:]
    flips = np.zeros(len(a), dtype=bool)
    if not len(a):
        return flips
    inside_a = _inside(a, image_shape)
    distances = []
    for candidate in (b, b[:, ::-1]):
        both = inside_a & _inside(candidate, image_shape)
        d = np.linalg.norm(a - candidate, axis=-1)
        counted = both.sum(1)
        distances.append(np.where(counted >= 10, (d * both).sum(1) / np.maximum(counted, 1), np.nan))
    same, reverse = distances
    with np.errstate(invalid="ignore"):
        flips = reverse < same
    return np.asarray(flips, dtype=bool)


def _window_clean(bad_pair: NDArray[np.bool_], fitted: NDArray[np.bool_], window: int) -> NDArray[np.bool_]:
    """Rows whose ``±window`` rows are all fitted and hold no bad consecutive pair (the window must lie in the range)."""

    n = len(fitted)
    ok = np.zeros(n, dtype=bool)
    if n < 2 * window + 1:
        return ok
    unfitted = np.concatenate(([0], np.cumsum(~fitted)))
    bad = np.concatenate(([0], np.cumsum(bad_pair)))  # bad[k] = bad pairs among (0,1)..(k-1,k)
    rows = np.arange(window, n - window)
    lo, hi = rows - window, rows + window
    ok[rows] = (unfitted[hi + 1] - unfitted[lo] == 0) & (bad[hi] - bad[lo] == 0)
    return ok


def select_rows(
    state: dict[str, NDArray[Any]],
    algorithm: NDArray[Any],
    stretches: Sequence[Sequence[int]],
    params: SelectionParams = SelectionParams(),
    *,
    image_shape: tuple[int, int] | None = None,
    excluded: NDArray[np.bool_] | None = None,
) -> dict[str, Any]:
    """The rows of a workspace state to pseudo-label, as ``{"rows", "kind", "counts"}``.

    ``state`` is a workspace's ``state.npz`` (with the ambiguity stage's
    flags), ``algorithm`` its provenance algorithm per row, ``stretches``
    the propagation summary's ``[first, last]`` row pairs and ``excluded``
    the rows that may not be used.  ``kind`` is ``"clear"`` or ``"bridged"``.
    """

    frame_index = np.asarray(state["frame_index"])
    n = len(frame_index)
    fitted = np.asarray(state["fitted"], dtype=bool)
    iou = np.nan_to_num(np.asarray(state["iou"], dtype=np.float64), nan=0.0)
    algorithm = np.asarray(algorithm).astype(str)
    flag = lambda name: np.asarray(state[f"flag_{name}"], dtype=bool)
    width = np.asarray(state["width_px"], dtype=np.float64)
    length = np.asarray(state["body_length_px"], dtype=np.float64)

    flips = orientation_flips(state["centerline_xy"], image_shape)
    jump = np.asarray(state["pose_jump_px"], dtype=np.float64)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        pose_jump = jump > np.nan_to_num(width[1:], nan=np.inf)
        length_jump = np.abs(np.log(length[1:] / length[:-1])) > params.length_jump
    bad_pair = flips | pose_jump | np.nan_to_num(length_jump, nan=True).astype(bool)
    consistent = _window_clean(bad_pair, fitted, params.window)

    clear_quality = fitted & (iou >= params.clear_iou) & ~np.any([flag(f) for f in FAILURE_FLAGS], axis=0)
    clear = clear_quality & (algorithm == "independent_fit")

    bridged = np.zeros(n, dtype=bool)
    stretch_rows = 0
    for first, last in stretches:
        first, last = int(first), int(last)
        if last - first + 1 > params.max_stretch or first < 1 or last + 1 >= n:
            continue
        anchors = iou[[first - 1, last + 1]]
        if not (fitted[first - 1] and fitted[last + 1] and anchors.max() >= params.anchor_iou and anchors.min() >= params.bridged_iou):
            continue
        rows = np.arange(first, last + 1)
        stretch_rows += len(rows)
        bridged[rows] = True
    bridged &= (
        fitted & (iou >= params.bridged_iou) & np.isin(algorithm, CHAIN_ALGORITHMS)
        & ~np.any([flag(f) for f in BRIDGED_FAILURE_FLAGS], axis=0)
    )
    usable = consistent & (np.ones(n, dtype=bool) if excluded is None else ~np.asarray(excluded, dtype=bool))
    keep_clear = clear & usable & (frame_index % params.clear_stride == 0)
    keep_bridged = bridged & usable & (frame_index % params.bridged_stride == 0)
    rows = np.nonzero(keep_clear | keep_bridged)[0]
    kind = np.where(keep_bridged[rows], "bridged", "clear")
    counts = {
        "rows": n, "fitted": int(fitted.sum()), "consistent": int(consistent.sum()), "orientation_flips": int(flips.sum()),
        "clear_candidates": int(clear.sum()), "clear_consistent": int((clear & usable).sum()),
        "stretch_rows_with_anchors": stretch_rows, "bridged_candidates": int(bridged.sum()),
        "bridged_consistent": int((bridged & usable).sum()),
        "excluded": 0 if excluded is None else int(np.asarray(excluded).sum()),
        "kept_clear": int(keep_clear.sum()), "kept_bridged": int(keep_bridged.sum()),
    }
    return {"rows": rows, "kind": kind, "counts": counts}


# --------------------------------------------------------------------------- targets


def mask_target(clean: NDArray[np.bool_], tube: NDArray[np.bool_], band_px: float) -> NDArray[np.uint8]:
    """The tube as the mask target (0/1), with disagreement within ``band_px`` of the tube's edge ignored (255)."""

    from scipy.ndimage import distance_transform_edt

    tube = np.asarray(tube, dtype=bool)
    clean = np.asarray(clean, dtype=bool)
    target = tube.astype(np.uint8)
    disagree = tube != clean
    if disagree.any():
        inside = distance_transform_edt(tube)
        outside = distance_transform_edt(~tube)
        near_edge = np.where(tube, inside, outside) <= band_px
        target[disagree & near_edge] = IGNORE
    return target


def target_rows(item: dict[str, NDArray[Any]]) -> NDArray[np.float32]:
    """The ``[8,H,W]`` body-field target stack of an item; the tail heatmap is NaN (it trains nothing).

    Overlap trains nothing either: the item's is the tube's, which also marks
    body parts that only touch (:func:`model_eval.overlap_state`).
    """

    from .body_net import _TAIL, body_target_rows

    targets = {"ap": item["ap"], "head_xy": item["head_xy"], "tail_xy": np.full(2, np.nan), "diameter_px": float(item["diameter_px"])}
    rows = body_target_rows(item["mask"], targets, None)
    rows[_TAIL] = np.nan
    return rows


def render_item_targets(clean: NDArray[np.bool_], centerline: NDArray[np.generic], width_profile: NDArray[np.generic],
                        band_px: float) -> dict[str, NDArray[Any]]:
    """Mask target, A-P, overlap, head and diameter of one pose over its cleaned mask."""

    from .body_targets import render_body_targets
    from .pose_run import render_tube

    height, width = clean.shape
    points = np.asarray(centerline, dtype=np.float64)
    reach = float(np.max(width_profile))
    # (x0, x1, y0, y1) around the tube; render_tube grows it by its margin.
    window = (int(points[:, 0].min() - reach), int(points[:, 0].max() + reach) + 1,
              int(points[:, 1].min() - reach), int(points[:, 1].max() + reach) + 1)
    tube = render_tube(centerline, width_profile, height, width, window=window, device="cpu")
    mask = mask_target(clean, tube, band_px)
    body = render_body_targets(mask == 1, centerline, width_profile)
    return {"mask": mask, "ap": body.ap, "overlap": body.overlap, "head_xy": body.head_xy, "diameter_px": np.float64(body.diameter_px)}


# --------------------------------------------------------------------------- building items from a workspace


def workspace_selection(workspace_path: Path, params: SelectionParams = SelectionParams()) -> dict[str, Any]:
    """:func:`select_rows` on an analysed workspace, with :func:`temporal_eval.excluded` applied to every context frame."""

    from .temporal_eval import excluded
    from .workspace import Workspace

    workspace = Workspace.open(Path(workspace_path))
    state = workspace.load_state()
    algorithm = workspace.load_provenance()["algorithm"]
    summary = json.loads((Path(workspace_path) / "summary.json").read_text())
    stretches = (summary.get("propagation") or {}).get("stretches") or []
    recording = str(workspace.info.recording)
    frames = np.asarray(state["frame_index"])
    rejected = np.array([any(excluded(recording, int(f) + o) for o in (-16, 0, 16)) for f in frames])
    selection = select_rows(state, algorithm, stretches, params, image_shape=workspace.image_shape, excluded=rejected)
    selection.update(workspace=str(workspace_path), recording=recording)
    return selection


def build_items(workspace_path: Path, out_dir: Path, params: SelectionParams = SelectionParams(), *, limit: int | None = None,
                seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Write the items of one workspace into ``out_dir`` (``<workspace>_<frame>.npz``); their index entries and the selection counts.

    ``limit`` caps the clear items (a random subset, ``seed``); bridged items are all kept.
    """

    from .pipeline import Frames
    from .workspace import Workspace

    workspace = Workspace.open(Path(workspace_path))
    selection = workspace_selection(workspace_path, params)
    state = workspace.load_state()
    rows, kinds = selection["rows"], selection["kind"]
    if limit is not None and (kinds == "clear").sum() > limit:
        rng = np.random.default_rng(seed)
        clear = np.nonzero(kinds == "clear")[0]
        drop = rng.choice(clear, size=len(clear) - limit, replace=False)
        keep = np.setdiff1d(np.arange(len(rows)), drop)
        rows, kinds = rows[keep], kinds[keep]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(workspace_path).name
    entries = []
    masks = workspace.get_masks(rows)
    with Frames(Path(workspace.info.recording)) as source:
        for row, kind in zip(rows, kinds):
            frame = int(state["frame_index"][row])
            indices = [frame + o for o in CONTEXT_OFFSETS]
            valid = np.array([0 <= i < source.total for i in indices])
            clamped = [min(max(i, 0), source.total - 1) for i in indices]
            frames, _, _ = source.corrected(clamped)
            targets = render_item_targets(masks[int(row)], state["centerline_xy"][row], state["width_profile"][row], params.band_px)
            path = out_dir / f"{name}_{frame:06d}.npz"
            temporary = path.with_name("." + path.name + ".tmp")
            with open(temporary, "wb") as handle:
                np.savez_compressed(
                    handle, frames=frames, offsets=np.asarray(CONTEXT_OFFSETS), valid=valid, centerline_xy=state["centerline_xy"][row],
                    width_profile=state["width_profile"][row], **targets,
                )
            os.replace(temporary, path)
            entries.append({
                "file": path.name, "workspace": name, "recording": str(workspace.info.recording), "frame": frame, "kind": str(kind),
                "iou": float(state["iou"][row]), "ignored_px": int((targets["mask"] == IGNORE).sum()),
                "overlap_px": int(targets["overlap"].sum()), "self_contact": bool(state["flag_self_contact"][row]),
            })
    return entries, selection["counts"]


def write_index(out_dir: Path, entries: list[dict[str, Any]], params: SelectionParams, extra: dict[str, Any] | None = None,
                offsets: Sequence[int] = CONTEXT_OFFSETS) -> None:
    index = {"params": asdict(params), "context_offsets": [int(o) for o in offsets], "items": entries, **(extra or {})}
    path = Path(out_dir) / INDEX
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(index, indent=1))
    os.replace(temporary, path)


def read_index(directory: Path) -> dict[str, Any]:
    return json.loads((Path(directory) / INDEX).read_text())


# --------------------------------------------------------------------------- training data


class PseudoLabelDataset:
    """Pseudo-label items as body-field network inputs and targets, in :class:`model_training.LabelDataset`'s item layout."""

    def __init__(self, directories: Sequence[Path | str], *, lags: Sequence[int], augment: bool = False, crop_size: int | None = None,
                 seed: int = 0) -> None:
        self.items: list[Path] = []
        self.entries: list[dict[str, Any]] = []
        self.lags = tuple(int(lag) for lag in lags)
        for directory in directories:
            index = read_index(Path(directory))
            missing = sorted({sign * lag for lag in self.lags for sign in (-1, 1)} - set(index["context_offsets"]))
            if missing:
                raise ValueError(f"{directory}: items store offsets {index['context_offsets']}; the model needs {missing}")
            self.items += [Path(directory) / entry["file"] for entry in index["items"]]
            self.entries += index["items"]
        self.augment = augment
        self.crop_size = crop_size
        self.seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def counts(self) -> dict[str, Any]:
        """The items by kind and their recordings, for the model card's ``trained_on`` (no train/val counts: they are not labels)."""

        kinds = [entry["kind"] for entry in self.entries]
        return {
            "counts": {"pseudo": len(self.entries), **{kind: kinds.count(kind) for kind in sorted(set(kinds))}},
            "recordings": len({Path(entry["recording"]).stem for entry in self.entries}),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        from .body_net import TARGET_ROWS, _augment
        from .segmenter import INPUT_MEAN, INPUT_STD
        from .temporal_context import MAX_LAG, difference_channels

        path = self.items[index]
        with np.load(path) as archive:
            item = {name: archive[name] for name in archive.files}
        targets = target_rows(item)
        offsets = [int(o) for o in item["offsets"]]
        frames = item["frames"].astype(np.float32)
        if self.augment:
            rng = np.random.default_rng((self.seed, 1 << 20, index, int(torch.initial_seed()) % (1 << 31)))
            frames, targets = _augment(frames, targets, rng, self.crop_size)
        centre = offsets.index(0)
        image = ((frames[centre] / 255.0 - INPUT_MEAN) / INPUT_STD)[None]
        stack = np.zeros((2 * MAX_LAG + 1, *frames.shape[1:]), dtype=np.float32)
        valid = np.zeros(2 * MAX_LAG + 1, dtype=bool)
        for k, offset in enumerate(offsets):
            stack[MAX_LAG + offset] = frames[k]
            valid[MAX_LAG + offset] = bool(item["valid"][k])
        inputs = np.concatenate((image, difference_channels(stack, valid, self.lags)), axis=0)
        assert targets.shape[0] == TARGET_ROWS
        return {"key": f"pseudo/{path.stem}", "image": torch.as_tensor(inputs), "targets": torch.as_tensor(targets)}
