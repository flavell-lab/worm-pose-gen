"""Temporal consistency of a body-field network on fixed clips, measured without labels.

The clips are listed in ``eval_sets/temporal_eval_set.json``: the seven sequence
clips, held-out stretches of test recordings (two of them through the
self-contacts of the ``nir-flv-labels-b2`` benchmark) and stretches of another
project's recordings no model trains on.  Their flat-fielded frames, with
``MAX_LAG`` frames of margin on each side, are cached once
(:func:`cached_clip`) so every model reads exactly the same input.

A network is run on every clip frame (:func:`predict_clip`) and judged on
how its outputs change from frame to frame.  At 20 Hz a worm moves a few
pixels per frame and its A-P field moves with the body, so what changes
faster than that is the network's own noise:

``mask_flicker``      share of the body area whose mask value differs from both
                      neighbouring frames while they agree (``m[t-1] == m[t+1] != m[t]``)
``ap_jitter``         mean ``|ap[t] - (ap[t-1] + ap[t+1]) / 2|`` on pixels in all three masks
``ap_flip_frames``    frame pairs whose A-P fields on their shared pixels are
                      anticorrelated: the ends swapped between two frames
``head_jumps``        frame pairs whose head peaks lie more than one body width apart
``head_swaps``        head jumps that land within a body width of the previous tail
``head_jitter_px``    median second difference of the head position (px)
``head_dropouts``     a present head that is absent for one frame and back the next
``overlap_flicker``   ``mask_flicker`` of the crossing map (``overlap >= 0.5``)

The tail metrics are reported beside the head's but the tail is the end
people cannot label consistently, so they are a weaker signal.

Predictions are kept per clip (:func:`save_prediction`, packed masks and
8-bit A-P) so videos comparing several models can be rendered later
(``scripts/render_field_comparison.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from .body_net import OUTPUTS
from .body_proposal import END_THRESHOLD, OVERLAP_THRESHOLD, _probabilities, network_inputs
from .settings import CACHE_DIR
from .temporal_context import MAX_LAG


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "eval_sets" / "temporal_eval_set.json"
DEFAULT_CACHE = CACHE_DIR / "temporal_clips"
# Median fitted body width on the sequence set (px); a head that moves more
# than this between frames (0.05 s) has jumped.
BODY_WIDTH_PX = 41.0
# Shared pixels below which a frame pair's A-P correlation is not judged.
MIN_SHARED_PX = 200
PREDICT_BATCH = 8


@dataclass(frozen=True)
class Clip:
    name: str
    group: str
    recording: str
    start: int
    frames: int

    @property
    def recording_name(self) -> str:
        return Path(self.recording).stem


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def clips(path: Path = DEFAULT_MANIFEST, names: Sequence[str] | None = None) -> list[Clip]:
    manifest = load_manifest(path)
    chosen = [Clip(**{k: c[k] for k in ("name", "group", "recording", "start", "frames")}) for c in manifest["clips"]]
    if names:
        missing = set(names) - {c.name for c in chosen}
        if missing:
            raise KeyError(f"unknown clips: {sorted(missing)}")
        chosen = [c for c in chosen if c.name in names]
    return chosen


def excluded(recording: str, frame: int, path: Path = DEFAULT_MANIFEST) -> bool:
    """Whether unlabeled-video training must not use this frame (a test or ood recording, or near an evaluation clip)."""

    manifest = load_manifest(path)
    name = Path(recording).stem
    if name in manifest["excluded_recordings"]:
        return True
    margin = int(manifest["exclusion_margin"])
    return any(
        Path(c["recording"]).stem == name and c["start"] - margin - MAX_LAG <= frame < c["start"] + c["frames"] + margin + MAX_LAG
        for c in manifest["clips"]
    )


def cached_clip(clip: Clip, cache_dir: Path = DEFAULT_CACHE, dataset_path: str = "/img_nir") -> tuple[NDArray[np.uint8], int]:
    """The clip's flat-fielded frames ``start - MAX_LAG .. start + frames + MAX_LAG`` (clipped to the recording) and the first one's index."""

    from .pipeline import Frames

    path = Path(cache_dir) / f"{clip.name}.npz"
    if path.exists():
        with np.load(path) as archive:
            return archive["frames"], int(archive["first"])
    with Frames(Path(clip.recording), dataset=dataset_path) as source:
        first = max(0, clip.start - MAX_LAG)
        last = min(source.total, clip.start + clip.frames + MAX_LAG)
        frames, _, _ = source.corrected(list(range(first, last)))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(handle, frames=frames, first=first)
    temporary.replace(path)
    return frames, first


@dataclass
class ClipPrediction:
    """A network's outputs on a clip's frames: masks, 8-bit A-P, crossings and end peaks."""

    mask: NDArray[np.bool_]       # [T,H,W] mask probability >= 0.5
    ap: NDArray[np.uint8]         # [T,H,W] round(254 * ap) + 1 on the mask, 0 off it
    overlap: NDArray[np.bool_]    # [T,H,W] overlap probability >= OVERLAP_THRESHOLD, on the mask
    head: NDArray[np.float64]     # [T,3] x, y, peak value
    tail: NDArray[np.float64]

    def ap_float(self, t: int) -> NDArray[np.float32]:
        return np.where(self.ap[t] > 0, (self.ap[t].astype(np.float32) - 1.0) / 254.0, np.nan)


Predictor = Callable[[torch.Tensor], torch.Tensor]


def predict_clip(module: Any, frames: NDArray[np.uint8], first: int, clip: Clip, *, batch_size: int = PREDICT_BATCH) -> ClipPrediction:
    """Run ``module`` (a :class:`body_net.BodyFieldModule` or anything with its ``lags``, ``device`` and forward) on every clip frame.

    Neighbours outside the cached frames (the recording's ends) give zero
    difference channels, as in the pipeline.
    """

    device = module.device
    lags = tuple(module.lags)
    targets = list(range(clip.start, clip.start + clip.frames))
    stack = {first + k: torch.as_tensor(frame) for k, frame in enumerate(frames)}
    shape = frames.shape[1:]
    out = ClipPrediction(
        mask=np.zeros((len(targets), *shape), bool), ap=np.zeros((len(targets), *shape), np.uint8),
        overlap=np.zeros((len(targets), *shape), bool), head=np.zeros((len(targets), 3)), tail=np.zeros((len(targets), 3)),
    )
    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        needed = {f + o for f in batch for o in (0, *lags, *(-lag for lag in lags))}
        on_device = {f: stack[f].to(device) for f in needed if f in stack}
        probability = _probabilities(module, network_inputs(on_device, batch, lags)).cpu().numpy()
        for k, p in enumerate(probability):
            row = start + k
            mask = p[OUTPUTS.index("mask")] >= 0.5
            out.mask[row] = mask
            out.ap[row] = np.where(mask, np.rint(254.0 * p[OUTPUTS.index("ap")]) + 1, 0).astype(np.uint8)
            out.overlap[row] = mask & (p[OUTPUTS.index("overlap")] >= OVERLAP_THRESHOLD)
            for name, target in (("head", out.head), ("tail", out.tail)):
                heat = p[OUTPUTS.index(name)]
                index = int(np.argmax(heat))
                target[row] = (index % shape[1], index // shape[1], float(heat.flat[index]))
    return out


def save_prediction(path: Path, prediction: ClipPrediction, clip: Clip) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle, mask=np.packbits(prediction.mask, axis=-1), overlap=np.packbits(prediction.overlap, axis=-1), ap=prediction.ap,
            head=prediction.head, tail=prediction.tail, width=prediction.mask.shape[-1], start=clip.start, name=clip.name,
        )
    temporary.replace(path)


def load_prediction(path: Path) -> ClipPrediction:
    with np.load(path) as archive:
        width = int(archive["width"])
        return ClipPrediction(
            mask=np.unpackbits(archive["mask"], axis=-1, count=width).astype(bool), ap=archive["ap"],
            overlap=np.unpackbits(archive["overlap"], axis=-1, count=width).astype(bool), head=archive["head"], tail=archive["tail"],
        )


def _flicker(maps: NDArray[np.bool_]) -> tuple[int, int]:
    """Pixels that differ from both neighbours while those agree, and the body area they are measured against."""

    flicker = 0
    area = 0
    for t in range(1, len(maps) - 1):
        before, now, after = maps[t - 1], maps[t], maps[t + 1]
        flicker += int(((before == after) & (now != before)).sum())
        area += int(now.sum())
    return flicker, area


def _end_metrics(points: NDArray[np.float64], other: NDArray[np.float64]) -> dict[str, float]:
    present = points[:, 2] >= END_THRESHOLD
    other_present = other[:, 2] >= END_THRESHOLD
    xy = points[:, :2]
    both = present[1:] & present[:-1]
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    jumps = both & (step > BODY_WIDTH_PX)
    to_other = np.linalg.norm(xy[1:] - other[:-1, :2], axis=1)
    swaps = jumps & other_present[:-1] & (to_other < BODY_WIDTH_PX)
    three = present[2:] & present[1:-1] & present[:-2]
    second = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=1)[three]
    dropouts = present[:-2] & ~present[1:-1] & present[2:]
    return {
        "jumps": int(jumps.sum()), "swaps": int(swaps.sum()), "jitter_px": float(np.median(second)) if len(second) else float("nan"),
        "dropouts": int(dropouts.sum()), "present": float(present.mean()),
    }


def clip_metrics(prediction: ClipPrediction) -> dict[str, Any]:
    """The metrics of the module docstring for one clip's prediction."""

    frames = len(prediction.mask)
    mask_flicker, area = _flicker(prediction.mask)
    overlap_flicker, overlap_area = _flicker(prediction.overlap)
    jitter_sum = 0.0
    jitter_px = 0
    flips = 0
    judged = 0
    steps: list[float] = []
    ap = [prediction.ap_float(t) for t in range(frames)]
    for t in range(frames - 1):
        shared = prediction.mask[t] & prediction.mask[t + 1]
        if shared.sum() >= MIN_SHARED_PX:
            a, b = ap[t][shared], ap[t + 1][shared]
            judged += 1
            steps.append(float(np.mean(np.abs(a - b))))
            if a.std() > 1e-3 and b.std() > 1e-3 and np.corrcoef(a, b)[0, 1] < 0:
                flips += 1
        if 0 < t:
            inner = prediction.mask[t - 1] & prediction.mask[t] & prediction.mask[t + 1]
            if inner.any():
                jitter_sum += float(np.abs(ap[t][inner] - 0.5 * (ap[t - 1][inner] + ap[t + 1][inner])).sum())
                jitter_px += int(inner.sum())
    head = _end_metrics(prediction.head, prediction.tail)
    tail = _end_metrics(prediction.tail, prediction.head)
    return {
        "frames": frames,
        "mask_flicker": mask_flicker / max(area, 1),
        "ap_jitter": jitter_sum / max(jitter_px, 1),
        "ap_step": float(np.median(steps)) if steps else float("nan"),
        "ap_flip_frames": flips, "ap_pairs_judged": judged,
        "overlap_flicker": overlap_flicker / max(overlap_area, 1), "overlap_area_px": overlap_area / max(frames, 1),
        "mask_area_px": float(prediction.mask.sum((1, 2)).mean()),
        **{f"head_{k}": v for k, v in head.items()}, **{f"tail_{k}": v for k, v in tail.items()},
    }


SUMMED = ("ap_flip_frames", "head_jumps", "head_swaps", "head_dropouts", "tail_jumps", "tail_swaps", "tail_dropouts", "frames")
FRAME_WEIGHTED = ("mask_flicker", "ap_jitter", "ap_step", "overlap_flicker", "head_jitter_px", "tail_jitter_px", "head_present", "tail_present")


def summarize(per_clip: dict[str, dict[str, Any]], groups: dict[str, str]) -> dict[str, dict[str, float]]:
    """Totals of the counts and frame-weighted means of the rates, over all clips and per group."""

    def combine(names: list[str]) -> dict[str, float]:
        rows = [per_clip[n] for n in names]
        weights = np.array([r["frames"] for r in rows], float)
        result: dict[str, float] = {k: float(sum(r[k] for r in rows)) for k in SUMMED}
        for k in FRAME_WEIGHTED:
            values = np.array([r[k] for r in rows], float)
            ok = np.isfinite(values)
            result[k] = float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")
        result["clips"] = len(rows)
        return result

    summary = {"all": combine(list(per_clip))}
    for group in sorted(set(groups.values())):
        summary[group] = combine([n for n in per_clip if groups[n] == group])
    return summary
