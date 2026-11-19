#!/usr/bin/env python3
"""Calibrate the stage against the image and measure what substrate alignment does (docs/STAGE_ALIGNMENT.md).

For each ``--recording``: background shifts of frame pairs at lags 1, 4 and
16, the calibration fit to them (and to each quarter of the recording on
its own), written to the stage-calibration cache.  With ``--workspace``,
the same for the workspace's recording with the workspace's worm masks,
then the background difference energy ``frame[t+lag] - frame[t-lag]``
(worm masked out, flat-fielded) unaligned, stage-aligned and registered,
and a lag-16 figure of frames around ``--figure-frames``.  Writes
``<output-dir>/report.json`` and the figure.

With ``--fill`` the recordings keep their cached calibrations and the
report is on the registration fill instead (``substrate_shift_px``): the
untrusted frames and fill stretches, the time the fill's registrations take
on first use, and on ``--energy-frames`` frames sampled from the stretches
the difference energy (stage only, filled, registered) and the residual of
stage-only and filled offsets against direct registration at lags 4 and 16,
pairs the fill does not use.  Writes ``<output-dir>/fill_report.json``.
Examples::

    scripts/project_env.sh env OMP_NUM_THREADS=4 uv run --no-sync --frozen python scripts/stage_alignment_report.py \\
        --workspace <data dir>/workspaces/2023-03-30-01 --recording /storage/fs/data2/prj_aversion/data_raw/2023-08-22/2023-08-22-01.h5
    scripts/project_env.sh env OMP_NUM_THREADS=1 uv run --no-sync --frozen python scripts/stage_alignment_report.py --fill \\
        --recording /storage/fs/data2/prj_aversion/data_raw/2023-03-30/2023-03-30-01.h5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from scipy import ndimage

from worm_pose_gen.recordings import RecordingSource
from worm_pose_gen.recording_files import recording_name
from worm_pose_gen.settings import CACHE_DIR, RUNS_DIR
from worm_pose_gen.stage_alignment import (
    CALIBRATION_LAGS,
    DEFAULT_CACHE,
    MASK_DILATION_PX,
    StageCalibration,
    StageTrack,
    anchor_frames,
    bandpass,
    dark_mask,
    fill_stretches,
    fit_calibration,
    measure_shifts,
    read_stage,
    recording_calibration,
    register_shift,
    shift_image,
    stage_shift_px,
    substrate_shift_px,
    untrusted_frames,
)
from worm_pose_gen.workspace import Workspace


ENERGY_MODES = ("unaligned", "stage, offset 0", "stage, offset -0.5", "stage, fitted offset", "registered")
FILL_MODES = ("unaligned", "stage", "filled", "registered")
MaskOf = Callable[[int], np.ndarray | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", action="append", default=[], help="recording to calibrate (dark-pixel worm masks)")
    parser.add_argument("--workspace", type=Path, help="workspace whose recording is calibrated with its masks and measured")
    parser.add_argument("--anchors", type=int, default=400, help="frames spread over each recording, each registered at lags 1, 4, 16")
    parser.add_argument(
        "--energy-frames", type=int, default=150, help="frames for the energy table (spread over the workspace's recording, or from the fill stretches)"
    )
    parser.add_argument("--figure-frames", type=int, nargs=2, default=(4700, 4730), help="first and last frame t of the figure")
    parser.add_argument("--fill", action="store_true", help="report on the registration fill with the cached calibrations")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "stage_alignment")
    return parser.parse_args()


def summarize(recording: Path, shifts: np.ndarray, calibration: StageCalibration) -> dict[str, Any]:
    """Calibration, its geometry, residuals per lag, and the per-quarter fits."""

    matrix = np.asarray(calibration.matrix)
    scales = np.linalg.svd(matrix, compute_uv=False)
    track = read_stage(recording)
    t, lag = shifts[:, 0], shifts[:, 1]
    predicted = (track.at(t + lag + calibration.offset_frames) - track.at(t + calibration.offset_frames)) @ matrix.T
    residual = np.linalg.norm(shifts[:, 2:4] - predicted, axis=1)
    finite = np.isfinite(residual)
    quarters = []
    for q in range(4):
        part = (t >= q * len(track.filled) / 4) & (t < (q + 1) * len(track.filled) / 4)
        fit = fit_calibration(track, shifts[part])
        quarters.append({"matrix": fit.matrix, "offset_frames": fit.offset_frames, "residual_px": fit.residual_px})
    return {
        "recording": str(recording),
        "calibration": calibration.to_dict(),
        "um_per_px": [float(0.1 / s) for s in scales],
        "rotation_deg": float(np.degrees(np.arctan2(matrix[1, 0] - matrix[0, 1], matrix[0, 0] + matrix[1, 1]))),
        "determinant_sign": float(np.sign(np.linalg.det(matrix))),
        "stage_rows_missing": float(np.mean(~np.isfinite(track.raw).all(axis=1))),
        "frames_unreliable": float(np.mean(~track.reliable)),
        "residual_by_lag": {
            int(l): {
                "median_px": float(np.median(residual[finite & (lag == l)])),
                "p90_px": float(np.percentile(residual[finite & (lag == l)], 90)),
                "over_3px": float(np.mean(residual[finite & (lag == l)] > 3.0)),
            }
            for l in CALIBRATION_LAGS
        },
        "quarters": quarters,
    }


def calibrate(recording: Path, anchors: int, mask_of: MaskOf | None) -> dict[str, Any]:
    track = read_stage(recording)
    reliable = np.flatnonzero(track.reliable)
    frames = np.unique(np.linspace(reliable[0], reliable[-1] - max(CALIBRATION_LAGS), anchors).astype(np.int64))
    shifts = measure_shifts(recording, frames, CALIBRATION_LAGS, mask_of)
    calibration = fit_calibration(track, shifts)
    calibration.save(DEFAULT_CACHE / f"{recording_name(recording)}.json")
    summary = summarize(recording, shifts, calibration)
    print(json.dumps({k: summary[k] for k in ("recording", "calibration", "um_per_px", "rotation_deg", "residual_by_lag")}), flush=True)
    return summary


def with_offset(calibration: StageCalibration, offset: float) -> StageCalibration:
    return StageCalibration(calibration.matrix, offset, calibration.residual_px, calibration.pairs, calibration.outliers)


class Pair:
    """Frames t-lag, t, t+lag of a recording, flat-fielded, with dilated worm masks (``dark_mask`` where ``mask_of`` gives none)."""

    def __init__(self, source: RecordingSource, mask_of: MaskOf | None, t: int, lag: int) -> None:
        self.frames = {k: source.corrected(t + k)[1].astype(np.float64) for k in (-lag, 0, lag)}
        masks = {k: None if mask_of is None else mask_of(t + k) for k in (-lag, 0, lag)}
        self.masks = {
            k: ndimage.binary_dilation(dark_mask(self.frames[k]) if mask is None else mask, iterations=MASK_DILATION_PX)
            for k, mask in masks.items()
        }
        self.t, self.lag = t, lag

    def registered(self) -> tuple[np.ndarray, np.ndarray]:
        moved = []
        for k in (-self.lag, self.lag):
            shift, _ = register_shift(bandpass(self.frames[0]), bandpass(self.frames[k]), self.masks[0] | self.masks[k])
            moved.append(shift)
        return np.round(moved[0]), np.round(moved[1])

    def from_offsets(self, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """Whole-pixel moves that put frame t-lag's and t+lag's background where frame t has it, from per-frame substrate offsets."""

        t, lag = self.t, self.lag
        if not np.isfinite(offsets[[t - lag, t, t + lag]]).all():
            return None
        return np.round(offsets[t - lag] - offsets[t]), np.round(offsets[t + lag] - offsets[t])

    def difference(self, before: np.ndarray, after: np.ndarray, filtered: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``frame[t+lag] - frame[t-lag]`` after the moves, the valid pixels off the worm in all three frames, and the valid worm pixels."""

        lag = self.lag
        a, b = (bandpass(self.frames[k]) if filtered else self.frames[k] for k in (-lag, lag))
        a, valid_a = shift_image(a, before)
        b, valid_b = shift_image(b, after)
        worm_a = shift_image(self.masks[-lag].astype(np.float64), before)[0] > 0.01
        worm_b = shift_image(self.masks[lag].astype(np.float64), after)[0] > 0.01
        worm = worm_a | worm_b | self.masks[0]
        return b - a, valid_a & valid_b & ~worm, valid_a & valid_b & worm


def energy_table(
    recording: Path, mask_of: MaskOf | None, anchors: np.ndarray, offsets: dict[str, np.ndarray], modes: tuple[str, ...]
) -> dict[str, Any]:
    """Median over ``anchors`` of the mean square difference per lag and region, for every mode on the same pixels.

    ``offsets`` maps a mode to per-frame substrate offsets; "unaligned" and
    "registered" need none.  A frame counts only where every mode is defined.
    """

    source = RecordingSource(recording, CACHE_DIR / "flat_fields")
    zeros = np.zeros((source.frame_count, 2))
    regions = ("background raw", "background band-passed", "worm raw")
    energy = {(lag, region, mode): [] for lag in CALIBRATION_LAGS for region in regions for mode in modes}
    for t in anchors:
        for lag in CALIBRATION_LAGS:
            if t - lag < 0 or t + lag >= source.frame_count:
                continue
            pair = Pair(source, mask_of, int(t), lag)
            moves = {mode: pair.registered() if mode == "registered" else pair.from_offsets(offsets.get(mode, zeros)) for mode in modes}
            if any(move is None for move in moves.values()):
                continue
            for filtered in (False, True):
                differences = {mode: pair.difference(*moves[mode], filtered) for mode in modes}
                # The same pixels for every mode: background off the worm everywhere, worm wherever any mode has it.
                background = np.logical_and.reduce([keep for _, keep, _ in differences.values()])
                worm = np.logical_or.reduce([on for _, _, on in differences.values()]) & np.logical_and.reduce(
                    [keep | on for _, keep, on in differences.values()]
                )
                if background.sum() < 10_000:
                    continue
                for mode, (difference, _, _) in differences.items():
                    region = "background band-passed" if filtered else "background raw"
                    energy[(lag, region, mode)].append(float(np.mean(difference[background] ** 2)))
                    if not filtered and worm.any():
                        energy[(lag, "worm raw", mode)].append(float(np.mean(difference[worm] ** 2)))
    source.close()
    table: dict[str, Any] = {}
    for lag in CALIBRATION_LAGS:
        for region in regions:
            key = f"lag {lag}, {region}"
            values = {mode: energy[(lag, region, mode)] for mode in modes}
            table[key] = {mode: float(np.median(v)) if v else None for mode, v in values.items()}
            table[key]["frames"] = len(values[modes[0]])
            print(key, json.dumps(table[key]), flush=True)
    return table


def figure(workspace: Workspace, calibration: StageCalibration, first: int, last: int, path: Path) -> list[int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recording = workspace.recording
    track = read_stage(recording)
    stage = stage_shift_px(track, np.arange(len(track.raw)), calibration)
    usable = [t for t in range(first, last + 1) if np.isfinite(stage[[t - 16, t, t + 16]]).all()]
    if not usable:
        return []
    chosen = sorted({usable[0], usable[len(usable) // 2], usable[-1]})
    source = RecordingSource(recording, CACHE_DIR / "flat_fields")
    mask_of = lambda frame: workspace.effective_mask(workspace.row_of(frame))
    modes = ("unaligned", "stage, fitted offset", "registered")
    fig, axes = plt.subplots(2 * len(chosen), len(modes), figsize=(5.0 * len(modes), 3.9 * 2 * len(chosen)))
    for row, t in enumerate(chosen):
        pair = Pair(source, mask_of, t, 16)
        moves = {"unaligned": (np.zeros(2), np.zeros(2)), "stage, fitted offset": pair.from_offsets(stage), "registered": pair.registered()}
        for column, mode in enumerate(modes):
            for k, filtered in enumerate((False, True)):
                difference, keep, _ = pair.difference(*moves[mode], filtered)
                ax = axes[2 * row + k, column]
                limit = 8.0 if filtered else 40.0
                ax.imshow(np.where(keep | pair.masks[0], difference, 0.0), cmap="gray", vmin=-limit, vmax=limit)
                ax.contour(pair.masks[0], levels=[0.5], colors="c", linewidths=0.6)
                kind = "band-passed" if filtered else "raw"
                ax.set_title(f"t={t}: frame[t+16] - frame[t-16], {kind}, {mode}\nbackground mean square {np.mean(difference[keep] ** 2):.1f}", fontsize=8)
                ax.axis("off")
    source.close()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return chosen


def residuals(values: np.ndarray) -> dict[str, Any]:
    """Count, median, 90th percentile and the fractions over 3 and 8 px; NaN (no offset) counts as over."""

    finite = values[np.isfinite(values)]
    return {
        "pairs": int(len(values)),
        "nan": float(np.mean(~np.isfinite(values))),
        "median_px": float(np.median(finite)) if len(finite) else None,
        "p90_px": float(np.percentile(finite, 90)) if len(finite) else None,
        "over_3px": float(np.mean(~(values <= 3.0))),
        "over_8px": float(np.mean(~(values <= 8.0))),
    }


def fill_report(recording: Path, mask_of: MaskOf | None, count: int, rng: np.random.Generator) -> dict[str, Any]:
    """The fill's stretches and cost, then stage-only vs filled on frames sampled from the stretches."""

    calibration = recording_calibration(recording, mask_of)
    track: StageTrack = read_stage(recording)
    n = len(track.raw)
    stage = stage_shift_px(track, np.arange(n), calibration)
    untrusted = untrusted_frames(track, calibration)
    stretches = fill_stretches(untrusted, anchor_frames(track, calibration, untrusted))
    inside = np.zeros(n, dtype=bool)
    for start, stop in stretches:
        inside[start:stop] = True
    began = time.perf_counter()
    filled = substrate_shift_px(recording, np.arange(n), calibration)
    seconds = time.perf_counter() - began
    lags = (4, 16)
    candidates = np.flatnonzero(inside[: n - max(CALIBRATION_LAGS)])
    candidates = candidates[candidates >= max(CALIBRATION_LAGS)]
    sample = np.sort(rng.choice(candidates, size=min(count, len(candidates)), replace=False))
    summary: dict[str, Any] = {
        "recording": str(recording),
        "frames": n,
        "untrusted": float(untrusted.mean()),
        "stretches": len(stretches),
        "frames_in_stretches": float(inside.mean()),
        "fill_seconds": seconds,
        "stage_nan": float(np.mean(~np.isfinite(stage).all(axis=1))),
        "filled_nan": float(np.mean(~np.isfinite(filled).all(axis=1))),
        "sampled_frames": int(len(sample)),
    }
    print(json.dumps(summary), flush=True)
    direct = measure_shifts(recording, sample, lags, mask_of)
    held_out: dict[str, Any] = {}
    for lag in lags:
        rows = direct[(direct[:, 1] == lag) & (direct[:, 4] >= 15.0)]
        t = rows[:, 0].astype(np.int64)
        errors = {
            name: np.linalg.norm(-(offsets[t + lag] - offsets[t]) - rows[:, 2:4], axis=1)
            for name, offsets in (("stage", stage), ("filled", filled))
        }
        defined = np.isfinite(errors["stage"])
        for name, error in errors.items():
            held_out[f"lag {lag}, {name}"] = residuals(error)
            held_out[f"lag {lag}, {name}, where the stage is defined"] = residuals(error[defined])
        for key in [key for key in held_out if key.startswith(f"lag {lag},")]:
            print(key, json.dumps(held_out[key]), flush=True)
    summary["held_out"] = held_out
    summary["energy"] = energy_table(recording, mask_of, sample, {"stage": stage, "filled": filled}, FILL_MODES)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fill:
        rng = np.random.default_rng(0)
        targets: list[tuple[Path, MaskOf | None]] = [(Path(recording), None) for recording in args.recording]
        if args.workspace is not None:
            workspace = Workspace.open(args.workspace)
            targets.insert(0, (workspace.recording, lambda frame: workspace.effective_mask(workspace.row_of(frame))))
        report = {"fill": [fill_report(recording, mask_of, args.energy_frames, rng) for recording, mask_of in targets]}
        (args.output_dir / "fill_report.json").write_text(json.dumps(report, indent=1))
        print("done", args.output_dir)
        return
    report: dict[str, Any] = {"recordings": []}
    if args.workspace is not None:
        workspace = Workspace.open(args.workspace)
        mask_of = lambda frame: workspace.effective_mask(workspace.row_of(frame))
        summary = calibrate(workspace.recording, args.anchors, mask_of)
        report["recordings"].append(summary)
        calibration = StageCalibration.from_dict(summary["calibration"])
        name = workspace.recording_name(recording)
        report["figure_frames"] = figure(
            workspace, calibration, *args.figure_frames, args.output_dir / f"{name}_lag16_aligned_vs_unaligned.png"
        )
        track = read_stage(workspace.recording)
        frames = np.arange(len(track.raw))
        offsets = {
            "stage, offset 0": stage_shift_px(track, frames, with_offset(calibration, 0.0)),
            "stage, offset -0.5": stage_shift_px(track, frames, with_offset(calibration, -0.5)),
            "stage, fitted offset": stage_shift_px(track, frames, calibration),
        }
        lag = max(CALIBRATION_LAGS)
        anchors = np.linspace(lag + 1, len(track.raw) - lag - 2, args.energy_frames).astype(np.int64)
        report["energy"] = energy_table(workspace.recording, mask_of, anchors, offsets, ENERGY_MODES)
    for recording in args.recording:
        report["recordings"].append(calibrate(Path(recording), args.anchors, None))
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1))
    print("done", args.output_dir)


if __name__ == "__main__":
    main()
