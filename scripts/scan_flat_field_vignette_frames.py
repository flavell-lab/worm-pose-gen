#!/usr/bin/env python3
"""Find frames whose worm sits in the vignette and measure what flat-fielding changes.

The uniformly sampled 30-frame stress set rarely places the animal in a dark
corner, so it cannot show whether the illumination fall-off clips the mask.
This scan samples one recording densely, keeps frames whose Section 3
component overlaps the low-illumination region (flat-field gain >= 1.3), and
compares the component with and without correction.  For the frames with the
largest mask change it also runs the mask fit both ways and reports the
fitted length and overlap, which is where a clipped tail shows up.

No annotations are used; every number is agreement with the segmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from worm_pose_gen.classical import ClassicalConfig, segment_dark_ridge
from worm_pose_gen.flat_field import apply_flat_field, estimate_flat_field
from worm_pose_gen.frame_reader import DATASET_PATH, open_frames
from worm_pose_gen.mask_fit import (
    MaskFitConfig,
    fill_narrow_holes,
    fit_mask,
    standard_initializations,
)


DEFAULT_RECORDING = Path(
    "/store1/shared/all_data_raw/prj_aversion/2024-01-31/2024-01-31-02.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, default=DEFAULT_RECORDING)
    parser.add_argument("--dataset", default=DATASET_PATH, help="the HDF5 dataset (ignored for a video)")
    parser.add_argument("--scan-frames", type=int, default=300)
    parser.add_argument("--flat-field-frames", type=int, default=64)
    parser.add_argument("--dark-gain", type=float, default=1.3)
    parser.add_argument("--min-dark-fraction", type=float, default=0.05)
    parser.add_argument("--fit-top", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ClassicalConfig()
    fit_config = MaskFitConfig()
    rows: list[dict[str, float | int]] = []
    with open_frames(args.recording, args.dataset) as reader:
        count = reader.frame_count
        calibration = reader.read(np.linspace(0, count - 1, args.flat_field_frames, dtype=np.int64))
        field = estimate_flat_field(
            calibration, temporal_quantile=0.8, spatial_radius=31,
            smoothing_passes=2, min_gain=0.5, max_gain=2.5,
        )
        del calibration
        dark = field.gain >= args.dark_gain
        print(
            json.dumps(
                {
                    "recording": args.recording.stem,
                    "dark_region_fraction": float(dark.mean()),
                    "gain_max": float(field.gain.max()),
                }
            ),
            flush=True,
        )
        frames: dict[int, np.ndarray] = {}
        for index in np.linspace(0, count - 1, args.scan_frames, dtype=np.int64):
            frame = reader[int(index)]
            component = segment_dark_ridge(frame, cfg).component
            area = int(component.sum())
            if area < cfg.min_area:
                continue
            fraction = float((component & dark).sum() / area)
            if fraction < args.min_dark_fraction:
                continue
            corrected = segment_dark_ridge(
                apply_flat_field(frame, field, clip=(0.0, 255.0)), cfg
            ).component
            rows.append(
                {
                    "frame_index": int(index),
                    "component_fraction_in_dark_region": fraction,
                    "area_raw_px": area,
                    "area_flat_fielded_px": int(corrected.sum()),
                    "pixels_gained": int((corrected & ~component).sum()),
                    "pixels_lost": int((component & ~corrected).sum()),
                }
            )
            frames[int(index)] = frame

    deltas = np.asarray([r["area_flat_fielded_px"] - r["area_raw_px"] for r in rows], dtype=float)
    summary = {
        "scanned_frames": args.scan_frames,
        "frames_in_dark_region": len(rows),
        "area_delta_px_median": float(np.median(deltas)) if len(rows) else None,
        "area_delta_fraction_median": (
            float(np.median(deltas / np.asarray([r["area_raw_px"] for r in rows]))) if len(rows) else None
        ),
        "frames_with_net_gain": int(np.sum(deltas > 0)),
    }
    print(json.dumps(summary), flush=True)

    fits = []
    for row in sorted(rows, key=lambda r: r["pixels_gained"] - r["pixels_lost"], reverse=True)[: args.fit_top]:
        frame = frames[int(row["frame_index"])]
        record: dict[str, float | int | str] = {"frame_index": int(row["frame_index"])}
        for label, image in (
            ("raw", frame.astype(np.float64)),
            ("flat_fielded", apply_flat_field(frame, field, clip=(0.0, 255.0))),
        ):
            started = time.perf_counter()
            component = segment_dark_ridge(image, cfg).component
            target, _ = fill_narrow_holes(component, 8, device=args.device)
            starts = standard_initializations(target, config=fit_config)
            result = fit_mask(target, starts, config=fit_config, device=args.device)
            record[f"{label}_iou"] = result.records[result.best_index]["final_iou"]
            record[f"{label}_length_px"] = result.body_length_px
            record[f"{label}_width_px"] = result.width_px
            record[f"{label}_points_in_fov"] = result.points_in_fov
            record[f"{label}_seconds"] = time.perf_counter() - started
        fits.append(record)
        print(json.dumps(record), flush=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "recording": str(args.recording),
                    "dark_gain_threshold": args.dark_gain,
                    "summary": summary,
                    "frames": rows,
                    "fits": fits,
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
