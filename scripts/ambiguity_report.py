#!/usr/bin/env python3
"""Ambiguity report for stored pose runs.

Recomputes the per-frame ambiguity signals of ``worm_pose_gen.ambiguity``
from a run's ``poses.npz`` (runs made before the signals existed included),
relates them to the fit overlap, and lists the flagged frames.  Writes one
JSON with an entry per run.

Example:

    scripts/project_env.sh uv run --no-sync --frozen python scripts/ambiguity_report.py \\
        <data dir>/runs/poses/<run> --output <data dir>/runs/ambiguity_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worm_pose_gen.ambiguity import FLAG_NAMES, AmbiguityThresholds, compute_ambiguity, summarize_ambiguity
from worm_pose_gen.frame_reader import open_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", type=Path, nargs="+", help="run directories with summary.json and poses.npz")
    parser.add_argument("--output", type=Path, default=None, help="JSON to write (default: print only)")
    parser.add_argument("--max-listed", type=int, default=60, help="flagged frames listed per run")
    return parser.parse_args()


def image_shape_of(summary: dict) -> tuple[int, int] | None:
    recording = Path(summary["recording"])
    if not recording.exists():
        return None
    with open_frames(recording) as frames:
        return frames.shape


def report_run(run: Path, max_listed: int) -> dict:
    arrays = dict(np.load(run / "poses.npz"))
    summary = json.loads((run / "summary.json").read_text())
    thresholds = AmbiguityThresholds()
    arrays.update(compute_ambiguity(arrays, prior=summary.get("prior"), image_shape=image_shape_of(summary), thresholds=thresholds))
    result = summarize_ambiguity(arrays, thresholds)
    fitted = arrays["fitted"]
    low = fitted & (arrays["iou"] < thresholds.low_iou)
    other = np.zeros_like(low)
    for name in FLAG_NAMES:
        if name != "low_iou":
            other |= arrays[f"flag_{name}"]
    result["low_iou_frames"] = int(low.sum())
    result["low_iou_frames_caught_by_another_flag"] = int((low & other).sum())
    result["frames_flagged_with_iou_at_least_0.9"] = int((other & ~low & fitted).sum())
    order = np.lexsort((arrays["iou"], -arrays["ambiguity_score"]))
    listed = []
    for row in order:
        if not fitted[row] or arrays["ambiguity_score"][row] == 0 or len(listed) >= max_listed:
            break
        listed.append({
            "frame_index": int(arrays["frame_index"][row]),
            "iou": round(float(arrays["iou"][row]), 3),
            "area_ratio": round(float(arrays["area_ratio"][row]), 3),
            "self_contact_px": round(float(arrays["self_contact_px"][row]), 1),
            "pose_jump_px": None if not np.isfinite(arrays["pose_jump_px"][row]) else round(float(arrays["pose_jump_px"][row]), 1),
            "flags": [name for name in FLAG_NAMES if arrays[f"flag_{name}"][row]],
        })
    result["flagged_frames"] = listed
    return {"run": str(run), "recording": summary["recording"], "frames": summary["frames"], "frame_count": int(fitted.sum()), **result}


def main() -> int:
    args = parse_args()
    output = {}
    for run in args.runs:
        entry = report_run(run, args.max_listed)
        output[run.name] = entry
        print(f"== {run.name}: {entry['frame_count']} fitted frames")
        print("   flags:", entry["flag_counts"])
        print(f"   score>=1: {entry['frames_with_score_at_least_1']}  score>=2: {entry['frames_with_score_at_least_2']}")
        print("   iou by score:", {k: (v["frames"], round(v["iou_median"], 3)) for k, v in entry["iou_by_score"].items()})
        print(f"   low-IoU frames {entry['low_iou_frames']}, caught by another flag {entry['low_iou_frames_caught_by_another_flag']};"
              f" flagged with IoU >= 0.9: {entry['frames_flagged_with_iou_at_least_0.9']}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
