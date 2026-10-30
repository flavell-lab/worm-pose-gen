#!/usr/bin/env python3
"""Side-by-side residual images of several pose runs on the same frames.

Every run must cover the same recording.  For each requested frame the mask
is segmented once and each run's stored pose is rendered on it as a residual
panel (mask the tube misses in blue, tube outside the mask in red, head
square, tail circle), cropped to the union of the runs' crop windows.  Panels
are laid out left to right in the order the runs are given.

Example:

    scripts/project_env.sh uv run --no-sync --frozen python scripts/compare_pose_runs.py \\
        --run <step1 run> --run <6 coefficients> --run <12 coefficients> \\
        --frames 131,1021,1052 --output-dir <data dir>/runs/pose_run_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from worm_pose_gen.recordings import RecordingSource
from worm_pose_gen.pose_run import clean_mask, cleanup_options, draw_residual, render_tube, run_label
from worm_pose_gen.settings import CACHE_DIR, RUNS_DIR
from worm_pose_gen.segmenter import load_segmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt"
GAP_PX = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", type=Path, dest="runs", required=True, help="run directory (repeat, in display order)")
    parser.add_argument("--label", action="append", dest="labels", help="panel label per run (default: the run's width model)")
    parser.add_argument("--frames", required=True, help="comma-separated frame indices")
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "pose_run_comparison")
    parser.add_argument("--prefix", default="residual", help="output file prefix")
    parser.add_argument("--pad", type=int, default=24, help="pixels around the union of crop windows")
    parser.add_argument("--max-width", type=int, default=1800, help="downscale wider strips to this width")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality")
    parser.add_argument(
        "--checkpoint", type=Path, default=None,
        help="segmenter for every panel's mask (default: each run's own checkpoint from its summary, else the promoted one)",
    )
    parser.add_argument("--dataset-root", type=Path, default=CACHE_DIR)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = [json.loads((run / "summary.json").read_text()) for run in args.runs]
    arrays = [dict(np.load(run / "poses.npz")) for run in args.runs]
    recordings = {s["recording"] for s in summaries}
    if len(recordings) != 1:
        raise SystemExit(f"runs cover different recordings: {sorted(recordings)}")
    recording = Path(recordings.pop())
    labels = args.labels or [run_label(s) for s in summaries]
    if len(labels) != len(args.runs):
        raise SystemExit("one --label per --run is required")
    frames = [int(v) for v in args.frames.split(",") if v.strip()]
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    # Each run is drawn against the mask it was fit to: its own segmenter and cleanup.
    checkpoint_paths = [
        args.checkpoint or Path((s.get("checkpoint") or {}).get("path") or DEFAULT_CHECKPOINT) for s in summaries
    ]
    checkpoint_paths = [p if p.exists() else DEFAULT_CHECKPOINT for p in checkpoint_paths]
    modules = {}
    for path in checkpoint_paths:
        if path not in modules:
            modules[path] = load_segmenter(path, device)
    thresholds = [float(s.get("threshold", 0.5)) for s in summaries]
    cleanups = [cleanup_options(s) for s in summaries]
    source = RecordingSource(recording, args.dataset_root / "flat_fields")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index in frames:
            rows = []
            for a in arrays:
                hits = np.nonzero(a["frame_index"] == index)[0]
                rows.append(int(hits[0]) if len(hits) and a["fitted"][hits[0]] else None)
            if all(r is None for r in rows):
                print(f"frame {index}: no run fitted it, skipped", flush=True)
                continue
            frame = source.corrected(index)[1]
            height, width = frame.shape
            probabilities = {path: modules[path].predict_probability_batch(frame[None], batch_size=1)[0] for path in modules}
            masks = [
                clean_mask(probabilities[path], threshold, device=device, **c)[0]
                for path, threshold, c in zip(checkpoint_paths, thresholds, cleanups, strict=True)
            ]
            crops = np.array([a["crop"][r] for a, r in zip(arrays, rows, strict=True) if r is not None])
            x0, y0 = max(0, int(crops[:, 0].min()) - args.pad), max(0, int(crops[:, 2].min()) - args.pad)
            x1, y1 = min(width, int(crops[:, 1].max()) + args.pad), min(height, int(crops[:, 3].max()) + args.pad)
            window = (x0, x1, y0, y1)
            panels = []
            for a, row, label, mask in zip(arrays, rows, labels, masks, strict=True):
                if row is None:
                    panels.append(draw_residual(frame, mask, np.zeros_like(mask), None, f"{label}: no fit", window=window))
                    continue
                tube = render_tube(a["centerline_xy"][row], a["width_profile"][row], height, width, window=tuple(a["crop"][row]), device=device)
                caption = f"{label}: IoU {float(a['iou'][row]):.3f}"
                if "taper_asymmetry" in a and np.isfinite(a["taper_asymmetry"][row]):
                    caption += f", taper {float(a['taper_asymmetry'][row]):+.2f}"
                panels.append(draw_residual(frame, mask, tube, a["centerline_xy"][row], caption, window=window))
            strip = Image.new("RGB", (sum(p.width for p in panels) + GAP_PX * (len(panels) - 1), max(p.height for p in panels)), (0, 0, 0))
            x = 0
            for panel in panels:
                strip.paste(panel, (x, 0))
                x += panel.width + GAP_PX
            if strip.width > args.max_width:
                strip = strip.resize((args.max_width, int(round(strip.height * args.max_width / strip.width))), Image.LANCZOS)
            path = args.output_dir / f"{args.prefix}_frame_{index:05d}.jpg"
            strip.save(path, quality=args.quality, optimize=True)
            print(f"wrote {path} ({strip.width}x{strip.height})", flush=True)
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
