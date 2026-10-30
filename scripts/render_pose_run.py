#!/usr/bin/env python3
"""Render the overlay video and residual images of a stored pose run.

Takes a run directory written by ``scripts/fit_recording.py`` (``summary.json``
and ``poses.npz``) and produces, without refitting, the same ``overlay.mp4``
the fitter writes with ``--video`` and the same residual images it writes
for its worst frames (mask the tube misses in blue, tube outside the mask in
red).  Frames are re-read from the recording and flat-fielded; residual
images re-run the segmenter on the few frames they need.

Example:

    scripts/project_env.sh uv run --no-sync --frozen python scripts/render_pose_run.py \\
        <data dir>/runs/poses/<run> --scale 0.5 --residual-frames 6 --frames 451,1052
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from worm_pose_gen.recordings import RecordingSource
from worm_pose_gen.pose_run import clean_mask, cleanup_options, draw_residual, render_tube, residual_caption, residual_rows, write_overlay_video
from worm_pose_gen.settings import CACHE_DIR
from worm_pose_gen.segmenter import load_segmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="run directory with summary.json and poses.npz")
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True, help="write overlay.mp4 (skipped when present unless --force)")
    parser.add_argument("--force", action="store_true", help="rewrite an existing overlay.mp4")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=float, default=1.0, help="resize factor for the video")
    parser.add_argument("--quality", type=int, default=5, help="imageio/ffmpeg quality, 0 worst to 10 best")
    parser.add_argument("--slab", type=int, default=64, help="frames read from disk at a time")
    parser.add_argument("--residual-frames", type=int, default=5, help="residual images for this many lowest-IoU frames")
    parser.add_argument("--frames", default="", help="comma-separated frame indices that also get residual images")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="segmenter used for the residual masks")
    parser.add_argument("--dataset-root", type=Path, default=CACHE_DIR, help="where the flat field cache lives")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    arrays = dict(np.load(args.run / "poses.npz"))
    recording = Path(summary["recording"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    source = RecordingSource(recording, args.dataset_root / "flat_fields")
    field = source.flat_field()
    n_points = arrays["centerline_xy"].shape[1]
    frame_index = arrays["frame_index"]
    try:
        video_path = args.run / "overlay.mp4"
        if args.video and (args.force or not video_path.exists()):
            started = time.perf_counter()
            write_overlay_video(
                video_path, source.frames, field, arrays, caption_prefix=recording.stem,
                fps=args.fps, scale=args.scale, quality=args.quality, slab=args.slab, device=device,
            )
            print(f"wrote {video_path} in {time.perf_counter() - started:.0f} s", flush=True)
        requested = [int(v) for v in args.frames.split(",") if v.strip()]
        rows = residual_rows(arrays, args.residual_frames, requested)
        if rows:
            module = load_segmenter(args.checkpoint, device)
            threshold = float(summary.get("threshold", 0.5))
            cleanup = cleanup_options(summary)
            for row in rows:
                index = int(frame_index[row])
                frame = source.corrected(index)[1]
                probability = module.predict_probability_batch(frame[None], batch_size=1)[0]
                mask, _ = clean_mask(probability, threshold, device=device, **cleanup)
                tube = render_tube(
                    arrays["centerline_xy"][row], arrays["width_profile"][row], *frame.shape, window=tuple(arrays["crop"][row]), device=device
                )
                image = draw_residual(frame, mask, tube, arrays["centerline_xy"][row], residual_caption(index, arrays, row, mask))
                path = args.run / f"frame_{index:06d}_iou{float(arrays['iou'][row]):.3f}.png"
                image.save(path)
                print(f"wrote {path.name}", flush=True)
    finally:
        source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
