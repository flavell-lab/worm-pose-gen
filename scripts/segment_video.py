#!/usr/bin/env python3
"""Run the promoted segmenter over a stretch of a recording and write a video.

Frames are read from the recording in slabs, flat-fielded with the same
correction the labeling app uses (estimated once per recording and cached
under the dataset root), pushed through the network in batches, and stitched
into an MP4: the flat-fielded frame in gray, the worm mask (probability at or
above ``--threshold``, then the narrow holes it encloses filled unless
``--no-fill-holes``; every component is kept) filled in magenta with a green outline, and, with
``--show-uncertain``, the band between ``0.2`` and ``0.8`` tinted yellow.
A JSON file beside the video records per-frame worm and uncertain pixel
counts, component counts and pixels outside the largest component before
cleanup, pixels added by hole filling, and throughput.  ``--scale 0.5`` halves the output size for sharing.

Example (one minute at 20 fps of an unseen recording):

    scripts/project_env.sh uv run --no-sync --frozen python scripts/segment_video.py \\
        --recording /store1/shared/all_data_raw/prj_aversion/2024-05-28/2024-05-28-02.h5 \\
        --start 0 --frames 1200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

from worm_pose_gen.classical import _erode, _largest_component
from worm_pose_gen.flat_field import apply_flat_field
from worm_pose_gen.mask_fit import fill_narrow_holes
from worm_pose_gen.recordings import RecordingSource
from worm_pose_gen.run_records import checkpoint_fingerprint, utc_now
from worm_pose_gen.settings import CACHE_DIR
from worm_pose_gen.segmenter import load_segmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "checkpoints" / "segmenter" / "videos"
UNCERTAIN_BAND = (0.2, 0.8)
HOLE_FILL_RADIUS_PX = 8
WORM_RGB = np.array([255, 80, 165], dtype=np.float32)
EDGE_RGB = np.array([90, 220, 140], dtype=np.float32)
UNCERTAIN_RGB = np.array([255, 210, 60], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--start", type=int, default=0, help="first frame index")
    parser.add_argument("--frames", type=int, default=1200, help="number of frames (1200 = one minute at 20 fps)")
    parser.add_argument("--fps", type=float, default=20.0, help="frame rate of the output video")
    parser.add_argument("--batch-size", type=int, default=16, help="frames per network forward pass")
    parser.add_argument("--slab", type=int, default=64, help="frames read from disk at once")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--show-uncertain", action="store_true", help="tint the 0.2-0.8 probability band yellow")
    parser.add_argument("--fill-holes", action=argparse.BooleanOptionalAction, default=True, help="fill the narrow holes the mask encloses")
    parser.add_argument("--hole-radius", type=int, default=HOLE_FILL_RADIUS_PX, help="largest hole width to fill, in pixels")
    parser.add_argument("--scale", type=float, default=1.0, help="resize factor for the output video")
    parser.add_argument("--quality", type=int, default=5, help="imageio/ffmpeg quality, 0 worst to 10 best")
    parser.add_argument("--dataset-root", type=Path, default=CACHE_DIR, help="where the flat field cache lives")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def component_stats(mask: np.ndarray) -> tuple[int, int]:
    """(number of components, worm pixels outside the largest component)."""

    if not mask.any():
        return 0, 0
    largest, _, count = _largest_component(mask)
    return int(count), int(mask.sum() - largest.sum())


def render(frame: np.ndarray, probability: np.ndarray, worm: np.ndarray, caption: str, *, show_uncertain: bool, scale: float) -> np.ndarray:
    rgb = np.repeat(frame[:, :, None].astype(np.float32), 3, axis=2)
    edge = worm & ~_erode(worm, 1)
    rgb[worm] = 0.55 * rgb[worm] + 0.45 * WORM_RGB
    if show_uncertain:
        uncertain = (probability > UNCERTAIN_BAND[0]) & (probability < UNCERTAIN_BAND[1]) & ~worm
        rgb[uncertain] = 0.6 * rgb[uncertain] + 0.4 * UNCERTAIN_RGB
    rgb[edge] = EDGE_RGB
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    if scale != 1.0:
        size = (max(2, int(round(image.width * scale)) // 2 * 2), max(2, int(round(image.height * scale)) // 2 * 2))
        image = image.resize(size, Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 8 + 7 * len(caption), 18), fill=(0, 0, 0))
    draw.text((4, 3), caption, fill=(255, 255, 255))
    return np.asarray(image)


def main() -> int:
    args = parse_args()
    started = utc_now()
    module = load_segmenter(args.checkpoint, args.device)
    source = RecordingSource(args.recording, args.dataset_root / "flat_fields")
    field_seconds = time.perf_counter()
    field = source.flat_field()
    field_seconds = time.perf_counter() - field_seconds
    total = source.frame_count
    frames_shape = source.shape
    start = max(0, args.start)
    stop = min(total, start + args.frames)
    if stop <= start:
        raise SystemExit(f"no frames in [{start}, {stop}) of {total}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.recording.stem}_f{start:06d}-{stop - 1:06d}" + (f"_x{args.scale:g}" if args.scale != 1.0 else "")
    video_path = args.output_dir / f"{stem}.mp4"
    writer = imageio.get_writer(
        str(video_path), fps=args.fps, codec="libx264", quality=args.quality, macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    rows = []
    timing = {"read": 0.0, "flat_field": 0.0, "network": 0.0, "render": 0.0}
    try:
        for slab_start in range(start, stop, args.slab):
            slab_stop = min(stop, slab_start + args.slab)
            t0 = time.perf_counter()
            raw = source.frames[slab_start:slab_stop]
            t1 = time.perf_counter()
            corrected = np.stack([
                np.clip(np.rint(apply_flat_field(frame, field, clip=(0.0, 255.0))), 0, 255).astype(np.uint8)
                for frame in raw
            ])
            t2 = time.perf_counter()
            probability = module.predict_probability_batch(corrected, batch_size=args.batch_size)
            if module.device.type == "cuda":
                torch.cuda.synchronize()
            t3 = time.perf_counter()
            for offset, (frame, prob) in enumerate(zip(corrected, probability)):
                index = slab_start + offset
                raw_mask = prob >= args.threshold
                uncertain_pixels = int(((prob > UNCERTAIN_BAND[0]) & (prob < UNCERTAIN_BAND[1])).sum())
                components, extra_pixels = component_stats(raw_mask)
                worm, filled_pixels = fill_narrow_holes(raw_mask, args.hole_radius, device=module.device) if args.fill_holes else (raw_mask, 0)
                worm_pixels = int(worm.sum())
                caption = (
                    f"{args.recording.stem} frame {index}  worm {worm_pixels} px  raw components {components}  "
                    f"outside largest {extra_pixels} px  filled {filled_pixels} px"
                )
                writer.append_data(render(frame, prob, worm, caption, show_uncertain=args.show_uncertain, scale=args.scale))
                rows.append({
                    "frame_index": index, "worm_pixels": worm_pixels, "raw_worm_pixels": int(raw_mask.sum()),
                    "uncertain_pixels": uncertain_pixels, "components": components,
                    "pixels_outside_largest_component": extra_pixels, "pixels_filled": filled_pixels,
                })
            t4 = time.perf_counter()
            timing["read"] += t1 - t0
            timing["flat_field"] += t2 - t1
            timing["network"] += t3 - t2
            timing["render"] += t4 - t3
            print(f"frames {slab_start}-{slab_stop - 1}: network {(t3 - t2) / (slab_stop - slab_start) * 1000:.1f} ms/frame", flush=True)
    finally:
        writer.close()
    source.close()
    n = len(rows)
    worm_counts = np.array([r["worm_pixels"] for r in rows])
    summary = {
        "started_at": started,
        "finished_at": utc_now(),
        "recording": str(args.recording),
        "frames": [start, stop - 1],
        "frame_count": n,
        "checkpoint": checkpoint_fingerprint(args.checkpoint),
        "threshold": args.threshold,
        "mask_cleanup": {"fill_holes": args.fill_holes, "fill_holes_radius_px": args.hole_radius},
        "batch_size": args.batch_size,
        "device": str(module.device),
        "flat_field_seconds": field_seconds,
        "seconds": timing,
        "network_ms_per_frame": 1000.0 * timing["network"] / n,
        "total_ms_per_frame": 1000.0 * sum(timing.values()) / n,
        "worm_pixels": {"median": float(np.median(worm_counts)), "min": int(worm_counts.min()), "max": int(worm_counts.max())},
        "frames_without_worm": int((worm_counts == 0).sum()),
        "frames_with_multiple_components": int(sum(r["components"] > 1 for r in rows)),
        "pixels_outside_largest_component": {
            "median": float(np.median([r["pixels_outside_largest_component"] for r in rows])),
            "max": int(max(r["pixels_outside_largest_component"] for r in rows)),
        },
        "pixels_filled": {
            "median": float(np.median([r["pixels_filled"] for r in rows])),
            "max": int(max(r["pixels_filled"] for r in rows)),
            "frames_with_filling": int(sum(r["pixels_filled"] > 0 for r in rows)),
        },
        "uncertain_fraction_median": float(np.median([r["uncertain_pixels"] for r in rows]) / (rows and (frames_shape[0] * frames_shape[1]) or 1)),
        "uncertain_pixels_median": float(np.median([r["uncertain_pixels"] for r in rows])),
        "video": str(video_path),
        "per_frame": rows,
    }
    (args.output_dir / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_frame"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
