#!/usr/bin/env python3
"""Find candidate clips with coils, self-contact, holes, fragments, and camera exits.

Scans a recording at a frame stride with the segmenter only (no fitting) and
records mask statistics per sampled frame: area, whether the mask touches
the image border, filled hole pixels, pixels outside the largest component,
the bounding-box fill of the mask, and the longest skeleton path.  A coiled
or self-touching body has a short skeleton for its area and a well-filled
bounding box; a tight omega turn encloses background that the hole fill
removes; a body leaving the camera touches the border.  Flagged samples are
grouped into windows, and a montage of each window's peak frame is written
so the clips can be picked by eye.

Outputs: ``<name>_candidates.json`` and ``<name>_candidates.jpg`` in
``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from worm_pose_gen.flat_field import apply_flat_field
from worm_pose_gen.recordings import RecordingSource
from worm_pose_gen.mask_fit import MaskFitConfig, init_from_skeleton
from worm_pose_gen.pose_run import boundary, clean_mask, touches_border
from worm_pose_gen.settings import CACHE_DIR, RUNS_DIR
from worm_pose_gen.segmenter import load_segmenter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "segmenter" / "best.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None, help="frames to cover (default: whole recording)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hole-radius", type=int, default=8)
    parser.add_argument("--min-worm-pixels", type=int, default=500)
    parser.add_argument("--short-skeleton", type=float, default=0.75, help="skeleton shorter than this fraction of the whole-worm median marks a coil")
    parser.add_argument("--holes-px", type=int, default=200)
    parser.add_argument("--fragments-px", type=int, default=500)
    parser.add_argument("--gap", type=int, default=3, help="samples of quiet allowed inside one window")
    parser.add_argument("--min-window-samples", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "clip_candidates")
    parser.add_argument("--dataset-root", type=Path, default=CACHE_DIR)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    module = load_segmenter(args.checkpoint, device)
    source = RecordingSource(args.recording, args.dataset_root / "flat_fields")
    field = source.flat_field()
    config = MaskFitConfig()
    started = time.perf_counter()
    rows: list[dict] = []
    peaks: dict[int, np.ndarray] = {}
    try:
        total = source.frame_count
        stop = total if args.frames is None else min(total, args.start + args.frames)
        indices = list(range(args.start, stop, args.stride))
        unreadable = 0
        for chunk_start in range(0, len(indices), args.batch_size):
            chunk = indices[chunk_start : chunk_start + args.batch_size]
            readable = []
            for i in chunk:
                try:
                    readable.append((i, source.read(i)))
                except OSError:
                    # Some recordings hold chunks compressed with an HDF5 filter plugin that is not installed.
                    unreadable += 1
            if not readable:
                continue
            chunk = [i for i, _ in readable]
            raw = np.stack([frame for _, frame in readable])
            corrected = np.stack([np.clip(np.rint(apply_flat_field(f, field, clip=(0.0, 255.0))), 0, 255).astype(np.uint8) for f in raw])
            probability = module.predict_probability_batch(corrected, batch_size=args.batch_size)
            for index, frame, prob in zip(chunk, corrected, probability, strict=True):
                mask, stats = clean_mask(prob, args.threshold, args.hole_radius, device)
                row = {"frame_index": int(index), **stats, "touches_border": False, "bbox_fill": float("nan"), "skeleton_px": float("nan")}
                if stats["worm_pixels"] >= args.min_worm_pixels:
                    yy, xx = np.nonzero(mask)
                    row["touches_border"] = touches_border(mask, 2)
                    row["bbox_fill"] = float(mask.sum()) / float((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1))
                    skeleton = init_from_skeleton(mask, config=config)
                    row["skeleton_px"] = float(skeleton.latent[17]) if skeleton is not None else float("nan")
                rows.append(row)
                peaks[int(index)] = np.stack((frame, boundary(mask).astype(np.uint8)))
            if chunk_start % (args.batch_size * 20) == 0:
                print(f"{chunk[-1]}/{stop} frames scanned", flush=True)
    finally:
        source.close()

    skeleton = np.array([r["skeleton_px"] for r in rows])
    whole = np.array([not r["touches_border"] and np.isfinite(r["skeleton_px"]) for r in rows])
    reference = float(np.nanmedian(skeleton[whole])) if whole.any() else float("nan")
    for r in rows:
        reasons = []
        if np.isfinite(r["skeleton_px"]) and not r["touches_border"] and r["skeleton_px"] < args.short_skeleton * reference:
            reasons.append("short_skeleton")
        if r["pixels_filled"] > args.holes_px:
            reasons.append("holes")
        if r["pixels_outside_largest"] > args.fragments_px:
            reasons.append("fragments")
        if r["touches_border"]:
            reasons.append("border")
        r["reasons"] = reasons
        r["ambiguous"] = bool(set(reasons) & {"short_skeleton", "holes", "fragments"})
    # Windows of ambiguous samples (border-only samples are common and listed separately).
    windows = []
    current: list[dict] = []
    quiet = 0
    for r in rows:
        if r["ambiguous"]:
            current.append(r)
            quiet = 0
        elif current:
            quiet += 1
            if quiet > args.gap:
                windows.append(current)
                current, quiet = [], 0
    if current:
        windows.append(current)
    windows = [w for w in windows if len(w) >= args.min_window_samples]
    candidates = []
    for w in windows:
        counts: dict[str, int] = {}
        for r in w:
            for reason in r["reasons"]:
                counts[reason] = counts.get(reason, 0) + 1
        peak = min(w, key=lambda r: (r["skeleton_px"] / reference if np.isfinite(r["skeleton_px"]) else 9) - 0.001 * r["pixels_filled"])
        candidates.append({
            "start": w[0]["frame_index"], "end": w[-1]["frame_index"], "samples": len(w), "reasons": counts,
            "peak_frame": peak["frame_index"], "peak_skeleton_fraction": float(peak["skeleton_px"] / reference) if np.isfinite(peak["skeleton_px"]) else None,
            "border_samples": sum(r["touches_border"] for r in w),
        })
    border_fraction = float(np.mean([r["touches_border"] for r in rows if r["worm_pixels"] >= args.min_worm_pixels])) if rows else float("nan")
    output = {
        "recording": str(args.recording), "stride": args.stride, "frames_scanned": len(rows), "frames_unreadable": unreadable, "seconds": time.perf_counter() - started,
        "whole_worm_median_skeleton_px": reference, "fraction_touching_border": border_fraction,
        "thresholds": {"short_skeleton": args.short_skeleton, "holes_px": args.holes_px, "fragments_px": args.fragments_px},
        "candidates": candidates, "samples": rows,
    }
    name = args.recording.stem
    (args.output_dir / f"{name}_candidates.json").write_text(json.dumps(output, indent=1, default=float))
    print(f"{name}: {len(rows)} samples, whole-worm skeleton {reference:.0f} px, {100 * border_fraction:.0f}% touch the border, {len(candidates)} candidate windows", flush=True)
    for c in candidates:
        print(f"  frames {c['start']}-{c['end']} ({c['samples']} samples) {c['reasons']} peak {c['peak_frame']}", flush=True)
    # Montage of peak frames.
    tiles = []
    for c in candidates[:16]:
        frame, edge = peaks[c["peak_frame"]]
        rgb = np.repeat(frame[:, :, None], 3, axis=2).copy()
        rgb[edge > 0] = (80, 160, 255)
        image = Image.fromarray(rgb)
        ImageDraw.Draw(image).text((8, 8), f"{name} frame {c['peak_frame']} {','.join(c['reasons'])} [{c['start']}-{c['end']}]", fill=(255, 255, 0))
        image.thumbnail((480, 480))
        tiles.append(image)
    if tiles:
        cols = 4
        rows_n = (len(tiles) + cols - 1) // cols
        canvas = Image.new("RGB", (cols * 484, rows_n * 484), (0, 0, 0))
        for k, tile in enumerate(tiles):
            canvas.paste(tile, ((k % cols) * 484, (k // cols) * 484))
        canvas.save(args.output_dir / f"{name}_candidates.jpg", quality=80, optimize=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
