#!/usr/bin/env python3
"""Fixed runtime and quality benchmark of the pose pipeline with the body-field network.

Three 300-frame clips are fit end to end by ``scripts/fit_recording.py`` with
``--body-net checkpoints/body_net/best.ckpt``, propagation and the track pass
on (the defaults), no video and no residual images, and each recording's
prior pinned to its cached file (no bootstrap):

``spiral_0131``    a tight spiral (``eval_sets/sequence_eval_set.json``)
``edge_0528``      the body at the image border with pieces breaking off (same manifest)
``ordinary_0131``  2024-01-31-02 frames 4350-4649: ordinary crawling, fully in
                   view, outside every window of the mask scan
                   (from ``scripts/find_sequence_clips.py``)

Per run: wall seconds, each stage's seconds and ms per frame (the run's
``summary.json``), and quality: frames with IoU below 0.9, median IoU, and
``propagation.continuity_summary`` (pose jumps over a width, length jumps
over 3%, in-view changes).  Then the held-out benchmark,
``scripts/evaluate_field_fitting.py --variants both``: within half a width,
A-P error mean, head-correct rate and IoU median for the traced and the
accepted records.  The JSON report gives, over the ``--repeat`` repeats, the
minimum of every stage's clip-set ms per frame, the minimum clip-set total
(``timing.total_ms_per_frame``, the sum of the timed stages: the KEEP
metric), the minimum overhead outside the stages (imports, model loading,
compilation, saving), and the worst quality of any repeat.

Timing protocol.  Run on flv-c3 GPU 2 (RTX A5500) with
``CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2``, the baseline and
every candidate alike.  While it times, nothing else may run on that GPU and
nothing heavy on the CPU (each run starts eight start-building processes):
no other fit, training or benchmark on c3, and a 1-minute load average near
zero (recorded per run as ``load_1min_at_start``, with the processes on the
GPU as ``gpu_processes_at_start``).  Use ``--repeat 2``.  Repeats are
interleaved (every clip once, then again), so a slow first process, such as
one filling the compiler cache after a code change, is discarded by the
minimum.

Noise, measured with the same code on GPU 2 in two invocations of
``--repeat 2`` (2026-10-07): the four single-repeat clip-set totals spanned
1.1% (807-816 ms per frame) and single clips up to 2.1%; the reported
minimum differed by 0.2% between the invocations, the large stages' minima
by at most 0.3%; the overhead outside the stages by up to 1.0 s; quality and
the held-out benchmark were bit-identical.  Twice the noise (2.2%) is below
3%, so the floor sets the threshold.  Quality is deterministic, so the
quality tolerances need no widening for noise: any change in them is a real
change of the fits.

KEEP rule for a runtime optimization (``--baseline`` applies it and writes
the verdict into the report).  KEEP only if all hold:

- ``timing.total_ms_per_frame`` drops by more than 3% of the baseline's
  (max(3%, twice the noise));
- the overhead outside the stages rises by at most 2 s over the three runs
  (twice its noise; work moved out of the timed stages is not a speedup);
- per clip, worst repeat against worst repeat: frames with IoU below 0.9
  rise by at most 2, pose jumps by at most 1, length jumps by at most 3;
- held-out, traced and accepted records: within half a width does not drop,
  A-P error mean rises by at most 0.005, head-correct rate stays 1.0.

Command (about 28 minutes; runs go to a new directory under ``--runs-dir``):

    ssh -o BatchMode=yes flv-c3 "cd /home/alex/git/worm-pose-gen; and env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \\
        uv run --no-sync python scripts/benchmark_pose_runtime.py --repeat 2 --baseline <data dir>/runs/runtime_benchmark/<run>/runtime.json"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np

from worm_pose_gen.pipeline import DEFAULT_PRIOR_CACHE
from worm_pose_gen.settings import RUNS_DIR
from worm_pose_gen.propagation import continuity_summary
from worm_pose_gen.recording_files import recording_name
from worm_pose_gen.run_records import git_revision, timestamp_slug, utc_now


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "eval_sets" / "sequence_eval_set.json"
BODY_NET = PROJECT_ROOT / "checkpoints" / "body_net" / "best.ckpt"
MANIFEST_CLIPS = ("spiral_0131", "edge_0528")
ORDINARY = {"name": "ordinary_0131", "recording": "/store1/shared/all_data_raw/prj_aversion/2024-01-31/2024-01-31-02.h5", "start": 4350, "frames": 300}
QUALITY_KEYS = ("frames_iou_below_0.9", "iou_median", "pose_jumps", "length_jumps", "in_view_changes")
HELDOUT_KEYS = ("n", "within_half_width", "ap_error_mean", "head_correct", "iou_median")

# The KEEP rule (module docstring).
MIN_SPEEDUP = 0.03
MAX_OVERHEAD_RISE_S = 2.0
MAX_CLIP_RISE = {"frames_iou_below_0.9": 2, "pose_jumps": 1, "length_jumps": 3}
MAX_AP_ERROR_RISE = 0.005


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeat", type=int, default=2, help="runs per clip; every stage reports its minimum")
    parser.add_argument("--baseline", type=Path, default=None, help="baseline report to apply the KEEP rule against")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR / "runtime_benchmark", help="fit runs and the held-out output go to a new directory here")
    parser.add_argument("--output", type=Path, default=None, help="report path (default: benchmark.json in the new runs directory)")
    return parser.parse_args()


def benchmark_clips() -> list[dict[str, Any]]:
    manifest = {clip["name"]: clip for clip in json.loads(MANIFEST.read_text())["clips"]}
    clips = [{k: manifest[name][k] for k in ORDINARY} for name in MANIFEST_CLIPS] + [ORDINARY]
    for clip in clips:
        # The cached prior of the fit's six width coefficients, as fit_recording resolves it by default.
        prior = DEFAULT_PRIOR_CACHE / f"{recording_name(clip['recording'])}_k6.json"
        clip.update(prior_file=str(prior), prior_sha256=hashlib.sha256(prior.read_bytes()).hexdigest())
    return clips


def nvidia_smi(*query: str) -> list[str] | None:
    """``nvidia-smi`` rows for the visible GPU (indices in PCI bus order), or ``None`` without one."""

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not gpu:
        return None
    completed = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader", "-i", gpu], capture_output=True, text=True)
    return [line for line in completed.stdout.splitlines() if line.strip()] if completed.returncode == 0 else None


def fit_clip(clip: dict[str, Any], name: str, runs_dir: Path) -> tuple[Path, float]:
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "fit_recording.py"),
        "--recording", clip["recording"], "--start", str(clip["start"]), "--frames", str(clip["frames"]),
        "--body-net", str(BODY_NET), "--prior-file", clip["prior_file"], "--residual-frames", "0",
        "--output-dir", str(runs_dir), "--name", name,
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    wall = time.perf_counter() - started
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr, flush=True)
        raise SystemExit(f"fit_recording failed on {name} (exit {completed.returncode}); its stderr tail is above")
    (run_dir,) = runs_dir.glob(f"*_{name}")
    (run_dir / "fit_recording.log").write_text(completed.stdout)
    return run_dir, wall


def clip_quality(run_dir: Path) -> dict[str, Any]:
    with np.load(run_dir / "poses.npz") as arrays:
        fitted = arrays["fitted"].astype(bool)
        iou = arrays["iou"][fitted]
        continuity = continuity_summary(arrays)
    return {
        "frames_fitted": int(fitted.sum()), "frames_iou_below_0.9": int((iou < 0.9).sum()), "iou_median": float(np.median(iou)),
        "pose_jumps": continuity["pose_jumps_over_width"], "length_jumps": continuity["length_jumps_over_fraction"],
        "in_view_changes": continuity["in_view_changes"],
    }


def heldout(out_dir: Path) -> dict[str, dict[str, float]]:
    command = [sys.executable, str(PROJECT_ROOT / "scripts" / "evaluate_field_fitting.py"), "--variants", "both", "--checkpoint", str(BODY_NET), "--out", str(out_dir)]
    completed = subprocess.run(command, capture_output=True, text=True, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        print(completed.stderr[-4000:], file=sys.stderr, flush=True)
        raise SystemExit(f"evaluate_field_fitting failed (exit {completed.returncode}); its stderr tail is above")
    summary = json.loads((out_dir / "summary.json").read_text())["summary"]
    return {group: {k: summary[f"{group}/both"][k] for k in HELDOUT_KEYS} for group in ("traced", "accepted")}


def timing(runs: list[dict[str, Any]], clips: list[dict[str, Any]], repeat: int) -> dict[str, Any]:
    """Minimum over repeats of the clip-set ms per frame of every stage, of the total, and of the overhead outside the stages."""

    frames = sum(clip["frames"] for clip in clips)
    by_repeat = [[run for run in runs if run["repeat"] == r] for r in range(repeat)]
    stages = list(runs[0]["seconds"])

    def ms(seconds: float) -> float:
        return 1000.0 * seconds / frames

    totals = [ms(sum(run["staged_seconds"] for run in group)) for group in by_repeat]
    per_clip = {}
    for clip in clips:
        mine = [run for run in runs if run["clip"] == clip["name"]]
        per_clip[clip["name"]] = {
            "ms_per_frame": {s: min(1000.0 * run["seconds"][s] / clip["frames"] for run in mine) for s in stages},
            "total_ms_per_frame": min(1000.0 * run["staged_seconds"] / clip["frames"] for run in mine),
            "wall_seconds": min(run["wall_seconds"] for run in mine),
        }
    return {
        "frames": frames,
        "ms_per_frame": {s: min(ms(sum(run["seconds"][s] for run in group)) for group in by_repeat) for s in stages},
        "total_ms_per_frame": min(totals),
        "total_ms_per_frame_by_repeat": totals,
        "repeat_spread": (max(totals) - min(totals)) / min(totals),
        "wall_seconds": min(sum(run["wall_seconds"] for run in group) for group in by_repeat),
        "overhead_seconds": min(sum(run["wall_seconds"] - run["staged_seconds"] for run in group) for group in by_repeat),
        "clips": per_clip,
    }


def worst_quality(runs: list[dict[str, Any]], clips: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per clip, the worst value of any repeat (most frames or jumps, lowest median IoU)."""

    out = {}
    for clip in clips:
        mine = [run["quality"] for run in runs if run["clip"] == clip["name"]]
        out[clip["name"]] = {k: (min if k == "iou_median" else max)(q[k] for q in mine) for k in QUALITY_KEYS}
    return out


def keep_verdict(report: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    """The KEEP rule of the module docstring against a baseline report."""

    failures = []
    new, base = report["timing"]["total_ms_per_frame"], baseline["timing"]["total_ms_per_frame"]
    speedup = 1.0 - new / base
    if speedup <= MIN_SPEEDUP:
        failures.append(f"total {new:.1f} vs {base:.1f} ms/frame ({speedup:+.1%}), needs a drop over {MIN_SPEEDUP:.0%}")
    overhead_rise = report["timing"]["overhead_seconds"] - baseline["timing"]["overhead_seconds"]
    if overhead_rise > MAX_OVERHEAD_RISE_S:
        failures.append(f"overhead outside the stages rose {overhead_rise:.1f} s (limit {MAX_OVERHEAD_RISE_S:.0f} s)")
    for clip, quality in report["quality"].items():
        for key, limit in MAX_CLIP_RISE.items():
            rise = quality[key] - baseline["quality"][clip][key]
            if rise > limit:
                failures.append(f"{clip}: {key} rose by {rise} (limit {limit})")
    for group, entry in report["heldout"].items():
        reference = baseline["heldout"][group]
        if entry["within_half_width"] < reference["within_half_width"]:
            failures.append(f"held-out {group}: within half a width {entry['within_half_width']} < {reference['within_half_width']}")
        if entry["ap_error_mean"] - reference["ap_error_mean"] > MAX_AP_ERROR_RISE:
            failures.append(f"held-out {group}: A-P error mean {entry['ap_error_mean']:.4f} vs {reference['ap_error_mean']:.4f}")
        if entry["head_correct"] < 1.0:
            failures.append(f"held-out {group}: head-correct rate {entry['head_correct']:.3f}")
    return {"keep": not failures, "speedup": speedup, "overhead_rise_seconds": overhead_rise, "failures": failures}


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    baseline = None if args.baseline is None else json.loads(args.baseline.read_text())
    clips = benchmark_clips()
    started = utc_now()
    runs_dir = args.runs_dir / timestamp_slug(started)
    runs_dir.mkdir(parents=True, exist_ok=False)
    output = args.output or runs_dir / "benchmark.json"
    diff = subprocess.run(["git", "diff", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, check=True).stdout
    report: dict[str, Any] = {
        "generated_at": started,
        "git": {**git_revision(PROJECT_ROOT), "diff_sha256": hashlib.sha256(diff).hexdigest()},
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": nvidia_smi("--query-gpu=name"),
        "repeat": args.repeat,
        "body_net": str(BODY_NET.relative_to(PROJECT_ROOT)),
        "clips": clips,
        "runs_dir": str(runs_dir),
        "runs": [],
    }
    for repeat in range(args.repeat):
        for clip in clips:
            load, gpu_processes = os.getloadavg()[0], nvidia_smi("--query-compute-apps=pid")
            run_dir, wall = fit_clip(clip, f"{clip['name']}_r{repeat}", runs_dir)
            summary = json.loads((run_dir / "summary.json").read_text())
            run = {
                "clip": clip["name"], "repeat": repeat, "run": str(run_dir), "load_1min_at_start": load,
                "gpu_processes_at_start": None if gpu_processes is None else len(gpu_processes),
                "wall_seconds": wall, "staged_seconds": sum(summary["seconds"].values()), "seconds": summary["seconds"],
                "total_ms_per_frame": summary["total_ms_per_frame"], "quality": clip_quality(run_dir),
            }
            report["runs"].append(run)
            output.write_text(json.dumps(report, indent=1))
            print(
                f"repeat {repeat} {clip['name']:>14s}: {run['total_ms_per_frame']:.0f} ms/frame staged, {wall:.0f} s wall,"
                f" load {load:.1f}, quality {run['quality']}",
                flush=True,
            )
    report["heldout"] = heldout(runs_dir / "heldout")
    report["timing"] = timing(report["runs"], clips, args.repeat)
    report["quality"] = worst_quality(report["runs"], clips)
    if baseline is not None:
        report["verdict"] = {"baseline": str(args.baseline), **keep_verdict(report, baseline)}
    report["finished_at"] = utc_now()
    output.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("timing", "quality", "heldout", *(("verdict",) if baseline is not None else ()))}, indent=1))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
