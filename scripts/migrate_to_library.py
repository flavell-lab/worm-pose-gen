#!/usr/bin/env python3
"""Build the first lab library from the segmentation store, the app corpus and two trained models.

Writes into a fresh ``--out`` directory (the real run into
``/storage/fs/store1/shared/worm-pose-models`` happens after review):

- setup ``nir-flv``: ``/img_nir`` flat-fielded, the frame rate measured from
  the recordings' frame timestamps, the pixel size left null (no recording
  or code states it), the two raw-data roots, and the two models below as
  its defaults (logged with the reason);
- the setup's label collection: the union of the source stores' live labels
  (``--store``, by default ``segmentation_v1`` and the app corpus).  A
  (recording, frame) labeled in more than one place keeps the newest save.
  A recording is named by its file stem, so the same file under two mount
  points (``/store1/shared/all_data_raw/...`` and
  ``/storage/fs/data2/.../data_raw/...``) and the corpus's ``rec_<hash>``
  identities (``identities.json`` maps them back to sample names, which are
  per file) all become one recording.  Each label takes its 33 context
  frames from the store's ``body_fields/`` record (all frames invalid
  without one), the nose landmarks of those frames from the recording
  (only the record's chosen landmark when the recording is unreadable), and
  the human body fields from the record: a rejected record is mask-only, a
  traced one keeps its trace, an accepted or hand-flipped one keeps its
  head end as a manual orientation, and the rest stay automatic;
- dataset ``nir-labels``, with splits per recording: a recording any of whose old labels was a training
  label is train, else test if any was test, else validation.  The old
  store split frames, and both migrated models trained on its training
  frames, so this is the split that holds out, for them too, every frame of
  a validation or test recording (the balancing rule alone put a recording
  with 30 of their training frames into test);
- benchmark ``nir-v1``: the test labels;
- model cards for the segmenter run (``--segmenter-run``, outputs
  ``mask``) and the body-field net run (``--body-run``, all five outputs),
  with copies of their best checkpoints and run records, and in
  ``training/labels.json`` which library labels each run trained on.

``--seed-cache <personal library>`` also stores each label's existing
body-field targets in that library's target cache, so its labels have fit
IoUs without refitting.  The legacy store (Katie's labels) is not migrated,
and the source stores are only read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import getpass
import json
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

from worm_pose_gen.body_fields import TRACE_METHODS
from worm_pose_gen.head_tracking import read_head_tracking
from worm_pose_gen.library import (
    Collection, Libraries, benchmark_labels, create_dataset, labels, make_inputs, write_benchmark, write_model, write_setup,
)
from worm_pose_gen.library.datasets import SPLITS, fingerprint
from worm_pose_gen.library.labels import MAX_LAG, LabelRecord
from worm_pose_gen.library.roots import write_json
from worm_pose_gen.library.setups import log_default
from worm_pose_gen.recording_files import recording_name
from worm_pose_gen.library.targets import describe_body, write_targets
from worm_pose_gen.segmentation_dataset import DEFAULT_DATASET_ROOT
from worm_pose_gen.workspace import DEFAULT_WORKSPACES_ROOT, utc_now


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_ID, DATASET_ID, BENCHMARK_ID = "nir-flv", "nir-labels", "nir-v1"
SETUP, DATASET = f"lab:{SETUP_ID}", f"lab:{DATASET_ID}"
# The NIR camera (FLIR BFS behind the 10x objective): 791 px per mm, the lab's
# FLIR_BFS_PIX_SIZE in BehaviorDataNIR.jl (src/unit.jl).
PIXEL_SIZE_UM = 1000 / 791
# Copied from the source records into the target cache when seeding.
TARGET_ARRAYS = ("centerline_xy", "width_profile", "ap", "overlap", "head_xy", "tail_xy", "diameter_px", "nose_xy")
TARGET_META = ("has_body", "fit_method", "orientation", "nose_offset", "fit_iou", "overlap_px", "orientation_margin",
               "head_off_camera", "tail_off_camera", "visible_share", "chain_anchor_offset", "independent_fit_iou")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="the new lab library (must not exist or be empty)")
    parser.add_argument("--store", type=Path, action="append", dest="stores",
                        help=f"a segmentation store or app corpus to migrate (repeatable; default {DEFAULT_DATASET_ROOT} and {DEFAULT_WORKSPACES_ROOT / 'corpus'})")
    parser.add_argument("--segmenter-run", type=Path, default=PROJECT_ROOT / "checkpoints" / "segmenter" / "runs" / "2026-10-06T16-06-01Z_r4-hand284")
    parser.add_argument("--segmenter-name", default="nir-hand284")
    parser.add_argument("--body-run", type=Path, required=True, help="a run directory of scripts/train_body_net.py")
    parser.add_argument("--body-name", default=None, help="the body net's model id (default nir-body-lags<number of lags>)")
    parser.add_argument("--seed-cache", type=Path, default=None, help="a personal library whose target cache to fill from the old records")
    parser.add_argument("--author", default=getpass.getuser())
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- sources


def live_samples(stores: list[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    """The newest live sample of every (recording, frame) across the stores, with its store and the older ones (``others``)."""

    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for store in stores:
        index_path = store / "index.json"
        if not index_path.exists():
            print(f"{store}: no index.json, nothing to migrate", flush=True)
            continue
        index = json.loads(index_path.read_text())
        print(f"{store}: {len(index)} live samples", flush=True)
        for sample_id, record in index.items():
            key = (recording_name(record["source_path"]), int(record["frame_index"]))
            candidate = {**record, "sample_id": sample_id, "store": store, "others": []}
            previous = chosen.get(key)
            if previous is None:
                chosen[key] = candidate
            elif candidate["saved_at"] >= previous["saved_at"]:
                candidate["others"] = [*previous.pop("others"), previous]
                chosen[key] = candidate
            else:
                previous["others"].append(candidate)
    return chosen


def body_record(sample: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    path = sample["store"] / "body_fields" / f"{sample['sample_id']}.npz"
    if not path.exists():
        return None
    with np.load(path) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files if not name.startswith("proposal_")}
    return arrays, json.loads(str(arrays.pop("meta")))


def nose_landmarks(paths: list[str], frame: int, shape: tuple[int, int], fallback: tuple[dict, dict] | None) -> tuple[np.ndarray, np.ndarray, str]:
    """The nose landmark of each context frame from the first readable copy of the recording, else the record's chosen one."""

    offsets = np.arange(-MAX_LAG, MAX_LAG + 1)
    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                count = int(handle["/img_nir"].shape[0])
        except (OSError, KeyError):
            continue
        frames = offsets + frame
        inside = (frames >= 0) & (frames < count)
        tracking = read_head_tracking(path, np.clip(frames, 0, count - 1), shape)
        if tracking.provenance.get("status") == "available":
            valid = tracking.valid & inside
            return np.where(valid[:, None], tracking.xy, np.nan).astype(np.float64), valid, "recording"
    xy, valid = np.full((len(offsets), 2), np.nan), np.zeros(len(offsets), bool)
    if fallback is not None and "nose_xy" in fallback[0] and fallback[1].get("nose_offset") is not None:
        position = MAX_LAG + int(fallback[1]["nose_offset"])
        xy[position], valid[position] = fallback[0]["nose_xy"], True
        return xy, valid, "record"
    return xy, valid, "none"


def human_fields(arrays: dict[str, np.ndarray], meta: dict[str, Any], current: bool) -> dict[str, Any]:
    """The decisions a person made in the old Body fields screen, as label fields.

    A record built for an older mask keeps only its trace, as a rebuild did.
    """

    if "trace_xy" in arrays and meta.get("fit_method") in TRACE_METHODS:
        return {"trace_xy": arrays["trace_xy"], "mask_only": current and meta.get("review") == "rejected"}
    if not current or not meta.get("has_body"):
        return {"mask_only": False}
    if meta.get("review") == "rejected":
        return {"mask_only": True}
    if meta.get("review") == "accepted" or meta.get("orientation") == "manual":
        return {"orientation": "manual", "head_xy": arrays["centerline_xy"][0], "mask_only": False}
    return {"mask_only": False}


def measured_fps(paths: list[str]) -> float | None:
    """Frames per second from the saved frames' camera timestamps (nanoseconds) of the first readable recording."""

    for path in paths:
        try:
            with h5py.File(path, "r") as handle:
                group = handle["/img_metadata"]
                stamps = np.asarray(group["img_timestamp"][:4000], dtype=np.float64)
                saved = (np.asarray(group["q_iter_save"][:4000]) == 1) & (np.asarray(group["q_recording"][:4000]) == 1)
        except (OSError, KeyError):
            continue
        steps = np.diff(stamps[saved])
        if len(steps):
            return round(1e9 / float(np.median(steps)), 2)
    return None


# --------------------------------------------------------------------------- splits


def recording_splits(samples: dict[tuple[str, int], dict[str, Any]]) -> dict[str, str]:
    """Each recording's split: train if any of its old labels was trained on, else test if any was test, else val.

    The old store split frames, not recordings, and both migrated models
    trained on its training frames; a recording any of whose frames they
    trained on cannot hold out anything for them.
    """

    old: dict[str, set[str]] = defaultdict(set)
    for (recording, _), sample in samples.items():
        old[recording].add(sample["split"])
    return {recording: "train" if "train" in splits else "test" if "test" in splits else "val" for recording, splits in sorted(old.items())}


# --------------------------------------------------------------------------- models


def run_labels(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """The run's split membership, from either trainer's run.json."""

    splits = run["splits"]
    if "train" not in splits:  # train_segmenter.py keys it by store
        merged: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for by_split in splits.values():
            for split, entries in by_split.items():
                merged[split].extend(entries)
        splits = merged
    return dict(splits)


def model_labels(run: dict[str, Any], migrated: dict[str, LabelRecord], sources: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Which library labels a run trained on, and the card's ``trained_on`` entry.

    The library keeps only each frame's newest revision, so a label revised
    after the run is named by its current revision and counted.
    """

    used, rows = [], []
    for split, entries in run_labels(run).items():
        for entry in entries:
            record = migrated.get(entry["sample_id"])
            rows.append({"old_sample_id": entry["sample_id"], "old_revision": entry.get("revision"), "old_split": split,
                         "label": None if record is None else record.identity, "split_now": None if record is None else record.split,
                         "revised_since": record is not None and sources[entry["sample_id"]]["revision"] != entry.get("revision")})
            if record is not None and split in ("train", "val"):
                used.append(record)
    trained = [r for r in rows if r["old_split"] in ("train", "val")]
    entry = {
        "dataset": DATASET, "fingerprint": fingerprint(used),
        "counts": {split: sum(r["old_split"] == split for r in rows) for split in SPLITS},
        "recordings": len({r.recording for r in used}),
        "before_library": {
            "splits": "per frame (segmentation_v1)",
            "missing_from_library": sum(r["label"] is None for r in rows),
            "revised_since": sum(r["revised_since"] for r in trained),
            "trained_on_now_in": dict(Counter(r["split_now"] for r in trained)),
        },
    }
    return rows, entry


def migrate_model(out: Path, model_id: str, run_dir: Path, *, kind: str, outputs: list[str], fps: float | None, author: str,
                  migrated: dict[str, LabelRecord], sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    run = json.loads((run_dir / "run.json").read_text())
    lags = [int(lag) for lag in run.get("lags", [])]
    rows, trained_on = model_labels(run, migrated, sources)
    hparams = {k: v for k, v in run["args"].items()
               if k not in ("dataset_root", "checkpoint_dir", "name", "num_workers", "promote", "no_promote", "train_labels", "init")}
    test = {k.removeprefix("test_"): v for k, v in (run.get("test") or {}).items()}
    card = {
        "name": model_id, "kind": kind, "setup": SETUP, "inputs": make_inputs(lags, fps=fps, pixel_size_um=PIXEL_SIZE_UM),
        "outputs": outputs, "trained_on": [trained_on], "parent": None, "hparams": hparams, "author": author,
        "created_at": run.get("finished_at") or run.get("started_at"),
        "notes": (f"Run {run['name']} ({run_dir.name}), trained before the library on segmentation_v1 with per-frame "
                  f"splits from ImageNet weights; best epoch {run.get('best_epoch')}, validation loss {run.get('best_val_loss'):.4f}; "
                  f"test IoU on its old split {test.get('iou', float('nan')):.3f}. Migrated by scripts/migrate_to_library.py."),
    }
    files = {name: run_dir / name for name in ("run.json", "hparams.yaml", "metrics.csv") if (run_dir / name).exists()}
    write_model(out, model_id, card, run_dir / "best.ckpt", training_files=files, training_records={"labels.json": rows})
    return trained_on


# --------------------------------------------------------------------------- main


def seed_targets(seed: Libraries, record: LabelRecord, arrays: dict[str, np.ndarray], meta: dict[str, Any], label_fields: dict[str, Any],
                 source: Path, builder: str) -> None:
    """The old record's targets as the cache entry of the migrated revision: they were built from the same mask, with
    the segmenter that becomes the setup's mask default, so they are filed under that builder."""

    entry = {key: meta[key] for key in TARGET_META if key in meta}
    entry.setdefault("fit_method", "independent")
    if label_fields.get("orientation") == "manual" and "trace_xy" not in label_fields:
        entry["orientation"] = "manual"
    entry.update(label=record.identity, builder=builder, built_at=utc_now(), fit_preset=meta.get("fit_preset"), max_lag=meta.get("max_lag"), seeded_from=str(source))
    targets = {name: arrays[name] for name in TARGET_ARRAYS if name in arrays} if meta.get("has_body") else {}
    if meta.get("has_body"):
        describe_body(entry, arrays["centerline_xy"], arrays["width_profile"], (record.height, record.width))
    write_targets(seed, record.sha256, builder, entry, targets)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty; the migration writes a fresh library")
    stores = [p.resolve() for p in (args.stores or [DEFAULT_DATASET_ROOT, DEFAULT_WORKSPACES_ROOT / "corpus"])]
    samples = live_samples(stores)
    if not samples:
        sys.exit("no labels to migrate")
    paths_of: dict[str, list[str]] = defaultdict(list)
    for (recording, _), sample in sorted(samples.items(), key=lambda item: item[1]["saved_at"]):
        if sample["source_path"] not in paths_of[recording]:
            paths_of[recording].append(sample["source_path"])

    # The same libraries object reads the new root as the lab library; ``personal`` is written only with --seed-cache.
    libraries = Libraries(lab=out, personal=(args.seed_cache or out).resolve())
    body_run = args.body_run.resolve()
    body_lags = json.loads((body_run / "run.json").read_text()).get("lags", [])
    body_id = args.body_name or (f"nir-body-lags{len(body_lags)}" if body_lags else "nir-body-nolags")
    fps = measured_fps([p for paths in paths_of.values() for p in paths])
    print(f"frame rate {fps} fps (from the frame timestamps)", flush=True)
    write_setup(
        out, SETUP_ID, name="NIR behaviour camera (flv rigs)",
        description="The 850 nm NIR camera of the lab's tracking rigs; 732x968 frames in /img_nir.",
        fps=fps, pixel_size_um=PIXEL_SIZE_UM,
        defaults={"mask": f"lab:{args.segmenter_name}", "body": f"lab:{body_id}"},
    )
    log_default(out, SETUP, "mask", f"lab:{args.segmenter_name}", previous=None, who=args.author,
                reason=f"migrated: {args.segmenter_run.name} was the promoted segmenter (checkpoints/segmenter/promotions.jsonl)")
    log_default(out, SETUP, "body", f"lab:{body_id}", previous=None, who=args.author,
                reason=f"migrated: the body-field net {body_run.name} given to the migration")

    splits = recording_splits(samples)
    if not {"val", "test"} <= set(splits.values()):
        sys.exit(f"no recording is held out entirely in validation or test: {splits}")
    dataset = create_dataset(libraries, DATASET_ID, setup=SETUP, name="NIR hand labels", splits=splits,
                             description="Hand labels of segmentation_v1 and the app corpus, migrated " + utc_now()[:10], author=args.author, scope="lab")
    collection = Collection(libraries, SETUP)
    print(f"recording splits: {splits}", flush=True)
    order = sorted(samples)

    migrated: dict[str, LabelRecord] = {}
    sources: dict[str, dict[str, Any]] = {}
    nose_sources: Counter[str] = Counter()
    for number, key in enumerate(order, 1):
        sample = samples[key]
        recording, frame = key
        with np.load(sample["store"] / "samples" / f"{sample['sample_id']}.npz") as archive:
            image, image_raw, mask = (np.asarray(archive[name]) for name in ("image", "image_raw", "mask"))
        # The human fields come only from the sample's own record; the context
        # frames may come from another store's record of the same frame.
        found = body_record(sample)
        fields, saved_at, current = {"mask_only": False}, sample["saved_at"], False
        if found is not None:
            current = found[1].get("mask_revision") == sample["revision"]
            fields = human_fields(*found, current)
            saved_at = max(sample["saved_at"], found[1].get("reviewed_at") or "")
        with_context = next((r for r in [found, *map(body_record, sample["others"])]
                             if r is not None and np.array_equal(r[0]["context"][MAX_LAG], image)), None)
        if with_context is not None:
            context, context_valid = with_context[0]["context"], with_context[0]["context_valid"].astype(bool)
        else:
            context = np.repeat(image[None], 2 * MAX_LAG + 1, axis=0)
            context_valid = np.arange(2 * MAX_LAG + 1) == MAX_LAG
        nose_xy, nose_valid, nose_source = nose_landmarks(paths_of[recording], frame, image.shape, with_context)
        nose_sources[nose_source] += 1
        record = collection.save(
            recording=recording, frame=frame, image=image, image_raw=image_raw, mask=mask, context=context,
            context_valid=context_valid, nose_xy=nose_xy, nose_valid=nose_valid, origin="migrated", author=args.author,
            saved_at=saved_at, source_path=sample["source_path"], dataset_path=sample.get("dataset_path") or "/img_nir",
            extra_meta={"migrated_from": {
                "store": str(sample["store"]), "sample_id": sample["sample_id"], "revision": sample["revision"],
                "label_source": sample["label_source"], "split": sample["split"],
                "body_review": None if found is None else found[1].get("review", "unreviewed"),
                "body_fit_method": None if found is None else found[1].get("fit_method"),
                "nose_landmarks": nose_source,
            }},
            scope="lab", **fields,
        )
        record = replace(record, split=splits[recording])
        migrated[sample["sample_id"]], sources[sample["sample_id"]] = record, sample
        if args.seed_cache is not None and found is not None and current:
            seed_targets(libraries, record, found[0], found[1], fields, sample["store"] / "body_fields" / f"{sample['sample_id']}.npz",
                         f"lab:{args.segmenter_name}")
        if number % 25 == 0 or number == len(order):
            print(f"{number}/{len(order)} labels", flush=True)

    records = labels(libraries, DATASET)
    test_recordings = {r.recording for r in records if r.split == "test"}
    val_recordings = {r.recording for r in records if r.split == "val"}
    if not test_recordings or not val_recordings:
        sys.exit("the split left no validation or no test recording; the library is incomplete")
    write_benchmark(out, BENCHMARK_ID, setup=SETUP, dataset=DATASET, records=records, author=args.author,
                    description="The test recordings of nir-labels at migration.")
    write_json(dataset.root / "migration.json", {
        "migrated_at": utc_now(), "stores": [str(s) for s in stores], "author": args.author,
        "split_rule": "train if any old label of the recording was train, else test if any was test, else val",
        "splits": splits, "nose_landmarks": dict(nose_sources),
        "labels": {sample_id: record.identity for sample_id, record in sorted(migrated.items())},
    })

    seg = migrate_model(out, args.segmenter_name, args.segmenter_run.resolve(), kind="segmenter", outputs=["mask"], fps=fps,
                        author=args.author, migrated=migrated, sources=sources)
    body = migrate_model(out, body_id, body_run, kind="body_net", outputs=["mask", "ap", "head", "tail", "overlap"], fps=fps,
                         author=args.author, migrated=migrated, sources=sources)

    print(f"\n{out}")
    print(f"labels {len(records)}: " + ", ".join(f"{s} {sum(r.split == s for r in records)}" for s in SPLITS))
    print("status " + json.dumps(dict(Counter(r.status for r in records))) + f"; nose landmarks {dict(nose_sources)}")
    for recording in sorted({r.recording for r in records}):
        group = [r for r in records if r.recording == recording]
        print(f"  {recording:16s} {group[0].split:5s} {len(group):3d}  " + json.dumps(dict(Counter(r.status for r in group))))
    print(f"benchmark {BENCHMARK_ID}: {len(benchmark_labels(libraries, f'lab:{BENCHMARK_ID}'))} labels from {len(test_recordings)} recordings")
    for name, entry in ((args.segmenter_name, seg), (body_id, body)):
        print(f"model {name}: trained on {entry['counts']}; before the library: {json.dumps(entry['before_library'])}")
    if args.seed_cache is not None:
        print(f"target cache seeded in {libraries.personal}")


if __name__ == "__main__":
    main()
