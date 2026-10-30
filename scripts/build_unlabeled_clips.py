#!/usr/bin/env python3
"""Build a store of unlabeled video segments for semi-supervised body-field training.

Draws ``--per-recording`` segments of ``--length`` frames from every readable
recording of the setup (its recording folders and the recordings registered to
it), skipping every frame :func:`temporal_eval.excluded` rejects and every
recording the dataset puts in ``val`` or ``test`` (so early stopping on the
validation split stays honest), flat-fields them as the pipeline does and
saves them as uint8 (:mod:`worm_pose_gen.unlabeled_clips`).  Example::

    scripts/build_unlabeled_clips.py --setup lab:nir-flv --dataset mine:on-food-finetune \\
        --output <data dir>/runs/unlabeled_clips --per-recording 10 --length 64 --workers 32
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

from worm_pose_gen import library
from worm_pose_gen.model_eval import add_library_arguments, libraries_from
from worm_pose_gen.recordings import find_recordings
from worm_pose_gen.settings import CACHE_DIR
from worm_pose_gen.unlabeled_clips import build_store


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_library_arguments(parser)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--dataset", required=True, help="the dataset trained on; its val and test recordings are left out")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-recording", type=int, default=10)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    started = time.monotonic()
    libraries = libraries_from(args)
    setup = library.get_setup(libraries, args.setup)
    dataset = library.Dataset(libraries, args.dataset)
    held_out = sorted(name for name, split in dataset.splits().items() if split in ("val", "test"))
    recordings = [str(p) for p in find_recordings(library.recording_sources(libraries, args.setup))]
    print(f"{len(recordings)} recording files; leaving out {held_out}", flush=True)
    written = build_store(
        args.output, recordings, per_recording=args.per_recording, length=args.length, seed=args.seed,
        excluded_recordings=held_out, dataset=setup.video.get("dataset_path", "/img_nir"), workers=args.workers,
        flat_field_cache=CACHE_DIR / "flat_fields", log=lambda m: print(m, flush=True),
    )
    print(f"{len(written)} segments from {len({s.recording for s in written})} recordings "
          f"in {(time.monotonic() - started) / 60:.1f} min", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
