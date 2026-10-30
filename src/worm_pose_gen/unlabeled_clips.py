"""A store of short unlabeled video segments for semi-supervised training of the body-field network.

Semi-supervised training (:mod:`semi_supervised`) needs windows of a few
consecutive frames, each with the ``MAX_LAG`` frames of context its
difference channels read, from as many recordings as possible.  Reading them
from the raw recordings at training time is slow (NFS, chunked datasets), so
a fixed sample of contiguous segments is flat-fielded once, exactly as the
pipeline does it (:meth:`pipeline.Frames.corrected`), and kept as uint8
arrays:

    <root>/index.json                       the segments and how they were drawn
    <root>/segments/<recording>_<start>.npy  [length, H, W] uint8
    <root>/flat_fields/<recording>.npz       the flat fields used (pipeline cache format)

Segments are spread over the recordings (a few per recording rather than
many from one, for diversity of animals, days and illumination) and never
contain a frame that :func:`temporal_eval.excluded` rejects (test and ood
recordings, evaluation clips with their margin) or a frame of a recording the
caller excludes (the validation split, so early stopping stays honest).  A
recording that cannot be read (the store1 HDF5 corruption) is skipped.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .frame_reader import open_frames
from .temporal_context import MAX_LAG


INDEX = "index.json"
SEGMENTS = "segments"
FLAT_FIELDS = "flat_fields"


@dataclass(frozen=True)
class Segment:
    recording: str   # path of the HDF5 file
    start: int       # first frame of the segment in the recording
    length: int

    @property
    def name(self) -> str:
        return f"{Path(self.recording).stem}_{self.start:06d}"


def segment_allowed(
    recording: str, start: int, length: int, excluded: Callable[[str, int], bool], excluded_recordings: Iterable[str] = (),
) -> bool:
    """Whether no frame of ``start .. start + length - 1`` is excluded (``excluded(recording, frame)`` checks every frame)."""

    if Path(recording).stem in set(excluded_recordings):
        return False
    return not any(excluded(recording, frame) for frame in range(start, start + length))


def plan_segments(
    recordings: Sequence[tuple[str, int]], *, per_recording: int, length: int, seed: int,
    excluded: Callable[[str, int], bool], excluded_recordings: Iterable[str] = (), attempts: int = 50,
) -> list[Segment]:
    """``per_recording`` non-overlapping segments of ``length`` frames drawn uniformly from each ``(path, frame count)``.

    Segments that touch an excluded frame are redrawn (up to ``attempts``
    draws per segment); a recording with no room gets fewer.  Of two files
    with the same name only the first is used.
    """

    excluded_recordings = set(excluded_recordings)
    rng = np.random.default_rng(seed)
    chosen: list[Segment] = []
    seen: set[str] = set()
    for path, total in recordings:
        # A recording copied under two roots is one recording.
        if Path(path).stem in excluded_recordings | seen or total < length:
            continue
        seen.add(Path(path).stem)
        starts: list[int] = []
        for _ in range(per_recording * attempts):
            if len(starts) == per_recording:
                break
            start = int(rng.integers(0, total - length + 1))
            if any(abs(start - other) < length for other in starts):
                continue
            if segment_allowed(path, start, length, excluded):
                starts.append(start)
        chosen.extend(Segment(str(path), start, length) for start in sorted(starts))
    return chosen


def _write_segment(segment: Segment, root: str, dataset: str) -> tuple[Segment, str | None]:
    """Flat-field and save one segment; returns the error text instead of raising (unreadable chunks)."""

    from .pipeline import Frames

    target = Path(root) / SEGMENTS / f"{segment.name}.npy"
    if target.exists():
        return segment, None
    try:
        source = Frames(Path(segment.recording), dataset_root=root, dataset=dataset)
        frames, _, _ = source.corrected(list(range(segment.start, segment.start + segment.length)))
    except (OSError, KeyError, ValueError) as error:
        return segment, f"{type(error).__name__}: {error}"
    temporary = target.with_name("." + target.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.save(handle, np.ascontiguousarray(frames, dtype=np.uint8))
    temporary.replace(target)
    return segment, None


def frame_count(path: str, dataset: str) -> int | None:
    """The recording's frame count, or ``None`` when it cannot be opened."""

    try:
        with open_frames(path, dataset) as frames:
            # A corrupted file can open and still fail on its first frame.
            frames[0]
            return frames.frame_count
    except (OSError, KeyError, ValueError):
        return None


def build_store(
    root: Path, recordings: Sequence[str], *, per_recording: int, length: int, seed: int = 0,
    excluded: Callable[[str, int], bool] | None = None, excluded_recordings: Iterable[str] = (),
    dataset: str = "/img_nir", workers: int = 8, flat_field_cache: Path | None = None, log: Callable[[str], None] = print,
) -> list[Segment]:
    """Draw, flat-field and save the segments (see the module docstring); returns the segments written.

    ``excluded`` defaults to :func:`temporal_eval.excluded`.  Existing flat
    fields in ``flat_field_cache`` (the pipeline's) are copied first so the
    frames are corrected exactly as the pipeline corrects them.
    """

    if excluded is None:
        from .temporal_eval import excluded as default_excluded

        excluded = default_excluded
    root = Path(root)
    (root / SEGMENTS).mkdir(parents=True, exist_ok=True)
    (root / FLAT_FIELDS).mkdir(parents=True, exist_ok=True)
    if flat_field_cache is not None:
        for path in recordings:
            cached = Path(flat_field_cache) / f"{Path(path).stem}.npz"
            if cached.exists() and not (root / FLAT_FIELDS / cached.name).exists():
                shutil.copy2(cached, root / FLAT_FIELDS / cached.name)
    counts: list[tuple[str, int]] = []
    unreadable: list[str] = []
    for path in recordings:
        total = frame_count(str(path), dataset)
        if total is None:
            unreadable.append(str(path))
        else:
            counts.append((str(path), total))
    log(f"{len(counts)} readable recordings, {len(unreadable)} unreadable")
    planned = plan_segments(counts, per_recording=per_recording, length=length, seed=seed, excluded=excluded,
                            excluded_recordings=excluded_recordings)
    log(f"{len(planned)} segments planned")
    written: list[Segment] = []
    failed: dict[str, str] = {}
    # One recording's flat field is fitted once: its segments go to the same worker in order.
    by_recording: dict[str, list[Segment]] = {}
    for segment in planned:
        by_recording.setdefault(segment.recording, []).append(segment)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_write_recording, segments, str(root), dataset) for segments in by_recording.values()]
        for future in as_completed(futures):
            for segment, error in future.result():
                if error is None:
                    written.append(segment)
                else:
                    failed[segment.name] = error
            log(f"{len(written)} segments written, {len(failed)} failed")
    written.sort(key=lambda s: (Path(s.recording).stem, s.start))
    index = {
        "segments": [asdict(s) for s in written], "length": length, "per_recording": per_recording, "seed": seed,
        "dataset": dataset, "max_lag": MAX_LAG, "unreadable_recordings": unreadable, "failed_segments": failed,
        "excluded_recordings": sorted(set(excluded_recordings)),
        "exclusion": "temporal_eval.excluded on every frame, plus excluded_recordings",
    }
    (root / INDEX).write_text(json.dumps(index, indent=1))
    return written


def _write_recording(segments: list[Segment], root: str, dataset: str) -> list[tuple[Segment, str | None]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return [_write_segment(segment, root, dataset) for segment in segments]


class ClipStore:
    """The segments of a store, memory-mapped, and the training windows they hold.

    A window is a centre frame ``t`` and its neighbours ``t + o`` for
    ``o in offsets``; every frame of the window has its full ``±max_lag``
    context inside the segment.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        index = json.loads((self.root / INDEX).read_text())
        self.segments = [Segment(**s) for s in index["segments"]]
        if not self.segments:
            raise ValueError(f"{self.root} holds no segments")
        self._arrays: dict[int, NDArray[np.uint8]] = {}

    def __len__(self) -> int:
        return len(self.segments)

    def frames(self, index: int) -> NDArray[np.uint8]:
        """``[length, H, W]`` uint8 frames of segment ``index`` (memory-mapped)."""

        if index not in self._arrays:
            self._arrays[index] = np.load(self.root / SEGMENTS / f"{self.segments[index].name}.npy", mmap_mode="r")
        return self._arrays[index]

    def recordings(self) -> list[str]:
        return sorted({Path(s.recording).stem for s in self.segments})

    def centres(self, index: int, reach: int, max_lag: int = MAX_LAG) -> range:
        """Positions in segment ``index`` whose window (``±reach`` frames, each with ``±max_lag`` context) fits."""

        length = self.segments[index].length
        return range(reach + max_lag, length - reach - max_lag)
