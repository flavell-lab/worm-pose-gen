"""Bounded reader for ConfocalTrackerControl acquisition nose landmarks.

Schema verified against ConfocalTrackerControl.jl commit
359832f79da8f037324e3d824f2909e21e176043, src/gui_loop.jl and src/data.jl:
https://github.com/flavell-lab/ConfocalTrackerControl.jl/tree/359832f79da8f037324e3d824f2909e21e176043/src
Julia writes [landmark, coordinate, time], exposed by h5py as [time,
coordinate, landmark]. Landmarks are nose, mid, pharynx; coordinates are
one-based image x, y, confidence. Features and images are appended together.
Stage coordinates are never used as image landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
from numpy.typing import NDArray

from .frame_reader import VIDEO_SUFFIXES


@dataclass(frozen=True)
class HeadTracking:
    xy: NDArray[np.float32]
    valid: NDArray[np.bool_]
    confidence: NDArray[np.float32]
    frame_indices: NDArray[np.int64]
    source_frame_ids: NDArray[np.int64]
    source_timestamps: NDArray[np.int64]
    provenance: dict[str, Any]


def _rows(dataset: h5py.Dataset, indices: NDArray[np.int64]) -> np.ndarray:
    """Read only requested rows, preserving duplicates and caller ordering."""
    unique, inverse = np.unique(indices, return_inverse=True)
    return np.asarray(dataset[unique])[inverse]


def read_head_tracking(
    recording_path: str | Path,
    frame_indices: Sequence[int] | NDArray[np.int64],
    image_shape: tuple[int, int],
    *, minimum_confidence: float = 0.9,
) -> HeadTracking:
    """Return zero-based image XY nose cues, rejecting uncertain observations.

    Missing/unrecognized metadata returns NaN heads and a false valid mask;
    provenance explains the rejection. Camera metadata is optional: direct
    /pos_feature to /img_nir alignment is established by the acquisition writer.
    If present, inconsistent save flags invalidate the cues. Only sampled
    feature rows are read; camera save flags are scanned in bounded chunks.
    A confidence > 0.9 follows the acquisition default, not an inferred status
    saying that stage tracking was enabled (that status is not stored).
    """
    raw = np.asarray(frame_indices)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in 'iu'):
        raise ValueError('frame_indices must be a one-dimensional integer sequence')
    indices = raw.astype(np.int64)
    if len(image_shape) != 2 or min(image_shape) <= 0:
        raise ValueError('image_shape must be positive (height, width)')
    if not np.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in [0, 1]")
    n = len(indices)
    xy = np.full((n, 2), np.nan, np.float32)
    valid = np.zeros(n, bool)
    confidence = np.full(n, np.nan, np.float32)
    ids = np.full(n, -1, np.int64)
    timestamps = ids.copy()
    provenance: dict[str, Any] = {
        'recording_path': str(recording_path), 'dataset': '/pos_feature',
        'landmark': 'nose', 'coordinate_system': 'zero_based_image_xy',
        'source_coordinate_system': 'one_based_image_xy',
        'confidence_threshold': float(minimum_confidence), 'alignment': 'same_saved_image_index',
        'schema': 'ConfocalTrackerControl.jl/359832f79da8f037324e3d824f2909e21e176043',
        'status': 'unavailable',
    }
    result = HeadTracking(xy, valid, confidence, indices, ids, timestamps, provenance)
    # A video carries no landmarks: it is rejected as an HDF5 recording without them is.
    if Path(recording_path).suffix.lower() in VIDEO_SUFFIXES:
        provenance['reason'] = 'missing_img_nir_or_pos_feature'
        return result
    try:
        with h5py.File(recording_path, 'r') as h:
            if '/img_nir' not in h or '/pos_feature' not in h:
                provenance['reason'] = 'missing_img_nir_or_pos_feature'
                return result
            video, features = h['/img_nir'], h['/pos_feature']
            if video.ndim != 3 or tuple(video.shape[1:]) != tuple(image_shape):
                provenance['reason'] = 'image_shape_mismatch'
                return result
            total = video.shape[0]
            if features.shape != (total, 3, 3):
                provenance['reason'] = 'unrecognized_feature_shape_or_frame_count'
                return result
            if np.any(indices < 0) or np.any(indices >= total):
                raise ValueError('frame_indices outside recording')
            group = h.get('/img_metadata')
            if group is not None:
                required = ('q_iter_save', 'q_recording', 'img_id', 'img_timestamp')
                if not all(k in group for k in required):
                    provenance['reason'] = 'incomplete_camera_metadata'
                    return result
                count = group['q_iter_save'].shape
                if len(count) != 1 or any(group[k].shape != count for k in required):
                    provenance['reason'] = 'camera_metadata_shape_mismatch'
                    return result
                # Keep only requested camera rows; never materialize all metadata.
                wanted, inverse = np.unique(indices, return_inverse=True)
                camera_rows = np.full(len(wanted), -1, np.int64)
                saved = 0
                for start in range(0, count[0], 65536):
                    stop = min(start + 65536, count[0])
                    save = np.asarray(group['q_iter_save'][start:stop])
                    recording = np.asarray(group['q_recording'][start:stop])
                    if not np.all(np.isin(save, [0, 1])) or not np.all(np.isin(recording, [0, 1])):
                        provenance['reason'] = 'invalid_camera_save_flags'
                        return result
                    rows = np.flatnonzero((save == 1) & (recording == 1)) + start
                    lo, hi = np.searchsorted(wanted, [saved, saved + len(rows)])
                    camera_rows[lo:hi] = rows[wanted[lo:hi] - saved]
                    saved += len(rows)
                if saved != total:
                    provenance['reason'] = 'camera_saved_frame_count_mismatch'
                    return result
                if n:
                    ids[:] = _rows(group['img_id'], camera_rows)[inverse]
                    timestamps[:] = _rows(group['img_timestamp'], camera_rows)[inverse]
                provenance['camera_alignment'] = 'q_iter_save_and_q_recording'
            else:
                provenance['camera_alignment'] = 'metadata_absent'
            if n:
                sampled = _rows(features, indices)
                candidate = sampled[:, :2, 0].astype(np.float32) - 1
                confidence[:] = sampled[:, 2, 0]
                height, width = image_shape
                valid[:] = (np.isfinite(candidate).all(axis=1)
                            & np.isfinite(confidence) & (confidence > minimum_confidence) & (confidence <= 1)
                            & (candidate[:, 0] >= 0) & (candidate[:, 0] <= width - 1)
                            & (candidate[:, 1] >= 0) & (candidate[:, 1] <= height - 1))
                xy[valid] = candidate[valid]
            provenance['status'] = 'available'
            provenance['valid_count'] = int(valid.sum())
            provenance['invalid_count'] = int(n - valid.sum())
    except (OSError, KeyError, TypeError) as error:
        xy[:] = np.nan
        valid[:] = False
        provenance['reason'] = f'{type(error).__name__}: {error}'
    return result
