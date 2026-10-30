import tempfile
import unittest
from pathlib import Path
from unittest import mock

import av
import h5py
import numpy as np
from scipy import ndimage

from worm_pose_gen import frame_reader
from worm_pose_gen.stage_alignment import (
    StageCalibration,
    align_frame,
    anchor_frames,
    bandpass,
    clean_stage,
    fill_shifts,
    fill_stretches,
    fit_calibration,
    read_stage,
    recording_calibration,
    recording_steps,
    register_shift,
    shift_image,
    stage_shift_px,
    substrate_shift_px,
    untrusted_frames,
)


def speckle(shape: tuple[int, int] = (128, 160), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 + 40.0 * ndimage.gaussian_filter(rng.normal(size=shape), 1.5)


class StageTrackTests(unittest.TestCase):
    def test_gaps_are_interpolated_and_long_ones_flagged(self) -> None:
        raw = np.stack([np.arange(12.0) * 10, np.arange(12.0) * -5], axis=1)
        raw[[0, 3, 4, 7, 8, 9]] = np.nan  # leading read, a gap of 2, a gap of 3
        track = clean_stage(raw, max_gap=2)
        np.testing.assert_allclose(track.filled[1:], np.stack([np.arange(1, 12.0) * 10, np.arange(1, 12.0) * -5], axis=1))
        np.testing.assert_array_equal(track.gap, [1, 0, 0, 2, 2, 0, 0, 3, 3, 3, 0, 0])
        np.testing.assert_array_equal(track.reliable, [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1])

    def test_fractional_lookup_is_nan_where_unreliable(self) -> None:
        raw = np.stack([np.arange(10.0), np.zeros(10)], axis=1)
        raw[5:8] = np.nan
        track = clean_stage(raw, max_gap=2)
        at = track.at([1.25, 3.5, 4.5, 8.5, 9.5, -0.5])
        np.testing.assert_allclose(at[[0, 1, 3], 0], [1.25, 3.5, 8.5])
        self.assertTrue(np.isnan(at[[2, 4, 5]]).all())  # touches the gap of 3, or leaves the recording


class RegistrationTests(unittest.TestCase):
    def test_synthetic_shift_is_recovered_with_a_masked_object(self) -> None:
        reference = speckle()
        shift = np.array([5.3, -2.7])
        moving = ndimage.shift(reference, shift[::-1], order=3, mode="wrap")
        invalid = np.zeros(reference.shape, dtype=bool)
        invalid[50:80, 60:100] = True
        moving[invalid] = 0.0  # a dark worm that does not move with the background
        found, score = register_shift(bandpass(moving), bandpass(reference), invalid)
        np.testing.assert_allclose(found, shift, atol=0.15)
        self.assertGreater(score, 10.0)

    def test_shift_image_moves_content_and_marks_pixels_from_outside(self) -> None:
        image = speckle((40, 50))
        shifted, valid = shift_image(image, (3.0, -2.0))
        np.testing.assert_allclose(shifted[5:30, 10:40], image[7:32, 7:37])
        self.assertFalse(valid[:, :3].any())
        self.assertFalse(valid[-2:, :].any())
        self.assertTrue(valid[:-2, 3:].all())

    def test_align_frame_uses_the_difference_of_substrate_shifts(self) -> None:
        image = speckle((40, 50))
        direct, _ = shift_image(image, (1.5, 0.5))
        aligned, valid = align_frame(image, (2.0, -1.0), (0.5, -1.5))
        np.testing.assert_allclose(aligned[valid], direct[valid])


class CalibrationTests(unittest.TestCase):
    def test_fit_recovers_matrix_and_time_offset(self) -> None:
        rng = np.random.default_rng(1)
        raw = np.cumsum(rng.normal(scale=40.0, size=(400, 2)), axis=0)
        track = clean_stage(raw)
        matrix = np.array([[0.078, 0.002], [-0.001, 0.079]])
        offset = -0.5
        t = rng.integers(0, 380, size=150).astype(np.float64)
        lag = rng.choice([1.0, 4.0, 16.0], size=150)
        truth = (track.at(t + lag + offset) - track.at(t + offset)) @ matrix.T
        measured = truth + rng.normal(scale=0.1, size=truth.shape)
        measured[:10] = 0.0  # registrations locked onto the camera's fixed pattern
        shifts = np.column_stack([t, lag, measured, np.full(150, 30.0)])
        calibration = fit_calibration(track, shifts)
        np.testing.assert_allclose(calibration.matrix, matrix, atol=2e-3)
        self.assertAlmostEqual(calibration.offset_frames, offset, places=6)
        self.assertGreaterEqual(calibration.outliers, 8)
        self.assertLess(calibration.residual_px, 0.3)

    def test_stage_shift_cancels_stage_motion_and_calibration_round_trips(self) -> None:
        raw = np.array([[0.0, 0.0], [10.0, 0.0], [np.nan, np.nan], [30.0, 20.0], [40.0, 40.0]])
        calibration = StageCalibration(matrix=((0.1, 0.0), (0.0, -0.2)), offset_frames=0.0, residual_px=0.5, pairs=10, outliers=0)
        shift = stage_shift_px(clean_stage(raw), [0, 2, 4], calibration)
        with tempfile.TemporaryDirectory() as directory:
            calibration.save(Path(directory) / "c.json")
            self.assertEqual(StageCalibration.load(Path(directory) / "c.json"), calibration)
        # The background moves by +A d in the image; adding the shift undoes it.
        np.testing.assert_allclose(shift, [[0.0, 0.0], [-2.0, 2.0], [-4.0, 8.0]])


IDENTITY = StageCalibration(matrix=((1.0, 0.0), (0.0, 1.0)), offset_frames=0.0, residual_px=0.5, pairs=10, outliers=0)


class FillTests(unittest.TestCase):
    """Stage units are pixels here (identity matrix, no time offset), so the true offset of frame t is -position[t]."""

    def test_jumps_speed_and_unknown_frames_are_untrusted_and_anchors_keep_clear(self) -> None:
        raw = np.zeros((30, 2))
        raw[:, 0] = np.arange(30.0)  # 1 px per frame
        raw[10, 1] = 9.0  # a reading off by 9 px: second differences of 9 and 18
        raw[20:22] = np.nan  # a short gap, interpolated: trusted, but no anchor
        raw[25:, 0] = 24.0 + 12.0 * np.arange(1, 6)  # 12 px per frame
        track = clean_stage(raw)
        untrusted = untrusted_frames(track, IDENTITY)
        np.testing.assert_array_equal(np.flatnonzero(untrusted), [9, 10, 11, 24, 25, 26, 27, 28])
        anchors = anchor_frames(track, IDENTITY, untrusted, guard=2)
        self.assertFalse(anchors[7:14].any())
        self.assertTrue(anchors[[6, 14, 19]].all())
        self.assertFalse(anchors[20:22].any())

    def test_stretches_reach_margin_anchors_and_merge(self) -> None:
        untrusted = np.zeros(40, dtype=bool)
        untrusted[[10, 16, 37]] = True
        anchors = ~ndimage.binary_dilation(untrusted, iterations=1)
        # 10 and 16 are 3 anchors apart, under two margins of 2: one stretch; 37 has one anchor after it and runs to the end.
        self.assertEqual(fill_stretches(untrusted, anchors, margin=2), [(7, 20), (34, 40)])
        anchors[6] = False  # not an anchor (a filled-in read): no change, the margin is 7 and 8
        self.assertEqual(fill_stretches(untrusted, anchors, margin=2)[0], (7, 20))
        anchors[8] = False  # now the margin is 7 and 5
        self.assertEqual(fill_stretches(untrusted, anchors, margin=2)[0], (5, 20))

    def test_fill_follows_registration_between_anchors_and_spreads_drift(self) -> None:
        n = 40
        truth = np.zeros((n, 2))
        truth[:, 0] = -np.arange(n) * 2.0
        truth[15:25, 1] = 20.0 * np.sin(np.arange(10) * 2.0)  # a fast oscillation
        stage = truth.copy()
        stage[15:25, 1] = 0.0  # the stage misses it
        stage[18:22] = np.nan
        anchors = np.ones(n, dtype=bool)
        anchors[12:28] = False
        steps = np.column_stack([-np.diff(truth, axis=0), np.full(n - 1, 50.0)])
        steps[:, 0] += 0.05  # registration bias: 0.05 px per step, 0.75 px over the stretch
        filled = fill_shifts(stage, anchors, steps, [(8, 32)])
        np.testing.assert_array_equal(filled[:8], stage[:8])
        np.testing.assert_allclose(filled[8:32], truth[8:32], atol=0.15)
        # The anchor runs (8-11, 28-31) are matched at their middle frames, the drift spread between them.
        np.testing.assert_allclose(filled[[9, 10, 29, 30]], truth[[9, 10, 29, 30]], atol=0.05)

    def test_failed_steps_fall_back_to_readings_or_nan(self) -> None:
        n = 30
        truth = np.zeros((n, 2))
        truth[:, 1] = np.arange(n) * 1.5
        stage = truth.copy()
        stage[12:22] = np.nan
        stage[16] = truth[16] + [0.0, 3.0]  # one reading inside, 3 px off
        anchors = np.isfinite(stage).all(axis=1)
        anchors[10:24] = False
        steps = np.column_stack([-np.diff(truth, axis=0), np.full(n - 1, 50.0)])
        steps[[13, 19], 2] = 5.0  # two failed registrations split the stretch into three pieces
        filled = fill_shifts(stage, anchors, steps, [(6, 28)])
        np.testing.assert_allclose(filled[6:14], truth[6:14], atol=1e-9)  # anchored on the left margin
        np.testing.assert_allclose(filled[14:20], truth[14:20] + [0.0, 3.0], atol=1e-9)  # only the one reading
        np.testing.assert_allclose(filled[20:28], truth[20:28], atol=1e-9)  # anchored on the right margin
        stage[16] = np.nan
        filled = fill_shifts(stage, anchors, steps, [(6, 28)])
        self.assertTrue(np.isnan(filled[14:20]).all())
        self.assertTrue(np.isfinite(filled[np.r_[6:14, 20:28]]).all())


class SubstrateShiftTests(unittest.TestCase):
    def test_registration_fills_a_long_gap_and_a_missed_excursion(self) -> None:
        n, shape = 48, (96, 128)
        rng = np.random.default_rng(3)
        texture = 120.0 + 25.0 * ndimage.gaussian_filter(rng.normal(size=(200, 240)), 1.2)
        position = np.zeros((n, 2))  # background displacement in the image (x, y), whole pixels
        position[:, 0] = np.arange(n)
        position[18:24, 1] = [7, -7, 7, -7, 7, -7]  # fast motion while the stage reads nothing
        position[35, 1] = 4  # a short excursion the one reading overstates
        raw = position.copy()
        raw[18:24] = np.nan  # a gap of 6 frames: unreliable
        raw[35, 1] = 14.0
        frames = np.stack([texture[60 - int(dy) : 60 - int(dy) + shape[0], 60 - int(dx) : 60 - int(dx) + shape[1]] for dx, dy in position])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rec.h5"
            with h5py.File(path, "w") as handle:
                handle["/pos_stage"] = raw
                handle["/img_nir"] = np.clip(frames, 0, 255).astype(np.uint8)
            stage = stage_shift_px(clean_stage(raw), np.arange(n), IDENTITY)
            cache = Path(directory) / "steps"
            filled = substrate_shift_px(path, np.arange(n), IDENTITY, cache, Path(directory) / "flat")
            with mock.patch("worm_pose_gen.stage_alignment.measure_shifts", side_effect=AssertionError("steps not cached")):
                again = substrate_shift_px(path, np.arange(n), IDENTITY, cache, Path(directory) / "flat")
        np.testing.assert_array_equal(again, filled)
        self.assertTrue(np.isnan(stage[18:24]).all())
        np.testing.assert_allclose(filled, -position, atol=0.5)
        self.assertGreater(np.abs(stage[35, 1] + position[35, 1]), 9.0)


class VideoTests(unittest.TestCase):
    def test_a_video_registers_its_frames_and_has_no_stage(self) -> None:
        n, shape = 12, (96, 128)
        rng = np.random.default_rng(4)
        texture = np.clip(120.0 + 25.0 * ndimage.gaussian_filter(rng.normal(size=(120, 160)), 1.2), 0, 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(frame_reader, "INDEX_DIR", Path(directory) / "index"):
            path = Path(directory) / "rec.avi"
            # Lossless gray frames, the background moving 1 px right a frame.
            with av.open(str(path), "w") as container:
                stream = container.add_stream("ffv1", rate=20)
                stream.width, stream.height, stream.pix_fmt = shape[1], shape[0], "gray"
                for t in range(n):
                    image = np.ascontiguousarray(texture[10 : 10 + shape[0], 20 - t : 20 - t + shape[1]])
                    container.mux(stream.encode(av.VideoFrame.from_ndarray(image, format="gray")))
                container.mux(stream.encode(None))
            steps = recording_steps(path, np.arange(n), Path(directory) / "steps", Path(directory) / "flat")
            # As for an HDF5 recording without /pos_stage: KeyError, which the export and the pipeline read as "no stage".
            with self.assertRaisesRegex(KeyError, "no /pos_stage"):
                read_stage(path)
            with self.assertRaisesRegex(KeyError, "no /pos_stage"):
                recording_calibration(path, cache_dir=Path(directory) / "calibrations")
        np.testing.assert_allclose(steps[:-1, :2], np.tile([1.0, 0.0], (n - 1, 1)), atol=0.2)
        self.assertTrue(np.isnan(steps[-1]).all())


if __name__ == "__main__":
    unittest.main()
