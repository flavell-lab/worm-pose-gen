import h5py
import numpy as np
import unittest
import tempfile
from pathlib import Path

from worm_pose_gen.head_tracking import read_head_tracking
from tests.test_frame_reader import HEIGHT, WIDTH, _write_video


def recording(tmp_path, *, metadata=True):
    path = tmp_path / 'recording.h5'
    with h5py.File(path, 'w') as h:
        h.create_dataset('img_nir', shape=(4, 20, 30), dtype='u1')
        features = np.zeros((4, 3, 3), np.float32)
        features[:, 0, 0] = [2, 5, 9, 12]
        features[:, 1, 0] = [3, 6, 10, 13]
        features[:, 2, 0] = [.99, .99, .2, .99]
        features[:, :2, 1:] = 99  # Other landmarks must not become the head.
        h['pos_feature'] = features
        h['pos_stage'] = np.full((4, 2), 50000)
        if metadata:
            h['img_metadata/q_iter_save'] = [0, 1, 0, 1, 0, 1, 0, 1]
            h['img_metadata/q_recording'] = np.ones(8, dtype='u1')
            h['img_metadata/img_id'] = np.arange(8) + 100
            h['img_metadata/img_timestamp'] = np.arange(8) * 25
    return path


class HeadTrackingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.tmp_path = Path(self.directory.name)

    def test_nose_xy_origin_sparse_order_and_camera_alignment(self):
        tmp_path = self.tmp_path
        result = read_head_tracking(recording(tmp_path), [3, 0, 3, 2], (20, 30))
        np.testing.assert_array_equal(result.valid, [True, True, True, False])
        np.testing.assert_array_equal(result.xy[:3], [[11, 12], [1, 2], [11, 12]])
        assert np.isnan(result.xy[3]).all()
        np.testing.assert_array_equal(result.source_frame_ids, [107, 101, 107, 105])
        np.testing.assert_array_equal(result.source_timestamps, [175, 25, 175, 125])


    def test_missing_landmarks_does_not_use_stage(self):
        tmp_path = self.tmp_path
        path = recording(tmp_path)
        with h5py.File(path, 'a') as h:
            del h['pos_feature']
        result = read_head_tracking(path, [0], (20, 30))
        assert not result.valid.any()
        assert result.provenance['reason'] == 'missing_img_nir_or_pos_feature'


    def test_video_has_no_landmarks(self):
        path = self.tmp_path / 'recording.mp4'
        _write_video(path, 'libx264', frames=4)
        result = read_head_tracking(path, [3, 0], (HEIGHT, WIDTH))
        assert not result.valid.any() and np.isnan(result.xy).all()
        assert result.provenance['status'] == 'unavailable'
        assert result.provenance['reason'] == 'missing_img_nir_or_pos_feature'


    def test_bad_confidence_and_bounds(self):
        tmp_path = self.tmp_path
        path = recording(tmp_path, metadata=False)
        with h5py.File(path, 'a') as h:
            h['pos_feature'][0, 0, 0] = 31
            h['pos_feature'][1, 1, 0] = np.nan
            h['pos_feature'][3, 2, 0] = 1.1
        result = read_head_tracking(path, [0, 1, 2, 3], (20, 30))
        assert not result.valid.any()
        assert np.isnan(result.xy).all()


    def test_bad_alignment_rejects_tracking(self):
        tmp_path = self.tmp_path
        path = recording(tmp_path)
        with h5py.File(path, 'a') as h:
            h['img_metadata/q_iter_save'][0] = 1
        result = read_head_tracking(path, [0], (20, 30))
        assert not result.valid.any()
        assert result.provenance['reason'] == 'camera_saved_frame_count_mismatch'


    def test_optional_camera_metadata_and_custom_threshold(self):
        tmp_path = self.tmp_path
        path = recording(tmp_path, metadata=False)
        result = read_head_tracking(path, [2], (20, 30), minimum_confidence=.1)
        assert result.valid[0]
        assert result.source_frame_ids[0] == -1
        assert result.provenance['camera_alignment'] == 'metadata_absent'


    def test_empty_and_invalid_indices(self):
        tmp_path = self.tmp_path
        path = recording(tmp_path)
        assert read_head_tracking(path, [], (20, 30)).xy.shape == (0, 2)
        with self.assertRaises(ValueError):
            read_head_tracking(path, [4], (20, 30))
        with self.assertRaises(ValueError):
            read_head_tracking(path, [1.5], (20, 30))
