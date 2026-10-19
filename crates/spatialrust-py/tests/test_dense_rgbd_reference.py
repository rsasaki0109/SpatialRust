"""Closed-form independent final-warp counts, errors, and missing-corner rules."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from audit_tum_dense_rgbd import reference_statistics
from dense_rgbd_protocol import CONTROLS

CAMERA = dict(fx=64., fy=64., cx=8., cy=6., depth_unit_m=.0002)


def inputs():
    y, x = np.mgrid[:12, :16]
    depth = np.full((12, 16), 10000, dtype=np.uint16)
    source = (30+x+3*y).astype(np.uint8)
    target = (29+x+3*y).astype(np.uint8)
    pose = np.eye(4)
    pose[0, 3] = .03125
    return depth, source, target, pose


def test_one_pixel_metric_translation_has_exact_full_row_counts_and_zero_errors():
    depth, source, target, pose = inputs()
    result = reference_statistics(depth, depth, source, target, pose, CAMERA, CONTROLS)
    assert result['valid_source_pixels'] == 16*12
    assert result['correspondences'] == (16-2)*(12-1)
    assert result['correspondence_fraction'] == 154/192
    assert result['photometric_rmse'] < 1e-15 and result['depth_rmse_m'] == 0


def test_missing_target_corner_rejects_all_four_affected_source_rows():
    depth, source, target, pose = inputs()
    target_depth = depth.copy()
    target_depth[5, 6] = 0
    result = reference_statistics(depth, target_depth, source, target, pose, CAMERA, CONTROLS)
    assert result['valid_source_pixels'] == 192 and result['correspondences'] == 150
    assert result['depth_rmse_m'] == 0


def test_missing_source_row_changes_the_complete_denominator():
    depth, source, target, pose = inputs()
    source_depth = depth.copy()
    source_depth[2, 2] = 0
    result = reference_statistics(source_depth, depth, source, target, pose, CAMERA, CONTROLS)
    assert result['valid_source_pixels'] == 191 and result['correspondences'] == 153
    assert result['correspondence_fraction'] == 153/191


def test_no_measurement_is_not_a_zero_error_success():
    depth, source, target, pose = inputs()
    result = reference_statistics(depth, np.zeros_like(depth), source, target, pose, CAMERA, CONTROLS)
    assert result['valid_source_pixels'] == 192 and result['correspondences'] == 0
    assert result['photometric_rmse'] is None and result['depth_rmse_m'] is None
