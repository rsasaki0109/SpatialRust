"""Paired claims require unchanged measurements, controls and comparator results."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from report_pnp_repair import compare


def receipts():
    before = dict(
        schema='spatialrust.opencv-pnp-study.v1', native_sha256='a' * 64,
        source_sha256='b' * 64, calculation_sha256='c' * 64,
        seeds=1, max_iterations=300, threshold_pixels=3, confidence=.99,
        camera_intrinsics=[700, 710, 320, 240], thread_environment={},
        opencv_threads=1, versions={'opencv': '4.12.0'}, rows=[],
    )
    for method in ('spatialrust_plain', 'spatialrust_ransac', 'opencv_plain', 'opencv_ransac'):
        before['rows'].append(dict(
            geometry='plane', condition='noise', seed=0, method=method,
            object_sha256='d' * 64, image_sha256='e' * 64, correspondences=100,
            known_wrong_correspondences=0, generating_pose_ambiguity_known=False,
            status='success', recovered=False, rotation_error_degrees=2.,
            translation_error_metres=.01, positive_depth_fraction=1.,
        ))
    after = copy.deepcopy(before)
    after['native_sha256'] = 'f' * 64
    after['source_sha256'] = '0' * 64
    after['rows'][0].update(recovered=True, rotation_error_degrees=.1)
    return before, after


def test_controlled_gain():
    before, after = receipts()
    result = compare(before, after)
    assert result['opencv_controls_exact']
    assert len(result['methods']['spatialrust_plain']['gains']) == 1
    assert result['methods']['spatialrust_plain']['regressions'] == []
    assert result['methods']['spatialrust_ransac']['gains'] == []


@pytest.mark.parametrize('change', [
    'input', 'threshold', 'calculation', 'opencv', 'duplicate', 'recovery', 'missing_method',
])
def test_invalid_paired_claim_rejected(change):
    before, after = receipts()
    if change == 'input':
        for row in after['rows']:
            row['image_sha256'] = '1' * 64
    elif change == 'threshold':
        after['threshold_pixels'] = 4
    elif change == 'calculation':
        after['calculation_sha256'] = '1' * 64
    elif change == 'opencv':
        after['rows'][2]['rotation_error_degrees'] = 3.
    elif change == 'duplicate':
        after['rows'].append(copy.deepcopy(after['rows'][0]))
    elif change == 'recovery':
        after['rows'][0]['rotation_error_degrees'] = 3.
    else:
        after['rows'].pop()
    with pytest.raises(ValueError):
        compare(before, after)
