"""Known metric motion, missing-depth/shape/budget rejection and owned calls."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import spatialrust as sr


def images(shift=0, width=96, height=72):
    y, x = np.mgrid[:height, :width]
    x = x.astype(np.float64) - shift
    gray = np.floor(128 + 45*np.sin(.26*x) + 40*np.cos(.19*y)
                    + 25*np.sin(.13*(x+y)) + .5).astype(np.uint8)
    return gray, np.full((height, width), 2, dtype=np.float32)


def estimate(source, target, **kwargs):
    return sr.estimate_dense_rgbd_odometry(*source, *target, 64., 64., 48., 36., **kwargs)


def test_known_metric_source_to_target_motion_and_owned_outputs():
    source, target = images(), images(1)
    originals = [x.copy() for x in (*source, *target)]
    rotation, translation, report = estimate(source, target)
    np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=1e-4)
    np.testing.assert_allclose(translation, [.03125, 0, 0], rtol=0, atol=1e-4)
    assert report['correspondence_fraction'] > .9
    assert report['depth_rmse_m'] < 1e-5 and report['photometric_rmse'] < 1e-4
    assert report['valid_source_pixels'] == 96*72
    assert report['input_copy_bytes'] == 96*72*10
    assert report['final_support_only'] is True
    assert report['working_bytes'] <= 128*1024**2 and 1 <= report['workers'] <= 4
    for actual, expected in zip((*source, *target), originals):
        np.testing.assert_array_equal(actual, expected)
    assert all(not np.shares_memory(output, x)
               for output in (rotation, translation) for x in (*source, *target))


def test_strided_inputs_and_explicit_initial_pose_keep_identity():
    source = images()
    # Same logical pixels with noncontiguous/reversed underlying storage.
    strided = tuple(np.repeat(x, 2, axis=1)[:, ::2] for x in source)
    result = estimate(strided, strided, initial=np.eye(4))
    np.testing.assert_allclose(result[0], np.eye(3), rtol=0, atol=1e-10)
    np.testing.assert_allclose(result[1], np.zeros(3), rtol=0, atol=1e-10)
    assert all(not x.flags.c_contiguous for x in strided)


@pytest.mark.parametrize('kwargs', [
    {'iterations': []}, {'iterations': [0]}, {'iterations': [1]*7},
    {'min_depth': -1}, {'max_depth': .05}, {'max_depth_difference': 0},
    {'epsilon': float('nan')}, {'depth_weight': 2},
    {'min_correspondences': 5}, {'min_correspondence_fraction': 0},
    {'max_working_bytes': 1}, {'initial': np.zeros((3, 3))},
    {'initial': np.diag([1., 1., -1., 1.])},
    {'initial': np.diag([1., 1., 1., 2.])},
])
def test_invalid_math_schema_and_storage_options_fail(kwargs):
    source = images()
    with pytest.raises((ValueError, OverflowError)):
        estimate(source, source, **kwargs)


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_missing_depth_does_not_become_an_identity_success(value):
    gray, depth = images()
    depth[:] = value
    with pytest.raises(ValueError):
        estimate((gray, depth), (gray, depth))


def test_shape_dtype_and_unobservable_scene_rejection():
    gray, depth = images()
    with pytest.raises(ValueError):
        estimate((gray, depth), (gray[:, :-1], depth[:, :-1]))
    with pytest.raises(TypeError):
        estimate((gray, depth.astype(np.float64)), (gray, depth))
    flat = np.full_like(gray, 100)
    with pytest.raises(ValueError):
        estimate((flat, depth), (flat, depth))


def test_low_final_overlap_is_rejected_even_with_zero_residual():
    gray, depth = images()
    target = np.zeros_like(depth)
    target[:, 30:55] = 2
    with pytest.raises(ValueError, match='returned full-resolution pose'):
        estimate((gray, depth), (gray, target), iterations=[1])


def test_simultaneous_native_calls_keep_independent_owned_results():
    a, b = images(width=256, height=160), images(width=256, height=160)
    expected = estimate(a, b, iterations=[1])
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(lambda _: estimate(a, b, iterations=[1]), range(2)))
    for rotation, translation, report in results:
        np.testing.assert_array_equal(rotation, expected[0])
        np.testing.assert_array_equal(translation, expected[1])
        assert report == expected[2]
