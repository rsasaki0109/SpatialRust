"""RGB provenance, fixed protocol, noncommuting direction and lost-frame retention."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from run_tum_rgbd_odometry import (DEFAULT_CONTROLS, ESTIMATOR, align_rgbd,
                                   validate_rgbd_plan)
from run_tum_sequence import checked_rgb, decode_rgb_gray, generate
from tum_sequence_reference import file_sha256, prepare
from test_tum_sequence_reference import fixture, PLAN

RGBD_PLAN = PLAN.with_name('tum-fr1-xyz-rgbd-pnp-plan.json')


def test_fixed_limits_and_settings_cannot_be_relaxed():
    plan = json.loads(RGBD_PLAN.read_text(encoding='utf-8'))
    validate_rgbd_plan(plan)
    for field, value in [('pnp_threshold', 4.), ('min_tracks', 10), ('max_features', 400)]:
        changed = copy.deepcopy(plan)
        changed['controls']['rgbd_odometry'][field] = value
        with pytest.raises(ValueError):
            validate_rgbd_plan(changed)
    for value in [.2, True]:
        changed = copy.deepcopy(plan)
        changed['acceptance']['maximum_first_pose_ate_rmse_m'] = value
        with pytest.raises(ValueError):
            validate_rgbd_plan(changed)
    for field, value in [('min_support_fraction', .1), ('max_depth_m', 20.)]:
        changed = copy.deepcopy(plan)
        changed['controls'][field] = value
        with pytest.raises(ValueError):
            validate_rgbd_plan(changed)


def test_rgb_bytes_bind_full_geometry_and_reject_paths_and_corruption(tmp_path):
    archive, sha = fixture(tmp_path)
    prepared = tmp_path/'prepared'
    manifest = prepare(archive, sha, PLAN, prepared)
    camera = json.loads(PLAN.read_text(encoding='utf-8'))['calibration']
    entry = manifest['frames'][0]['rgb']
    gray = decode_rgb_gray(checked_rgb(prepared, entry, 2**24), camera)
    assert gray.shape == (480, 640) and gray.dtype == np.uint8
    for bad in [None, dict(entry, path='../outside.png'), dict(entry, path='depth/other.png'),
                dict(entry, bytes=1), dict(entry, sha256='a'*64)]:
        with pytest.raises((ValueError, OSError)):
            checked_rgb(prepared, bad, 2**24)
    with pytest.raises(ValueError):
        decode_rgb_gray(checked_rgb(prepared, entry, 2**24), dict(camera, width=10))


def identity_image_adapter(previous_depth, previous_gray, current_gray, camera, controls):
    assert previous_depth.shape == previous_gray.shape == current_gray.shape == (480, 640)
    return np.eye(4), dict(identity_fixture=True)


@pytest.mark.parametrize('corruption', ['missing_rgb', 'wrong_delta'])
def test_image_generation_requires_explicit_adapter_and_preserves_all_failure_rows(tmp_path, corruption):
    archive, sha = fixture(tmp_path, count=3)
    prepared = tmp_path/'prepared'
    manifest = prepare(archive, sha, RGBD_PLAN, prepared)
    bound = file_sha256(prepared/'manifest.json')
    with pytest.raises(ValueError):
        generate(prepared, bound, tmp_path/'missing-adapter', method=ESTIMATOR)
    with pytest.raises(ValueError):
        generate(prepared, bound, tmp_path/'wrong-adapter', rgbd_aligner=identity_image_adapter)
    # Preserve first-frame initialization; a later missing association loses the
    # connected chain rather than relocalizing to an invented identity.
    if corruption == 'missing_rgb':
        manifest['frames'][1]['rgb'] = None
        manifest['frames'][1]['rgb_delta_ns'] = None
    else:
        manifest['frames'][1]['rgb_delta_ns'] = 1
    (prepared/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    result = generate(prepared, file_sha256(prepared/'manifest.json'), tmp_path/'result',
                      rgbd_aligner=identity_image_adapter, method=ESTIMATOR, additional_bindings=[__file__])
    assert result['planned_poses'] == 3 and result['generated_poses'] == 1
    rows = json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert [row['status'] for row in rows] == ['success', 'error', 'error']
    assert rows[0]['rgb_sha256']
    assert 'earlier frame' in rows[2]['error']


def test_successful_image_adapter_keeps_the_original_point_support_gate(tmp_path):
    archive, sha = fixture(tmp_path, count=2)
    prepared = tmp_path/'prepared'
    prepare(archive, sha, RGBD_PLAN, prepared)
    result = generate(prepared, file_sha256(prepared/'manifest.json'), tmp_path/'result',
                      rgbd_aligner=identity_image_adapter, method=ESTIMATOR, additional_bindings=[__file__])
    assert result['generated_poses'] == result['planned_poses'] == 2
    rows = json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert rows[1]['registration']['support_fraction'] == 1.
    assert rows[1]['registration']['identity_fixture'] is True


def test_pnp_direction_is_inverted_once_with_noncommuting_rotation(monkeypatch):
    import run_tum_rgbd_odometry as runner
    settings = dict(DEFAULT_CONTROLS)
    depth = np.full((480, 640), 5000, np.uint16)
    image = np.zeros((480, 640), np.uint8)
    pixels = np.array([[50.+i*10, 70.+i*5] for i in range(25)])
    class Keypoint:
        def __init__(self, point):
            self.x, self.y = point
    monkeypatch.setattr(runner.sr, 'shi_tomasi_keypoints', lambda *a, **k: [Keypoint(p) for p in pixels])
    monkeypatch.setattr(runner.sr, 'track_points_forward_backward', lambda a,b,p,**k: (p.copy(),np.ones(len(p),bool),np.zeros(len(p))))
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([1., 2., 3.])
    monkeypatch.setattr(runner.sr, 'estimate_rgbd_odometry', lambda *a, **k: (rotation, translation, np.ones(len(pixels),bool), 0))
    controls = dict(rgbd_odometry=settings, min_depth_m=.1, max_depth_m=5.)
    transform, report = align_rgbd(depth, image, image, dict(width=640,height=480,depth_unit_m=.0002,fx=525.,fy=525.,cx=319.5,cy=239.5), controls)
    expected = np.eye(4)
    expected[:3, :3] = rotation.T
    expected[:3, 3] = [-2., 1., -3.]
    np.testing.assert_array_equal(transform, expected)
    np.testing.assert_allclose(transform @ np.array(report['current_from_previous']), np.eye(4))
