"""Independent unit/coordinate/hash/denominator checks for Redwood preparation."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from redwood_frame_reference import checked_bytes, depth_points, load_plan, read_log, relative_pose, score


def camera():
    return dict(width=3, height=2, fx=2., fy=4., cx=1., cy=.5,
                depth_scale=1000., depth_trunc_m=3.)


def log_bytes(poses):
    lines = []
    for i, pose in enumerate(poses):
        lines.append(f'{i} {i} {i+1}')
        lines.extend(' '.join(str(x) for x in row) for row in pose)
    return ('\n'.join(lines) + '\n').encode('ascii')


def test_depth_units_axes_invalid_and_truncation():
    depth = np.array([[1000, 0, 3000], [2000, 1000, 65535]], dtype=np.uint16)
    xyz = depth_points(depth, camera())
    np.testing.assert_array_equal(xyz, [[-.5, -.125, 1], [-1, .25, 2], [0, .125, 1]])
    assert xyz.dtype == np.float32


@pytest.mark.parametrize('depth', [np.zeros((2, 3), np.uint8), np.zeros((3, 2), np.uint16)])
def test_depth_rejects_lossy_or_wrong_dimensions(depth):
    with pytest.raises(ValueError):
        depth_points(depth, camera())


@pytest.mark.parametrize('key,value', [('fx', 0), ('fy', -2), ('depth_scale', 0), ('cx', float('nan'))])
def test_depth_rejects_invalid_calibration(key, value):
    c = camera()
    c[key] = value
    with pytest.raises(ValueError):
        depth_points(np.ones((2, 3), np.uint16), c)


def test_reference_direction_with_noncommuting_motion():
    source = np.eye(4)
    source[:3, 3] = [1, 2, 3]
    target = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6], [0, 0, 0, 1]], float)
    poses = read_log(log_bytes([source, target]), 2)
    relative = relative_pose(*poses)
    np.testing.assert_allclose(relative[:3, 3], [-3, 3, -3], atol=1e-12)
    np.testing.assert_allclose(target @ relative, source, atol=1e-12)
    assert not np.allclose(relative, np.linalg.inv(source) @ target)


@pytest.mark.parametrize('mode', ['identity', 'short', 'reflection', 'nonfinite'])
def test_log_rejects_bad_identity_length_or_pose(mode):
    pose = np.eye(4)
    if mode == 'reflection':
        pose[0, 0] = -1
    if mode == 'nonfinite':
        pose[0, 3] = float('nan')
    data = log_bytes([pose])
    if mode == 'identity':
        data = data.replace(b'0 0 1', b'1 1 2')
    if mode == 'short':
        data = b'\n'.join(data.splitlines()[:-1])
    with pytest.raises(ValueError):
        read_log(data, 1)


def test_wrong_publisher_hash_is_rejected(tmp_path):
    path = tmp_path / 'depth.zip'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='official dataset MD5'):
        checked_bytes(path, hashlib.md5(b'original').hexdigest())


def fixture():
    plan = dict(frame_count=2, pairs=[[0, 1]], seeds=[7], accuracy_limits=dict(translation_m=.05, rotation_degrees=5),
                download_prefix='https://example.test/', trajectory=dict(name='traj.txt'), dataset='synthetic fixture', scope='test')
    hashes = dict(source='a'*64, target='b'*64)
    report = dict(schema_version='spatialrust.python-alignment.v1', input_file_sha256=hashes,
                  initial_transform_supplied=False, transform_source_to_target=np.eye(4).tolist())
    rows = [dict(pair=[0, 1], seed=7, method='spatialrust', status='success', report=report),
            dict(pair=[0, 1], seed=7, method='open3d', status='error', error='retained failure')]
    frozen = dict(schema='spatialrust.redwood-frozen-registrations.v1', plan=copy.deepcopy(plan), rows=rows,
                  preparation=dict(frames={'0':dict(pcd_sha256='a'*64), '1':dict(pcd_sha256='b'*64)}))
    return frozen, plan


def test_all_rows_count_and_accuracy_is_separate_from_generation():
    frozen, plan = fixture()
    target = np.eye(4)
    target[0, 3] = 1
    result = score(frozen, log_bytes([np.eye(4), target]), plan)
    assert result['totals']['spatialrust'] == dict(denominator=1, generated=1, accurate=0)
    assert result['totals']['open3d'] == dict(denominator=1, generated=0, accurate=0)
    assert result['rows'][0]['accuracy']['translation_error'] == 1
    assert 'accuracy' not in frozen['rows'][0]


@pytest.mark.parametrize('mode', ['missing_row', 'wrong_hash', 'different_plan'])
def test_evaluation_rejects_missing_rows_or_binding_mismatch(mode):
    frozen, plan = fixture()
    if mode == 'missing_row':
        frozen['rows'].pop()
    elif mode == 'wrong_hash':
        frozen['rows'][0]['report']['input_file_sha256']['source'] = 'c'*64
    else:
        frozen['plan']['seeds'] = [8]
    with pytest.raises(ValueError):
        score(frozen, log_bytes([np.eye(4), np.eye(4)]), plan)


def test_plan_binds_runner_controls(tmp_path):
    root = Path(__file__).resolve().parents[3]
    plan, _ = load_plan(root / 'benchmarks/redwood-livingroom1-plan.json')
    plan['controls']['coarse_leaf_m'] = .2
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='controls differ'):
        load_plan(path)
