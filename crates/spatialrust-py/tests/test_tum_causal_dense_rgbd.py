"""Causal seed direction/ownership, independent auditing and fixed full gates."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from audit_tum_dense_rgbd import check_causal_initialization
from causal_dense_rgbd_protocol import ESTIMATOR, PARENTS, validate_causal_dense_plan
from dense_rgbd_protocol import CONTROLS
from run_tum_causal_dense_rgbd import PreviousMotionDenseRgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256, prepare
from test_tum_sequence_reference import fixture

PLAN = PARENTS['rgbd_dataset_freiburg1_xyz'].with_name('tum-fr1-xyz-causal-dense-plan.json')


def args():
    depth = np.full((32, 32), 5000, dtype=np.uint16)
    gray = np.full((32, 32), 100, dtype=np.uint8)
    camera = json.loads(PLAN.read_text(encoding='utf-8'))['calibration']
    return depth, depth.copy(), gray, gray.copy(), camera, dict(dense_rgbd_odometry=CONTROLS)


def native_with_pose(pose, calls, *, fail_at=None):
    def solve(*values, **kwargs):
        calls.append(kwargs['initial'].copy())
        if len(calls) == fail_at:
            raise ValueError('controlled native failure')
        source = values[1]
        valid = int(np.count_nonzero((source >= .1) & (source <= 5.)))
        return pose[:3, :3].copy(), pose[:3, 3].copy(), dict(valid_source_pixels=valid,
            correspondences=valid, correspondence_fraction=1., photometric_rmse=0.,
            depth_rmse_m=0., final_support_only=True)
    return solve


def nonidentity():
    pose = np.array([[0., -1., 0., .01], [1., 0., 0., .02], [0., 0., 1., .03], [0., 0., 0., 1.]])
    return pose


def test_seed_is_prior_forward_increment_and_does_not_alias_returned_rows(monkeypatch):
    pose, calls = nonidentity(), []
    monkeypatch.setattr(sr, 'estimate_dense_rgbd_odometry', native_with_pose(pose, calls))
    aligner = PreviousMotionDenseRgbd()
    inverse, report = aligner.align(*args())
    np.testing.assert_array_equal(calls[0], np.eye(4))
    np.testing.assert_allclose(inverse, np.linalg.inv(pose), rtol=0, atol=1e-15)
    assert report['initial_previous_pair_index'] is None
    assert check_causal_initialization(report, None, 1) == 0
    report['current_from_previous'][0][3] = 999
    report['initial_current_from_previous'][0][3] = 999
    inverse[:] = 0
    _, second = aligner.align(*args())
    np.testing.assert_array_equal(calls[1], pose)
    assert second['initial_previous_pair_index'] == 1
    assert check_causal_initialization(second, pose, 2) == 0


def test_instances_are_independent_and_native_failure_does_not_advance_seed(monkeypatch):
    pose, calls = nonidentity(), []
    monkeypatch.setattr(sr, 'estimate_dense_rgbd_odometry', native_with_pose(pose, calls, fail_at=2))
    a, b = PreviousMotionDenseRgbd(), PreviousMotionDenseRgbd()
    a.align(*args())
    with pytest.raises(ValueError, match='controlled'):
        a.align(*args())
    _, resumed = a.align(*args())
    np.testing.assert_array_equal(calls[2], pose)
    assert resumed['initial_previous_pair_index'] == 1
    _, fresh = b.align(*args())
    np.testing.assert_array_equal(calls[3], np.eye(4))
    assert fresh['initial_previous_pair_index'] is None


@pytest.mark.parametrize('group,key,value', [
    ('dense_initialization', 'time_rescaling', True),
    ('dense_initialization', 'retries', True),
    ('dense_initialization', 'reset_after_loss', True),
    ('dense_initialization', 'first_pair', 'reference'),
    ('acceptance', 'minimum_generated_fraction', .99),
    ('acceptance', 'maximum_first_pose_ate_rmse_m', .2),
    ('controls', 'min_support_fraction', .4),
    ('calibration', 'fx', 517.3),
])
def test_fixed_plan_rejects_seed_and_final_gate_changes(group, key, value):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    validate_causal_dense_plan(plan)
    plan[group][key] = value
    with pytest.raises(ValueError):
        validate_causal_dense_plan(plan)


@pytest.mark.parametrize('field,value', [
    ('initial_previous_pair_index', True), ('initial_previous_pair_index', 0),
    ('initial_previous_pair_index', 2), ('initial_seed_policy', 'reference'),
    ('initial_current_from_previous', np.eye(4).tolist()),
])
def test_independent_seed_audit_rejects_wrong_lineage_and_pose(field, value):
    pose = nonidentity()
    report = dict(initial_previous_pair_index=1, initial_seed_policy='previous-relative-increment',
                  initial_current_from_previous=pose.tolist())
    report[field] = value
    with pytest.raises(ValueError):
        check_causal_initialization(report, pose, 2)


@pytest.mark.parametrize('bad', ['missing-index', 'nonidentity', 'prior-pose'])
def test_first_pair_requires_explicit_identity_without_prior_history(bad):
    report = dict(initial_previous_pair_index=None, initial_seed_policy='previous-relative-increment',
                  initial_current_from_previous=np.eye(4).tolist())
    previous = None
    if bad == 'missing-index':
        report.pop('initial_previous_pair_index')
    elif bad == 'nonidentity':
        report['initial_current_from_previous'] = nonidentity().tolist()
    else:
        previous = np.eye(4)
    with pytest.raises(ValueError):
        check_causal_initialization(report, previous, 1)


def test_plan_preserves_both_parents_controls_and_acceptance():
    for parent_path in PARENTS.values():
        parent = json.loads(parent_path.read_text(encoding='utf-8'))
        path = parent_path.with_name(parent_path.name.replace('-dense-final-overlap-plan', '-causal-dense-plan'))
        plan = validate_causal_dense_plan(json.loads(path.read_text(encoding='utf-8')))
        for key in ('controls', 'acceptance', 'calibration', 'limits'):
            assert plan[key] == parent[key]


@pytest.mark.parametrize('fail_support', [False, True])
def test_generator_checks_original_support_and_never_reuses_a_rejected_chain(tmp_path, monkeypatch, fail_support):
    archive, digest = fixture(tmp_path, count=3)
    prepared = tmp_path/'prepared'
    prepare(archive, digest, PLAN, prepared)
    calls = []
    monkeypatch.setattr(sr, 'estimate_dense_rgbd_odometry', native_with_pose(np.eye(4), calls))
    if fail_support:
        monkeypatch.setattr(sr, 'distance_gated_support', lambda *args: (0, 0., None))
    aligner = PreviousMotionDenseRgbd()
    result = generate(prepared, file_sha256(prepared/'manifest.json'), tmp_path/'result',
        dense_rgbd_aligner=aligner.align, method=ESTIMATOR, additional_bindings=[__file__])
    rows = json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    if fail_support:
        assert result['generated_poses'] == 1 and len(calls) == 1
        assert [r['status'] for r in rows] == ['success', 'error', 'error']
        assert 'earlier frame' in rows[2]['error']
    else:
        assert result['generated_poses'] == 3 and len(calls) == 2
        assert rows[2]['registration']['initial_previous_pair_index'] == 1
        assert rows[2]['registration']['support_fraction'] >= .5
