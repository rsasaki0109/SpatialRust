"""Fixed dense plans and explicit adjacent-depth orchestration with old gates."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from dense_rgbd_protocol import ESTIMATOR, validate_dense_plan
from run_tum_sequence import checked_depth, decode_depth, generate
from tum_sequence_reference import file_sha256, prepare
from test_tum_sequence_reference import fixture, PLAN

DENSE_PLAN = PLAN.with_name('tum-fr1-xyz-dense-rgbd-plan.json')


@pytest.mark.parametrize('field,value', [
    ('depth_weight', .5), ('iterations', [5, 10, 20]),
    ('max_working_bytes', True), ('min_correspondence_fraction', .1),
    ('min_depth', 0), ('max_depth_difference', .1),
])
def test_dense_options_are_fixed_before_full_motion(field, value):
    plan = json.loads(DENSE_PLAN.read_text(encoding='utf-8'))
    validate_dense_plan(plan)
    plan['controls']['dense_rgbd_odometry'][field] = value
    with pytest.raises(ValueError):
        validate_dense_plan(plan)


@pytest.mark.parametrize('group,field,value', [
    ('acceptance', 'maximum_first_pose_ate_rmse_m', .2),
    ('acceptance', 'minimum_generated_fraction', .99),
    ('controls', 'min_support_fraction', .1),
    ('calibration', 'fx', 517.3),
])
def test_dense_model_cannot_relax_existing_geometry_or_accuracy(group, field, value):
    plan = json.loads(DENSE_PLAN.read_text(encoding='utf-8'))
    changed = copy.deepcopy(plan)
    changed[group][field] = value
    with pytest.raises(ValueError):
        validate_dense_plan(changed)


def test_generator_supplies_both_adjacent_depths_and_keeps_original_support(tmp_path):
    archive, sha = fixture(tmp_path, count=3)
    prepared = tmp_path/'prepared'
    manifest = prepare(archive, sha, DENSE_PLAN, prepared)
    camera = json.loads(DENSE_PLAN.read_text(encoding='utf-8'))['calibration']
    depths = [decode_depth(checked_depth(prepared, f['depth'], 2**24), camera)
              for f in manifest['frames']]
    calls = []

    def adapter(previous_depth, current_depth, previous_gray, current_gray, camera, controls):
        assert previous_depth is not current_depth
        assert previous_gray.shape == current_gray.shape == (480, 640)
        ordinal = len(calls)
        np.testing.assert_array_equal(previous_depth, depths[ordinal])
        np.testing.assert_array_equal(current_depth, depths[ordinal+1])
        calls.append(ordinal)
        return np.eye(4), dict(explicit_depth_pair=True)

    result = generate(prepared, file_sha256(prepared/'manifest.json'), tmp_path/'result',
        method=ESTIMATOR, dense_rgbd_aligner=adapter, additional_bindings=[__file__])
    assert result['generated_poses'] == result['planned_poses'] == 3
    assert calls == [0, 1]
    rows = json.loads((tmp_path/'result/estimates.json').read_text())['poses']
    assert all(row['registration']['support_fraction'] >= .5 for row in rows[1:])


def test_missing_or_crosswired_adapters_fail_before_generation(tmp_path):
    for kwargs in ({}, {'rgbd_aligner': lambda *args: None},
                   {'dense_rgbd_aligner': lambda *args: None, 'pair_aligner': lambda *args: None}):
        with pytest.raises(ValueError):
            generate(tmp_path/'missing', '0'*64, tmp_path/'out', method=ESTIMATOR, **kwargs)
    with pytest.raises(ValueError):
        generate(tmp_path/'missing', '0'*64, tmp_path/'out', dense_rgbd_aligner=lambda *args: None)


def test_dense_failure_keeps_complete_timeline_without_chain_reset(tmp_path):
    archive, sha = fixture(tmp_path, count=3)
    prepared = tmp_path/'prepared'
    prepare(archive, sha, DENSE_PLAN, prepared)

    def fail(*args):
        raise ValueError('insufficient dense projective support')

    result = generate(prepared, file_sha256(prepared/'manifest.json'), tmp_path/'result',
        method=ESTIMATOR, dense_rgbd_aligner=fail, additional_bindings=[__file__])
    assert result['generated_poses'] == 1 and result['planned_poses'] == 3
    rows = json.loads((tmp_path/'result/estimates.json').read_text())['poses']
    assert [row['status'] for row in rows] == ['success', 'error', 'error']
    assert 'earlier frame' in rows[2]['error']
