"""No output-gate relaxation and explicit native-policy/pose-direction binding."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from dense_rgbd_protocol import CONTROLS, final_overlap_plan, validate_final_overlap_plan
from run_tum_dense_rgbd import align_dense_rgbd
from run_tum_final_overlap_rgbd import align_final_overlap_rgbd
from tum_sequence_reference import file_sha256

ROOT = Path(__file__).resolve().parents[3]


def fixed_plan():
    path = ROOT/'benchmarks/tum-fr1-xyz-dense-rgbd-plan.json'
    parent = json.loads(path.read_text(encoding='utf-8'))
    return parent, final_overlap_plan(parent, file_sha256(path))


def test_revision_preserves_every_numeric_control_and_final_acceptance():
    parent, plan = fixed_plan()
    validate_final_overlap_plan(plan)
    assert plan['controls'] == parent['controls']
    assert plan['acceptance'] == parent['acceptance']
    for name in ('calibration', 'limits', 'sequence'):
        assert plan[name] == parent[name]


@pytest.mark.parametrize('group,key,value', [
    ('acceptance', 'maximum_first_pose_ate_rmse_m', .2),
    ('acceptance', 'minimum_generated_fraction', .9),
    ('controls', 'min_support_fraction', .4),
    ('controls', 'max_depth_m', 10.),
])
def test_output_and_accuracy_gate_changes_are_rejected(group, key, value):
    _, plan = fixed_plan()
    changed = copy.deepcopy(plan)
    changed[group][key] = value
    with pytest.raises(ValueError):
        validate_final_overlap_plan(changed)


def fake_native(final_support_only):
    def solve(*args, **kwargs):
        np.testing.assert_array_equal(kwargs['initial'], np.eye(4))
        assert args[1].dtype == args[3].dtype == np.float32
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        return rotation, np.array([.01, .02, .03]), dict(valid_source_pixels=1024,
            correspondences=1024, correspondence_fraction=1., photometric_rmse=0.,
            depth_rmse_m=0., final_support_only=final_support_only)
    return solve


def args():
    camera = fixed_plan()[0]['calibration']
    depth = np.full((32, 32), 5000, dtype=np.uint16)
    gray = np.full((32, 32), 100, dtype=np.uint8)
    return depth, depth.copy(), gray, gray.copy(), camera, dict(dense_rgbd_odometry=CONTROLS)


@pytest.mark.parametrize('actual', [False, 'true'])
def test_revised_adapter_rejects_wrong_or_ambiguous_native_policy(monkeypatch, actual):
    monkeypatch.setattr(sr, 'estimate_dense_rgbd_odometry', fake_native(actual))
    with pytest.raises(ValueError, match='policy'):
        align_final_overlap_rgbd(*args())


def test_original_adapter_rejects_relabeling_the_new_native_policy(monkeypatch):
    monkeypatch.setattr(sr, 'estimate_dense_rgbd_odometry', fake_native(True))
    with pytest.raises(ValueError, match='policy'):
        align_dense_rgbd(*args())


def test_revised_pose_is_inverted_exactly_once(monkeypatch):
    monkeypatch.setattr(sr, 'estimate_dense_rgbd_odometry', fake_native(True))
    inverse, report = align_final_overlap_rgbd(*args())
    forward = np.array(report['current_from_previous'])
    np.testing.assert_allclose(inverse, np.linalg.inv(forward), rtol=0, atol=1e-15)
    assert report['final_support_only'] is True
