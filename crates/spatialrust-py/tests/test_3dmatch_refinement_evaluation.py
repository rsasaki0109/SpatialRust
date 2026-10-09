"""Paired controls prevent attributing initializer or gate changes to trimming."""
import copy
from pathlib import Path

import numpy as np
import pytest


def fixture():
    initial, final = np.eye(4), np.eye(4)
    initial[0, 3], final[0, 3] = .1, .3
    binding = dict(source='a' * 64, target='b' * 64)
    report = dict(input_file_sha256=binding, initial_transform_supplied=True,
        initial_transform_source_to_target=initial.tolist(), transform_source_to_target=final.tolist(),
        schedule=[dict(leaf=None, max_distance=.05, iterations=50, trim_fraction=1.)],
        stages=[dict(leaf_metres=None, max_correspondence_distance_metres=.05,
                     convergence_criteria={}, trim_fraction=1.)])
    original = copy.deepcopy(report)
    original['initial_transform_supplied'] = False
    trimmed = copy.deepcopy(report)
    trimmed['transform_source_to_target'] = initial.tolist()
    trimmed['schedule'][0]['trim_fraction'] = trimmed['stages'][0]['trim_fraction'] = .5
    comparison = dict(schema='spatialrust.public-global-comparison.v1', native_sha256='c' * 64,
        input_file_sha256=binding, seeds=[7],
        rows=[dict(method='spatialrust', seed=7, status='success', report=original)])
    study = dict(schema='spatialrust.public-refinement-study.v1', native_sha256='c' * 64,
        input_file_sha256=binding, rows=[dict(seed=7, trim_fraction=1., report=report),
                                       dict(seed=7, trim_fraction=.5, report=trimmed)])
    return study, comparison


@pytest.mark.parametrize('change', [None, 'missing', 'duplicate', 'initial', 'control_pose', 'gate', 'iterations', 'trim'])
def test_paired_score_and_control_rejection(monkeypatch, change):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    from evaluate_3dmatch_refinement import evaluate_replay
    study, comparison = fixture()
    if change == 'missing':
        study['rows'].pop(0)
    elif change == 'duplicate':
        study['rows'].append(copy.deepcopy(study['rows'][0]))
    elif change == 'initial':
        study['rows'][1]['report']['initial_transform_source_to_target'][0][3] = .2
    elif change == 'control_pose':
        study['rows'][0]['report']['transform_source_to_target'][0][3] = .2
    elif change == 'gate':
        study['rows'][1]['report']['stages'][0]['max_correspondence_distance_metres'] = .1
    elif change == 'iterations':
        study['rows'][1]['report']['schedule'][0]['iterations'] = 10
    elif change == 'trim':
        study['rows'][1]['report']['stages'][0]['trim_fraction'] = .8
    if change:
        with pytest.raises((ValueError, AssertionError)):
            evaluate_replay(study, comparison, np.eye(4), np.eye(6))
    else:
        result = evaluate_replay(study, comparison, np.eye(4), np.eye(6))
        assert result['summary'] == {'1.0': dict(planned=1, correct=0), '0.5': dict(planned=1, correct=1)}
        assert result['exact_untrimmed_controls'] == 1
        assert result['deployment_policy_established'] is False
