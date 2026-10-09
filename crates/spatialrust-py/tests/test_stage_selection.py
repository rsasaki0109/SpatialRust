from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import study_stage_selection
    return study_stage_selection


def report():
    identity = np.eye(4).tolist()
    return dict(initial_transform_source_to_target=identity, transform_source_to_target=identity,
                stages=[dict(name='fine', transform_source_to_target=identity)])


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'final', 'nonrigid'])
def test_invalid_stage_sequence(module, change):
    value = report()
    if change == 'duplicate':
        value['stages'][0]['name'] = 'initial'
    elif change == 'missing':
        value['stages'] = []
    elif change == 'final':
        value['transform_source_to_target'] = np.zeros((4, 4)).tolist()
    else:
        value['initial_transform_source_to_target'][0][0] = 2
    with pytest.raises(ValueError):
        module.saved_stages(value)


def test_stage_selection_tie_preserves_initializer(module):
    stages = module.saved_stages(report())
    assert [name for name, pose in stages] == ['initial', 'fine']
    metrics = dict(forward=dict(fraction=.5, rmse=.01), reverse=dict(fraction=.6, rmse=.02))
    assert max(range(2), key=lambda i: module.rank_key(metrics, 'balanced')) == 0


def test_motion_uses_source_centroid_not_translation_component(module):
    centroid = np.array([10., 20., 30.])
    pose = np.eye(4)
    pose[:3, :3] = np.diag([-1., -1., 1.])
    pose[:3, 3] = centroid - pose[:3, :3] @ centroid
    motion = module.pose_motion(np.eye(4), pose, centroid)
    assert motion['rotation_degrees'] == pytest.approx(180)
    assert motion['source_centroid_displacement_metres'] == pytest.approx(0)
    pose[:3, 3] += [0., 0., .3]
    assert module.pose_motion(np.eye(4), pose, centroid)['source_centroid_displacement_metres'] == pytest.approx(.3)


def test_nonfinite_centroid_rejected(module):
    with pytest.raises(ValueError):
        module.pose_motion(np.eye(4), np.eye(4), [0, float('nan'), 0])


def test_summary_preserves_failures_and_both_directions(module):
    def row(values, forward, balanced):
        return dict(status='success', stages=[dict(correct=v) for v in values],
                    selected=dict(initial=0, final=2, forward=forward, balanced=balanced))
    rows = [row([True, True, False], 0, 2), row([False, False, True], 1, 2), dict(status='error')]
    summary = module.summarize(rows, 3)
    assert summary['forward'] == dict(planned=3, evaluated=2, correct=1, gained_vs_final=1,
                                      lost_vs_final=1, unavailable=1)
    assert summary['balanced']['correct'] == 1
    assert summary['initial']['lost_vs_final'] == 1


def test_motion_guard_boundary_and_labels(module):
    stages = [dict(correct=False), dict(correct=True, motion_from_initial=dict(source_centroid_displacement_metres=.2))]
    assert module.motion_selection(stages, .2) == 1
    stages[-1]['motion_from_initial']['source_centroid_displacement_metres'] = .2001
    assert module.motion_selection(stages, .2) == 0
    stages[0]['correct'], stages[1]['correct'] = True, False
    assert module.motion_selection(stages, .2) == 0


@pytest.mark.parametrize('limit', [0, -.1, float('nan'), True])
def test_invalid_motion_limit(module, limit):
    with pytest.raises(ValueError):
        module.motion_selection([{}, {}], limit)


def test_inconsistent_rule_denominator_rejected(module):
    row = dict(status='success', stages=[dict(correct=True)], selected=dict(final=0))
    with pytest.raises(ValueError):
        module.summarize([row], 0)
