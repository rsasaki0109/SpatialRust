from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import study_global_candidate_selection
    return study_global_candidate_selection


def candidate(forward, reverse, correct=False, method='spatialrust'):
    return dict(status='success', method=method, publisher_correct=correct,
                metrics=dict(forward=dict(fraction=forward, rmse=.01 if forward else None),
                             reverse=dict(fraction=reverse, rmse=.01 if reverse else None)))


def test_balanced_rule_penalizes_one_direction_and_ignores_truth(module):
    rows = [candidate(.9, .1, True), candidate(.6, .6)]
    assert module.select(rows, 'forward', ('spatialrust',)) == 0
    assert module.select(rows, 'balanced', ('spatialrust',)) == 1
    for row in rows:
        row['publisher_correct'] = not row['publisher_correct']
    assert module.select(rows, 'balanced', ('spatialrust',)) == 1


def test_ties_method_pools_and_failed_candidates(module):
    rows = [dict(status='error', method='spatialrust'), candidate(.5, .5), candidate(.5, .5, method='open3d')]
    assert module.select(rows, 'balanced', ('spatialrust', 'open3d')) == 1
    assert module.select(rows, 'balanced', ('open3d',)) == 2
    assert module.select(rows[:1], 'balanced', ('spatialrust',)) is None
    assert module.rank_key(candidate(0, 0)['metrics'], 'balanced')[0] == 0


@pytest.mark.parametrize('field,value', [('fraction', -1), ('fraction', float('nan')),
    ('fraction', True), ('fraction', '0.5'), ('rmse', -1), ('rmse', float('inf')), ('rmse', None), ('rmse', '0.1')])
def test_invalid_metrics_rejected(module, field, value):
    metrics = candidate(.5, .5)['metrics']
    metrics['forward'][field] = value
    with pytest.raises(ValueError):
        module.rank_key(metrics, 'balanced')


def test_distance_scores_agree_with_numpy(module):
    import numpy as np
    import spatialrust as sr
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    query = sr.PointCloud.from_xyz(xyz + np.float32(.01))
    index = sr.DistanceSupportIndex(sr.PointCloud.from_xyz(xyz))
    score = module.support(index, query, .05)
    assert score['count'] == 3 and score['fraction'] == 1
    distances = np.linalg.norm((xyz + np.float32(.01)) - xyz, axis=1)
    assert score['rmse'] == pytest.approx(np.sqrt(np.mean(distances ** 2)), rel=1e-6)
    assert module.support(index, query, .001)['rmse'] is None
