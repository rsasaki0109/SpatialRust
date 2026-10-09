import copy
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import summarize_3dmatch_holdout
    return summarize_3dmatch_holdout


def fixture():
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    plan = dict(cases=[dict(case_id='good'), dict(case_id='failed')], seeds=[7, 8], ransac_iterations=10000,
        cases_manifest_sha256='manifest', native_sha256='native', open3d_native_sha256='other',
        comparison_controls=dict(gate=.2), primary_rules=dict(trim_fractions=[1., .8], motion_limit_metres=.2))
    originals = [dict(method='spatialrust', seed=seed, status='success', publisher_correct=correct,
        report=dict(initial_transform_source_to_target=copy.deepcopy(identity), transform_source_to_target=copy.deepcopy(identity)))
        for seed, correct in [(7, False), (8, True)]]
    baseline = dict(schema='spatialrust.3dmatch-live-comparison.v1', cases_manifest_sha256='manifest',
        seeds=[7, 8], ransac_iterations=10000, cases=[dict(case_id='good', status='completed', rows=originals,
            controls=dict(gate=.2), native_sha256='native', open3d_native_sha256='other'),
            dict(case_id='failed', status='case_error')])
    refinement = dict(schema='spatialrust.3dmatch-refinement-batch.v1', cases=[
        dict(case_id='good', planned_seeds=2, status='completed', evaluation=dict(rows=[
            dict(seed=seed, trim_fraction=fraction, correct=correct if fraction == 1 else True)
            for seed, correct in [(7, False), (8, True)] for fraction in [1., .8]])),
        dict(case_id='failed', planned_seeds=2, status='case_error')])
    stages = dict(schema='spatialrust.stage-selection.v1', leaf=.05, gate=.05, motion_limits_metres=[.2], rows=[
        dict(case_id='good', seed=seed, status='success', selected=dict(motion_0_2m=0), stages=[
            dict(pose=copy.deepcopy(identity), correct=not correct), dict(pose=copy.deepcopy(identity), correct=correct,
                motion_from_initial=dict(source_centroid_displacement_metres=.3))]) for seed, correct in [(7, False), (8, True)]])
    for row in stages['rows']:
        row['selected'] = {'motion_0.2m': 0}
    return plan, baseline, refinement, stages


def test_gains_losses_and_missing_case_denominator(module):
    summary = module.summarize_primary(*fixture())
    assert summary['baseline']['correct'] == 1 and summary['baseline']['planned'] == 4
    assert summary['trim_0_8']['correct'] == 2 and summary['trim_0_8']['gained'] == 1
    assert summary['motion_0_2m'] == dict(planned=4, evaluated=2, correct=1, gained=1, lost=1, unavailable=2)


@pytest.mark.parametrize('change', ['case_order', 'seed', 'control', 'duplicate', 'motion', 'missing_stage', 'endpoint'])
def test_unpaired_or_changed_results_rejected(module, change):
    plan, baseline, refinement, stages = copy.deepcopy(fixture())
    if change == 'case_order':
        baseline['cases'].reverse()
    elif change == 'seed':
        baseline['seeds'] = [7, 9]
    elif change == 'control':
        refinement['cases'][0]['evaluation']['rows'][0]['correct'] = True
    elif change == 'duplicate':
        refinement['cases'][0]['evaluation']['rows'].append(refinement['cases'][0]['evaluation']['rows'][0])
    elif change == 'motion':
        stages['rows'][0]['selected']['motion_0.2m'] = 1
    elif change == 'missing_stage':
        stages['rows'].pop()
    else:
        stages['rows'][0]['stages'][-1]['pose'][0][3] = 1
    with pytest.raises(ValueError):
        module.summarize_primary(plan, baseline, refinement, stages)
