"""Plot data preserves undefined scores, failed denominators and seed pairing."""
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import render_refinement_batch
    return render_refinement_batch


def fixture():
    return dict(schema='spatialrust.3dmatch-refinement-batch.v1', cases=[
        dict(case_id='a', scene='scene', planned_seeds=2, status='completed', evaluation=dict(rows=[
            dict(seed=7, trim_fraction=1., score_squared=.01, correct=True),
            dict(seed=7, trim_fraction=.8, score_squared=.05, correct=False),
            dict(seed=8, trim_fraction=1., score_squared=None, correct=False),
            dict(seed=8, trim_fraction=.8, score_squared=.02, correct=True)])),
        dict(case_id='b', scene='scene', planned_seeds=2, status='case_error')])


def test_undefined_pair_and_failed_case_are_retained(module):
    rows, failed, planned = module.paired_rows(fixture(), .8)
    assert planned == 4 and len(rows) == 2 and len(failed) == 1
    assert rows[0]['category'] == 'lost'
    assert rows[1]['category'] == 'gained' and rows[1]['baseline'] is None


@pytest.mark.parametrize('change', ['duplicate', 'negative', 'nan', 'label', 'missing'])
def test_invalid_plot_data_is_rejected(module, change):
    study = fixture()
    rows = study['cases'][0]['evaluation']['rows']
    if change == 'duplicate':
        rows.append(rows[0])
    elif change == 'negative':
        rows[0]['score_squared'] = -.1
    elif change == 'nan':
        rows[0]['score_squared'] = float('nan')
    elif change == 'label':
        rows[0]['correct'] = False
    else:
        rows.pop()
    with pytest.raises(ValueError):
        module.paired_rows(study, .8)
