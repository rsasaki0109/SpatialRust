from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import inspect_github_ci
    return inspect_github_ci


def job(i, status='queued', conclusion=None):
    return dict(id=i, status=status, conclusion=conclusion, runner_id=0)


def test_more_than_one_hundred_jobs_is_not_truncated(module):
    pages = [dict(total_count=103, jobs=[job(i) for i in range(100)]),
             dict(total_count=103, jobs=[job(100), job(101), job(102, 'completed', 'failure')])]
    paths = []
    def fetch(path):
        paths.append(path)
        return pages.pop(0)
    jobs = module.all_jobs(fetch, 123)
    assert len(jobs) == 103 and len(paths) == 2
    assert module.summarize(jobs)['completed_conclusions'] == {'failure': 1}
    assert not module.summarize(jobs)['all_success']


@pytest.mark.parametrize('second', [dict(total_count=2, jobs=[]), dict(total_count=2, jobs=[job(1)])])
def test_missing_or_duplicate_page_fails_closed(module, second):
    pages = [dict(total_count=2, jobs=[job(1)]), second]
    with pytest.raises(ValueError):
        module.all_jobs(lambda path: pages.pop(0), 123)


def test_zero_skipped_or_pending_jobs_do_not_establish_success(module):
    assert not module.summarize([])['all_success']
    assert not module.summarize([job(1, 'completed', 'skipped')])['all_success']
    assert not module.summarize([job(1, 'in_progress')])['all_success']
    assert module.summarize([job(1, 'completed', 'success')])['all_success']
