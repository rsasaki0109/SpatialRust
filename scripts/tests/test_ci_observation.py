"""Reject incomplete pagination and CI evidence from another commit or attempt."""
import copy
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location('ci', Path(__file__).resolve().parents[1] / 'inspect_github_ci.py')
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
HEAD = 'a' * 40


def run():
    return dict(id=42, head_sha=HEAD, run_attempt=1, name='CI', status='queued', conclusion=None, html_url='https://github.com/example/project/actions/runs/42')


def job(number):
    return dict(id=number, run_id=42, head_sha=HEAD, run_attempt=1, status='completed', conclusion='success', runner_id=1)


class CiObservationTests(unittest.TestCase):
    def test_rerun_uses_only_the_explicit_attempt_endpoint(self):
        requests = []
        record = run()
        record['run_attempt'] = 2
        retried = job(1)
        retried['run_attempt'] = 2
        def fetch(path):
            requests.append(path)
            if '/jobs?' in path:
                self.assertEqual(path, '/actions/runs/42/attempts/2/jobs?per_page=100&page=1')
                return dict(total_count=1, jobs=[retried])
            return record
        result = ci.observe_run(fetch, 42, HEAD)
        self.assertEqual(result['run_attempt'], 2)
        self.assertTrue(result['summary']['all_success'])
        self.assertNotIn('/actions/runs/42/jobs?per_page=100&page=1', requests)

    def test_explicit_attempt_rejects_stale_jobs(self):
        record = run()
        record['run_attempt'] = 2
        def fetch(path):
            return dict(total_count=1, jobs=[job(1)]) if '/jobs?' in path else record
        with self.assertRaisesRegex(ValueError, 'different run'):
            ci.observe_run(fetch, 42, HEAD)

    def test_invalid_explicit_attempt_rejected_before_fetch(self):
        for attempt in (0, -1, True, '2'):
            with self.subTest(attempt=attempt), self.assertRaisesRegex(ValueError, 'positive integer'):
                ci.all_jobs(lambda path: self.fail('invalid attempt must not fetch'), 42, attempt)

    def test_all_103_jobs_are_required(self):
        requests = []
        def fetch(path):
            requests.append(path)
            if path.endswith('&page=1'):
                return dict(total_count=103, jobs=[job(i) for i in range(100)])
            return dict(total_count=103, jobs=[job(i) for i in range(100, 103)])
        jobs = ci.all_jobs(fetch, 42)
        self.assertEqual(len(jobs), 103)
        self.assertEqual(len(requests), 2)
        self.assertTrue(ci.summarize(jobs)['all_success'])

    def test_changed_total_rejected_even_when_second_page_matches_new_total(self):
        def fetch(path):
            if path.endswith('&page=1'):
                return dict(total_count=103, jobs=[job(i) for i in range(100)])
            return dict(total_count=102, jobs=[job(100), job(101)])
        with self.assertRaisesRegex(ValueError, 'count changed'):
            ci.all_jobs(fetch, 42)

    def test_duplicate_or_truncated_page_rejected(self):
        for final_jobs in ([], [job(99), job(100), job(101)]):
            with self.subTest(final_jobs=final_jobs):
                def fetch(path):
                    return dict(total_count=103, jobs=[job(i) for i in range(100)] if path.endswith('&page=1') else final_jobs)
                with self.assertRaises(ValueError):
                    ci.all_jobs(fetch, 42)

    def test_old_commit_cannot_validate_current_head(self):
        with self.assertRaisesRegex(ValueError, 'expected head'):
            ci.observe_run(lambda path: run(), 42, 'b' * 40)

    def test_job_identity_and_attempt_must_match(self):
        for key, value in [('run_id', 43), ('head_sha', 'b' * 40), ('run_attempt', 2)]:
            with self.subTest(key=key):
                wrong_job = job(1)
                wrong_job[key] = value
                def fetch(path):
                    return dict(total_count=1, jobs=[wrong_job]) if '/jobs?' in path else run()
                with self.assertRaisesRegex(ValueError, 'different run'):
                    ci.observe_run(fetch, 42, HEAD)

    def test_rerun_during_collection_rejected(self):
        calls = 0
        def fetch(path):
            nonlocal calls
            if '/jobs?' in path:
                return dict(total_count=1, jobs=[job(1)])
            calls += 1
            record = copy.deepcopy(run())
            if calls == 2:
                record['run_attempt'] = 2
            return record
        with self.assertRaisesRegex(ValueError, 'changed while fetching'):
            ci.observe_run(fetch, 42, HEAD)

    def test_queued_skipped_failed_and_empty_are_distinct_from_all_success(self):
        self.assertFalse(ci.summarize([])['all_success'])
        for status, conclusion in [('queued', None), ('completed', 'skipped'), ('completed', 'failure')]:
            with self.subTest(status=status, conclusion=conclusion):
                unfinished = job(2)
                unfinished.update(status=status, conclusion=conclusion, runner_id=0)
                result = ci.summarize([job(1), unfinished])
                self.assertFalse(result['all_success'])
                self.assertEqual(result['assigned'], 1)
                if status == 'completed':
                    self.assertEqual(result['completed_conclusions'][conclusion], 1)


if __name__ == '__main__':
    unittest.main()
