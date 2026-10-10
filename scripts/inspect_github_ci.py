"""Read complete public Actions job metadata without treating a page as a run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.request


def all_jobs(fetch, run_id, run_attempt=None):
    if run_attempt is not None and (type(run_attempt) is not int or run_attempt <= 0):
        raise ValueError('run attempt must be a positive integer')
    endpoint = f'/actions/runs/{run_id}'
    if run_attempt is not None:
        endpoint += f'/attempts/{run_attempt}'
    jobs, total = {}, None
    for page in range(1, 21):
        data = fetch(f'{endpoint}/jobs?per_page=100&page={page}')
        reported_total = data['total_count']
        if type(reported_total) is not int or not 0 <= reported_total <= 2000:
            raise ValueError('unsupported job count')
        if total is not None and reported_total != total:
            raise ValueError('job count changed across pages; retry a fresh snapshot')
        total = reported_total
        for job in data['jobs']:
            if job['id'] in jobs:
                raise ValueError('duplicate job across pages; retry a fresh snapshot')
            jobs[job['id']] = job
        if len(jobs) == total:
            return list(jobs.values())
        if len(jobs) > total or not data['jobs']:
            raise ValueError('incomplete or changing job pagination')
    raise ValueError('job pagination exceeded bound')


def summarize(jobs):
    states = Counter(job['status'] for job in jobs)
    conclusions = Counter(job['conclusion'] or 'unset' for job in jobs if job['status'] == 'completed')
    return dict(observed_jobs=len(jobs), assigned=sum(bool(job.get('runner_id')) for job in jobs),
                states=dict(states), completed_conclusions=dict(conclusions),
                all_success=bool(jobs) and all(job['status'] == 'completed' and job['conclusion'] == 'success' for job in jobs))


def observe_run(fetch, run_id, expected_head_sha=None):
    run = fetch(f'/actions/runs/{run_id}')
    if run['id'] != run_id:
        raise ValueError('run ID does not match request')
    if expected_head_sha is not None and run['head_sha'] != expected_head_sha:
        raise ValueError('run does not belong to the expected head SHA')
    jobs = all_jobs(fetch, run_id, run['run_attempt'])
    for job in jobs:
        if (job['run_id'], job['head_sha'], job['run_attempt']) != (run_id, run['head_sha'], run['run_attempt']):
            raise ValueError('job belongs to a different run, head SHA or attempt')
    current = fetch(f'/actions/runs/{run_id}')
    if (current['id'], current['head_sha'], current['run_attempt']) != (run_id, run['head_sha'], run['run_attempt']):
        raise ValueError('run identity or attempt changed while fetching jobs; retry a fresh snapshot')
    return dict(run_id=run_id, head_sha=run['head_sha'], run_attempt=run['run_attempt'], name=run['name'],
                status=current['status'], conclusion=current['conclusion'], html_url=run['html_url'],
                summary=summarize(jobs), jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--run-ids', type=int, nargs='+', required=True)
    parser.add_argument('--expected-head-sha', help='Require every run and job to belong to this full commit SHA')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo) or any(i <= 0 for i in args.run_ids):
        parser.error('require a repository owner/name and positive run IDs')
    if len(set(args.run_ids)) != len(args.run_ids):
        parser.error('run IDs must be distinct')
    if args.expected_head_sha is not None and not re.fullmatch(r'[0-9a-f]{40}', args.expected_head_sha):
        parser.error('expected head SHA must be 40 lowercase hexadecimal characters')
    if args.output.exists():
        parser.error('output already exists')
    def fetch(path):
        request = urllib.request.Request('https://api.github.com/repos/'+args.repo+path,
                                         headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'SpatialRust-CI-observation'})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    runs = []
    for run_id in args.run_ids:
        runs.append(observe_run(fetch, run_id, args.expected_head_sha))
    output = dict(schema='spatialrust.ci-observation.v1', repository=args.repo,
                  expected_head_sha=args.expected_head_sha,
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), pagination_complete=True,
                  limits='Public metadata sampled across requests; queued run status does not imply all jobs are queued.', runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps([{k: r[k] for k in ('run_id', 'name', 'head_sha', 'summary')} for r in runs]))


if __name__ == '__main__':
    main()
