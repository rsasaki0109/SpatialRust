"""Read complete public Actions job metadata without treating a page as a run."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import urllib.request


def all_jobs(fetch, run_id):
    jobs, total = {}, None
    for page in range(1, 21):
        data = fetch(f'/actions/runs/{run_id}/jobs?per_page=100&page={page}')
        total = data['total_count']
        if type(total) is not int or not 0 <= total <= 2000:
            raise ValueError('unsupported job count')
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--run-ids', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repo) or any(i <= 0 for i in args.run_ids):
        parser.error('require a repository owner/name and positive run IDs')
    if args.output.exists():
        parser.error('output already exists')
    def fetch(path):
        request = urllib.request.Request('https://api.github.com/repos/'+args.repo+path,
                                         headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'SpatialRust-CI-observation'})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    runs = []
    for run_id in args.run_ids:
        run = fetch(f'/actions/runs/{run_id}')
        jobs = all_jobs(fetch, run_id)
        runs.append(dict(run_id=run_id, head_sha=run['head_sha'], name=run['name'],
                         status=run['status'], conclusion=run['conclusion'], html_url=run['html_url'],
                         summary=summarize(jobs), jobs=jobs))
    output = dict(schema='spatialrust.ci-observation.v1', repository=args.repo,
                  checked_at_utc=datetime.now(timezone.utc).isoformat(), pagination_complete=True,
                  limits='Public metadata sampled across requests; queued run status does not imply all jobs are queued.', runs=runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps([{k: r[k] for k in ('run_id', 'name', 'head_sha', 'summary')} for r in runs]))


if __name__ == '__main__':
    main()
