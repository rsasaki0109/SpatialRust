"""Checkpointed paired trim validation on a fixed list of official-fragment cases."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from compare_3dmatch_cases import validate_cases


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def summarize(results, fractions):
    planned = sum(case['planned_seeds'] for case in results)
    summary = {}
    for fraction in fractions:
        paired = []
        for case in results:
            if case['status'] != 'completed':
                continue
            rows = case['evaluation']['rows']
            control = {row['seed']: row['correct'] for row in rows if row['trim_fraction'] == 1.}
            paired.extend((control[row['seed']], row['correct']) for row in rows if row['trim_fraction'] == fraction)
        summary[str(fraction)] = dict(planned=planned, evaluated=len(paired),
            correct=sum(b for _, b in paired), gained=sum(not a and b for a, b in paired),
            lost=sum(a and not b for a, b in paired), case_error_rows=planned-len(paired))
    return summary


def run_case(case, root, fractions, timeout, resume):
    directory = root / case['case_id']
    directory.mkdir(exist_ok=resume)
    checkpoint = directory / 'case.json'
    if resume and checkpoint.exists():
        result = json.loads(checkpoint.read_bytes())
        if result['case_id'] != case['case_id'] or result['planned_seeds'] != len(case['seeds']):
            raise ValueError('checkpoint case or denominator differs')
        if result['status'] == 'completed':
            for name, expected in result['artifact_sha256'].items():
                if digest(directory / name) != expected:
                    raise ValueError('checkpoint artifact bytes changed')
            if result['evaluation'] != json.loads((directory / 'evaluation/evaluation.json').read_bytes()):
                raise ValueError('checkpoint evaluation differs from its bound artifact')
        elif result['status'] != 'case_error':
            raise ValueError('unknown checkpoint status')
        return result
    result = dict(case_id=case['case_id'], scene=case['scene'], planned_seeds=len(case['seeds']), status='case_error')
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    replay = directory / 'replay'
    evaluation = directory / 'evaluation'
    commands = []
    if not (replay / 'study.json').exists():
        commands.append([sys.executable, str(Path(__file__).with_name('study_public_refinement.py')),
            '--source', case['source'], '--target', case['target'], '--reference', case['reference_file'],
            '--comparison', case['comparison_file'], '--trim-fractions', *map(str, fractions), '--output-dir', str(replay)])
    if not (evaluation / 'evaluation.json').exists():
        commands.append([sys.executable, str(Path(__file__).with_name('evaluate_3dmatch_refinement.py')),
            '--study', str(replay / 'study.json'), '--comparison', case['comparison_file'],
            '--reference', case['reference_file'], '--output-dir', str(evaluation)])
    try:
        for command in commands:
            completed = subprocess.run(command, env=environment, capture_output=True, timeout=timeout)
            if completed.returncode:
                raise ValueError(completed.stderr[-4000:].decode('utf-8', errors='replace'))
        replay_data = json.loads((replay / 'study.json').read_bytes())
        evaluated = json.loads((evaluation / 'evaluation.json').read_bytes())
        if replay_data['comparison_sha256'] != case['comparison_sha256'] or replay_data['reference_sha256'] != case['reference_sha256']:
            raise ValueError('replay input artifact bindings differ')
        if {row['trim_fraction'] for row in replay_data['rows']} != set(fractions):
            raise ValueError('replay fractions differ from the pre-fit plan')
        expected_slots = {(seed, fraction) for seed in case['seeds'] for fraction in fractions}
        slots = [(row['seed'], row['trim_fraction']) for row in evaluated['rows']]
        if len(slots) != len(expected_slots) or set(slots) != expected_slots:
            raise ValueError('evaluation omitted or duplicated planned rows')
        bindings = evaluated['input_artifact_sha256']
        if bindings != dict(study=digest(replay / 'study.json'), comparison=case['comparison_sha256'], reference=case['reference_sha256']):
            raise ValueError('post-fit evaluation bindings differ')
        result.update(status='completed', evaluation=evaluated,
            artifact_sha256={name: digest(directory / name) for name in ('replay/study.json', 'evaluation/evaluation.json')})
    except (subprocess.TimeoutExpired, ValueError) as error:
        result['error'] = str(error)
    atomic_json(checkpoint, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--comparisons', type=Path, required=True)
    parser.add_argument('--exclude-case-id', nargs='*', default=[])
    parser.add_argument('--trim-fractions', nargs='+', type=float, default=[1., .8])
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--case-timeout', type=int, default=1800)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.case_timeout <= 3600:
        parser.error('workers 1..4 and timeout 1..3600 required')
    if not 2 <= len(args.trim_fractions) <= 8 or len(set(args.trim_fractions)) != len(args.trim_fractions) or 1. not in args.trim_fractions or any(not np.isfinite(v) or not 0 < v <= 1 for v in args.trim_fractions):
        parser.error('require 2..8 distinct valid fractions including 1.0')
    manifest = json.loads(args.cases.read_bytes())
    all_cases = validate_cases(manifest)
    if not set(args.exclude_case_id) <= {case['case_id'] for case in all_cases}:
        parser.error('unknown excluded case ID')
    cases = []
    for original in all_cases:
        if original['case_id'] in args.exclude_case_id:
            continue
        case = dict(original)
        comparison_path = args.comparisons / case['case_id'] / 'comparison.json'
        comparison = json.loads(comparison_path.read_bytes())
        case.update(comparison_file=str(comparison_path.resolve()), comparison_sha256=digest(comparison_path),
                    reference_sha256=digest(case['reference_file']), seeds=comparison['seeds'])
        for name in ('source', 'target'):
            if digest(case[name]) != comparison['input_file_sha256'][name]:
                raise ValueError('baseline point-file binding differs')
        cases.append(case)
    if not cases:
        parser.error('no cases remain')
    repository = Path(__file__).resolve().parents[1]
    helpers = [Path(__file__), Path(__file__).with_name('study_public_refinement.py'),
        Path(__file__).with_name('evaluate_3dmatch_refinement.py'), Path(__file__).with_name('evaluate_pose_reference.py'),
        Path(__file__).with_name('evaluate_redwood_logs.py'), Path(__file__).with_name('redwood_reference.py'),
        Path(__file__).with_name('prepare_numpy_reference.py')] + [repository / 'crates/spatialrust-py/examples' / name
        for name in ('align_multiscale.py', 'align_point_clouds.py')]
    native = Path(importlib.import_module('spatialrust.spatialrust').__file__)
    if any(json.loads(Path(case['comparison_file']).read_bytes())['native_sha256'] != digest(native) for case in cases):
        raise ValueError('all comparisons must use the current native build')
    plan = dict(schema='spatialrust.3dmatch-refinement-plan.v1', cases_manifest_sha256=digest(args.cases),
        cases=cases, excluded_case_ids=args.exclude_case_id, fractions=args.trim_fractions,
        case_timeout_seconds=args.case_timeout, native_sha256=digest(native),
        source_sha256={str(path.resolve()): digest(path) for path in helpers},
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        thread_environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1'))
    if args.resume:
        if json.loads((args.output_dir / 'plan.json').read_bytes()) != plan:
            raise ValueError('resume plan, data, native or sources differ')
    else:
        args.output_dir.mkdir()
        atomic_json(args.output_dir / 'plan.json', plan)
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, case, args.output_dir, args.trim_fractions, args.case_timeout, args.resume) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results[result['case_id']] = result
    if digest(native) != plan['native_sha256'] or any(digest(path) != expected for path, expected in plan['source_sha256'].items()):
        raise ValueError('native or sources changed during batch')
    ordered = [results[case['case_id']] for case in cases]
    summary = summarize(ordered, args.trim_fractions)
    receipt = dict(schema='spatialrust.3dmatch-refinement-batch.v1', plan_sha256=digest(args.output_dir / 'plan.json'),
        summary=summary, cases=ordered, by_scene={scene: summarize([c for c in ordered if c['scene']==scene], args.trim_fractions)
            for scene in sorted({c['scene'] for c in ordered})}, default_trim_changed=False)
    atomic_json(args.output_dir / 'study.json', receipt)
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired trim validation</title>'
        '<h1>Fixed-case paired trim validation</h1><p>Saved generated priors are identical. Reference labels enter only '
        'post-fit evaluation. Exact untrimmed controls are required. Errors remain in planned denominators. '
        'This selected positive-pair study is not benchmark recall/precision or an independent dataset holdout.</p><pre>'
        + html.escape(json.dumps(dict(summary=summary, by_scene=receipt['by_scene']), indent=2)) + '</pre></html>')
    (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
