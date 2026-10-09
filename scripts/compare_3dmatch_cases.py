"""Run bounded, isolated public-pair comparisons; retain every planned failure."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from evaluate_pose_reference import validate_reference
from evaluate_redwood_logs import squared_information_error
from redwood_reference import load_records


def validate_cases(manifest):
    cases = manifest.get('cases')
    if manifest.get('schema') != 'spatialrust.3dmatch-cases.v1' or not isinstance(cases, list) or not 1 <= len(cases) <= 1024:
        raise ValueError('require a supported manifest with 1..1024 cases')
    seen = set()
    for case in cases:
        name = case.get('case_id')
        if not isinstance(name, str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]*', name) or name in seen:
            raise ValueError('case IDs must be distinct safe directory names')
        seen.add(name)
        if any(not isinstance(case.get(key), str) or not case[key] for key in ('scene', 'source', 'target', 'reference_file')):
            raise ValueError('case requires scene and input/reference paths')
        pair = case.get('pair')
        if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair) or not 0 <= pair[0] < pair[1] - 1:
            raise ValueError('case requires an ordered nonconsecutive pair')
    return cases


def validate_saved_receipt(receipt, reference, reference_bytes, seeds, iterations):
    if receipt.get('schema') != 'spatialrust.public-global-comparison.v1':
        raise ValueError('unsupported saved comparison schema')
    if receipt.get('seeds') != seeds or receipt.get('ransac_iterations') != iterations:
        raise ValueError('saved comparison seed order or budget differs')
    if receipt.get('input_file_sha256') != reference['input_file_sha256']:
        raise ValueError('saved comparison input binding differs')
    if receipt.get('reference_sha256') != hashlib.sha256(reference_bytes).hexdigest():
        raise ValueError('reference changed during subprocess evaluation')
    expected = {(seed, method) for seed in seeds for method in ('spatialrust', 'open3d')}
    if len(receipt['rows']) != len(expected) or {(row['seed'], row['method']) for row in receipt['rows']} != expected:
        raise ValueError('subprocess omitted or duplicated planned rows')
    for row in receipt['rows']:
        if row['status'] not in ('success', 'error'):
            raise ValueError('unsupported solver status')
        if row['status'] == 'success' and row['report'].get('input_file_sha256') != reference['input_file_sha256']:
            raise ValueError('saved row input binding differs')


def execute(case, directory, seeds, iterations, timeout, reuse=False):
    reference_path = Path(case['reference_file'])
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    validate_reference(reference)
    poses, pose_hash = load_records(reference_path.parent / 'gt.log', project_rotation=True)
    information, info_hash = load_records(reference_path.parent / 'gt.info', 6)
    pair = tuple(case['pair'])
    if pair not in poses or pair not in information:
        raise ValueError('selected pair absent from scene metadata')
    if pose_hash != reference['provenance'].get('gt_log_sha256') or info_hash != reference['provenance'].get('gt_info_sha256'):
        raise ValueError('case reference and GT/information hashes differ')
    np.testing.assert_array_equal(reference['transform_source_to_target'], poses[pair]['matrix'])
    command = [sys.executable, str(Path(__file__).with_name('compare_public_global.py')),
               '--source', case['source'], '--target', case['target'], '--reference', str(reference_path),
               '--seeds', *map(str, seeds), '--iterations', str(iterations), '--output-dir', str(directory)]
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    if not reuse:
        try:
            completed = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return dict(case_id=case['case_id'], scene=case['scene'], status='case_error', error=f'case exceeded {timeout} seconds')
        if completed.returncode:
            return dict(case_id=case['case_id'], scene=case['scene'], status='case_error',
                        error=completed.stderr[-4000:] or completed.stdout[-4000:], exit_code=completed.returncode)
    receipt_path = directory / 'comparison.json'
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    validate_saved_receipt(receipt, reference, reference_bytes, seeds, iterations)
    if reuse:
        for key in ('source', 'target'):
            digest = hashlib.sha256(Path(case[key]).read_bytes()).hexdigest()
            if digest != reference['input_file_sha256'][key]:
                raise ValueError('saved comparison input file changed')
    for row in receipt['rows']:
        row['publisher_correct'] = False
        if row['status'] == 'success':
            try:
                score = squared_information_error(poses[pair]['original_matrix'],
                    np.array(row['report']['transform_source_to_target']), information[pair]['matrix'])
                row.update(publisher_score_squared=score, publisher_correct=score <= .04)
            except ValueError as error:
                row['publisher_score_error'] = str(error)
    return dict(case_id=case['case_id'], scene=case['scene'], pair=case['pair'], status='completed',
                comparison_sha256=hashlib.sha256(receipt_bytes).hexdigest(),
                comparison_file=str(receipt_path.resolve()), native_sha256=receipt['native_sha256'],
                open3d_native_sha256=receipt['open3d_native_sha256'],
                fitting_runner_sha256=receipt['source_sha256'], controls=receipt['controls'],
                algorithm_source_sha256=receipt['algorithm_source_sha256'], rows=receipt['rows'])


def summarize(cases, seeds):
    summary = {}
    for method in ('spatialrust', 'open3d'):
        rows = [row for case in cases if case['status'] == 'completed' for row in case['rows'] if row['method'] == method]
        planned = len(cases) * len(seeds)
        summary[method] = dict(planned=planned, completed_rows=len(rows),
                              solver_outputs=sum(row['status'] == 'success' for row in rows),
                              publisher_correct=sum(row['publisher_correct'] for row in rows),
                              case_error_rows=planned - len(rows))
    return summary


def complementarity(cases):
    correct = {method: {(case['case_id'], row['seed']) for case in cases if case['status'] == 'completed'
                       for row in case['rows'] if row['method'] == method and row['publisher_correct']}
               for method in ('spatialrust', 'open3d')}
    first, second = correct['spatialrust'], correct['open3d']
    return dict(both_correct=len(first & second), spatialrust_only=len(first - second),
                open3d_only=len(second - first), oracle_union_correct=len(first | second),
                deployable_selector_established=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 8, 9])
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--case-timeout', type=int, default=600)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-comparisons', type=Path,
                        help='aggregate existing complete case receipts without running registration')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    if not 1 <= args.workers <= 4 or not 1 <= args.case_timeout <= 3600 or not 1 <= args.iterations <= 1000000:
        parser.error('workers 1..4, timeout 1..3600 and iterations 1..1000000 required')
    if not 1 <= len(args.seeds) <= 16 or len(set(args.seeds)) != len(args.seeds) or any(not 0 <= seed < 2**31 for seed in args.seeds):
        parser.error('require 1..16 distinct nonnegative signed-32-bit seeds')
    manifest_bytes = args.cases.read_bytes()
    manifest = json.loads(manifest_bytes)
    cases = validate_cases(manifest)
    args.output_dir.mkdir()
    results = {}
    # Per-case outputs are independent; retain them if later validation fails.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        root = args.reuse_comparisons or args.output_dir
        futures = {executor.submit(execute, case, (root / case['case_id']).resolve(),
                                   args.seeds, args.iterations, args.case_timeout,
                                   args.reuse_comparisons is not None): case for case in cases}
        for future in as_completed(futures):
            result = future.result()
            results[result['case_id']] = result
            print(f'{result["case_id"]}: {result["status"]}', flush=True)
    ordered = [results[case['case_id']] for case in cases]
    completed = [case for case in ordered if case['status'] == 'completed']
    for key in ('native_sha256', 'open3d_native_sha256', 'fitting_runner_sha256', 'algorithm_source_sha256', 'controls'):
        if len({json.dumps(case[key], sort_keys=True) for case in completed}) > 1:
            raise ValueError('native, fitting sources or controls changed between cases')
    summary = summarize(ordered, args.seeds)
    by_scene = {scene: summarize([case for case in ordered if case['scene'] == scene], args.seeds)
                for scene in sorted({case['scene'] for case in ordered})}
    receipt = dict(schema='spatialrust.3dmatch-live-comparison.v1',
                   cases_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   planned_cases=len(cases), seeds=args.seeds, workers=args.workers,
                   ransac_iterations=args.iterations, case_timeout_seconds=args.case_timeout,
                   registration_executed=args.reuse_comparisons is None,
                   reused_comparisons_directory=str(args.reuse_comparisons.resolve()) if args.reuse_comparisons else None,
                   child_thread_environment=dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1'),
                   summary=summary, by_scene=by_scene,
                   complementarity=complementarity(ordered), cases=ordered)
    table = []
    for case in ordered:
        if case['status'] != 'completed':
            table.append(f'<tr><td>{html.escape(case["case_id"])}</td><td colspan="5">{html.escape(case["error"])}</td></tr>')
            continue
        for row in case['rows']:
            score = row.get('publisher_score_squared')
            error = row.get('pose_accuracy')
            table.append(f'<tr><td>{html.escape(case["case_id"])}</td><td>{row["method"]}</td><td>{row["seed"]}</td>'
                         f'<td>{score if score is not None else "undefined"}</td><td>{row["publisher_correct"]}</td>'
                         f'<td>{error["translation_error"] if error else "no pose"}</td></tr>')
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Live 3DMatch comparison</title>'
                '<h1>Live no-prior selected-pair comparison</h1><p>Official sensor fragments, fixed header-quantile '
                'case selection before fitting. No reference-based initialization or candidate selection. Publisher '
                'information score squared ≤0.04. This is a selected positive-pair study, not full benchmark '
                'recall/precision. Failed cases remain in planned denominators. Concurrent worker timing is not a speed ranking.</p>'
                '<pre>' + html.escape(json.dumps(dict(summary=summary, by_scene=by_scene,
                                                     complementarity=receipt['complementarity']), indent=2)) + '</pre>'
                '<p>Oracle union uses reference labels after fitting. It is not a deployed selection method. '
                'Proximity support does not certify reference correctness.</p>'
                '<table><tr><th>Case</th><th>Method</th><th>Seed</th><th>Squared score</th><th>Correct</th><th>Translation error (m)</th></tr>'
                + ''.join(table) + '</table></html>')
    (args.output_dir / 'study.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
