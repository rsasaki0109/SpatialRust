"""Replay fixed generated initializations to separate trimming from global sampling."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'crates/spatialrust-py/examples'))
from align_multiscale import align_multiscale_clouds, validate_schedule
from align_point_clouds import read_bound_cloud, rigid_matrix
from evaluate_pose_reference import evaluate, validate_reference


def initializations(comparison, fingerprints):
    if comparison.get('schema') != 'spatialrust.public-global-comparison.v1' or comparison.get('input_file_sha256') != fingerprints:
        raise ValueError('comparison schema or input hashes mismatch')
    selected = []
    seen = set()
    for row in comparison['rows']:
        if row['method'] != 'spatialrust' or row['status'] != 'success':
            continue
        if type(row['seed']) is not int or row['seed'] in seen:
            raise ValueError('duplicate/invalid native seed')
        seen.add(row['seed'])
        report = row['report']
        candidates = report['candidate_selection']['candidates']
        if len(candidates) != 1 or candidates[0]['seed'] != row['seed'] or report.get('initial_transform_supplied') is not False:
            raise ValueError('replay requires one generated candidate per seed')
        if report.get('input_file_sha256') != fingerprints:
            raise ValueError('saved report input hashes mismatch')
        selected.append((row['seed'], rigid_matrix(candidates[0]['initial_transform_source_to_target'])))
    if not selected:
        raise ValueError('no generated initialization to replay')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'reference', 'comparison'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--trim-fractions', type=float, nargs='+', default=[1., .8, .6, .4])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    if not 1 <= len(args.trim_fractions) <= 16 or len(set(args.trim_fractions)) != len(args.trim_fractions):
        parser.error('require 1..16 distinct trim fractions')
    for trim in args.trim_fractions:
        validate_schedule([dict(leaf=None, max_distance=.05, iterations=50, trim_fraction=trim)])
    reference_bytes, comparison_bytes = args.reference.read_bytes(), args.comparison.read_bytes()
    reference, comparison = json.loads(reference_bytes), json.loads(comparison_bytes)
    validate_reference(reference)
    if comparison.get('reference_sha256') != hashlib.sha256(reference_bytes).hexdigest():
        parser.error('replay reference differs from the saved comparison')
    if reference['length_unit'] != 'm':
        parser.error('replay gates use metres')
    source, source_hash = read_bound_cloud(args.source)
    target, target_hash = read_bound_cloud(args.target)
    fingerprints = dict(source=source_hash, target=target_hash)
    if fingerprints != reference['input_file_sha256']:
        parser.error('input hashes differ from reference')
    generated = initializations(comparison, fingerprints)
    native = list(Path(sr.__file__).parent.glob('*.so'))
    if len(native) != 1:
        parser.error('expected one native extension')
    native_hash = hashlib.sha256(native[0].read_bytes()).hexdigest()
    if comparison['native_sha256'] != native_hash:
        parser.error('replay must use the same native build as the saved comparison')
    source_bytes = Path(__file__).read_bytes()
    rows = []
    for seed, initial in generated:
        for trim in args.trim_fractions:
            schedule = [dict(leaf=.05, max_distance=.2, iterations=50, trim_fraction=trim),
                        dict(leaf=None, max_distance=.05, iterations=50, trim_fraction=trim)]
            _, report = align_multiscale_clouds(source, target, schedule, initial_transform=initial)
            report['input_file_sha256'] = dict(fingerprints)
            rows.append(dict(seed=seed, trim_fraction=trim, report=report))
    # Metrics are computed after every refinement finishes; no pose selection.
    for row in rows:
        row['accuracy'] = evaluate(row['report'], reference)
        initial_report = dict(row['report'], transform_source_to_target=row['report']['initial_transform_source_to_target'])
        row['initial_accuracy'] = evaluate(initial_report, reference)
    if hashlib.sha256(native[0].read_bytes()).hexdigest() != native_hash or Path(__file__).read_bytes() != source_bytes:
        raise ValueError('native or study runner changed during replay')
    result = dict(schema='spatialrust.public-refinement-study.v1', native_sha256=native_hash,
                  source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  comparison_sha256=hashlib.sha256(comparison_bytes).hexdigest(),
                  input_file_sha256=fingerprints, reference_provenance=reference['provenance'], rows=rows)
    table = []
    for row in rows:
        angle, translation = row['accuracy']['rotation_error_degrees'], row['accuracy']['translation_error']
        support = row['report']['aligned_support']['query_fraction']
        table.append(f'<tr><td>{row["seed"]}</td><td>{row["trim_fraction"]}</td><td>{angle:.6g}°</td><td>{translation:.6g} m</td><td>{support:.6g}</td></tr>')
    dots = []
    maximum = max(row['accuracy']['translation_error'] for row in rows) or 1
    for row in rows:
        x = 40 + row['report']['aligned_support']['query_fraction'] * 500
        y = 240 - row['accuracy']['translation_error'] / maximum * 200
        label = html.escape(f'seed={row["seed"]}, trim={row["trim_fraction"]}, translation={row["accuracy"]["translation_error"]:.6g} m')
        dots.append(f'<circle cx="{x:.5g}" cy="{y:.5g}" r="4" fill="#3978b5"><title>{label}</title></circle>')
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Fixed-prior refinement study</title>'
                '<h1>Trimming and partial-cloud drift</h1><p>Fixed FPFH-generated initializations, same native build and points. '
                'Only trim fraction changes. Reference enters evaluation after fitting; no best trim is selected by truth. '
                'Sparse overlap and initialization can make aggressive trimming worse.</p>'
                '<svg viewBox="0 0 580 290" width="580"><path d="M40 40 V240 H540" fill="none" stroke="black"/>'
                '<text x="150" y="275">Proximity fraction (0..1)</text><text x="45" y="25">Translation error (0..' + f'{maximum:.4g} m)</text>'
                + ''.join(dots) + '</svg><table><tr><th>Seed</th><th>Trim</th><th>Rotation error</th><th>Translation error</th><th>Proximity fraction</th></tr>'
                + ''.join(table) + '</table><p>Reference provenance: '
                + html.escape(json.dumps(reference['provenance'], sort_keys=True))
                + '. This experiment does not independently certify sensor ground truth.</p></html>')
    serialized = json.dumps(result, indent=2, allow_nan=False)
    args.output_dir.mkdir()
    try:
        (args.output_dir / 'study.json').write_text(serialized, encoding='utf-8')
        (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Replayed {len(rows)} refinements without rerunning global initialization')


if __name__ == '__main__':
    main()
