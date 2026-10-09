"""Evaluate saved Redwood/3DMatch predictions with the publisher's information metric.

This evaluates supplied log files; it never runs or selects a registration method.
It uses original matrices for the publisher formula, even when near-rigid records
are explicitly allowed. Normalized matrices are reserved for reference import.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import platform
import shutil

import numpy as np

from redwood_reference import load_records


def squared_information_error(reference, prediction, information):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        relative = np.linalg.inv(reference) @ prediction
        scalar_squared = (1 + np.trace(relative[:3, :3])) / 4
        if not np.isfinite(relative).all() or scalar_squared <= 1e-16:
            raise ValueError('publisher quaternion formula is singular near a half turn')
        scalar = math.sqrt(float(scalar_squared))
        rotation = relative[:3, :3]
        vector = np.array([rotation[2, 1] - rotation[1, 2],
                           rotation[0, 2] - rotation[2, 0],
                           rotation[1, 0] - rotation[0, 1]]) / (4 * scalar)
        error = np.concatenate([relative[:3, 3], vector])
        score = float(error @ information @ error / information[0, 0])
    if not math.isfinite(score):
        raise ValueError('information score overflows')
    return max(score, 0.)


def evaluate_predictions(predictions, truth, information, threshold_squared=.04):
    if not truth or set(truth) != set(information):
        raise ValueError('GT pose/information pair sets must match and be nonempty')
    if isinstance(threshold_squared, bool) or not isinstance(threshold_squared, (int, float)) or not math.isfinite(threshold_squared) or threshold_squared <= 0:
        raise ValueError('threshold_squared must be finite and positive')
    counts = {record['fragments'] for record in truth.values()}
    if len(counts) != 1 or any(record['fragments'] not in counts for record in information.values()):
        raise ValueError('GT fragment counts mismatch')
    if any(record['fragments'] not in counts for record in predictions.values()):
        raise ValueError('prediction fragment count differs from GT')
    eligible = {pair for pair in truth if pair[1] - pair[0] > 1}
    if not eligible:
        raise ValueError('GT requires nonconsecutive pairs for benchmark recall')
    rows = []
    for pair, record in predictions.items():
        if pair[1] - pair[0] <= 1:
            continue
        row = dict(pair=list(pair), correct=False, score_squared=None, status='false_positive')
        if pair in eligible:
            try:
                score = squared_information_error(truth[pair]['original_matrix'], record['original_matrix'], information[pair]['matrix'])
                row.update(score_squared=score, correct=score <= threshold_squared, status='scored')
            except ValueError as error:
                row.update(status='undefined_score', error=str(error))
        rows.append(row)
    good = sum(row['correct'] for row in rows)
    return dict(eligible_gt_pairs=len(eligible), evaluated_predictions=len(rows), correct=good,
                false_positive_pairs=sum(row['status'] == 'false_positive' for row in rows),
                undefined_scores=sum(row['status'] == 'undefined_score' for row in rows),
                recall=good / len(eligible), precision=good / len(rows) if rows else None,
                threshold_squared=threshold_squared, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-log', type=Path, required=True)
    parser.add_argument('--gt-info', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True)
    parser.add_argument('--provenance-url', required=True)
    parser.add_argument('--allow-near-rigid-records', action='store_true')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    if not args.provenance_url.startswith('https://'):
        parser.error('require an HTTPS provenance URL')
    truth, gt_hash = load_records(args.gt_log, project_rotation=args.allow_near_rigid_records)
    information, info_hash = load_records(args.gt_info, 6)
    methods = []
    for path in args.predictions:
        data = path.read_bytes()
        if data.strip():
            predicted, digest = load_records(path, project_rotation=args.allow_near_rigid_records)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('prediction file changed while loading')
        else:
            predicted, digest = {}, hashlib.sha256(data).hexdigest()
        result = evaluate_predictions(predicted, truth, information)
        result.update(prediction_file=str(path.resolve()), prediction_sha256=digest)
        methods.append(result)
    overlap = []
    for i, first in enumerate(methods):
        for second in methods[i + 1:]:
            a = {tuple(row['pair']) for row in first['rows'] if row['correct']}
            b = {tuple(row['pair']) for row in second['rows'] if row['correct']}
            overlap.append(dict(first=first['prediction_file'], second=second['prediction_file'],
                                both_correct=len(a & b), first_only=len(a - b), second_only=len(b - a),
                                oracle_union_correct=len(a | b),
                                deployable_selector_established=False))
    receipt = dict(schema='spatialrust.redwood-log-evaluation.v1', provenance_url=args.provenance_url,
                   gt_log_sha256=gt_hash, gt_info_sha256=info_hash,
                   runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   versions=dict(numpy=np.__version__, python=platform.python_version()),
                   helper_source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('redwood_reference.py', 'prepare_numpy_reference.py', 'evaluate_pose_reference.py')},
                   allow_near_rigid_records=args.allow_near_rigid_records,
                   scoring_matrices='original_logged_matrices', methods=methods, complementarity=overlap)
    table = []
    for result in methods:
        precision = 'undefined (no predictions)' if result['precision'] is None else f'{result["precision"]:.6g}'
        table.append(f'<tr><td>{html.escape(result["prediction_file"])}</td><td>{result["correct"]}/{result["eligible_gt_pairs"]}</td>'
                     f'<td>{result["recall"]:.6g}</td><td>{precision}</td><td>{result["undefined_scores"]}</td></tr>')
    complementary = []
    for pair in overlap:
        names = html.escape(Path(pair['first']).name + ' + ' + Path(pair['second']).name)
        complementary.append(f'<tr><td>{names}</td><td>{pair["first_only"]}</td><td>{pair["second_only"]}</td>'
                             f'<td>{pair["both_correct"]}</td><td>{pair["oracle_union_correct"]}</td></tr>')
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Redwood log evaluation</title>'
                '<h1>Saved-log benchmark evaluation</h1><p>Nonconsecutive pairs only. Publisher normalized information '
                'error squared ≤0.04. Original logged matrices are scored. Supplied logs do not measure current native implementations.</p>'
                '<table><tr><th>Prediction log</th><th>Correct/eligible GT</th><th>Recall</th><th>Precision</th><th>Undefined scores</th></tr>'
                + ''.join(table) + '</table><h2>Complementary correct pairs</h2>'
                '<table><tr><th>Log pair</th><th>First only</th><th>Second only</th><th>Both</th><th>Oracle union</th></tr>'
                + ''.join(complementary) + '</table><p>Pairwise complementarity is saved in JSON. Its oracle union is an upper '
                'bound that uses reference labels, not an implemented deployable selection policy. Singular half-turn '
                'scores count as incorrect and remain explicitly recorded.</p></html>')
    serialized = json.dumps(receipt, indent=2, allow_nan=False)
    args.output_dir.mkdir()
    try:
        (args.output_dir / 'evaluation.json').write_text(serialized, encoding='utf-8')
        (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Evaluated {len(methods)} supplied prediction logs')


if __name__ == '__main__':
    main()
