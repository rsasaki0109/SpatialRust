"""Render a saved paired study as a standalone scientific plot and exact table.

Matplotlib is optional and required only by this reporting CLI, never fitting.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path


def paired_rows(study, fraction):
    if study.get('schema') != 'spatialrust.3dmatch-refinement-batch.v1':
        raise ValueError('unsupported batch schema')
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError('intervention fraction must be between zero and one')
    pairs, failed, planned, seen = [], [], 0, set()
    for case in study['cases']:
        if case['case_id'] in seen:
            raise ValueError('duplicate case ID')
        seen.add(case['case_id'])
        count = case['planned_seeds']
        if type(count) is not int or count < 1:
            raise ValueError('invalid planned seed count')
        planned += count
        if case['status'] == 'case_error':
            failed.append(case)
            continue
        if case['status'] != 'completed':
            raise ValueError('unknown case status')
        selections = {}
        for value in (1., fraction):
            rows = [row for row in case['evaluation']['rows'] if row['trim_fraction'] == value]
            keyed = {row['seed']: row for row in rows}
            if len(rows) != count or len(keyed) != count:
                raise ValueError('missing or duplicated paired seed')
            for row in rows:
                score = row['score_squared']
                if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or score < 0):
                    raise ValueError('score must be nonnegative finite or explicitly undefined')
                if row['correct'] is not (score is not None and score <= .04):
                    raise ValueError('correct flag differs from publisher threshold')
            selections[value] = keyed
        if set(selections[1.]) != set(selections[fraction]):
            raise ValueError('control/intervention seed sets differ')
        for seed, control in selections[1.].items():
            intervention = selections[fraction][seed]
            category = ('both_correct' if intervention['correct'] else 'lost') if control['correct'] else (
                'gained' if intervention['correct'] else 'neither_correct')
            pairs.append(dict(case_id=case['case_id'], scene=case['scene'], seed=seed,
                baseline=control['score_squared'], intervention=intervention['score_squared'], category=category))
    return pairs, failed, planned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--fraction', type=float, default=.8)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    data = args.study.read_bytes()
    pairs, failed, planned = paired_rows(json.loads(data), args.fraction)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    finite = [pair for pair in pairs if pair['baseline'] is not None and pair['intervention'] is not None]
    positive = [value for pair in finite for value in (pair['baseline'], pair['intervention']) if value > 0]
    floor = min(positive) / 3 if positive else 1e-8
    maximum = max([.4] + [max(pair['baseline'], pair['intervention']) * 2 for pair in finite])
    figure, axes = plt.subplots(figsize=(7.5, 6), layout='constrained')
    for category, label, color in [('both_correct', 'Both correct', '#2563eb'),
            ('gained', 'Recovery gained', '#16a34a'), ('lost', 'Recovery lost', '#ea580c'),
            ('neither_correct', 'Neither correct', '#64748b')]:
        selected = [pair for pair in finite if pair['category'] == category]
        if selected:
            axes.scatter([max(floor, pair['baseline']) for pair in selected],
                         [max(floor, pair['intervention']) for pair in selected],
                         color=color, alpha=.8, label=f'{label} ({len(selected)} plotted)', s=42)
    axes.plot([floor, maximum], [floor, maximum], color='gray', linewidth=1, linestyle=':')
    axes.axvline(.04, color='black', linewidth=1, linestyle='--')
    axes.axhline(.04, color='black', linewidth=1, linestyle='--')
    axes.set(xscale='log', yscale='log', xlim=(floor, maximum), ylim=(floor, maximum),
             xlabel='Untrimmed publisher score squared',
             ylabel=f'Retain {args.fraction:g}: publisher score squared',
             title='Fixed generated priors: trimming sensitivity')
    if finite:
        axes.legend(loc='upper left', fontsize=9)
    axes.grid(True, which='major', alpha=.2)
    args.output_dir.mkdir()
    figure.savefig(args.output_dir / 'paired_scores.svg', metadata={'Date': None})
    figure.savefig(args.output_dir / 'paired_scores.png', dpi=160)
    plt.close(figure)
    table = ''.join('<tr><td>' + html.escape(pair['case_id']) + '</td><td>' + html.escape(str(pair['seed']))
        + '</td><td>' + str(pair['baseline']) + '</td><td>' + str(pair['intervention']) + '</td><td>'
        + pair['category'] + '</td></tr>' for pair in pairs)
    counts = {category: sum(pair['category'] == category for pair in pairs)
              for category in ('both_correct', 'gained', 'lost', 'neither_correct')}
    receipt = dict(schema='spatialrust.refinement-plot.v1', input_sha256=hashlib.sha256(data).hexdigest(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), matplotlib_version=matplotlib.__version__,
        fraction=args.fraction, planned_pairs=planned, evaluated_pairs=len(pairs), plotted_pairs=len(finite),
        case_error_pairs=planned-len(pairs), categories=counts, plot_floor=floor, rows=pairs,
        failed_cases=failed, registration_executed=False)
    rendered = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired refinement scores</title>'
        '<h1>Paired refinement scores</h1><p>Lower scores are better. Dashed lines show the publisher threshold 0.04. '
        'Points below the diagonal improve the score. Categories include undefined scores as incorrect; those '
        'pairs remain in the table but cannot be placed on this plot. Zero scores plot at the lower axis bound. Selected positive '
        'pairs and repeated seeds are not independent benchmark samples. No reference-based trim selector is deployed.</p>'
        '<img src="paired_scores.svg" alt="Logarithmic paired score scatter plot"><pre>'
        + html.escape(json.dumps({key: receipt[key] for key in ('planned_pairs', 'evaluated_pairs', 'plotted_pairs', 'case_error_pairs', 'categories')}, indent=2))
        + '</pre><table><tr><th>Case</th><th>Seed</th><th>Untrimmed</th><th>Trimmed</th><th>Transition</th></tr>'
        + table + '</table></html>')
    (args.output_dir / 'plot.json').write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')


if __name__ == '__main__':
    main()
