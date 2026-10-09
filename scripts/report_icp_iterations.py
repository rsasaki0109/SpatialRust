"""Compare validated public-pair runs with different iteration caps offline."""
import argparse
import html
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+', help='directories containing receipt.json and alignment.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    trials, control = [], None
    for root in args.runs:
        receipt = json.loads((root / 'receipt.json').read_text())
        alignment = json.loads((root / 'alignment.json').read_text())
        settings = dict(receipt['settings'])
        cap = settings.pop('iterations')
        signature = (receipt['files_sha256'], receipt['initial_transforms'], settings)
        if control is not None and control != signature:
            parser.error('runs must share input hashes, initial poses and all settings except iterations')
        control = signature
        if not receipt['optimized_matches_independent'] or not receipt['aligned_xyz_roundtrip_exact']:
            parser.error('run lacks successful equivalence or roundtrip validation')
        selection = alignment['candidate_selection']
        trials.append(dict(iteration_cap_per_stage=cap, selected_index=selection['selected_index'],
            elapsed_seconds=receipt['optimized_seconds'], candidates=selection['candidates']))
    trials.sort(key=lambda row: row['iteration_cap_per_stage'])
    rows = []
    for trial in trials:
        for candidate in trial['candidates']:
            label = f'{candidate["index"]}' + (' (selected)' if candidate['index'] == trial['selected_index'] else '')
            start = f'<tr><td>{html.escape(str(trial["iteration_cap_per_stage"]))}</td><td>{html.escape(label)}</td>'
            if candidate['status'] == 'error':
                rows.append(start+f'<td colspan="4">{html.escape(candidate["error"])}</td></tr>')
                continue
            forward, reverse = candidate['aligned_support'], candidate['aligned_reverse_support']
            residual = forward['gated_rmse_metres']
            residual = 'None' if residual is None else f'{residual*1000:.3f} mm'
            rows.append(start+f'<td><meter min="0" max="1" value="{forward["query_fraction"]}"></meter> '
                f'{forward["query_fraction"]:.2%}</td><td>{reverse["query_fraction"]:.2%}</td>'
                f'<td>{residual}</td><td>{html.escape(str(candidate["converged"]))}</td></tr>')
    timing = ''.join(f'<li>Cap {html.escape(str(trial["iteration_cap_per_stage"]))}: {trial["elapsed_seconds"]:.2f} seconds</li>' for trial in trials)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'comparison.json').write_text(json.dumps(dict(trials=trials,
        controlled_inputs_and_settings=True, ground_truth_pose_available=False), indent=2, allow_nan=False)+'\n')
    (args.output_dir / 'report.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Public ICP iteration study</title><style>body{font:16px system-ui;margin:2rem}'
        'td,th{padding:.5rem;text-align:left}meter{width:140px}</style><h1>Public ICP iteration study</h1>'
        '<p>Same public pair, supplied poses and distance gates. Iteration cap applies separately to each ICP stage. '
        'Convergence and proximity support do not certify true pose. Times are single observations, not repeated benchmarks.</p>'
        '<ul>'+timing+'</ul><table><tr><th>Cap / stage</th><th>Candidate</th><th>Forward support</th>'
        '<th>Reverse support</th><th>Forward gated RMSE</th><th>Converged</th></tr>'+''.join(rows)+'</table>')
    print(json.dumps([dict(cap=trial['iteration_cap_per_stage'], selected=trial['selected_index'],
        seconds=trial['elapsed_seconds'], converged=sum(c.get('converged', False) for c in trial['candidates'])) for trial in trials]))


if __name__ == '__main__':
    main()
