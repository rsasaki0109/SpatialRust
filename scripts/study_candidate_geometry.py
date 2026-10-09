"""Candidate support versus pose correctness across controlled geometries."""
import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np
import spatialrust as sr

from study_alignment_filters import matrix, pose_errors, rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'crates/spatialrust-py/examples'))
from align_point_clouds import align_files
from align_pose_candidates import evaluate_candidates


def geometry(kind, seed):
    rng = np.random.default_rng(seed)
    if kind in ('volume', 'plane'):
        xyz = rng.uniform(-1, 1, (400, 3))
        if kind == 'plane':
            xyz[:, 2] = 0
        return xyz
    angles = np.arange(180) * 2*np.pi/180
    ring = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(180)))
    if kind == 'ring':
        return ring
    # Three marked protrusions break the generating shape's exact symmetry.
    return np.concatenate((ring, [[1.4, .3, .2], [1.4, .33, .25], [1.43, .3, .22]]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'target/candidate-geometry-study')
    root = parser.parse_args().output_dir
    root.mkdir(parents=True, exist_ok=True)
    trials = []
    for kind in ('volume', 'plane', 'ring', 'marked_ring'):
        # Ring fixtures have no random draws: their repeated runs are controls,
        # not independent geometry samples.
        for seed in range(5 if kind in ('volume', 'plane') else 1):
            target = geometry(kind, seed)
            r, t = rotation(20), np.array([.012, -.008, .005])
            source = target @ r.T + t
            truth = matrix(r.T, -r.T @ t)
            values = [np.eye(4).tolist(), truth.tolist(),
                      (matrix(rotation(180), np.zeros(3)) @ truth).tolist()]
            paths = [root / f'{kind}-{seed}-{name}.pcd' for name in ('source', 'target')]
            for path, xyz in zip(paths, (source, target)):
                sr.write(str(path), sr.PointCloud.from_xyz(xyz.astype(np.float32)))
            settings = dict(leaf=.025, max_distance=.15, evaluation_distance=.05, iterations=100)
            _, report = evaluate_candidates(*paths, values, **settings)
            selection = report['candidate_selection']
            # Selection is fixed before evaluating any generating-pose errors.
            selected = selection['selected_index']
            independent = []
            for pose in values:
                try:
                    independent.append(align_files(*paths, initial_transform=pose, **settings)[1])
                except ValueError as error:
                    independent.append(dict(error=str(error)))
            ordinary = {k:v for k,v in report.items() if k != 'candidate_selection'}
            assert ordinary == independent[selected]
            assessments = []
            for record, separate in zip(selection['candidates'], independent):
                if record['status'] == 'error':
                    assert record['error'] == separate['error']
                    assessments.append(dict(index=record['index'], error=record['error']))
                    continue
                for field in ('transform_source_to_target', 'aligned_support', 'aligned_reverse_support', 'converged'):
                    assert record[field] == separate[field]
                angle, translation = pose_errors(np.asarray(record['transform_source_to_target']), truth)
                assessments.append(dict(index=record['index'], rotation_error_degrees=angle,
                    translation_error_metres=translation, correct=angle < 1 and translation < .01,
                    forward_support=record['aligned_support'], reverse_support=record['aligned_reverse_support']))
            trials.append(dict(geometry=kind, seed=seed, selected_index=selected,
                selected_correct=assessments[selected]['correct'], candidates=assessments,
                initial_transforms=values, truth_source_to_target=truth.tolist(),
                optimized_matches_independent=True))
    summary = []
    for kind in ('volume', 'plane', 'ring', 'marked_ring'):
        subset = [row for row in trials if row['geometry'] == kind]
        candidates = [c for row in subset for c in row['candidates'] if 'error' not in c]
        summary.append(dict(geometry=kind, trials=len(subset),
            selected_correct=sum(row['selected_correct'] for row in subset),
            wrong_high_support=sum(not c['correct'] and c['forward_support']['query_fraction'] >= .98 for c in candidates)))
    result = dict(trials=trials, summary=summary, settings=settings,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / 'crates/spatialrust-py/examples/align_point_clouds.py',
                      ROOT / 'crates/spatialrust-py/examples/align_pose_candidates.py')},
        limits='Synthetic geometry; truth-informed candidate grid for controlled diagnosis. Ring cases are deterministic single fixtures.')
    (root / 'study.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    cells = []
    for trial in trials:
        for candidate in trial['candidates']:
            label = f'{trial["geometry"]} seed {trial["seed"]} candidate {candidate["index"]}'
            selected = 'selected' if candidate['index'] == trial['selected_index'] else ''
            if 'error' in candidate:
                cells.append(f'<tr><td>{label}</td><td>{selected}</td><td colspan="4">{html.escape(candidate["error"])}</td></tr>')
                continue
            fraction = candidate['forward_support']['query_fraction']
            color = '#047857' if candidate['correct'] else '#b91c1c'
            cells.append(f'<tr><td>{label}</td><td>{selected}</td><td><meter min="0" max="1" value="{fraction}"></meter> '
                f'{fraction:.1%}</td><td>{candidate["forward_support"]["gated_rmse_metres"]}</td>'
                f'<td style="color:{color}">{candidate["rotation_error_degrees"]:.2f}°</td>'
                f'<td>{candidate["translation_error_metres"]:.5f} m</td></tr>')
    (root / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Candidate geometry study</title>'
        '<style>body{font:16px system-ui;margin:2rem}td,th{padding:.5rem;text-align:left}meter{width:120px}</style>'
        '<h1>Candidate support versus pose error</h1><p>Selection uses support and RMSE only. '
        'Generating-pose errors are assessed afterward. Red rotation errors fail the study criterion.</p>'
        '<p>'+html.escape(result['limits'])+'</p><table><tr><th>Condition</th><th>Selection</th>'
        '<th>Forward support</th><th>Gated RMSE (m)</th><th>Rotation error</th><th>Translation error</th></tr>'
        +''.join(cells)+'</table>', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
