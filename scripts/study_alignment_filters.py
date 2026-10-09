"""Compare public outlier filters before ICP on five reproducible synthetic seeds."""
import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import spatialrust as sr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'crates/spatialrust-py/examples'))
from align_point_clouds import align_files


def rotation(degrees):
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def matrix(r, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r, t
    return m


def pose_errors(estimated, truth):
    u, _, vh = np.linalg.svd(estimated[:3, :3])
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vh
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(r @ truth[:3, :3].T)-1)/2, -1, 1))))
    return angle, float(np.linalg.norm(estimated[:3, 3]-truth[:3, 3]))


FILTERS = [
    ('none', {}),
    ('sor', {'k_neighbors': 20, 'std_mul': 1.0}),
    ('sor', {'k_neighbors': 20, 'std_mul': 0.5}),
    ('radius', {'radius': 0.3, 'min_neighbors': 2}),
    ('radius', {'radius': 0.4, 'min_neighbors': 5}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=REPO/'target/filter-study')
    args = parser.parse_args()
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in range(5):
        target = np.random.default_rng(seed).uniform(-1, 1, (400, 3))
        r, t = rotation(60), np.array([5., -2., .5])
        source = target @ r.T + t
        rng = np.random.default_rng(20000+seed)
        outliers = rng.uniform(-1, 1, (80, 3))
        outliers[:, 0] = rng.uniform(1.2, 2, 80)
        source[-80:] = outliers @ r.T + t
        source += np.random.default_rng(10000+seed).normal(0, .005, source.shape)
        original = sr.PointCloud.from_xyz(source.astype(np.float32))
        original_xyz = original.xyz()
        # Exact float32 identity checks avoid guessing labels by geometric proximity.
        membership = {tuple(point): index < 320 for index, point in enumerate(original_xyz)}
        if len(membership) != 400:
            raise RuntimeError('duplicate source coordinates make membership ambiguous')
        tp = root/f'target-{seed}.pcd'
        sr.write(str(tp), sr.PointCloud.from_xyz(target.astype(np.float32)))
        truth = matrix(r.T, -r.T @ t)
        prior = matrix(rotation(40), np.array([.03, -.02, .01])) @ truth
        for index, (kind, parameters) in enumerate(FILTERS):
            label = kind + (':' + ','.join(f'{k}={v}' for k, v in parameters.items()) if parameters else '')
            common = dict(seed=seed, filter=label, filter_parameters=parameters,
                          original_inliers=320, original_outliers=80)
            try:
                filtered = (original if kind == 'none' else
                            sr.statistical_outlier_removal(original, **parameters) if kind == 'sor' else
                            sr.radius_outlier_removal(original, **parameters))
                retained = [membership[tuple(point)] for point in filtered.xyz()]
                common.update(retained_inliers=sum(retained), retained_outliers=len(retained)-sum(retained),
                              retained_points=len(retained))
                sp = root/f'source-{seed}-filter-{index}.pcd'
                sr.write(str(sp), filtered)
                filter_error = None
            except Exception as exc:
                filter_error = exc
            for gate in [.3, .6, 1.2]:
                row = dict(common, gate_metres=gate, success=False)
                try:
                    if filter_error is not None:
                        raise filter_error
                    _, diagnostics = align_files(sp, tp, leaf=.025, max_distance=gate,
                        iterations=100, initial_transform=prior, evaluation_distance=.05)
                    estimated = np.asarray(diagnostics['transform_source_to_target'])
                    angle, translation = pose_errors(estimated, truth)
                    full = sr.apply_transform(original, estimated.astype(np.float32))
                    count, fraction, rmse = sr.distance_gated_support(full, sr.read(str(tp)), .05)
                    support = dict(query_count=400, supported_count=count,
                                   query_fraction=fraction, gated_rmse_metres=rmse)
                    row.update(success=angle < 1 and translation < .01,
                               rotation_error_degrees=angle, translation_error_metres=translation,
                               converged=diagnostics['converged'],
                               filtered_forward_support=diagnostics['aligned_support'],
                               original_forward_support=support, diagnostics=diagnostics)
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, error=str(exc))
                rows.append(row)
    summaries = []
    for kind, parameters in FILTERS:
        label = kind + (':' + ','.join(f'{k}={v}' for k, v in parameters.items()) if parameters else '')
        for gate in [.3, .6, 1.2]:
            group = [row for row in rows if row['filter'] == label and row['gate_metres'] == gate]
            summaries.append(dict(filter=label, gate_metres=gate, trials=len(group),
                successes=sum(row['success'] for row in group),
                errors=sum('error' in row for row in group),
                retained_inliers=[row.get('retained_inliers') for row in group],
                retained_outliers=[row.get('retained_outliers') for row in group],
                original_support=[row.get('original_forward_support') for row in group]))
    result = dict(schema='spatialrust.alignment-filter-study.v1', seeds=list(range(5)),
        search_gates_metres=[.3, .6, 1.2], evaluation_gate_metres=.05, leaf_metres=.025,
        iterations_per_stage=100, source_noise_std_metres=.005,
        initial_rotation_error_degrees=40, initial_translation_error_metres=[.03, -.02, .01],
        initial_perturbation_convention='target-frame perturbation @ correct inverse',
        outlier_generation='replace last 80 of 400 with target-frame x in [1.2,2], y/z in [-1,1], then noise',
        correctness='rotation error < 1 degree and translation error < 0.01 m',
        membership='exact float32 source-coordinate subset, before alignment',
        limitations=['Five synthetic uniform seeds; no real sensors or general filter recommendation.',
                     'Density-based filters may remove legitimate sparse points and retain clustered outliers.',
                     'Support is evaluated on all original 400 source points; proximity is not pose correctness.'],
        rows=rows, summary=summaries)
    (root/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    table = []
    for item in summaries:
        keep = lambda key: ', '.join(str(x) for x in item[key])
        def retention_bar(key, denominator, label, color):
            values = [v for v in item[key] if v is not None]
            if not values:
                return 'Unavailable: filter did not complete'
            mean = sum(values) / len(values) / denominator
            low, high = min(values) / denominator, max(values) / denominator
            return (f'<svg viewBox="0 0 100 10" role="img" aria-label="{label}: mean {mean:.1%}">'
                    f'<rect width="100" height="10" fill="#eee"/><rect width="{mean*100:.9g}" height="10" fill="{color}"/></svg>'
                    f'<br>Mean {mean:.1%}; seed range {low:.1%}–{high:.1%}; {len(values)}/5 filter runs')
        table.append(f'<tr><td>{html.escape(item["filter"])}</td><td>{item["gate_metres"]}</td>'
                     f'<td>{item["successes"]}/5</td><td>{item["errors"]}</td>'
                     f'<td>{keep("retained_inliers")}</td><td>{keep("retained_outliers")}</td>'
                     f'<td>{retention_bar("retained_inliers", 320, "Genuine points retained", "#0369a1")}</td>'
                     f'<td>{retention_bar("retained_outliers", 80, "Nonmatching points retained", "#b45309")}</td></tr>')
    (root/'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Outlier filters before ICP</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto}'
        'td,th{padding:.5rem;text-align:left}tr:nth-child(even){background:#eee}svg{width:150px}</style>'
        '<h1>Outlier filters before ICP</h1><p>Five seeds; 320 original inliers and 80 replacement outliers; '
        '5 mm per-axis noise; 40 degree initial error. Fixed evaluation gate: 0.05 m.</p>'
        '<table><tr><th>Filter</th><th>Search gate (m)</th><th>Correct poses</th><th>Errors</th>'
        '<th>Retained inliers by seed</th><th>Retained outliers by seed</th>'
        '<th>Genuine points retained / 320</th><th>Nonmatching points retained / 80</th></tr>' + ''.join(table) + '</table>'
        '<p>Blue bars show genuine point retention; orange bars show nonmatching point retention. '
        'Bars are mean fractions across completed filter runs; ranges expose variation across seeds. '
        'A low orange bar can still accompany loss of genuine points. The same filtering result is reused across search gates.</p>'
        '<p>Correct poses require rotation error &lt;1 degree and translation error &lt;0.01 m. '
        'JSON includes support on every original point, including removed points. Five synthetic seeds do not establish '
        'a general filter recommendation; clustered outliers can survive density filters while valid sparse points are lost.</p></html>', encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
