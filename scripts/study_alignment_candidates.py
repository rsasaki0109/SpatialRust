"""Bounded multistart ICP study; select without using ground-truth poses."""
import argparse
import html
import json
import time
from pathlib import Path

import numpy as np
import spatialrust as sr

from study_alignment_filters import align_files, matrix, pose_errors, rotation

ANGLES = [-40, -20, 0, 20, 40]
FILTERS = [('none', {}), ('sor', {'k_neighbors': 20, 'std_mul': .5})]


def selection_key(row):
    """Ground-truth-free ordering; stable list order resolves exact ties."""
    support = row['original_forward_support']
    rmse = support['gated_rmse_metres']
    return (-support['supported_count'], float('inf') if rmse is None else rmse)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'target/candidate-study')
    root = parser.parse_args().output_dir
    root.mkdir(parents=True, exist_ok=True)
    trials = []
    for seed in range(5):
        target = np.random.default_rng(seed).uniform(-1, 1, (400, 3))
        r, t = rotation(60), np.array([5., -2., .5])
        source = target @ r.T + t
        rng = np.random.default_rng(20000 + seed)
        outliers = rng.uniform(-1, 1, (80, 3))
        outliers[:, 0] = rng.uniform(1.2, 2, 80)
        source[-80:] = outliers @ r.T + t
        source += np.random.default_rng(10000 + seed).normal(0, .005, source.shape)
        original = sr.PointCloud.from_xyz(source.astype(np.float32))
        target_cloud = sr.PointCloud.from_xyz(target.astype(np.float32))
        tp = root / f'target-{seed}.pcd'
        sr.write(str(tp), target_cloud)
        truth = matrix(r.T, -r.T @ t)
        prior = matrix(rotation(40), np.array([.03, -.02, .01])) @ truth
        for kind, parameters in FILTERS:
            sp = root / f'source-{seed}-{kind}.pcd'
            try:
                filtered = (original if kind == 'none' else
                            sr.statistical_outlier_removal(original, **parameters))
                sr.write(str(sp), filtered)
                filter_error = None
            except Exception as exc:
                filter_error = exc
            for gate in [.3, .6]:
                candidates = []
                for angle in ANGLES:
                    row = dict(additional_rotation_degrees=angle)
                    started = time.perf_counter()
                    try:
                        if filter_error is not None:
                            raise filter_error
                        candidate = matrix(rotation(angle), np.zeros(3)) @ prior
                        _, diagnostics = align_files(sp, tp, leaf=.025, max_distance=gate,
                            iterations=100, initial_transform=candidate, evaluation_distance=.05)
                        estimated = np.asarray(diagnostics['transform_source_to_target'])
                        full = sr.apply_transform(original, estimated.astype(np.float32))
                        count, fraction, rmse = sr.distance_gated_support(full, target_cloud, .05)
                        row.update(transform_source_to_target=estimated.tolist(),
                            converged=diagnostics['converged'],
                            original_forward_support=dict(query_count=400,
                                supported_count=count, query_fraction=fraction,
                                gated_rmse_metres=rmse))
                    except Exception as exc:
                        row.update(error_type=type(exc).__name__, error=str(exc))
                    row["elapsed_seconds"] = time.perf_counter() - started
                    candidates.append(row)
                # Selection is finalized before any ground-truth assessment.
                valid = [i for i, row in enumerate(candidates) if 'error' not in row]
                selected = min(valid, key=lambda i: selection_key(candidates[i])) if valid else None
                baseline = ANGLES.index(0)
                for row in candidates:
                    row['success'] = False
                    if 'error' not in row:
                        angle_error, translation_error = pose_errors(
                            np.asarray(row['transform_source_to_target']), truth)
                        row.update(rotation_error_degrees=angle_error,
                            translation_error_metres=translation_error,
                            success=angle_error < 1 and translation_error < .01)
                trials.append(dict(seed=seed, filter=kind, filter_parameters=parameters,
                    gate_metres=gate, retained_points=(len(filtered.xyz()) if filter_error is None else None),
                    selected_index=selected, baseline_index=baseline,
                    total_candidate_elapsed_seconds=sum(c["elapsed_seconds"] for c in candidates),
                    baseline_elapsed_seconds=candidates[baseline]["elapsed_seconds"],
                    selected_success=selected is not None and candidates[selected]['success'],
                    baseline_success=candidates[baseline]['success'], candidates=candidates))
    summaries = []
    for kind, _ in FILTERS:
        for gate in [.3, .6]:
            group = [row for row in trials if row['filter'] == kind and row['gate_metres'] == gate]
            summaries.append(dict(filter=kind, gate_metres=gate, trials=len(group),
                baseline_successes=sum(row['baseline_success'] for row in group),
                selected_successes=sum(row['selected_success'] for row in group),
                candidate_errors=sum('error' in c for row in group for c in row['candidates']),
                selected_angles=[None if row['selected_index'] is None else
                    row['candidates'][row['selected_index']]['additional_rotation_degrees'] for row in group]))
    limitations = [
        'Five synthetic uniform seeds with a deliberately biased rotational prior; no real sensor validation.',
        'The angle grid includes a near-correct pose for these synthetic shifts; this is not a global registration guarantee.',
        'Selection uses original-source support at 0.05 m, then smaller gated RMSE, then stable candidate order. Ground truth is used only afterward for assessment.',
        'High support can favor incorrect poses in repetitive or symmetric scenes; this study does not certify pose correctness.',
        'Multistart uses five alignment runs per trial, versus one for the baseline; no speed advantage is claimed. Recorded wall times include candidate alignment and support scoring and are illustrative single-run timings, not benchmarks.']
    result = dict(schema='spatialrust.alignment-candidate-study.v1', seeds=list(range(5)),
        candidate_additional_target_frame_rotations_degrees=ANGLES,
        candidate_convention='additional target-frame rotation @ same prior',
        baseline='additional rotation 0 degrees (original prior)',
        search_gates_metres=[.3, .6], evaluation_gate_metres=.05, leaf_metres=.025,
        iterations_per_stage=100, source_noise_std_metres=.005,
        original_inliers=320, original_outliers=80,
        initial_rotation_error_degrees=40, initial_translation_error_metres=[.03, -.02, .01],
        initial_perturbation_convention='target-frame perturbation @ correct inverse',
        outlier_generation='replace last 80 of 400 with target-frame x in [1.2,2], y/z in [-1,1], then noise',
        selector='maximize original400 supported_count; minimize gated_rmse; stable candidate order',
        correctness='SO(3) rotation error < 1 degree and translation error < 0.01 m',
        alignment_runs=sum(len(row['candidates']) for row in trials),
        limitations=limitations, summary=summaries, trials=trials)
    (root / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    summary_rows = ''.join(f'<tr><td>{item["filter"]}</td><td>{item["gate_metres"]}</td>'
        f'<td>{item["baseline_successes"]}/5</td><td>{item["selected_successes"]}/5</td>'
        f'<td>{item["candidate_errors"]}</td><td>{item["selected_angles"]}</td></tr>' for item in summaries)
    details = []
    for row in trials:
        for index, candidate in enumerate(row['candidates']):
            support = candidate.get('original_forward_support', {})
            mark = ('selected ' if row['selected_index'] == index else '') + ('baseline' if row['baseline_index'] == index else '')
            details.append(f'<tr><td>{row["seed"]}</td><td>{row["filter"]}</td><td>{row["gate_metres"]}</td>'
                f'<td>{candidate["additional_rotation_degrees"]}</td><td>{mark}</td>'
                f'<td>{support.get("supported_count", "—")}/400</td><td>{support.get("gated_rmse_metres", "—")}</td>'
                f'<td>{candidate.get("rotation_error_degrees", "—")}</td>'
                f'<td>{candidate.get("translation_error_metres", "—")}</td>'
                f'<td>{html.escape(candidate.get("error", str(candidate["success"])))}</td></tr>')
    (root / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<title>Multistart ICP candidates</title><style>body{font:16px system-ui;max-width:1300px;margin:2rem auto}'
        'td,th{padding:.45rem;text-align:left}tr:nth-child(even){background:#eee}table{border-collapse:collapse}</style>'
        '<h1>Multistart ICP candidates</h1><p>100 alignment runs: five seeds × two filters × two search gates × '
        'five additional target-frame rotations. Every candidate is scored on all original 400 source points.</p><p>Fixed evaluation gate: 0.05 m. Correct poses require SO(3) error &lt;1° and translation error &lt;0.01 m.</p>'
        '<ul>' + ''.join('<li>' + html.escape(item) + '</li>' for item in limitations) + '</ul>'
        '<table><tr><th>Filter</th><th>Search gate (m)</th><th>Baseline correct</th><th>Selected correct</th>'
        '<th>Candidate errors</th><th>Selected angles by seed (°)</th></tr>' + summary_rows + '</table>'
        '<h2>All candidates, including failures</h2><table><tr><th>Seed</th><th>Filter</th><th>Gate (m)</th>'
        '<th>Additional angle (°)</th><th>Role</th><th>Supported count</th><th>Gated RMSE (m)</th>'
        '<th>Truth rotation error (°)</th><th>Truth translation error (m)</th><th>Correct / error</th></tr>'
        + ''.join(details) + '</table></html>', encoding='utf-8')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
