"""Render an alignment.json diagnostic as a standalone HTML report."""
import argparse
import html
import json
import math
from pathlib import Path


def _number(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside the supported range')
    return value


def _count(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _candidate_rotation(candidate):
    matrix = candidate.get('transform_source_to_target')
    if matrix is None:
        return None  # Older support-only reports can still be rendered.
    if not isinstance(matrix, list) or len(matrix) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in matrix):
        raise ValueError('candidate transform must be a rigid 4x4 matrix')
    for row in matrix:
        for value in row:
            _number(value, 'candidate transform', minimum=-math.inf)
    if any(abs(matrix[3][j] - (1 if j == 3 else 0)) > 1e-5 for j in range(4)):
        raise ValueError('candidate transform must be rigid')
    r = [row[:3] for row in matrix[:3]]
    for i in range(3):
        for j in range(3):
            if abs(sum(r[k][i]*r[k][j] for k in range(3)) - (1 if i == j else 0)) > 1e-4:
                raise ValueError('candidate rotation must be orthogonal')
    determinant = (r[0][0]*(r[1][1]*r[2][2]-r[1][2]*r[2][1])
                   - r[0][1]*(r[1][0]*r[2][2]-r[1][2]*r[2][0])
                   + r[0][2]*(r[1][0]*r[2][1]-r[1][1]*r[2][0]))
    if abs(determinant - 1) > 1e-4:
        raise ValueError('candidate rotation must be proper')
    return r


def _candidate_table(selection, source, target, gate):
    if not isinstance(selection, dict):
        raise ValueError('candidate_selection must be an object')
    candidates = selection.get('candidates')
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
        raise ValueError('candidate_selection requires 1 to 16 candidates')
    if type(selection.get('candidate_count')) is not int or selection['candidate_count'] != len(candidates):
        raise ValueError('candidate count mismatch')
    selected = selection.get('selected_index')
    if type(selected) is not int or not 0 <= selected < len(candidates):
        raise ValueError('invalid selected candidate index')
    selected_rotation = (_candidate_rotation(candidates[selected])
                         if isinstance(candidates[selected], dict) else None)
    centroid = selection.get('source_centroid_xyz_metres')
    if centroid is not None:
        if not isinstance(centroid, list) or len(centroid) != 3:
            raise ValueError('candidate source centroid must contain three finite coordinates')
        for value in centroid:
            _number(value, 'candidate source centroid', minimum=-math.inf)
    rows, scores = [], []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict) or type(candidate.get('index')) is not int or candidate['index'] != index:
            raise ValueError('candidate indices must match input order')
        if candidate.get('status') == 'error':
            if not isinstance(candidate.get('error'), str) or index == selected:
                raise ValueError('invalid candidate failure')
            rows.append(f'<tr><td>{index}</td><td colspan="6">Failed: {html.escape(candidate["error"])}</td></tr>')
            continue
        if candidate.get('status') != 'success' or type(candidate.get('converged')) is not bool:
            raise ValueError('invalid candidate status or convergence')
        displays = []
        for key, denominator in [('aligned_support', source), ('aligned_reverse_support', target)]:
            support = candidate.get(key)
            if not isinstance(support, dict) or _count(support.get('query_points'), 'candidate query_points') != denominator:
                raise ValueError('candidate support denominator mismatch')
            count = support.get('distance_gated_points')
            fraction = _number(support.get('query_fraction'), 'candidate fraction', maximum=1)
            if type(count) is not int or not 0 <= count <= denominator or not math.isclose(fraction, count / denominator, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('candidate support count/fraction mismatch')
            if _number(support.get('distance_metres'), 'candidate distance_metres') != gate:
                raise ValueError('candidate evaluation gate mismatch')
            rmse = support.get('gated_rmse_metres')
            if count == 0:
                if rmse is not None:
                    raise ValueError('empty candidate support must have null RMSE')
            else:
                _number(rmse, 'candidate RMSE')
                if rmse > gate and not math.isclose(rmse, gate, rel_tol=1e-6):
                    raise ValueError('candidate RMSE exceeds gate')
            displays.append(f'{count}/{denominator} ({fraction:.1%})')
            if key == 'aligned_support':
                scores.append((index, -count, math.inf if rmse is None else rmse))
                residual = 'None' if rmse is None else f'{rmse:.6g} m'
        label = f'{index} (selected)' if index == selected else str(index)
        rotation = _candidate_rotation(candidate)
        difference = 'Unavailable'
        position_difference = 'Unavailable'
        if rotation is not None and selected_rotation is not None:
            cosine = (sum(rotation[i][j]*selected_rotation[i][j] for i in range(3) for j in range(3)) - 1) / 2
            angle = 0. if rotation == selected_rotation else math.degrees(math.acos(max(-1., min(1., cosine))))
            difference = f'<meter min="0" max="180" value="{angle:.6g}"></meter> {angle:.2f}°'
            if centroid is not None:
                pose = candidate['transform_source_to_target']
                chosen = candidates[selected]['transform_source_to_target']
                delta = [sum((pose[i][j] - chosen[i][j])*centroid[j] for j in range(3))
                         + pose[i][3] - chosen[i][3] for i in range(3)]
                separation = _number(math.hypot(*delta), 'candidate centroid separation')
                position_difference = f'{separation:.6g} m'
        rows.append(f'<tr><td>{label}</td><td>{displays[0]}</td><td>{displays[1]}</td>'
                    f'<td>{residual}</td><td>{str(candidate["converged"]).lower()}</td><td>{difference}</td>'
                    f'<td>{position_difference}</td></tr>')
    if not scores or type(selection.get('successful_candidates')) is not int or selection['successful_candidates'] != len(scores):
        raise ValueError('candidate success count mismatch')
    if min(scores, key=lambda score: (score[1], score[2]))[0] != selected:
        raise ValueError('selected candidate does not match support/RMSE ordering')
    rule = 'maximum_forward_supported_points_then_minimum_gated_rmse_then_input_order'
    if selection.get('rule') != rule:
        raise ValueError('unsupported candidate selection rule')
    return ('<section><h2>Pose candidates</h2><table><tr><th>Index</th><th>Forward support</th>'
            '<th>Reverse support</th><th>Forward gated RMSE</th><th>Converged</th><th>Rotation from selected</th>'
            '<th>Source centroid from selected</th></tr>'
            + ''.join(rows) + '</table><p>Selection maximizes forward supported points, then minimizes gated RMSE; '
            'input order breaks ties. Reverse support and convergence are shown for inspection. '
            'Selection does not certify pose correctness. Rotation from selected measures candidate disagreement, '
            'not ground-truth error or a confidence probability. Similar scores with different poses can indicate '
            'geometric ambiguity. Source centroid from selected measures the distance between transformed '
            'source centroids, not the difference between pose translation vectors. Rotations about the '
            'centroid can have zero centroid distance.</p></section>')


def _trace_chart(title, series, labels):
    """Linear axes with gaps for unavailable measurements; no external assets."""
    colors = ('#0369a1', '#c2410c')
    maximum = max((v for values in series for v in values if v is not None), default=0)
    scale = maximum if maximum > 0 else 1
    count = len(series[0])
    elements = []
    for values, color in zip(series, colors):
        segment = []
        def flush():
            if segment:
                elements.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2"/>')
                segment.clear()
        for i, value in enumerate(values):
            if value is None:
                flush()
                continue
            x, y = 65 + 420 * i / max(1, count - 1), 145 - 110 * value / scale
            segment.append(f'{x:.4f},{y:.4f}')
            elements.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="2" fill="{color}"/>')
        flush()
    legend = ', '.join(html.escape(label) for label in labels)
    return (f'<figure><figcaption>{html.escape(title)} — {legend}</figcaption>'
            f'<svg class="trace-chart" viewBox="0 0 520 180" role="img" aria-label="{html.escape(title, quote=True)} by iteration">'
            '<path d="M65 35V145H485" stroke="#64748b" fill="none"/>'
            f'<text x="2" y="40">{maximum:.4g}</text><text x="40" y="150">0</text>'
            f'<text x="65" y="170">1</text><text x="440" y="170">{count}</text>'
            + ''.join(elements) + '</svg><p>Horizontal axis: iteration; vertical axis: linear, zero to '
            f'{maximum:.6g}. Blue: {html.escape(labels[0])}'
            + (f'; orange: {html.escape(labels[1])}' if len(labels) > 1 else '') + '.</p></figure>')


def _trace_sections(report):
    sections = []
    stages = report.get('stages', [])
    if not isinstance(stages, list):
        raise ValueError('stages must be a list')
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError('stage must be an object')
        if 'icp_history' not in stage:
            continue  # Older and ordinary reports remain readable.
        history = stage['icp_history']
        iterations = _count(stage.get('iterations'), 'trace iterations')
        source = _count(stage.get('source_points'), 'trace source_points')
        if not isinstance(history, list) or len(history) != iterations:
            raise ValueError('trace history length must match iterations')
        reason = stage.get('stop_reason')
        if reason not in ('fitness_threshold', 'transform_threshold', 'iteration_limit'):
            raise ValueError('unsupported trace stop_reason')
        if type(stage.get('converged')) is not bool or stage['converged'] != (reason != 'iteration_limit'):
            raise ValueError('trace stop_reason and convergence disagree')
        name = stage.get('name')
        if not isinstance(name, str):
            raise ValueError('trace stage name must be text')
        criteria = stage.get('convergence_criteria')
        criteria_html = ''
        if criteria is not None:
            if not isinstance(criteria, dict) or set(criteria) != {'translation_epsilon', 'rotation_epsilon', 'fitness_epsilon'}:
                raise ValueError('trace convergence criteria fields are invalid')
            for value in criteria.values():
                _number(value, 'trace convergence threshold')
            criteria_html = (f'<p>Stopping thresholds: translation {criteria["translation_epsilon"]:.6g} m; '
                             f'rotation {criteria["rotation_epsilon"]:.6g} rad; absolute MSE {criteria["fitness_epsilon"]:.6g} m². '
                             'Zero disables a test; both transform thresholds must be met.</p>')
        previous = None
        rows = []
        for i, row in enumerate(history, 1):
            if not isinstance(row, dict) or type(row.get('iteration')) is not int or row['iteration'] != i:
                raise ValueError('trace iteration numbers must be contiguous')
            for key in ('correspondences', 'evaluated_correspondences'):
                if type(row.get(key)) is not int or not 0 <= row[key] <= source:
                    raise ValueError('trace correspondence count is invalid')
            fitness = row.get('fitness_metres_squared')
            if row['evaluated_correspondences'] == 0:
                if fitness is not None:
                    raise ValueError('empty trace correspondence set must have null fitness')
            else:
                _number(fitness, 'trace fitness')
            change = row.get('fitness_change_metres_squared')
            if i == 1:
                if change is not None:
                    raise ValueError('first trace fitness change must be null')
            elif previous is not None and fitness is not None:
                _number(change, 'trace fitness change', minimum=-math.inf)
                if not math.isclose(change, previous - fitness, rel_tol=1e-12, abs_tol=1e-18):
                    raise ValueError('trace fitness change disagrees with history')
            elif change is not None:
                _number(change, 'trace fitness change', minimum=-math.inf)
            previous = fitness
            _number(row.get('translation_delta_metres'), 'trace translation delta')
            _number(row.get('rotation_delta_radians'), 'trace rotation delta', maximum=math.pi)
            residual = 'Unavailable' if fitness is None else f'{math.sqrt(fitness):.6g}'
            rows.append(f'<tr><td>{i}</td><td>{row["correspondences"]}</td><td>{row["evaluated_correspondences"]}</td>'
                        f'<td>{residual}</td><td>{row["translation_delta_metres"]:.6g}</td>'
                        f'<td>{math.degrees(row["rotation_delta_radians"]):.6g}</td></tr>')
        if stage.get('kernel_fitness_metres_squared') != previous:
            raise ValueError('trace final fitness differs from stage result')
        if criteria is not None:
            final = history[-1]
            expected = ('fitness_threshold' if previous is not None and previous < criteria['fitness_epsilon']
                        else 'transform_threshold' if final['translation_delta_metres'] < criteria['translation_epsilon']
                        and final['rotation_delta_radians'] < criteria['rotation_epsilon'] else 'iteration_limit')
            if reason != expected:
                raise ValueError('trace stop_reason disagrees with thresholds')
        charts = [
            _trace_chart('Gated correspondence count',
                         [[r[k] for r in history] for k in ('correspondences', 'evaluated_correspondences')],
                         ['used for update', 'after rematching']),
            _trace_chart('Gated RMSE (m)', [[None if r['fitness_metres_squared'] is None else math.sqrt(r['fitness_metres_squared']) for r in history]], ['after rematching']),
            _trace_chart('Translation update (m)', [[r['translation_delta_metres'] for r in history]], ['update length']),
            _trace_chart('Rotation update (degrees)', [[math.degrees(r['rotation_delta_radians']) for r in history]], ['shortest angle']),
        ]
        sections.append(f'<section><h2>ICP iteration history: {html.escape(name)}</h2><p>Stop reason: {reason}.</p>'
                        + criteria_html + ''.join(charts) + '<details><summary>All update measurements</summary><table><tr>'
                        '<th>Iteration</th><th>Used matches</th><th>Rematched</th><th>RMSE (m)</th>'
                        '<th>Translation (m)</th><th>Rotation (degrees)</th></tr>' + ''.join(rows) + '</table></details></section>')
    if not sections:
        return ''
    return (''.join(sections) + '<p>Correspondence membership changes after each update. '
            'Lower gated RMSE may reflect excluded points rather than a better pose. '
            'An iteration-limit exit differs from a met stopping threshold; neither certifies accuracy.</p>')


def _trim_summary(report):
    entries = []
    stages = report.get('stages', [])
    if not isinstance(stages, list):
        raise ValueError('stages must be a list')
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError('stage must be an object')
        if 'trim_fraction' not in stage:
            continue
        fraction = _number(stage['trim_fraction'], 'trim_fraction', maximum=1)
        if fraction == 0:
            raise ValueError('trim_fraction must be positive')
        if not isinstance(stage.get('name'), str):
            raise ValueError('trimming stage name must be text')
        entries.append(f'<li>{html.escape(stage["name"])}: retain {fraction:.1%} of gated pairs.</li>')
    if not entries:
        return ''
    return ('<section><h2>ICP pair selection</h2><ul>' + ''.join(entries)
            + '</ul><p>Updates use the lowest-distance pairs; ties follow source order. '
            'Fitness evaluates all rematched gated points, and support evaluates the full query cloud. '
            'A poor prior can cause trimming to discard useful correct points.</p></section>')


def render_report(report):
    """Validate diagnostic fields and return HTML; no network or scripts required."""
    if not isinstance(report, dict) or report.get('schema_version') != 'spatialrust.python-alignment.v1':
        raise ValueError('unsupported alignment report schema')
    source = _count(report.get('source_points'), 'source_points')
    target = _count(report.get('target_points'), 'target_points')
    optimization_gate = _number(report.get('max_distance_metres'), 'max_distance_metres')
    fine_gate = _number(report.get('fine_distance_metres', optimization_gate), 'fine_distance_metres')
    if fine_gate <= 0:
        raise ValueError('fine distance gate must be positive')
    gate = _number(report.get('evaluation_distance_metres', optimization_gate), 'evaluation_distance_metres')
    if gate == 0 or optimization_gate == 0:
        raise ValueError('max_distance_metres must be positive')
    if type(report.get('converged')) is not bool:
        raise ValueError('converged must be boolean')
    prior = report.get('initial_transform_supplied', 'unknown')
    if 'initial_transform_supplied' in report and type(prior) is not bool:
        raise ValueError('initial_transform_supplied must be boolean')
    prior_label = ('supplied by caller' if prior is True else
                   'default identity' if prior is False else 'not recorded in this report')
    for name in ('source_file', 'target_file'):
        if not isinstance(report.get(name), str):
            raise ValueError(f'{name} must be text')
    rows = []
    support_rows = [
        ('before_support', 'Before: source → target', source),
        ('aligned_support', 'After: source → target', source),
        ('aligned_reverse_support', 'After: target → source', target),
    ]
    if 'initial_support' in report:
        support_rows.insert(1, ('initial_support', 'Initial pose: source → target', source))
    for key, label, expected in support_rows:
        support = report.get(key)
        if not isinstance(support, dict):
            raise ValueError(f'{key} must be an object')
        if _count(support.get('query_points'), f'{key}.query_points') != expected:
            raise ValueError(f'{key} query count does not match cloud count')
        count = support.get('distance_gated_points')
        if type(count) is not int or not 0 <= count <= expected:
            raise ValueError(f'{key} supported count is invalid')
        fraction = _number(support.get('query_fraction'), f'{key}.query_fraction', maximum=1)
        if not math.isclose(fraction, count / expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{key} fraction does not match counts')
        if _number(support.get('distance_metres'), f'{key}.distance_metres') != gate:
            raise ValueError(f'{key} distance gate does not match report')
        rmse = support.get('gated_rmse_metres')
        if count == 0:
            if rmse is not None:
                raise ValueError(f'{key} empty support must have null RMSE')
            residual = 'None (no points within gate)'
        else:
            _number(rmse, f'{key}.gated_rmse_metres')
            # Native distance gates use float32; allow its rounding at the boundary.
            if rmse > gate and not math.isclose(rmse, gate, rel_tol=1e-6):
                raise ValueError(f'{key} RMSE exceeds the distance gate')
            residual = f'{rmse:.6g} m'
        rows.append(f'<section><h2>{label}</h2><svg viewBox="0 0 100 8" role="img" '
                    f'aria-label="Supported fraction {fraction:.1%}; {expected - count} points excluded from RMSE"><rect width="100" height="8" fill="#e2e8f0"/>'
                    f'<rect width="{fraction * 100:.12g}" height="8" fill="#0369a1"/></svg>'
                    f'<p>{count} / {expected} points ({fraction:.1%}); gated RMSE: {residual}</p>'
                    f'<p>{expected - count} / {expected} points excluded from RMSE ({1 - fraction:.1%}).</p></section>')
    candidates_html = (_candidate_table(report['candidate_selection'], source, target, gate)
                       if 'candidate_selection' in report else '')
    if candidates_html:
        selection = report['candidate_selection']
        selected = selection['candidates'][selection['selected_index']]
        for key in ['aligned_support', 'aligned_reverse_support']:
            if selected[key] != report[key]:
                raise ValueError('selected candidate support differs from main report')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Alignment support report</title><style>body{font:16px system-ui;max-width:850px;margin:2rem auto;padding:1rem}'
            'svg{width:100%;max-height:55px}.trace-chart{max-height:250px}figure{margin:1rem 0}h2{font-size:1.1rem}section{margin:2rem 0}td,th{padding:.5rem;text-align:left}</style>'
            '<main><h1>Alignment support report</h1>'
            f'<p>Source: {html.escape(report["source_file"])}<br>Target: {html.escape(report["target_file"])}</p>'
            f'<p>Evaluation distance gate: {gate:.6g} m. ICP correspondence gate: {optimization_gate:.6g} m. '
            f'ICP fine correspondence gate: {fine_gate:.6g} m. '
            f'ICP converged: {str(report["converged"]).lower()}.</p>'
            f'<p>Initial pose: {prior_label}. A supplied prior is not independently verified.</p>'
            + ''.join(rows) + candidates_html + _trim_summary(report) + _trace_sections(report) +
            '<p>Each direction uses its own query point count. High forward support with lower reverse support '
            'can indicate partial overlap or different sampling densities; it does not identify the cause.</p>'
            '<p>RMSE includes only points within the distance gate. Low RMSE and convergence do not certify '
            'the correct pose. Repeated geometry can produce convincing but incorrect matches.</p></main></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    rendered = render_report(json.loads(args.report.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as destination:
        destination.write(rendered)


if __name__ == '__main__':
    main()
