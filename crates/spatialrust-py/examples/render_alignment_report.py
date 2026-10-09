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


def render_report(report):
    """Validate diagnostic fields and return HTML; no network or scripts required."""
    if not isinstance(report, dict) or report.get('schema_version') != 'spatialrust.python-alignment.v1':
        raise ValueError('unsupported alignment report schema')
    source = _count(report.get('source_points'), 'source_points')
    target = _count(report.get('target_points'), 'target_points')
    gate = _number(report.get('max_distance_metres'), 'max_distance_metres')
    if gate == 0:
        raise ValueError('max_distance_metres must be positive')
    if type(report.get('converged')) is not bool:
        raise ValueError('converged must be boolean')
    for name in ('source_file', 'target_file'):
        if not isinstance(report.get(name), str):
            raise ValueError(f'{name} must be text')
    rows = []
    for key, label, expected in (
        ('before_support', 'Before: source → target', source),
        ('aligned_support', 'After: source → target', source),
        ('aligned_reverse_support', 'After: target → source', target),
    ):
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
                    f'aria-label="Supported fraction {fraction:.1%}"><rect width="100" height="8" fill="#e2e8f0"/>'
                    f'<rect width="{fraction * 100:.12g}" height="8" fill="#0369a1"/></svg>'
                    f'<p>{count} / {expected} points ({fraction:.1%}); gated RMSE: {residual}</p></section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>Alignment support report</title><style>body{font:16px system-ui;max-width:850px;margin:2rem auto;padding:1rem}'
            'svg{width:100%;max-height:55px}h2{font-size:1.1rem}section{margin:2rem 0}</style>'
            '<main><h1>Alignment support report</h1>'
            f'<p>Source: {html.escape(report["source_file"])}<br>Target: {html.escape(report["target_file"])}</p>'
            f'<p>Distance gate: {gate:.6g} m. ICP converged: {str(report["converged"]).lower()}.</p>'
            + ''.join(rows) +
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
