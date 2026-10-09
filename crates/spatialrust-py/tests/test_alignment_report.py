"""Standalone report contracts without requiring the native extension."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import json

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'render_alignment_report.py'
spec = importlib.util.spec_from_file_location('alignment_report', EXAMPLE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    def support(n, count):
        return dict(query_points=n, distance_gated_points=count, query_fraction=count/n,
                    distance_metres=.1, gated_rmse_metres=0. if count else None)
    return dict(schema_version='spatialrust.python-alignment.v1', source_points=3,
                target_points=4, source_file='<script>alert("x")</script>', target_file='target.pcd',
                max_distance_metres=.1, converged=True, before_support=support(3, 0),
                aligned_support=support(3, 3), aligned_reverse_support=support(4, 3))


def test_partial_overlap_and_escaped_content():
    rendered = module.render_report(fixture())
    assert '3 / 3 points (100.0%)' in rendered
    assert '3 / 4 points (75.0%)' in rendered
    assert 'None (no points within gate)' in rendered
    assert '&lt;script&gt;' in rendered and '<script>' not in rendered
    assert 'do not certify' in rendered
    assert 'different sampling densities' in rendered
    assert '<script' not in rendered and 'src=' not in rendered


@pytest.mark.parametrize('key,value', [
    ('schema_version', 'unknown'), ('source_points', True), ('target_points', 0),
    ('max_distance_metres', float('inf')), ('max_distance_metres', 0),
    ('converged', 1), ('source_file', None), ('aligned_support', []),
])
def test_invalid_report_fields(key, value):
    report = fixture()
    report[key] = value
    with pytest.raises(ValueError):
        module.render_report(report)


@pytest.mark.parametrize('key,value', [
    ('query_points', 4), ('distance_gated_points', -1), ('distance_gated_points', 4),
    ('distance_gated_points', True), ('query_fraction', .5), ('query_fraction', float('nan')),
    ('distance_metres', .2), ('gated_rmse_metres', None), ('gated_rmse_metres', -.1),
    ('gated_rmse_metres', .2), ('gated_rmse_metres', float('inf')),
])
def test_invalid_support_fields(key, value):
    report = copy.deepcopy(fixture())
    report['aligned_support'][key] = value
    with pytest.raises(ValueError):
        module.render_report(report)


def test_empty_support_requires_null_rmse():
    report = fixture()
    report['before_support']['gated_rmse_metres'] = 0
    with pytest.raises(ValueError):
        module.render_report(report)


def test_native_float32_gate_rounding_is_accepted():
    report = fixture()
    report['aligned_support']['gated_rmse_metres'] = .10000000149
    assert 'gated RMSE: 0.1 m' in module.render_report(report)


def test_cli_exclusive_output_and_validation_before_write(tmp_path):
    report = tmp_path / 'alignment.json'
    output = tmp_path / 'report.html'
    report.write_text(json.dumps(fixture()), encoding='utf-8')
    subprocess.run([sys.executable, str(EXAMPLE), str(report), str(output)], check=True)
    original = output.read_bytes()
    result = subprocess.run([sys.executable, str(EXAMPLE), str(report), str(output)], capture_output=True)
    assert result.returncode != 0
    assert output.read_bytes() == original
    report.write_text('{}', encoding='utf-8')
    absent = tmp_path / 'invalid.html'
    result = subprocess.run([sys.executable, str(EXAMPLE), str(report), str(absent)], capture_output=True)
    assert result.returncode != 0
    assert not absent.exists()
