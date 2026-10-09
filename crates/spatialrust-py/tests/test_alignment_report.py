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
    assert '3 / 3 points excluded from RMSE (100.0%)' in rendered
    assert '1 / 4 points excluded from RMSE (25.0%)' in rendered
    assert '<script' not in rendered and 'src=' not in rendered


def test_report_displays_distinct_fine_gate_and_rejects_invalid_value():
    report = fixture()
    report['fine_distance_metres'] = .02
    assert 'ICP fine correspondence gate: 0.02 m' in module.render_report(report)
    assert 'Evaluation distance gate: 0.1 m' in module.render_report(report)
    report['fine_distance_metres'] = 0
    with pytest.raises(ValueError, match='fine distance'):
        module.render_report(report)


def trace_fixture():
    report = fixture()
    history = [dict(iteration=i, correspondences=3, evaluated_correspondences=3,
                    fitness_metres_squared=f, fitness_change_metres_squared=change,
                    translation_delta_metres=0, rotation_delta_radians=0)
               for i, f, change in [(1, .001, None), (2, .0005, .0005)]]
    report['stages'] = [dict(name='<trace>', source_points=3, iterations=2,
                            converged=False, stop_reason='iteration_limit',
                            kernel_fitness_metres_squared=.0005, icp_history=history)]
    return report


def test_trace_charts_are_escaped_standalone_and_preserve_input():
    report = trace_fixture()
    original = copy.deepcopy(report)
    rendered = module.render_report(report)
    assert rendered.count('class="trace-chart"') == 4
    assert 'ICP iteration history: &lt;trace&gt;' in rendered
    assert 'Horizontal axis: iteration' in rendered
    assert 'membership changes' in rendered
    assert '<script' not in rendered and 'src=' not in rendered
    assert report == original


@pytest.mark.parametrize('field,value', [
    ('iteration', 4), ('correspondences', True), ('evaluated_correspondences', 4),
    ('fitness_metres_squared', float('nan')), ('fitness_change_metres_squared', 7),
    ('translation_delta_metres', -1), ('rotation_delta_radians', 4),
])
def test_malformed_trace_row_is_rejected(field, value):
    report = trace_fixture()
    report['stages'][0]['icp_history'][1][field] = value
    with pytest.raises(ValueError):
        module.render_report(report)


@pytest.mark.parametrize('field,value', [
    ('stop_reason', 'unknown'), ('converged', True), ('iterations', 3),
    ('kernel_fitness_metres_squared', .002),
])
def test_inconsistent_trace_stage_is_rejected(field, value):
    report = trace_fixture()
    report['stages'][0][field] = value
    with pytest.raises(ValueError):
        module.render_report(report)


def test_trace_reason_is_checked_against_explicit_criteria():
    report = trace_fixture()
    stage = report['stages'][0]
    stage['convergence_criteria'] = dict(translation_epsilon=1e-3, rotation_epsilon=1e-4, fitness_epsilon=0)
    with pytest.raises(ValueError, match='thresholds'):
        module.render_report(report)
    stage.update(stop_reason='transform_threshold', converged=True)
    assert 'Stopping thresholds:' in module.render_report(report)
    stage['convergence_criteria']['fitness_epsilon'] = 1
    with pytest.raises(ValueError, match='thresholds'):
        module.render_report(report)


def candidate_fixture():
    report = fixture()
    identity = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    quarter = [[0,-1,0,0], [1,0,0,0], [0,0,1,0], [0,0,0,1]]
    half = [[-1,0,0,0], [0,-1,0,0], [0,0,1,0], [0,0,0,1]]
    report['candidate_selection'] = dict(
        rule='maximum_forward_supported_points_then_minimum_gated_rmse_then_input_order',
        candidate_count=3, successful_candidates=3, selected_index=0,
        candidates=[dict(index=i, status='success', converged=True,
            aligned_support=copy.deepcopy(report['aligned_support']),
            aligned_reverse_support=copy.deepcopy(report['aligned_reverse_support']),
            transform_source_to_target=pose) for i, pose in enumerate((identity, quarter, half))])
    return report


def test_candidate_rotation_disagreement_without_numpy():
    report = candidate_fixture()
    unchanged = copy.deepcopy(report)
    rendered = module.render_report(report)
    for angle in ('0.00°', '90.00°', '180.00°'):
        assert angle in rendered
    assert 'Rotation from selected' in rendered
    assert 'not ground-truth error' in rendered
    assert report == unchanged


def test_older_candidate_report_without_poses_stays_readable():
    report = candidate_fixture()
    for candidate in report['candidate_selection']['candidates']:
        candidate.pop('transform_source_to_target')
    assert module.render_report(report).count('Unavailable') == 6


def test_centroid_disagreement_is_invariant_to_source_origin():
    report = candidate_fixture()
    selection = report['candidate_selection']
    selection['source_centroid_xyz_metres'] = [10, 0, 0]
    # Quarter turn about the source centroid preserves its location despite
    # differing pose translation. The last candidate moves it by (3, 4, 0).
    selection['candidates'][1]['transform_source_to_target'][0][3] = 10
    selection['candidates'][1]['transform_source_to_target'][1][3] = -10
    selection['candidates'][2]['transform_source_to_target'][0][3] = 23
    selection['candidates'][2]['transform_source_to_target'][1][3] = 4
    rendered = module.render_report(report)
    assert rendered.count('<td>0 m</td></tr>') == 2
    assert '<td>5 m</td></tr>' in rendered
    shift = [100, -20, 4]
    selection['source_centroid_xyz_metres'] = [10 + shift[0], shift[1], shift[2]]
    for candidate in selection['candidates']:
        pose = candidate['transform_source_to_target']
        for i in range(3):
            pose[i][3] -= sum(pose[i][j]*shift[j] for j in range(3))
    assert module.render_report(report) == rendered


@pytest.mark.parametrize('centroid', [[], [0, 0], [0, 0, 0, 0],
    [True, 0, 0], [float('nan'), 0, 0], [0, float('inf'), 0]])
def test_invalid_candidate_centroid_rejected(centroid):
    report = candidate_fixture()
    report['candidate_selection']['source_centroid_xyz_metres'] = centroid
    with pytest.raises(ValueError, match='centroid'):
        module.render_report(report)


@pytest.mark.parametrize('pose', [[], [[1]*4]*3,
    [[2,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    [[-1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    [[1,0,0,float('nan')], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    [[True,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]],
    [[1,0,0,0], [0,1,0,0], [0,0,1,0], [1,0,0,1]]])
def test_invalid_candidate_pose_rejected(pose):
    report = candidate_fixture()
    report['candidate_selection']['candidates'][1]['transform_source_to_target'] = pose
    with pytest.raises(ValueError, match='candidate'):
        module.render_report(report)


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


@pytest.mark.parametrize('supplied,label', [(True, 'supplied by caller'), (False, 'default identity')])
def test_prior_provenance(supplied, label):
    report = fixture()
    assert 'not recorded in this report' in module.render_report(report)
    report['initial_transform_supplied'] = supplied
    assert f'Initial pose: {label}' in module.render_report(report)
    report['initial_transform_supplied'] = 1
    with pytest.raises(ValueError, match='initial_transform_supplied'):
        module.render_report(report)


def test_low_rmse_does_not_hide_excluded_source_points():
    report = fixture()
    report['source_points'] = 5
    for key in ['before_support', 'aligned_support']:
        report[key].update(query_points=5, distance_gated_points=3,
                           query_fraction=.6, gated_rmse_metres=1e-8)
    rendered = module.render_report(report)
    assert '2 / 5 points excluded from RMSE (40.0%)' in rendered
    assert 'gated RMSE: 1e-08 m' in rendered


def test_initial_support_is_optional_validated_and_between_before_and_after():
    report = fixture()
    assert 'Initial pose: source → target' not in module.render_report(report)
    report['initial_support'] = copy.deepcopy(report['aligned_support'])
    rendered = module.render_report(report)
    assert rendered.index('Before: source') < rendered.index('Initial pose: source') < rendered.index('After: source')
    report['initial_support']['query_fraction'] = .5
    with pytest.raises(ValueError, match='fraction does not match'):
        module.render_report(report)


def test_distinct_optimization_and_evaluation_gates():
    report = fixture()
    report['evaluation_distance_metres'] = .1
    report['max_distance_metres'] = .6
    rendered = module.render_report(report)
    assert 'Evaluation distance gate: 0.1 m' in rendered
    assert 'ICP correspondence gate: 0.6 m' in rendered
    report['evaluation_distance_metres'] = 0
    with pytest.raises(ValueError):
        module.render_report(report)


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
