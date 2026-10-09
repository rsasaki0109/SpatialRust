"""Reference direction, strict records and independent publisher-score calculations."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import spatialrust as sr

SCRIPTS = Path(__file__).resolve().parents[3] / 'scripts'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import redwood_reference
    import evaluate_redwood_logs
    return redwood_reference, evaluate_redwood_logs


def log(pair=(0, 2), matrix=None, count=4):
    matrix = np.eye(4) if matrix is None else matrix
    return f'{pair[0]} {pair[1]} {count}\n' + '\n'.join(' '.join(str(v) for v in row) for row in matrix) + '\n'


def test_pair_direction_and_inverse(modules):
    reader, _ = modules
    pose = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], float)
    records = reader.parse_records(log(matrix=pose))
    direct, pair, inverted = reader.select_pose(records, 2, 0)
    np.testing.assert_array_equal(direct, pose)
    assert pair == (0, 2) and not inverted
    inverse, _, inverted = reader.select_pose(records, 0, 2)
    expected = np.array([[0, 1, 0, -2], [-1, 0, 0, 1], [0, 0, 1, -3], [0, 0, 0, 1]], float)
    np.testing.assert_allclose(inverse, expected, atol=1e-15)
    assert inverted
    direct[0, 0] = 123
    assert records[pair]['matrix'][0, 0] == 0


@pytest.mark.parametrize('text', ['', '0 2 4\n1 0 0 0\n', log() + log(),
    log(pair=(2, 0)), log(pair=(0, 4)), log() + log(pair=(1, 3), count=5),
    log().replace('1.0 0.0 0.0 0.0', 'nan 0 0 0'),
    log().replace('0 2 4', '0.0 2 4'), log().replace('1.0 0.0 0.0 0.0', '1 0 0')])
def test_bad_pose_records_rejected(modules, text):
    with pytest.raises(ValueError):
        modules[0].parse_records(text)


@pytest.mark.parametrize('bad', ['asymmetric', 'negative', 'zero_normalizer'])
def test_information_constraints(modules, bad):
    matrix = np.eye(6)
    if bad == 'asymmetric':
        matrix[0, 1] = .1
    elif bad == 'negative':
        matrix[5, 5] = -1
    else:
        matrix[0, 0] = 0
    with pytest.raises(ValueError):
        modules[0].parse_records(log(matrix=matrix), 6)


def test_large_information_matrix_remains_finite(modules):
    records = modules[0].parse_records(log(matrix=np.eye(6) * 1e308), 6)
    assert np.isfinite(records[(0, 2)]['matrix']).all()


def test_near_rigid_opt_in_preserves_original_scoring_matrix(modules):
    reader, _ = modules
    matrix = np.eye(4)
    matrix[:3, :3] *= .99996
    with pytest.raises(ValueError):
        reader.parse_records(log(matrix=matrix))
    record = reader.parse_records(log(matrix=matrix), project_rotation=True)[(0, 2)]
    np.testing.assert_allclose(record['matrix'], np.eye(4), atol=1e-15)
    np.testing.assert_array_equal(record['original_matrix'], matrix)
    assert record['rotation_correction']['correction_frobenius'] > 0


def test_information_error_matches_analytic_quaternion_with_cross_term(modules):
    _, metric = modules
    angle = np.deg2rad(30)
    delta = np.array([[np.cos(angle), -np.sin(angle), 0, .1],
                      [np.sin(angle), np.cos(angle), 0, .2],
                      [0, 0, 1, .3], [0, 0, 0, 1]])
    gt = np.array([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], float)
    info = np.eye(6)
    info[0, 5] = info[5, 0] = .2
    expected = .1**2 + .2**2 + .3**2 + np.sin(angle/2)**2 + .4 * .1 * np.sin(angle/2)
    assert metric.squared_information_error(gt, gt @ delta, info) == pytest.approx(expected)
    assert metric.squared_information_error(gt, gt, info) == 0


def test_recall_precision_false_positives_and_consecutive_exclusion(modules):
    reader, metric = modules
    truth = reader.parse_records(log((0, 2)) + log((0, 3)) + log((1, 2)))
    information = reader.parse_records(''.join(log(pair, np.eye(6)) for pair in truth), 6)
    wrong = np.eye(4)
    wrong[0, 3] = 1
    predictions = reader.parse_records(log((0, 2)) + log((0, 3), wrong) + log((1, 3)) + log((1, 2)))
    result = metric.evaluate_predictions(predictions, truth, information)
    assert result['recall'] == .5 and result['precision'] == pytest.approx(1/3)
    assert result['false_positive_pairs'] == 1 and result['evaluated_predictions'] == 3
    empty = metric.evaluate_predictions({}, truth, information)
    assert empty['recall'] == 0 and empty['precision'] is None
    with pytest.raises(ValueError, match='pair sets'):
        metric.evaluate_predictions(predictions, truth, {})


def test_half_turn_is_explicit_undefined_score(modules):
    reader, metric = modules
    truth = reader.parse_records(log())
    information = reader.parse_records(log(matrix=np.eye(6)), 6)
    predictions = reader.parse_records(log(matrix=np.diag([-1, -1, 1, 1])))
    result = metric.evaluate_predictions(predictions, truth, information)
    assert result['undefined_scores'] == 1
    assert result['correct'] == 0 and result['rows'][0]['score_squared'] is None


def test_original_fragment_binding_cli(tmp_path):
    xyz = np.eye(3, dtype=np.float32)
    paths = [tmp_path / 'cloud_bin_0.ply', tmp_path / 'cloud_bin_2.ply']
    for path in paths:
        sr.write(str(path), sr.PointCloud.from_xyz(xyz))
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    gt = tmp_path / 'gt.log'
    gt.write_text(log(matrix=pose))
    output = tmp_path / 'reference'
    command = [sys.executable, str(SCRIPTS / 'prepare_3dmatch_reference.py'),
        '--fragments-dir', str(tmp_path), '--gt-log', str(gt), '--source-id', '0', '--target-id', '2',
        '--provenance-url', 'https://example.com/pinned-fixture', '--description', 'Test pair', '--output-dir', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    reference = json.loads((output / 'reference.json').read_text())
    assert reference['provenance']['stored_pose_inverted']
    np.testing.assert_array_equal(np.array(reference['transform_source_to_target'])[:3, 3], [-1, -2, -3])
    assert reference['input_file_sha256']['source'] == hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert reference['input_file_sha256']['target'] == hashlib.sha256(paths[1].read_bytes()).hexdigest()
    assert reference['provenance']['gt_log_sha256'] == hashlib.sha256(gt.read_bytes()).hexdigest()
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_saved_log_cli_complementarity_and_empty_precision(tmp_path):
    truth = tmp_path / 'gt.log'
    information = tmp_path / 'gt.info'
    truth.write_text(log((0, 2)) + log((0, 3)))
    information.write_text(log((0, 2), np.eye(6)) + log((0, 3), np.eye(6)))
    predictions = [tmp_path / 'first.log', tmp_path / 'second.log', tmp_path / 'empty.log']
    for path, data in zip(predictions, [log((0, 2)), log((0, 3)), '']):
        path.write_text(data)
    output = tmp_path / 'evaluation'
    command = [sys.executable, str(SCRIPTS / 'evaluate_redwood_logs.py'), '--gt-log', str(truth),
        '--gt-info', str(information), '--predictions', *map(str, predictions),
        '--provenance-url', 'https://example.com/pinned-results', '--output-dir', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    result = json.loads((output / 'evaluation.json').read_text())
    assert [r['recall'] for r in result['methods']] == [.5, .5, 0]
    assert result['methods'][2]['precision'] is None
    pair = result['complementarity'][0]
    assert pair['first_only'] == pair['second_only'] == 1
    assert pair['oracle_union_correct'] == 2 and not pair['deployable_selector_established']
    assert result['gt_log_sha256'] == hashlib.sha256(truth.read_bytes()).hexdigest()
    assert 'Oracle union' in (output / 'report.html').read_text()
    assert subprocess.run(command, capture_output=True).returncode != 0
