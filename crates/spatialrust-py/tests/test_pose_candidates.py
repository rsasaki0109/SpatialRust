"""File candidate selection records failures and preserves exclusive output."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import spatialrust as sr
from test_alignment_pipeline import fixture_files

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'align_pose_candidates.py'


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))
    spec = importlib.util.spec_from_file_location('pose_candidate_example', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def poses():
    wrong = np.eye(4)
    wrong[:3, 3] = 100
    return [wrong.tolist(), np.eye(4).tolist(), np.eye(4).tolist()]


def test_real_candidates_keep_failures_and_stable_ties(tmp_path, example):
    paths, source, target = fixture_files(tmp_path)
    with patch.object(sr, 'read', wraps=sr.read) as reader:
        cloud, report = example.evaluate_candidates(*paths, poses())
    assert reader.call_count == 2
    assert [call.args[0] for call in reader.call_args_list] == list(map(str, paths))
    selection = report['candidate_selection']
    assert selection['selected_index'] == 1
    assert [c['status'] for c in selection['candidates']] == ['error', 'success', 'success']
    assert 'correspondences' in selection['candidates'][0]['error']
    assert len(cloud) == len(source)
    np.testing.assert_allclose(cloud.xyz(), target, atol=3e-5)
    json.dumps(report, allow_nan=False)
    from render_alignment_report import render_report
    rendered = render_report(report)
    assert 'Pose candidates' in rendered and '1 (selected)' in rendered
    assert 'Failed:' in rendered
    report['candidate_selection']['candidates'][0]['error'] = '<script>bad</script>'
    assert '&lt;script&gt;bad&lt;/script&gt;' in render_report(report)
    report['candidate_selection']['selected_index'] = 2
    with pytest.raises(ValueError, match='ordering'):
        render_report(report)


@pytest.mark.parametrize('values', [[], [np.eye(4).tolist()] * 17, [np.eye(3).tolist()],
                                  [np.eye(4).tolist(), np.eye(3).tolist()], 'invalid'])
def test_all_pose_validation_precedes_io(values, example):
    with patch.object(sr, 'read', side_effect=AssertionError('IO before validation')):
        with pytest.raises(ValueError):
            example.evaluate_candidates('missing', 'missing', values)


def test_candidate_cli_outputs_and_refuses_overwrite(tmp_path):
    paths, _, target = fixture_files(tmp_path)
    pose_path = tmp_path / 'poses.json'
    pose_path.write_text(json.dumps(poses()))
    output = tmp_path / 'selected'
    command = [sys.executable, str(EXAMPLE), *map(str, paths), '--initial-transforms',
               str(pose_path), '--output-dir', str(output), '--html-report']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'alignment.json').read_text())
    assert report['candidate_selection']['selected_index'] == 1
    assert (output / 'report.html').exists()
    assert '1 (selected)' in (output / 'report.html').read_text()
    np.testing.assert_allclose(sr.read(str(output / 'aligned.pcd')).xyz(), target, atol=3e-5)
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    assert subprocess.run(command, capture_output=True, timeout=60).returncode != 0
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original


def test_all_candidates_failed_no_output(tmp_path):
    paths, _, _ = fixture_files(tmp_path)
    pose_path = tmp_path / 'bad-poses.json'
    pose_path.write_text(json.dumps(poses()[:1]))
    output = tmp_path / 'failed'
    result = subprocess.run([sys.executable, str(EXAMPLE), *map(str, paths),
                             '--initial-transforms', str(pose_path), '--output-dir', str(output)],
                            capture_output=True, timeout=60)
    assert result.returncode != 0
    assert not output.exists()


def test_write_failure_cleans_new_output(tmp_path, example):
    paths, _, _ = fixture_files(tmp_path)
    pose_path = tmp_path / 'poses.json'
    pose_path.write_text(json.dumps(poses()))
    output = tmp_path / 'failed-save'
    with patch.object(sys, 'argv', [str(EXAMPLE), *map(str, paths), '--initial-transforms',
                                    str(pose_path), '--output-dir', str(output)]), patch.object(sr, 'write', side_effect=OSError('injected save failure')):
        with pytest.raises(OSError, match='injected'):
            example.main()
    assert not output.exists()


def test_invalid_settings_precede_shared_reads(example):
    with patch.object(sr, 'read', side_effect=AssertionError('IO before settings validation')):
        with pytest.raises(ValueError, match='evaluation_distance'):
            example.evaluate_candidates('missing', 'missing', poses(), evaluation_distance=0)


def test_shared_target_voxels_match_independent_candidates(tmp_path, example):
    paths, _, _ = fixture_files(tmp_path)
    from align_point_clouds import align_files
    independent = [align_files(*paths, initial_transform=pose)[1] for pose in poses()[1:]]
    voxel = sr.voxel_downsample
    registration = sr.register_icp
    coarse_targets = []

    def check_readonly_target(source, target, *args):
        before = target.xyz().copy()
        coarse_targets.append(target)
        result = registration(source, target, *args)
        np.testing.assert_array_equal(target.xyz(), before)
        return result

    with patch.object(sr, 'voxel_downsample', wraps=voxel) as downsample, \
            patch.object(sr, 'register_icp', side_effect=check_readonly_target), \
            patch.object(sr, 'distance_gated_support', wraps=sr.distance_gated_support) as support:
        _, report = example.evaluate_candidates(*paths, poses())
    # Three source voxelizations, one shared target voxelization, including the
    # candidate that fails during registration.
    assert downsample.call_count == 4
    assert support.call_count == 7  # One before query plus three per success.
    assert coarse_targets[0] is coarse_targets[1] is coarse_targets[3]
    selection = report.pop('candidate_selection')
    assert report == independent[0]
    for actual, expected in zip(selection['candidates'][1:], independent):
        for field in ('transform_source_to_target', 'aligned_support',
                      'aligned_reverse_support', 'converged'):
            assert actual[field] == expected[field]


def test_before_support_cache_is_local_and_reports_are_independent(tmp_path, example):
    paths, _, _ = fixture_files(tmp_path)
    from align_point_clouds import _align_clouds, align_files
    source, target = [sr.read(str(path)) for path in paths]
    cache = []
    first = _align_clouds(source, target, before_support_cache=cache)[1]
    first['before_support']['distance_gated_points'] = -1
    second = _align_clouds(source, target, before_support_cache=cache)[1]
    assert second['before_support'] == align_files(*paths)[1]['before_support']
    # Separate searches must not retain an old evaluation gate or old support.
    for gate in (.1, .001):
        _, report = example.evaluate_candidates(*paths, poses()[1:], evaluation_distance=gate)
        assert report['before_support'] == align_files(*paths, evaluation_distance=gate)[1]['before_support']
