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
    cloud, report = example.evaluate_candidates(*paths, poses())
    selection = report['candidate_selection']
    assert selection['selected_index'] == 1
    assert [c['status'] for c in selection['candidates']] == ['error', 'success', 'success']
    assert 'correspondences' in selection['candidates'][0]['error']
    assert len(cloud) == len(source)
    np.testing.assert_allclose(cloud.xyz(), target, atol=3e-5)
    json.dumps(report, allow_nan=False)


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
