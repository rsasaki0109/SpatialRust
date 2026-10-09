"""Analytical post-fit attribution and reference-safe refinement replay."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_pose_reference import reference

SCRIPTS = Path(__file__).resolve().parents[3] / 'scripts'


@pytest.mark.parametrize('initial,final,expected', [(True, True, 'retained'),
    (True, False, 'lost'), (False, True, 'recovered'), (False, False, 'unrecovered')])
def test_transition_accounts_for_both_endpoints(monkeypatch, initial, final, expected):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from diagnose_3dmatch_stages import transition
    assert transition(initial, final) == expected


def test_stage_scores_locate_refinement_loss(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from diagnose_3dmatch_stages import stage_scores
    initial, final = np.eye(4), np.eye(4)
    initial[0, 3], final[0, 3] = .1, .3
    report = dict(initial_transform_source_to_target=initial.tolist(),
                  transform_source_to_target=final.tolist(),
                  stages=[dict(name='fine', transform_source_to_target=final.tolist())])
    result = stage_scores(report, np.eye(4), np.eye(6))
    assert [row['correct'] for row in result] == [True, False]
    assert [row['score_squared'] for row in result] == pytest.approx([.01, .09])
    report['transform_source_to_target'] = initial.tolist()
    with pytest.raises(ValueError, match='final stage'):
        stage_scores(report, np.eye(4), np.eye(6))


def test_undefined_half_turn_is_retained_as_incorrect(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from diagnose_3dmatch_stages import stage_scores
    final = np.diag([-1., -1., 1., 1.]).tolist()
    report = dict(initial_transform_source_to_target=np.eye(4).tolist(),
                  transform_source_to_target=final,
                  stages=[dict(name='fine', transform_source_to_target=final)])
    result = stage_scores(report, np.eye(4), np.eye(6))
    assert result[-1]['correct'] is False
    assert result[-1]['score_squared'] is None
    assert 'singular' in result[-1]['error']


def test_replay_rejects_changed_reference_before_loading_points(tmp_path):
    ref, comparison = tmp_path / 'reference.json', tmp_path / 'comparison.json'
    ref.write_text(json.dumps(reference()))
    comparison.write_text(json.dumps(dict(reference_sha256='c' * 64)))
    output = tmp_path / 'output'
    completed = subprocess.run([sys.executable, str(SCRIPTS / 'study_public_refinement.py'),
        '--source', str(tmp_path / 'absent-source.ply'), '--target', str(tmp_path / 'absent-target.ply'),
        '--reference', str(ref), '--comparison', str(comparison), '--output-dir', str(output)],
        capture_output=True, text=True)
    assert completed.returncode != 0
    assert 'replay reference differs' in completed.stderr
    assert not output.exists()
