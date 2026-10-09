"""Optional shared-input no-prior reference comparison smoke test."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('open3d')
sr = pytest.importorskip('spatialrust')
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from compare_public_global import run
from evaluate_pose_reference import evaluate
from study_public_refinement import initializations


def test_known_transform_without_reference_in_solver():
    target = np.random.default_rng(12).uniform(-1, 1, (120, 3)).astype(np.float32)
    offset = np.array([.02, -.01, .03], np.float32)
    source = target + offset
    hashes = dict(source='a' * 64, target='b' * 64)
    rows = run(sr.PointCloud.from_xyz(source), sr.PointCloud.from_xyz(target), hashes, [7], 2000)
    reference = dict(schema='spatialrust.pose-reference.v1', length_unit='m',
        input_file_sha256=hashes, provenance=dict(kind='synthetic', description='Known translation'),
        transform_source_to_target=np.eye(4).tolist())
    reference['transform_source_to_target'][0][3] = -.02
    reference['transform_source_to_target'][1][3] = .01
    reference['transform_source_to_target'][2][3] = -.03
    assert len(rows) == 2
    comparison = dict(schema='spatialrust.public-global-comparison.v1', input_file_sha256=hashes, rows=rows)
    generated = initializations(comparison, hashes)
    assert len(generated) == 1 and generated[0][0] == 7
    for row in rows:
        assert row['status'] == 'success'
        assert not row['report']['initial_transform_supplied']
        error = evaluate(row['report'], reference)
        assert error['rotation_error_degrees'] < .001
        assert error['translation_error'] < 1e-5
        assert row['common_support']['count'] == 120


def test_replay_rejects_changed_inputs_and_empty_comparison():
    hashes = dict(source='a' * 64, target='b' * 64)
    comparison = dict(schema='spatialrust.public-global-comparison.v1', input_file_sha256=hashes, rows=[])
    with pytest.raises(ValueError, match='no generated'):
        initializations(comparison, hashes)
    with pytest.raises(ValueError, match='input hashes'):
        initializations(comparison, dict(source='c' * 64, target='b' * 64))
