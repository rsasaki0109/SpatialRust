"""Explicit near-rotation correction, safe arrays and exact prepared PCD bytes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / 'scripts'


@pytest.fixture
def prepare(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('numpy_reference', SCRIPTS / 'prepare_numpy_reference.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_correction_requires_opt_in_and_records_original(prepare):
    original = np.eye(4)
    original[:3, :3] *= .99996
    with pytest.raises(ValueError):
        prepare.reference_pose(original)
    corrected, receipt = prepare.reference_pose(original, True)
    np.testing.assert_allclose(corrected, np.eye(4), atol=1e-15)
    np.testing.assert_array_equal(receipt['original_rotation'], original[:3, :3])
    assert receipt['correction_frobenius'] == pytest.approx(np.sqrt(3) * .00004)
    np.testing.assert_array_equal(original[:3, :3], np.eye(3) * .99996)


@pytest.mark.parametrize('value', [-1, .99, float('nan')])
def test_large_or_invalid_reference_correction_rejected(prepare, value):
    matrix = np.eye(4)
    matrix[0, 0] = value
    with pytest.raises(ValueError):
        prepare.reference_pose(matrix, True)


@pytest.mark.parametrize('array', [np.ones((2, 3)), np.ones((3, 4)),
    np.full((3, 3), np.nan), np.full((3, 3), 1e100), np.ones((3, 3), dtype=object)])
def test_invalid_points_rejected(prepare, array):
    with pytest.raises(ValueError):
        prepare.points(array)


def test_numpy_cli_roundtrip_and_reference_binding(tmp_path):
    xyz = np.array([[0, 0, 0], [1.123456789, 0, 0], [0, 1, 0]], np.float64)
    for name, value in [('source', xyz), ('target', xyz), ('pose', np.eye(4))]:
        np.save(tmp_path / (name + '.npy'), value)
    output = tmp_path / 'prepared'
    command = [sys.executable, str(SCRIPTS / 'prepare_numpy_reference.py'),
        '--source', str(tmp_path / 'source.npy'), '--target', str(tmp_path / 'target.npy'),
        '--pose', str(tmp_path / 'pose.npy'), '--length-unit', 'm',
        '--provenance-url', 'https://example.com/pinned-pair', '--description', 'Test reference',
        '--output-dir', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    reference = json.loads((output / 'reference.json').read_text())
    assert reference['provenance']['reference_rotation_correction'] is None
    assert reference['provenance']['f32_conversion_max_abs_error']['source'] > 0
    np.testing.assert_array_equal(sr.read(str(output / 'source.pcd')).xyz(), xyz.astype(np.float32))
    assert subprocess.run(command, capture_output=True).returncode != 0
