"""The real audit keeps its exact 20GiB gate before any input read or output."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_tum_dense_rgbd as dense_audit


@pytest.mark.parametrize('available', [0, 20 * 1024**3 - 1])
def test_audit_rejects_low_space_before_inputs_or_output(tmp_path, monkeypatch, synthetic_dense_audit_space, available):
    synthetic_dense_audit_space.free = available
    monkeypatch.setattr(dense_audit, 'file_sha256', lambda _path: pytest.fail('input read before storage admission'))
    output = tmp_path / 'audit'
    with pytest.raises(ValueError, match='20GiB'):
        dense_audit.audit(tmp_path / 'prepared', '1' * 64, tmp_path / 'estimates.json', '2' * 64, output)
    assert not output.exists()


def test_audit_exact_free_space_boundary_reaches_frozen_input_check(tmp_path, monkeypatch, synthetic_dense_audit_space):
    synthetic_dense_audit_space.free = 20 * 1024**3
    reads = []
    monkeypatch.setattr(dense_audit, 'file_sha256', lambda path: reads.append(path) or '0' * 64)
    output = tmp_path / 'audit'
    with pytest.raises(ValueError, match='frozen inputs/estimates changed'):
        dense_audit.audit(tmp_path / 'prepared', '1' * 64, tmp_path / 'estimates.json', '2' * 64, output)
    assert reads == [tmp_path / 'prepared' / 'manifest.json']
    assert not output.exists()
