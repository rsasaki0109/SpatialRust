"""End-to-end study receipt and ambiguous-geometry safeguards using real bindings."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'scripts/study_icp_convergence.py'


def test_controlled_study_records_false_convergence_and_rejects_overwrite(tmp_path):
    output = tmp_path / 'study'
    command = [sys.executable, str(SCRIPT), '--output-dir', str(output), '--seeds', '1',
               '--iterations', '40', '--scales', '.001', '1', '--angles', '0', '20']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output/'study.json').read_text())
    assert receipt['case_count'] == len(receipt['results']) == 60
    assert len(receipt['native_sha256']) == 64
    for name, expected in receipt['source_sha256'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    rows = receipt['results']
    assert all(row['status'] == 'success' for row in rows)
    clean = [row for row in rows if row['case'] == 'clean_volume' and row['initial_error_degrees'] == 0]
    assert all(row['recovered_generating_pose'] for row in clean)
    ring = [row for row in rows if row['case'] == 'symmetric_ring' and row['initial_error_degrees'] == 20]
    assert all(row['forward_support'] > .99 and not row['recovered_generating_pose'] for row in ring)
    assert any(row['converged'] and row['false_convergence'] for row in ring)
    budget = [row for row in rows if row['policy'] == 'budget_only']
    assert all(row['iterations'] == 40 and row['stop_reason'] == 'iteration_limit' for row in budget)
    assert any(not row['recovered_generating_pose'] for row in budget)
    for row in rows:
        assert len(row['history']) == row['iterations']
        assert row['false_convergence'] == (row['converged'] and not row['recovered_generating_pose'])
    rendered = (output/'report.html').read_text()
    assert 'multiple equivalent poses' in rendered and 'no universal ranking' in rendered
    assert '<script' not in rendered and 'src=' not in rendered
    original = (output/'study.json').read_bytes()
    repeat = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeat.returncode != 0 and (output/'study.json').read_bytes() == original


def test_invalid_study_settings_do_not_create_output(tmp_path):
    output = tmp_path/'invalid'
    result = subprocess.run([sys.executable, str(SCRIPT), '--output-dir', str(output),
                             '--scales', 'nan'], capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and not output.exists()
