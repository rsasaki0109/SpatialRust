"""A real-binding paired study keeps improvements and regressions visible."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT/'scripts/study_trimmed_icp.py'


def test_paired_trim_study_reports_counterexamples_and_full_source_support(tmp_path):
    output = tmp_path/'trim-study'
    command = [sys.executable, str(SCRIPT), '--output-dir', str(output), '--seeds', '1']
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    receipt = json.loads((output/'study.json').read_text())
    rows = receipt['rows']
    assert len(rows) == 162 and all(r['status'] == 'success' for r in rows)
    for path, expected in receipt['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    assert len(receipt['native_sha256']) == 64
    def case(name, angle, fraction):
        return next(r for r in rows if (r['case'],r['initial_error_degrees'],r['gate'],r['trim_fraction']) == (name,angle,.6,fraction))
    assert not case('near_outliers',0,1.)['recovered']
    assert case('near_outliers',0,.8)['recovered']
    assert case('clean',40,1.)['recovered']
    assert not case('clean',40,.8)['recovered']
    for row in rows:
        assert row['evaluation_distance'] == .05
        assert row['first_update_retained_points'] == row['first_update_retained_inliers'] + row['first_update_retained_outliers']
        assert row['forward_support'] == row['forward_supported_points']/row['source_points']
    ring = case('symmetric_ring',20,.8)
    assert ring['pose_ambiguity_known'] and not ring['recovered']
    rendered = (output/'report.html').read_text()
    assert 'paired recoveries gained' in rendered and 'lost versus all pairs' in rendered
    assert 'multiple equivalent poses' in rendered
    assert '<script' not in rendered and 'src=' not in rendered
    original = (output/'study.json').read_bytes()
    repeat = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert repeat.returncode != 0 and original == (output/'study.json').read_bytes()
