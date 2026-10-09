"""Offline comparison rejects mixed builds and unrelated parameter changes."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[3]/'scripts/report_icp_iterations.py'


def run_receipt(root, trim=None):
    root.mkdir()
    settings = dict(iterations=30, max_distance=.1, fine_distance=.02, evaluation_distance=.02)
    if trim is not None:
        settings['trim_fraction'] = trim
    receipt = dict(settings=settings, files_sha256={'source':'a','target':'b'}, initial_transforms=['same poses'],
                   native_sha256='same native', source_sha256={'example':'same source'},
                   optimized_matches_independent=True, aligned_xyz_roundtrip_exact=True, optimized_seconds=1.)
    (root/'receipt.json').write_text(json.dumps(receipt))
    support = dict(query_fraction=.5, gated_rmse_metres=.01)
    candidate = dict(index=0, status='success', converged=False, aligned_support=support, aligned_reverse_support=support)
    (root/'alignment.json').write_text(json.dumps(dict(candidate_selection=dict(selected_index=0,candidates=[candidate]))))
    return receipt


def command(first, second, output):
    return [sys.executable,str(SCRIPT),str(first),str(second),'--vary','trim-fraction','--output-dir',str(output)]


def test_default_trim_is_normalized_and_report_displays_both_fractions(tmp_path):
    first, second, output = [tmp_path/name for name in ('all','trimmed','output')]
    run_receipt(first)
    run_receipt(second,.8)
    result = subprocess.run(command(first,second,output),capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr
    comparison = json.loads((output/'comparison.json').read_text())
    assert comparison['controlled_inputs_and_settings']
    assert {r['trim_fraction'] for r in comparison['trials']} == {1.,.8}
    assert '80.0%' in (output/'report.html').read_text()


@pytest.mark.parametrize('change',['native','missing_native','source','gate'])
def test_trimming_comparison_rejects_uncontrolled_changes(tmp_path,change):
    first, second, output = [tmp_path/name for name in ('all','trimmed','output')]
    run_receipt(first)
    receipt = run_receipt(second,.8)
    if change == 'native':
        receipt['native_sha256'] = 'different native'
    elif change == 'missing_native':
        receipt.pop('native_sha256')
    elif change == 'source':
        receipt['source_sha256']['example'] = 'changed pipeline'
    else:
        receipt['settings']['max_distance'] = .2
    (second/'receipt.json').write_text(json.dumps(receipt))
    result = subprocess.run(command(first,second,output),capture_output=True,text=True,timeout=30)
    assert result.returncode != 0 and not output.exists()
