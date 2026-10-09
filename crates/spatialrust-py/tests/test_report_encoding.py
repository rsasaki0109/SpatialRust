"""CLI artifacts use UTF-8 even when the process locale does not."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_pose_reference import reference, report
from test_redwood_reference import log

SCRIPTS = Path(__file__).resolve().parents[3] / 'scripts'


@pytest.mark.parametrize('workflow', ['pose', 'publisher'])
def test_cli_report_encoding_is_independent_of_locale(tmp_path, workflow):
    output = tmp_path / 'output'
    if workflow == 'pose':
        ref = reference()
        ref['provenance']['description'] = '参照姿勢の検証'
        paths = [tmp_path / 'reference.json', tmp_path / 'alignment.json']
        for path, value in zip(paths, [ref, report(ref)]):
            path.write_bytes(json.dumps(value, ensure_ascii=False).encode('utf-8'))
        args = ['evaluate_pose_reference.py', '--reference', str(paths[0]), '--reports', str(paths[1])]
        expected = '参照姿勢の検証'
    else:
        paths = [tmp_path / name for name in ('gt.log', 'gt.info', 'prediction.log')]
        for path, value in zip(paths, [log(), log(matrix=np.eye(6)), log()]):
            path.write_bytes(value.encode('utf-8'))
        args = ['evaluate_redwood_logs.py', '--gt-log', str(paths[0]), '--gt-info', str(paths[1]),
                '--predictions', str(paths[2]), '--provenance-url', 'https://example.org/reference']
        expected = '≤0.04'
    environment = dict(os.environ, PYTHONUTF8='0', PYTHONCOERCECLOCALE='0', LC_ALL='C')
    completed = subprocess.run([sys.executable, str(SCRIPTS / args[0]), *args[1:],
                                '--output-dir', str(output)], env=environment, capture_output=True)
    assert completed.returncode == 0, completed.stderr.decode('utf-8', errors='replace')
    rendered = (output / 'report.html').read_bytes().decode('utf-8')
    assert expected in rendered
    assert 'charset="utf-8"' in rendered
    receipts = list(output.glob('*.json'))
    assert len(receipts) == 1
    json.loads(receipts[0].read_bytes().decode('utf-8'))
