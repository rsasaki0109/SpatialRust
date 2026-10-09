"""Fixed case selection, safe archive members and failure-preserving accounting."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest
import spatialrust as sr

from test_redwood_reference import log

SCRIPTS = Path(__file__).resolve().parents[3] / 'scripts'


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    import prepare_3dmatch_cases
    import compare_3dmatch_cases
    return prepare_3dmatch_cases, compare_3dmatch_cases


def test_selection_is_fixed_header_quantiles(modules):
    records = {(0, 1): {}, (0, 2): {}, (0, 3): {}, (1, 3): {}, (2, 4): {}, (3, 5): {}}
    selected = modules[0].selected_pairs(records, 3)
    assert selected == [(0, 2), (1, 3), (3, 5)]
    assert selected == modules[0].selected_pairs(dict(reversed(list(records.items()))), 3)
    with pytest.raises(ValueError):
        modules[0].selected_pairs(records, 64)


def test_archive_member_is_exact_and_bounded(tmp_path, modules):
    path = tmp_path / 'archive.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('scene/cloud_bin_0.ply', b'abc')
        archive.writestr('../outside', b'not extracted')
    with zipfile.ZipFile(path) as archive:
        assert modules[0].member_bytes(archive, 'scene/cloud_bin_0.ply') == b'abc'
        with pytest.raises(ValueError):
            modules[0].member_bytes(archive, 'scene/cloud_bin_1.ply')
        with pytest.raises(ValueError):
            modules[0].member_bytes(archive, 'scene/cloud_bin_0.ply', limit=2)


def test_case_failure_stays_in_planned_denominator(modules):
    completed = dict(status='completed', rows=[
        dict(method='spatialrust', seed=7, status='success', publisher_correct=True),
        dict(method='open3d', seed=7, status='success', publisher_correct=False),
        dict(method='spatialrust', seed=8, status='error', publisher_correct=False),
        dict(method='open3d', seed=8, status='success', publisher_correct=True),
    ])
    result = modules[1].summarize([completed, dict(status='case_error')], [7, 8])
    assert result['spatialrust'] == dict(planned=4, completed_rows=2, solver_outputs=1,
                                       publisher_correct=1, case_error_rows=2)
    assert result['open3d']['solver_outputs'] == 2
    assert result['open3d']['case_error_rows'] == 2


def test_complementarity_pairs_case_and_seed(modules):
    cases = [dict(case_id='a', status='completed', rows=[
        dict(method='spatialrust', seed=7, publisher_correct=True),
        dict(method='open3d', seed=7, publisher_correct=True),
        dict(method='spatialrust', seed=8, publisher_correct=True),
        dict(method='open3d', seed=8, publisher_correct=False)]),
        dict(case_id='b', status='completed', rows=[
            dict(method='spatialrust', seed=7, publisher_correct=False),
            dict(method='open3d', seed=7, publisher_correct=True)]), dict(status='case_error')]
    assert modules[1].complementarity(cases) == dict(both_correct=1, spatialrust_only=1,
        open3d_only=1, oracle_union_correct=3, deployable_selector_established=False)


@pytest.mark.parametrize('change', [None, 'budget', 'seeds', 'input', 'reference', 'duplicate', 'row_input'])
def test_saved_receipt_preserves_controls_and_bindings(modules, change):
    reference_bytes = b'reference'
    binding = dict(source='a' * 64, target='b' * 64)
    reference = dict(input_file_sha256=binding)
    receipt = dict(schema='spatialrust.public-global-comparison.v1', seeds=[7], ransac_iterations=10000,
                   input_file_sha256=dict(binding), reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                   rows=[dict(seed=7, method=method, status='success', report=dict(input_file_sha256=dict(binding)))
                         for method in ('spatialrust', 'open3d')])
    if change == 'budget':
        receipt['ransac_iterations'] = 5
    elif change == 'seeds':
        receipt['seeds'] = [8]
    elif change == 'input':
        receipt['input_file_sha256']['source'] = 'c' * 64
    elif change == 'reference':
        receipt['reference_sha256'] = 'c' * 64
    elif change == 'duplicate':
        receipt['rows'][1] = receipt['rows'][0]
    elif change == 'row_input':
        receipt['rows'][0]['report']['input_file_sha256']['source'] = 'c' * 64
    if change:
        with pytest.raises(ValueError):
            modules[1].validate_saved_receipt(receipt, reference, reference_bytes, [7], 10000)
    else:
        modules[1].validate_saved_receipt(receipt, reference, reference_bytes, [7], 10000)


@pytest.mark.parametrize('change', ['path', 'duplicate', 'adjacent', 'schema', 'missing'])
def test_bad_cases_rejected_before_subprocess(modules, change):
    case = dict(case_id='scene-2-to-0', scene='scene', source='source.ply', target='target.ply',
                reference_file='reference.json', pair=[0, 2])
    manifest = dict(schema='spatialrust.3dmatch-cases.v1', cases=[case])
    if change == 'path':
        case['case_id'] = '../outside'
    elif change == 'duplicate':
        manifest['cases'].append(dict(case))
    elif change == 'adjacent':
        case['pair'] = [0, 1]
    elif change == 'schema':
        manifest['schema'] = 'unknown'
    else:
        del case['reference_file']
    with pytest.raises(ValueError):
        modules[1].validate_cases(manifest)


def test_case_preparation_cli_preserves_original_ply(tmp_path):
    archives = tmp_path / 'archives'
    archives.mkdir()
    scene = 'test-scene'
    ply = tmp_path / 'cloud.ply'
    xyz = np.eye(3, dtype=np.float32)
    sr.write(str(ply), sr.PointCloud.from_xyz(xyz))
    original = ply.read_bytes()
    with zipfile.ZipFile(archives / (scene + '.zip'), 'w') as archive:
        for index in (0, 2):
            archive.writestr(f'{scene}/cloud_bin_{index}.ply', original)
        archive.writestr('../outside', b'ignored unsafe archive path')
    with zipfile.ZipFile(archives / (scene + '-evaluation.zip'), 'w') as archive:
        archive.writestr(f'{scene}-evaluation/gt.log', log())
        archive.writestr(f'{scene}-evaluation/gt.info', log(matrix=np.eye(6)))
    output = tmp_path / 'prepared'
    command = [sys.executable, str(SCRIPTS / 'prepare_3dmatch_cases.py'), '--archive-dir', str(archives),
        '--scenes', scene, '--cases-per-scene', '1', '--output-dir', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    manifest = json.loads((output / 'cases.json').read_text())
    case = manifest['cases'][0]
    assert Path(case['source']).read_bytes() == original == Path(case['target']).read_bytes()
    reference = json.loads(Path(case['reference_file']).read_text())
    assert reference['input_file_sha256']['source'] == hashlib.sha256(original).hexdigest()
    assert reference['provenance']['source_fragment_id'] == 2
    assert not (tmp_path / 'outside').exists()
    assert subprocess.run(command, capture_output=True).returncode != 0
