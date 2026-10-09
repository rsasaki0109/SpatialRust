"""Planned-denominator accounting, checkpoint integrity and subprocess failures."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import spatialrust as sr

import pytest

@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import study_3dmatch_refinement_cases
    return study_3dmatch_refinement_cases


def test_gains_losses_and_failures_are_paired(module):
    cases = [dict(status='completed', planned_seeds=3, evaluation=dict(rows=[
        dict(seed=7, trim_fraction=1., correct=True), dict(seed=7, trim_fraction=.8, correct=False),
        dict(seed=8, trim_fraction=1., correct=False), dict(seed=8, trim_fraction=.8, correct=True),
        dict(seed=9, trim_fraction=1., correct=True), dict(seed=9, trim_fraction=.8, correct=True)])),
        dict(status='case_error', planned_seeds=3)]
    result = module.summarize(cases, [1., .8])
    assert result['0.8'] == dict(planned=6, evaluated=3, correct=2, gained=1, lost=1, case_error_rows=3)
    assert result['1.0']['gained'] == result['1.0']['lost'] == 0


@pytest.mark.parametrize('timeout', [False, True])
def test_failed_subprocess_is_checkpointed(tmp_path, module, monkeypatch, timeout):
    case = dict(case_id='case', scene='scene', seeds=[7, 8, 9], source='source', target='target',
                reference_file='reference', comparison_file='comparison')
    def fail(command, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 1, stdout=b'', stderr=b'controlled failure')
    monkeypatch.setattr(module.subprocess, 'run', fail)
    result = module.run_case(case, tmp_path, [1., .8], 1, False)
    assert result['status'] == 'case_error' and result['planned_seeds'] == 3
    assert json.loads((tmp_path / 'case/case.json').read_bytes()) == result
    assert module.run_case(case, tmp_path, [1., .8], 1, True) == result


def test_resume_rejects_changed_artifact(tmp_path, module):
    directory = tmp_path / 'case'
    directory.mkdir()
    artifact = directory / 'artifact.json'
    artifact.write_bytes(b'original')
    checkpoint = dict(case_id='case', planned_seeds=1, status='completed',
                      artifact_sha256={'artifact.json': module.digest(artifact)})
    module.atomic_json(directory / 'case.json', checkpoint)
    artifact.write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact bytes changed'):
        module.run_case(dict(case_id='case', seeds=[7]), tmp_path, [1., .8], 1, True)


def test_real_cli_replay_and_resume(tmp_path):
    from test_redwood_reference import log
    root = Path(__file__).resolve().parents[3]
    target = np.random.default_rng(12).uniform(-1, 1, (120, 3)).astype(np.float32)
    source = target + np.array([.02, -.01, .03], np.float32)
    paths = [tmp_path / 'source.ply', tmp_path / 'target.ply']
    for path, xyz in zip(paths, [source, target]):
        sr.write(str(path), sr.PointCloud.from_xyz(xyz))
    bindings = dict(zip(['source', 'target'], [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]))
    pose = np.eye(4)
    pose[:3, 3] = [-.02, .01, -.03]
    for name, text in [('gt.log', log(matrix=pose)), ('gt.info', log(matrix=np.eye(6)))]:
        (tmp_path / name).write_text(text, encoding='utf-8')
    reference = dict(schema='spatialrust.pose-reference.v1', length_unit='m', input_file_sha256=bindings,
        transform_source_to_target=pose.tolist(), provenance=dict(kind='synthetic', description='Known generated translation',
        source_fragment_id=2, target_fragment_id=0, gt_log_sha256=hashlib.sha256((tmp_path / 'gt.log').read_bytes()).hexdigest(),
        gt_info_sha256=hashlib.sha256((tmp_path / 'gt.info').read_bytes()).hexdigest()))
    reference_path = tmp_path / 'reference.json'
    reference_path.write_text(json.dumps(reference), encoding='utf-8')
    case_id = 'synthetic-2-to-0'
    comparisons = tmp_path / 'comparisons'
    (comparisons / case_id).mkdir(parents=True)
    # Generate the saved prior in the same isolated single-thread configuration.
    code = '''import hashlib,importlib,json,sys
from pathlib import Path
import spatialrust as sr
sys.path.insert(0,sys.argv[1])
from align_global import align_global_clouds
source,target=map(sr.read,sys.argv[2:4])
_,report=align_global_clouds(source,target,[dict(leaf=.05,max_distance=.2,iterations=50),dict(leaf=None,max_distance=.05,iterations=50)],seeds=[7],ransac_iterations=2000)
bindings=json.loads(sys.argv[4]); report['input_file_sha256']=bindings
native=Path(importlib.import_module('spatialrust.spatialrust').__file__)
reference=Path(sys.argv[5])
receipt=dict(schema='spatialrust.public-global-comparison.v1',input_file_sha256=bindings,seeds=[7],native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),rows=[dict(seed=7,method='spatialrust',status='success',report=report)])
Path(sys.argv[6]).write_text(json.dumps(receipt),encoding='utf-8')
'''
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    subprocess.run([sys.executable, '-c', code, str(root / 'crates/spatialrust-py/examples'),
        *map(str, paths), json.dumps(bindings), str(reference_path), str(comparisons / case_id / 'comparison.json')],
        env=environment, check=True, capture_output=True)
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(dict(schema='spatialrust.3dmatch-cases.v1', cases=[dict(case_id=case_id,
        scene='synthetic', source=str(paths[0]), target=str(paths[1]), reference_file=str(reference_path), pair=[0, 2])])), encoding='utf-8')
    output = tmp_path / 'output'
    command = [sys.executable, str(root / 'scripts/study_3dmatch_refinement_cases.py'), '--cases', str(manifest),
        '--comparisons', str(comparisons), '--workers', '1', '--output-dir', str(output)]
    subprocess.run(command, env=environment, check=True, capture_output=True)
    study = json.loads((output / 'study.json').read_bytes())
    assert study['summary']['1.0']['correct'] == study['summary']['0.8']['correct'] == 1
    assert study['cases'][0]['evaluation']['exact_untrimmed_controls'] == 1
    original_bytes = (output / case_id / 'replay/study.json').read_bytes()
    subprocess.run(command + ['--resume'], env=environment, check=True, capture_output=True)
    assert (output / case_id / 'replay/study.json').read_bytes() == original_bytes
    assert json.loads((output / 'study.json').read_bytes()) == study
