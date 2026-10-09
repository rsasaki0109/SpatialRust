"""Independent known-angle errors, file binding and post-fit reference evaluation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / 'crates/spatialrust-py/examples'
SCRIPT = ROOT / 'scripts/evaluate_pose_reference.py'
spec = importlib.util.spec_from_file_location('pose_reference', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def reference():
    return dict(schema='spatialrust.pose-reference.v1', length_unit='m',
                input_file_sha256=dict(source='a' * 64, target='b' * 64),
                provenance=dict(kind='synthetic', description='Known analytic rigid transform'),
                transform_source_to_target=np.eye(4).tolist())


def report(ref):
    return dict(schema_version='spatialrust.python-alignment.v1',
                input_file_sha256=copy.deepcopy(ref['input_file_sha256']),
                transform_source_to_target=np.eye(4).tolist())


@pytest.mark.parametrize('degrees', [0, 1e-7, 45, 90, 179.9999, 180])
def test_known_rotation_translation_errors(degrees):
    ref = reference()
    estimate = report(ref)
    angle = np.deg2rad(degrees)
    matrix = np.array([[np.cos(angle), -np.sin(angle), 0, 3],
                       [np.sin(angle), np.cos(angle), 0, 4],
                       [0, 0, 1, 0], [0, 0, 0, 1]])
    estimate['transform_source_to_target'] = matrix.tolist()
    result = module.evaluate(estimate, ref)
    assert result['rotation_error_degrees'] == pytest.approx(degrees, abs=1e-8)
    assert result['translation_error'] == 5
    # A shared target-frame rigid transform preserves both errors.
    frame = np.array([[0, 0, 1, 10], [1, 0, 0, 20], [0, 1, 0, 30], [0, 0, 0, 1]])
    ref['transform_source_to_target'] = frame.tolist()
    estimate['transform_source_to_target'] = (frame @ matrix).tolist()
    changed = module.evaluate(estimate, ref)
    assert changed['rotation_error_degrees'] == pytest.approx(degrees, abs=1e-8)
    assert changed['translation_error'] == 5


@pytest.mark.parametrize('bad', ['swapped', 'missing_hash', 'reflection', 'scale', 'nan', 'unit', 'direction', 'provenance'])
def test_invalid_reference_or_pair_rejected(bad):
    ref = reference()
    estimate = report(ref)
    if bad == 'swapped':
        estimate['input_file_sha256'] = dict(source='b' * 64, target='a' * 64)
    elif bad == 'missing_hash':
        del estimate['input_file_sha256']
    elif bad in ('reflection', 'scale', 'nan'):
        estimate['transform_source_to_target'][0][0] = {'reflection': -1, 'scale': 2, 'nan': float('nan')}[bad]
    elif bad == 'unit':
        ref['length_unit'] = 'unknown'
    elif bad == 'direction':
        ref['transform_target_to_source'] = ref.pop('transform_source_to_target')
    else:
        ref['provenance'] = dict(kind='dataset_reference', description='No reference source')
    with pytest.raises(ValueError):
        module.evaluate(estimate, ref)


def test_changed_input_during_read_rejected(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import align_point_clouds as pipeline
    path = tmp_path / 'input.pcd'
    sr.write(str(path), sr.PointCloud.from_xyz(np.eye(3, dtype=np.float32)))
    original = sr.read
    def changing_read(name):
        cloud = original(name)
        path.write_bytes(path.read_bytes() + b'\n')
        return cloud
    monkeypatch.setattr(sr, 'read', changing_read)
    with pytest.raises(ValueError, match='changed'):
        pipeline.read_bound_cloud(path)


@pytest.mark.parametrize('workflow', ['ordinary', 'multiscale', 'global', 'candidates'])
def test_file_workflow_and_accuracy_cli(tmp_path, monkeypatch, workflow):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    import align_point_clouds
    import align_multiscale
    import align_global
    import align_pose_candidates
    target = np.random.default_rng(12).uniform(-1, 1, (120, 3)).astype(np.float32)
    source = target + np.array([.02, -.01, .03], np.float32)
    paths = [tmp_path / 'source.pcd', tmp_path / 'target.pcd']
    for path, xyz in zip(paths, [source, target]):
        sr.write(str(path), sr.PointCloud.from_xyz(xyz))
    schedule = [dict(leaf=None, max_distance=.1, iterations=30)]
    if workflow == 'ordinary':
        _, diagnostic = align_point_clouds.align_files(*paths, max_distance=.1)
    elif workflow == 'multiscale':
        _, diagnostic = align_multiscale.align_multiscale_files(*paths, schedule)
    elif workflow == 'global':
        _, diagnostic = align_global.align_global_files(*paths, schedule, seeds=[7], ransac_iterations=1000)
    else:
        _, diagnostic = align_pose_candidates.evaluate_candidates(*paths, [np.eye(4).tolist()])
    hashes = dict(zip(['source', 'target'], [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]))
    assert diagnostic['input_file_sha256'] == hashes
    ref = reference()
    ref['input_file_sha256'] = hashes
    pose = np.eye(4)
    pose[:3, 3] = [-.02, .01, -.03]
    ref['transform_source_to_target'] = pose.tolist()
    reference_path, report_path = tmp_path / 'reference.json', tmp_path / 'alignment.json'
    reference_path.write_text(json.dumps(ref))
    report_path.write_text(json.dumps(diagnostic))
    output = tmp_path / 'accuracy'
    command = [sys.executable, str(SCRIPT), '--reference', str(reference_path),
               '--reports', str(report_path), '--output-dir', str(output)]
    subprocess.run(command, check=True, capture_output=True)
    result = json.loads((output / 'accuracy.json').read_text())
    assert result['evaluations'][0]['translation_error'] < 1e-5
    assert result['evaluations'][0]['rotation_error_degrees'] < .001
    assert result['reference_sha256'] == hashlib.sha256(reference_path.read_bytes()).hexdigest()
    assert 'Declared provenance' in (output / 'report.html').read_text()
    assert subprocess.run(command, capture_output=True).returncode != 0
