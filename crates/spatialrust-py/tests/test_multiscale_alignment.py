"""Schedule validation, frame composition and full-source file contracts."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
import spatialrust as sr

from test_alignment_pipeline import fixture_files

EXAMPLES = Path(__file__).resolve().parents[1]/'examples'


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location('multiscale_example',EXAMPLES/'align_multiscale.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def schedule():
    return [dict(leaf=.05,max_distance=.1,iterations=50),dict(leaf=None,max_distance=.1,iterations=50)]


def test_two_stage_schedule_matches_existing_pipeline(module,tmp_path):
    from align_point_clouds import align_files
    paths,_,_ = fixture_files(tmp_path)
    ordinary,report = align_files(*paths)
    scaled,diagnostic = module.align_multiscale_files(*paths,schedule())
    np.testing.assert_array_equal(ordinary.xyz(),scaled.xyz())
    for key in ('transform_source_to_target','aligned_support','aligned_reverse_support','converged'):
        assert diagnostic[key] == report[key]
    assert diagnostic['total_iterations'] == sum(s['iterations'] for s in diagnostic['stages'])
    from render_alignment_report import render_report
    assert render_report(diagnostic).count('class="trace-chart"') == 8


@pytest.mark.parametrize('value',[
    [], 'bad', [dict(leaf=.1,max_distance=.2,iterations=10)],
    [dict(leaf=None,max_distance=.1,iterations=0)],
    [dict(leaf=None,max_distance=float('nan'),iterations=10)],
    [dict(leaf=None,max_distance=.1,iterations=10,trim_fraction=0)],
    [dict(leaf=None,max_distance=.1,iterations=10,convergence={'rotation_epsilon':-1})],
    [dict(leaf=None,max_distance=.1,iterations=10,unknown=True)],
    [dict(leaf=None,max_distance=.2,iterations=10),dict(leaf=None,max_distance=.1,iterations=10)],
    [dict(leaf=.1,max_distance=.1,iterations=10),dict(leaf=None,max_distance=.2,iterations=10)],
    [dict(leaf=.1,max_distance=.3,iterations=10),dict(leaf=.2,max_distance=.2,iterations=10),dict(leaf=None,max_distance=.1,iterations=10)],
])
def test_invalid_schedule_rejected_before_io(module,value):
    with patch.object(sr,'read',side_effect=AssertionError('IO before validation')):
        with pytest.raises(ValueError):
            module.align_multiscale_files('missing','missing',value)


def test_three_stage_composition_keeps_original_source_and_reuses_target_voxels(module):
    xyz = np.random.default_rng(23).uniform(-1,1,(300,3)).astype(np.float32)
    source = sr.PointCloud.from_xyz(xyz+[np.float32(.02),np.float32(-.01),np.float32(.005)])
    target = sr.PointCloud.from_xyz(xyz)
    settings = [dict(leaf=.05,max_distance=.15,iterations=20,trim_fraction=.8),
                dict(leaf=.05,max_distance=.1,iterations=20),dict(leaf=None,max_distance=.05,iterations=20)]
    with patch.object(sr,'voxel_downsample',wraps=sr.voxel_downsample) as voxel:
        aligned,report = module.align_multiscale_clouds(source,target,settings)
    assert voxel.call_count == 3
    assert len(aligned) == len(source) and report['aligned_support']['query_points'] == len(source)
    composed = np.eye(4,dtype=np.float32)
    for stage in report['stages']:
        composed = (np.asarray(stage['correction_in_target_frame'],np.float64) @ composed.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(composed,np.asarray(stage['transform_source_to_target'],np.float32))
    np.testing.assert_array_equal(aligned.xyz(),sr.apply_transform(source,composed).xyz())
    np.testing.assert_array_equal(source.xyz(),xyz+[np.float32(.02),np.float32(-.01),np.float32(.005)])


def test_cli_roundtrip_and_exclusive_output(module,tmp_path):
    paths,_,_ = fixture_files(tmp_path)
    config = tmp_path/'schedule.json'
    config.write_text(json.dumps(schedule()))
    output = tmp_path/'aligned'
    command = [sys.executable,str(EXAMPLES/'align_multiscale.py'),*map(str,paths),
               '--schedule',str(config),'--output-dir',str(output),'--html-report']
    result = subprocess.run(command,capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stderr
    assert len(sr.read(str(output/'aligned.pcd'))) == 120
    original = (output/'aligned.pcd').read_bytes()
    assert subprocess.run(command,capture_output=True,timeout=60).returncode != 0
    assert (output/'aligned.pcd').read_bytes() == original


def test_prior_and_attributes_survive_all_stages(module,tmp_path):
    import pyarrow as pa
    from test_alignment_attributes import write_rich_pcd
    xyz = np.random.default_rng(41).uniform(-1,1,(100,3)).astype(np.float32)
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32)
    translation = np.array([5,-2,.5],np.float32)
    source_xyz = xyz @ rotation.T + translation
    normals = np.tile([0,1,0],(len(xyz),1)).astype(np.float32)
    path = tmp_path/'rich.pcd'
    intensity,labels,timestamps = write_rich_pcd(path,source_xyz,normals)
    source = sr.read(str(path))
    prior = np.eye(4,dtype=np.float32)
    prior[:3,:3] = rotation.T
    prior[:3,3] = -rotation.T @ translation
    original_prior = prior.copy()
    aligned,report = module.align_multiscale_clouds(source,sr.PointCloud.from_xyz(xyz),
                                                   schedule(),initial_transform=prior)
    np.testing.assert_allclose(aligned.xyz(),xyz,atol=3e-5)
    np.testing.assert_array_equal(prior,original_prior)
    np.testing.assert_array_equal(source.xyz(),source_xyz)
    assert aligned.field_names() == source.field_names()
    columns = pa.array(aligned)
    for name,expected in [('intensity',intensity),('label',labels),('timestamp',timestamps)]:
        np.testing.assert_array_equal(columns.field(name).to_numpy(),expected)
    actual = np.column_stack([columns.field('normal_'+axis).to_numpy() for axis in 'xyz'])
    final = np.asarray(report['transform_source_to_target'])
    np.testing.assert_allclose(actual,normals @ final[:3,:3].T,atol=2e-6)
    assert report['initial_transform_supplied']


def test_failed_write_removes_only_reserved_output(module,tmp_path,monkeypatch):
    paths,_,_ = fixture_files(tmp_path)
    config = tmp_path/'schedule.json'
    config.write_text(json.dumps(schedule()))
    output = tmp_path/'new-output'
    sentinel = tmp_path/'existing.txt'
    sentinel.write_text('keep')
    monkeypatch.setattr(sys,'argv',['align_multiscale',*map(str,paths),
        '--schedule',str(config),'--output-dir',str(output)])
    def broken_write(path,cloud):
        Path(path).write_bytes(b'partial')
        raise OSError('disk write failed')
    monkeypatch.setattr(sr,'write',broken_write)
    with pytest.raises(OSError,match='disk write failed'):
        module.main()
    assert not output.exists()
    assert sentinel.read_text() == 'keep'
