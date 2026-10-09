"""File alignment must preserve scalar attributes and transform normals."""
import json
import subprocess
import sys

import numpy as np
import pytest
import spatialrust as sr
from test_alignment_pipeline import EXAMPLE


def write_rich_pcd(path, xyz, normals):
    n = len(xyz)
    intensity = np.arange(n, dtype=np.float32)/8
    labels = np.arange(n, dtype=np.int32)-3
    timestamp = 1e12 + np.arange(n, dtype=np.float64)/8
    header = ('VERSION .7\nFIELDS x y z intensity label normal_x normal_y normal_z timestamp\n'
              'SIZE 4 4 4 4 4 4 4 4 8\nTYPE F F F F I F F F F\nCOUNT 1 1 1 1 1 1 1 1 1\n'
              f'WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n')
    rows = [' '.join(map(str, [*p, intensity[i], labels[i], *normals[i], timestamp[i]])) for i,p in enumerate(xyz)]
    path.write_text(header+'\n'.join(rows)+'\n')
    return intensity, labels, timestamp


def test_file_pipeline_preserves_attributes_and_rotates_normals(tmp_path):
    pa = pytest.importorskip('pyarrow')
    target = np.random.default_rng(27).uniform(-1,1,(80,3)).astype(np.float32)
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float32)
    translation = np.array([5,-2,.5], dtype=np.float32)
    source = target @ rotation.T + translation
    normals = np.tile(np.array([0,1,0], dtype=np.float32), (len(source),1))
    path = tmp_path/'source.pcd'
    intensity, labels, timestamp = write_rich_pcd(path, source, normals)
    target_path = tmp_path/'target.pcd'
    sr.write(str(target_path), sr.PointCloud.from_xyz(target))
    initial = np.eye(4, dtype=np.float32)
    initial[:3,:3] = rotation.T
    initial[:3,3] = -rotation.T @ translation
    pose = tmp_path/'pose.json'
    pose.write_text(json.dumps(initial.tolist()))
    output = tmp_path/'aligned'
    result = subprocess.run([sys.executable,str(EXAMPLE),str(path),str(target_path),'--initial-transform',str(pose),'--leaf','.01','--output-dir',str(output)], capture_output=True,text=True,timeout=60)
    assert result.returncode == 0, result.stderr
    saved = sr.read(str(output/'aligned.pcd'))
    original = sr.read(str(path))
    assert saved.field_names() == original.field_names()
    np.testing.assert_array_equal(saved.labels(), labels)
    np.testing.assert_allclose(saved.xyz(), target, atol=3e-5)
    columns = pa.array(saved)
    np.testing.assert_array_equal(columns.field('intensity').to_numpy(), intensity)
    np.testing.assert_array_equal(columns.field('timestamp').to_numpy(), timestamp)
    actual_normals = np.column_stack([columns.field('normal_'+axis).to_numpy() for axis in 'xyz'])
    report = json.loads((output/'alignment.json').read_text())
    final = np.asarray(report['transform_source_to_target'])
    expected = normals @ final[:3,:3].T
    np.testing.assert_allclose(actual_normals, expected, atol=2e-6)
    np.testing.assert_allclose(actual_normals, np.tile([1,0,0],(len(source),1)), atol=3e-5)
    # Serialized source remains intact after both coarse and fine transforms.
    np.testing.assert_array_equal(original.xyz(), source)
    assert columns.type.field('label').type == pa.int32()
    assert columns.type.field('timestamp').type == pa.float64()
