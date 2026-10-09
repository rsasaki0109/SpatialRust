import hashlib
from pathlib import Path

import numpy as np
import pytest
import spatialrust as sr


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / 'scripts'))
    import render_saved_pose_geometry
    return render_saved_pose_geometry


def fixture(tmp_path):
    xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    offset = np.array([.4, -.2, .1], dtype=np.float32)
    case, hashes = {}, {}
    for role, points in [('source', xyz), ('target', xyz+offset)]:
        path = tmp_path/(role+'.pcd')
        sr.write(str(path), sr.PointCloud.from_xyz(points))
        case[role] = str(path)
        hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    pose = np.eye(4)
    pose[:3, 3] = offset
    row = dict(method='spatialrust', seed=7, status='success', common_support=dict(count=4, fraction=1, rmse=0),
               report=dict(input_file_sha256=hashes, transform_source_to_target=pose.tolist()))
    return case, dict(schema='spatialrust.public-global-comparison.v1', input_file_sha256=hashes, rows=[row]), xyz+offset


def test_display_pose_maps_source_into_target_frame(module, tmp_path):
    case, comparison, expected = fixture(tmp_path)
    result = module.prepare(case, comparison, [('spatialrust', 7)])
    actual = np.array(result['views'][0]['aligned_xyz'])
    assert {tuple(np.round(p, 5)) for p in actual} == {tuple(np.round(p.astype(float), 5)) for p in expected}
    assert len(result['source_xyz']) == 4 and result['point_limit_per_cloud'] == 1500


@pytest.mark.parametrize('change', ['bytes', 'duplicate', 'failed'])
def test_unbound_or_ambiguous_pose_rejected(module, tmp_path, change):
    case, comparison, _ = fixture(tmp_path)
    if change == 'bytes':
        Path(case['source']).write_bytes(b'changed source')
    elif change == 'duplicate':
        comparison['rows'].append(comparison['rows'][0])
    else:
        comparison['rows'][0]['status'] = 'error'
    with pytest.raises(ValueError):
        module.prepare(case, comparison, [('spatialrust', 7)])
