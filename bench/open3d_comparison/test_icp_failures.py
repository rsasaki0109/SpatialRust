"""Optional direct-comparison checks, requiring the Open3D comparison environment."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('open3d')
pytest.importorskip('scipy')
pytest.importorskip('spatialrust')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import compare_icp_failures as study


def test_assessment_uses_original_denominator_and_known_pose():
    xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[10,10,10]],np.float32)
    row=study.assess(xyz,xyz[:3],np.eye(4),np.eye(4))
    assert row['forward_support'] == .75
    assert row['reverse_support'] == 1
    assert row['gated_rmse_metres'] == 0
    assert row['recovered']
    pose=np.eye(4);pose[:3,3]=[1,0,0]
    assert not study.assess(xyz,xyz[:3],pose,np.eye(4))['recovered']


def test_native_and_open3d_complete_paired_fixed_budget_study():
    rows=study.run(seeds=1,iterations=100)
    assert len(rows) == 108
    pairs={}
    for row in rows:
        assert row['status'] == 'success'
        assert 0 <= row['forward_support'] <= 1
        assert 0 <= row['reverse_support'] <= 1
        pairs.setdefault((row['case'],row['seed'],row['initial_error_degrees'],row['gate']),{})[row['backend']]=row
    assert all(set(pair)=={'spatialrust','open3d'} for pair in pairs.values())
    clean=pairs[('clean',0,0,.3)]
    for row in clean.values():
        assert row['recovered']
        assert row['source_points'] == 400
        assert row['forward_support'] == 1
    np.testing.assert_allclose(clean['spatialrust']['pose'],clean['open3d']['pose'],atol=3e-5)
    assert all(row['executed_iterations']==100 for row in rows if row['backend']=='spatialrust')
    assert 'SpatialRust' in study.render(rows)


def test_cli_rejects_existing_directory_before_study(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['compare','--output-dir',str(tmp_path)])
    monkeypatch.setattr(study,'run',lambda *args:pytest.fail('work before output validation'))
    with pytest.raises(SystemExit):
        study.main()
