"""Optional three-library registration checks with an explicitly supplied PCL binary."""
import os
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip('open3d')
pytest.importorskip('scipy')
pytest.importorskip('spatialrust')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from compare_icp_failures import run,render
from pcl_icp_process import PclIcpProcess

binary=os.environ.get('SPATIALRUST_PCL_ICP_BINARY')
pytestmark=pytest.mark.skipif(not binary,reason='explicit PCL comparator binary required')


def test_protocol_and_exact_clean_alignment():
    xyz=np.random.default_rng(17).uniform(-1,1,(80,3)).astype(np.float32)
    moved=xyz+np.array([.01,-.02,.005],np.float32)
    with PclIcpProcess(binary) as pcl:
        with pytest.raises(ValueError):
            pcl.align(moved.astype(np.float64),xyz,.2,30)
        pose,updates,seconds=pcl.align(moved,xyz,.2,30)
        assert 1 <= updates <= 30 and seconds >= 0
        assert pcl.version
        np.testing.assert_allclose(moved@pose[:3,:3].T+pose[:3,3],xyz,atol=3e-5)


def test_three_library_paired_failure_study():
    with PclIcpProcess(binary) as pcl:
        rows=run(seeds=1,iterations=100,pcl=pcl)
    assert len(rows)==162
    assert all(row['status']=='success' for row in rows)
    assert {row['backend'] for row in rows}=={'spatialrust','open3d','pcl'}
    clean=[row for row in rows if row['case']=='clean' and row['initial_error_degrees']==0 and row['gate']==.3]
    assert len(clean)==3 and all(row['recovered'] for row in clean)
    assert all(1 <= row['executed_iterations'] <= 100 for row in rows if row['backend']=='pcl')
    assert '<th>PCL</th>' in render(rows)
