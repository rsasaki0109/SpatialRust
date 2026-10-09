"""Public voxel boundary, native scheduling and shared-input contracts."""
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

import numpy as np
import pytest
import spatialrust as sr


@pytest.mark.parametrize('leaf', [0, -1, float('nan'), float('inf')])
def test_invalid_leaf_is_named_value_error(leaf):
    cloud = sr.PointCloud.from_xyz(np.zeros((3,3), dtype=np.float32))
    with pytest.raises(ValueError, match='leaf_size'):
        sr.voxel_downsample(cloud, leaf, 'cpu')


def test_nonfinite_coordinates_rejected_without_damaging_valid_inputs():
    cloud = sr.PointCloud.from_xyz(np.array([[0,0,0],[1,np.nan,0]], dtype=np.float32))
    with pytest.raises(ValueError, match='finite XYZ'):
        sr.voxel_downsample(cloud, .1, 'cpu')
    valid = sr.PointCloud.from_xyz(np.array([[0,0,0],[.02,0,0],[1,0,0]], dtype=np.float32))
    assert len(sr.voxel_downsample(valid, .1, 'cpu')) == 2
    assert len(valid) == 3


def test_empty_cloud_remains_valid_empty_result():
    cloud = sr.PointCloud.from_xyz(np.empty((0,3), dtype=np.float32))
    assert len(sr.voxel_downsample(cloud, .1, 'cpu')) == 0


def test_cpu_voxel_releases_gil_and_concurrent_results_match():
    cloud = sr.PointCloud.from_xyz(np.random.default_rng(21).uniform(-1,1,(120000,3)).astype(np.float32))
    ready, start = threading.Event(), threading.Event()
    progress = []
    def observer():
        ready.set()
        start.wait()
        progress.append(True)
    worker = threading.Thread(target=observer)
    worker.start()
    assert ready.wait(5)
    previous = sys.getswitchinterval()
    try:
        sys.setswitchinterval(60)
        start.set()
        result = sr.voxel_downsample(cloud, .05, 'cpu')
        progressed = bool(progress)
    finally:
        sys.setswitchinterval(previous)
        worker.join(timeout=5)
    assert progressed
    expected = result.xyz()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(sr.voxel_downsample, cloud, .05, 'cpu') for _ in range(2)]
        for future in futures:
            np.testing.assert_array_equal(future.result(timeout=30).xyz(), expected)
