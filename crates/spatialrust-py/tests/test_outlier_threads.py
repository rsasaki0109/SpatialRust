"""Outlier filters release Python scheduling and keep shared inputs intact."""
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

import numpy as np
import pytest
import spatialrust as sr


def statistical(cloud):
    return sr.statistical_outlier_removal(cloud, k_neighbors=16, std_mul=1.0)


def radius(cloud):
    return sr.radius_outlier_removal(cloud, radius=.06, min_neighbors=3)


@pytest.mark.parametrize("filter_cloud", [statistical, radius], ids=["statistical", "radius"])
def test_filter_releases_gil_and_concurrent_shared_inputs_match(filter_cloud):
    xyz = np.random.default_rng(73).uniform(-1, 1, (12000, 3)).astype(np.float32)
    cloud = sr.PointCloud.from_xyz(xyz)
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
        result = filter_cloud(cloud)
        progressed = bool(progress)
    finally:
        sys.setswitchinterval(previous)
        worker.join(timeout=5)
    assert progressed, "Python observer must run during native filtering"
    expected = result.xyz()
    assert 0 < len(expected) < len(xyz)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(filter_cloud, cloud) for _ in range(2)]
        for future in futures:
            np.testing.assert_array_equal(future.result(timeout=30).xyz(), expected)
    np.testing.assert_array_equal(cloud.xyz(), xyz)


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("filter_cloud", [statistical, radius], ids=["statistical", "radius"])
def test_nonfinite_xyz_is_rejected_and_input_remains_unchanged(filter_cloud, coordinate):
    xyz = np.array([[0, 0, 0], [1, coordinate, 0]], dtype=np.float32)
    cloud = sr.PointCloud.from_xyz(xyz)
    with pytest.raises(ValueError, match="finite XYZ"):
        filter_cloud(cloud)
    np.testing.assert_array_equal(cloud.xyz(), xyz)


def test_radius_impossible_neighbor_count_still_rejects_nonfinite_xyz():
    cloud = sr.PointCloud.from_xyz(np.array([[0, np.nan, 0]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite XYZ"):
        sr.radius_outlier_removal(cloud, radius=.06, min_neighbors=2**63)
