"""Owned support indexes preserve results, lifetime and read-only concurrency."""
from concurrent.futures import ThreadPoolExecutor
import gc
import sys
import threading

import numpy as np
import pytest
import spatialrust as sr


def test_index_matches_function_after_target_is_released():
    xyz = np.random.default_rng(17).uniform(-1, 1, (313, 3)).astype(np.float32)
    target = sr.PointCloud.from_xyz(xyz)
    queries = [sr.PointCloud.from_xyz(xyz[:117]),
               sr.PointCloud.from_xyz(xyz + np.float32(.02)),
               sr.PointCloud.from_xyz(xyz + np.float32(100))]
    expected = [sr.distance_gated_support(q, target, .05) for q in queries]
    index = sr.DistanceSupportIndex(target)
    del target
    gc.collect()
    xyz[:] = 1000
    assert [index.support(q, .05) for q in queries] == expected
    assert expected[-1] == (0, 0., None)


def test_same_index_concurrent_queries_and_exact_boundary():
    target = sr.PointCloud.from_xyz(np.array([[0, 0, 0], [0, 0, 0]], np.float32))
    query = sr.PointCloud.from_xyz(np.array([[.25, 0, 0], [2, 0, 0], [0, 0, 0]], np.float32))
    index = sr.DistanceSupportIndex(target)
    expected = sr.distance_gated_support(query, target, .25)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: index.support(query, .25), range(32))) == [expected] * 32
    np.testing.assert_array_equal(target.xyz(), [[0, 0, 0], [0, 0, 0]])
    assert expected == (2, 2 / 3, np.sqrt(.25**2 / 2))


@pytest.mark.parametrize('xyz', [np.empty((0, 3), np.float32),
    np.array([[np.nan, 0, 0]], np.float32), np.array([[0, np.inf, 0]], np.float32)])
def test_invalid_index_reference_or_query(xyz):
    invalid = sr.PointCloud.from_xyz(xyz)
    with pytest.raises(ValueError):
        sr.DistanceSupportIndex(invalid)
    index = sr.DistanceSupportIndex(sr.PointCloud.from_xyz(np.zeros((1, 3), np.float32)))
    with pytest.raises(ValueError):
        index.support(invalid, .1)


@pytest.mark.parametrize('gate', [0, -1, float('nan'), float('inf'), 1e30, 1e-30])
def test_invalid_index_gate(gate):
    cloud = sr.PointCloud.from_xyz(np.zeros((1, 3), np.float32))
    with pytest.raises(ValueError):
        sr.DistanceSupportIndex(cloud).support(cloud, gate)


@pytest.mark.parametrize('operation', ['construct', 'query'])
def test_index_operations_release_gil(operation):
    cloud = sr.PointCloud.from_xyz(np.random.default_rng(9).uniform(-1, 1, (120000, 3)).astype(np.float32))
    index = sr.DistanceSupportIndex(cloud)
    ready, start = threading.Event(), threading.Event()
    progress = []
    def observer():
        ready.set()
        start.wait()
        progress.append(True)
    worker = threading.Thread(target=observer)
    worker.start()
    assert ready.wait(timeout=5)
    previous = sys.getswitchinterval()
    try:
        sys.setswitchinterval(60)
        start.set()
        result = sr.DistanceSupportIndex(cloud) if operation == 'construct' else index.support(cloud, .1)
        progressed = bool(progress)
    finally:
        sys.setswitchinterval(previous)
        worker.join(timeout=5)
    assert progressed
    if operation == 'query':
        assert result == (len(cloud), 1., 0.)
