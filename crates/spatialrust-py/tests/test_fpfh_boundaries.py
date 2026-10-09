"""Global registration validation, deterministic seeds and bounded normal scratch."""
import numpy as np
import pytest
import spatialrust as sr
import sys
import threading


def cloud():
    xyz=np.random.default_rng(47).normal(size=(80,3)).astype(np.float32)
    return sr.PointCloud.from_xyz(xyz)


@pytest.mark.parametrize('name',['feature_radius','max_correspondence_distance'])
@pytest.mark.parametrize('value',[0,-1,float('nan'),float('inf'),1e30,1e-30])
def test_bad_distance_rejected(name,value):
    points=cloud()
    with pytest.raises(ValueError,match='finite nonzero f32 square'):
        sr.register_fpfh_ransac(points,points,**{name:value})


@pytest.mark.parametrize('settings',[{'ransac_iterations':0},{'k_neighbors':0},{'k_neighbors':2}])
def test_bad_budget_and_neighborhood_rejected(settings):
    points=cloud()
    with pytest.raises(ValueError):
        sr.register_fpfh_ransac(points,points,**settings)


@pytest.mark.parametrize('bad',[np.full((2,3),0,np.float32),np.full((4,3),np.nan,np.float32),np.full((4,3),np.inf,np.float32)])
def test_invalid_xyz_rejected_before_normal_estimation(bad):
    points=cloud()
    invalid=sr.PointCloud.from_xyz(bad)
    for source,target in [(points,invalid),(invalid,points)]:
        with pytest.raises(ValueError,match='finite XYZ'):
            sr.register_fpfh_ransac(source,target)


def test_seed_reproducibility_and_large_k_is_bounded():
    points=cloud()
    settings=dict(feature_radius=2,max_correspondence_distance=.1,ransac_iterations=100,seed=7)
    first=sr.register_fpfh_ransac(points,points,k_neighbors=10**12,**settings)
    second=sr.register_fpfh_ransac(points,points,k_neighbors=len(points),**settings)
    np.testing.assert_array_equal(first.transform(),second.transform())
    assert first.fitness == second.fitness
    assert first.converged
    np.testing.assert_allclose(first.transform(),np.eye(4),atol=1e-5)


def test_global_registration_releases_gil():
    points=cloud()
    ready,gate,tick=threading.Event(),threading.Event(),threading.Event()
    def python_worker():
        ready.set()
        gate.wait()
        tick.set()
    thread=threading.Thread(target=python_worker)
    previous=sys.getswitchinterval()
    try:
        # Prevent ordinary bytecode switching from masquerading as native GIL release.
        sys.setswitchinterval(10)
        thread.start()
        assert ready.wait(2)
        gate.set()
        sr.register_fpfh_ransac(points,points,feature_radius=2,
            max_correspondence_distance=.1,ransac_iterations=100000)
        assert tick.is_set(), 'Python worker could not execute during native registration'
    finally:
        sys.setswitchinterval(previous)
        gate.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
