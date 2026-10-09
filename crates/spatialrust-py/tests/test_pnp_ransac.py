"""Robust calibrated pose, independent reprojection and owned result contracts."""
import gc
import sys
import threading
import numpy as np
import pytest
import spatialrust as sr


def fixture(outliers=True):
    objects=np.random.default_rng(23).uniform(-1,1,(100,3))
    angle=.2
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    translation=np.array([.3,-.2,5.])
    camera=objects@rotation.T+translation
    images=camera[:,:2]/camera[:,2,None]*[700.,710.]+[320.,240.]
    if outliers:
        images+=np.random.default_rng(24).normal(0,.2,images.shape)
        images[-20:]+=[200.,-150.]
    return objects,images,rotation,translation


def test_ransac_recovers_with_outliers_and_matches_independent_residuals():
    objects,images,expected_rotation,expected_translation=fixture()
    original_objects,original_images=objects.copy(),images.copy()
    rotation,translation,mask,residual=sr.solve_pnp_ransac(objects,images,700,710,320,240,threshold=3,seed=7)
    np.testing.assert_allclose(rotation,expected_rotation,atol=4e-4)
    np.testing.assert_allclose(translation,expected_translation,atol=.002)
    camera=objects@rotation.T+translation
    projected=camera[:,:2]/camera[:,2,None]*[700.,710.]+[320.,240.]
    independent=np.linalg.norm(projected-images,axis=1)
    np.testing.assert_allclose(residual,independent,atol=1e-9)
    np.testing.assert_array_equal(mask,residual<=3)
    assert mask.dtype==np.bool_ and residual.dtype==np.float64
    assert mask.shape==residual.shape==(100,)
    assert mask[:80].all() and not mask[80:].any()
    np.testing.assert_array_equal(objects,original_objects)
    np.testing.assert_array_equal(images,original_images)
    again=sr.solve_pnp_ransac(objects,images,700,710,320,240,threshold=3,seed=7)
    for first,second in zip((rotation,translation,mask,residual),again):
        np.testing.assert_array_equal(first,second)
    del objects,images
    gc.collect()
    assert np.isfinite(residual).all() and mask.sum()==80


def test_strided_inputs_and_positional_camera_are_supported():
    objects,images,expected_rotation,expected_translation=fixture(False)
    rotation,translation,mask,residual=sr.solve_pnp_ransac(objects[::-1],images[::-1],700,710,320,240,640,480)
    np.testing.assert_allclose(rotation,expected_rotation,atol=1e-6)
    np.testing.assert_allclose(translation,expected_translation,atol=1e-6)
    assert mask.all() and residual.max()<1e-6


@pytest.mark.parametrize('settings',[
    {'threshold':0},{'threshold':float('nan')},{'threshold':float('inf')},
    {'confidence':0},{'confidence':1},{'confidence':float('inf')},{'max_iterations':0},
])
def test_invalid_robust_settings_rejected(settings):
    objects,images,_,_=fixture()
    with pytest.raises(ValueError):
        sr.solve_pnp_ransac(objects,images,700,710,320,240,**settings)


@pytest.mark.parametrize('kind',['short','shape','mismatch','nonfinite_object','nonfinite_image','bad_focal','bad_size'])
def test_invalid_camera_or_correspondences_rejected(kind):
    objects,images,_,_=fixture()
    fx=700;width=640
    if kind=='short':objects,images=objects[:5],images[:5]
    elif kind=='shape':objects=objects[:,:2]
    elif kind=='mismatch':images=images[:-1]
    elif kind=='nonfinite_object':objects[0,0]=np.nan
    elif kind=='nonfinite_image':images[0,0]=np.inf
    elif kind=='bad_focal':fx=0
    elif kind=='bad_size':width=0
    with pytest.raises(ValueError):sr.solve_pnp_ransac(objects,images,fx,710,320,240,width=width)


def test_ransac_native_work_releases_gil():
    objects,_,_,_=fixture()
    # Unrelated pixels prevent early all-inlier termination of the long native call.
    images=np.random.default_rng(801).uniform([0,0],[640,480],(100,2))
    ready,gate,tick=threading.Event(),threading.Event(),threading.Event()
    def worker():
        ready.set();gate.wait();tick.set()
    thread=threading.Thread(target=worker)
    previous=sys.getswitchinterval()
    try:
        sys.setswitchinterval(10)
        thread.start();assert ready.wait(2);gate.set()
        try:
            sr.solve_pnp_ransac(objects,images,700,710,320,240,max_iterations=2000)
        except ValueError:
            pass # No accepted model is valid for these deliberately unrelated pairs.
        assert tick.is_set()
    finally:
        sys.setswitchinterval(previous);gate.set();thread.join(timeout=2)
    assert not thread.is_alive()
