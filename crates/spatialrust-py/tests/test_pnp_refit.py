"""The installed robust-PnP API must survive an optional invalid DLT refit."""
import gc

import numpy as np
import pytest
import spatialrust as sr


def synthetic_near_plane():
    state=5
    mask=(1<<64)-1
    def uniform():
        nonlocal state
        state^=(state<<13)&mask
        state^=state>>7
        state^=(state<<17)&mask
        return (state>>11)*(1./(1<<53))
    rotation=np.array([[.9983758533596424,-.03588494159791196,-.04424846206277005],
                       [.034957430317503015,.999156210831159,-.021560250998475538],
                       [.044984814037660234,.019978421460035138,.9987878799735018]])
    translation=[.01,-.004,.007]
    objects=[];images=[]
    for index in range(90):
        x=-.95+1.9*uniform();y=-.7+1.4*uniform();z=2.+.2*x+.1*y+.002*(uniform()-.5)
        objects.append([x,y,z])
        moved=[float(rotation[row,0])*x+float(rotation[row,1])*y+float(rotation[row,2])*z+translation[row] for row in range(3)]
        u=525*(moved[0]/moved[2])+319.5+.5*(uniform()-.5)
        v=525*(moved[1]/moved[2])+239.5+.5*(uniform()-.5)
        if index%3==0:u=40+560*uniform();v=40+400*uniform()
        images.append([u,v])
    return np.array(objects),np.array(images)


@pytest.mark.parametrize('strided',[False,True])
def test_optional_refit_retains_a_valid_owned_full_row_estimate(strided):
    objects,images=synthetic_near_plane()
    if strided:
        object_storage=np.zeros((180,6));object_storage[::2,::2]=objects;objects=object_storage[::2,::2]
        image_storage=np.zeros((180,4));image_storage[::2,::2]=images;images=image_storage[::2,::2]
    before_objects=objects.copy();before_images=images.copy()
    output=sr.solve_pnp_ransac(objects,images,525.,525.,319.5,239.5,width=640,height=480,
                             threshold=2.,confidence=.99,max_iterations=2000,seed=0)
    rotation,translation,inliers,residuals=output
    assert rotation.shape==(3,3) and translation.shape==(3,)
    assert inliers.shape==residuals.shape==(90,) and np.count_nonzero(inliers)>=58
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-10)
    assert np.linalg.det(rotation)>0.999999999
    moved=objects@rotation.T+translation
    assert np.all(moved[:,2]>0)
    projected=np.column_stack((525*(moved[:,0]/moved[:,2])+319.5,525*(moved[:,1]/moved[:,2])+239.5))
    expected=np.linalg.norm(projected-images,axis=1)
    np.testing.assert_allclose(residuals,expected,rtol=0,atol=2e-10)
    np.testing.assert_array_equal(inliers,expected<=2.)
    repeated=sr.solve_pnp_ransac(objects,images,525.,525.,319.5,239.5,width=640,height=480,
                               threshold=2.,confidence=.99,max_iterations=2000,seed=0)
    for first,second in zip(output,repeated):np.testing.assert_array_equal(first,second)
    np.testing.assert_array_equal(objects,before_objects)
    np.testing.assert_array_equal(images,before_images)
    snapshots=[array.copy() for array in output]
    assert all(not np.shares_memory(array,objects) and not np.shares_memory(array,images) for array in output)
    objects[:]=np.nan;images[:]=np.nan
    del objects,images
    gc.collect()
    for actual,expected_output in zip(output,snapshots):np.testing.assert_array_equal(actual,expected_output)
