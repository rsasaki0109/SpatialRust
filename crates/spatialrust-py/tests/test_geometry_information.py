"""Analytical ranks, independent Jacobians and geometry invariance."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec=importlib.util.spec_from_file_location('geometry_example',EXAMPLES/'analyze_geometry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def plane():
    x,y=np.meshgrid(np.linspace(-1,1,7),np.linspace(-1,1,7))
    return np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))


def test_models_distinguish_plane_line_and_coincident_geometry(module):
    planar=plane();normals=np.tile([0,0,1.],(len(planar),1))
    paired=module.analyze_geometry(planar)
    surface=module.analyze_geometry(planar,normals)
    assert paired['rank']==6
    assert surface['rank']==3 and len(surface['weak_directions'])==3
    np.testing.assert_allclose(surface['relative_eigenvalues'],[0,0,0,.5,.5,1],atol=1e-12)
    line=np.column_stack((np.arange(20),np.zeros(20),np.zeros(20)))
    assert module.analyze_geometry(line)['rank']==5
    coincident=module.analyze_geometry(np.zeros((20,3)))
    assert coincident['rank']==3 and coincident['normalization_radius_metres']==1


@pytest.mark.parametrize('surface',[False,True])
def test_information_matches_independent_finite_difference_jacobian(module,surface):
    xyz=np.random.default_rng(71).normal(size=(200,3))
    normals=np.random.default_rng(72).normal(size=xyz.shape)
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    diagnostic=module.analyze_geometry(xyz,normals if surface else None)
    points=(xyz-np.asarray(diagnostic['centroid_xyz_metres']))/diagnostic['normalization_radius_metres']
    columns=[]
    step=1e-5
    for axis in range(6):
        if axis<3:
            vector=np.eye(3)[axis]
            cross=np.cross(vector,points)
            # Rodrigues' formula for +/- angle; the even terms cancel.
            plus=points*np.cos(step)+cross*np.sin(step)+(points@vector)[:,None]*vector*(1-np.cos(step))
            minus=points*np.cos(step)-cross*np.sin(step)+(points@vector)[:,None]*vector*(1-np.cos(step))
        else:
            delta=np.eye(3)[axis-3]*step
            plus,minus=points+delta,points-delta
        derivative=(plus-minus)/(2*step)
        columns.append(np.sum(derivative*normals,axis=1) if surface else derivative)
    jacobian=np.stack(columns,axis=-1)
    expected=jacobian.T@jacobian/len(points) if surface else np.einsum('nki,nkj->ij',jacobian,jacobian)/len(points)
    np.testing.assert_allclose(diagnostic['information_matrix_dimensionless'],expected,rtol=1e-8,atol=1e-10)
    assert diagnostic['rank']==6


def test_scale_origin_rotation_and_normal_sign_invariance(module):
    xyz=np.array([[x,y,z] for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    baseline=module.analyze_geometry(xyz)
    for moved in (xyz*1e-4,xyz*1e4,xyz+[1e9,-1e9,1e9],xyz[:,[2,0,1]]):
        np.testing.assert_allclose(module.analyze_geometry(moved)['relative_eigenvalues'],baseline['relative_eigenvalues'],atol=1e-12)
    normals=xyz/np.sqrt(3)
    positive=module.analyze_geometry(xyz,normals)
    negative=module.analyze_geometry(xyz,-normals)
    np.testing.assert_array_equal(positive['information_matrix_dimensionless'],negative['information_matrix_dimensionless'])


def test_symmetric_ring_can_have_full_local_information(module):
    angles=np.arange(180)*2*np.pi/180
    xyz=np.column_stack((np.cos(angles),np.sin(angles),np.zeros(180)))
    report=module.analyze_geometry(xyz)
    assert report['rank']==6
    rendered=module.render_geometry_section(report,'<sensor>')
    assert '&lt;sensor&gt;' in rendered
    assert 'multiple global poses' in rendered
    assert 'statistical confidence' in module.__doc__


@pytest.mark.parametrize('xyz',[np.zeros((2,3)),np.full((4,3),np.nan),np.full((4,3),np.inf),np.zeros((4,2))])
def test_invalid_geometry_rejected(module,xyz):
    with pytest.raises(ValueError):module.analyze_geometry(xyz)


@pytest.mark.parametrize('normals',[np.zeros((49,3)),np.full((49,3),np.nan),np.tile([0,0,2.],(49,1)),np.ones((48,3))])
def test_invalid_surface_normals_rejected(module,normals):
    with pytest.raises(ValueError):module.analyze_geometry(plane(),normals)


@pytest.mark.parametrize('threshold',[0,1,-1,float('nan'),True])
def test_invalid_rank_threshold_rejected(module,threshold):
    with pytest.raises(ValueError):module.analyze_geometry(plane(),relative_threshold=threshold)


def test_overflow_and_malformed_report_rejected(module):
    xyz=np.array([[1e308,0,0],[-1e308,0,0],[0,0,0]])
    with pytest.raises(ValueError,match='overflow'):module.analyze_geometry(xyz)
    report=module.analyze_geometry(plane())
    report['rank']=3
    with pytest.raises(ValueError,match='rank'):module.render_geometry_section(report)


def test_cli_failed_second_output_rolls_back_reserved_json(module,tmp_path,monkeypatch):
    import spatialrust as sr
    input_path=tmp_path/'plane.pcd';sr.write(str(input_path),sr.PointCloud.from_xyz(plane().astype(np.float32)))
    output=tmp_path/'information.json'
    monkeypatch.setattr(sys,'argv',['analyze',str(input_path),'--output',str(output),'--html',str(tmp_path/'missing'/'report.html')])
    with pytest.raises(FileNotFoundError):module.main()
    assert not output.exists() and input_path.exists()
