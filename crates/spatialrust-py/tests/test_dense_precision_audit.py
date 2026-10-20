"""Independent geometry, IEEE storage and fixed synthetic schema expectations."""
import copy
import math
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
import audit_dense_precision as audit

@pytest.mark.parametrize('axis',[0,1,2])
def test_single_axis_retractions_match_physical_quarter_turns(axis):
    delta=np.zeros(6);delta[axis]=math.pi/2
    expected=[np.array([[1,0,0],[0,0,-1],[0,1,0]]),np.array([[0,0,1],[0,1,0],[-1,0,0]]),np.array([[0,-1,0],[1,0,0],[0,0,1]])][axis]
    for kind in ('vector','euler'):
        result=audit.retract(np.eye(4),delta,kind)
        assert np.max(np.abs(result[:3,:3]-expected))<1e-14


def test_mixed_rotations_have_distinct_finite_retractions():
    delta=np.array([.15,-.25,.35,.04,-.02,.03])
    a=audit.retract(np.eye(4),delta,'vector');b=audit.retract(np.eye(4),delta,'euler')
    assert np.max(np.abs(a[:3,:3]-b[:3,:3]))>.01
    assert np.array_equal(a[:3,3],delta[3:]) and np.array_equal(b[:3,3],delta[3:])
    audit.proper_pose(a);audit.proper_pose(b)


@pytest.mark.parametrize('kind',['vector','euler'])
def test_left_update_has_correct_physical_point_action(kind):
    initial=np.eye(4);initial[:3,:3]=audit.euler_rotation([.2,-.1,.3]);initial[:3,3]=[1,2,3]
    delta=np.array([-.1,.2,.05,.03,.01,-.02]);step=audit.retract(np.eye(4),delta,kind);result=audit.retract(initial,delta,kind)
    points=np.array([[0,0,0],[1,2,3],[-2,.2,1.]])
    original_world=points@initial[:3,:3].T+initial[:3,3]
    expected=original_world@step[:3,:3].T+delta[3:]
    assert np.max(np.abs(points@result[:3,:3].T+result[:3,3]-expected))<1e-12


@pytest.mark.parametrize('change',['reflection','scale','nan','bottom','shape'])
def test_improper_or_nonfinite_poses_reject(change):
    pose=np.eye(4)
    if change=='reflection':pose[0,0]=-1
    elif change=='scale':pose[0,0]=1.01
    elif change=='nan':pose[1,1]=np.nan
    elif change=='bottom':pose[3,0]=1
    else:pose=pose[:3]
    with pytest.raises(ValueError):audit.proper_pose(pose)


@pytest.mark.parametrize('kind',['vector','euler'])
def test_zero_retraction_preserves_the_anchor_exactly(kind):
    pose=np.eye(4);pose[:3,:3]=audit.euler_rotation([.1,.2,.3]);pose[:3,3]=[1,2,3]
    assert np.array_equal(audit.retract(pose,np.zeros(6),kind),pose)


def test_separable_filters_match_constant_and_affine_analytics():
    y,x=np.mgrid[:8,:8];values=(3*x+5*y+7).astype(np.float32)
    gx=audit._separable32(values,[-1,0,1],[1,2,1]);gy=audit._separable32(values,[1,2,1],[-1,0,1])
    smooth=audit._separable32(values,[.25,.5,.25],[.25,.5,.25])
    assert np.all(gx[1:-1,1:-1]==24) and np.all(gy[1:-1,1:-1]==40)
    assert np.array_equal(smooth[1:-1,1:-1],values[1:-1,1:-1]) and smooth[0,0]==9
    assert np.array_equal(audit._separable32(np.full((4,5),2,np.float32),[.25,.5,.25],[.25,.5,.25]),np.full((4,5),2,np.float32))


def test_float32_filter_store_rounds_a_known_quarter_unit():
    values=np.array([[2**24,1,0]]*3,np.float32)
    reference=audit._separable32(values,[.25,.5,.25],[.25,.5,.25])
    native=audit._kernel64(values,[1,2,1],[1,2,1])/16
    assert reference.dtype==np.float32 and reference[0,0]==12582912
    assert native[0,0]==12582912.25 and native[0,0]-reference[0,0]==.25


@pytest.mark.parametrize('precision',['native','reference'])
@pytest.mark.parametrize('index',range(6))
def test_fixed_synthetic_normal_systems_are_symmetric_solvable_and_proper(index,precision):
    plan=audit.load_plan();config=plan['cases'][index];payload=audit.fixture(config)
    result=audit.normal_model(payload,config,plan,precision);a=np.asarray(result['matrix']);b=np.asarray(result['rhs']);delta=np.asarray(result['delta'])
    assert np.all(np.isfinite(a)) and np.max(np.abs(a-a.T))<1e-9
    assert np.min(np.linalg.eigvalsh(a))>0 and result['condition']<=plan['tolerances']['maximum_normal_condition']
    assert np.max(np.abs(a@delta-b))/max(1.,np.max(np.abs(b)))<=plan['tolerances']['normal_relative_residual']
    assert result['modeled_correspondences']==config['width']*config['height']
    for value in result['predictions'].values():audit.proper_pose(value)
    if config['shift']==[0.,0.]:
        assert np.array_equal(delta,np.zeros(6))
        assert np.array_equal(result['predictions']['vector'],np.eye(4))


@pytest.mark.parametrize('change',['name','shape','boolean','nonfinite','range'])
def test_synthetic_input_schema_never_coerces_invalid_values(change):
    config=audit.load_plan()['cases'][0];payload=audit.fixture(config)
    if change=='name':payload['name']='other'
    elif change=='shape':payload['source_gray'].pop()
    elif change=='boolean':payload['source_gray'][0][0]=True
    elif change=='nonfinite':payload['source_depth'][0][0]=float('nan')
    else:payload['target_depth'][0][0]=6
    with pytest.raises(ValueError):audit.arrays(payload,config)


@pytest.mark.parametrize('change',['whitespace','mixed','threshold'])
def test_frozen_plan_cannot_change_tolerances_or_line_ending_policy(tmp_path,change):
    raw=(ROOT/'benchmarks/dense-precision-audit-plan.json').read_bytes().replace(b'\r\n',b'\n')
    if change=='whitespace':raw+=b' '
    elif change=='mixed':raw=raw.replace(b'\n',b'\r\n',1)
    else:raw=raw.replace(b'1e-08',b'1e-07')
    path=tmp_path/'plan.json';path.write_bytes(raw)
    with pytest.raises(ValueError):audit.load_plan(path)


def test_case_omission_is_rejected_before_comparison():
    plan=audit.load_plan();inputs=audit.make_inputs(plan)
    with pytest.raises(ValueError,match='all fixed'):
        audit.audit(plan,inputs,dict(cases=[]),dict(cases=[]))
