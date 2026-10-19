"""Native preprocessing ownership, raw final measurements and fixed full plans."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import spatialrust as sr

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'scripts'))
from audit_tum_dense_rgbd import audit,check_causal_initialization,reference_statistics
from dense_rgbd_protocol import CONTROLS
from gaussian_dense_rgbd_protocol import ESTIMATOR,PARENTS,POLICY,validate_gaussian_dense_plan
from run_tum_gaussian_dense_rgbd import PreviousMotionGaussianRgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256,prepare
from test_dense_rgbd_odometry import images,estimate
from test_tum_sequence_reference import fixture

PLAN=ROOT/'benchmarks/tum-fr1-xyz-gaussian3-dense-plan.json'


def test_gaussian_known_metric_motion_and_independent_original_raw_statistics():
    source,target=images(),images(1);original=[v.copy() for v in (*source,*target)]
    rotation,translation,report=estimate(source,target,preprocessing='gaussian3')
    np.testing.assert_allclose(rotation,np.eye(3),rtol=0,atol=5e-4)
    np.testing.assert_allclose(translation,[.03125,0,0],rtol=0,atol=5e-4)
    assert report['fitting_preprocessing']=='gaussian3' and report['final_support_only'] is True
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=translation
    camera=dict(width=96,height=72,fx=64.,fy=64.,cx=48.,cy=36.,depth_unit_m=.0002)
    counts=np.full((72,96),10000,np.uint16)
    expected=reference_statistics(counts,counts,source[0],target[0],pose,camera,CONTROLS)
    for key in ('valid_source_pixels','correspondences'):assert report[key]==expected[key]
    for key in ('correspondence_fraction','photometric_rmse','depth_rmse_m'):
        assert abs(report[key]-expected[key])<1e-10
    for actual,wanted in zip((*source,*target),original):np.testing.assert_array_equal(actual,wanted)


def test_raw_default_stays_explicit_raw_and_filtered_bound_charges_retained_planes():
    source=images();a=estimate(source,source);b=estimate(source,source,preprocessing='raw')
    for x,y in zip(a[:2],b[:2]):np.testing.assert_array_equal(x,y)
    assert a[2]==b[2] and a[2]['fitting_preprocessing']=='raw'
    filtered=estimate(source,source,preprocessing='gaussian3')[2]
    assert filtered['working_bytes']-a[2]['working_bytes']==32*96*72
    with pytest.raises(ValueError,match='budget'):
        estimate(source,source,preprocessing='gaussian3',max_working_bytes=filtered['working_bytes']-1)


def test_missing_depth_keeps_raw_denominator_and_does_not_fill_the_input():
    gray,depth=images();depth[36,48]=np.nan
    _,_,report=estimate((gray,depth),(gray,depth),preprocessing='gaussian3')
    assert report['valid_source_pixels']==96*72-1 and np.isnan(depth[36,48])
    assert report['photometric_rmse']<1e-12 and report['depth_rmse_m']<1e-12


@pytest.mark.parametrize('policy',['','Gaussian3','unknown',None,1])
def test_unknown_or_non_string_policy_rejects(policy):
    source=images()
    with pytest.raises((ValueError,TypeError)):estimate(source,source,preprocessing=policy)


@pytest.mark.parametrize('value',[0.,-1.,np.nan,np.inf])
def test_all_missing_or_invalid_depth_cannot_pass_filtered_fitting(value):
    gray,depth=images();depth[:]=value
    with pytest.raises(ValueError):estimate((gray,depth),(gray,depth),preprocessing='gaussian3')


def test_strided_filtered_calls_keep_independent_owned_deterministic_results():
    source=tuple(np.repeat(v,2,axis=1)[:,::2] for v in images(width=256,height=160))
    assert all(not v.flags.c_contiguous for v in source)
    def run(_):return estimate(source,source,iterations=[1],preprocessing='gaussian3')
    with ThreadPoolExecutor(max_workers=2) as workers:a,b=workers.map(run,range(2))
    for x,y in zip(a[:2],b[:2]):np.testing.assert_array_equal(x,y)
    assert a[2]==b[2]
    for output in a[:2]:assert all(not np.shares_memory(output,v) for v in source)


@pytest.mark.parametrize('scene',['xyz','rpy'])
def test_prospective_plan_preserves_all_parent_numeric_and_final_controls(scene):
    plan=json.loads((ROOT/'benchmarks'/('tum-fr1-'+scene+'-gaussian3-dense-plan.json')).read_text(encoding='utf-8'))
    validate_gaussian_dense_plan(plan);parent=json.loads(PARENTS[plan['sequence']].read_text(encoding='utf-8'))
    for key in ('calibration','limits','acceptance','dense_initialization'):assert plan[key]==parent[key]
    for key,value in parent['controls'].items():assert plan['controls'][key]==value


@pytest.mark.parametrize('group,key,value',[
    ('dense_initialization','time_rescaling',True),('acceptance','maximum_first_pose_ate_rmse_m',.2),
    ('controls','min_support_fraction',.4),('calibration','fx',517.3),
    ('dense_fitting_preprocessing','intensity_normalization',True),
    ('dense_fitting_preprocessing','depth_missing','fill missing taps')])
def test_plan_rejects_any_model_seed_calibration_or_gate_change(group,key,value):
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    target=plan['controls'][group] if group=='dense_fitting_preprocessing' else plan[group]
    target[key]=value
    with pytest.raises(ValueError):validate_gaussian_dense_plan(plan)


def alignment_args():
    depth=np.full((32,32),5000,np.uint16);gray=np.full((32,32),100,np.uint8)
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    return depth,depth.copy(),gray,gray.copy(),plan['calibration'],plan['controls']


def mock_native(calls, *, fail_at=None, wrong_policy=False, audit_stats=False):
    def solve(*args,**kwargs):
        calls.append(kwargs['initial'].copy());assert kwargs['preprocessing']=='gaussian3'
        if len(calls)==fail_at:raise ValueError('controlled Gaussian3 failure')
        depth=args[1];valid=int(np.count_nonzero(depth>0));count=(depth.shape[0]-1)*(depth.shape[1]-1) if audit_stats else valid
        return np.eye(3),np.zeros(3),dict(valid_source_pixels=valid,correspondences=count,
            correspondence_fraction=count/valid,photometric_rmse=0.,depth_rmse_m=0.,
            final_support_only=True,fitting_preprocessing='raw' if wrong_policy else 'gaussian3')
    return solve


def test_owned_causal_policy_receipts_and_failure_do_not_advance_state(monkeypatch):
    calls=[];monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',mock_native(calls,fail_at=2))
    adapter=PreviousMotionGaussianRgbd();_,first=adapter.align(*alignment_args())
    first['fitting_policy']['kernel'][0]=99
    assert POLICY['kernel']==[1,2,1]
    with pytest.raises(ValueError):adapter.align(*alignment_args())
    _,third=adapter.align(*alignment_args());assert third['initial_previous_pair_index']==1
    check_causal_initialization(third,np.eye(4),2)


def test_wrong_native_fitting_policy_cannot_be_accepted(monkeypatch):
    calls=[];monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',mock_native(calls,wrong_policy=True))
    with pytest.raises(ValueError,match='policy'):PreviousMotionGaussianRgbd().align(*alignment_args())


@pytest.mark.parametrize('reject_support',[False,True])
def test_full_generator_retains_raw_audit_and_never_calls_after_loss(tmp_path,monkeypatch,reject_support):
    archive,digest=fixture(tmp_path,count=3);prepared=tmp_path/'prepared'
    prepare(archive,digest,PLAN,prepared);calls=[]
    monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',mock_native(calls,audit_stats=True))
    if reject_support:monkeypatch.setattr(sr,'distance_gated_support',lambda *args:(0,0.,None))
    result=generate(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result',
        dense_rgbd_aligner=PreviousMotionGaussianRgbd().align,method=ESTIMATOR,additional_bindings=[__file__])
    rows=json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert len(rows)==3 and len(calls)==(1 if reject_support else 2)
    if reject_support:assert [r['status'] for r in rows]==['success','error','error']
    else:
        checked=audit(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result/estimates.json',result['estimates_sha256'],tmp_path/'audit')
        assert checked['successful_pairs']==checked['causal_initialization_pairs']==2
