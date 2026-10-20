"""Explicit precision model: owned inputs, raw final admission and fixed lineage."""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest
import spatialrust as sr

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from huber_dense_rgbd_protocol import (ESTIMATOR,POLICY,PARENTS,
    fixed_json,huber_plan,source_identity,validate_huber_plan,
    validate_transition)
from run_tum_huber_dense_rgbd import PreviousMotionHuberRgbd
from test_dense_rgbd_odometry import images,estimate
from audit_tum_dense_rgbd import reference_statistics
from dense_rgbd_protocol import CONTROLS


def precision(source,target,**kwargs):
    return estimate(source,target,fitting_model='reference-precision-huber',**kwargs)


def test_identity_entire_schedule_raw_measurements_and_no_aliases():
    gray,depth=images();depth[36,48]=np.nan
    before=[gray.copy(),depth.copy()]
    rotation,translation,report=precision((gray,depth),(gray,depth))
    np.testing.assert_allclose(rotation,np.eye(3),rtol=0,atol=1e-12)
    np.testing.assert_allclose(translation,np.zeros(3),rtol=0,atol=1e-12)
    assert report['iterations']==35 and report['fitting_model']=='reference-precision-huber'
    assert report['fitting_preprocessing']=='raw' and report['final_support_only'] is True
    assert report['valid_source_pixels']==96*72-1
    assert 1<=report['workers']<=4 and report['input_copy_bytes']==96*72*10
    for value,old in zip((gray,depth),before):np.testing.assert_array_equal(value,old)
    assert all(not np.shares_memory(v,o) for v in (gray,depth) for o in (rotation,translation))


def test_brightness_gain_fitting_keeps_original_raw_final_error():
    gray,depth=images();gray=(gray//2).astype(np.uint8);target=gray*2
    rotation,translation,report=precision((gray,depth),(target,depth))
    pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=translation
    camera=dict(width=96,height=72,fx=64.,fy=64.,cx=48.,cy=36.,depth_unit_m=.0002)
    counts=np.full((72,96),10000,np.uint16)
    expected=reference_statistics(counts,counts,gray,target,pose,camera,CONTROLS)
    assert report['photometric_rmse']>.1
    for key in ('valid_source_pixels','correspondences'):assert report[key]==expected[key]
    for key in ('correspondence_fraction','photometric_rmse','depth_rmse_m'):
        assert abs(report[key]-expected[key])<1e-10


@pytest.mark.parametrize('policy',['gaussian3','gaussian3-gray-only','gaussian3-gray-only-pair-mean'])
def test_mixed_preprocessing_is_rejected(policy):
    source=images()
    with pytest.raises(ValueError,match='preprocessing must be raw'):
        precision(source,source,preprocessing=policy)


def test_budget_boundary_nan_denominator_and_invalid_mean():
    source=images();required=precision(source,source)[2]['working_bytes']
    assert precision(source,source,max_working_bytes=required)[2]['working_bytes']==required
    with pytest.raises(ValueError,match='budget'):precision(source,source,max_working_bytes=required-1)
    with pytest.raises(ValueError):precision(images(width=7,height=5),images(width=7,height=5),iterations=[1,1])
    gray,depth=source;gray.fill(0)
    with pytest.raises(ValueError,match='pair means'):precision((gray,depth),(gray,depth))
    assert not np.any(gray)


def test_readonly_strided_concurrent_inputs_and_proper_rotation():
    source=tuple(np.repeat(v,2,axis=1)[:,::2] for v in images(width=256,height=160))
    for v in source:v.setflags(write=False)
    def run(_):return precision(source,source,iterations=[1])
    with ThreadPoolExecutor(max_workers=2) as pool:a,b=pool.map(run,range(2))
    for x,y in zip(a[:2],b[:2]):np.testing.assert_array_equal(x,y)
    assert a[2]==b[2] and a[2]['input_copy_bytes']==256*160*10
    np.testing.assert_allclose(a[0].T@a[0],np.eye(3),rtol=0,atol=1e-12)
    assert abs(np.linalg.det(a[0])-1.)<1e-12


@pytest.mark.parametrize('scene',['xyz','rpy','plant','room'])
def test_fixed_observed_plans_retain_all_other_controls_and_gates(scene):
    plan=json.loads((ROOT/'benchmarks'/('tum-fr1-'+scene+'-huber-dense-plan.json')).read_text(encoding='utf-8'))
    validate_huber_plan(plan);parent,_=fixed_json(PARENTS[plan['sequence']])
    assert plan['observed_development'] is True and 'heldout_validation' not in plan
    for key in ('calibration','limits','acceptance','dense_initialization'):assert plan[key]==parent[key]
    for key,value in parent['controls'].items():
        if key!='dense_fitting_model':assert plan['controls'][key]==value
    validate_transition(parent,plan)
    altered=copy.deepcopy(parent);altered['controls']['min_support_fraction']=.4
    with pytest.raises(ValueError):validate_transition(altered,plan)


@pytest.mark.parametrize('group,key,value',[
    ('acceptance','maximum_first_pose_ate_rmse_m',.2),('controls','min_support_fraction',.4),
    ('calibration','fx',517.3),('dense_initialization','time_rescaling',True),
    ('dense_fitting_model','pose_update','Rodrigues'),('dense_fitting_model','storage','f32 buffers'),
    ('dense_fitting_model','final_admission','filtered')])
def test_no_silent_policy_calibration_or_gate_changes(group,key,value):
    plan=huber_plan('rgbd_dataset_freiburg1_xyz')
    target=plan['controls'][group] if group=='dense_fitting_model' else plan[group];target[key]=value
    with pytest.raises(ValueError):validate_huber_plan(plan)


def test_source_pin_and_uniform_line_endings(tmp_path):
    raw=(ROOT/'benchmarks/huber-dense-plan.json').read_bytes().replace(b'\r\n',b'\n')
    path=tmp_path/'fixed.json';path.write_bytes(raw.replace(b'\n',b'\r\n'))
    from huber_dense_rgbd_protocol import PLAN_LF_SHA256
    assert fixed_json(path,PLAN_LF_SHA256)[0]['slice']=='146V'
    path.write_bytes(raw.replace(b'\n',b'\r\n',1))
    with pytest.raises(ValueError):fixed_json(path,PLAN_LF_SHA256)
    with pytest.raises(ValueError):source_identity('rgbd_dataset_freiburg1_room','0'*64)


def test_causal_adapter_failed_call_does_not_advance_and_report_is_owned(monkeypatch):
    calls=[]
    def solve(*args,**kwargs):
        calls.append(kwargs['initial'].copy());assert kwargs['fitting_model']=='reference-precision-huber'
        if len(calls)==2:raise ValueError('controlled fit failure')
        return np.eye(3),np.zeros(3),dict(valid_source_pixels=1024,correspondences=1024,
            correspondence_fraction=1.,photometric_rmse=0.,depth_rmse_m=0.,iterations=35,
            final_support_only=True,fitting_preprocessing='raw',fitting_model='reference-precision-huber')
    monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',solve)
    plan=huber_plan('rgbd_dataset_freiburg1_xyz');d=np.full((32,32),5000,np.uint16);g=np.full((32,32),100,np.uint8)
    args=(d,d.copy(),g,g.copy(),plan['calibration'],plan['controls'])
    adapter=PreviousMotionHuberRgbd();_,first=adapter.align(*args)
    first['fitting_policy']['pose_update']='mutated caller report'
    with pytest.raises(ValueError,match='controlled fit failure'):adapter.align(*args)
    _,last=adapter.align(*args)
    assert last['fitting_policy']==POLICY and last['initial_previous_pair_index']==1
    np.testing.assert_array_equal(calls[1],calls[2])


def test_depth_delta_underflow_rejected_before_estimation():
    source=images()
    with pytest.raises(ValueError,match='underflowed Huber depth delta'):
        precision(source,source,max_depth_difference=np.nextafter(0.,1.))


@pytest.mark.parametrize('field,value',[('photometric_delta',.04),('depth_delta','max_depth_difference / 2')])
def test_robust_thresholds_cannot_change_after_protocol(field,value):
    plan=huber_plan('rgbd_dataset_freiburg1_xyz')
    plan['controls']['dense_fitting_model']['robust_loss'][field]=value
    with pytest.raises(ValueError):validate_huber_plan(plan)


def test_huber_reuse_preserves_parent_bytes_and_shares_payloads(tmp_path,monkeypatch):
    import shutil
    import reuse_tum_prepared as reuse
    from tum_sequence_reference import file_sha256,prepare
    from test_tum_sequence_reference import fixture
    monkeypatch.setattr(reuse.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(40*1024**3,1,30*1024**3))
    archive,digest=fixture(tmp_path,count=3);source=tmp_path/'source';output=tmp_path/'huber'
    prepare(archive,digest,PARENTS['rgbd_dataset_freiburg1_xyz'],source)
    bound=file_sha256(source/'manifest.json');before=(source/'manifest.json').read_bytes()
    plan=ROOT/'benchmarks/tum-fr1-xyz-huber-dense-plan.json'
    with pytest.raises(ValueError,match='controls cannot change'):reuse.reuse(source,bound,plan,output)
    assert not output.exists()
    result=reuse.reuse(source,bound,plan,output,huber_revision=True)
    assert result['reused_input']['new_image_payload_bytes']==0
    assert (source/'manifest.json').read_bytes()==before
    for name,entry in result['files'].items():
        assert (output/name).samefile(source/name) and file_sha256(output/name)==entry['sha256']
    assert str(ROOT/'scripts/huber_dense_rgbd_protocol.py') in result['bindings']


@pytest.mark.parametrize('flags',[dict(huber_revision=1),dict(huber_revision=True,capture_revision=True),dict(huber_revision=True,reference_precision_revision=True)])
def test_ambiguous_reuse_rejected_before_io(tmp_path,flags):
    from reuse_tum_prepared import reuse
    with pytest.raises(ValueError,match='mutually exclusive'):
        reuse(tmp_path/'missing','0'*64,tmp_path/'missing.json',tmp_path/'output',**flags)
    assert not (tmp_path/'output').exists()
