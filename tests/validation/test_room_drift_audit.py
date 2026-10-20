"""Closed-form full-timeline drift and strict frozen IO/coverage evidence."""
import copy
import hashlib
import json
import math
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import audit_room_drift as audit
from timestamped_trajectory_reference import evaluate


def fixture(label='reference-precision',n=17,dx=.02,yaw_step=0.,true_step=.1,gap=None,fail_at=None):
    c=copy.deepcopy(audit.load_plan()['controls']);c['planned_frames']=n
    base=10**18;dt=250000000;times=[base+i*dt for i in range(n)]
    gauge=np.eye(4);gauge[:3,:3]=Rotation.from_euler('z',.3).as_matrix();gauge[:3,3]=[2.,-1.,.5]
    worlds=[];reference=[]
    for i,t in enumerate(times):
        e=np.eye(4);e[:3,:3]=Rotation.from_euler('z',i*yaw_step).as_matrix();e[0,3]=(true_step+dx)*i;worlds.append(e)
        if i==gap:continue
        truth=gauge.copy();truth[:3,3]+=gauge[:3,:3]@np.array([true_step*i,0.,0.]);q=Rotation.from_matrix(truth[:3,:3]).as_quat()
        reference.append(str(t//10**9)+'.'+f'{t%10**9:09d}'+' '+' '.join(map(str,[*truth[:3,3],*q])))
    reference=('\n'.join(reference)+'\n').encode('ascii')
    poses=[];checks=[];previous=None
    for i,t in enumerate(times):
        row=dict(timestamp_ns=t,status='success',world_from_sensor=worlds[i].tolist())
        if fail_at is not None and i>=fail_at:row.update(status='error',error='retained synthetic chain loss');poses.append(row);continue
        if i:
            relative=np.linalg.inv(worlds[i])@worlds[i-1];inverse=np.linalg.inv(relative)
            report=dict(fitting_model=label,fitting_preprocessing='raw',fitting_policy=copy.deepcopy(audit.POLICIES[label]),dense_settings=copy.deepcopy(audit.CONTROLS),iterations=35,final_support_only=True,valid_source_pixels=1000,correspondences=900,correspondence_fraction=.9,photometric_rmse=.01+.0001*i,depth_rmse_m=.002+.00001*i,support_count=990,support_fraction=.99,support_rmse_m=.015,current_from_previous=relative.tolist(),previous_from_current=inverse.tolist(),initial_seed_policy='previous-relative-increment',initial_previous_pair_index=i-1 if i>1 else None,initial_current_from_previous=(np.eye(4) if previous is None else previous).tolist())
            row['registration']=report;previous=relative
            checks.append(dict(index=i,statistics={k:report[k] for k in ['valid_source_pixels','correspondences','correspondence_fraction','photometric_rmse','depth_rmse_m']}))
        poses.append(row)
    frozen=dict(schema='spatialrust.timestamped-trajectory.v1',length_unit='m',sensor_frame='tum_rgb_optical',reference_used_for_generation=False,source_sha256='1'*64,plan_sha256='2'*64,calibration_sha256='3'*64,manifest_sha256='4'*64,poses=poses)
    scored=evaluate(frozen,reference,'tum_rgb_optical',c['max_reference_gap_ns'],c['rpe_delta_ns']);scored.update(source_sha256=frozen['source_sha256'],plan_sha256=frozen['plan_sha256'])
    raw=dict(schema='spatialrust.dense-rgbd-audit.v1',reference_motion_used=False,manifest_sha256=frozen['manifest_sha256'],planned_frames=n,successful_pairs=len(checks),causal_initialization_pairs=len(checks),pair_checks=checks,failed_frame_indices=[i for i,p in enumerate(poses) if p['status']=='error'])
    keys=['planned_poses','generated_poses','evaluated_poses','unmatched_reference_poses','rpe_planned_pairs','rpe_evaluated_pairs']
    score=dict(schema='spatialrust.tum-independent-audit.v1',counts={k:scored[k] for k in keys},aggregates={k:scored[k] for k in ['ate_raw_rmse_m','ate_first_pose_aligned_rmse_m','rpe_translation_rmse_m']})
    return [label,frozen,scored,raw,score,reference,c]


def test_linear_drift_closed_form_reconstructs_every_bin_and_original_gauge():
    result,_,_=audit.analyze_case(*fixture())
    assert result['counts']['planned_poses']==result['counts']['evaluated_poses']==17
    expected=math.sqrt(sum((.02*i)**2 for i in range(17))/17)
    assert abs(result['metrics']['aligned']['translation_m']['rmse']-expected)<1e-12
    assert abs(result['metrics']['adjacent']['translation_m']['rmse']-.02)<1e-12
    assert abs(result['metrics']['rpe']['translation_m']['rmse']-.08)<1e-12
    assert result['anchor_index']==0 and np.linalg.norm(np.asarray(result['original_first_pose_gauge'])[:3,3])>2.
    assert result['original_acceptance'] is False
    bins=result['chronological_bins'];assert sum(b['planned_frames'] for b in bins)==17
    assert [i for b in bins for i in b['indices']]==list(range(17))
    assert sum(b['metrics']['rpe']['translation_m']['count'] for b in bins)==13
    assert abs(sum(b['metrics']['aligned']['translation_m']['whole_sse_fraction'] for b in bins)-1.)<1e-12
    assert abs(result['rows'][-1]['aligned_error']['translation_m']-.32)<1e-12


@pytest.mark.parametrize('label',['rounded-sobel','reference-precision'])
def test_rotation_only_drift_closed_form_and_exact_legacy_metadata(label):
    result,_,_=audit.analyze_case(*fixture(label=label,dx=0.,true_step=0.,yaw_step=.01))
    rms=math.degrees(.01)*math.sqrt(sum(i*i for i in range(17))/17)
    assert abs(result['metrics']['aligned']['rotation_degrees']['rmse']-rms)<1e-10
    assert abs(result['metrics']['adjacent']['rotation_degrees']['rmse']-math.degrees(.01))<1e-10
    assert abs(result['metrics']['rpe']['rotation_degrees']['rmse']-math.degrees(.04))<1e-10
    assert result['metrics']['aligned']['translation_m']['rmse']<1e-12
    assert result['original_acceptance'] is True


def test_reference_gap_and_generation_failure_are_retained_in_full_denominators():
    result,_,_=audit.analyze_case(*fixture(gap=6))
    assert len(result['rows'])==17 and result['counts']['evaluated_poses']==16
    assert result['rows'][6]['reference_matched'] is False and result['rows'][6]['aligned_error'] is None
    assert result['rows'][6]['adjacent'] is result['rows'][7]['adjacent'] is None
    assert len(result['rpe_pairs'])==13 and sum(not p['evaluated'] for p in result['rpe_pairs'])==2
    assert result['original_acceptance'] is False
    lost,_,_=audit.analyze_case(*fixture(fail_at=5))
    assert len(lost['rows'])==17 and lost['counts']['generated_poses']==5
    assert sum(r['status']=='error' for r in lost['rows'])==12


@pytest.mark.parametrize('angle',[0.,1e-10,90.,179.999999,180.])
def test_principal_rotation_and_rigid_translation_are_analytical(angle):
    axis=np.asarray([1.,2.,-3.]);axis/=np.linalg.norm(axis)
    pose=np.eye(4);pose[:3,:3]=Rotation.from_rotvec(axis*math.radians(angle)).as_matrix();pose[:3,3]=[3.,4.,12.]
    result=audit.pose_error(pose,np.eye(4))
    assert result['translation_m']==13.
    assert abs(result['rotation_degrees']-angle)<1e-9
    assert abs(np.linalg.norm(result['rotation_vector_degrees'])-angle)<1e-9
    reflected=pose.copy();reflected[:3,0]*=-1
    with pytest.raises(ValueError):audit.pose_error(reflected,np.eye(4))


def test_exact_large_epoch_interpolation_and_rpe_earlier_tie():
    values=fixture();base=values[1]['poses'][0]['timestamp_ns']
    refs=audit.references(values[5],[base+125000000],125000000)
    expected=np.array([2.,-1.,.5])+Rotation.from_euler('z',.3).apply([.05,0,0])
    np.testing.assert_allclose(refs[0][:3,3],expected,rtol=0,atol=1e-12)
    assert audit.references(values[5],[base-1],20000000)==[None]
    assert audit.pair_indices([0,700000000,1300000000],1000000000,300000000)==[(0,1)]


def test_pearson_scale_constant_and_sample_count_rules_are_explicit():
    assert abs(audit.pearson([1e308,5e307,0.],[-1e308,-5e307,0.])['value']+1.)<1e-14
    assert audit.pearson([1.,1.],[1.,2.])['undefined_reason']=='constant-covariate-or-error'
    assert audit.pearson([1.],[2.])['undefined_reason']=='fewer-than-two-observations'
    with pytest.raises(ValueError):audit.pearson([1.,math.nan],[1.,2.])
    with pytest.raises(ValueError):audit.pearson([1.,2.],[1.])
    with pytest.raises(ValueError):audit.moments([1e308])
    assert audit.moments([])['rmse'] is None


@pytest.mark.parametrize('kind',[
    'drop-pose','duplicate-time','nonfinite','reflection','score-mask','raw-count','support',
    'seed-index-bool','seed-pose','inverse','policy','iterations','rpe-denominator','rpe-time',
    'rpe-error','aggregate','independent-count','independent-aggregate','raw-denominator',
    'reset','lineage','scalar-bool'])
def test_corrupt_source_schema_gauge_causal_scores_and_denominators_reject(kind):
    values=fixture();_,frozen,scored,raw,score,_,_=values;r=frozen['poses'][2]['registration']
    if kind=='drop-pose':frozen['poses'].pop()
    elif kind=='duplicate-time':frozen['poses'][2]['timestamp_ns']=frozen['poses'][1]['timestamp_ns']
    elif kind=='nonfinite':frozen['poses'][2]['world_from_sensor'][0][3]=math.inf
    elif kind=='reflection':frozen['poses'][2]['world_from_sensor'][0][0]=-1.
    elif kind=='score-mask':scored['rows'][2]['reference_matched']=False
    elif kind=='raw-count':raw['pair_checks'][1]['statistics']['correspondences']+=1
    elif kind=='support':r['support_fraction']=.49
    elif kind=='seed-index-bool':r['initial_previous_pair_index']=True
    elif kind=='seed-pose':r['initial_current_from_previous'][0][3]+=.01
    elif kind=='inverse':r['previous_from_current'][0][3]+=.01
    elif kind=='policy':r['fitting_policy']['workers']=99
    elif kind=='iterations':r['iterations']=34
    elif kind=='rpe-denominator':scored['rpe_pairs'].pop()
    elif kind=='rpe-time':scored['rpe_pairs'][0]['delta_ns']+=1
    elif kind=='rpe-error':scored['rpe_pairs'][0]['error']['translation_m']+=.01
    elif kind=='aggregate':scored['ate_first_pose_aligned_rmse_m']=0.
    elif kind=='independent-count':score['counts']['generated_poses']-=1
    elif kind=='independent-aggregate':score['aggregates']['rpe_translation_rmse_m']=0.
    elif kind=='raw-denominator':raw['pair_checks'].pop()
    elif kind=='reset':frozen['poses'][2].update(status='error',error='lost')
    elif kind=='lineage':scored['source_sha256']='9'*64
    elif kind=='scalar-bool':r['depth_rmse_m']=True
    with pytest.raises(ValueError):audit.analyze_case(*values)


def test_comparison_uses_only_original_gauges_and_keeps_reference_gap():
    a,pa,ta=audit.analyze_case(*fixture(label='rounded-sobel',dx=.01,gap=6))
    b,pb,tb=audit.analyze_case(*fixture(dx=.02,gap=6))
    result=audit.compare_cases({'rounded-sobel':a,'reference-precision':b},{'rounded-sobel':pa,'reference-precision':pb},{'rounded-sobel':ta,'reference-precision':tb})
    assert len(result['rows'])==17 and result['rows'][6]['error'] is None
    assert abs(result['rows'][-1]['error']['translation_m']-.16)<1e-12


def test_bound_file_uniform_plan_and_reparse_reject(tmp_path,monkeypatch):
    path=tmp_path/'frozen.json';raw=b'{"value": 1}';path.write_bytes(raw)
    entry=dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    assert audit.read_bound(entry,1024)==raw
    with pytest.raises(ValueError):audit.read_bound(dict(entry,bytes=True),1024)
    with pytest.raises(ValueError):audit.read_bound(entry,2)
    with pytest.raises(ValueError):audit.read_bound(dict(entry,path='relative.json'),1024)
    original=Path.lstat
    with monkeypatch.context() as m:
        m.setattr(Path,'lstat',lambda p:SimpleNamespace(st_mode=stat.S_IFREG,st_file_attributes=0x400) if p==path else original(p))
        with pytest.raises(ValueError):audit.read_bound(entry,1024)
    path.write_bytes(b'{"value": 2}')
    with pytest.raises(ValueError):audit.read_bound(entry,1024)
    plan=audit.PLAN.read_bytes().replace(b'\r\n',b'\n');p=tmp_path/'plan.json';p.write_bytes(plan.replace(b'\n',b'\r\n'))
    assert audit.load_plan(p)['slice']=='146U'
    p.write_bytes(plan.replace(b'\n',b'\r\n',1))
    with pytest.raises(ValueError):audit.load_plan(p)


@pytest.mark.parametrize('raw',[b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e400}',b'{"x":1,"x":2}'])
def test_nonfinite_and_duplicate_json_reject(raw):
    with pytest.raises(ValueError):audit.json_value(raw)


def test_full_timeline_plot_preserves_counts_and_missing_gaps(tmp_path):
    a,_,_=audit.analyze_case(*fixture(label='rounded-sobel',gap=6))
    b,_,_=audit.analyze_case(*fixture(gap=6))
    result=audit.render(dict(cases={'rounded-sobel':a,'reference-precision':b}),tmp_path)
    assert all(c==dict(all_frame_samples=17,all_planned_rpe_pairs=13) for c in result['plotted_counts'].values())
    assert result['path_simplification'] is False
    assert (tmp_path/'drift.svg').stat().st_size>10000 and (tmp_path/'preview.png').stat().st_size>10000
