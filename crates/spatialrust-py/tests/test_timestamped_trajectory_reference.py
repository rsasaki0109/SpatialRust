"""Exact epoch association, rigid gauge, drift and missing-coverage regression."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from timestamped_trajectory_reference import associate,decimal_timestamp_ns,evaluate,read_tum


def test_decimal_epoch_and_one_nanosecond_are_exact():
    assert decimal_timestamp_ns('1607987782.123456789')==1607987782123456789
    assert decimal_timestamp_ns('1607987782.123456790')==1607987782123456790


@pytest.mark.parametrize('value',['nan','inf','-1','0.0000000001','1607987782.12345678900000000000001',
                                '9223372036.854775808','x',True,1.0,'1e-10000000000'])
def test_inexact_nonfinite_or_out_of_range_timestamps_rejected(value):
    with pytest.raises(ValueError):decimal_timestamp_ns(value)


def test_shortest_path_slerp_bounded_bracket_and_no_extrapolation():
    data=b'1607987782.000000000 0 0 0 0 0 0 1\n1607987782.000000002 2 0 0 0 0 -1 0\n'
    records=read_tum(data);times=[r['timestamp_ns'] for r in records]
    pose=associate(records,times,times[0]+1,1)
    np.testing.assert_allclose(pose[:3,3],[1,0,0],atol=1e-15)
    np.testing.assert_allclose(pose[:3,:3],[[0,1,0],[-1,0,0],[0,0,1]],atol=3e-16)
    assert associate(records,times,times[0]-1,10) is None
    assert associate(records,times,times[-1]+1,10) is None
    assert associate(records,times,times[0]+1,0) is None
    sign=read_tum(b'0 0 0 0 0 0 0 1\n2 2 0 0 0 0 0 -1\n')
    np.testing.assert_array_equal(associate(sign,[0,2000000000],1000000000,1000000000)[:3,:3],np.eye(3))


@pytest.mark.parametrize('data',[b'',b'0 0 0 0 0 0 0 0\n',b'0 nan 0 0 0 0 0 1\n',
                               b'0 0 0 0 0 0 0 1\n0 0 0 0 0 0 0 1\n',b'0 0 0 0 0 0 1\n'])
def test_bad_reference_rows_are_rejected(data):
    with pytest.raises(ValueError):read_tum(data)


def fixture():
    frozen=dict(schema='spatialrust.timestamped-trajectory.v1',length_unit='m',sensor_frame='camera_optical',
                reference_used_for_generation=False,source_sha256='a'*64,calibration_sha256='b'*64,
                plan_sha256='c'*64,poses=[])
    lines=[]
    for i in range(4):
        pose=np.eye(4);pose[0,3]=i
        frozen['poses'].append(dict(timestamp_ns=i*10**9,status='success',world_from_sensor=pose.tolist()))
        lines.append(f'{i} {i} 0 0 0 0 0 1')
    return frozen,('\n'.join(lines)+'\n').encode()


def test_noncommuting_rigid_gauge_and_rpe_are_separate_from_raw_error():
    frozen,ref=fixture()
    gauge=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,30],[0,0,0,1]],float)
    for row in frozen['poses']:row['world_from_sensor']=(gauge @ np.array(row['world_from_sensor'])).tolist()
    result=evaluate(frozen,ref,'camera_optical')
    assert result['evaluated_poses']==4
    assert result['ate_raw_rmse_m']>30
    assert result['ate_first_pose_aligned_rmse_m']==0
    assert result['rpe_translation_rmse_m']==0
    np.testing.assert_array_equal(result['first_pose_alignment'],np.linalg.inv(gauge))


def test_scale_error_is_never_fit_away_and_failures_remain_in_denominator():
    frozen,ref=fixture()
    for row in frozen['poses']:row['world_from_sensor'][0][3]*=2
    frozen['poses'][2]=dict(timestamp_ns=2*10**9,status='error',error='tracking lost')
    result=evaluate(frozen,ref,'camera_optical')
    assert (result['planned_poses'],result['generated_poses'],result['evaluated_poses'])==(4,3,3)
    assert (result['rpe_planned_pairs'],result['rpe_evaluated_pairs'])==(3,1)
    assert result['ate_first_pose_aligned_rmse_m']==pytest.approx(np.sqrt(10/3))
    assert result['rpe_translation_rmse_m']==1


def test_unmatched_reference_is_retained_without_extrapolation():
    frozen,ref=fixture()
    frozen['poses'][-1]['timestamp_ns']=4*10**9
    result=evaluate(frozen,ref,'camera_optical')
    assert result['planned_poses']==4 and result['evaluated_poses']==3
    assert result['unmatched_reference_poses']==1
    assert result['rows'][-1]['reference_matched'] is False


def test_no_reference_overlap_is_unevaluated_and_never_a_zero_error():
    frozen,ref=fixture()
    for row in frozen['poses']:row['timestamp_ns']+=10*10**9
    result=evaluate(frozen,ref,'camera_optical')
    assert result['evaluated_poses']==0
    assert result['unmatched_reference_poses']==4
    assert result['first_pose_alignment'] is None
    assert result['ate_first_pose_aligned_rmse_m'] is None
    assert result['rpe_translation_rmse_m'] is None


def test_association_controls_are_positive_integer_nanoseconds():
    frozen,ref=fixture()
    for controls in ((0,10**9),(20000000,0),(20000000.,10**9),(True,10**9)):
        with pytest.raises(ValueError):evaluate(frozen,ref,'camera_optical',*controls)


@pytest.mark.parametrize('kind',['frame','unit','binding','reference_used','float_time','duplicate_time','reflection'])
def test_false_frame_units_provenance_time_or_pose_rejected(kind):
    frozen,ref=fixture()
    if kind=='frame':frozen['sensor_frame']='other_camera'
    if kind=='unit':frozen['length_unit']='mm'
    if kind=='binding':frozen['calibration_sha256']='invalid'
    if kind=='reference_used':frozen['reference_used_for_generation']=True
    if kind=='float_time':frozen['poses'][0]['timestamp_ns']=0.
    if kind=='duplicate_time':frozen['poses'][1]['timestamp_ns']=0
    if kind=='reflection':frozen['poses'][0]['world_from_sensor'][0][0]=-1
    with pytest.raises(ValueError):evaluate(frozen,ref,'camera_optical')


def test_hash_frozen_cli_rejects_changed_input_and_existing_output(tmp_path):
    frozen,ref=fixture()
    raw=(json.dumps(frozen)+'\n').encode()
    (tmp_path/'estimates.json').write_bytes(raw);(tmp_path/'truth.txt').write_bytes(ref)
    cli=Path(__file__).resolve().parents[3]/'scripts/evaluate_timestamped_trajectory.py'
    output=tmp_path/'accuracy.json'
    command=[sys.executable,str(cli),'--estimates',str(tmp_path/'estimates.json'),
             '--estimates-sha256',hashlib.sha256(raw).hexdigest(),'--reference',str(tmp_path/'truth.txt'),
             '--reference-sha256',hashlib.sha256(ref).hexdigest(),'--reference-frame','camera_optical','--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    saved=output.read_bytes()
    assert json.loads(saved)['evaluated_poses']==4
    assert subprocess.run(command,capture_output=True).returncode!=0
    assert output.read_bytes()==saved
    output.unlink();(tmp_path/'estimates.json').write_bytes(raw+b' ')
    assert subprocess.run(command,capture_output=True).returncode!=0
    assert not output.exists()
