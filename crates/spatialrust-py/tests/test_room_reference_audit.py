"""Fixed full-room reference admission, independent geometry and complete comparison."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_room_reference import compare_rows,modeled_pairs
from room_reference_rgbd_protocol import (ESTIMATOR,PARENT,PLAN,PLAN_LF_SHA256,SELECTION,
    room_reference_plan,source_identity,validate_room_reference_plan,validate_transition)
from reference_precision_dense_rgbd_protocol import fixed_json
from run_tum_open3d_rgbd import PreviousMotionOpen3dRgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256,prepare
import test_tum_sequence_reference as fixture_module


def case(name):return copy.deepcopy(next(c for c in SELECTION['correspondence_audit']['cases'] if c['name']==name))


@pytest.mark.parametrize('config',SELECTION['correspondence_audit']['cases'],ids=lambda c:c['name'])
def test_source_equations_match_fixed_analytic_counts_and_multiplicity(config):
    a=modeled_pairs(config,'native-source');b=modeled_pairs(config,'reference-source')
    assert a==b and len(a)==config['expected_rows']
    assert len({tuple(row[2:]) for row in a})==config['expected_unique_targets']


def test_contraction_retains_four_sources_per_target_without_zbuffer_selection():
    rows=modeled_pairs(case('contraction-many-to-one'),'native-source')
    for u,v in [(1,1),(2,1),(1,2),(2,2)]:
        assert sum(row[2:]==[u,v] for row in rows)==4
    assert rows[:4]==[[0,0,1,1],[1,0,1,1],[2,0,2,1],[3,0,2,1]]


def test_signed_rounding_is_truncation_toward_zero_and_keeps_original_border():
    rows=modeled_pairs(case('signed-negative-rounding-edge'),'reference-source')
    assert rows[:3]==[[1,0,0,0],[2,0,1,0],[3,0,2,0]]
    assert all(row[0]>0 for row in rows)


@pytest.mark.parametrize('change',[dict(width=True),dict(width=65),dict(camera=[0.,4.,1.5,1.5]),dict(translation=[0.,0.]),dict(source_depth_m=float('inf')),dict(target_holes=[[4,0]])])
def test_source_model_rejects_unbounded_or_undefined_geometry(change):
    config=case('identity');config.update(change)
    with pytest.raises(ValueError):modeled_pairs(config,'native-source')


def test_fixed_plan_retains_calibration_original_gates_and_native_controls():
    plan=room_reference_plan();validate_room_reference_plan(plan);parent,_=fixed_json(PARENT)
    for key in ('sequence','calibration','acceptance','limits','dense_initialization'):assert plan[key]==parent[key]
    for key,value in parent['controls'].items():
        if key!='dense_fitting_model':assert plan['controls'][key]==value
    assert plan['estimator']==ESTIMATOR and 'dense_fitting_model' not in plan['controls']
    validate_transition(parent,plan)
    changed=copy.deepcopy(parent);changed['controls']['min_support_fraction']=.4
    with pytest.raises(ValueError):validate_transition(changed,plan)


@pytest.mark.parametrize('group,key,value',[
    ('acceptance','maximum_first_pose_ate_rmse_m',.2),('controls','min_support_fraction',.4),
    ('calibration','fx',517.3),('dense_initialization','time_rescaling',True),
    ('top','sequence','rgbd_dataset_freiburg1_plant'),('top','open3d_cpu_module_sha256','0'*64)])
def test_fixed_plan_rejects_any_scope_calibration_gate_or_source_change(group,key,value):
    plan=room_reference_plan();(plan if group=='top' else plan[group])[key]=value
    with pytest.raises(ValueError):validate_room_reference_plan(plan)


def test_pins_and_uniform_line_endings(tmp_path):
    raw=PLAN.read_bytes().replace(b'\r\n',b'\n');p=tmp_path/'plan.json';p.write_bytes(raw.replace(b'\n',b'\r\n'))
    assert fixed_json(p,PLAN_LF_SHA256)[0]['slice']=='146W'
    p.write_bytes(raw.replace(b'\n',b'\r\n',1))
    with pytest.raises(ValueError):fixed_json(p,PLAN_LF_SHA256)
    with pytest.raises(ValueError):source_identity(SELECTION['sequence'],'0'*64)
    with pytest.raises(ValueError):source_identity('rgbd_dataset_freiburg1_plant',SELECTION['source_sha256'])


def trajectory():
    return [dict(timestamp_ns=i+1,status='success',world_from_sensor=np.eye(4).tolist()) for i in range(3)]


def test_frozen_comparison_matches_independent_translation_and_axis_rotation():
    a,b=trajectory(),trajectory();p=np.eye(4);p[:3,3]=[3.,4.,0.];p[:3,:3]=[[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]];b[1]['world_from_sensor']=p.tolist()
    result=compare_rows(a,b)
    assert result['complete_comparison'] and result['planned_rows']==result['paired_rows']==3
    assert result['translation_max_m']==5. and abs(result['translation_rms_m']-5/np.sqrt(3))<1e-12
    assert result['rotation_max_degrees']==90. and abs(result['rotation_rms_degrees']-90/np.sqrt(3))<1e-12


def test_missing_comparison_rows_remain_explicit_and_cannot_claim_complete():
    a,b=trajectory(),trajectory();b[1]=dict(timestamp_ns=2,status='error',error='preserved failure')
    result=compare_rows(a,b)
    assert not result['complete_comparison'] and result['paired_rows']==2 and result['unpaired_rows']==1
    assert len(result['rows'])==3 and result['rows'][1]['translation_difference_m'] is None


@pytest.mark.parametrize('kind',['short','timestamp','unordered','boolean-time','status','anchor','improper'])
def test_comparison_rejects_trim_rescale_unknown_status_and_invalid_poses(kind):
    a,b=trajectory(),trajectory()
    if kind=='short':b.pop()
    elif kind=='timestamp':b[1]['timestamp_ns']=20
    elif kind=='unordered':a[1]['timestamp_ns']=a[0]['timestamp_ns']
    elif kind=='boolean-time':a[0]['timestamp_ns']=True
    elif kind=='status':b[1]['status']='unknown'
    elif kind=='anchor':b[0]['world_from_sensor'][0][3]=.1
    else:b[1]['world_from_sensor'][0][0]=2.
    with pytest.raises(ValueError):compare_rows(a,b)


def prepared_room(tmp_path,monkeypatch):
    import reuse_tum_prepared as reuse
    monkeypatch.setattr(fixture_module,'ROOT',SELECTION['sequence'])
    monkeypatch.setattr(reuse.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(40*1024**3,1,30*1024**3))
    archive,digest=fixture_module.fixture(tmp_path,count=3);source=tmp_path/'source'
    prepare(archive,digest,PARENT,source)
    plan=ROOT/'benchmarks/tum-fr1-room-reference-comparator-plan.json';bound=file_sha256(source/'manifest.json');before=(source/'manifest.json').read_bytes()
    output=tmp_path/'reference'
    with pytest.raises(ValueError,match='controls cannot change'):reuse.reuse(source,bound,plan,output)
    result=reuse.reuse(source,bound,plan,output,room_reference_revision=True)
    assert (source/'manifest.json').read_bytes()==before and result['reused_input']['new_image_payload_bytes']==0
    for name in result['files']:assert (source/name).samefile(output/name)
    return output,file_sha256(output/'manifest.json')


def test_exact_revision_reuse_and_complete_reference_generation(tmp_path,monkeypatch):
    prepared,digest=prepared_room(tmp_path,monkeypatch)
    adapter=PreviousMotionOpen3dRgbd(_compute=lambda *args:(True,np.eye(4),np.eye(6)))
    result=generate(prepared,digest,tmp_path/'run',dense_rgbd_aligner=adapter.align,method=ESTIMATOR,additional_bindings=[__file__])
    assert result['generated_poses']==result['planned_poses']==3 and not result['tracking_lost']
    rows=json.loads((tmp_path/'run/estimates.json').read_text(encoding='utf-8'))['poses']
    assert rows[2]['registration']['initial_previous_pair_index']==1
    assert rows[2]['registration']['external_final_support_validation'] is True


@pytest.mark.parametrize('flags',[dict(room_reference_revision=1),dict(room_reference_revision=True,huber_revision=True),dict(room_reference_revision=True,capture_revision=True),dict(room_reference_revision=True,reference_precision_revision=True)])
def test_ambiguous_revisions_fail_before_io(tmp_path,flags):
    from reuse_tum_prepared import reuse
    with pytest.raises(ValueError,match='mutually exclusive'):reuse(tmp_path/'missing','0'*64,tmp_path/'missing.json',tmp_path/'out',**flags)
    assert not (tmp_path/'out').exists()
