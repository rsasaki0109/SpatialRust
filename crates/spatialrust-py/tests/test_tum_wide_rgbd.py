"""Fixed wide capture retains native row ownership, pose direction and every gate."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import spatialrust as sr

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import run_tum_wide_rgbd as runner
import run_tum_rgbd_odometry as original
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256,prepare
from test_tum_sequence_reference import fixture


def plan_path(scene,method):
    return runner.PLAN_ROOT/('tum-fr1-'+scene+'-'+method+'-wide-capture-plan.json')


@pytest.mark.parametrize('scene',['xyz','rpy'])
@pytest.mark.parametrize('method',['rgbd','adaptive'])
def test_four_plans_only_expand_the_fixed_capture_geometry(scene,method):
    plan=json.loads(plan_path(scene,method).read_text(encoding='utf-8'));before=copy.deepcopy(plan)
    assert runner.validate_wide_rgbd_plan(plan)==method
    assert plan==before and runner.WIDE_CONTROLS['window_radius']==10 and runner.WIDE_CONTROLS['pyramid_levels']==4
    parent=json.loads(runner.parent_plan(scene,method).read_text(encoding='utf-8'))
    expected=dict(parent['controls']['rgbd_odometry'],window_radius=10,pyramid_levels=4)
    assert plan['controls']['rgbd_odometry']==expected
    assert original.DEFAULT_CONTROLS['window_radius']==5 and original.DEFAULT_CONTROLS['pyramid_levels']==3


@pytest.mark.parametrize('method',['rgbd','adaptive'])
@pytest.mark.parametrize('case',['radius','levels_type','inliers','roundtrip','support','calibration','coverage','prior'])
def test_plan_rejects_changed_capture_identity_or_weakened_admission(method,case):
    plan=json.loads(plan_path('xyz',method).read_text(encoding='utf-8'))
    if case=='radius':plan['controls']['rgbd_odometry']['window_radius']=9
    if case=='levels_type':plan['controls']['rgbd_odometry']['pyramid_levels']=4.
    if case=='inliers':plan['controls']['rgbd_odometry']['min_inlier_fraction']=.4
    if case=='roundtrip':plan['controls']['rgbd_odometry']['max_forward_backward_error']=1.
    if case=='support':plan['controls']['min_support_fraction']=.4
    if case=='calibration':plan['calibration']['fx']=526.
    if case=='coverage':plan['acceptance']['minimum_evaluated_fraction']=.9
    if case=='prior':plan['prior_consensus_plan_sha256']='0'*64
    with pytest.raises(ValueError):runner.validate_wide_rgbd_plan(plan)


@pytest.mark.parametrize('method',['rgbd','adaptive'])
def test_wide_controls_reach_native_tracking_and_preserve_pose_direction_and_ids(monkeypatch,method):
    plan=json.loads(plan_path('xyz',method).read_text(encoding='utf-8'));controls=copy.deepcopy(plan['controls']);camera=plan['calibration']
    pixels=np.array([[80.+i%10*40,80.+i//10*40] for i in range(70)],np.float64)
    class Point:
        def __init__(self,p):self.x,self.y=p
    def detect(*a,**kw):
        assert kw['max_corners']==(300 if method=='rgbd' else 900)
        return [Point(p) for p in pixels]
    captured=[]
    def track(a,b,p,**kw):
        captured.append((p.copy(),dict(kw)));valid=np.ones(len(p),bool)
        if len(captured)==1:valid[[0,3]]=False
        return p+np.array([.25,.125]),valid,np.zeros(len(p))
    sources=[];poses=[]
    def pnp(depth,source,target,*a,**kw):
        assert kw==dict(depth_scale=1.,threshold=2.)
        sources.append(source.copy());np.testing.assert_allclose(depth,1.,rtol=0,atol=0)
        theta=.02*len(sources);c,s=np.cos(theta),np.sin(theta);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        translation=np.array([.001,.002,.003]);pose=np.eye(4);pose[:3,:3]=rotation;pose[:3,3]=translation;poses.append(pose)
        mask=np.zeros(len(source),bool);mask[:60]=True
        return rotation,translation,mask,0
    monkeypatch.setattr(sr,'shi_tomasi_keypoints',detect);monkeypatch.setattr(sr,'track_points_forward_backward',track);monkeypatch.setattr(sr,'estimate_rgbd_odometry',pnp)
    depth=np.full((480,640),5000,np.uint16);images=[np.zeros((480,640),np.uint8) for _ in range(3)];before=depth.copy()
    tracker=runner.WideAdaptiveRgbdTracker();adapter=runner.align_wide_rgbd if method=='rgbd' else tracker.align
    first,report=adapter(depth,images[0],images[1],camera,controls)
    np.testing.assert_allclose(first,np.linalg.inv(poses[0]),rtol=0,atol=1e-12)
    np.testing.assert_array_equal(sources[0],np.delete(pixels,[0,3],axis=0))
    assert report['inlier_count']==60 and report['rejected_depth_count']==0
    assert captured[0][1]['window_radius']==10 and captured[0][1]['pyramid_levels']==4
    assert captured[0][1]['max_forward_backward_error']==.75
    if method=='adaptive':
        second,report=adapter(depth,images[1],images[2],camera,controls)
        np.testing.assert_allclose(first@second,np.linalg.inv(poses[1]),rtol=0,atol=1e-12)
        np.testing.assert_array_equal(sources[1],sources[0][:60]);assert not report['early_keyframe_promotion'] and tracker.pose_ordinal==2
        np.testing.assert_allclose(captured[1][0],sources[0][:60]+[.25,.125],rtol=0,atol=0)
    np.testing.assert_array_equal(depth,before)
    assert controls==plan['controls'] and original.DEFAULT_CONTROLS['window_radius']==5
    with pytest.raises(ValueError):original.align_rgbd(depth,images[0],images[1],camera,controls)


@pytest.mark.parametrize('method',['rgbd','adaptive'])
def test_complete_wide_timeline_preserves_image_bindings_and_lost_chain(tmp_path,method):
    archive,sha=fixture(tmp_path,count=3);prepared=tmp_path/'prepared';plan=plan_path('xyz',method)
    manifest=prepare(archive,sha,plan,prepared);estimator=runner.ESTIMATORS[method]
    with pytest.raises(ValueError):generate(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'missing',method=estimator)
    def fail(*a):raise ValueError('fixed wide test tracking lost')
    result=generate(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result',rgbd_aligner=fail,method=estimator,additional_bindings=[__file__])
    rows=json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert result['planned_poses']==3 and result['generated_poses']==1 and result['tracking_lost']
    assert [r['status'] for r in rows]==['success','error','error'] and 'earlier frame' in rows[2]['error']
    assert rows[0]['rgb_sha256']==manifest['frames'][0]['rgb']['sha256']


def test_native_wide_tracking_keeps_known_motion_and_owned_full_rows():
    y,x=np.mgrid[:480,:640].astype(float)
    def texture(dx,dy):
        a,b=x-dx,y-dy
        return np.round(120+35*np.sin(a*.17)+25*np.cos(b*.21)+20*np.sin(a*.11+b*.13)).clip(0,255).astype(np.uint8)
    previous,following=texture(0,0),texture(8.,-6.)
    points=np.array([[180.,180.],[300.,220.],[440.,300.]])
    before=[v.copy() for v in [previous,following,points]]
    result=sr.track_points_forward_backward(previous,following,points,window_radius=10,pyramid_levels=4,max_forward_backward_error=.75,min_eigenvalue=.0001)
    coordinates,valid,errors=result;np.testing.assert_array_equal(valid,True);np.testing.assert_allclose(coordinates,points+[8.,-6.],rtol=0,atol=.2)
    assert np.all(errors<=.75)
    for current,original_input in zip([previous,following,points],before):np.testing.assert_array_equal(current,original_input)
    snapshots=[v.copy() for v in result];previous[:]=0;following[:]=0;points[:]=0
    for current,saved in zip(result,snapshots):np.testing.assert_array_equal(current,saved)
@pytest.mark.parametrize('method',['rgbd','adaptive'])
def test_explicit_reuse_checks_exact_parent_and_preserves_all_payloads(tmp_path,monkeypatch,method):
    import reuse_tum_prepared as reuse
    import shutil
    monkeypatch.setattr(reuse.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(40*1024**3,1,30*1024**3))
    archive,sha=fixture(tmp_path,count=3);source=tmp_path/'source';output=tmp_path/'wide'
    prepare(archive,sha,runner.parent_plan('xyz',method),source);bound=file_sha256(source/'manifest.json');before=(source/'manifest.json').read_bytes()
    with pytest.raises(ValueError,match='controls cannot change'):reuse.reuse(source,bound,plan_path('xyz',method),output)
    assert not output.exists()
    result=reuse.reuse(source,bound,plan_path('xyz',method),output,capture_revision=True)
    assert result['reused_input']['new_image_payload_bytes']==0 and (source/'manifest.json').read_bytes()==before
    for name,entry in result['files'].items():assert (output/name).samefile(source/name) and file_sha256(output/name)==entry['sha256']
    assert file_sha256(source/'manifest.json')==bound
    rule=Path(runner.__file__).with_name('wide_rgbd_protocol.py').resolve()
    assert result['bindings'][str(rule)]==file_sha256(rule)


@pytest.mark.parametrize('case',['parent','support','integer_type','frontend'])
def test_capture_reuse_rejects_unbound_or_loosened_plan_before_creation(tmp_path,monkeypatch,case):
    import reuse_tum_prepared as reuse
    import shutil
    monkeypatch.setattr(reuse.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(40*1024**3,1,30*1024**3))
    archive,sha=fixture(tmp_path,count=2);source=tmp_path/'source';output=tmp_path/'wide';prepare(archive,sha,runner.parent_plan('xyz','rgbd'),source)
    plan=json.loads(plan_path('xyz','rgbd').read_text(encoding='utf-8'))
    if case=='parent':plan['prior_consensus_plan_sha256']='0'*64
    if case=='support':plan['controls']['min_support_fraction']=.1
    if case=='integer_type':plan['controls']['rgbd_odometry']['window_radius']=10.
    if case=='frontend':plan['estimator']=runner.ESTIMATORS['adaptive']
    changed=tmp_path/'changed.json';changed.write_text(json.dumps(plan),encoding='utf-8')
    with pytest.raises(ValueError):reuse.reuse(source,file_sha256(source/'manifest.json'),changed,output,capture_revision=True)
    assert not output.exists()