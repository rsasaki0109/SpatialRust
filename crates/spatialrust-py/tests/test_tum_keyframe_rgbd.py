"""Persistent depth/pixel identity, metric pose direction, state failure and hard-link IO."""
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import run_tum_keyframe_rgbd as runner
import reuse_tum_prepared as reuse_module
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256, prepare
from test_tum_sequence_reference import fixture

PLAN = runner.PARENT_PLAN.with_name('tum-fr1-xyz-keyframe-rgbd-pnp-plan.json')
CAMERA = dict(width=640, height=480, fx=525., fy=525., cx=319.5, cy=239.5, depth_unit_m=.0002)
CONTROLS = dict(rgbd_odometry=runner.DEFAULT_CONTROLS, rgbd_keyframe=runner.KEYFRAME_CONTROLS,
                min_depth_m=.1, max_depth_m=5.)
PIXELS = np.array([[70.+i%5*90, 70.+i//5*65] for i in range(30)], dtype=np.float64)


def fake_features(monkeypatch):
    class Point:
        def __init__(self, point): self.x, self.y = point
    monkeypatch.setattr(runner.sr, 'shi_tomasi_keypoints', lambda *args, **kw: [Point(p) for p in PIXELS])


def pose(axis, angle, shift):
    unit = np.eye(3)[axis]; x,y,z=unit*np.sin(angle/2); w=np.cos(angle/2)
    p = np.eye(4)
    p[:3,:3] = [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]
    p[:3,3] = shift
    return p


@pytest.mark.parametrize('protocol',[PLAN,PLAN.with_name('tum-fr1-rpy-keyframe-rgbd-pnp-plan.json')])
def test_fixed_plan_preserves_parent_accuracy_geometry_and_typed_controls(protocol):
    plan = json.loads(protocol.read_text(encoding='utf-8')); runner.validate_keyframe_rgbd_plan(plan)
    for field,value in [('interval',15),('interval_type',30.),('threshold',4.),('ate',.2),('depth',10.),('features_type',300.)]:
        bad = copy.deepcopy(plan)
        if field.startswith('interval'): bad['controls']['rgbd_keyframe']['interval_frames']=value
        if field=='threshold': bad['controls']['rgbd_odometry']['pnp_threshold']=value
        if field=='ate': bad['acceptance']['maximum_first_pose_ate_rmse_m']=value
        if field=='depth': bad['controls']['max_depth_m']=value
        if field=='features_type': bad['controls']['rgbd_odometry']['max_features']=value
        with pytest.raises(ValueError): runner.validate_keyframe_rgbd_plan(bad)


def test_native_pnp_uses_original_metric_landmarks_and_noncommuting_deltas(monkeypatch):
    fake_features(monkeypatch)
    depth = np.zeros((480,640), np.uint16)
    z = 1. + np.arange(len(PIXELS))*.037
    depth[PIXELS[:,1].astype(int),PIXELS[:,0].astype(int)] = np.floor(z*5000+.5).astype(np.uint16)
    z = depth[PIXELS[:,1].astype(int),PIXELS[:,0].astype(int)].astype(float)*.0002
    objects = np.column_stack(((PIXELS[:,0]-319.5)*z/525, (PIXELS[:,1]-239.5)*z/525,z))
    transforms = [pose(1,.025,[.012,-.004,.003]), pose(2,-.018,[.019,.007,-.002])]
    calls=iter(transforms)
    def track(previous,current,pixels,**kw):
        transform=next(calls); moved=objects@transform[:3,:3].T+transform[:3,3]
        following=np.column_stack((525*moved[:,0]/moved[:,2]+319.5,525*moved[:,1]/moved[:,2]+239.5))
        return following, np.ones(len(pixels),bool), np.zeros(len(pixels))
    monkeypatch.setattr(runner.sr,'track_points_forward_backward',track)
    tracker=runner.KeyframeRgbdTracker()
    gray0=np.zeros((480,640),np.uint8);gray1=gray0.copy();gray2=gray0.copy()
    first,_=tracker.align(depth,gray0,gray1,CAMERA,CONTROLS)
    # Changing later depth cannot change the retained source landmarks.
    second,report=tracker.align(np.zeros_like(depth),gray1,gray2,CAMERA,CONTROLS)
    np.testing.assert_allclose(first,np.linalg.inv(transforms[0]),rtol=0,atol=2e-5)
    np.testing.assert_allclose(second,transforms[0]@np.linalg.inv(transforms[1]),rtol=0,atol=2e-5)
    np.testing.assert_allclose(first@second,np.linalg.inv(transforms[1]),rtol=0,atol=2e-5)
    assert report['keyframe_pose_ordinal']==0 and report['keyframe_promoted_from_previous'] is False


def test_promotions_own_depth_and_keep_original_pixel_ids(monkeypatch):
    fake_features(monkeypatch)
    monkeypatch.setattr(runner.sr,'track_points_forward_backward',lambda a,b,p,**kw:(p+np.array([.1,.2]),np.ones(len(p),bool),np.zeros(len(p))))
    seen=[]
    def pnp(depth,source,target,*args,**kw):
        seen.append((depth.copy(),source.copy(),target.copy()))
        return np.eye(3),np.array([.01,0,0]),np.ones(len(source),bool),0
    monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',pnp)
    tracker=runner.KeyframeRgbdTracker(interval_frames=2)
    depth=np.full((480,640),5000,np.uint16);gray=np.zeros((480,640),np.uint8)
    next_gray=gray.copy();tracker.align(depth,gray,next_gray,CAMERA,CONTROLS);depth[:]=0
    third=gray.copy();tracker.align(np.zeros_like(depth),next_gray,third,CAMERA,CONTROLS)
    fourth=gray.copy();_,report=tracker.align(np.full_like(depth,10000),third,fourth,CAMERA,CONTROLS)
    np.testing.assert_array_equal(seen[0][1],seen[1][1])
    np.testing.assert_allclose(seen[1][2],PIXELS+np.array([.2,.4]),rtol=0,atol=1e-12)
    assert np.all(seen[0][0]==1.) and np.all(seen[1][0]==1.) and np.all(seen[2][0]==2.)
    assert report['keyframe_pose_ordinal']==2 and report['keyframe_promoted_from_previous'] is True


@pytest.mark.parametrize('case',['tracks','inliers','rejected_depth','rigidity','previous_image'])
def test_failed_update_cannot_promote_or_reset_the_chain(monkeypatch,case):
    fake_features(monkeypatch)
    monkeypatch.setattr(runner.sr,'track_points_forward_backward',lambda a,b,p,**kw:(p.copy(),np.ones(len(p),bool),np.zeros(len(p))))
    good=lambda d,s,t,*a,**k:(np.eye(3),np.zeros(3),np.ones(len(s),bool),0)
    monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',good)
    tracker=runner.KeyframeRgbdTracker();depth=np.full((480,640),5000,np.uint16)
    first=np.zeros((480,640),np.uint8);second=first.copy();tracker.align(depth,first,second,CAMERA,CONTROLS)
    if case=='tracks': monkeypatch.setattr(runner.sr,'track_points_forward_backward',lambda a,b,p,**kw:(p,np.zeros(len(p),bool),np.zeros(len(p))))
    if case=='inliers': monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',lambda d,s,t,*a,**k:(np.eye(3),np.zeros(3),np.zeros(len(s),bool),0))
    if case=='rejected_depth': monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',lambda d,s,t,*a,**k:(np.eye(3),np.zeros(3),np.ones(len(s),bool),1))
    if case=='rigidity': monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',lambda d,s,t,*a,**k:(np.zeros((3,3)),np.zeros(3),np.ones(len(s),bool),0))
    with pytest.raises(ValueError): tracker.align(depth,second.copy() if case=='previous_image' else second,first,CAMERA,CONTROLS)
    assert tracker.lost and tracker.pose_ordinal==1 and tracker.keyframe_ordinal==0
    with pytest.raises(ValueError,match='resets forbidden'): tracker.align(depth,second,first,CAMERA,CONTROLS)


@pytest.fixture
def reuse_fixture(tmp_path,monkeypatch):
    archive,sha=fixture(tmp_path,count=3);source=tmp_path/'prepared'
    prepare(archive,sha,runner.PARENT_PLAN,source)
    monkeypatch.setattr(reuse_module.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(40*1024**3,1,30*1024**3))
    return source,file_sha256(source/'manifest.json'),tmp_path/'reused'


def test_reuse_shares_payloads_preserves_source_and_all_failed_rows(reuse_fixture):
    source,sha,output=reuse_fixture
    before=(source/'manifest.json').read_bytes()
    result=reuse_module.reuse(source,sha,PLAN,output)
    assert (source/'manifest.json').read_bytes()==before
    assert result['reused_input']['new_image_payload_bytes']==0
    for name,entry in result['files'].items():
        assert (output/name).samefile(source/name) and file_sha256(output/name)==entry['sha256']
    assert not list(output.glob('groundtruth*'))
    with pytest.raises(ValueError): reuse_module.reuse(source,sha,PLAN,output)
    tracker=runner.KeyframeRgbdTracker()
    receipt=generate(output,file_sha256(output/'manifest.json'),output.parent/'result',
        rgbd_aligner=tracker.align,method=runner.ESTIMATOR,additional_bindings=[__file__])
    assert receipt['planned_poses']==3 and receipt['generated_poses']==1
    rows=json.loads((output.parent/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert rows[0]['identity_anchor'] and 'earlier frame' in rows[2]['error']


@pytest.mark.parametrize('case',['manifest','payload','low_space','nested','plan'])
def test_reuse_rejects_changed_identity_or_storage_before_creation(reuse_fixture,monkeypatch,case):
    source,sha,output=reuse_fixture
    plan=PLAN
    if case=='manifest': sha='0'*64
    if case=='payload':
        entry=json.loads((source/'manifest.json').read_text())['frames'][0]['depth'];(source/entry['path']).write_bytes(b'changed')
    if case=='low_space': monkeypatch.setattr(reuse_module.shutil,'disk_usage',lambda _:shutil._ntuple_diskusage(1,1,0))
    if case=='nested': output=source/'nested'
    if case=='plan':
        bad=json.loads(PLAN.read_text(encoding='utf-8'));bad['controls']['min_support_fraction']=.1
        plan=output.parent/'bad-plan.json';plan.write_text(json.dumps(bad),encoding='utf-8')
    with pytest.raises(ValueError): reuse_module.reuse(source,sha,plan,output)
    assert not output.exists()


def test_link_failure_preserves_inventory_without_copy_fallback(reuse_fixture,monkeypatch):
    source,sha,output=reuse_fixture
    original=reuse_module.os.link;calls=[]
    def fail_second(a,b):
        calls.append((a,b))
        if len(calls)==2: raise OSError('test link failure')
        return original(a,b)
    monkeypatch.setattr(reuse_module.os,'link',fail_second)
    with pytest.raises(OSError): reuse_module.reuse(source,sha,PLAN,output)
    partial=json.loads((output/'reuse.partial.json').read_text())
    assert partial['accepted'] is False and len(partial['files'])==1
    assert not (output/'manifest.json').exists()


def test_partial_track_and_pnp_masks_keep_the_same_landmark_ids(monkeypatch):
    fake_features(monkeypatch)
    valid=np.ones(len(PIXELS),bool);valid[[0,3,7]]=False
    first_inliers=np.ones(int(valid.sum()),bool);first_inliers[[1,5]]=False
    tracked_inputs=[];sources=[]
    def track(a,b,p,**kw):
        tracked_inputs.append(p.copy())
        mask=valid if len(tracked_inputs)==1 else np.ones(len(p),bool)
        return p+np.array([.1,.2]),mask,np.zeros(len(p))
    def pnp(d,s,t,*a,**kw):
        sources.append(s.copy())
        mask=first_inliers if len(sources)==1 else np.ones(len(s),bool)
        return np.eye(3),np.zeros(3),mask,0
    monkeypatch.setattr(runner.sr,'track_points_forward_backward',track)
    monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',pnp)
    depth=np.full((480,640),5000,np.uint16);images=[np.zeros((480,640),np.uint8) for _ in range(3)]
    tracker=runner.KeyframeRgbdTracker()
    tracker.align(depth,images[0],images[1],CAMERA,CONTROLS)
    tracker.align(depth,images[1],images[2],CAMERA,CONTROLS)
    expected=PIXELS[valid][first_inliers]
    np.testing.assert_array_equal(sources[1],expected)
    np.testing.assert_allclose(tracked_inputs[1],expected+np.array([.1,.2]),rtol=0,atol=1e-12)
    assert tracker.pose_ordinal==2 and not tracker.lost
