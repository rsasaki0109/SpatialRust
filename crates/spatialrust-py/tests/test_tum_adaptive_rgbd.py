"""Exact depth-count consensus, bounded candidate selection and input-only promotion."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import run_tum_adaptive_rgbd as runner
from test_tum_keyframe_rgbd import CAMERA, CONTROLS as BASE_CONTROLS, pose

CONTROLS=dict(BASE_CONTROLS,rgbd_adaptive_keyframe=runner.CONTROLS)
PIXELS=np.array([[70.+i%10*50,70.+i//10*50] for i in range(70)],dtype=np.float64)


def features(monkeypatch,pixels=PIXELS):
    class Point:
        def __init__(self,p):self.x,self.y=p
    def detect(*args,**kw):
        assert kw['max_corners']==900
        return [Point(p) for p in pixels]
    monkeypatch.setattr(runner.sr,'shi_tomasi_keypoints',detect)


@pytest.mark.parametrize('scene',['xyz','rpy'])
def test_separate_protocol_keeps_coverage_and_original_inlier_gates(scene):
    path=runner.PLAN_ROOT/('tum-fr1-'+scene+'-adaptive-rgbd-pnp-plan.json')
    plan=json.loads(path.read_text(encoding='utf-8'));runner.validate_adaptive_rgbd_plan(plan)
    for case in ['candidates','count_type','deviation','tracks','inliers','ate','prior']:
        bad=copy.deepcopy(plan)
        if case=='candidates':bad['controls']['rgbd_adaptive_keyframe']['maximum_candidate_features']=901
        if case=='count_type':bad['controls']['rgbd_adaptive_keyframe']['minimum_retained_tracks']=60.
        if case=='deviation':bad['controls']['rgbd_adaptive_keyframe']['absolute_deviation_depth_counts']=101
        if case=='tracks':bad['controls']['rgbd_odometry']['min_tracks']=10
        if case=='inliers':bad['controls']['rgbd_odometry']['min_inlier_fraction']=.4
        if case=='ate':bad['acceptance']['maximum_first_pose_ate_rmse_m']=.2
        if case=='prior':bad['prior_fixed_keyframe_plan_sha256']='0'*64
        with pytest.raises(ValueError):runner.validate_adaptive_rgbd_plan(bad)


@pytest.mark.parametrize('median,deviation,admitted',[(5000,100,True),(5000,101,False),(10000,200,True),(10000,201,False),(500,0,True),(25000,0,True)])
def test_integer_consensus_closed_boundaries_and_owned_median(monkeypatch,median,deviation,admitted):
    features(monkeypatch)
    depth=np.full((480,640),median,np.uint16);depth[70,70]=median+deviation;before=depth.copy()
    tracker=runner.AdaptiveRgbdTracker()
    metres,points=tracker.seed(depth,np.zeros((480,640),np.uint8),CAMERA,CONTROLS)
    assert any(np.array_equal(p,PIXELS[0]) for p in points)==admitted
    if admitted:assert metres[70,70]==np.float32(median*.0002)
    np.testing.assert_array_equal(depth,before)
    assert not np.shares_memory(depth,metres)


@pytest.mark.parametrize('bad_count',[0,499,25001])
def test_local_holes_and_out_of_range_neighbors_are_rejected(monkeypatch,bad_count):
    features(monkeypatch)
    depth=np.full((480,640),5000,np.uint16);depth[69,69]=bad_count
    _,points=runner.AdaptiveRgbdTracker().seed(depth,np.zeros((480,640),np.uint8),CAMERA,CONTROLS)
    assert not any(np.array_equal(p,PIXELS[0]) for p in points)
    assert len(points)==len(PIXELS)-1


def test_source_depth_selection_keeps_detector_order_and_300_landmark_budget(monkeypatch):
    pixels=np.array([[43.+i%30*18,43.+i//30*24] for i in range(450)],dtype=np.float64)
    features(monkeypatch,pixels)
    tracker=runner.AdaptiveRgbdTracker()
    _,points=tracker.seed(np.full((480,640),5000,np.uint16),np.zeros((480,640),np.uint8),CAMERA,CONTROLS)
    np.testing.assert_array_equal(points,pixels[:300])
    assert tracker.seed_statistics['depth_consistent_candidates']==450
    assert tracker.seed_statistics['retained_features']==300


@pytest.mark.parametrize('retained',[59,60])
def test_early_promotion_preserves_metric_world_composition(monkeypatch,retained):
    features(monkeypatch)
    monkeypatch.setattr(runner.sr,'track_points_forward_backward',lambda a,b,p,**kw:(p.copy(),np.ones(len(p),bool),np.zeros(len(p))))
    first_pose=pose(2,.03,[.01,.02,.003]);second_pose=pose(1,-.02,[-.004,.015,.002])
    seen=[]
    def pnp(depth,source,target,*a,**kw):
        seen.append(depth.copy());transform=first_pose if len(seen)==1 else second_pose
        inliers=np.ones(len(source),bool)
        if len(seen)==1:inliers[retained:]=False
        return transform[:3,:3],transform[:3,3],inliers,0
    monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',pnp)
    tracker=runner.AdaptiveRgbdTracker();images=[np.zeros((480,640),np.uint8) for _ in range(3)]
    first,_=tracker.align(np.full((480,640),5000,np.uint16),images[0],images[1],CAMERA,CONTROLS)
    second,report=tracker.align(np.full((480,640),10000,np.uint16),images[1],images[2],CAMERA,CONTROLS)
    early=retained<60
    expected=np.linalg.inv(first_pose)@np.linalg.inv(second_pose) if early else np.linalg.inv(second_pose)
    np.testing.assert_allclose(first@second,expected,rtol=0,atol=1e-12)
    assert report['early_keyframe_promotion'] is early
    assert report['keyframe_pose_ordinal']==(1 if early else 0)
    assert np.all(seen[1]==(2. if early else 1.))
    assert tracker.pose_ordinal==2 and not tracker.lost


def test_failed_early_promotion_keeps_prior_state_and_lost_chain(monkeypatch):
    features(monkeypatch)
    monkeypatch.setattr(runner.sr,'track_points_forward_backward',lambda a,b,p,**kw:(p.copy(),np.ones(len(p),bool),np.zeros(len(p))))
    def pnp(depth,source,target,*a,**kw):
        inliers=np.zeros(len(source),bool);inliers[:59]=True
        return np.eye(3),np.zeros(3),inliers,0
    monkeypatch.setattr(runner.sr,'estimate_rgbd_odometry',pnp)
    tracker=runner.AdaptiveRgbdTracker();a=np.zeros((480,640),np.uint8);b=a.copy()
    tracker.align(np.full((480,640),5000,np.uint16),a,b,CAMERA,CONTROLS);old=tracker.keyframe_depth
    with pytest.raises(ValueError,match='stable-depth'):
        tracker.align(np.zeros((480,640),np.uint16),b,a,CAMERA,CONTROLS)
    assert tracker.lost and tracker.keyframe_depth is old and tracker.pose_ordinal==1 and tracker.keyframe_ordinal==0
    with pytest.raises(ValueError,match='resets forbidden'):tracker.align(np.full((480,640),5000,np.uint16),b,a,CAMERA,CONTROLS)


def test_insufficient_consensus_has_no_raw_depth_fallback(monkeypatch):
    features(monkeypatch,PIXELS[:19])
    tracker=runner.AdaptiveRgbdTracker()
    with pytest.raises(ValueError,match='require 20'):
        tracker.align(np.full((480,640),5000,np.uint16),np.zeros((480,640),np.uint8),np.zeros((480,640),np.uint8),CAMERA,CONTROLS)
    assert tracker.lost and tracker.pose_ordinal==0


def test_native_candidate_detection_feeds_bounded_owned_depth_landmarks():
    rows,columns=np.indices((480,640))
    image=np.ascontiguousarray((((rows//20+columns//20)%2)*255).astype(np.uint8))
    depth=np.full((480,640),10000,np.uint16)
    before=image.copy()
    tracker=runner.AdaptiveRgbdTracker()
    metres,points=tracker.seed(depth,image,CAMERA,CONTROLS)
    assert 20<=len(points)<=300
    assert tracker.seed_statistics['detected_candidates']<=900
    assert tracker.seed_statistics['retained_features']==len(points)
    assert points.dtype==np.float64 and points.flags.c_contiguous
    assert metres.dtype==np.float32 and np.all(metres==2.)
    np.testing.assert_array_equal(image,before)
    assert np.all(depth==10000)
