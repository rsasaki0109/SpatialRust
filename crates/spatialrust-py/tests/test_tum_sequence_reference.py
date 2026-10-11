"""Full timeline, archive isolation, camera chaining and frozen post-fit checks."""
import copy
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import zlib

import numpy as np
import pytest
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from evaluate_tum_sequence import score
from run_tum_sequence import TrackingChain, align_pair, checked_depth, compose_camera_pose, decode_depth, generate, project
from run_tum_point_to_plane import align_point_to_plane, validate_plane_plan
from run_tum_keyframe import ESTIMATOR as KEYFRAME_ESTIMATOR, KeyframeTracker, validate_keyframe_plan
from run_tum_adaptive_keyframe import ESTIMATOR as OVERLAP_ESTIMATOR, OverlapKeyframeTracker, validate_overlap_plan
from tum_sequence_reference import (associate_rgb, file_sha256, json_bytes, prepare,
                                    read_image_index, safe_relative, validate_plan)

PLAN = Path(__file__).resolve().parents[3]/'benchmarks/tum-fr1-xyz-plan.json'
PLANE_PLAN = PLAN.with_name('tum-fr1-xyz-plane-plan.json')
ROOT = 'rgbd_dataset_freiburg1_xyz'


def png(depth=True, count=5000):
    bits, color, channels = (16, 0, 2) if depth else (8, 2, 3)
    pixel = struct.pack('>H', count) if depth else b'\x00\x10\x20'
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data)&0xffffffff)
    header = struct.pack('>IIBBBBB', 640, 480, bits, color, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', header)+chunk(b'IDAT', zlib.compress((b'\0'+pixel*640)*480))+chunk(b'IEND', b'')


def fixture(tmp_path, count=3, failure_index=None, extra=None, omit=None):
    files, depth, rgb, truth = {}, [], [], []
    for i in range(count):
        timestamp = '1607987782.{:09d}'.format(i*10000000+1)
        for kind, index in (('depth', depth), ('rgb', rgb)):
            name = kind+'/'+timestamp+'.png'
            index.append(timestamp+' '+name)
            files[name] = png(kind == 'depth', 0 if i == failure_index else 5000)
        truth.append(timestamp+' 0 0 0 0 0 0 1')
    files['depth.txt'] = ('\n'.join(depth)+'\n').encode()
    files['rgb.txt'] = ('\n'.join(rgb)+'\n').encode()
    files['groundtruth.txt'] = ('\n'.join(truth)+'\n').encode()
    if omit:
        del files[omit]
    archive = tmp_path/'source.tgz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(ROOT+'/'+name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            info, data = extra
            tar.addfile(info, io.BytesIO(data))
    return archive, file_sha256(archive)


def test_complete_sequence_beyond_sample_cap_keeps_integer_times_and_no_truth(tmp_path):
    archive, sha = fixture(tmp_path, count=40)
    out = tmp_path/'prepared'
    manifest = prepare(archive, sha, PLAN, out)
    assert manifest['depth_frames'] == manifest['rgb_frames'] == 40
    assert manifest['frames'][0]['timestamp_ns'] == 1607987782000000001
    assert manifest['frames'][-1]['timestamp_ns'] == 1607987782390000001
    assert not (out/'groundtruth.txt').exists()
    assert manifest['withheld_members'][0]['excluded_from_generation'] is True
    assert manifest['reference_used_for_generation'] is False
    assert all(f['depth']['sha256'] == file_sha256(out/f['depth']['path']) for f in manifest['frames'])
    frozen = (out/'manifest.json').read_bytes()
    with pytest.raises(FileExistsError): prepare(archive, sha, PLAN, out)
    assert (out/'manifest.json').read_bytes() == frozen
    with pytest.raises(ValueError): prepare(archive, 'a'*64, PLAN, tmp_path/'tampered')
    assert not (tmp_path/'tampered').exists()


@pytest.mark.parametrize('path', ['../escape', '/absolute', 'depth/../escape.png', 'depth\\a.png',
                                'depth//a.png', './depth/a.png', 'depth/a\n.png'])
def test_paths_cannot_escape_or_alias(path):
    with pytest.raises(ValueError): safe_relative(path)


@pytest.mark.parametrize('kind', ['link', 'wrong_root', 'duplicate', 'member_budget', 'total_budget', 'unindexed'])
def test_archive_refuses_unsafe_or_incomplete_full_sequence(tmp_path, kind):
    info = tarfile.TarInfo(ROOT+'/depth/extra.png')
    data = png()
    plan = json.loads(PLAN.read_bytes())
    if kind == 'link':
        info.type = tarfile.SYMTYPE
        info.linkname = '/tmp/escape'
        data = b''
    elif kind == 'wrong_root': info.name = 'other/depth/extra.png'
    elif kind == 'duplicate': info.name = ROOT+'/depth.txt'
    elif kind == 'member_budget': plan['limits']['max_members'] = 1
    elif kind == 'total_budget': plan['limits']['max_total_bytes'] = 1
    info.size = len(data)
    archive, sha = fixture(tmp_path, count=1, extra=(info, data))
    plan_path = tmp_path/'plan.json'
    plan_path.write_bytes(json_bytes(plan))
    out = tmp_path/'prepared'
    with pytest.raises(ValueError): prepare(archive, sha, plan_path, out)
    assert not (out/'manifest.json').exists()
    assert not (tmp_path/'escape').exists()


def test_missing_png_and_nanosecond_rgb_ties_are_not_silently_dropped(tmp_path):
    archive, sha = fixture(tmp_path, count=1, omit='depth/1607987782.000000001.png')
    with pytest.raises(ValueError): prepare(archive, sha, PLAN, tmp_path/'prepared')
    rgb = [dict(timestamp_ns=10, path='rgb/a.png'), dict(timestamp_ns=12, path='rgb/b.png')]
    frames = associate_rgb([dict(timestamp_ns=11, path='depth/x.png'), dict(timestamp_ns=20, path='depth/y.png')], rgb, 1)
    assert frames[0]['rgb']['timestamp_ns'] == 10 and frames[0]['rgb_delta_ns'] == -1
    assert frames[1]['rgb'] is None and len(frames) == 2


@pytest.mark.parametrize('data', [b'1 depth/a.png\n1 depth/b.png\n', b'1 depth/a.png\n2 depth/a.png\n',
                                 b'1 rgb/a.png\n', b'0.0000000001 depth/a.png\n'])
def test_image_indexes_reject_duplicate_time_path_or_inexact_time(data):
    with pytest.raises(ValueError): read_image_index(data, 'depth', 10000)


@pytest.mark.parametrize('kind', ['unit', 'frame', 'extrinsic', 'budget', 'nan', 'bool', 'stage'])
def test_protocol_rejects_false_calibration_and_unbounded_controls(kind):
    plan = copy.deepcopy(json.loads(PLAN.read_bytes()))
    if kind == 'unit': plan['calibration']['depth_unit_m'] = .001
    if kind == 'frame': plan['calibration']['sensor_frame'] = 'depth_ir'
    if kind == 'extrinsic': plan['calibration']['extrinsic_applied'] = True
    if kind == 'budget': plan['limits']['max_frames'] = 10001
    if kind == 'nan': plan['controls']['max_depth_m'] = float('nan')
    if kind == 'bool': plan['controls']['min_points'] = True
    if kind == 'stage': plan['controls']['stages'][1]['leaf_m'] = .2
    with pytest.raises(ValueError): validate_plan(plan)


def test_camera_composition_order_and_tracking_loss_never_reset():
    first = np.array([[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]], float)
    second = np.eye(4); second[0,3] = 4
    np.testing.assert_allclose(compose_camera_pose(first, second), first @ second, atol=1e-15)
    chain = TrackingChain()
    chain.fail()  # Initial bad image does not establish a gauge.
    np.testing.assert_array_equal(chain.accept(), np.eye(4))
    np.testing.assert_allclose(chain.accept(first), first)
    np.testing.assert_allclose(chain.accept(second), first @ second)
    chain.fail()
    with pytest.raises(ValueError): chain.accept()
    with pytest.raises(ValueError): chain.accept(np.eye(4))


def test_native_projection_exact_depth_boundaries_and_current_to_previous_icp():
    plan = json.loads(PLAN.read_bytes())
    controls = copy.deepcopy(plan['controls']); controls['min_points'] = 1
    depth = np.zeros((480,640), np.uint16)
    depth[0,0], depth[1,0], depth[2,0] = 500, 25000, 25001
    cloud, xyz, valid, error = project(depth, plan['calibration'], controls, np.empty((480,640,3), np.float32))
    assert len(cloud) == 2 and depth[valid].tolist() == [500,25000] and error < 1e-6
    np.testing.assert_allclose(xyz[:,2], [.1,5], atol=2e-8, rtol=0)
    previous = np.random.RandomState(42).uniform(-.5,.5,(80,3)).astype(np.float32)
    delta = np.eye(4); delta[:3,3] = [.002,-.003,.001]
    current = previous-delta[:3,3].astype(np.float32)
    controls['stages'] = [dict(leaf_m=.005,gate_m=.05,iterations=20)]
    controls['min_support_fraction'] = .99
    transform, report = align_pair(sr.PointCloud.from_xyz(current),sr.PointCloud.from_xyz(previous),controls)
    np.testing.assert_allclose(transform, delta, atol=2e-6, rtol=0)
    assert report['support_fraction'] == 1


def test_changed_depth_cannot_satisfy_frozen_input_hash(tmp_path):
    archive, sha = fixture(tmp_path, count=1)
    out = tmp_path/'prepared'
    manifest = prepare(archive, sha, PLAN, out)
    entry = manifest['frames'][0]['depth']
    assert checked_depth(out, entry, 16777216)
    (out/entry['path']).write_bytes(b'changed')
    with pytest.raises(ValueError): checked_depth(out, entry, 16777216)


@pytest.mark.parametrize('method,adapter', [('unknown',None),('open3d_cpu',None),('spatialrust_point_to_plane_cpu',None),('spatialrust_keyframe_point_to_plane_cpu',None),('spatialrust_overlap_keyframe_point_to_plane_cpu',None),('spatialrust_cpu',lambda: None)])
def test_generation_method_cannot_silently_substitute_an_adapter(tmp_path,method,adapter):
    with pytest.raises(ValueError):
        generate(tmp_path/'missing-input','a'*64,tmp_path/'output',method=method,pair_aligner=adapter)
    assert not (tmp_path/'output').exists()


def test_depth_decoder_preserves_original_counts_and_rejects_color_or_broken_png():
    camera = json.loads(PLAN.read_bytes())['calibration']
    depth = decode_depth(png(count=65535), camera)
    assert depth.dtype == np.uint16 and depth.shape == (480,640) and np.all(depth == 65535)
    for raw in (png(depth=False), b'broken png'):
        with pytest.raises((ValueError, OSError)): decode_depth(raw, camera)


@pytest.mark.parametrize('failure_index', [None, 1])
def test_native_fixture_freezes_every_pose_retains_loss_then_scores_separately(tmp_path, failure_index):
    archive, sha = fixture(tmp_path, count=3, failure_index=failure_index)
    prepared = tmp_path/'prepared'
    prepare(archive, sha, PLAN, prepared)
    assert not (prepared/'groundtruth.txt').exists()
    output = tmp_path/'generated'
    receipt = generate(prepared, file_sha256(prepared/'manifest.json'), output)
    assert receipt['method'] == 'spatialrust_cpu'
    expected_count = 3 if failure_index is None else 1
    assert receipt['planned_poses'] == 3 and receipt['generated_poses'] == expected_count
    assert receipt['tracking_lost'] is (failure_index is not None)
    estimates = output/'estimates.json'
    frozen = estimates.read_bytes()
    poses = json.loads(frozen)['poses']
    assert [p['status'] for p in poses] == (['success']*3 if failure_index is None else ['success','error','error'])
    if failure_index is not None:
        assert 'earlier frame' in poses[2]['error']
    assert poses[0]['io']['exact_typed_attributes'] is True
    result = score(archive, estimates, receipt['estimates_sha256'], PLAN, tmp_path/'evaluated')
    assert (result['planned_poses'], result['evaluated_poses']) == (3,expected_count)
    assert result['ate_first_pose_aligned_rmse_m'] == 0  # Analytic fixture only.
    assert estimates.read_bytes() == frozen
    with pytest.raises(FileExistsError): generate(prepared, file_sha256(prepared/'manifest.json'), output)
    with pytest.raises(FileExistsError): score(archive, estimates, receipt['estimates_sha256'], PLAN, tmp_path/'evaluated')
    estimates.write_bytes(frozen+b' ')
    with pytest.raises(ValueError): score(archive, estimates, receipt['estimates_sha256'], PLAN, tmp_path/'changed')
    assert not (tmp_path/'changed').exists()


def test_point_to_plane_recovers_known_current_to_previous_motion():
    axis = np.linspace(.3, 1.5, 18)
    u, v = np.meshgrid(axis, axis)
    target = np.vstack((np.column_stack((u.ravel(), v.ravel(), np.full(u.size, 2.))),
                        np.column_stack((np.full(u.size, -.8), u.ravel(), v.ravel()+.7)),
                        np.column_stack((u.ravel(), np.full(u.size, -.8), v.ravel()+.7)))).astype(np.float32)
    angle = .015
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    translation = np.array([.022, -.016, .024])
    current = ((target-translation) @ rotation).astype(np.float32)
    controls = json.loads(PLANE_PLAN.read_bytes())['controls']
    pose, report = align_point_to_plane(sr.PointCloud.from_xyz(current), sr.PointCloud.from_xyz(target), controls)
    np.testing.assert_allclose(pose[:3, 3], translation, atol=.002, rtol=0)
    np.testing.assert_allclose(pose[:3, :3], rotation, atol=.002, rtol=0)
    assert report['support_fraction'] >= controls['min_support_fraction']
    assert all(0 < s['iterations'] <= 20 and s['normal_neighbors'] == 20 for s in report['stages'])


def test_point_to_plane_does_not_hide_unobservable_plane_motion():
    u, v = np.meshgrid(np.linspace(0, 2, 20), np.linspace(0, 2, 20))
    plane = sr.PointCloud.from_xyz(np.column_stack((u.ravel(), v.ravel(), np.full(u.size, 2.))).astype(np.float32))
    with pytest.raises(ValueError, match='singular'):
        align_point_to_plane(plane, plane, json.loads(PLANE_PLAN.read_bytes())['controls'])


@pytest.mark.parametrize('change', ['normal', 'calibration', 'controls', 'source_plan'])
def test_point_to_plane_rejects_unfrozen_settings(change):
    plan = json.loads(PLANE_PLAN.read_bytes())
    if change == 'normal': plan['estimator']['normal_neighbors'] = 30
    elif change == 'calibration': plan['calibration']['fx'] = 530.
    elif change == 'controls': plan['controls']['stages'][0]['iterations'] = 40
    else: plan['prior_native_plan_sha256'] = 'a'*64
    with pytest.raises(ValueError): validate_plane_plan(plan, PLAN)


def test_second_sequence_has_same_fixed_geometry_and_archive_isolation(tmp_path):
    holdout = json.loads(PLAN.with_name('tum-fr1-rpy-plane-plan.json').read_bytes())
    assert validate_plane_plan(holdout, PLAN)['sequence'] == 'rgbd_dataset_freiburg1_rpy'
    archive, sha = fixture(tmp_path, count=1)
    with pytest.raises(ValueError, match='wrong-root'):
        prepare(archive, sha, PLAN.with_name('tum-fr1-rpy-plane-plan.json'), tmp_path/'wrong-sequence')
    holdout['sequence'] = 'rgbd_dataset_freiburg3_xyz'
    with pytest.raises(ValueError): validate_plan(holdout)


def test_keyframe_native_motion_remains_in_one_world_gauge_across_promotions():
    axis = np.linspace(.3, 1.5, 18)
    u, v = np.meshgrid(axis, axis)
    world_points = np.vstack((np.column_stack((u.ravel(), v.ravel(), np.full(u.size, 2.))),
                              np.column_stack((np.full(u.size, -.8), u.ravel(), v.ravel()+.7)),
                              np.column_stack((u.ravel(), np.full(u.size, -.8), v.ravel()+.7)))).astype(np.float32)
    tracker, chain = KeyframeTracker(2), TrackingChain()
    previous = sr.PointCloud.from_xyz(world_points)
    chain.accept()
    controls = json.loads(PLANE_PLAN.read_bytes())['controls']
    targets = []
    for i in range(1, 5):
        angle = i*.012
        expected = np.eye(4)
        expected[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]]
        expected[:3, 3] = np.array([.018, -.011, .014])*i
        current_points = ((world_points-expected[:3, 3]) @ expected[:3, :3]).astype(np.float32)
        current = sr.PointCloud.from_xyz(current_points)
        delta, report = tracker.align_pair(current, previous, controls)
        actual = chain.accept(delta)
        np.testing.assert_allclose(actual, expected, atol=.005, rtol=0)
        assert report['current_pose_ordinal'] == i
        assert report['keyframe_promoted'] is (i % 2 == 0)
        assert report['support_fraction'] >= controls['min_support_fraction']
        assert 'keyframe_correction' in report['stages'][0]
        assert 'previous_from_current' not in report['stages'][0]
        targets.append(report['keyframe_pose_ordinal'])
        previous = current
    assert targets == [0, 0, 2, 2]
    assert tracker.keyframe is current and tracker.keyframe_ordinal == 4


def test_keyframe_failure_cannot_promote_or_restart_tracking(monkeypatch):
    import run_tum_keyframe as module
    tracker = KeyframeTracker(1)
    cloud = sr.PointCloud.from_xyz(np.ones((30, 3), np.float32))
    def singular(*args):
        raise ValueError('singular geometry')
    monkeypatch.setattr(module, 'align_point_to_plane', singular)
    with pytest.raises(ValueError, match='singular'):
        tracker.align_pair(cloud, cloud, {})
    assert tracker.lost and tracker.pose_ordinal == tracker.keyframe_ordinal == 0
    np.testing.assert_array_equal(tracker.keyframe_from_previous, np.eye(4))
    with pytest.raises(ValueError, match='resets forbidden'):
        tracker.align_pair(cloud, cloud, {})


@pytest.mark.parametrize('interval', [0, -1, True, 1.5, 1001])
def test_keyframe_interval_is_a_bounded_integer(interval):
    with pytest.raises(ValueError): KeyframeTracker(interval)


@pytest.mark.parametrize('change', ['interval', 'initialization', 'normal', 'calibration', 'controls', 'source_plan', 'acceptance'])
def test_keyframe_protocol_refuses_unfrozen_parameters(change):
    plan = json.loads(PLAN.with_name('tum-fr1-xyz-keyframe-plan.json').read_bytes())
    if change == 'interval': plan['estimator']['keyframe_interval_frames'] = 20
    elif change == 'initialization': plan['estimator']['initialization'] = 'reference'
    elif change == 'normal': plan['estimator']['normal_neighbors'] = 30
    elif change == 'calibration': plan['calibration']['fx'] = 530.
    elif change == 'controls': plan['controls']['stages'][0]['iterations'] = 40
    elif change == 'source_plan': plan['prior_native_plan_sha256'] = 'a'*64
    else: plan['acceptance']['maximum_first_pose_ate_rmse_m'] = .6
    with pytest.raises(ValueError): validate_keyframe_plan(plan, PLAN)


def test_desk_keyframe_sequence_is_declared_but_wrong_archive_is_rejected(tmp_path):
    path = PLAN.with_name('tum-fr1-desk-keyframe-plan.json')
    plan = json.loads(path.read_bytes())
    assert validate_keyframe_plan(plan, PLAN)['sequence'] == 'rgbd_dataset_freiburg1_desk'
    assert validate_plane_plan(json.loads(PLAN.with_name('tum-fr1-desk-plane-plan.json').read_bytes()), PLAN)
    archive, sha = fixture(tmp_path, count=1)
    with pytest.raises(ValueError, match='wrong-root'):
        prepare(archive, sha, path, tmp_path/'wrong-sequence')


def test_keyframe_frozen_generation_binds_both_helpers_and_keeps_all_lost_rows(tmp_path):
    archive, sha = fixture(tmp_path, count=3)
    prepared, output = tmp_path/'prepared', tmp_path/'generated'
    plan = PLAN.with_name('tum-fr1-xyz-keyframe-plan.json')
    prepare(archive, sha, plan, prepared)
    tracker = KeyframeTracker()
    helper = PLAN.parent.parent/'scripts/run_tum_point_to_plane.py'
    result = generate(prepared, file_sha256(prepared/'manifest.json'), output,
                      pair_aligner=tracker.align_pair, method=KEYFRAME_ESTIMATOR['method'],
                      additional_bindings=(helper,))
    assert result['method'] == KEYFRAME_ESTIMATOR['method']
    assert result['planned_poses'] == 3 and result['generated_poses'] == 1 and result['tracking_lost']
    poses = json.loads((output/'estimates.json').read_bytes())['poses']
    assert [p['status'] for p in poses] == ['success', 'error', 'error']
    assert 'singular' in poses[1]['error'] and 'earlier frame' in poses[2]['error']
    assert poses[0]['io']['exact_typed_attributes']
    assert result['bindings'][str(helper.resolve())] == file_sha256(helper)
    assert any(p.endswith('/run_tum_keyframe.py') for p in result['bindings'])
    assert not (prepared/'groundtruth.txt').exists()


@pytest.mark.parametrize('support,promoted', [(.749, True), (.75, False), (.751, False)])
def test_overlap_margin_promotes_only_a_valid_current_frame_without_changing_world(monkeypatch, support, promoted):
    import run_tum_keyframe as module
    tracker, chain = OverlapKeyframeTracker(), TrackingChain()
    chain.accept()
    def accepted(*args):
        delta = np.eye(4); delta[0, 3] = .01
        return delta, dict(stages=[dict(previous_from_current=delta.tolist())],
                           support_fraction=support, support_count=30, support_rmse_m=.01)
    monkeypatch.setattr(module, 'align_point_to_plane', accepted)
    previous = sr.PointCloud.from_xyz(np.ones((30, 3), np.float32))
    for i in range(1, 4):
        current = sr.PointCloud.from_xyz(np.ones((30, 3), np.float32))
        delta, report = tracker.align_pair(current, previous, {})
        actual = chain.accept(delta)
        np.testing.assert_allclose(actual[:3, 3], [.01*i, 0, 0], atol=1e-12, rtol=0)
        assert report['keyframe_promoted'] is promoted
        assert report['keyframe_promotion_cause'] == ('geometric_overlap_margin' if promoted else 'retained')
        if promoted:assert tracker.tracker.keyframe is current
        previous = current


def test_overlap_tracker_does_not_retry_a_failed_registration(monkeypatch):
    import run_tum_keyframe as module
    calls = []
    def unsupported(*args):
        calls.append(args)
        raise ValueError('support below fixed minimum')
    monkeypatch.setattr(module, 'align_point_to_plane', unsupported)
    tracker = OverlapKeyframeTracker()
    cloud = sr.PointCloud.from_xyz(np.ones((30, 3), np.float32))
    with pytest.raises(ValueError, match='fixed minimum'):tracker.align_pair(cloud, cloud, {})
    with pytest.raises(ValueError, match='resets forbidden'):tracker.align_pair(cloud, cloud, {})
    assert len(calls) == 1 and tracker.tracker.keyframe_ordinal == 0


@pytest.mark.parametrize('change', ['margin', 'minimum_support', 'acceptance', 'interval'])
def test_overlap_plan_cannot_relax_support_accuracy_or_change_promotion(change):
    p = json.loads(PLAN.with_name('tum-fr1-rpy-overlap-keyframe-plan.json').read_bytes())
    if change == 'margin':p['estimator']['promotion_support_fraction'] = .6
    elif change == 'minimum_support':p['controls']['min_support_fraction'] = .4
    elif change == 'acceptance':p['acceptance']['maximum_first_pose_ate_rmse_m'] = .6
    else:p['estimator']['keyframe_interval_frames'] = 20
    with pytest.raises(ValueError):validate_overlap_plan(p, PLAN)


def test_overlap_generation_binds_its_policy_and_both_native_helpers(tmp_path):
    archive, sha = fixture(tmp_path, count=3)
    prepared, output = tmp_path/'prepared', tmp_path/'generated'
    plan = PLAN.with_name('tum-fr1-xyz-overlap-keyframe-plan.json')
    assert validate_overlap_plan(json.loads(plan.read_bytes()), PLAN)
    prepare(archive, sha, plan, prepared)
    tracker = OverlapKeyframeTracker()
    helpers = [PLAN.parent.parent/('scripts/'+name) for name in ('run_tum_keyframe.py','run_tum_point_to_plane.py')]
    result = generate(prepared, file_sha256(prepared/'manifest.json'), output,
                      pair_aligner=tracker.align_pair, method=OVERLAP_ESTIMATOR['method'],
                      additional_bindings=helpers)
    assert result['method'] == OVERLAP_ESTIMATOR['method'] and result['generated_poses'] == 1
    assert result['planned_poses'] == 3 and result['tracking_lost']
    assert all(result['bindings'][str(h.resolve())] == file_sha256(h) for h in helpers)
    assert any(p.endswith('/run_tum_adaptive_keyframe.py') for p in result['bindings'])
    assert not (prepared/'groundtruth.txt').exists()
