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
from tum_sequence_reference import (associate_rgb, file_sha256, json_bytes, prepare,
                                    read_image_index, safe_relative, validate_plan)

PLAN = Path(__file__).resolve().parents[3]/'benchmarks/tum-fr1-xyz-plan.json'
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
