"""Owned CPU comparator state, unchanged gates and independently checked raw warps."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_tum_dense_rgbd import audit, check_causal_initialization, reference_statistics
from dense_rgbd_protocol import CONTROLS as VALIDATION
from open3d_rgbd_protocol import CONTROLS, ESTIMATOR, PARENTS, validate_comparator_plan
from open3d_rgbd_validation import raw_statistics
from run_tum_open3d_rgbd import PreviousMotionOpen3dRgbd, owned_cpu_rgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256, prepare
from test_tum_sequence_reference import fixture

PLAN = ROOT/'benchmarks/tum-fr1-xyz-open3d-dense-plan.json'


@pytest.mark.parametrize('scene', ['xyz','rpy'])
@pytest.mark.parametrize('ending', ['lf','crlf'])
def test_fixed_parent_identity_survives_uniform_checkout_endings(tmp_path,monkeypatch,scene,ending):
    plan=json.loads((ROOT/'benchmarks'/('tum-fr1-'+scene+'-open3d-dense-plan.json')).read_text())
    original=PARENTS[plan['sequence']].read_bytes().replace(b'\r\n',b'\n')
    raw=original if ending=='lf' else original.replace(b'\n',b'\r\n')
    parent=tmp_path/'parent.json';parent.write_bytes(raw)
    monkeypatch.setitem(PARENTS,plan['sequence'],parent)
    assert validate_comparator_plan(plan) is plan
    assert parent.read_bytes()==raw
    bad=copy.deepcopy(plan);bad['prior_causal_dense_plan_sha256']='0'*64
    with pytest.raises(ValueError):validate_comparator_plan(bad)


@pytest.mark.parametrize('scene', ['xyz','rpy'])
@pytest.mark.parametrize('damage', ['whitespace','mixed_endings','stray_cr'])
def test_fixed_parent_identity_rejects_non_checkout_byte_changes(tmp_path,monkeypatch,scene,damage):
    plan=json.loads((ROOT/'benchmarks'/('tum-fr1-'+scene+'-open3d-dense-plan.json')).read_text())
    raw=PARENTS[plan['sequence']].read_bytes().replace(b'\r\n',b'\n')
    if damage=='whitespace':raw=b' '+raw
    elif damage=='mixed_endings':raw=raw.replace(b'\n',b'\r\n',1)
    else:raw=raw.replace(b'\n',b'\r',1)
    parent=tmp_path/'parent.json';parent.write_bytes(raw)
    monkeypatch.setitem(PARENTS,plan['sequence'],parent)
    with pytest.raises(ValueError):validate_comparator_plan(plan)


def inputs():
    camera = dict(width=8, height=8, fx=20., fy=20., cx=3.5, cy=3.5, depth_unit_m=.0002)
    depth = np.full((8, 8), 5000, np.uint16)
    gray = np.full((8, 8), 128, np.uint8)
    controls = dict(dense_rgbd_odometry=copy.deepcopy(VALIDATION),
                    open3d_rgbd_odometry=copy.deepcopy(CONTROLS))
    return depth, depth.copy(), gray, gray.copy(), camera, controls


def test_owned_relative_seed_direction_and_report_independence():
    calls = []
    returned = np.eye(4);returned[0, 3] = .001
    def compute(*args):
        calls.append(args[4].copy())
        args[4][:] = 99
        return True, returned, np.eye(6)
    adapter = PreviousMotionOpen3dRgbd(_compute=compute)
    inverse, report = adapter.align(*inputs())
    np.testing.assert_array_equal(calls[0], np.eye(4))
    assert inverse[0, 3] == -.001
    assert report['initial_previous_pair_index'] is None
    expected = returned.copy()
    inverse[:] = 88;report['current_from_previous'][0][3] = 77;returned[0, 3] = .002
    _, second = adapter.align(*inputs())
    np.testing.assert_array_equal(calls[1], expected)
    assert second['initial_previous_pair_index'] == 1
    check_causal_initialization(second, expected, 2)


def test_raw_inputs_are_snapshotted_before_external_backend_mutation():
    arguments = inputs();depth, _, gray, _, _, _ = arguments
    def compute(source_gray, source_depth, target_gray, target_depth, initial, camera):
        assert source_depth.dtype == source_gray.dtype == np.float32
        assert source_depth.flags.c_contiguous
        depth[:] = 0;gray[:] = 0
        source_depth[:] = 0;source_gray[:] = 0
        return True, np.eye(4), np.eye(6)
    _, report = PreviousMotionOpen3dRgbd(_compute=compute).align(*arguments)
    assert report['valid_source_pixels'] == 64 and report['correspondences'] == 49
    assert report['photometric_rmse'] == report['depth_rmse_m'] == 0
    assert report['owned_raw_input_bytes'] == 384 and report['owned_float_input_bytes'] == 1024


def test_returned_inverse_handles_rotation_and_translation_together():
    pose=np.eye(4);angle=.01
    pose[:2,:2]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    pose[:2,3]=[.001,.002]
    adapter=PreviousMotionOpen3dRgbd(_compute=lambda *args:(True,pose,np.eye(6)))
    inverse,report=adapter.align(*inputs())
    np.testing.assert_allclose(inverse@pose,np.eye(4),atol=1e-12,rtol=0)
    np.testing.assert_allclose(pose@inverse,np.eye(4),atol=1e-12,rtol=0)
    np.testing.assert_array_equal(report['current_from_previous'],pose)


def test_installed_legacy_cpu_images_own_float_buffers_without_estimation():
    cpu = pytest.importorskip('open3d.cpu.pybind')
    gray = np.arange(64, dtype=np.float32).reshape(8, 8)/np.float32(255)
    depth = np.full((8, 8), 1.5, np.float32)
    expected_gray, expected_depth = gray.copy(), depth.copy()
    image = owned_cpu_rgbd(cpu, gray, depth)
    gray[:] = 0;depth[:] = 0
    np.testing.assert_array_equal(np.asarray(image.color), expected_gray)
    np.testing.assert_array_equal(np.asarray(image.depth), expected_depth)
    assert np.asarray(image.color).dtype == np.asarray(image.depth).dtype == np.float32


@pytest.mark.parametrize('failure', ['flag','pose','information','support','api'])
def test_failure_does_not_advance_seed_or_pair_counter(failure):
    attempt = [0]
    observed = []
    def compute(*args):
        observed.append(args[4].copy());attempt[0] += 1
        pose = np.eye(4)
        if attempt[0] == 1:
            if failure == 'api':raise TypeError('unsupported native signature')
            if failure == 'flag':return False, pose, np.eye(6)
            if failure == 'pose':pose[0, 0] = 2
            if failure == 'information':return True, pose, np.full((6, 6), np.nan)
            if failure == 'support':pose[2, 3] = .2
        return True, pose, np.eye(6)
    adapter = PreviousMotionOpen3dRgbd(_compute=compute)
    with pytest.raises(ValueError):adapter.align(*inputs())
    _, report = adapter.align(*inputs())
    np.testing.assert_array_equal(observed[1], np.eye(4))
    assert report['initial_previous_pair_index'] is None


def test_instances_share_no_retained_motion():
    pose = np.eye(4);pose[0, 3] = .001
    a = PreviousMotionOpen3dRgbd(_compute=lambda *args: (True, pose, np.eye(6)))
    b = PreviousMotionOpen3dRgbd(_compute=lambda *args: (True, np.eye(4), np.eye(6)))
    a.align(*inputs());_, report = b.align(*inputs())
    np.testing.assert_array_equal(report['initial_current_from_previous'], np.eye(4))


@pytest.mark.parametrize('mutation', ['dtype','shape','camera','controls'])
def test_invalid_inputs_reject_before_external_call(mutation):
    arguments = list(inputs());calls = []
    if mutation == 'dtype':arguments[0] = arguments[0].astype(np.float32)
    if mutation == 'shape':arguments[2] = arguments[2][:-1]
    if mutation == 'camera':arguments[4]['fx'] = float('nan')
    if mutation == 'controls':arguments[5]['open3d_rgbd_odometry']['depth_diff_max'] = .1
    adapter = PreviousMotionOpen3dRgbd(_compute=lambda *args: calls.append(args))
    with pytest.raises(ValueError):adapter.align(*arguments)
    assert calls == []


@pytest.mark.parametrize('seed',[10,20,30])
def test_tiled_validation_matches_independent_full_frame_statistics(seed):
    rng = np.random.default_rng(seed);shape = (35, 43)
    source = rng.integers(4500, 5500, shape, dtype=np.uint16);target = source.copy()
    source[2:4, 4:6] = 0;target[18:21, 15:17] = 0
    gray = rng.integers(0, 256, shape, dtype=np.uint8)
    camera = dict(width=43,height=35,fx=40.,fy=40.,cx=21.,cy=17.,depth_unit_m=.0002)
    pose=np.eye(4);pose[0,3]=.0005;pose[2,3]=.001
    produced=raw_statistics(source,target,gray,gray,pose,camera,VALIDATION)
    reference=reference_statistics(source,target,gray,gray,pose,camera,VALIDATION)
    for key in ('valid_source_pixels','correspondences'):assert produced[key]==reference[key]
    for key in ('correspondence_fraction','photometric_rmse','depth_rmse_m'):
        assert produced[key] == pytest.approx(reference[key],abs=1e-12,rel=0)


def test_plan_keeps_native_geometry_and_accuracy_for_both_scenes():
    for parent_path in PARENTS.values():
        parent=json.loads(parent_path.read_text(encoding='utf-8'))
        path=parent_path.with_name(parent_path.name.replace('-causal-dense-plan','-open3d-dense-plan'))
        plan=validate_comparator_plan(json.loads(path.read_text(encoding='utf-8')))
        assert all(plan[key]==parent[key] for key in ('calibration','limits','acceptance','dense_initialization'))
        assert all(plan['controls'][key]==value for key,value in parent['controls'].items())


@pytest.mark.parametrize('change',['accuracy','support','package','seed','camera'])
def test_comparator_contract_rejects_changed_acceptance_or_model(change):
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    if change=='accuracy':plan['acceptance']['maximum_first_pose_ate_rmse_m']=.2
    if change=='support':plan['controls']['min_support_fraction']=.1
    if change=='package':plan['controls']['open3d_rgbd_odometry']['package_version']='0.20.0'
    if change=='seed':plan['dense_initialization']['first_pair']='reference'
    if change=='camera':plan['calibration']['fx']=517.3
    with pytest.raises(ValueError):validate_comparator_plan(plan)


@pytest.mark.parametrize('reject_support',[False,True])
def test_generator_preserves_world_io_and_no_call_after_loss(tmp_path,monkeypatch,reject_support):
    archive,digest=fixture(tmp_path,count=3);prepared=tmp_path/'prepared'
    prepare(archive,digest,PLAN,prepared);calls=[]
    def compute(*args):calls.append(args[4].copy());return True,np.eye(4),np.eye(6)
    if reject_support:monkeypatch.setattr(sr,'distance_gated_support',lambda *args:(0,0.,None))
    adapter=PreviousMotionOpen3dRgbd(_compute=compute)
    result=generate(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result',
        dense_rgbd_aligner=adapter.align,method=ESTIMATOR,additional_bindings=[__file__])
    rows=json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert len(rows)==3
    if reject_support:
        assert result['generated_poses']==1 and len(calls)==1
        assert [row['status'] for row in rows]==['success','error','error']
    else:
        assert result['generated_poses']==3 and len(calls)==2
        checked=audit(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result/estimates.json',
            result['estimates_sha256'],tmp_path/'audit')
        assert checked['causal_initialization_pairs']==checked['successful_pairs']==2
