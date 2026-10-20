"""Fixed complete model, unchanged raw admission and owned native inputs."""
import ast
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_tum_dense_rgbd import audit, check_causal_initialization, reference_statistics
from dense_rgbd_protocol import CONTROLS
from rounded_sobel_dense_rgbd_protocol import ESTIMATOR, PARENTS, POLICY, parent_plan, validate_rounded_sobel_dense_plan
from run_tum_rounded_sobel_dense_rgbd import PreviousMotionRoundedSobelRgbd
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256, prepare
from test_dense_rgbd_odometry import images, estimate
from test_tum_sequence_reference import fixture

PLAN = ROOT/'benchmarks/tum-fr1-xyz-rounded-sobel-dense-plan.json'


def rounded(source, target, **kwargs):
    return estimate(source, target, fitting_model='rounded-sobel', **kwargs)


def test_gain_invariance_keeps_independent_original_raw_measurements():
    gray, depth = images(); gray = (gray//2).astype(np.uint8)
    target = (gray*2, depth.copy())
    rotation, translation, report = rounded((gray, depth), target)
    np.testing.assert_allclose(rotation, np.eye(3), rtol=0, atol=1e-10)
    np.testing.assert_allclose(translation, np.zeros(3), rtol=0, atol=1e-10)
    assert report['iterations'] == 35 and report['fitting_model'] == 'rounded-sobel'
    assert report['photometric_rmse'] > .1 and report['depth_rmse_m'] < 1e-12
    pose = np.eye(4); pose[:3,:3] = rotation; pose[:3,3] = translation
    camera = dict(width=96,height=72,fx=64.,fy=64.,cx=48.,cy=36.,depth_unit_m=.0002)
    counts = np.full((72,96),10000,np.uint16)
    expected = reference_statistics(counts,counts,gray,target[0],pose,camera,CONTROLS)
    for key in ('valid_source_pixels','correspondences'): assert report[key] == expected[key]
    for key in ('correspondence_fraction','photometric_rmse','depth_rmse_m'):
        assert abs(report[key]-expected[key]) < 1e-10


def test_integer_motion_proper_pose_and_original_input_ownership():
    source, target = images(), images(1)
    originals = [value.copy() for value in (*source,*target)]
    rotation, translation, report = rounded(source,target)
    # One-half input pixel in metric x, fixed for the rounded surrogate model.
    np.testing.assert_allclose(translation,[.03125,0.,0.],rtol=0,atol=.016)
    np.testing.assert_allclose(rotation,np.eye(3),rtol=0,atol=.01)
    np.testing.assert_allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-10)
    assert abs(np.linalg.det(rotation)-1.) < 1e-10
    assert report['iterations'] == 35 and report['final_support_only'] is True
    for value, original in zip((*source,*target), originals): np.testing.assert_array_equal(value,original)
    assert all(not np.shares_memory(output,value) for output in (rotation,translation) for value in (*source,*target))


@pytest.mark.parametrize('policy',['raw','gaussian3','gaussian3-gray-only','gaussian3-gray-only-pair-mean'])
def test_explicit_bilinear_retains_every_old_selector_and_exact_report(policy):
    source = images()
    a = estimate(source,source,preprocessing=policy)
    b = estimate(source,source,preprocessing=policy,fitting_model='bilinear')
    for x,y in zip(a[:2],b[:2]): np.testing.assert_array_equal(x,y)
    assert a[2] == b[2] and 'fitting_model' not in a[2]


def test_bound_boundary_raw_denominator_and_missing_input_preserved():
    gray, depth = images(); depth[36,48] = np.nan
    report = rounded((gray,depth),(gray,depth))[2]
    assert report['valid_source_pixels'] == 96*72-1 and np.isnan(depth[36,48])
    required = report['working_bytes']
    assert rounded((gray,depth),(gray,depth),max_working_bytes=required)[2]['working_bytes'] == required
    with pytest.raises(ValueError,match='budget'):
        rounded((gray,depth),(gray,depth),max_working_bytes=required-1)
    with pytest.raises(ValueError):
        rounded(images(width=7,height=5),images(width=7,height=5),iterations=[1,1])


@pytest.mark.parametrize('model',['','RoundedSobel','unknown',None,1])
def test_invalid_complete_model_rejects(model):
    source = images()
    with pytest.raises((ValueError,TypeError)): estimate(source,source,fitting_model=model)


@pytest.mark.parametrize('policy',['gaussian3','gaussian3-gray-only','gaussian3-gray-only-pair-mean'])
def test_mixed_legacy_filter_and_complete_model_reject(policy):
    source = images()
    with pytest.raises(ValueError,match='preprocessing must be raw'):
        rounded(source,source,preprocessing=policy)


@pytest.mark.parametrize('value',[0.,-1.,np.nan,np.inf])
def test_all_invalid_depth_cannot_masquerade_as_motion(value):
    gray, depth = images(); depth.fill(value)
    with pytest.raises(ValueError): rounded((gray,depth),(gray,depth))


def test_zero_brightness_rejects_before_mutating_inputs():
    gray,depth = images(); gray.fill(0); before = depth.copy()
    with pytest.raises(ValueError,match='pair means'): rounded((gray,depth),(gray.copy(),depth.copy()))
    assert not np.any(gray); np.testing.assert_array_equal(depth,before)


def test_strided_readonly_concurrent_calls_own_independent_deterministic_results():
    source = tuple(np.repeat(value,2,axis=1)[:,::2] for value in images(width=256,height=160))
    for value in source: value.setflags(write=False)
    assert all(not value.flags.c_contiguous for value in source)
    def run(_): return rounded(source,source,iterations=[1])
    with ThreadPoolExecutor(max_workers=2) as pool: a,b = pool.map(run,range(2))
    for x,y in zip(a[:2],b[:2]): np.testing.assert_array_equal(x,y)
    assert a[2] == b[2] and a[2]['input_copy_bytes'] == 256*160*10
    assert 1 <= a[2]['workers'] <= 4
    assert all(not np.shares_memory(value,output) for value in source for output in a[:2])


def test_model_keyword_is_present_in_checked_public_stub():
    tree = ast.parse((ROOT/'crates/spatialrust-py/spatialrust.pyi').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name == 'estimate_dense_rgbd_odometry')
    assert 'fitting_model' in [node.arg for node in function.args.kwonlyargs]


@pytest.mark.parametrize('scene',['xyz','rpy'])
def test_fixed_plan_preserves_all_source_numeric_and_final_gates(scene):
    plan = json.loads((ROOT/'benchmarks'/('tum-fr1-'+scene+'-rounded-sobel-dense-plan.json')).read_text(encoding='utf-8'))
    validate_rounded_sobel_dense_plan(plan)
    parent = json.loads(PARENTS[plan['sequence']].read_text(encoding='utf-8'))
    for key in ('calibration','limits','acceptance','dense_initialization'): assert plan[key] == parent[key]
    for key,value in parent['controls'].items():
        if key != 'dense_fitting_preprocessing': assert plan['controls'][key] == value


@pytest.mark.parametrize('group,key,value',[
    ('dense_initialization','time_rescaling',True),('acceptance','maximum_first_pose_ate_rmse_m',.2),
    ('controls','min_support_fraction',.4),('calibration','fx',517.3),
    ('dense_fitting_model','workers',8),('dense_fitting_model','iterations','early stop'),
    ('dense_fitting_model','final_statistics','filtered')])
def test_no_model_seed_calibration_or_gate_change_is_accepted(group,key,value):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    target = plan['controls'][group] if group == 'dense_fitting_model' else plan[group]
    target[key] = value
    with pytest.raises(ValueError): validate_rounded_sobel_dense_plan(plan)


@pytest.mark.parametrize('crlf',[False,True])
def test_parent_identity_accepts_only_exact_uniform_line_endings(tmp_path,crlf):
    raw = next(iter(PARENTS.values())).read_bytes().replace(b'\r\n',b'\n')
    path = tmp_path/'parent.json'; path.write_bytes(raw.replace(b'\n',b'\r\n') if crlf else raw)
    assert parent_plan(path)[0]['estimator'] == 'spatialrust_pair_mean_dense_rgbd_cpu'
    path.write_bytes(raw.replace(b'\n',b'\r\n',1))
    with pytest.raises(ValueError): parent_plan(path)


def alignment_args():
    depth = np.full((32,32),5000,np.uint16); gray = np.full((32,32),100,np.uint8)
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    return depth,depth.copy(),gray,gray.copy(),plan['calibration'],plan['controls']


def mock_native(calls, *, fail_at=None, wrong_model=False, audit_stats=False):
    def solve(*args,**kwargs):
        calls.append(kwargs['initial'].copy()); assert kwargs['fitting_model'] == 'rounded-sobel'
        if len(calls) == fail_at: raise ValueError('controlled rounded-Sobel failure')
        depth = args[1]; valid = int(np.count_nonzero(depth>0))
        count = (depth.shape[0]-1)*(depth.shape[1]-1) if audit_stats else valid
        return np.eye(3),np.zeros(3),dict(valid_source_pixels=valid,correspondences=count,
            correspondence_fraction=count/valid,photometric_rmse=0.,depth_rmse_m=0.,iterations=35,
            final_support_only=True,fitting_preprocessing='raw',fitting_model='wrong' if wrong_model else 'rounded-sobel')
    return solve


def test_owned_causal_receipts_and_failed_calls_do_not_advance_state(monkeypatch):
    calls = []; monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',mock_native(calls,fail_at=2))
    adapter = PreviousMotionRoundedSobelRgbd(); _,first = adapter.align(*alignment_args())
    first['fitting_policy']['initial_filter']['kernel'][0] = 99
    assert POLICY['initial_filter']['kernel'] == [1,2,1]
    with pytest.raises(ValueError): adapter.align(*alignment_args())
    _,third = adapter.align(*alignment_args()); assert third['initial_previous_pair_index'] == 1
    check_causal_initialization(third,np.eye(4),2)


def test_wrong_native_model_receipt_rejects(monkeypatch):
    calls = []; monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',mock_native(calls,wrong_model=True))
    with pytest.raises(ValueError,match='model'): PreviousMotionRoundedSobelRgbd().align(*alignment_args())


@pytest.mark.parametrize('reject_support',[False,True])
def test_complete_generator_keeps_raw_audit_and_stops_after_loss(tmp_path,monkeypatch,reject_support,synthetic_dense_audit_space):
    archive,digest = fixture(tmp_path,count=3); prepared = tmp_path/'prepared'
    prepare(archive,digest,PLAN,prepared); calls = []
    monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',mock_native(calls,audit_stats=True))
    if reject_support: monkeypatch.setattr(sr,'distance_gated_support',lambda *args:(0,0.,None))
    result = generate(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result',
        dense_rgbd_aligner=PreviousMotionRoundedSobelRgbd().align,method=ESTIMATOR,additional_bindings=[__file__])
    rows = json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert len(rows) == 3 and len(calls) == (1 if reject_support else 2)
    if reject_support: assert [row['status'] for row in rows] == ['success','error','error']
    else:
        checked = audit(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result/estimates.json',result['estimates_sha256'],tmp_path/'audit')
        assert checked['successful_pairs'] == checked['causal_initialization_pairs'] == 2
