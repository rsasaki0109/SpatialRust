"""Fixed held-out identity, fail-closed acquisition and complete original timelines."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import spatialrust as sr

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
import acquire_heldout_rgbd as download
from heldout_dense_rgbd_protocol import (ESTIMATOR,PARENT,SELECTION,SELECTION_LF_SHA256,
    heldout_plan,selection,source_identity,validate_heldout_plan)
from tum_sequence_reference import file_sha256,prepare,validate_plan
from run_tum_sequence import generate
from run_tum_rounded_sobel_dense_rgbd import PreviousMotionRoundedSobelRgbd
from audit_tum_dense_rgbd import audit
import test_tum_sequence_reference as fixture_module
from test_rounded_sobel_dense_rgbd import mock_native


def pin_value(plant_sha='a'*64,room_sha='b'*64):
    rows=[]
    for row,digest in zip(selection()['sources'],[plant_sha,room_sha]):
        rows.append(dict(sequence=row['sequence'],url=row['url'],etag=row['etag'],bytes=row['content_length'],sha256=digest))
    return dict(schema='spatialrust.heldout-source-identities.v1',selection_lf_sha256=SELECTION_LF_SHA256,
        frozen_at_utc='2026-10-20T00:00:00+00:00',sources=rows)


def write_pin(tmp_path,value):
    path=tmp_path/'identities.json';raw=(json.dumps(value,sort_keys=True)+'\n').encode()
    path.write_bytes(raw);return path,hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('crlf',[False,True])
def test_exact_selection_accepts_only_its_uniform_line_endings(tmp_path,crlf):
    raw=SELECTION.read_bytes().replace(b'\r\n',b'\n');path=tmp_path/'selection.json'
    path.write_bytes(raw.replace(b'\n',b'\r\n') if crlf else raw)
    assert selection(path)['candidate']['model']=='rounded-sobel'


@pytest.mark.parametrize('change',['mixed','whitespace','candidate','limit'])
def test_selection_cannot_change_model_or_resource_choices(tmp_path,change):
    raw=SELECTION.read_bytes().replace(b'\r\n',b'\n')
    if change=='mixed':raw=raw.replace(b'\n',b'\r\n',1)
    elif change=='whitespace':raw+=b' '
    elif change=='candidate':raw=raw.replace(b'rounded-sobel',b'other-model',1)
    else:raw=raw.replace(b'8589934592',b'8589934593',1)
    path=tmp_path/'selection.json';path.write_bytes(raw)
    with pytest.raises(ValueError):selection(path)


@pytest.mark.parametrize('scene',['plant','room'])
def test_added_recordings_require_exact_plans_and_all_original_controls(scene):
    plan=heldout_plan('rgbd_dataset_freiburg1_'+scene)
    assert validate_heldout_plan(plan)==validate_plan(plan)==plan
    parent=json.loads(PARENT.read_text(encoding='utf-8'))
    for name in ('calibration','controls','acceptance','dense_initialization','dense_support_policy'):
        assert plan[name]==parent[name]
    assert plan['limits']['max_total_bytes']==2*1024**3
    assert plan['heldout_validation']['candidate']['development_rpy_accepted'] is False


@pytest.mark.parametrize('group,key,value',[
    ('acceptance','maximum_first_pose_ate_rmse_m',.2),('calibration','fx',517.3),
    ('dense_initialization','reset_after_loss',True),('limits','max_total_bytes',8*1024**3),
    ('controls','min_support_fraction',.4),('heldout_validation','candidate',{}),
    ('heldout_validation','selection_lf_sha256','a'*64)])
def test_new_recordings_cannot_admit_changed_camera_gates_or_candidate(group,key,value):
    plan=heldout_plan('rgbd_dataset_freiburg1_plant');plan[group][key]=value
    with pytest.raises(ValueError):validate_plan(plan)


@pytest.mark.parametrize('scene',['plant','room'])
def test_renaming_an_old_plan_does_not_bypass_new_identity_admission(scene):
    old=json.loads(PARENT.read_text(encoding='utf-8'));old['sequence']='rgbd_dataset_freiburg1_'+scene
    with pytest.raises(ValueError):validate_plan(old)


@pytest.mark.parametrize('name',['rgbd_dataset_freiburg1_desk','rgbd_dataset_freiburg1_desk2','rgbd_dataset_freiburg2_xyz','unknown'])
def test_observed_or_undeclared_recordings_are_not_held_out(name):
    with pytest.raises(ValueError):heldout_plan(name)


def test_source_pin_matches_both_exact_official_descriptors_and_manifest_hash(tmp_path):
    path,digest=write_pin(tmp_path,pin_value())
    assert source_identity(path,digest,'rgbd_dataset_freiburg1_plant','a'*64)['bytes']==695620135
    with pytest.raises(ValueError,match='prepared source'):source_identity(path,digest,'rgbd_dataset_freiburg1_plant','c'*64)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='changed'):source_identity(path,digest,'rgbd_dataset_freiburg1_plant','a'*64)


@pytest.mark.parametrize('change',['missing','duplicate','url','etag','bytes','boolean-bytes','sha','timestamp','row-type'])
def test_incomplete_or_changed_source_pin_rejects(tmp_path,change):
    value=pin_value()
    if change=='missing':value['sources'].pop()
    elif change=='duplicate':value['sources'][1]=copy.deepcopy(value['sources'][0])
    elif change=='timestamp':value['frozen_at_utc']='not a timezone timestamp'
    elif change=='row-type':value['sources'][0]=None
    else:
        field='bytes' if change=='boolean-bytes' else 'sha256' if change=='sha' else change
        value['sources'][0][field]=True if change=='boolean-bytes' else 'wrong'
    path,digest=write_pin(tmp_path,value)
    with pytest.raises(ValueError):source_identity(path,digest,'rgbd_dataset_freiburg1_plant','a'*64)


class Response(io.BytesIO):
    def __init__(self,body,source,**changes):
        super().__init__(body);self.status=200;self.url=source['url']
        self.headers={'Content-Length':str(source['content_length']),'ETag':source['etag']}
        self.headers.update(changes)


class Opener:
    def __init__(self,response):self.response=response;self.calls=0
    def open(self,request,timeout):self.calls+=1;return self.response


def acquisition_case(tmp_path,monkeypatch,body=b'\x1f\x8bopaque archive'):
    root=tmp_path/'input';root.mkdir()
    source=copy.deepcopy(selection()['sources'][0]);source['content_length']=len(body)
    resources=copy.deepcopy(selection()['resources']);resources['archives_max_bytes']=1024**2
    monkeypatch.setattr(download.shutil,'disk_usage',lambda _:SimpleNamespace(free=32*1024**3))
    return root,source,resources,body


def test_opaque_download_pins_bytes_once_and_preserves_prior_acquisition(tmp_path,monkeypatch):
    root,source,resources,body=acquisition_case(tmp_path,monkeypatch)
    opener=Opener(Response(body,source));result=download.acquire(source,root,resources,opener=opener)
    archive=Path(result['archive']['path'])
    assert archive.read_bytes()==body and result['archive']['sha256']==hashlib.sha256(body).hexdigest()
    assert result['status']=='complete' and result['image_or_reference_interpretation'] is False
    assert result['additional_payload_copies']==0 and opener.calls==1
    with pytest.raises(FileExistsError):download.acquire(source,root,resources,opener=opener)
    assert opener.calls==1 and archive.read_bytes()==body


@pytest.mark.parametrize('free',[0,20*1024**3-1])
def test_disk_floor_refuses_before_network_or_case_creation(tmp_path,monkeypatch,free):
    root,source,resources,body=acquisition_case(tmp_path,monkeypatch)
    monkeypatch.setattr(download.shutil,'disk_usage',lambda _:SimpleNamespace(free=free))
    opener=Opener(Response(body,source))
    with pytest.raises(ValueError,match='20GiB'):download.acquire(source,root,resources,opener=opener)
    assert opener.calls==0 and list(root.iterdir())==[]


@pytest.mark.parametrize('change',['etag','length','encoding','truncated','not-gzip','duration','budget'])
def test_changed_or_partial_source_never_becomes_an_admitted_archive(tmp_path,monkeypatch,change):
    root,source,resources,body=acquisition_case(tmp_path,monkeypatch)
    actual=body;headers={}
    if change=='etag':headers['ETag']='different'
    elif change=='length':headers['Content-Length']=str(len(body)+1)
    elif change=='encoding':headers['Content-Encoding']='gzip'
    elif change=='truncated':actual=body[:-1]
    elif change=='not-gzip':actual=b'no'+body[2:]
    elif change=='duration':resources['source_command_max_seconds']=0
    elif change=='budget':
        resources['archives_max_bytes']=len(body)
        (root/'existing.tgz.part').write_bytes(b'x')
    opener=Opener(Response(actual,source,**headers))
    with pytest.raises(ValueError):download.acquire(source,root,resources,opener=opener)
    case=root/source['sequence'];journal=json.loads((case/'acquisition.json').read_text(encoding='utf-8'))
    assert journal['status']=='failed' and not (case/(source['sequence']+'.tgz')).exists()
    if change in ('truncated','not-gzip'):assert Path(journal['preserved_partial']['path']).exists()
    if change=='budget':assert opener.calls==0


def test_source_redirects_and_repository_output_are_rejected(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='redirect'):download.NoRedirect().redirect_request(None,None,None,None,None,None)
    root,source,resources,body=acquisition_case(tmp_path,monkeypatch)
    monkeypatch.setattr(download,'ROOT',tmp_path)
    with pytest.raises(ValueError,match='repository'):download.acquire(source,root,resources,opener=Opener(Response(body,source)))


@pytest.mark.parametrize('scene',['plant','room'])
@pytest.mark.parametrize('reject_support',[False,True])
def test_full_heldout_generator_keeps_all_rows_pin_and_no_reset(tmp_path,monkeypatch,scene,reject_support,synthetic_dense_audit_space):
    sequence='rgbd_dataset_freiburg1_'+scene;monkeypatch.setattr(fixture_module,'ROOT',sequence)
    archive,digest=fixture_module.fixture(tmp_path,count=3)
    plan=ROOT/'benchmarks'/('tum-fr1-'+scene+'-heldout-dense-plan.json')
    prepared=tmp_path/'prepared';prepare(archive,digest,plan,prepared)
    assert not (prepared/'groundtruth.txt').exists()
    pin=pin_value(plant_sha=digest if scene=='plant' else 'a'*64,room_sha=digest if scene=='room' else 'b'*64)
    identities,pin_sha=write_pin(tmp_path,pin)
    calls=[];monkeypatch.setattr(sr,'estimate_dense_rgbd_odometry',mock_native(calls,audit_stats=True))
    if reject_support:monkeypatch.setattr(sr,'distance_gated_support',lambda *args:(0,0.,None))
    result=generate(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result',
        dense_rgbd_aligner=PreviousMotionRoundedSobelRgbd().align,method=ESTIMATOR,additional_bindings=[__file__,identities])
    rows=json.loads((tmp_path/'result/estimates.json').read_text(encoding='utf-8'))['poses']
    assert len(rows)==3 and len(calls)==(1 if reject_support else 2)
    if reject_support:assert [r['status'] for r in rows]==['success','error','error']
    else:
        with pytest.raises(ValueError,match='source identity pin'):
            audit(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result/estimates.json',result['estimates_sha256'],tmp_path/'missing-pin-audit')
        checked=audit(prepared,file_sha256(prepared/'manifest.json'),tmp_path/'result/estimates.json',result['estimates_sha256'],tmp_path/'audit',source_identities=identities,source_identities_sha256=pin_sha)
        assert checked['successful_pairs']==checked['causal_initialization_pairs']==2
        assert checked['source_identities_lf_sha256']==pin_sha
