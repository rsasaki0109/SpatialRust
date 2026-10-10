"""Timestamp, row layout, calibration provenance and native attribute checks."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import zipfile

import numpy as np
import pytest
import spatialrust as sr

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from realsense_reference import (camera_parameters, check_attributes, decode_depth, inventory,
                                reference_geometry, save_attribute_input, timestamp_ns)
from run_realsense_reference import cpu_execution_receipt,validate_plan, verified_source


def camera_message():
    return NS(width=3, height=2, distortion_model='None', D=np.zeros(5),
              K=np.array([2.,0,1,0,4,.5,0,0,1]), R=np.zeros(9), P=np.zeros(12))


def test_sensor_timestamp_keeps_nanoseconds_past_float_precision():
    assert timestamp_ns(NS(sec=1607987782,nanosec=123456789)) == 1607987782123456789
    assert timestamp_ns(NS(sec=1607987782,nanosec=123456790)) == 1607987782123456790


@pytest.mark.parametrize('stamp', [NS(sec=-1,nanosec=0),NS(sec=1,nanosec=10**9),
                                 NS(sec=1.0,nanosec=0),NS(sec=1,nanosec=True)])
def test_invalid_timestamp_rejected(stamp):
    with pytest.raises(ValueError):timestamp_ns(stamp)


@pytest.mark.parametrize('big_endian', [0,1])
def test_depth_reads_padded_rows_and_explicit_endian(big_endian):
    rows=np.array([[1,256,65535,999],[400,800,1200,999]],dtype='>u2' if big_endian else '<u2')
    message=NS(width=3,height=2,step=8,encoding='mono16',is_bigendian=big_endian,
               data=np.frombuffer(rows.tobytes(),np.uint8))
    np.testing.assert_array_equal(decode_depth(message,6), [[1,256,65535],[400,800,1200]])
    with pytest.raises(ValueError):decode_depth(message,5)
    message.data=message.data[:-1]
    with pytest.raises(ValueError):decode_depth(message,6)


@pytest.mark.parametrize('key,value', [('encoding','rgb8'),('is_bigendian',2),('step',3),('width',0)])
def test_depth_rejects_wrong_encoding_layout(key,value):
    message=NS(width=1,height=1,step=2,encoding='mono16',is_bigendian=0,data=np.array([0,1],np.uint8))
    setattr(message,key,value)
    with pytest.raises(ValueError):decode_depth(message,10)


@pytest.mark.parametrize('kind', ['distortion','nonfinite','skew','dimension','focal'])
def test_no_silent_distortion_or_intrinsic_assumption(kind):
    message=camera_message()
    if kind=='distortion':message.distortion_model='Inverse Brown Conrady'
    if kind=='nonfinite':message.K[2]=np.nan
    if kind=='skew':message.K[1]=1
    if kind=='dimension':message.width=0
    if kind=='focal':message.K[0]=0
    with pytest.raises(ValueError):camera_parameters(message)


def test_optical_axes_raw_depth_and_typed_native_roundtrip(tmp_path):
    camera=camera_parameters(camera_message())
    depth=np.array([[1000,0,4000],[2000,1000,65535]],np.uint16)
    controls=dict(min_depth_m=.1,max_depth_m=3.)
    valid,expected=reference_geometry(depth,camera,.001,controls)
    np.testing.assert_array_equal(expected,[[-.5,-.125,1],[-1,.25,2],[0,.125,1]])
    native=sr.depth_to_xyz(depth.astype(np.float32),2.,4.,1.,.5,depth_scale=.001,min_depth=.1,max_depth=3.)
    np.testing.assert_allclose(native[valid],expected,rtol=0,atol=2e-7)
    path=tmp_path/'original.pcd'
    fields=save_attribute_input(path,native[valid],depth,valid)
    cloud=sr.read(str(path))
    check_attributes(cloud,fields)
    np.testing.assert_array_equal(fields['pixel_u'],[0,0,1])
    np.testing.assert_array_equal(fields['pixel_v'],[0,1,1])
    out=tmp_path/'restored.pcd'
    sr.write(str(out),cloud)
    check_attributes(sr.read(str(out)),fields)
    with pytest.raises(FileExistsError):save_attribute_input(path,native[valid],depth,valid)
    with pytest.raises(ValueError):check_attributes(sr.PointCloud.from_xyz(native[valid]),fields)


def test_checked_archive_rejects_tamper_and_unexpected_members(tmp_path):
    path=tmp_path/'input.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('x.bag',b'bag')
    def source():return dict(name='x.bag',archive=path.name,md5=hashlib.md5(path.read_bytes()).hexdigest(),
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    frozen=source()
    assert verified_source(tmp_path,frozen).read_bytes()==b'bag'
    (tmp_path/'x.bag').write_bytes(b'changed')
    with pytest.raises(ValueError):verified_source(tmp_path,frozen)
    with zipfile.ZipFile(path,'a') as z:z.writestr('../escape.bag',b'x')
    with pytest.raises(ValueError):verified_source(tmp_path,frozen)
    with pytest.raises(ValueError):verified_source(tmp_path,source())


def test_plan_rejects_unbounded_controls_and_path_traversal():
    path=Path(__file__).resolve().parents[3]/'benchmarks/realsense-l515-plan.json'
    plan=json.loads(path.read_bytes())
    validate_plan(plan)
    for kind in ('stride','depth','path','backend'):
        modified=copy.deepcopy(plan)
        if kind=='stride':modified['controls']['save_stride']=False
        if kind=='depth':modified['controls']['max_depth_m']=-1
        if kind=='path':modified['inputs'][0]['name']='../outside.bag'
        if kind=='backend':modified['controls']['pipeline_policy']='auto'
        with pytest.raises(ValueError):validate_plan(modified)


def test_cpu_execution_accepts_both_documented_cpu_policies():
    pipeline=NS(resolved_policies=['CpuParallel','CpuSingle','CpuSingle','CpuSingle'],
                host_to_device_bytes=0,device_to_device_bytes=0,device_to_host_bytes=0)
    assert cpu_execution_receipt(pipeline)['resolved_policies']==pipeline.resolved_policies


@pytest.mark.parametrize('kind',['gpu','auto','incomplete','h2d','d2d','d2h'])
def test_cpu_execution_rejects_device_work_or_unresolved_backend(kind):
    pipeline=NS(resolved_policies=['CpuSingle']*4,
                host_to_device_bytes=0,device_to_device_bytes=0,device_to_host_bytes=0)
    if kind=='gpu':pipeline.resolved_policies[0]='Gpu'
    if kind=='auto':pipeline.resolved_policies[0]='Auto'
    if kind=='incomplete':pipeline.resolved_policies.pop()
    if kind=='h2d':pipeline.host_to_device_bytes=1
    if kind=='d2d':pipeline.device_to_device_bytes=1
    if kind=='d2h':pipeline.device_to_host_bytes=1
    with pytest.raises(ValueError):cpu_execution_receipt(pipeline)


def fake_inventory(monkeypatch,tmp_path,mode='valid'):
    stream='/depth'
    messages=[(stream+'/info/camera_info',1,camera_message()),
              ('/sensor/option/Depth_Units/value',1,NS(data=.001)),
              (stream+'/image/data',100,NS(header=NS(stamp=NS(sec=2,nanosec=3),seq=10,frame_id='0'))),
              (stream+'/image/metadata',100,NS(key='timestamp_domain',value='Hardware Clock')),
              (stream+'/image/data',200,NS(header=NS(stamp=NS(sec=2,nanosec=4),seq=11,frame_id='0')))]
    if mode=='late_calibration':messages[0]=(messages[0][0],101,messages[0][2])
    if mode=='missing_units':messages.pop(1)
    if mode=='negative_units':messages[1][2].data=-1
    if mode=='nonmonotonic':messages[-1][2].header.stamp.nanosec=3
    if mode=='duplicate_metadata':messages.append(messages[3])
    if mode=='orphan_metadata':messages.append((stream+'/image/metadata',300,NS(key='x',value='y')))
    if mode=='changing_calibration':messages.append(messages[0])
    payloads={str(i).encode():m for i,(_,_,m) in enumerate(messages)}
    class Reader:
        def __init__(self,paths):
            topics={topic for topic,_,_ in messages}
            self.connections=[NS(topic=t,msgcount=sum(t==topic for topic,_,_ in messages),msgtype='fake') for t in topics]
            self.duration=201
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def messages(self,connections):
            by_topic={c.topic:c for c in connections}
            for i,(topic,t,_) in enumerate(messages):
                if topic in by_topic:yield by_topic[topic],t,str(i).encode()
        def deserialize(self,raw,msgtype):return payloads[raw]
    monkeypatch.setitem(sys.modules,'rosbags.highlevel',NS(AnyReader=Reader))
    path=tmp_path/'source.bag';path.write_bytes(b'fixture')
    return inventory(path,stream,10)


def test_inventory_preserves_distinct_clock_domains_and_source_hash(monkeypatch,tmp_path):
    info=fake_inventory(monkeypatch,tmp_path)
    assert info['frames'][0]['bag_timestamp_ns']==100
    assert info['frames'][0]['sensor_timestamp_ns']==2000000003
    assert info['source_sha256']==hashlib.sha256(b'fixture').hexdigest()
    assert info['clock']['declared_domains']==['Hardware Clock','unknown']
    assert info['clock']['measured_synchronization_available'] is False
    assert info['independent_trajectory_truth_available'] is False


@pytest.mark.parametrize('mode',['late_calibration','missing_units','negative_units','nonmonotonic',
                               'duplicate_metadata','orphan_metadata','changing_calibration'])
def test_inventory_rejects_invalid_source_evidence(monkeypatch,tmp_path,mode):
    with pytest.raises(ValueError):fake_inventory(monkeypatch,tmp_path,mode)
