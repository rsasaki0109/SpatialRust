"""Independent public GPU input projection, original pixels and source binding."""
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from prepare_public_gpu_rgbd import (CAMERA, SOURCE_SHA256, checked_image, decode_depth,
                                     decode_rgb, prepare, project_reference, sha256)


def png(values):
    stream = io.BytesIO()
    Image.fromarray(values).save(stream, format='PNG')
    return stream.getvalue()


def test_projection_retains_closed_depth_limits_and_row_order():
    camera = dict(CAMERA, width=3, height=2, fx=2., fy=4., cx=.5, cy=.5)
    depth = np.array([[0,500,25000],[25001,1000,65535]],dtype=np.uint16)
    xyz = project_reference(depth,camera)
    expected = np.array([[.025,-.0125,.1],[3.75,-.625,5.],[.05,.025,.2]],dtype='<f4')
    np.testing.assert_array_equal(xyz,expected)
    assert xyz.flags.c_contiguous and xyz.dtype == np.dtype('<f4')
    np.testing.assert_array_equal(project_reference(depth[:,::-1][:,::-1],camera),expected)
    with pytest.raises(ValueError):
        project_reference(depth.astype(np.float32),camera)


def test_png_decoding_preserves_counts_and_rgb_without_conversion():
    camera = dict(CAMERA,width=2,height=1)
    depth = np.array([[500,65535]],dtype=np.uint16)
    rgb = np.array([[[0,17,255],[1,2,3]]],dtype=np.uint8)
    np.testing.assert_array_equal(decode_depth(png(depth),camera),depth)
    np.testing.assert_array_equal(decode_rgb(png(rgb),camera),rgb)
    with pytest.raises(ValueError):
        decode_depth(png(depth.astype(np.uint8)),camera)
    with pytest.raises(ValueError):
        decode_rgb(png(depth),camera)
    with pytest.raises(ValueError):
        decode_rgb(png(rgb),dict(camera,width=3))


@pytest.mark.parametrize('change',[dict(path='../escape.png'),dict(path='rgb/../escape.png'),
                                 dict(path='depth/frame.png'),dict(bytes=True),dict(sha256='0'*64)])
def test_image_identity_rejects_aliases_and_corruption(tmp_path,change):
    directory = tmp_path/'rgb'; directory.mkdir()
    payload = png(np.zeros((1,1,3),dtype=np.uint8))
    (directory/'frame.png').write_bytes(payload)
    record = dict(path='rgb/frame.png',bytes=len(payload),sha256=sha256(payload))
    assert checked_image(tmp_path,record,'rgb') == payload
    with pytest.raises((ValueError,OSError)):
        checked_image(tmp_path,dict(record,**change),'rgb')


def fixture(tmp_path):
    prepared = tmp_path/'prepared'; prepared.mkdir()
    files = {}
    for kind, values in [('rgb',np.zeros((480,640,3),dtype=np.uint8)),
                         ('depth',np.full((480,640),500,dtype=np.uint16))]:
        (prepared/kind).mkdir()
        raw = png(values); (prepared/kind/'frame.png').write_bytes(raw)
        files[kind] = dict(path=kind+'/frame.png',bytes=len(raw),sha256=sha256(raw),timestamp_ns=1_000_000_000)
    raw_camera = (json.dumps(CAMERA)+'\n').encode()
    (prepared/'calibration.json').write_bytes(raw_camera)
    frame = dict(rgb=files['rgb'],depth=files['depth'],timestamp_ns=1_000_000_000,rgb_delta_ns=0)
    manifest = dict(schema='spatialrust.tum-prepared.v1',source_sha256=SOURCE_SHA256,
                    reference_used_for_generation=False,calibration_sha256=sha256(raw_camera),
                    frames=[copy.deepcopy(frame) for _ in range(798)])
    return prepared, manifest


def test_preparation_binds_every_selected_original_and_refuses_overwrite(tmp_path):
    prepared, manifest = fixture(tmp_path)
    raw = json.dumps(manifest).encode(); (prepared/'manifest.json').write_bytes(raw)
    output = tmp_path/'frozen'
    digest, result = prepare(prepared,sha256(raw),output)
    assert sha256((output/'manifest.json').read_bytes()) == digest
    assert [frame['index'] for frame in result['frames']] == [0,399,797]
    assert all(frame['point_count'] == 307200 for frame in result['frames'])
    for frame in result['frames']:
        for name in ('rgb','positions'):
            payload = (output/frame[name]['path']).read_bytes()
            assert len(payload) == frame[name]['bytes'] and sha256(payload) == frame[name]['sha256']
    with pytest.raises(ValueError,match='overwrite'):
        prepare(prepared,sha256(raw),output)


@pytest.mark.parametrize('case',['manifest_hash','source_hash','timestamp','reference','calibration'])
def test_preparation_rejects_broken_source_contract_before_output(tmp_path,case):
    prepared, manifest = fixture(tmp_path)
    if case == 'source_hash': manifest['source_sha256'] = '0'*64
    if case == 'timestamp': manifest['frames'][399]['rgb_delta_ns'] = 1
    if case == 'reference': manifest['reference_used_for_generation'] = True
    if case == 'calibration': (prepared/'calibration.json').write_text('{}',encoding='utf-8')
    raw = json.dumps(manifest).encode(); (prepared/'manifest.json').write_bytes(raw)
    digest = '0'*64 if case == 'manifest_hash' else sha256(raw)
    output = tmp_path/'unwritten'
    with pytest.raises(ValueError): prepare(prepared,digest,output)
    assert not output.exists()
