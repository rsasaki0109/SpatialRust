"""Offline canonical source binding, coordinate direction, bounded CDR and disk admission."""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
import prepare_autoware_frame_evidence as adapter


def parameters():
    result = {}
    for frame, _ in adapter.SENSORS:
        result[frame] = {'filter_transform_vlp16_' + frame[len('lidar_'):]: {'ros__parameters': dict(
            input_frame_id=frame, output_frame_id='base_link', timeout_ms=110,
            static_transformer=dict(translation=dict(x=1., y=-2., z=3.),
                                    quaternion=dict(x=0., y=0., z=.6, w=.8)))}}
    return result


def header(frame, seconds=123, nanos=45, order='<'):
    raw = frame.encode('utf-8') + b'\0'
    return (b'\0\1\0\0' if order == '<' else b'\0\0\0\0') + struct.pack(order + 'iII', seconds, nanos, len(raw)) + raw


def test_direction_keeps_parent_from_sensor_translation_and_xyzw():
    edges = adapter.extract_edges(yaml.safe_dump(parameters()))
    assert [edge['child_frame'] for edge in edges] == ['lidar_front', 'lidar_rear']
    for edge in edges:
        assert edge == dict(parent_frame='base_link', child_frame=edge['child_frame'],
                            translation_m=[1., -2., 3.], rotation_xyzw=[0., 0., .6, .8], stamp_nanos=None)
    # The input includes a nontrivial rotation; swapping/conjugating xyzw fails.
    assert edges[0]['rotation_xyzw'] != [0., 0., -.6, .8]


@pytest.mark.parametrize('case', ['frame', 'missing', 'bool', 'nan', 'inf', 'huge', 'nonunit', 'axis', 'namespace'])
def test_extraction_rejects_invalid_schema_and_transform(case):
    document = parameters()
    node = document['lidar_front']['filter_transform_vlp16_front']['ros__parameters']
    if case == 'frame': node['output_frame_id'] = 'lidar_front'
    if case == 'missing': del node['static_transformer']['translation']
    if case == 'bool': node['static_transformer']['translation']['x'] = True
    if case == 'nan': node['static_transformer']['quaternion']['x'] = float('nan')
    if case == 'inf': node['static_transformer']['translation']['x'] = float('inf')
    if case == 'huge': node['static_transformer']['translation']['x'] = 10**400
    if case == 'nonunit': node['static_transformer']['quaternion']['w'] = 0.
    if case == 'axis': node['static_transformer']['translation']['w'] = 1.
    if case == 'namespace': document['another_sensor'] = {}
    with pytest.raises(ValueError):
        adapter.extract_edges(yaml.safe_dump(document))


def test_yaml_never_constructs_python_objects():
    with pytest.raises(yaml.YAMLError):
        adapter.extract_edges('!!python/object/apply:os.system [echo forbidden]')


@pytest.mark.parametrize('order', ['<', '>'])
def test_cdr_header_endianness_and_exact_integer_time(order):
    assert adapter.decode_header(header('lidar_front', 1600901887, 522903301, order)) == (1600901887522903301, 'lidar_front')


@pytest.mark.parametrize('case', ['short', 'representation', 'seconds', 'nanos', 'length', 'nul', 'embedded', 'utf8'])
def test_cdr_rejects_malformed_or_unbounded_strings(case):
    raw = bytearray(header('lidar_front'))
    if case == 'short': raw = raw[:15]
    if case == 'representation': raw[1] = 3
    if case == 'seconds': struct.pack_into('<i', raw, 4, -1)
    if case == 'nanos': struct.pack_into('<I', raw, 8, 1_000_000_000)
    if case == 'length': struct.pack_into('<I', raw, 12, 129)
    if case == 'nul': raw[-1] = 1
    if case == 'embedded': raw[18] = 0
    if case == 'utf8': raw[18] = 255
    with pytest.raises(ValueError): adapter.decode_header(raw)


@pytest.fixture
def source_fixture(tmp_path, monkeypatch):
    configs = tmp_path / 'upstream'; configs.mkdir()
    pinned = {}
    for name, (relative, _, _) in adapter.SOURCES.items():
        raw = yaml.safe_dump(parameters()).encode() if name == 'filter.yaml' else b'fixture source: not executed\n'
        (configs / name).write_bytes(raw)
        pinned[name] = (relative, len(raw), hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(adapter, 'SOURCES', pinned)
    bag = tmp_path / 'source.db3'
    with sqlite3.connect(str(bag)) as connection:
        connection.executescript('CREATE TABLE topics(id INTEGER,name TEXT,type TEXT,serialization_format TEXT); CREATE TABLE messages(id INTEGER,topic_id INTEGER,timestamp INTEGER,data BLOB);')
        for index, (frame, count) in enumerate(adapter.SENSORS, 1):
            connection.execute('INSERT INTO topics VALUES(?,?,?,?)', (index, '/' + frame + '/points_raw', 'sensor_msgs/msg/PointCloud2', 'cdr'))
            connection.executemany('INSERT INTO messages VALUES(?,?,?,?)', ((index*10000+j, index, j, header(frame, nanos=j)) for j in range(count)))
    monkeypatch.setattr(adapter, 'BAG_BYTES', bag.stat().st_size)
    monkeypatch.setattr(adapter, 'BAG_SHA256', adapter.digest_file(bag))
    monkeypatch.setattr(adapter.shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(40*1024**3, 1, 30*1024**3))
    return bag, configs, tmp_path / 'evidence'


def test_complete_offline_preparation_preserves_inputs_and_binds_every_output(source_fixture):
    bag, configs, output = source_fixture
    before = {path: adapter.digest_file(path) for path in [bag] + list(configs.iterdir())}
    manifest = adapter.prepare(bag, configs, output)
    document = json.loads((output / 'frame-evidence.json').read_text())
    assert document['schema'] == 'spatialrust.calibration.frame-evidence'
    assert document['source'] == dict(path=str(bag), size_bytes=bag.stat().st_size, sha256=before[bag])
    assert len(document['edges']) == 2
    assert manifest['artifact']['sha256'] == adapter.digest_file(output / 'frame-evidence.json')
    assert [topic['messages'] for topic in manifest['topics']] == [768, 767]
    assert manifest['clock_artifact_generated'] is manifest['calibrated_mapping_accepted'] is False
    assert set(path.name for path in output.iterdir()) == {'frame-evidence.json', 'manifest.json'}
    assert before == {path: adapter.digest_file(path) for path in before}
    assert not list(bag.parent.glob('source.db3-*'))
    with pytest.raises(ValueError, match='overwrite'): adapter.prepare(bag, configs, output)


@pytest.mark.parametrize('case', ['bag_size', 'bag_hash', 'source_size', 'source_hash', 'output_exists', 'low_space', 'space_falls'])
def test_source_and_storage_admission_fail_before_output(source_fixture, monkeypatch, case):
    bag, configs, output = source_fixture
    if case == 'bag_size': monkeypatch.setattr(adapter, 'BAG_BYTES', bag.stat().st_size + 1)
    if case == 'bag_hash': monkeypatch.setattr(adapter, 'BAG_SHA256', '0'*64)
    if case == 'source_size': (configs / 'launch.py').write_bytes(b'changed')
    if case == 'source_hash':
        path = configs / 'launch.py'; path.write_bytes(b'X' + path.read_bytes()[1:])
    if case == 'output_exists': output.mkdir(); (output / 'keep').write_text('existing')
    if case == 'low_space':
        monkeypatch.setattr(adapter.shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(1, 1, 0))
        monkeypatch.setattr(adapter, 'digest_file', lambda _: pytest.fail('bag opened before disk preflight'))
    if case == 'space_falls':
        free = iter([30*1024**3, adapter.MIN_FREE_BYTES - 1])
        monkeypatch.setattr(adapter.shutil, 'disk_usage', lambda _: shutil._ntuple_diskusage(1, 1, next(free)))
    with pytest.raises(ValueError): adapter.prepare(bag, configs, output)
    if case == 'output_exists': assert (output / 'keep').read_text() == 'existing'
    else: assert not output.exists()


@pytest.mark.parametrize('case', ['count', 'frame', 'time', 'schema', 'duplicate_topic'])
def test_inventory_rejects_wrong_source_structure(source_fixture, case):
    bag, _, _ = source_fixture
    with sqlite3.connect(str(bag)) as connection:
        if case == 'count': connection.execute('DELETE FROM messages WHERE id=10000')
        if case == 'frame': connection.execute('UPDATE messages SET data=? WHERE id=10000', (header('other'),))
        if case == 'time': connection.execute('UPDATE messages SET data=? WHERE id=10000', (header('lidar_front', nanos=100),))
        if case == 'schema': connection.execute("UPDATE topics SET serialization_format='other' WHERE id=1")
        if case == 'duplicate_topic': connection.execute('INSERT INTO topics SELECT 99,name,type,serialization_format FROM topics WHERE id=1')
    with pytest.raises(ValueError): adapter.inventory_bag(bag)


def test_relative_and_workspace_paths_are_rejected():
    for path in (Path('relative'), adapter.WORKSPACE, adapter.WORKSPACE / 'target' / 'raw'):
        with pytest.raises(ValueError): adapter.outside_workspace(path)
