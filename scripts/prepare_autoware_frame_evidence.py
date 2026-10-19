"""Bind publisher-configured Autoware extrinsics to the exact public canonical bag.

Offline only: supply the six hash-pinned upstream files listed in SOURCES.
This emits frame evidence and provenance, never a clock calibration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import struct


WORKSPACE = Path(__file__).resolve().parents[1]
REVISION = '15d2e91a2b3c8cd61b20b7e6e79da32919c2cbcd'
UPSTREAM = 'https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto/-/raw/' + REVISION + '/'
BAG_BYTES = 713670656
BAG_SHA256 = 'b00d31e25dc0b53cba89cfbe16e5b118079c514a1d8c6f4089fac9c0e3ffd7c8'
MIN_FREE_BYTES = 20 * 1024**3
# These complete byte identities anchor the manually audited launch association
# and transform direction. Downloaded Python/C++ is never imported or executed.
SOURCES = {
    'launch.py': ('src/launch/autoware_demos/launch/localization_rosbag.launch.py', 8450,
                  '5af895daa6b87aef2db5fa38e1f500083fb5fcf17f68e57a3f81a8fbebfa5cea'),
    'filter.yaml': ('src/perception/filters/point_cloud_filter_transform_nodes/param/vlp16_lexus_filter_transform.param.yaml', 1669,
                    'af1e4edcdf0640fd11097dfa26de3969f1233311172511687d79433684d61ebf'),
    'filter-transform.cpp': ('src/perception/filters/point_cloud_filter_transform_nodes/src/point_cloud_filter_transform_node.cpp', 8187,
                             '327cf688c9ca7c84f3a2363fb92805edbd39571bffc38a0490fe757087635df5'),
    'filter-node.hpp': ('src/perception/filters/point_cloud_filter_transform_nodes/include/point_cloud_filter_transform_nodes/point_cloud_filter_transform_node.hpp', 4615,
                       '6d4a3ff8ff811dc18f9c825438d826101ffe5d3f9e716d445749fd67c1079b45'),
    'point-cloud-utils.cpp': ('src/common/lidar_utils/src/point_cloud_utils.cpp', 10465,
                             'b8c8eb65a9572129c20ebb22dc2cb9318b488c14cd20a386c9d7e5f88e635d1e'),
    'point-cloud-utils.hpp': ('src/common/lidar_utils/include/lidar_utils/point_cloud_utils.hpp', 12619,
                             '9130ceee210c0f7bb8c822680c8a45883cc1593d447b1b412e5f4d6dbc2ea511'),
}
SENSORS = (('lidar_front', 768), ('lidar_rear', 767))


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def outside_workspace(path):
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('require an absolute external path')
    resolved = path.resolve()
    if resolved == WORKSPACE or WORKSPACE in resolved.parents:
        raise ValueError('source and output must remain outside the code workspace')
    return resolved


def checked_sources(directory):
    sources, payloads = [], {}
    for name, (relative, size, checksum) in SOURCES.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != directory:
            raise ValueError('require a regular pinned upstream file: ' + name)
        if path.stat().st_size != size:
            raise ValueError('upstream source size changed: ' + name)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError('upstream source checksum changed: ' + name)
        sources.append(dict(path=str(path), bytes=size, sha256=checksum, url=UPSTREAM + relative))
        payloads[name] = raw
    return sources, payloads


def numeric_vector(mapping, keys):
    if not isinstance(mapping, dict) or set(mapping) != set(keys):
        raise ValueError('transform components must have exactly the declared axes')
    values = [mapping[key] for key in keys]
    if any(type(value) not in (int, float) for value in values):
        raise ValueError('transform components must be finite numbers, not booleans')
    try:
        values = [float(value) for value in values]
    except OverflowError as error:
        raise ValueError('transform component outside finite float64 range') from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError('transform components must be finite numbers')
    return values


def extract_edges(raw):
    # The caller verifies complete source bytes before any YAML interpretation.
    import yaml
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or set(document) != {name for name, _ in SENSORS}:
        raise ValueError('require exactly the publisher front/rear sensor namespaces')
    edges = []
    for frame, _ in SENSORS:
        node = 'filter_transform_vlp16_' + frame[len('lidar_'):]
        try:
            parameters = document[frame][node]['ros__parameters']
            transform = parameters['static_transformer']
            if parameters['input_frame_id'] != frame or parameters['output_frame_id'] != 'base_link':
                raise ValueError('publisher input/output frame direction changed')
            translation = numeric_vector(transform['translation'], 'xyz')
            rotation = numeric_vector(transform['quaternion'], 'xyzw')
        except (KeyError, TypeError) as error:
            raise ValueError('incomplete publisher transform') from error
        if abs(math.sqrt(sum(value * value for value in rotation)) - 1.) > 1e-6:
            raise ValueError('publisher quaternion must be unit length within rounding tolerance')
        # StaticTransformer applies R*p_input+t and labels output base_link.
        # Preserve raw rounded values; the existing Rust mapper normalizes q.
        edges.append(dict(parent_frame='base_link', child_frame=frame,
                          translation_m=translation, rotation_xyzw=rotation, stamp_nanos=None))
    return edges


def decode_header(raw):
    """Read only a bounded ROS2 CDR Header prefix; never materialize cloud data."""
    if len(raw) < 17 or raw[:4] not in (b'\0\1\0\0', b'\0\0\0\0'):
        raise ValueError('unsupported or truncated CDR header')
    order = '<' if raw[1] == 1 else '>'
    seconds, nanos, size = struct.unpack_from(order + 'iII', raw, 4)
    if seconds < 0 or nanos >= 1_000_000_000 or not 1 <= size <= 128 or 16 + size > len(raw):
        raise ValueError('invalid or truncated CDR timestamp/frame string')
    frame = raw[16:16 + size]
    if frame[-1] != 0 or b'\0' in frame[:-1]:
        raise ValueError('CDR frame string must have one terminating NUL')
    return seconds * 1_000_000_000 + nanos, frame[:-1].decode('utf-8', errors='strict')


def inventory_bag(path):
    inventory = []
    with sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True) as connection:
        if connection.execute('SELECT COUNT(*) FROM messages').fetchone()[0] != sum(count for _, count in SENSORS):
            raise ValueError('canonical message count changed')
        for frame, count in SENSORS:
            name = '/' + frame + '/points_raw'
            topics = connection.execute('SELECT id,type,serialization_format FROM topics WHERE name=?', (name,)).fetchall()
            if len(topics) != 1 or topics[0][1:] != ('sensor_msgs/msg/PointCloud2', 'cdr'):
                raise ValueError('canonical PointCloud2 topic/schema changed')
            actual, stamps, previous = 0, [], None
            for (prefix,) in connection.execute('SELECT substr(data,1,256) FROM messages WHERE topic_id=? ORDER BY timestamp,id', (topics[0][0],)):
                stamp, observed = decode_header(prefix)
                if observed != frame or (previous is not None and stamp < previous):
                    raise ValueError('unexpected sensor frame or decreasing header timestamp')
                previous = stamp
                stamps.append(stamp)
                actual += 1
                if actual > count:
                    raise ValueError('canonical sensor count exceeded')
            if actual != count:
                raise ValueError('canonical sensor message count changed')
            inventory.append(dict(topic=name, frame_id=frame, messages=actual,
                                  header_min_ns=min(stamps), header_max_ns=max(stamps)))
    return inventory


def prepare(bag, upstream_dir, output):
    bag, upstream_dir = outside_workspace(bag), outside_workspace(upstream_dir)
    original_output = Path(output)
    output = outside_workspace(original_output)
    if original_output.is_symlink() or output.exists():
        raise ValueError('refuse to overwrite an evidence directory')
    if not output.parent.is_dir():
        raise ValueError('output parent must already exist')
    # Check the output filesystem before opening even the read-only bag.
    free_before = shutil.disk_usage(output.parent).free
    if free_before < MIN_FREE_BYTES:
        raise ValueError('output filesystem has less than the 20 GiB free-space floor')
    if not bag.is_file() or bag.stat().st_size != BAG_BYTES:
        raise ValueError('canonical bag size changed')
    if digest_file(bag) != BAG_SHA256:
        raise ValueError('canonical bag checksum changed')
    sources, payloads = checked_sources(upstream_dir)
    edges = extract_edges(payloads['filter.yaml'])
    topics = inventory_bag(bag)
    if digest_file(bag) != BAG_SHA256 or checked_sources(upstream_dir)[1] != payloads:
        raise ValueError('source changed during preparation')
    document = dict(schema='spatialrust.calibration.frame-evidence', version=1,
                    source=dict(path=str(bag), size_bytes=BAG_BYTES, sha256=BAG_SHA256),
                    method='Publisher-configured Autoware.Auto ' + REVISION + '; independent physical calibration accuracy unverified',
                    root_frame='base_link', edges=edges)
    raw = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    # Recheck after hashing/inventory; create the run directory exclusively.
    free_at_creation = shutil.disk_usage(output.parent).free
    if free_at_creation < MIN_FREE_BYTES + len(raw) + 65536:
        raise ValueError('output filesystem no longer meets the 20 GiB floor')
    output.mkdir()
    with (output / 'frame-evidence.json').open('xb') as stream:
        stream.write(raw)
    manifest = dict(schema='spatialrust.autoware-publisher-frame-provenance.v1',
                    source=document['source'], upstream_revision=REVISION, upstream_files=sources,
                    preparation_sha256=digest_file(Path(__file__)), topics=topics,
                    artifact=dict(path=str(output / 'frame-evidence.json'), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()),
                    transform_convention='p_base_link=R(q_xyzw)*p_lidar+t; raw publisher values preserved; Rust mapping normalizes q',
                    quaternion_norms={edge['child_frame']: math.sqrt(sum(v*v for v in edge['rotation_xyzw'])) for edge in edges},
                    evidence_level='publisher demo configuration; not independently measured physical accuracy',
                    clock_artifact_generated=False, calibrated_mapping_accepted=False,
                    clock_boundary='Header/recording delay does not independently measure sensor clock offset or drift',
                    min_output_free_bytes=MIN_FREE_BYTES, free_before_bytes=free_before,
                    free_at_creation_bytes=free_at_creation, free_after_artifact_bytes=shutil.disk_usage(output).free)
    with (output / 'manifest.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', required=True, type=Path)
    parser.add_argument('--upstream-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        manifest = prepare(args.bag, args.upstream_dir, args.output_dir)
    except (ValueError, OSError, sqlite3.Error) as error:
        parser.exit(1, str(error) + '\n')
    print(json.dumps(dict(artifact=manifest['artifact'], clock_artifact_generated=False)))


if __name__ == '__main__':
    main()
