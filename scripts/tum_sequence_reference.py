"""Bounded full-sequence TUM input preparation, excluding reference pose contents."""
from bisect import bisect_left
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import zlib

from timestamped_trajectory_reference import decimal_timestamp_ns


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def require_sha256(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('require a complete lowercase SHA-256')
    return value


def validate_plan(plan):
    # This protocol is specific to the declared official sequence and optical convention.
    if plan.get('schema') != 'spatialrust.tum-sequence-plan.v1' or plan.get('sequence') != 'rgbd_dataset_freiburg1_xyz':
        raise ValueError('unsupported TUM sequence protocol')
    c = plan['calibration']
    for name, value in dict(width=640, height=480, fx=525., fy=525., cx=319.5, cy=239.5,
                            depth_unit_m=.0002, sensor_frame='tum_rgb_optical',
                            distortion_applied=False, extrinsic_applied=False).items():
        if c.get(name) != value or (isinstance(value, bool) and type(c.get(name)) is not bool):
            raise ValueError('unsupported registered-depth calibration')
    require_sha256(plan['publisher_formats_sha256'])
    for key, maximum in dict(max_frames=10000, max_members=30000, max_member_bytes=16777216,
                             max_total_bytes=8589934592).items():
        v = plan['limits'][key]
        if type(v) is not int or not 1 <= v <= maximum:
            raise ValueError('invalid archive/frame bound')
    controls = plan['controls']
    for name, maximum in dict(rgb_max_gap_ns=100000000, max_tracking_gap_ns=1000000000,
                              io_stride=10000, min_points=307200,
                              evaluation_max_gap_ns=100000000, rpe_delta_ns=10000000000).items():
        if type(controls[name]) is not int or not 1 <= controls[name] <= maximum:
            raise ValueError('invalid integer control')
    for name in ('min_depth_m', 'max_depth_m', 'min_support_fraction', 'projection_tolerance_m'):
        value = controls[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 100:
            raise ValueError('invalid finite geometric control')
    if not controls['min_depth_m'] < controls['max_depth_m'] or controls['min_support_fraction'] > 1:
        raise ValueError('invalid depth range or support fraction')
    stages = controls['stages']
    if not isinstance(stages, list) or not 1 <= len(stages) <= 8:
        raise ValueError('require a bounded ICP schedule')
    leaf, gate = float('inf'), float('inf')
    for s in stages:
        if set(s) != {'leaf_m', 'gate_m', 'iterations'}:
            raise ValueError('unsupported ICP stage fields')
        for key, previous in (('leaf_m', leaf), ('gate_m', gate)):
            v = s[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 < v <= min(previous, 10):
                raise ValueError('require finite decreasing stage sizes')
        if type(s['iterations']) is not int or not 1 <= s['iterations'] <= 1000:
            raise ValueError('invalid iteration bound')
        leaf, gate = s['leaf_m'], s['gate_m']
    return plan


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name or any(ord(c) < 32 for c in name):
        raise ValueError('invalid archive/image path')
    p = PurePosixPath(name)
    if p.is_absolute() or any(s in ('', '.', '..') for s in name.split('/')):
        raise ValueError('archive/image path escapes or aliases the root')
    return p


def read_image_index(data, kind, max_frames):
    rows, names = [], set()
    for line in data.decode('ascii').splitlines():
        fields = line.split('#', 1)[0].split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError('image index requires timestamp and relative PNG path')
        ns = decimal_timestamp_ns(fields[0])
        p = safe_relative(fields[1])
        if len(p.parts) != 2 or p.parts[0] != kind or p.suffix != '.png':
            raise ValueError('image index names the wrong stream')
        if fields[1] in names or (rows and ns <= rows[-1]['timestamp_ns']):
            raise ValueError('duplicate image or non-increasing timestamp')
        rows.append(dict(timestamp_ns=ns, path=fields[1]))
        names.add(fields[1])
        if len(rows) > max_frames:
            raise ValueError('complete image index exceeds declared frame budget')
    if not rows:
        raise ValueError('empty image index')
    return rows


def associate_rgb(depth, rgb, max_gap_ns):
    times = [r['timestamp_ns'] for r in rgb]
    frames = []
    for d in depth:
        i = bisect_left(times, d['timestamp_ns'])
        candidates = [j for j in (i-1, i) if 0 <= j < len(rgb)]
        j = min(candidates, key=lambda j: (abs(times[j]-d['timestamp_ns']), j))
        gap = times[j]-d['timestamp_ns']
        match = rgb[j] if abs(gap) <= max_gap_ns else None
        frames.append(dict(timestamp_ns=d['timestamp_ns'], depth=dict(d), rgb=match,
                           rgb_delta_ns=gap if match else None))
    return frames


def png_header(path, kind, camera):
    # Header/layout check only: complete decoding remains the generation runner's job.
    with Path(path).open('rb') as stream:
        header = stream.read(33)
    if len(header) != 33 or header[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR':
        raise ValueError('invalid PNG header')
    if zlib.crc32(header[12:29]) & 0xffffffff != struct.unpack('>I', header[29:33])[0]:
        raise ValueError('PNG IHDR checksum mismatch')
    w, h, bits, color, compression, filtering, interlace = struct.unpack('>IIBBBBB', header[16:29])
    if (w, h) != (camera['width'], camera['height']) or (bits, color) != ((16, 0) if kind == 'depth' else (8, 2)):
        raise ValueError('unsupported registered PNG dimensions/type')
    if compression or filtering or interlace:
        raise ValueError('unsupported PNG encoding')


def prepare(archive, expected_sha256, plan_path, output):
    require_sha256(expected_sha256)
    plan_raw = Path(plan_path).read_bytes()
    plan = validate_plan(json.loads(plan_raw))
    if file_sha256(archive) != expected_sha256:
        raise ValueError('archive differs from supplied frozen SHA-256')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    limits, root = plan['limits'], plan['sequence']
    names, files, withheld = set(), {}, []
    total = 0
    # Streaming gzip/tar, bounded member count and expanded size. No extractall.
    with tarfile.open(archive, mode='r|gz') as source:
        for member in source:
            name = member.name.rstrip('/') if member.isdir() else member.name
            p = safe_relative(name)
            if len(names) >= limits['max_members'] or name in names or p.parts[0] != root:
                raise ValueError('duplicate, wrong-root or over-budget archive member')
            names.add(name)
            if member.isdir():
                if tuple(p.parts[1:]) not in ((), ('rgb',), ('depth',)):
                    raise ValueError('unexpected archive directory')
                continue
            if not member.isfile() or not 0 <= member.size <= limits['max_member_bytes']:
                raise ValueError('links, special files and oversized members are forbidden')
            total += member.size
            if total > limits['max_total_bytes']:
                raise ValueError('archive expanded size exceeds bound')
            relative = PurePosixPath(*p.parts[1:])
            if str(relative) in ('groundtruth.txt', 'accelerometer.txt', 'README', 'README.txt'):
                withheld.append(dict(path=str(relative), bytes=member.size, excluded_from_generation=True))
                continue
            image = len(relative.parts) == 2 and relative.parts[0] in ('rgb', 'depth') and relative.suffix == '.png'
            if not image and str(relative) not in ('depth.txt', 'rgb.txt'):
                raise ValueError('unexpected archive file')
            destination = output / str(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            count = 0
            with source.extractfile(member) as incoming, destination.open('xb') as outgoing:
                for chunk in iter(lambda: incoming.read(1048576), b''):
                    count += len(chunk)
                    digest.update(chunk)
                    outgoing.write(chunk)
            if count != member.size:
                raise ValueError('truncated archive member')
            if image:
                png_header(destination, relative.parts[0], plan['calibration'])
            files[str(relative)] = dict(path=str(relative), bytes=count, sha256=digest.hexdigest())
    if not {'depth.txt', 'rgb.txt'} <= set(files):
        raise ValueError('missing image indexes')
    depth = read_image_index((output/'depth.txt').read_bytes(), 'depth', limits['max_frames'])
    rgb = read_image_index((output/'rgb.txt').read_bytes(), 'rgb', limits['max_frames'])
    listed = {r['path'] for r in depth+rgb}
    available = {p for p in files if p.endswith('.png')}
    if listed != available:
        raise ValueError('missing or unindexed images; cannot claim a complete sequence')
    for row in depth+rgb:
        row.update(files[row['path']])
    frames = associate_rgb(depth, rgb, plan['controls']['rgb_max_gap_ns'])
    calibration_raw = json_bytes(plan['calibration'])
    (output/'calibration.json').write_bytes(calibration_raw)
    (output/'plan.json').write_bytes(plan_raw)
    manifest = dict(schema='spatialrust.tum-prepared.v1', sequence=root,
                    source_sha256=expected_sha256, plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
                    calibration_sha256=hashlib.sha256(calibration_raw).hexdigest(),
                    reference_used_for_generation=False, archive_member_count=len(names),
                    archive_expanded_bytes=total, files=files, withheld_members=withheld,
                    frames=frames, rgb_frames=len(rgb), depth_frames=len(depth),
                    rgb_unmatched_depth_frames=sum(f['rgb'] is None for f in frames),
                    image_validation='PNG header only; full decoding occurs during generation',
                    bindings={str(Path(__file__).resolve()):file_sha256(__file__),
                              str(Path(__file__).with_name('timestamped_trajectory_reference.py').resolve()):
                              file_sha256(Path(__file__).with_name('timestamped_trajectory_reference.py'))})
    if file_sha256(archive) != expected_sha256 or Path(plan_path).read_bytes() != plan_raw:
        raise ValueError('source archive or plan changed during preparation')
    with (output/'manifest.json').open('xb') as stream:
        stream.write(json_bytes(manifest))
    return manifest
