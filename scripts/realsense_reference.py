"""Bounded decoding and source-bound metadata for RealSense ROS1 recordings.

rosbags is an optional benchmark reader, imported only by inventory(). No ROS
runtime is required. Recorded hardware timestamps do not prove synchronization.
"""
import hashlib
from pathlib import Path

import numpy as np


def file_hash(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def timestamp_ns(stamp):
    if type(stamp.sec) is not int or type(stamp.nanosec) is not int or stamp.sec < 0 or not 0 <= stamp.nanosec < 10**9:
        raise ValueError('invalid integer sensor timestamp')
    return stamp.sec * 10**9 + stamp.nanosec


def decode_depth(message, max_pixels):
    h, w, step = message.height, message.width, message.step
    if any(type(x) is not int for x in (h, w, step)) or not 0 < h * w <= max_pixels or min(h, w) < 1:
        raise ValueError('invalid or oversized depth dimensions')
    if message.encoding != 'mono16' or message.is_bigendian not in (0, 1) or step < 2 * w or step % 2:
        raise ValueError('unsupported depth encoding, endian, or row stride')
    data = np.asarray(message.data)
    if data.dtype != np.uint8 or data.ndim != 1 or len(data) != h * step:
        raise ValueError('depth payload does not match its row stride')
    raw = data.tobytes()
    # Explicit endian and padded-row decoding; native projection receives f32.
    depth = np.ndarray((h, w), dtype='>u2' if message.is_bigendian else '<u2',
                       buffer=raw, strides=(step, 2)).astype(np.uint16)
    return depth


def camera_parameters(message):
    k, d = np.asarray(message.K, dtype=float), np.asarray(message.D, dtype=float)
    if k.shape != (9,) or d.shape != (5,) or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise ValueError('invalid camera calibration')
    if message.distortion_model != 'None' or np.any(d != 0):
        raise ValueError('depth distortion is unsupported; do not assume rectification')
    if not np.array_equal(k[[1, 3, 6, 7, 8]], [0, 0, 0, 0, 1]) or min(k[0], k[4]) <= 0:
        raise ValueError('unsupported intrinsic matrix')
    w, h = message.width, message.height
    if type(w) is not int or type(h) is not int or min(w, h) < 1 or not 0 <= k[2] < w or not 0 <= k[5] < h:
        raise ValueError('invalid intrinsic dimensions or principal point')
    return dict(width=w, height=h, fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]),
                distortion_model=message.distortion_model, coefficients=d.tolist(), intrinsic_matrix=k.tolist())


def reference_geometry(depth, camera, units, controls):
    """Independent f64 pinhole formula in optical x-right/y-down/z-forward axes."""
    if not np.isfinite(units) or units <= 0:
        raise ValueError('invalid recorded depth units')
    z = depth.astype(np.float64) * units
    valid = (z >= controls['min_depth_m']) & (z <= controls['max_depth_m']) & (depth > 0)
    v, u = np.nonzero(valid)
    points = np.column_stack(((u-camera['cx']) * z[valid] / camera['fx'],
                              (v-camera['cy']) * z[valid] / camera['fy'], z[valid]))
    return valid, points


def save_attribute_input(path, xyz, depth, valid):
    """Binary PCD retains original depth counts and pixel identities beside xyz."""
    v, u = np.nonzero(valid)
    dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                      ('raw_depth', '<u2'), ('pixel_u', '<u4'), ('pixel_v', '<u4')])
    records = np.empty(len(xyz), dtype=dtype)
    for i, name in enumerate(('x', 'y', 'z')):
        records[name] = xyz[:, i]
    records['raw_depth'], records['pixel_u'], records['pixel_v'] = depth[valid], u, v
    header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z raw_depth pixel_u pixel_v\n'
              'SIZE 4 4 4 2 4 4\nTYPE F F F U U U\nCOUNT 1 1 1 1 1 1\n'
              f'WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n')
    with Path(path).open('xb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(records.tobytes())
    return records


def check_attributes(cloud, expected):
    import pyarrow as pa
    # C Data export checks the native schema and typed attribute values, not only xyz.
    actual = pa.array(cloud)
    if set(actual.type.names) != set(expected.dtype.names):
        raise ValueError('cloud attribute schema changed')
    for name in expected.dtype.names:
        values = actual.field(name).to_numpy()
        if values.dtype != expected[name].dtype:
            raise ValueError(f'cloud attribute type changed: {name}')
        np.testing.assert_array_equal(values, expected[name])


def inventory(path, stream, max_frames):
    from rosbags.highlevel import AnyReader
    frames, metadata, cameras, transforms, units = [], {}, {}, {}, []
    with AnyReader([Path(path)]) as reader:
        connections = {c.topic: c for c in reader.connections}
        topic = stream + '/image/data'
        if topic not in connections or not 0 < connections[topic].msgcount <= max_frames:
            raise ValueError('missing depth stream or frame bound exceeded')
        selected = [c for c in reader.connections if c.topic == topic or
                    c.topic == stream + '/image/metadata' or c.topic.endswith('/info/camera_info') or
                    '/tf/' in c.topic or c.topic.endswith('/Depth_Units/value')]
        for connection, bag_ns, raw in reader.messages(connections=selected):
            message = reader.deserialize(raw, connection.msgtype)
            name = connection.topic
            if name == topic:
                frames.append(dict(index=len(frames), bag_timestamp_ns=bag_ns,
                                   sensor_timestamp_ns=timestamp_ns(message.header.stamp),
                                   sequence=message.header.seq, header_frame_id=message.header.frame_id,
                                   serialized_sha256=hashlib.sha256(raw).hexdigest()))
            elif name == stream + '/image/metadata':
                values = metadata.setdefault(bag_ns, {})
                if message.key in values:
                    raise ValueError('duplicate per-frame metadata key')
                values[message.key] = message.value
            elif name.endswith('/info/camera_info'):
                if name in cameras:
                    raise ValueError('changing camera calibration is unsupported')
                cameras[name] = dict(bag_timestamp_ns=bag_ns, serialized_sha256=hashlib.sha256(raw).hexdigest(),
                    width=message.width, height=message.height, distortion_model=message.distortion_model,
                    D=message.D.tolist(), K=message.K.tolist(), R=message.R.tolist(), P=message.P.tolist())
                if name == stream + '/info/camera_info':
                    depth_camera = camera_parameters(message)
            elif name.endswith('/Depth_Units/value'):
                units.append(dict(bag_timestamp_ns=bag_ns, meters_per_unit=float(message.data),
                                  serialized_sha256=hashlib.sha256(raw).hexdigest()))
            else:
                if name in transforms:
                    raise ValueError('changing stream extrinsics are unsupported')
                t, q = message.translation, message.rotation
                values = [t.x, t.y, t.z, q.x, q.y, q.z, q.w]
                if not np.isfinite(values).all() or abs(np.linalg.norm(values[3:]) - 1) > 1e-5:
                    raise ValueError('invalid recorded stream transform')
                transforms[name] = dict(bag_timestamp_ns=bag_ns, translation_m=values[:3], quaternion_xyzw=values[3:],
                                        serialized_sha256=hashlib.sha256(raw).hexdigest(), applied=False)
        if len(frames) != connections[topic].msgcount or stream + '/info/camera_info' not in cameras or len(units) != 1:
            raise ValueError('incomplete depth/calibration inventory')
        if units[0]['bag_timestamp_ns'] > frames[0]['bag_timestamp_ns'] or cameras[stream+'/info/camera_info']['bag_timestamp_ns'] > frames[0]['bag_timestamp_ns']:
            raise ValueError('calibration first appears after depth acquisition')
        if not np.isfinite(units[0]['meters_per_unit']) or units[0]['meters_per_unit'] <= 0:
            raise ValueError('invalid recorded depth units')
        for previous, current in zip(frames, frames[1:]):
            if any(current[k] <= previous[k] for k in ('bag_timestamp_ns', 'sensor_timestamp_ns', 'sequence')):
                raise ValueError('nonmonotonic frame time or sequence')
        for frame in frames:
            frame['metadata'] = metadata.pop(frame['bag_timestamp_ns'], {})
        if metadata:
            raise ValueError('depth metadata has no corresponding frame')
        return dict(frames=frames, camera=depth_camera, camera_info=cameras, recorded_transforms=transforms,
                    depth_units=units[0], topic_counts={c.topic: c.msgcount for c in reader.connections},
                    duration_ns=reader.duration, source_sha256=file_hash(path),
                    coordinate_frame='depth optical: x right, y down, z forward; no extrinsic applied',
                    clock=dict(declared_domains=sorted({f['metadata'].get('timestamp_domain', 'unknown') for f in frames}),
                               measured_synchronization_available=False),
                    independent_trajectory_truth_available=False, color_alignment_performed=False)
