"""Run all depth frames from the committed source-bound RealSense operation plan."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import spatialrust as sr

from realsense_reference import (check_attributes, decode_depth, file_hash,
                                inventory, reference_geometry, save_attribute_input, timestamp_ns)


def bindings():
    paths = [Path(__file__), Path(__file__).with_name('realsense_reference.py')]
    native = list(Path(sr.__file__).parent.glob('*.so'))
    if len(native) != 1:
        raise ValueError('expected one Linux native extension')
    paths += native
    return {str(p.resolve()): file_hash(p) for p in paths}


def validate_plan(plan):
    if plan.get('schema') != 'spatialrust.realsense-operation-plan.v1':
        raise ValueError('unsupported operation plan')
    c = plan['controls']
    if not 0 < c['min_depth_m'] < c['max_depth_m'] or any(not np.isfinite(c[k]) or c[k] <= 0 for k in
            ('projection_tolerance_m', 'leaf_size_m', 'plane_distance_m', 'cluster_tolerance_m')):
        raise ValueError('invalid geometric controls')
    if any(type(c[k]) is not int or c[k] <= 0 for k in
            ('save_stride', 'pipeline_stride', 'min_cluster_size', 'max_pixels', 'max_frames')):
        raise ValueError('invalid frame/resource controls')
    for source in plan['inputs']:
        if Path(source['name']).name != source['name'] or Path(source['archive']).name != source['archive']:
            raise ValueError('input names must be basenames')
        if len(source['md5']) != 32 or len(source['sha256']) != 64:
            raise ValueError('missing source hashes')
    return c


def verified_source(data_dir, source):
    path = data_dir / source['archive']
    if file_hash(path, 'md5') != source['md5'] or file_hash(path) != source['sha256']:
        raise ValueError('input does not match the official descriptor and frozen SHA-256')
    if source['archive'] == source['name']:
        return path
    # No general-purpose extraction: exactly one known, bounded bag member.
    bag_path = data_dir / source['name']
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != [source['name']] or archive.getinfo(source['name']).file_size > 128*1024*1024:
            raise ValueError('unexpected ZIP members or excessive bag size')
        data = archive.read(source['name'])
    if bag_path.exists():
        if bag_path.read_bytes() != data:
            raise ValueError('extracted bag no longer matches the verified archive')
    else:
        bag_path.write_bytes(data)
    return bag_path


def run_source(path, plan, output_dir):
    from rosbags.highlevel import AnyReader
    controls = plan['controls']
    output_dir.mkdir()
    info = inventory(path, plan['depth_stream'], controls['max_frames'])
    (output_dir / 'inventory.json').write_text(json.dumps(info, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    camera = info['camera']
    units = info['depth_units']['meters_per_unit']
    output_buffer = np.empty((camera['height'], camera['width'], 3), np.float32)
    rows = []
    started = time.perf_counter()
    with AnyReader([path]) as reader:
        selected = [c for c in reader.connections if c.topic == plan['depth_stream']+'/image/data']
        for index, (connection, bag_ns, raw) in enumerate(reader.messages(connections=selected)):
            frame = info['frames'][index]
            row = dict(index=index, bag_timestamp_ns=bag_ns, sensor_timestamp_ns=frame['sensor_timestamp_ns'], status='error')
            frame_started = time.perf_counter()
            try:
                if hashlib.sha256(raw).hexdigest() != frame['serialized_sha256'] or bag_ns != frame['bag_timestamp_ns']:
                    raise ValueError('frame source changed since inventory')
                message = reader.deserialize(raw, connection.msgtype)
                if timestamp_ns(message.header.stamp) != frame['sensor_timestamp_ns']:
                    raise ValueError('sensor timestamp changed')
                depth = decode_depth(message, controls['max_pixels'])
                if depth.shape != output_buffer.shape[:2]:
                    raise ValueError('image dimensions differ from source-bound calibration')
                native = sr.depth_to_xyz(depth.astype(np.float32), camera['fx'], camera['fy'], camera['cx'], camera['cy'],
                    depth_scale=units, min_depth=controls['min_depth_m'], max_depth=controls['max_depth_m'], out=output_buffer)
                if native is not output_buffer:
                    raise ValueError('dense output buffer was not reused')
                valid, expected = reference_geometry(depth, camera, units, controls)
                np.testing.assert_array_equal(np.isfinite(native).all(axis=2), valid)
                xyz = native[valid]
                np.testing.assert_allclose(xyz, expected, rtol=0, atol=controls['projection_tolerance_m'])
                if not len(xyz):
                    raise ValueError('no usable points after fixed depth bounds')
                row.update(points=len(xyz), invalid_pixels=int(depth.size-len(xyz)),
                           max_projection_error_m=float(np.max(np.abs(xyz-expected))))
                if index % controls['save_stride'] == 0 or index == len(info['frames'])-1:
                    original_path = output_dir / f'frame-{index:05d}-input.pcd'
                    expected_fields = save_attribute_input(original_path, xyz, depth, valid)
                    cloud = sr.read(str(original_path))
                    check_attributes(cloud, expected_fields)
                    restored_path = output_dir / f'frame-{index:05d}-restored.pcd'
                    sr.write(str(restored_path), cloud)
                    check_attributes(sr.read(str(restored_path)), expected_fields)
                    row['io'] = dict(input_sha256=file_hash(original_path), restored_sha256=file_hash(restored_path),
                                     fields=list(expected_fields.dtype.names), exact_typed_roundtrip=True)
                if index % controls['pipeline_stride'] == 0:
                    pipeline = sr.run_pipeline(sr.PointCloud.from_xyz(xyz), leaf_size=controls['leaf_size_m'],
                        plane_distance=controls['plane_distance_m'], cluster_tolerance=controls['cluster_tolerance_m'],
                        min_cluster_size=controls['min_cluster_size'])
                    row['pipeline'] = dict(output_points=len(pipeline.output), plane_inliers=pipeline.plane_inliers,
                                           clusters=pipeline.cluster_count)
                row['status'] = 'success'
            except (ValueError, RuntimeError, AssertionError) as error:
                row['error'] = str(error)
            row['elapsed_seconds'] = time.perf_counter()-frame_started
            rows.append(row)
            if index % 120 == 0:
                print(f'{path.name}: {index+1}/{len(info["frames"])} frames, failures={sum(r["status"]=="error" for r in rows)}', flush=True)
            # Preserve every completed outcome if an execution is interrupted.
            with (output_dir/'frames.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
    if len(rows) != len(info['frames']) or file_hash(path) != info['source_sha256']:
        raise ValueError('frame denominator or source hash changed')
    result = dict(source_sha256=info['source_sha256'], frame_count=len(rows),
                  successful_frames=sum(r['status']=='success' for r in rows),
                  total_projected_points=sum(r.get('points', 0) for r in rows),
                  io_verified_frames=sum('io' in r for r in rows), pipeline_frames=sum('pipeline' in r for r in rows),
                  max_projection_error_m=max((r.get('max_projection_error_m',0) for r in rows), default=0),
                  elapsed_seconds=time.perf_counter()-started, clock=info['clock'],
                  ground_truth_available=False, rows=rows, inventory_sha256=file_hash(output_dir/'inventory.json'))
    (output_dir/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def main():
    import resource  # This operational runner targets the prepared Linux environment.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory already exists; preserve previous outcomes')
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    validate_plan(plan)
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        parser.error('set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1')
    source_bindings = bindings()
    sources = [verified_source(args.data_dir, source) for source in plan['inputs']]
    args.output_dir.mkdir()
    results = [run_source(path, plan, args.output_dir/path.stem) for path in sources]
    if bindings() != source_bindings or args.plan.read_bytes() != plan_bytes:
        raise ValueError('runner, native extension, or plan changed')
    result = dict(schema='spatialrust.realsense-operation.v1', plan=plan,
        plan_sha256=hashlib.sha256(plan_bytes).hexdigest(), bindings=source_bindings,
        versions=dict(spatialrust=sr.__version__, numpy=np.__version__,
                      rosbags=metadata.version('rosbags')),
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sources={path.name: value for path,value in zip(sources,results)})
    (args.output_dir/'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in source.items() if k!='rows'} for name,source in result['sources'].items()}), flush=True)
    if any(r['successful_frames'] != r['frame_count'] for r in results):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
