"""Freeze reference-free CPU depth ICP estimates for every prepared TUM frame."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import spatialrust as sr

from evaluate_pose_reference import rigid
from realsense_reference import check_attributes, reference_geometry, save_attribute_input
from tum_sequence_reference import file_sha256, json_bytes, require_sha256, safe_relative, validate_plan


def compose_camera_pose(world_from_previous, previous_from_current):
    """Current-to-previous registration composes on the RIGHT of the world pose."""
    delta = rigid(previous_from_current).copy()
    # Native ICP returns f32 rotation: restore only its validated roundoff to SO(3).
    u, _, v = np.linalg.svd(delta[:3, :3])
    delta[:3, :3] = u @ v
    return rigid(rigid(world_from_previous) @ delta)


class TrackingChain:
    """One fixed gauge; lost tracking never starts a disconnected identity segment."""
    def __init__(self):
        self.pose = None
        self.lost = False

    def accept(self, previous_from_current=None):
        if self.lost:
            raise ValueError('tracking was lost; no reset or reference relocalization')
        if self.pose is None:
            if previous_from_current is not None:
                raise ValueError('first valid frame must use the sole identity anchor')
            self.pose = np.eye(4)
        elif previous_from_current is None:
            raise ValueError('existing trajectory requires a current-to-previous registration')
        else:
            self.pose = compose_camera_pose(self.pose, previous_from_current)
        return self.pose.copy()

    def fail(self):
        # A failed initial image cannot prevent later first initialization.
        if self.pose is not None:
            self.lost = True


def decode_depth(raw, camera):
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != 'PNG' or image.size != (camera['width'], camera['height']) or getattr(image, 'n_frames', 1) != 1:
            raise ValueError('unsupported depth image geometry')
        image.load()
        depth = np.asarray(image)
        if depth.ndim != 2 or depth.dtype.kind not in 'iu' or depth.dtype.itemsize < 2 or np.any(depth < 0) or np.any(depth > 65535):
            raise ValueError('depth must contain original unsigned 16-bit counts')
        return np.array(depth, dtype=np.uint16, copy=True)


def project(depth, camera, controls, output):
    valid, expected = reference_geometry(depth, camera, camera['depth_unit_m'], controls)
    # Convert original counts once in f64, then hand metres to the f32 kernel.
    # Multiplying .0002 inside f32 would exclude exact 0.1 m boundary pixels.
    metres = (depth.astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
    native = sr.depth_to_xyz(metres, camera['fx'], camera['fy'],
                            camera['cx'], camera['cy'], depth_scale=1.,
                            min_depth=controls['min_depth_m'], max_depth=controls['max_depth_m'], out=output)
    if not np.array_equal(np.isfinite(native).all(axis=2), valid):
        raise ValueError('native projection valid mask differs from independent formula')
    xyz = np.ascontiguousarray(native[valid])
    if len(xyz) < controls['min_points']:
        raise ValueError('too few valid depth points')
    error = float(np.max(np.abs(xyz.astype(np.float64)-expected)))
    if error > controls['projection_tolerance_m']:
        raise ValueError('native projection differs from independent optical formula')
    return sr.PointCloud.from_xyz(xyz), xyz, valid, error


def align_pair(current, previous, controls):
    # Current camera points map into previous camera coordinates. No motion prior.
    pose, stages = np.eye(4), []
    for stage in controls['stages']:
        moved = sr.apply_transform(current, pose.astype(np.float32))
        query = sr.voxel_downsample(moved, stage['leaf_m'], 'cpu')
        target = sr.voxel_downsample(previous, stage['leaf_m'], 'cpu')
        if min(len(query), len(target)) < 3:
            raise ValueError('fewer than three voxel points')
        trace = sr.register_icp_diagnostics(query, target, stage['gate_m'], stage['iterations'],
                                          translation_epsilon=1e-8, rotation_epsilon=1e-8,
                                          fitness_epsilon=1e-6, trim_fraction=1.)
        result = trace.result
        correction = result.transform()
        # Left-composition here is in target coordinates; world chaining is separate.
        pose = compose_camera_pose(correction, pose)
        stages.append(dict(iterations=result.iterations, converged=result.converged,
                           stop_reason=trace.stop_reason, query_points=len(query),
                           target_points=len(target), previous_from_current=pose.tolist()))
    count, fraction, rmse = sr.distance_gated_support(
        sr.apply_transform(current, pose.astype(np.float32)), previous, controls['stages'][-1]['gate_m'])
    if count < 3 or fraction < controls['min_support_fraction'] or rmse is None:
        raise ValueError('registration lacks preregistered geometric support: count={}, fraction={}, rmse={}'.format(count, fraction, rmse))
    return pose, dict(stages=stages, support_count=count, support_fraction=fraction, support_rmse_m=rmse)


def checked_depth(prepared, entry, limit):
    relative = safe_relative(entry['path'])
    if len(relative.parts) != 2 or relative.parts[0] != 'depth' or relative.suffix != '.png':
        raise ValueError('depth entry requires a relative depth PNG path')
    path = (prepared / str(relative)).resolve()
    path.relative_to(prepared.resolve())  # Reject filesystem links that escape the input.
    if type(entry['bytes']) is not int or not 0 < entry['bytes'] <= limit or path.stat().st_size != entry['bytes']:
        raise ValueError('invalid depth size')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != require_sha256(entry['sha256']):
        raise ValueError('depth differs from prepared image hash')
    return raw


def generate(prepared, manifest_sha256, output, *, pair_aligner=None,
             method='spatialrust_cpu', additional_bindings=()):
    """Run native ICP by default; explicit adapters serve separate baseline studies."""
    if method not in ('spatialrust_cpu', 'open3d_cpu') or (pair_aligner is None) != (method == 'spatialrust_cpu'):
        raise ValueError('method must match an explicit baseline adapter or the native default')
    prepared, output = Path(prepared), Path(output)
    raw = (prepared/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != require_sha256(manifest_sha256):
        raise ValueError('manifest differs from supplied frozen SHA-256')
    manifest = json.loads(raw)
    if manifest.get('schema') != 'spatialrust.tum-prepared.v1' or manifest.get('reference_used_for_generation') is not False:
        raise ValueError('unsupported prepared input or reference use')
    plan_raw = (prepared/'plan.json').read_bytes()
    calibration_raw = (prepared/'calibration.json').read_bytes()
    for data, name in ((plan_raw, 'plan_sha256'), (calibration_raw, 'calibration_sha256')):
        if hashlib.sha256(data).hexdigest() != require_sha256(manifest[name]):
            raise ValueError('plan or calibration differs from preparation')
    require_sha256(manifest['source_sha256'])
    plan = validate_plan(json.loads(plan_raw))
    camera, controls = plan['calibration'], plan['controls']
    if json.loads(calibration_raw) != camera or manifest['sequence'] != plan['sequence']:
        raise ValueError('prepared calibration/sequence does not match plan')
    frames = manifest['frames']
    if not 1 <= len(frames) == manifest['depth_frames'] <= plan['limits']['max_frames']:
        raise ValueError('incomplete or over-budget planned timeline')
    last = -1
    for frame in frames:
        ns = frame['timestamp_ns']
        if type(ns) is not int or not last < ns <= 2**63-1 or frame['depth']['timestamp_ns'] != ns:
            raise ValueError('timeline requires ordered integer nanoseconds')
        last = ns
    code = [Path(__file__), Path(__file__).with_name('tum_sequence_reference.py'),
            Path(__file__).with_name('realsense_reference.py'),
            Path(__file__).with_name('timestamped_trajectory_reference.py'),
            Path(__file__).with_name('evaluate_pose_reference.py')]
    if pair_aligner is not None:
        import inspect
        adapter_source = inspect.getsourcefile(pair_aligner)
        if adapter_source is None or not additional_bindings:
            raise ValueError('baseline adapter requires source and native-library bindings')
        code += [Path(adapter_source)] + [Path(p) for p in additional_bindings]
    pair_aligner = align_pair if pair_aligner is None else pair_aligner
    native = list(Path(sr.__file__).parent.glob('*.so')) + list(Path(sr.__file__).parent.glob('*.pyd'))
    if len(native) != 1:
        raise ValueError('require one installed native extension')
    bindings = {str(p.resolve()):file_sha256(p) for p in code+native}
    output.mkdir(parents=True, exist_ok=False)
    chain, previous, previous_ns = TrackingChain(), None, None
    dense = np.empty((camera['height'], camera['width'], 3), np.float32)
    poses, points, used_images = [], 0, []
    with (output/'frames.jsonl').open('x', encoding='utf-8') as checkpoint:
        for i, frame in enumerate(frames):
            row = dict(timestamp_ns=frame['timestamp_ns'], depth_sha256=frame['depth']['sha256'], status='error')
            try:
                if chain.lost:
                    raise ValueError('tracking lost at an earlier frame; disconnected resets forbidden')
                image_raw = checked_depth(prepared, frame['depth'], plan['limits']['max_member_bytes'])
                used_images.append(frame['depth'])
                depth = decode_depth(image_raw, camera)
                cloud, xyz, valid, error = project(depth, camera, controls, dense)
                if previous is None:
                    world = chain.accept()
                    row['identity_anchor'] = True
                else:
                    if frame['timestamp_ns']-previous_ns > controls['max_tracking_gap_ns']:
                        raise ValueError('depth interval exceeds preregistered tracking gap')
                    transform, report = pair_aligner(cloud, previous, controls)
                    world = chain.accept(transform)
                    row['registration'] = report
                if i % controls['io_stride'] == 0 or i == len(frames)-1:
                    path = output / 'frame-{:05d}-input.pcd'.format(i)
                    expected = save_attribute_input(path, xyz, depth, valid)
                    saved = sr.read(str(path))
                    check_attributes(saved, expected)
                    restored = path.with_name(path.name.replace('-input.pcd', '-restored.pcd'))
                    sr.write(str(restored), saved)
                    check_attributes(sr.read(str(restored)), expected)
                    row['io'] = dict(input_sha256=file_sha256(path), restored_sha256=file_sha256(restored),
                                     exact_typed_attributes=True)
                previous, previous_ns = cloud, frame['timestamp_ns']
                row.update(status='success', world_from_sensor=world.tolist(), points=len(cloud),
                           projection_max_error_m=error)
                points += len(cloud)
            except (ValueError, RuntimeError, OSError) as failure:
                chain.fail()
                row['error'] = str(failure)
            checkpoint.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
            checkpoint.flush()
            poses.append(row)
    # Check all inputs actually used and code again before publishing the frozen estimates.
    for image in used_images:
        checked_depth(prepared, image, plan['limits']['max_member_bytes'])
    if (prepared/'manifest.json').read_bytes() != raw or (prepared/'plan.json').read_bytes() != plan_raw or (prepared/'calibration.json').read_bytes() != calibration_raw:
        raise ValueError('prepared metadata changed during generation')
    if any(file_sha256(p) != sha for p, sha in bindings.items()):
        raise ValueError('generation code/native bytes changed during generation')
    frozen = dict(schema='spatialrust.timestamped-trajectory.v1', length_unit='m',
                  sensor_frame=camera['sensor_frame'], reference_used_for_generation=False,
                  source_sha256=manifest['source_sha256'], plan_sha256=manifest['plan_sha256'],
                  calibration_sha256=manifest['calibration_sha256'], manifest_sha256=manifest_sha256,
                  method=method, bindings=bindings, poses=poses)
    with (output/'estimates.json').open('xb') as stream:
        stream.write(json_bytes(frozen))
    receipt = dict(schema='spatialrust.tum-generation.v1', planned_poses=len(poses),
                   generated_poses=sum(p['status'] == 'success' for p in poses), projected_points=points,
                   tracking_lost=chain.lost, reference_used_for_generation=False,
                   estimates_sha256=file_sha256(output/'estimates.json'), bindings=bindings,
                   method=method, python=sys.version.split()[0], numpy=np.__version__)
    with (output/'receipt.json').open('xb') as stream:
        stream.write(json_bytes(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    r = generate(args.prepared_dir, args.manifest_sha256, args.output_dir)
    print('Frozen {}/{} poses; estimates SHA-256 {}'.format(r['generated_poses'], r['planned_poses'], r['estimates_sha256']))


if __name__ == '__main__':
    main()
