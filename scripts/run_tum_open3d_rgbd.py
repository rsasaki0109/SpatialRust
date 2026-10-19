"""Full-scene CPU hybrid RGB-D comparator with an owned causal relative seed."""
import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path

import numpy as np

from dense_rgbd_protocol import CONTROLS as VALIDATION
from evaluate_pose_reference import rigid
from open3d_rgbd_protocol import CONTROLS, CPU_MODULE_SHA256, ESTIMATOR, validate_comparator_plan
from open3d_rgbd_validation import raw_statistics
from run_tum_sequence import generate
from tum_sequence_reference import file_sha256
from wide_rgbd_protocol import exact_json


def compute_cpu(source_gray, source_depth, target_gray, target_depth, initial, camera):
    cpu = importlib.import_module('open3d.cpu.pybind')
    source = cpu.geometry.RGBDImage(cpu.geometry.Image(source_gray), cpu.geometry.Image(source_depth))
    target = cpu.geometry.RGBDImage(cpu.geometry.Image(target_gray), cpu.geometry.Image(target_depth))
    intrinsic = cpu.camera.PinholeCameraIntrinsic(camera['width'], camera['height'],
        camera['fx'], camera['fy'], camera['cx'], camera['cy'])
    options = cpu.pipelines.odometry.OdometryOption()
    options.iteration_number_per_pyramid_level = cpu.utility.IntVector(CONTROLS['iterations'])
    options.depth_min, options.depth_max = CONTROLS['depth_min'], CONTROLS['depth_max']
    options.depth_diff_max = CONTROLS['depth_diff_max']
    return cpu.pipelines.odometry.compute_rgbd_odometry(source, target, intrinsic, initial,
        cpu.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), options)


def owned_inputs(previous_depth, current_depth, previous_gray, current_gray, camera):
    try:
        height, width = camera['height'], camera['width']
        valid_camera = (type(height) is int and type(width) is int and height > 1 and width > 1
            and all(type(camera[key]) in (int, float) and np.isfinite(camera[key]) for key in
                    ('fx','fy','cx','cy','depth_unit_m'))
            and min(camera['fx'], camera['fy'], camera['depth_unit_m']) > 0)
    except (KeyError, TypeError):
        valid_camera = False
    if not valid_camera:
        raise ValueError('finite positive CPU comparator camera required')
    inputs = (previous_depth, current_depth, previous_gray, current_gray)
    for index, array in enumerate(inputs):
        array = np.asarray(array)
        if array.shape != (height, width) or array.dtype != (np.uint16 if index < 2 else np.uint8):
            raise ValueError('require original depth counts and grayscale image shapes/types')
    return tuple(np.array(a, copy=True, order='C') for a in inputs)


class PreviousMotionOpen3dRgbd:
    """One seed/counter per chain; no retained images or reference motion."""

    def __init__(self, *, _compute=None):
        self._previous = np.eye(4)
        self._successful_pairs = 0
        self._compute = compute_cpu if _compute is None else _compute

    def align(self, previous_depth, current_depth, previous_gray, current_gray, camera, controls):
        if (exact_json(controls.get('open3d_rgbd_odometry')) != exact_json(CONTROLS)
                or exact_json(controls.get('dense_rgbd_odometry')) != exact_json(VALIDATION)):
            raise ValueError('comparator model and raw final-validation controls are fixed')
        source_counts, target_counts, source_gray, target_gray = owned_inputs(
            previous_depth, current_depth, previous_gray, current_gray, camera)
        initial = self._previous.copy()
        source = (source_counts.astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
        target = (target_counts.astype(np.float64)*camera['depth_unit_m']).astype(np.float32)
        result = self._compute(source_gray.astype(np.float32)/np.float32(255), source,
            target_gray.astype(np.float32)/np.float32(255), target, initial.copy(), dict(camera))
        if not isinstance(result, tuple) or len(result) != 3 or result[0] is not True:
            raise ValueError('Open3D CPU RGB-D estimation did not succeed')
        candidate = rigid(result[1]).copy()
        information = np.array(result[2], dtype=np.float64, copy=True)
        if information.shape != (6, 6) or not np.all(np.isfinite(information)):
            raise ValueError('finite 6x6 Open3D diagnostic information required')
        stats = raw_statistics(source_counts, target_counts, source_gray, target_gray,
                               candidate, camera, VALIDATION)
        if (stats['correspondences'] < VALIDATION['min_correspondences']
                or stats['correspondence_fraction'] < VALIDATION['min_correspondence_fraction']
                or any(stats[key] is None or not np.isfinite(stats[key]) for key in
                       ('photometric_rmse','depth_rmse_m'))):
            raise ValueError('Open3D estimate fails unchanged raw final support: count={}, fraction={}'.format(
                stats['correspondences'], stats['correspondence_fraction']))
        inverse = np.eye(4)
        inverse[:3, :3], inverse[:3, 3] = candidate[:3, :3].T, -candidate[:3, :3].T@candidate[:3, 3]
        report = dict(stats, current_from_previous=candidate.tolist(), previous_from_current=rigid(inverse).tolist(),
            external_final_support_validation=True, open3d_package_version=CONTROLS['package_version'],
            objective='Open3D CPU legacy hybrid; common raw validation statistics are separate',
            open3d_information=information.tolist(), information_scope='diagnostic only; not physical uncertainty',
            initial_seed_policy='previous-relative-increment',
            initial_previous_pair_index=self._successful_pairs if self._successful_pairs else None,
            initial_current_from_previous=initial.tolist(),
            owned_raw_input_bytes=sum(a.nbytes for a in (source_counts,target_counts,source_gray,target_gray)),
            owned_float_input_bytes=source.size*4*4, retained_seed_bytes=128,
            external_native_working_memory_observed=False, dense_settings=VALIDATION)
        self._previous = candidate.copy()
        self._successful_pairs += 1
        return inverse.copy(), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    validate_comparator_plan(json.loads((args.prepared_dir/'plan.json').read_text(encoding='utf-8')))
    if metadata.version('open3d') != CONTROLS['package_version']:
        raise ValueError('pinned Open3D package version required')
    native = Path(importlib.import_module('open3d.cpu.pybind').__file__)
    if file_sha256(native) != CPU_MODULE_SHA256:
        raise ValueError('pinned Open3D CPU module required')
    adapter = PreviousMotionOpen3dRgbd()
    result = generate(args.prepared_dir, args.manifest_sha256, args.output_dir,
        dense_rgbd_aligner=adapter.align, method=ESTIMATOR,
        additional_bindings=[__file__, native, Path(__file__).with_name('open3d_rgbd_protocol.py'),
            Path(__file__).with_name('open3d_rgbd_validation.py'),
            Path(__file__).with_name('audit_tum_dense_rgbd.py')])
    print('Frozen Open3D CPU RGB-D {}/{} poses; SHA-256 {}'.format(
        result['generated_poses'],result['planned_poses'],result['estimates_sha256']))


if __name__ == '__main__':
    main()
