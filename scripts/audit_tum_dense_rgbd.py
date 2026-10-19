"""Independent NumPy final-warp statistics and world-chain audit; no pose fitting."""
import argparse
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from dense_rgbd_protocol import CONTROLS, FINAL_ESTIMATOR, validate_dense_experiment_plan
from causal_dense_rgbd_protocol import ESTIMATOR as CAUSAL_ESTIMATOR, validate_causal_dense_plan
from open3d_rgbd_protocol import ESTIMATOR as OPEN3D_ESTIMATOR, validate_comparator_plan
from gaussian_dense_rgbd_protocol import ESTIMATOR as GAUSSIAN_ESTIMATOR, POLICY as GAUSSIAN_POLICY, validate_gaussian_dense_plan
from wide_rgbd_protocol import exact_json
from evaluate_pose_reference import rigid
from tum_sequence_reference import file_sha256, json_bytes, require_sha256, safe_relative


def reference_statistics(source_depth, target_depth, source_gray, target_gray, pose, camera, settings):
    """Recompute every full-resolution source row, bilinear mask and residual."""
    pose = rigid(pose)
    source = (source_depth.astype(np.float64)*camera['depth_unit_m']).astype(np.float32).astype(np.float64)
    target = (target_depth.astype(np.float64)*camera['depth_unit_m']).astype(np.float32).astype(np.float64)
    valid = np.isfinite(source) & (source >= settings['min_depth']) & (source <= settings['max_depth'])
    target[~(np.isfinite(target) & (target >= settings['min_depth']) & (target <= settings['max_depth']))] = np.nan
    y, x = np.nonzero(valid)
    z = source[y, x]
    objects = [(x-camera['cx'])*z/camera['fx'], (y-camera['cy'])*z/camera['fy'], z]
    moved = [pose[a, 0]*objects[0] + pose[a, 1]*objects[1] + pose[a, 2]*objects[2] + pose[a, 3]
             for a in range(3)]
    positive = np.isfinite(moved[2]) & (moved[2] > 0)
    moved = [a[positive] for a in moved]
    x, y = x[positive], y[positive]
    u = camera['fx']*moved[0]/moved[2] + camera['cx']
    v = camera['fy']*moved[1]/moved[2] + camera['cy']
    height, width = source.shape
    inside = np.isfinite(u) & np.isfinite(v) & (u >= 0) & (v >= 0) & (u < width-1) & (v < height-1)
    u, v, zc, x, y = u[inside], v[inside], moved[2][inside], x[inside], y[inside]
    ix, iy = u.astype(np.int64), v.astype(np.int64)
    du, dv = u-ix, v-iy

    def interpolate(values):
        a, b, c, d = values[iy, ix], values[iy, ix+1], values[iy+1, ix], values[iy+1, ix+1]
        finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
        return (a*(1-du)+b*du)*(1-dv)+(c*(1-du)+d*du)*dv, finite

    depth, corners = interpolate(target)
    depth_error = depth-zc
    accepted = corners & (np.abs(depth_error) <= settings['max_depth_difference'])
    intensity, _ = interpolate(target_gray.astype(np.float64)/255.)
    photo_error = intensity-source_gray[y, x].astype(np.float64)/255.
    count, total = int(np.count_nonzero(accepted)), int(np.count_nonzero(valid))
    return dict(valid_source_pixels=total, correspondences=count,
                correspondence_fraction=count/total if total else 0.,
                photometric_rmse=float(np.sqrt(np.mean(photo_error[accepted]**2))) if count else None,
                depth_rmse_m=float(np.sqrt(np.mean(depth_error[accepted]**2))) if count else None)


def checked_image(prepared, entry, kind, camera, limit):
    relative = safe_relative(entry['path'])
    if len(relative.parts) != 2 or relative.parts[0] != kind or relative.suffix != '.png':
        raise ValueError('require original registered PNG path')
    path = (prepared/str(relative)).resolve()
    path.relative_to(prepared.resolve())
    if type(entry['bytes']) is not int or not 0 < entry['bytes'] <= limit or path.stat().st_size != entry['bytes']:
        raise ValueError('invalid source image size')
    if file_sha256(path) != require_sha256(entry['sha256']):
        raise ValueError('source image checksum changed')
    with Image.open(io.BytesIO(path.read_bytes())) as image:
        if image.format != 'PNG' or image.size != (camera['width'], camera['height']):
            raise ValueError('invalid original image geometry')
        values = np.array(image)
    if kind == 'depth':
        if values.ndim != 2 or values.dtype.kind not in 'iu' or values.dtype.itemsize < 2 or np.any(values < 0) or np.any(values > 65535):
            raise ValueError('invalid registered depth counts')
        return values.astype(np.uint16)
    if values.shape != (camera['height'], camera['width'], 3) or values.dtype != np.uint8:
        raise ValueError('require original RGB pixels')
    # Independently reproduce the already fixed CPU grayscale integer contract.
    rgb = values.astype(np.uint32)
    return ((4899*rgb[:, :, 0]+9617*rgb[:, :, 1]+1868*rgb[:, :, 2]+8192) >> 14).astype(np.uint8)


def check_causal_initialization(report, previous_relative, index):
    """Audit a reported seed using only the prior returned increment, no fitting."""
    if (report.get('initial_seed_policy') != 'previous-relative-increment'
            or 'initial_previous_pair_index' not in report):
        raise ValueError('causal seed policy receipt required')
    observed_index = report.get('initial_previous_pair_index')
    if index == 1:
        if observed_index is not None or previous_relative is not None:
            raise ValueError('first causal pair requires identity without prior index')
        expected = np.eye(4)
    else:
        if type(observed_index) is not int or observed_index != index-1 or previous_relative is None:
            raise ValueError('causal seed must identify the immediately preceding pair')
        expected = previous_relative
    try:
        actual = rigid(report['initial_current_from_previous'])
    except (KeyError, TypeError):
        raise ValueError('causal seed pose receipt required') from None
    error = float(np.max(np.abs(actual-expected)))
    if error > 1e-12:
        raise ValueError('causal seed differs from identity or the preceding relative pose')
    return error


def audit(prepared, manifest_sha256, estimates, estimates_sha256, output):
    prepared, estimates, output = Path(prepared), Path(estimates), Path(output)
    if output.exists() or shutil.disk_usage(output.parent).free < 20*1024**3:
        raise ValueError('require fresh external audit output and 20GiB free')
    if file_sha256(prepared/'manifest.json') != require_sha256(manifest_sha256) or file_sha256(estimates) != require_sha256(estimates_sha256):
        raise ValueError('frozen inputs/estimates changed')
    manifest = json.loads((prepared/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'spatialrust.tum-prepared.v1' or manifest.get('reference_used_for_generation') is not False:
        raise ValueError('require source-bound reference-free prepared frames')
    if file_sha256(prepared/'plan.json') != manifest['plan_sha256']:
        raise ValueError('prepared plan changed')
    plan = json.loads((prepared/'plan.json').read_text(encoding='utf-8'))
    external = plan.get('estimator') == OPEN3D_ESTIMATOR
    gaussian = plan.get('estimator') == GAUSSIAN_ESTIMATOR
    causal = external or gaussian or plan.get('estimator') == CAUSAL_ESTIMATOR
    plan = (validate_gaussian_dense_plan(plan) if gaussian else validate_comparator_plan(plan) if external else
            validate_causal_dense_plan(plan) if causal else validate_dense_experiment_plan(plan))
    if file_sha256(prepared/'calibration.json') != manifest['calibration_sha256'] or json.loads((prepared/'calibration.json').read_text(encoding='utf-8')) != plan['calibration']:
        raise ValueError('prepared calibration changed')
    frozen = json.loads(estimates.read_text(encoding='utf-8'))
    if (frozen.get('schema') != 'spatialrust.timestamped-trajectory.v1' or frozen.get('length_unit') != 'm'
            or frozen.get('method') != plan['estimator'] or frozen.get('manifest_sha256') != manifest_sha256
            or frozen.get('reference_used_for_generation') is not False
            or any(frozen.get(key) != manifest[key] for key in ('source_sha256', 'plan_sha256', 'calibration_sha256'))):
        raise ValueError('require the exact frozen native dense estimates')
    frames, rows = manifest['frames'], frozen['poses']
    if len(frames) != len(rows) or [r['timestamp_ns'] for r in rows] != [f['timestamp_ns'] for f in frames]:
        raise ValueError('complete original timeline required')
    camera, previous, world, lost = plan['calibration'], None, None, False
    previous_relative = None
    checks, failures = [], []
    for index, (frame, row) in enumerate(zip(frames, rows)):
        if row.get('status') not in ('success', 'error'):
            raise ValueError('unknown frame status')
        if row['status'] != 'success':
            lost = world is not None
            failures.append(index)
            continue
        if lost:
            raise ValueError('a disconnected trajectory reset cannot pass')
        depth = checked_image(prepared, frame['depth'], 'depth', camera, plan['limits']['max_member_bytes'])
        gray = checked_image(prepared, frame['rgb'], 'rgb', camera, plan['limits']['max_member_bytes'])
        observed_world = rigid(row['world_from_sensor'])
        if previous is None:
            if not np.allclose(observed_world, np.eye(4), rtol=0, atol=1e-12):
                raise ValueError('sole initial world anchor required')
        else:
            report = row['registration']
            if plan['estimator'] in (FINAL_ESTIMATOR, CAUSAL_ESTIMATOR, GAUSSIAN_ESTIMATOR) and report.get('final_support_only') is not True:
                raise ValueError('native final support policy receipt required')
            if gaussian and (report.get('fitting_preprocessing') != 'gaussian3'
                    or exact_json(report.get('fitting_policy')) != exact_json(GAUSSIAN_POLICY)):
                raise ValueError('fixed Gaussian3 fitting and original raw final diagnostics required')
            if external:
                information = np.asarray(report.get('open3d_information'), dtype=np.float64)
                if (report.get('external_final_support_validation') is not True
                        or report.get('open3d_package_version') != '0.19.0'
                        or information.shape != (6, 6) or not np.all(np.isfinite(information))):
                    raise ValueError('external CPU model/final-validation diagnostics required')
            pose = rigid(report['current_from_previous'])
            initial_error = check_causal_initialization(report, previous_relative, index) if causal else None
            reference = reference_statistics(previous[0], depth, previous[1], gray, pose, camera, CONTROLS)
            for key in ('valid_source_pixels', 'correspondences'):
                if type(report[key]) is not int or report[key] != reference[key]:
                    raise ValueError('dense full-row count differs: '+key)
            delta = {}
            for key, tolerance in [('correspondence_fraction', 1e-12), ('photometric_rmse', 1e-10), ('depth_rmse_m', 1e-10)]:
                if reference[key] is None or not np.isfinite(report[key]):
                    raise ValueError('missing/nonfinite dense statistic')
                delta[key] = abs(reference[key]-report[key])
                if delta[key] > tolerance:
                    raise ValueError('dense independent statistic differs: '+key)
            inverse = np.eye(4)
            inverse[:3, :3], inverse[:3, 3] = pose[:3, :3].T, -pose[:3, :3].T@pose[:3, 3]
            composition_error = float(np.max(np.abs(world@inverse-observed_world)))
            if composition_error > 1e-9 or not np.allclose(inverse, report['previous_from_current'], rtol=0, atol=1e-12):
                raise ValueError('dense direction/world composition differs')
            checks.append(dict(index=index, statistics=reference, differences=delta, world_max_error=composition_error))
            if causal:
                checks[-1]['initial_max_error'] = initial_error
                previous_relative = pose.copy()
        previous, world = (depth, gray), observed_world
        if index % 60 == 0:
            print('Independent dense rows {}/{}'.format(index+1, len(rows)), flush=True)
    if file_sha256(prepared/'manifest.json') != manifest_sha256 or file_sha256(estimates) != estimates_sha256:
        raise ValueError('frozen data changed during audit')
    result = dict(schema='spatialrust.dense-rgbd-audit.v1', planned_frames=len(rows),
        successful_pairs=len(checks), failed_frame_indices=failures, pair_checks=checks,
        estimates_sha256=estimates_sha256, manifest_sha256=manifest_sha256,
        auditor_sha256=file_sha256(__file__), reference_motion_used=False,
        scope='Final projective measurements and causal world chain; no fitting or trajectory-accuracy acceptance')
    if causal:
        result.update(causal_initialization_pairs=len(checks), initial_seed_policy='previous-relative-increment')
    output.mkdir()
    (output/'audit.json').write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--estimates', type=Path, required=True)
    parser.add_argument('--estimates-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.prepared_dir, args.manifest_sha256, args.estimates, args.estimates_sha256, args.output_dir)
    print('Audited {} dense pairs; {} failed frames retained'.format(result['successful_pairs'], len(result['failed_frame_indices'])))


if __name__ == '__main__':
    main()
