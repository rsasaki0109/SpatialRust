"""Source-bound synthetic dense sampling audit; never loads sensor frames."""
import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'benchmarks/dense-sampling-audit-plan.json'
PLAN_LF_SHA256 = '777449a2a647a47eab6f57a003f1f38a1b65d7f1e8639448e0a5f401f1ed1a9c'
MARKER = 'DENSE_SAMPLING_AUDIT_JSON:'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def load_plan(path=PLAN):
    raw = Path(path).read_bytes()
    lf = raw.replace(b'\r\n', b'\n')
    if len(raw) > 65536 or b'\r' in lf or raw not in (lf, lf.replace(b'\n', b'\r\n')) or hashlib.sha256(lf).hexdigest() != PLAN_LF_SHA256:
        raise ValueError('require the exact fixed audit plan with uniform LF or CRLF')
    return json.loads(lf)


def numbers(value, shape):
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in 'fiu' or not np.all(np.isfinite(array)):
        raise ValueError('finite numerical audit arrays with exact shapes required')
    return array.astype(np.float64)


def close(actual, expected, tolerance):
    wanted = np.asarray(expected, dtype=np.float64)
    observed = numbers(actual, wanted.shape)
    difference = float(np.max(np.abs(observed - wanted)))
    if difference > tolerance:
        raise ValueError('analytical audit expectation failed: difference {}'.format(difference))
    return difference


def dimensions(actual, expected):
    if type(actual) is not list or any(type(v) is not int for v in actual) or actual != expected:
        raise ValueError('fixed audit dimensions/coordinates required')


def boolean(actual, expected):
    if type(actual) is not bool or actual is not expected:
        raise ValueError('exact boolean audit observation required')


def row_inputs(plan):
    row = plan['row']
    return tuple(np.asarray(row[key], dtype=np.float32).astype(np.float64)
                 for key in ('point', 'gray_gradient', 'depth_gradient'))


def analytic_jacobian(point, intrinsics, gray_gradient, depth_gradient):
    x, y, z = numbers(point, (3,))
    fx, fy, _, _ = numbers(intrinsics, (4,))
    if min(z, fx, fy) <= 0:
        raise ValueError('positive finite point depth and focal lengths required')
    projection = np.array([[fx/z, 0., -fx*x/(z*z)], [0., fy/z, -fy*y/(z*z)]])
    action = np.array([[0., z, -y, 1., 0., 0.], [-z, 0., x, 0., 1., 0.], [y, -x, 0., 0., 0., 1.]])
    return np.vstack((numbers(gray_gradient, (2,)) @ projection @ action,
                      numbers(depth_gradient, (2,)) @ projection @ action - action[2]))


def native_observation(path):
    path = Path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError('native probe log exceeds its bound')
    lines = [line.split(MARKER, 1)[1] for line in path.read_text(encoding='utf-8').splitlines() if MARKER in line]
    if len(lines) != 1 or len(lines[0]) > 32768:
        raise ValueError('exactly one bounded native audit record required')
    return json.loads(lines[0])


def validate_native(observed, plan):
    keys = {'schema', 'even_dimensions', 'coarse_gray', 'odd_dimensions', 'hole_coarse_depth',
            'hole_admitted', 'border_admitted', 'subpixel', 'row_point', 'row_gray_gradient',
            'row_depth_gradient', 'row_jacobian'}
    if set(observed) != keys or observed['schema'] != 'spatialrust.native-sampling-probe.v1':
        raise ValueError('exact native sampling record schema required')
    dimensions(observed['even_dimensions'], [4, 4])
    dimensions(observed['odd_dimensions'], plan['odd']['native_level_one'])
    boolean(observed['hole_admitted'], False)
    boolean(observed['border_admitted'], False)
    point, gray, depth = row_inputs(plan)
    tol = plan['tolerances']['native_scalar']
    differences = dict(coarse_gray=close(observed['coarse_gray'], plan['affine']['native_coarse_gray'], tol),
        hole_depth=close(observed['hole_coarse_depth'], plan['affine']['depth_m'], tol),
        subpixel=close(observed['subpixel'], [plan['affine']['native_subpixel_gray'], *plan['affine']['gradient']], tol),
        point=close(observed['row_point'], point, tol), gray_gradient=close(observed['row_gray_gradient'], gray, tol),
        depth_gradient=close(observed['row_depth_gradient'], depth, tol),
        jacobian=close(observed['row_jacobian'], analytic_jacobian(point, plan['row']['intrinsics'], gray, depth), plan['tolerances']['unweighted_jacobian']))
    return differences


def verify_sources(path, plan):
    path = Path(path)
    if path.stat().st_size > 65536:
        raise ValueError('source manifest exceeds its bound')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    expected = {row['upstream_path']: row for row in plan['reference']['sources']}
    rows = manifest.get('files', [])
    if manifest.get('commit') != plan['reference']['commit'] or len(rows) != len(expected) or {r['upstream_path'] for r in rows} != set(expected):
        raise ValueError('exact pinned reference source set required')
    for row in rows:
        wanted = expected[row['upstream_path']]
        file = Path(row['path'])
        if not file.is_absolute() or file.stat().st_size != wanted['bytes'] or row['bytes'] != wanted['bytes'] or sha(file) != wanted['sha256'] or row['sha256'] != wanted['sha256']:
            raise ValueError('pinned reference source bytes changed')
    return manifest


def reference_observation(cpu, plan):
    affine = plan['affine']; height, width = affine['height'], affine['width']
    y, x = np.mgrid[:height, :width]
    values = (3*x + 5*y + 7).astype(np.float32) / np.float32(255)
    image = cpu.geometry.Image(values)
    coarse = np.asarray(image.create_pyramid(2, True)[1]).copy()
    odd = np.asarray(cpu.geometry.Image(np.zeros((5, 7), np.float32)).create_pyramid(2, True)[1]).copy()
    hole = np.full((8, 8), 2., np.float32); hole[2, 2] = np.nan
    hole = np.asarray(cpu.geometry.Image(hole).create_pyramid(2, False)[1]).copy()
    neighbor = np.full((8, 8), 2., np.float32); neighbor[2, 3] = np.nan
    owned_neighbor = np.asarray(cpu.geometry.Image(neighbor)).copy()
    gradient = [float(np.asarray(image.filter(mode))[2, 2]) * .125
                for mode in (cpu.geometry.ImageFilterType.Sobel3dx, cpu.geometry.ImageFilterType.Sobel3dy)]
    u, v = (int(coordinate + .5) for coordinate in affine['subpixel'])
    point, gray, depth = row_inputs(plan)
    def rgbd(color, z):
        result = cpu.geometry.RGBDImage()
        result.color = cpu.geometry.Image(np.full((8, 8), color, np.float32))
        result.depth = cpu.geometry.Image(np.full((8, 8), z, np.float32))
        return result
    row = plan['row']; source = rgbd(row['source_gray'], point[2]); target = rgbd(row['target_gray'], row['target_depth'])
    dx, dy = rgbd(gray[0]/.125, depth[0]/.125), rgbd(gray[1]/.125, depth[1]/.125)
    xyz = np.empty((8, 8, 3), np.float32); xyz[:] = point
    matrix = np.eye(3); matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2] = row['intrinsics']
    jacobian = [np.full((6, 1), np.nan), np.full((6, 1), np.nan)]
    residual, weights = cpu.utility.DoubleVector(), cpu.utility.DoubleVector()
    cpu.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm().compute_jacobian_and_residual(
        0, jacobian, residual, weights, source, target, cpu.geometry.Image(xyz), dx, dy,
        matrix, np.eye(4), cpu.utility.Vector4iVector(np.array([[2, 2, 2, 2]], np.int32)))
    observable = len(jacobian) == 2 and all(np.all(np.isfinite(value)) for value in jacobian)
    return dict(even_dimensions=[coarse.shape[1], coarse.shape[0]], coarse_gray=float(coarse[1, 1]),
        odd_dimensions=[odd.shape[1], odd.shape[0]], hole_coarse_valid=bool(np.isfinite(hole[1, 1])),
        integer_hole_valid=bool(np.isfinite(owned_neighbor[2, 2])), integer_border_valid=bool(np.isfinite(owned_neighbor[2, 7])),
        rounded_coordinates=[u, v], rounded_gray=float(values[v, u]), gradient=gradient,
        row_residual=list(residual), row_weights=list(weights), jacobian_observable=bool(observable),
        row_jacobian=np.asarray(jacobian).reshape(2, 6).tolist() if observable else None,
        jacobian_scope='mutated runtime arrays' if observable else 'Python list does not expose mutated C++ Jacobian vectors; source formula and native analytic check remain separate')


def validate_reference(observed, plan):
    dimensions(observed['even_dimensions'], [4, 4]); dimensions(observed['odd_dimensions'], plan['odd']['reference_level_one'])
    dimensions(observed['rounded_coordinates'], [2, 3])
    for key, wanted in [('hole_coarse_valid', False), ('integer_hole_valid', True), ('integer_border_valid', True)]:
        boolean(observed[key], wanted)
    tol = plan['tolerances']['reference_float32']; row = plan['row']
    scale = np.sqrt([1-row['depth_weight'], row['depth_weight']])
    residual = [float(np.float32(row['target_gray']))-float(np.float32(row['source_gray'])), float(np.float32(row['target_depth']))-float(np.float32(row['point'][2]))]
    differences = dict(coarse_gray=close(observed['coarse_gray'], plan['affine']['reference_coarse_gray'], tol),
        gradient=close(observed['gradient'], plan['affine']['gradient'], tol),
        rounded_gray=close(observed['rounded_gray'], plan['affine']['reference_integer_gray'], tol),
        row_residual=close(observed['row_residual'], scale*np.asarray(residual), plan['tolerances']['reference_row']),
        row_weights=close(observed['row_weights'], [1., 1.], 0.))
    if type(observed['jacobian_observable']) is not bool:
        raise ValueError('exact Jacobian observability flag required')
    if observed['jacobian_observable']:
        point, gray, depth = row_inputs(plan)
        differences['jacobian'] = close(observed['row_jacobian'], scale[:, None]*analytic_jacobian(point, row['intrinsics'], gray, depth), plan['tolerances']['reference_row'])
    elif observed['row_jacobian'] is not None:
        raise ValueError('unobserved Jacobian must not be presented as a runtime result')
    return differences


def run(native_log, native_sha256, native_exe, native_exe_sha256, sources, output):
    output = Path(output).resolve(); plan = load_plan()
    if output.exists() or not output.parent.is_dir() or shutil.disk_usage(output.parent).free < plan['resources']['minimum_free_bytes']:
        raise ValueError('fresh external audit output and 20GiB free required')
    if ROOT == output or ROOT in output.parents:
        raise ValueError('audit results must remain outside the repository')
    if sha(native_log) != native_sha256 or sha(native_exe) != native_exe_sha256:
        raise ValueError('frozen native log/executable changed')
    verify_sources(sources, plan)
    native = native_observation(native_log); native_errors = validate_native(native, plan)
    cpu = importlib.import_module('open3d.cpu.pybind'); module = Path(cpu.__file__)
    if metadata.version('open3d') != plan['reference']['version'] or sha(module) != plan['reference']['cpu_module_sha256']:
        raise ValueError('pinned installed reference CPU module required')
    reference = reference_observation(cpu, plan); reference_errors = validate_reference(reference, plan)
    report = dict(schema='spatialrust.dense-sampling-audit.v1', plan=record(PLAN), native_log=record(native_log), native_executable=record(native_exe),
        source_manifest=record(sources), reference_module=record(module), upstream_commit=plan['reference']['commit'],
        native=native, reference=reference, native_differences=native_errors, reference_differences=reference_errors,
        scope=plan['scope'], source_defined_lookup_scope='rounded coordinates use the pinned source rule; no exported runtime correspondence set is claimed',
        trajectory_acceptance=False, operational_80_percent_proven=False)
    raw = (json.dumps(report, indent=2, allow_nan=False)+'\n').encode('utf-8')
    if len(raw) > plan['resources']['audit_result_max_bytes']:
        raise ValueError('audit result exceeds its declared bound')
    output.mkdir(); (output/'audit.json').write_bytes(raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native-log', 'native-log-sha256', 'native-exe', 'native-exe-sha256', 'sources', 'output-dir'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    result = run(args.native_log, args.native_log_sha256, args.native_exe, args.native_exe_sha256, args.sources, args.output_dir)
    print(json.dumps(dict(native_jacobian_max_error=result['native_differences']['jacobian'], reference_jacobian_observable=result['reference']['jacobian_observable'], trajectory_acceptance=False)))


if __name__ == '__main__':
    main()
