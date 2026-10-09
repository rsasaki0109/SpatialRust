"""Read two point clouds, voxel-filter, align source to target, save full source.

Usage: python align_point_clouds.py source.pcd target.pcd --output-dir aligned-run
Output directory must not exist. Coordinates are interpreted as metres.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import spatialrust as sr


def rigid_matrix(value):
    """Validate a source-to-target proper rigid 4x4 matrix; return f32 copy."""
    raw = np.asarray(value)
    if raw.shape != (4, 4) or raw.dtype.kind not in 'fiu':
        raise ValueError('initial transform must be a numeric 4x4 matrix')
    matrix = raw.astype(np.float64)
    if not np.isfinite(matrix).all() or np.abs(matrix).max() > np.finfo(np.float32).max:
        raise ValueError('initial transform must be finite and f32-representable')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0,0,0,1], rtol=0, atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1, rtol=0, atol=1e-5)):
        raise ValueError('initial transform must be a proper rigid source-to-target pose')
    return matrix.astype(np.float32)


def align_files(source_path, target_path, *, leaf=.05, max_distance=.1, iterations=50, initial_transform=None):
    """Return (full-resolution aligned source, JSON-compatible diagnostics).

    Raises ValueError for invalid settings/input; native IO/registration errors
    propagate. ICP defaults to identity; convergence does not certify pose accuracy.
    XYZ transfer to NumPy for validation copies data explicitly.
    """
    for name, value in (('leaf', leaf), ('max_distance', max_distance)):
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
        with np.errstate(over='ignore', under='ignore'):
            converted = np.float32(value)
            squared = converted * converted
        if not np.isfinite(squared) or squared == 0:
            raise ValueError(f'{name} must have a finite nonzero f32 square')
    if type(iterations) is not int or iterations < 1:
        raise ValueError('iterations must be a positive integer')
    initial = rigid_matrix(np.eye(4) if initial_transform is None else initial_transform)
    source, target = sr.read(str(source_path)), sr.read(str(target_path))
    for name, cloud in (('source', source), ('target', target)):
        if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
            raise ValueError(f'{name} must contain at least three finite XYZ points')
    seeded = sr.apply_transform(source, initial)
    if not np.isfinite(seeded.xyz()).all():
        raise ValueError('initial transform overflowed source coordinates')
    coarse_source = sr.voxel_downsample(seeded, leaf, 'cpu')
    coarse_target = sr.voxel_downsample(target, leaf, 'cpu')
    if min(len(coarse_source), len(coarse_target)) < 3:
        raise ValueError('voxel clouds require at least three points; reduce leaf')
    coarse_result = sr.register_icp(coarse_source, coarse_target, max_distance, iterations)
    coarse_transform = (coarse_result.transform().astype(np.float64) @ initial.astype(np.float64)).astype(np.float32)
    if not np.isfinite(coarse_transform).all():
        raise ValueError('coarse registration returned an invalid transform')
    coarse_full = sr.apply_transform(source, coarse_transform)
    if not np.isfinite(coarse_full.xyz()).all():
        raise ValueError('coarse transform overflowed source coordinates')
    result = sr.register_icp(coarse_full, target, max_distance, iterations)
    transform = (result.transform().astype(np.float64) @ coarse_transform.astype(np.float64)).astype(np.float32)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('registration returned an invalid transform')
    aligned = sr.apply_transform(source, transform)
    if not np.isfinite(aligned.xyz()).all():
        raise ValueError('final transform overflowed source coordinates')
    def stage(name, registration, matrix, source_count, target_count):
        return dict(name=name, iterations=registration.iterations, converged=registration.converged,
                    source_points=source_count, target_points=target_count,
                    transform_source_to_target=matrix.tolist(),
                    kernel_fitness_metres_squared=registration.fitness if math.isfinite(registration.fitness) and registration.fitness < np.finfo(np.float64).max else None)
    stages = [stage('voxel', coarse_result, coarse_transform, len(coarse_source), len(coarse_target)),
              stage('full_resolution', result, transform, len(source), len(target))]
    def support(query, reference):
        count, fraction, rmse = sr.distance_gated_support(query, reference, max_distance)
        return dict(distance_metres=max_distance, query_points=len(query),
                    distance_gated_points=count, query_fraction=fraction, gated_rmse_metres=rmse)
    diagnostics = dict(schema_version='spatialrust.python-alignment.v1',
                       source_file=str(source_path), target_file=str(target_path),
                       source_points=len(source), target_points=len(target),
                       registration_source_points=len(coarse_source), registration_target_points=len(coarse_target),
                       leaf_metres=leaf, max_distance_metres=max_distance,
                       iterations=result.iterations, converged=result.converged,
                       max_iterations_per_stage=iterations, stages=stages,
                       initial_transform_supplied=initial_transform is not None,
                       initial_transform_source_to_target=initial.tolist(),
                       transform_source_to_target=transform.tolist(),
                       kernel_fitness_metres_squared=result.fitness if math.isfinite(result.fitness) and result.fitness < np.finfo(np.float64).max else None,
                       before_support=support(source, target),
                       aligned_support=support(aligned, target),
                       aligned_reverse_support=support(target, aligned),
                       pose_correctness='not_certified_by_convergence_or_residual')
    return aligned, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--leaf', type=float, default=.05)
    parser.add_argument('--max-distance', type=float, default=.1)
    parser.add_argument('--initial-transform', type=Path, help='JSON 4x4 source-to-target rigid matrix')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--html-report', action='store_true', help='also save standalone report.html with support diagnostics')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory already exists; choose a new path')
    initial = json.loads(args.initial_transform.read_text(encoding='utf-8')) if args.initial_transform else None
    aligned, diagnostics = align_files(args.source, args.target, leaf=args.leaf,
                                       max_distance=args.max_distance, iterations=args.iterations, initial_transform=initial)
    rendered = None
    if args.html_report:
        from render_alignment_report import render_report
        rendered = render_report(diagnostics)
    args.output_dir.mkdir()  # Exclusive reservation; never reuse an existing directory.
    try:
        sr.write(str(args.output_dir / 'aligned.pcd'), aligned)
        (args.output_dir / 'alignment.json').write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        if rendered is not None:
            (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Saved {len(aligned)} aligned points to {args.output_dir}; converged={diagnostics["converged"]}')


if __name__ == '__main__':
    main()
