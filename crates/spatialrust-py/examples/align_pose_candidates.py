"""Align files from 1–16 supplied rigid pose candidates and save the best support.

Usage: python align_pose_candidates.py source.pcd target.pcd
       --initial-transforms poses.json --output-dir candidate-run --html-report
Candidate selection uses forward support at one shared evaluation distance;
convergence and low residuals do not certify a correct pose.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import spatialrust as sr

from align_point_clouds import _align_clouds, rigid_matrix, validate_settings, validate_convergence, convergence_arguments, convergence_from_args


def evaluate_candidates(source_path, target_path, initial_transforms, *, leaf=.05,
                        max_distance=.1, evaluation_distance=None, iterations=50, fine_distance=None, trace=False, convergence=None):
    """Return the selected full source and diagnostics without saving files.

    Validate every candidate before reading either cloud. A ValueError from an
    individual alignment is recorded; other exceptions propagate. Selection
    maximizes forward supported points, minimizes gated RMSE, then keeps input
    order for ties. No ground-truth pose participates in selection.
    """
    if not isinstance(initial_transforms, list) or not 1 <= len(initial_transforms) <= 16:
        raise ValueError('initial_transforms must be a JSON list of 1 to 16 rigid matrices')
    poses = [rigid_matrix(value) for value in initial_transforms]
    evaluation_distance = validate_settings(leaf, max_distance, iterations, evaluation_distance, fine_distance)
    validate_convergence(convergence)
    source, target = sr.read(str(source_path)), sr.read(str(target_path))
    records = []
    selected = None
    selected_key = None
    selected_index = None
    target_voxel_cache = []
    before_support_cache = []
    target_support_index_cache = []
    for index, pose in enumerate(poses):
        try:
            aligned, report = _align_clouds(
                source, target, source_name=str(source_path), target_name=str(target_path), leaf=leaf, max_distance=max_distance,
                evaluation_distance=evaluation_distance, iterations=iterations,
                initial_transform=pose, target_voxel_cache=target_voxel_cache,
                before_support_cache=before_support_cache,
                target_support_index_cache=target_support_index_cache, fine_distance=fine_distance, trace=trace, convergence=convergence)
        except ValueError as error:
            records.append(dict(index=index, status='error', error=str(error),
                                initial_transform_source_to_target=pose.tolist()))
            continue
        support = report['aligned_support']
        rmse = support['gated_rmse_metres']
        key = (support['distance_gated_points'], -(math.inf if rmse is None else rmse))
        records.append(dict(
            index=index, status='success', initial_transform_source_to_target=pose.tolist(),
            transform_source_to_target=report['transform_source_to_target'],
            aligned_support=dict(support),
            aligned_reverse_support=dict(report['aligned_reverse_support']),
            converged=report['converged'], error=None))
        if selected_key is None or key > selected_key:
            selected = (aligned, report)
            selected_key = key
            selected_index = index
    if selected is None:
        failures = '; '.join(f"{record['index']}: {record['error']}" for record in records)
        raise ValueError(f'no candidate alignment succeeded ({failures})')
    aligned, report = selected
    diagnostics = dict(report)
    diagnostics['candidate_selection'] = dict(
        rule='maximum_forward_supported_points_then_minimum_gated_rmse_then_input_order',
        candidate_count=len(poses), successful_candidates=sum(record['error'] is None for record in records),
        selected_index=selected_index, candidates=records,
        source_centroid_xyz_metres=source.xyz().mean(axis=0, dtype=np.float64).tolist())
    return aligned, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--initial-transforms', type=Path, required=True,
                        help='JSON list of 1–16 source-to-target rigid 4x4 matrices')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--leaf', type=float, default=.05)
    parser.add_argument('--max-distance', type=float, default=.1)
    parser.add_argument('--fine-distance', type=float, help='full-resolution ICP gate; defaults to max-distance')
    parser.add_argument('--evaluation-distance', type=float,
                        help='shared support evaluation distance; defaults to max-distance')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--html-report', action='store_true')
    parser.add_argument('--trace', action='store_true', help='record selected candidate ICP history')
    convergence_arguments(parser)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory already exists; choose a new path')
    poses = json.loads(args.initial_transforms.read_text(encoding='utf-8'))
    aligned, diagnostics = evaluate_candidates(
        args.source, args.target, poses, leaf=args.leaf,
        max_distance=args.max_distance, evaluation_distance=args.evaluation_distance,
        iterations=args.iterations, fine_distance=args.fine_distance, trace=args.trace, convergence=convergence_from_args(args))
    serialized = json.dumps(diagnostics, indent=2, allow_nan=False) + '\n'
    rendered = None
    if args.html_report:
        from render_alignment_report import render_report
        rendered = render_report(diagnostics)
    args.output_dir.mkdir()
    try:
        sr.write(str(args.output_dir / 'aligned.pcd'), aligned)
        (args.output_dir / 'alignment.json').write_text(serialized, encoding='utf-8')
        if rendered is not None:
            (args.output_dir / 'report.html').write_text(rendered, encoding='utf-8')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Saved {len(aligned)} aligned points to {args.output_dir}; '
          f'candidate={diagnostics["candidate_selection"]["selected_index"]}; '
          f'converged={diagnostics["converged"]}')


if __name__ == '__main__':
    main()
