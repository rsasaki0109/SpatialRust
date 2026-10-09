"""Explicit coarse-to-fine ICP schedule, preserving the full original source.

Usage: python align_multiscale.py source.pcd target.pcd --schedule schedule.json
       --output-dir aligned --html-report
Schedule is a JSON list of stages: leaf (metres or null for full resolution),
max_distance, iterations, optional trim_fraction and convergence thresholds.
"""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import spatialrust as sr

from align_point_clouds import rigid_matrix, validate_settings, validate_convergence, validate_trim_fraction


def validate_schedule(schedule):
    """Return owned normalized settings; reject the entire schedule before IO."""
    if not isinstance(schedule, list) or not 1 <= len(schedule) <= 16:
        raise ValueError('schedule must be a list of 1..16 stages')
    normalized = []
    previous_leaf = previous_gate = math.inf
    for index, stage in enumerate(schedule):
        if not isinstance(stage, dict) or set(stage) - {'leaf','max_distance','iterations','trim_fraction','convergence'}:
            raise ValueError('schedule stage has unsupported fields')
        if not {'leaf','max_distance','iterations'} <= set(stage):
            raise ValueError('each stage requires leaf, max_distance and iterations')
        leaf, gate, iterations = stage['leaf'], stage['max_distance'], stage['iterations']
        validate_settings(gate if leaf is None else leaf, gate, iterations, gate)
        if leaf is None and index != len(schedule)-1:
            raise ValueError('full-resolution leaf=null is allowed only at the final stage')
        if leaf is not None and leaf > previous_leaf:
            raise ValueError('voxel leaf sizes must not increase')
        if gate > previous_gate:
            raise ValueError('correspondence gates must not increase')
        trim = stage.get('trim_fraction',1.)
        validate_trim_fraction(trim)
        criteria = validate_convergence(stage.get('convergence'))
        normalized.append(dict(leaf=leaf,max_distance=gate,iterations=iterations,trim_fraction=trim,convergence=criteria))
        previous_leaf = 0 if leaf is None else leaf
        previous_gate = gate
    if normalized[-1]['leaf'] is not None:
        raise ValueError('schedule must end with a full-resolution leaf=null stage')
    return normalized


def align_multiscale_clouds(source,target,schedule,*,initial_transform=None,evaluation_distance=None,
                            source_name='source',target_name='target'):
    """Run an explicit CPU schedule; each correction is composed in target frame."""
    schedule = validate_schedule(schedule)
    evaluation_distance = schedule[-1]['max_distance'] if evaluation_distance is None else evaluation_distance
    validate_settings(evaluation_distance,evaluation_distance,1,evaluation_distance)
    pose = rigid_matrix(np.eye(4) if initial_transform is None else initial_transform)
    initial = pose.copy()
    for name,cloud in [('source',source),('target',target)]:
        if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
            raise ValueError(f'{name} requires at least three finite XYZ points')
    stages, target_voxels = [], {}
    seeded = sr.apply_transform(source,initial)
    for index, settings in enumerate(schedule):
        # Always transform the original source, not the previous transformed copy.
        moved = sr.apply_transform(source,pose)
        if not np.isfinite(moved.xyz()).all():
            raise ValueError('multiscale pose overflowed source coordinates')
        leaf = settings['leaf']
        if leaf is None:
            query, reference = moved,target
        else:
            query = sr.voxel_downsample(moved,leaf,'cpu')
            if leaf not in target_voxels:
                target_voxels[leaf] = sr.voxel_downsample(target,leaf,'cpu')
            reference = target_voxels[leaf]
        if min(len(query),len(reference)) < 3:
            raise ValueError(f'stage {index} has fewer than three voxel points; reduce leaf')
        trace = sr.register_icp_diagnostics(query,reference,settings['max_distance'],settings['iterations'],
                                             trim_fraction=settings['trim_fraction'],**settings['convergence'])
        result = trace.result
        correction = result.transform()
        pose = rigid_matrix((correction.astype(np.float64) @ pose.astype(np.float64)).astype(np.float32))
        criteria = dict(translation_epsilon=1e-8,rotation_epsilon=1e-8,fitness_epsilon=1e-6)
        criteria.update(settings['convergence'])
        stages.append(dict(name=f'{index}: '+('full_resolution' if leaf is None else 'voxel'),
            leaf_metres=leaf,max_correspondence_distance_metres=settings['max_distance'],
            trim_fraction=settings['trim_fraction'],convergence_criteria=criteria,
            iterations=result.iterations,converged=result.converged,stop_reason=trace.stop_reason,
            source_points=len(query),target_points=len(reference),
            transform_source_to_target=pose.tolist(),correction_in_target_frame=correction.tolist(),
            kernel_fitness_metres_squared=result.fitness if result.fitness < np.finfo(np.float64).max else None,
            icp_history=[dict(iteration=h.iteration,correspondences=h.correspondences,
                evaluated_correspondences=h.evaluated_correspondences,
                fitness_metres_squared=h.fitness if h.evaluated_correspondences else None,
                fitness_change_metres_squared=h.fitness_change,translation_delta_metres=h.translation_delta,
                rotation_delta_radians=h.rotation_delta_radians) for h in trace.history]))
    aligned = sr.apply_transform(source,pose)
    if not np.isfinite(aligned.xyz()).all():
        raise ValueError('final multiscale pose overflowed source coordinates')
    index = sr.DistanceSupportIndex(target)
    def support(query,reference_index):
        count,fraction,rmse = reference_index.support(query,evaluation_distance)
        return dict(distance_metres=evaluation_distance,query_points=len(query),
                    distance_gated_points=count,query_fraction=fraction,gated_rmse_metres=rmse)
    report = dict(schema_version='spatialrust.python-alignment.v1',workflow='explicit_multiscale_icp',
        source_file=str(source_name),target_file=str(target_name),source_points=len(source),target_points=len(target),
        registration_source_points=stages[0]['source_points'],registration_target_points=stages[0]['target_points'],
        leaf_metres=schedule[0]['leaf'],max_distance_metres=schedule[0]['max_distance'],
        fine_distance_metres=schedule[-1]['max_distance'],evaluation_distance_metres=evaluation_distance,
        iterations=stages[-1]['iterations'],total_iterations=sum(s['iterations'] for s in stages),
        converged=stages[-1]['converged'],stages=stages,schedule=schedule,
        initial_transform_supplied=initial_transform is not None,initial_transform_source_to_target=initial.tolist(),
        transform_source_to_target=pose.tolist(),kernel_fitness_metres_squared=stages[-1]['kernel_fitness_metres_squared'],
        before_support=support(source,index),initial_support=support(seeded,index),aligned_support=support(aligned,index),
        aligned_reverse_support=support(target,sr.DistanceSupportIndex(aligned)),
        pose_correctness='not_certified_by_convergence_or_residual')
    return aligned,report


def align_multiscale_files(source_path,target_path,schedule,*,initial_transform=None,evaluation_distance=None):
    """Validate settings and prior before any file access."""
    schedule = validate_schedule(schedule)
    if initial_transform is not None:
        initial_transform = rigid_matrix(initial_transform)
    if evaluation_distance is not None:
        validate_settings(evaluation_distance,evaluation_distance,1,evaluation_distance)
    return align_multiscale_clouds(sr.read(str(source_path)),sr.read(str(target_path)),schedule,
        source_name=str(source_path),target_name=str(target_path),initial_transform=initial_transform,
        evaluation_distance=evaluation_distance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('target',type=Path)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--initial-transform',type=Path)
    parser.add_argument('--evaluation-distance',type=float)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--html-report',action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists; choose a new path')
    schedule = json.loads(args.schedule.read_text())
    prior = json.loads(args.initial_transform.read_text()) if args.initial_transform else None
    aligned,report = align_multiscale_files(args.source,args.target,schedule,
                                          initial_transform=prior,evaluation_distance=args.evaluation_distance)
    serialized = json.dumps(report,indent=2,allow_nan=False)+'\n'
    rendered = None
    if args.html_report:
        from render_alignment_report import render_report
        rendered = render_report(report)
    args.output_dir.mkdir()
    try:
        sr.write(str(args.output_dir/'aligned.pcd'),aligned)
        (args.output_dir/'alignment.json').write_text(serialized)
        if rendered is not None:
            (args.output_dir/'report.html').write_text(rendered)
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f'Saved {len(aligned)} full-resolution points; {len(schedule)} explicit stages')


if __name__ == '__main__':
    main()
