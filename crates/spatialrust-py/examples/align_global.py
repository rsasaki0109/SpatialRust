"""FPFH/RANSAC seeds followed by explicit multiscale ICP, without a supplied pose."""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import spatialrust as sr

from align_multiscale import align_multiscale_clouds, validate_schedule
from align_point_clouds import rigid_matrix, validate_settings


def validate_global_settings(schedule, leaf, radius, distance, iterations, neighbors, seeds, max_points, minimum_support):
    schedule=validate_schedule(schedule)
    validate_settings(leaf,distance,iterations,distance)
    validate_settings(radius,distance,iterations,distance)
    if type(neighbors) is not int or neighbors < 3:
        raise ValueError('normal_neighbors must be an integer at least 3')
    if type(max_points) is not int or not 3 <= max_points <= 10000:
        raise ValueError('max_coarse_points must be an integer in 3..10000')
    if not isinstance(seeds,list) or not 1 <= len(seeds) <= 16 or any(type(s) is not int or not 0 <= s < 2**64 for s in seeds):
        raise ValueError('seeds must contain 1..16 unsigned 64-bit integers')
    if len(set(seeds)) != len(seeds):
        raise ValueError('seeds must be distinct')
    if isinstance(minimum_support,bool) or not isinstance(minimum_support,(int,float)) or not math.isfinite(minimum_support) or not 0 <= minimum_support <= 1:
        raise ValueError('minimum_support must be finite in [0, 1]')
    return schedule,list(seeds)


def align_global_clouds(source,target,schedule,*,global_leaf=.1,feature_radius=.5,
        global_distance=.2,ransac_iterations=10000,normal_neighbors=20,seeds=None,
        max_coarse_points=5000,minimum_support=0,evaluation_distance=None,
        source_name='source',target_name='target',include_geometry=False):
    if type(include_geometry) is not bool:
        raise ValueError('include_geometry must be boolean')
    seeds=[7,8,9] if seeds is None else seeds
    schedule,seeds=validate_global_settings(schedule,global_leaf,feature_radius,global_distance,
        ransac_iterations,normal_neighbors,seeds,max_coarse_points,minimum_support)
    evaluation_distance=schedule[-1]['max_distance'] if evaluation_distance is None else evaluation_distance
    validate_settings(evaluation_distance,evaluation_distance,1,evaluation_distance)
    for name,cloud in [('source',source),('target',target)]:
        if len(cloud) < 3 or not np.isfinite(cloud.xyz()).all():
            raise ValueError(f'{name} requires at least three finite XYZ points')
    coarse_source=sr.voxel_downsample(source,global_leaf,'cpu')
    coarse_target=sr.voxel_downsample(target,global_leaf,'cpu')
    if min(len(coarse_source),len(coarse_target)) < 3:
        raise ValueError('too few global voxel points; reduce global_leaf')
    if max(len(coarse_source),len(coarse_target)) > max_coarse_points:
        raise ValueError('global voxel clouds exceed max_coarse_points; increase global_leaf')
    records=[]
    selected=selected_key=selected_index=None
    for index,seed in enumerate(seeds):
        record=dict(index=index,seed=seed,status='error',error=None)
        try:
            coarse=sr.register_fpfh_ransac(coarse_source,coarse_target,feature_radius,
                global_distance,ransac_iterations,normal_neighbors,seed=seed)
            record['global_converged']=coarse.converged
            record['global_feature_fitness_metres_squared']=coarse.fitness if math.isfinite(coarse.fitness) else None
            if not coarse.converged or not math.isfinite(coarse.fitness):
                raise ValueError('FPFH/RANSAC did not accept a finite hypothesis')
            prior=rigid_matrix(coarse.transform())
            record['initial_transform_source_to_target']=prior.tolist()
            aligned,report=align_multiscale_clouds(source,target,schedule,initial_transform=prior,
                evaluation_distance=evaluation_distance,source_name=source_name,target_name=target_name)
            support=report['aligned_support']
            if support['query_fraction'] < minimum_support:
                raise ValueError('refined hypothesis is below minimum_support')
            record.update(status='success',transform_source_to_target=report['transform_source_to_target'],
                aligned_support=dict(support),aligned_reverse_support=dict(report['aligned_reverse_support']),
                converged=report['converged'])
            rmse=support['gated_rmse_metres']
            key=(support['distance_gated_points'],-(math.inf if rmse is None else rmse))
            if selected_key is None or key > selected_key:
                selected=(aligned,report)
                selected_key,selected_index=key,index
        except ValueError as error:
            record['error']=str(error)
        records.append(record)
    if selected is None:
        raise ValueError('no global hypothesis succeeded: '+'; '.join(f"seed {r['seed']}: {r['error']}" for r in records))
    aligned,report=selected
    report['workflow']='fpfh_ransac_multiscale_icp'
    report['initial_transform_supplied']=False
    report['initialization_provenance']='generated_fpfh_ransac_hypothesis_no_caller_pose'
    report['global_initialization']=dict(leaf_metres=global_leaf,feature_radius_metres=feature_radius,
        max_correspondence_distance_metres=global_distance,ransac_iterations=ransac_iterations,
        normal_neighbors=normal_neighbors,seeds=seeds,max_coarse_points=max_coarse_points,
        source_points=len(coarse_source),target_points=len(coarse_target),minimum_support=minimum_support,
        normal_orientation='estimated_towards_default_viewpoint',
        fitness_definition='mean_squared_residual_over_accepted_feature_matches')
    report['candidate_selection']=dict(rule='maximum_forward_supported_points_then_minimum_gated_rmse_then_input_order',
        candidate_count=len(seeds),successful_candidates=sum(r['status']=='success' for r in records),
        selected_index=selected_index,candidates=records,
        source_centroid_xyz_metres=source.xyz().mean(axis=0,dtype=np.float64).tolist())
    if include_geometry:
        from analyze_geometry import analyze_geometry
        report['geometry_diagnostics']=dict(source=analyze_geometry(source.xyz()),target=analyze_geometry(target.xyz()))
    return aligned,report


def align_global_files(source_path,target_path,schedule,**settings):
    # Bind defaults and validate all options before sensor file access.
    import inspect
    bound=inspect.signature(align_global_clouds).bind(None,None,schedule,**settings)
    bound.apply_defaults()
    options=bound.arguments
    if type(options['include_geometry']) is not bool:
        raise ValueError('include_geometry must be boolean')
    seeds=[7,8,9] if options['seeds'] is None else options['seeds']
    validate_global_settings(schedule,options['global_leaf'],options['feature_radius'],options['global_distance'],
        options['ransac_iterations'],options['normal_neighbors'],seeds,options['max_coarse_points'],options['minimum_support'])
    if options['evaluation_distance'] is not None:
        gate=options['evaluation_distance']
        validate_settings(gate,gate,1,gate)
    settings.update(source_name=str(source_path),target_name=str(target_path))
    from align_point_clouds import read_bound_cloud
    source, source_hash = read_bound_cloud(source_path)
    target, target_hash = read_bound_cloud(target_path)
    aligned, report = align_global_clouds(source,target,schedule,**settings)
    report['input_file_sha256'] = dict(source=source_hash, target=target_hash)
    return aligned, report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('target',type=Path)
    parser.add_argument('--schedule',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--global-leaf',type=float,default=.1)
    parser.add_argument('--feature-radius',type=float,default=.5)
    parser.add_argument('--global-distance',type=float,default=.2)
    parser.add_argument('--ransac-iterations',type=int,default=10000)
    parser.add_argument('--normal-neighbors',type=int,default=20)
    parser.add_argument('--seeds',type=int,nargs='+',default=[7,8,9])
    parser.add_argument('--max-coarse-points',type=int,default=5000)
    parser.add_argument('--minimum-support',type=float,default=0)
    parser.add_argument('--evaluation-distance',type=float)
    parser.add_argument('--html-report',action='store_true')
    parser.add_argument('--geometry-diagnostics',action='store_true')
    args=parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    aligned,report=align_global_files(args.source,args.target,json.loads(args.schedule.read_text(encoding='utf-8')),
        global_leaf=args.global_leaf,feature_radius=args.feature_radius,global_distance=args.global_distance,
        ransac_iterations=args.ransac_iterations,normal_neighbors=args.normal_neighbors,seeds=args.seeds,
        max_coarse_points=args.max_coarse_points,minimum_support=args.minimum_support,evaluation_distance=args.evaluation_distance,
        include_geometry=args.geometry_diagnostics)
    serialized=json.dumps(report,indent=2,allow_nan=False)+'\n'
    rendered=None
    if args.html_report:
        from render_alignment_report import render_report
        rendered=render_report(report)
    args.output_dir.mkdir()
    try:
        sr.write(str(args.output_dir/'aligned.pcd'),aligned)
        (args.output_dir/'alignment.json').write_text(serialized, encoding='utf-8')
        if rendered is not None:
            (args.output_dir/'report.html').write_text(rendered, encoding='utf-8')
    except Exception:
        shutil.rmtree(args.output_dir)
        raise
    print(f"Saved {len(aligned)} full-resolution points; seed={report['candidate_selection']['candidates'][report['candidate_selection']['selected_index']]['seed']}")


if __name__ == '__main__':
    main()
