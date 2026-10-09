"""Controlled SpatialRust/Open3D/optional PCL ICP comparison with known synthetic poses.

Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 before starting this process.
Ground truth assesses outputs and constructs controlled perturbed priors; it
never selects an estimate. Initial perturbations are relative to truth, testing local ICP,
not independent global initialization. Both consume identical seeded f32 points.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import spatialrust as sr

from study_trimmed_icp import CASES, fixture
from study_icp_convergence import pose_errors, z_rotation


def assess(source, target, pose, truth):
    aligned = source.astype(np.float64) @ pose[:3, :3].T + pose[:3, 3]
    distance = cKDTree(target.astype(np.float64)).query(aligned)[0]
    reverse = cKDTree(aligned).query(target.astype(np.float64))[0]
    mask = distance <= .05
    rotation, translation = pose_errors(pose, truth, 1)
    return dict(rotation_error_degrees=rotation, translation_error_metres=translation,
        recovered=rotation < 1 and translation < .01,
        forward_support=float(mask.mean()), reverse_support=float((reverse <= .05).mean()),
        gated_rmse_metres=float(np.sqrt(np.mean(distance[mask]**2))) if mask.any() else None)


def run(seeds=5, iterations=100, pcl=None):
    rows = []
    for case in CASES:
        for seed in range(1 if case == 'symmetric_ring' else seeds):
            source_xyz, target_xyz, truth, _ = fixture(case,seed)
            source = sr.PointCloud.from_xyz(source_xyz)
            target = sr.PointCloud.from_xyz(target_xyz)
            reference = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target_xyz.astype(np.float64)))
            for angle in (0,20,40):
                delta = np.eye(4)
                delta[:3,:3] = z_rotation(angle)
                delta[:3,3] = [.03,-.02,.01]
                prior = (delta @ truth).astype(np.float32)
                seeded = sr.apply_transform(source,prior)
                # Make initial rounding identical, rather than let Open3D apply a
                # double-precision prior to the unseeded source internally.
                query = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(seeded.xyz().astype(np.float64)))
                for gate in (.3,.6,1.2):
                    for backend in (('spatialrust','open3d','pcl') if pcl else ('spatialrust','open3d')):
                        row = dict(case=case,seed=seed,initial_error_degrees=angle,gate=gate,
                            backend=backend,source_points=len(source),target_points=len(target),
                            symmetric_generating_pose_ambiguous=case == 'symmetric_ring',
                            recovered=False)
                        start = time.perf_counter()
                        try:
                            if backend == 'spatialrust':
                                result = sr.register_icp(seeded,target,gate,iterations,
                                    translation_epsilon=0,rotation_epsilon=0,fitness_epsilon=0)
                                correction = result.transform().astype(np.float64)
                                row['executed_iterations'] = result.iterations
                            elif backend == 'open3d':
                                result = o3d.pipelines.registration.registration_icp(query,reference,gate,np.eye(4),
                                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                                    o3d.pipelines.registration.ICPConvergenceCriteria(
                                        relative_fitness=0,relative_rmse=0,max_iteration=iterations))
                                correction = result.transformation
                            else:
                                correction,updates,kernel_seconds=pcl.align(seeded.xyz(),target_xyz,gate,iterations)
                                row['executed_iterations']=updates
                            elapsed = time.perf_counter()-start
                            pose = correction @ prior.astype(np.float64)
                            if not np.isfinite(pose).all():
                                raise ValueError('nonfinite estimated transform')
                            row.update(status='success',kernel_seconds=kernel_seconds if backend == 'pcl' else elapsed,pose=pose.tolist(),
                                       **assess(source_xyz,target_xyz,pose,truth))
                        except ValueError as error:
                            row.update(status='error',error=str(error),kernel_seconds=time.perf_counter()-start)
                        rows.append(row)
    return rows


def render(rows):
    lines=[]
    for case in CASES:
        for angle in (0,20,40):
            for gate in (.3,.6,1.2):
                cells=[]
                for backend in dict.fromkeys(r['backend'] for r in rows):
                    group=[r for r in rows if (r['case'],r['initial_error_degrees'],r['gate'],r['backend']) == (case,angle,gate,backend)]
                    if group:
                        count=sum(r['recovered'] for r in group)
                        cells.append(f'<td><meter min="0" max="{len(group)}" value="{count}"></meter> {count}/{len(group)}</td>')
                if cells:
                    lines.append(f'<tr><td>{html.escape(case)}</td><td>{angle}</td><td>{gate}</td>'+''.join(cells)+'</tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Controlled ICP comparison</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto}td,th{padding:.5rem}tr:nth-child(even){background:#eee}</style>'
        '<h1>Controlled local ICP failures</h1><p>Identical seeded f32 points, gates and update caps. '
        'SpatialRust computes f32 updates; Open3D computes f64 updates. Their nearest-neighbor tie rules '
        'and precision differ. Optional PCL uses f32 SVD updates and inclusive zero-delta stopping, '
        'so actual updates are recorded and can be fewer than the shared cap. '
        'Recovery means less than 1 degree and .01 m from the synthetic generating pose. '
        'The ring has equivalent poses, so its generating-pose error is not observable physical error.</p>'
        '<table><tr><th>Environment</th><th>Prior error °</th><th>Gate m</th><th>SpatialRust</th><th>Open3D</th>'
        + ('<th>PCL</th>' if any(r['backend']=='pcl' for r in rows) else '') + '</tr>'
        +''.join(lines)+'</table><p>No global initialization, real-sensor accuracy or universal speed ranking '
        'is established. Kernel timing excludes conversions and common SciPy evaluation. JSON records every '
        'pose and full-source support, including errors. Different library fitness definitions are not compared.</p></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--pcl-binary',type=Path,help='optional persistent native PCL comparator')
    args=parser.parse_args()
    if not 1 <= args.seeds <= 20 or not 1 <= args.iterations <= 1000:
        parser.error('seeds must be 1..20 and iterations 1..1000')
    if args.output_dir.exists():
        parser.error('output directory exists')
    if any(os.environ.get(name) != '1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        parser.error('start with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1')
    pcl_version=None
    if args.pcl_binary:
        from pcl_icp_process import PclIcpProcess
        with PclIcpProcess(args.pcl_binary) as pcl:
            rows=run(args.seeds,args.iterations,pcl)
            pcl_version=pcl.version
    else:
        rows=run(args.seeds,args.iterations)
    native=list(Path(sr.__file__).parent.glob('*.so')) if Path(sr.__file__).suffix != '.so' else [Path(sr.__file__)]
    if len(native) != 1:
        raise ValueError('cannot uniquely identify native extension')
    paths=[Path(__file__),Path(__file__).with_name('study_trimmed_icp.py'),Path(__file__).with_name('study_icp_convergence.py')]
    if args.pcl_binary:
        paths.extend([Path(__file__).with_name('pcl_icp_process.py'),Path(__file__).resolve().parents[1]/'bench/pcl_comparison/pcl_icp.cpp'])
    receipt=dict(schema='spatialrust.open3d-icp-study.v1',seeds=args.seeds,iterations=args.iterations,
        versions=dict(numpy=np.__version__,open3d=o3d.__version__),evaluation_distance_metres=.05,
        thread_environment={name:os.environ[name] for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        native_sha256=hashlib.sha256(native[0].read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},rows=rows)
    if args.pcl_binary:
        receipt['versions']['pcl']=pcl_version
        receipt['pcl_binary_sha256']=hashlib.sha256(args.pcl_binary.read_bytes()).hexdigest()
    serialized=json.dumps(receipt,indent=2,allow_nan=False)
    rendered=render(rows)
    args.output_dir.mkdir()
    (args.output_dir/'study.json').write_text(serialized, encoding='utf-8')
    (args.output_dir/'report.html').write_text(rendered, encoding='utf-8')
    print(f'{len(rows)} runs; '+', '.join(f'{b}: {sum(r["recovered"] for r in rows if r["backend"] == b)} recoveries' for b in dict.fromkeys(r['backend'] for r in rows)))


if __name__ == '__main__':
    main()
