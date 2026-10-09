"""Controlled PnP failures across geometry, image noise and wrong correspondences.

Compares SpatialRust calibrated initialization/refinement and six-point RANSAC against OpenCV
ITERATIVE PnP and its RANSAC path. Initializers/minimal sample sizes differ;
this is an outcome comparison, not matched numerical work or a speed ranking.
"""
import argparse
import ast
import hashlib
import html
import json
import os
import platform
from pathlib import Path

import cv2
import numpy as np
import spatialrust as sr

GEOMETRIES=('volume','thin','plane','line')
CONDITIONS=('clean','noise','outliers')
METHODS=('spatialrust_plain','spatialrust_ransac','opencv_plain','opencv_ransac')
CAMERA=np.array([[700.,0,320.],[0,710.,240.],[0,0,1.]])


def calculation_fingerprint(source):
    tree=ast.parse(source)
    selected=[node for node in tree.body if (isinstance(node,ast.FunctionDef) and node.name in ('fixture','run'))
        or (isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id in ('GEOMETRIES','CONDITIONS','METHODS','CAMERA') for target in node.targets))]
    if len(selected)!=6:raise ValueError('calculation source layout changed')
    return hashlib.sha256(ast.dump(ast.Module(body=selected,type_ignores=[]),include_attributes=False).encode()).hexdigest()


def fixture(geometry,condition,seed):
    objects=np.random.default_rng(seed).uniform(-1,1,(100,3))
    if geometry=='thin':objects[:,2]*=1e-4
    elif geometry=='plane':objects[:,2]=0
    elif geometry=='line':objects[:,1:]=0
    elif geometry!='volume':raise ValueError('unsupported geometry')
    angle=.2
    rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
    translation=np.array([.3,-.2,5.])
    camera=objects@rotation.T+translation
    pixels=camera[:,:2]/camera[:,2,None]*[700.,710.]+[320.,240.]
    images=pixels.copy()
    if condition in ('noise','outliers'):
        images+=np.random.default_rng(seed+1000).normal(0,1,images.shape)
    if condition=='outliers':
        images[-30:]+=np.random.default_rng(seed+2000).uniform([80,-200],[250,-80],(30,2))
    elif condition not in ('clean','noise'):raise ValueError('unsupported condition')
    return objects,images,rotation,translation,pixels


def run(seeds=5,budget=300):
    cv2.setNumThreads(1)
    rows=[]
    for geometry in GEOMETRIES:
        for condition in CONDITIONS:
            for seed in range(seeds):
                objects,images,truth_rotation,truth_translation,pixels=fixture(geometry,condition,seed)
                for method in METHODS:
                    row=dict(geometry=geometry,condition=condition,seed=seed,method=method,
                        correspondences=len(objects),known_wrong_correspondences=30 if condition=='outliers' else 0,
                        generating_pose_ambiguity_known=geometry=='line',recovered=False,
                        object_sha256=hashlib.sha256(objects.tobytes()).hexdigest(),image_sha256=hashlib.sha256(images.tobytes()).hexdigest())
                    try:
                        if method=='spatialrust_plain':
                            rotation,translation=sr.solve_pnp(objects,images,700,710,320,240)
                        elif method=='spatialrust_ransac':
                            rotation,translation,mask,residuals=sr.solve_pnp_ransac(objects,images,700,710,320,240,
                                threshold=3,confidence=.99,max_iterations=budget,seed=seed)
                            row['reported_inlier_count']=int(mask.sum())
                        else:
                            cv2.setRNGSeed(seed)
                            if method=='opencv_plain':
                                ok,vector,translation=cv2.solvePnP(objects,images,CAMERA,None,flags=cv2.SOLVEPNP_ITERATIVE)
                            else:
                                ok,vector,translation,inliers=cv2.solvePnPRansac(objects,images,CAMERA,None,
                                    iterationsCount=budget,reprojectionError=3,confidence=.99,flags=cv2.SOLVEPNP_ITERATIVE)
                                row['reported_inlier_count']=len(inliers) if inliers is not None else 0
                            if not ok:raise ValueError('OpenCV did not estimate a pose')
                            rotation=cv2.Rodrigues(vector)[0];translation=translation.reshape(3)
                        if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
                            raise ValueError('nonfinite estimated pose')
                        angle=float(np.degrees(np.arccos(np.clip((np.trace(rotation@truth_rotation.T)-1)/2,-1,1))))
                        translation_error=float(np.linalg.norm(translation-truth_translation))
                        camera=objects@rotation.T+translation
                        positive=camera[:,2]>1e-12
                        residual=None
                        if positive.all():
                            projection=camera[:,:2]/camera[:,2,None]*[700.,710.]+[320.,240.]
                            residual=float(np.sqrt(np.mean(np.sum((projection-pixels)**2,axis=1))))
                        row.update(status='success',rotation=rotation.tolist(),translation=translation.tolist(),
                            rotation_error_degrees=angle,translation_error_metres=translation_error,
                            generating_pixel_rmse=residual,positive_depth_fraction=float(positive.mean()),
                            recovered=angle<1 and translation_error<.05 and bool(positive.all()))
                    except (ValueError,cv2.error) as error:
                        row.update(status='error',error=str(error))
                    rows.append(row)
    return rows


def render(rows):
    lines=[]
    for geometry in GEOMETRIES:
        for condition in CONDITIONS:
            cells=[]
            for method in METHODS:
                group=[r for r in rows if (r['geometry'],r['condition'],r['method'])==(geometry,condition,method)]
                count=sum(r['recovered'] for r in group)
                errors=sum(r['status']=='error' for r in group)
                cells.append(f'<td><meter min="0" max="{len(group)}" value="{count}"></meter> {count}/{len(group)}<br>{errors} errors</td>')
            lines.append(f'<tr><td>{html.escape(geometry)}</td><td>{html.escape(condition)}</td>'+''.join(cells)+'</tr>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Controlled PnP failures</title>'
        '<style>body{font:15px system-ui;margin:2rem}td,th{padding:.5rem}tr:nth-child(even){background:#eee}</style>'
        '<h1>PnP assumptions and failures</h1><p>Same calibrated correspondences; plain fitting versus '
        'RANSAC with 3 pixel threshold and .99 confidence. SpatialRust uses calibrated DLT or its build-specific '
        'plane-aware homography initializer and six-point '
        'samples. OpenCV ITERATIVE has a planar initialization path and a different RANSAC sampler. '
        'No caller pose or truth-based selection. Recovery means less than 1 degree and .05 m from the '
        'generating pose, with all points in front of the camera.</p>'
        '<table><tr><th>Geometry</th><th>Environment</th>'+''.join(f'<th>{html.escape(m)}</th>' for m in METHODS)+'</tr>'
        +''.join(lines)+'</table><p>Noise is 1 pixel Gaussian; outliers replace the last 30% of image '
        'correspondences with large shifts. Thin geometry has z extent scaled by 1e-4; the plane has z=0. '
        'A line cannot determine all rotation degrees of freedom, so generating-pose error there is not '
        'observable physical error. JSON retains failures, every pose, full-row noiseless generating-pixel '
        'RMSE and input/code/native hashes. No distorted-camera, real-sensor or timing superiority claim.</p></html>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=300)
    args=parser.parse_args()
    if not 1 <= args.seeds <= 10 or not 1 <= args.iterations <= 2000:parser.error('seeds/iterations out of range')
    if args.output_dir.exists():parser.error('output directory exists')
    if any(os.environ.get(name)!='1' for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')):parser.error('set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1')
    native=list(Path(sr.__file__).parent.glob('*.so'));assert len(native)==1
    native_hash=hashlib.sha256(native[0].read_bytes()).hexdigest()
    source_snapshot=Path(__file__).read_bytes()
    rows=run(args.seeds,args.iterations)
    if hashlib.sha256(native[0].read_bytes()).hexdigest()!=native_hash:raise ValueError('native module changed during study')
    if Path(__file__).read_bytes()!=source_snapshot:raise ValueError('runner source changed during study')
    receipt=dict(schema='spatialrust.opencv-pnp-study.v1',versions=dict(opencv=cv2.__version__,numpy=np.__version__,python=platform.python_version(),spatialrust=sr.__version__),
        seeds=args.seeds,max_iterations=args.iterations,threshold_pixels=3,confidence=.99,
        camera_intrinsics=CAMERA.tolist(),thread_environment={name:os.environ[name] for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},opencv_threads=cv2.getNumThreads(),
        native_sha256=native_hash,source_sha256=hashlib.sha256(source_snapshot).hexdigest(),calculation_sha256=calculation_fingerprint(source_snapshot),rows=rows)
    serialized=json.dumps(receipt,indent=2,allow_nan=False);rendered=render(rows)
    args.output_dir.mkdir()
    (args.output_dir/'study.json').write_text(serialized)
    (args.output_dir/'report.html').write_text(rendered)
    print(f'{len(rows)} runs; '+', '.join(f'{m}: {sum(r["recovered"] for r in rows if r["method"]==m)} recoveries' for m in METHODS))


if __name__=='__main__':main()
