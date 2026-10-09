"""Conditional local rigid-motion information from point geometry and surface normals.

This diagnoses rank loss under fixed correspondences. It does not certify ICP
correspondences, global identifiability, physical overlap or statistical confidence.
"""
import argparse
from contextlib import ExitStack
import html
import json
import math
from pathlib import Path

import numpy as np


def analyze_geometry(xyz,normals=None,*,relative_threshold=1e-6):
    xyz=np.asarray(xyz,dtype=np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<3 or not np.isfinite(xyz).all():
        raise ValueError('geometry requires at least three finite XYZ points')
    if isinstance(relative_threshold,bool) or not isinstance(relative_threshold,(float,int)) or not math.isfinite(relative_threshold) or not 0 < relative_threshold < 1:
        raise ValueError('relative_threshold must be finite in (0, 1)')
    # Shift before averaging to retain small geometry at a large coordinate origin.
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            offset=xyz-xyz[0]
            mean=offset.mean(axis=0)
            centered=offset-mean
            centroid=xyz[0]+mean
            extent=float(np.max(np.abs(centered)))
            radius=extent*float(np.sqrt(np.mean(np.sum((centered/extent)**2,axis=1)))) if extent else 0.
            if not math.isfinite(radius):
                raise ValueError('geometry radius overflow')
            points=centered/radius if radius else np.zeros_like(centered)
    except FloatingPointError as error:
        raise ValueError('geometry coordinate range overflow') from error
    covariance=points.T@points/len(points)
    covariance_eigenvalues=np.maximum(np.linalg.eigvalsh(covariance),0)
    if normals is None:
        information=np.zeros((6,6))
        information[:3,:3]=np.trace(covariance)*np.eye(3)-covariance
        information[3:,3:]=np.eye(3)
        model='point_to_point_fixed_correspondences'
    else:
        normals=np.asarray(normals,dtype=np.float64)
        if normals.shape!=xyz.shape or not np.isfinite(normals).all():
            raise ValueError('normals must be finite Nx3 and match positions')
        lengths=np.linalg.norm(normals,axis=1)
        if not np.allclose(lengths,1,rtol=0,atol=1e-3):
            raise ValueError('surface normals must be unit vectors')
        rows=np.column_stack((np.cross(points,normals),normals))
        information=rows.T@rows/len(points)
        model='point_to_plane_fixed_correspondences'
    values,vectors=np.linalg.eigh(information)
    values=np.maximum(values,0)
    largest=float(values[-1])
    ratios=values/largest if largest else np.zeros(6)
    weak=ratios<=relative_threshold
    rank=int((~weak).sum())
    return dict(schema='spatialrust.geometry-information.v1',model=model,points=len(points),
        centroid_xyz_metres=centroid.tolist(),rms_radius_metres=radius,
        normalization_radius_metres=radius if radius else 1.,
        covariance_eigenvalues_dimensionless=covariance_eigenvalues.tolist(),
        information_matrix_dimensionless=information.tolist(),eigenvalues_dimensionless=values.tolist(),
        relative_eigenvalues=ratios.tolist(),rank=rank,relative_rank_threshold=relative_threshold,
        condition_number=float(values[-1]/values[0]) if rank==6 else None,
        weak_directions=[vectors[:,index].tolist() for index in np.flatnonzero(weak)],
        parameter_basis='rotation_about_centroid_radians_then_translation_divided_by_normalization_radius',
        interpretation='conditional_on_fixed_correspondences_not_pose_confidence')


def render_geometry_section(diagnostics,label='Geometry'):
    if not isinstance(diagnostics,dict) or diagnostics.get('schema')!='spatialrust.geometry-information.v1':
        raise ValueError('unsupported geometry diagnostics')
    if diagnostics.get('model') not in ('point_to_point_fixed_correspondences','point_to_plane_fixed_correspondences'):
        raise ValueError('unsupported geometry model')
    rank=diagnostics.get('rank')
    values=diagnostics.get('relative_eigenvalues')
    threshold=diagnostics.get('relative_rank_threshold')
    if type(rank) is not int or not 0 <= rank <= 6 or not isinstance(values,list) or len(values)!=6:
        raise ValueError('invalid geometry spectrum/rank')
    if isinstance(threshold,bool) or not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('invalid geometry rank threshold')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values) or values!=sorted(values) or sum(v>threshold for v in values)!=rank:
        raise ValueError('geometry eigenvalues disagree with rank')
    if diagnostics.get('interpretation')!='conditional_on_fixed_correspondences_not_pose_confidence':
        raise ValueError('geometry interpretation mismatch')
    bars=[]
    for index,value in enumerate(values):
        bars.append(f'<tr><td>{index+1}</td><td><meter min="0" max="1" value="{value:.12g}"></meter></td><td>{value:.6g}</td></tr>')
    return (f'<section><h2>{html.escape(label)}</h2><p>Model: {html.escape(diagnostics["model"])}; '
        f'local information rank {rank}/6 at relative threshold {threshold:.6g}.</p>'
        '<table><tr><th>Eigenmode</th><th>Relative information</th><th>Value</th></tr>'+''.join(bars)+'</table>'
        '<p>Rotation is about the cloud centroid; translation is normalized by RMS radius '
        '(a 1 m fallback for coincident points). '
        'Weak directions are available in JSON. Rank assumes fixed correspondences; it does not '
        'certify the estimated pose, capture basin, noise confidence or global identifiability. '
        'A repeated ring can have full local rank while admitting multiple global poses. '
        'Planar point-to-point pairs can have full rank, while planar surface-normal constraints '
        'leave in-plane motion unobserved. Actual retained ICP pairs may contain less information '
        'than the full cloud analyzed here.</p></section>')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--surface-normals',action='store_true',help='use normal_x/y/z columns via optional PyArrow')
    parser.add_argument('--html',type=Path)
    args=parser.parse_args()
    if args.output.exists() or (args.html and args.html.exists()) or args.html==args.output:
        parser.error('outputs must be distinct new paths')
    import spatialrust as sr
    cloud=sr.read(str(args.input))
    normals=None
    if args.surface_normals:
        import pyarrow as pa
        columns=pa.array(cloud)
        normals=np.column_stack([columns.field('normal_'+axis).to_numpy() for axis in 'xyz'])
    diagnostics=analyze_geometry(cloud.xyz(),normals)
    serialized=json.dumps(diagnostics,indent=2,allow_nan=False)+'\n'
    rendered='<!doctype html><html lang="en"><meta charset="utf-8"><title>Geometry information</title><main>'+render_geometry_section(diagnostics,str(args.input))+'</main></html>'
    reserved=[]
    try:
        with ExitStack() as stack:
            output=stack.enter_context(args.output.open('x'))
            reserved.append(args.output)
            webpage=None
            if args.html:
                webpage=stack.enter_context(args.html.open('x'))
                reserved.append(args.html)
            output.write(serialized)
            if webpage is not None:
                webpage.write(rendered)
    except Exception:
        for path in reserved:
            path.unlink()
        raise


if __name__=='__main__':
    main()
