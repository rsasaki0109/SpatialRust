"""Bounded source-model matching checks and complete frozen-trajectory comparison.

Source-model rows are analytical predictions, never exposed runtime pair arrays.
The actual public CPU trajectory is produced by the separate fixed runner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from evaluate_pose_reference import rigid
from room_reference_rgbd_protocol import ESTIMATOR,PLAN_LF_SHA256,ROOT,SELECTION
from tum_sequence_reference import file_sha256,json_bytes


def verify_source_snapshots():
    for row in SELECTION['external']['sources']['files']:
        p=Path(row['path'])
        if p.stat().st_size!=row['bytes'] or file_sha256(p)!=row['sha256']:
            raise ValueError('pinned public reference source changed')
    name=SELECTION['correspondence_audit']['native_source']
    saved=subprocess.check_output(['git','show',SELECTION['parent_merge']+':'+name],cwd=ROOT)
    current=(ROOT/name).read_bytes()
    if current.replace(b'\r\n',b'\n')!=saved.replace(b'\r\n',b'\n'):
        raise ValueError('native production source changed')
    return dict(native_lf_sha256=hashlib.sha256(saved.replace(b'\r\n',b'\n')).hexdigest(),
                reference_sources=SELECTION['external']['sources']['files'])


def modeled_pairs(config,kind):
    if kind not in ('native-source','reference-source'):
        raise ValueError('declared source-defined projection equation required')
    w,h=config['width'],config['height']
    if type(w) is not int or type(h) is not int or not 1<=w<=64 or not 1<=h<=64:
        raise ValueError('bounded source-model raster required')
    fx,fy,cx,cy=map(float,config['camera'])
    translation=np.asarray(config['translation'],dtype=np.float64)
    gate=config['depth_gate_m']
    gate=float(np.nextafter(.03125,0.)) if gate=='nextafter(0.03125,0)' else float(gate)
    if (translation.shape!=(3,) or not np.all(np.isfinite(translation))
            or not all(math.isfinite(v) for v in (fx,fy,cx,cy,gate)) or min(fx,fy,gate)<=0):
        raise ValueError('finite positive source-model geometry required')
    source=np.full((h,w),config['source_depth_m'],np.float32)
    target=np.full((h,w),config['target_depth_m'],np.float32)
    for plane,key in ((source,'source_holes'),(target,'target_holes')):
        for x,y in config[key]:
            if type(x) is not int or type(y) is not int or not 0<=x<w or not 0<=y<h:
                raise ValueError('source-model hole outside raster')
            plane[y,x]=np.nan
    if any(np.any(np.isfinite(a)&(a<=0)) or np.any(np.isinf(a)) for a in (source,target)):
        raise ValueError('source-model depths must be positive finite values or declared holes')
    k=np.array([[fx,0.,cx],[0.,fy,cy],[0.,0.,1.]])
    # The pinned reference groups K*R*K^-1 and K*t before visiting source pixels.
    krki=k@np.eye(3)@np.linalg.inv(k);kt=k@translation
    pairs=[]
    for y in range(h):
        for x in range(w):
            z=float(source[y,x])
            if not math.isfinite(z):continue
            if kind=='native-source':
                p=np.array([(x-cx)*z/fx,(y-cy)*z/fy,z])+translation
                projected=np.array([fx*p[0]/p[2]+cx,fy*p[1]/p[2]+cy]) if p[2]>0 else np.array([np.nan,np.nan])
                transformed_z=float(p[2])
            else:
                p=z*(krki@np.array([x,y,1.]))+kt
                projected=p[:2]/p[2] if p[2]>0 else np.array([np.nan,np.nan])
                transformed_z=float(p[2])
            if not np.all(np.isfinite(projected)) or not transformed_z>0:
                raise ValueError('fixed source-model domain excludes undefined projection/casts')
            u,v=[math.trunc(float(coordinate)+.5) for coordinate in projected]
            if 0<=u<w and 0<=v<h and math.isfinite(float(target[v,u])) and abs(transformed_z-float(target[v,u]))<=gate:
                pairs.append([x,y,u,v])
    return pairs


def source_model_audit():
    results=[]
    for config in SELECTION['correspondence_audit']['cases']:
        a=modeled_pairs(config,'native-source');b=modeled_pairs(config,'reference-source')
        unique=len({tuple(row[2:]) for row in a})
        if a!=b or len(a)!=config['expected_rows'] or unique!=config['expected_unique_targets']:
            raise ValueError('prospective analytical source-model expectation failed: '+config['name'])
        results.append(dict(name=config['name'],rows=len(a),unique_targets=unique,
            duplicate_target_rows=len(a)-unique,native_source_pairs=a,reference_source_pairs=b,accepted=True))
    return dict(schema='spatialrust.room-reference-source-model.v1',plan_lf_sha256=PLAN_LF_SHA256,
        sources=verify_source_snapshots(),source_modeled_only=True,runtime_correspondence_arrays_observed=False,cases=results)


def compare_rows(native,reference):
    if not native or len(native)!=len(reference):
        raise ValueError('complete equal planned timelines required')
    times=[row['timestamp_ns'] for row in native]
    if (any(type(t) is not int for t in times) or any(a>=b for a,b in zip(times,times[1:]))
            or times!=[row['timestamp_ns'] for row in reference]):
        raise ValueError('exact strictly increasing integer timestamps required')
    for rows in (native,reference):
        if any(row.get('status') not in ('success','error') for row in rows):
            raise ValueError('known complete trajectory statuses required')
        if rows[0]['status']!='success' or not np.allclose(rigid(rows[0]['world_from_sensor']),np.eye(4),rtol=0,atol=1e-12):
            raise ValueError('same original first-frame world anchors required')
    result=[]
    for a,b in zip(native,reference):
        row=dict(timestamp_ns=a['timestamp_ns'],native_status=a['status'],reference_status=b['status'],translation_difference_m=None,rotation_difference_degrees=None)
        if a['status']==b['status']=='success':
            pa,pb=rigid(a['world_from_sensor']),rigid(b['world_from_sensor'])
            r=pa[:3,:3]@pb[:3,:3].T
            skew=np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]])*.5
            angle=math.atan2(float(np.linalg.norm(skew)),float(np.clip((np.trace(r)-1.)*.5,-1.,1.)))
            row.update(translation_difference_m=float(np.linalg.norm(pa[:3,3]-pb[:3,3])),rotation_difference_degrees=math.degrees(angle))
        result.append(row)
    paired=[row for row in result if row['translation_difference_m'] is not None]
    return dict(planned_rows=len(result),paired_rows=len(paired),unpaired_rows=len(result)-len(paired),
        complete_comparison=len(paired)==len(result),gauge='unchanged common first-frame optical world; no alignment, trim or rescale',
        translation_rms_m=math.sqrt(sum(r['translation_difference_m']**2 for r in paired)/len(paired)),
        translation_max_m=max(r['translation_difference_m'] for r in paired),
        rotation_rms_degrees=math.sqrt(sum(r['rotation_difference_degrees']**2 for r in paired)/len(paired)),
        rotation_max_degrees=max(r['rotation_difference_degrees'] for r in paired),rows=result)


def bounded_json(path):
    p=Path(path)
    if not p.is_file() or not 0<p.stat().st_size<=128*1024**2:
        raise ValueError('bounded frozen trajectory JSON required')
    return json.loads(p.read_text(encoding='utf-8'))


def record(path):
    p=Path(path).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=file_sha256(p))


def comparison(reference_path,reference_sha256):
    baseline_path=ROOT/SELECTION['native_baseline_receipt']
    baseline_raw=baseline_path.read_bytes().replace(b'\r\n',b'\n')
    if hashlib.sha256(baseline_raw).hexdigest()!=SELECTION['native_baseline_receipt_lf_sha256']:
        raise ValueError('original native baseline receipt changed')
    baseline=json.loads(baseline_raw);native_file=baseline['cases']['room']['files']['estimates'];npth=Path(native_file['path'])
    if npth.stat().st_size!=native_file['bytes'] or file_sha256(npth)!=native_file['sha256'] or file_sha256(reference_path)!=reference_sha256:
        raise ValueError('frozen native/reference trajectory identity changed')
    native,reference=bounded_json(npth),bounded_json(reference_path)
    if (reference.get('method')!=ESTIMATOR or native.get('method')!='spatialrust_reference_precision_dense_rgbd_cpu'
            or any(v.get('reference_used_for_generation') is not False or v.get('length_unit')!='m' or v.get('sensor_frame')!='tum_rgb_optical' or v.get('source_sha256')!=SELECTION['source_sha256'] for v in (native,reference))
            or native.get('calibration_sha256')!=reference.get('calibration_sha256')
            or len(native['poses'])!=SELECTION['frames'] or len(reference['poses'])!=SELECTION['frames']):
        raise ValueError('fixed complete common-calibration reference-free room trajectories required')
    return dict(schema='spatialrust.room-reference-comparison.v1',plan_lf_sha256=PLAN_LF_SHA256,
        native=native_file,reference=record(reference_path),native_accuracy_accepted=baseline['cases']['room']['accepted'],
        operational_80_percent_proven=False,**compare_rows(native['poses'],reference['poses']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('source-models');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('compare');p.add_argument('--reference-estimates',type=Path,required=True);p.add_argument('--reference-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():raise ValueError('new output with an existing parent required')
    result=source_model_audit() if args.command=='source-models' else comparison(args.reference_estimates,args.reference_sha256)
    with args.output.open('xb') as stream:stream.write(json_bytes(result))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','sources','cases')}))


if __name__=='__main__':main()
