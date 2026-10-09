"""Validate controlled PnP before/after receipts and summarize paired changes."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re


def compare(before,after):
    for receipt in (before,after):
        if receipt.get('schema')!='spatialrust.opencv-pnp-study.v1':raise ValueError('unsupported PnP receipt')
        for key in ('native_sha256','source_sha256','calculation_sha256'):
            if not isinstance(receipt.get(key),str) or not re.fullmatch('[0-9a-f]{64}',receipt[key]):raise ValueError('missing/invalid receipt fingerprint')
        if any(key not in receipt for key in ('seeds','max_iterations','threshold_pixels','confidence','camera_intrinsics','thread_environment','opencv_threads','versions')):
            raise ValueError('missing comparison controls')
    for key in ('seeds','max_iterations','threshold_pixels','confidence','camera_intrinsics','thread_environment','opencv_threads','versions','calculation_sha256'):
        if before.get(key)!=after.get(key):raise ValueError(f'comparison control differs: {key}')
    if before['native_sha256']==after['native_sha256']:raise ValueError('expected different native builds')
    def index(receipt):
        rows=receipt.get('rows')
        if not isinstance(rows,list) or not rows:raise ValueError('receipt requires nonempty rows')
        result={}
        for row in rows:
            key=tuple(row.get(k) for k in ('geometry','condition','seed','method'))
            if key in result or type(row.get('recovered')) is not bool or row.get('status') not in ('success','error'):raise ValueError('duplicate/invalid study row')
            if row['status']=='error' and row['recovered']:raise ValueError('failed row cannot be recovered')
            for field in ('object_sha256','image_sha256'):
                if not isinstance(row.get(field),str) or not re.fullmatch('[0-9a-f]{64}',row[field]):raise ValueError('invalid paired input hash')
            if type(row.get('correspondences')) is not int or row['correspondences']<6:raise ValueError('invalid correspondence count')
            if row['status']=='success':
                angle,translation,positive=[row.get(k) for k in ('rotation_error_degrees','translation_error_metres','positive_depth_fraction')]
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in (angle,translation,positive)) or positive>1:
                    raise ValueError('invalid recovery measurements')
                if row['recovered']!=(angle<1 and translation<.05 and positive==1):raise ValueError('recovery flag contradicts measurements')
            result[key]=row
        groups={}
        for key,row in result.items():groups.setdefault(key[:3],[]).append(row)
        expected={'spatialrust_plain','spatialrust_ransac','opencv_plain','opencv_ransac'}
        for group in groups.values():
            if {row['method'] for row in group}!=expected or len(group)!=4:raise ValueError('missing comparison method')
            if any(len({row.get(field) for row in group})!=1 for field in ('object_sha256','image_sha256','correspondences')):raise ValueError('methods received different inputs')
        return result
    first,second=index(before),index(after)
    if set(first)!=set(second):raise ValueError('paired study conditions differ')
    for key,row in first.items():
        for field in ('object_sha256','image_sha256','correspondences','known_wrong_correspondences','generating_pose_ambiguity_known'):
            if row.get(field)!=second[key].get(field):raise ValueError(f'paired input differs: {field}')
        if key[-1].startswith('opencv') and row!=second[key]:raise ValueError('OpenCV control result changed')
    summary={}
    for method in ('spatialrust_plain','spatialrust_ransac'):
        keys=[key for key in first if key[-1]==method]
        if not keys:raise ValueError('missing native comparison method')
        gains=[key for key in keys if not first[key]['recovered'] and second[key]['recovered']]
        losses=[key for key in keys if first[key]['recovered'] and not second[key]['recovered']]
        summary[method]=dict(conditions=len(keys),before_recovered=sum(first[k]['recovered'] for k in keys),
            after_recovered=sum(second[k]['recovered'] for k in keys),gains=gains,regressions=losses)
    return dict(schema='spatialrust.pnp-repair-comparison.v1',before_native=before['native_sha256'],after_native=after['native_sha256'],
        calculation_sha256=before['calculation_sha256'],input_controls_match=True,opencv_controls_exact=True,methods=summary)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path);parser.add_argument('after',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('output directory exists')
    result=compare(json.loads(args.before.read_text()),json.loads(args.after.read_text()))
    result['receipt_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in [('before',args.before),('after',args.after)]}
    rows=[]
    for method,summary in result['methods'].items():
        rows.append(f'<tr><td>{html.escape(method)}</td><td>{summary["before_recovered"]}/{summary["conditions"]}</td>'
            f'<td>{summary["after_recovered"]}/{summary["conditions"]}</td><td>{len(summary["gains"])}</td><td>{len(summary["regressions"])}</td></tr>')
    rendered=('<!doctype html><html lang="en"><meta charset="utf-8"><title>Paired PnP repair</title>'
        '<h1>Planar initialization repair</h1><p>Identical input arrays, camera, robust controls and calculation code. '
        'OpenCV control rows match exactly; native builds differ.</p><table><tr><th>Method</th><th>Before</th><th>After</th>'
        '<th>Gained recoveries</th><th>Regressions</th></tr>'+''.join(rows)+'</table><p>Synthetic local evidence, '
        'not universal superiority. No truth-based pose selection. Collinear generating poses remain ambiguous.</p></html>')
    args.output_dir.mkdir()
    (args.output_dir/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (args.output_dir/'report.html').write_text(rendered)
    print(json.dumps(result['methods']))


if __name__=='__main__':main()
