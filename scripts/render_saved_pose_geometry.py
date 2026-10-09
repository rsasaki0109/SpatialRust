"""Plot hash-bound saved candidate geometry with common orthographic axes.

Optional Matplotlib; diagnostic projections only, no fitting or pose selection.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import spatialrust as sr

from compare_3dmatch_cases import validate_cases
from evaluate_pose_reference import rigid


def prepare(case, comparison, choices, limit=1500):
    if comparison.get('schema') != 'spatialrust.public-global-comparison.v1':
        raise ValueError('unsupported comparison')
    if type(limit) is not int or not 3 <= limit <= 5000 or not 1 <= len(choices) <= 4 or len(set(choices)) != len(choices):
        raise ValueError('require bounded distinct candidate choices')
    clouds, hashes = {}, {}
    for role in ('source', 'target'):
        path = Path(case[role])
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != comparison['input_file_sha256'][role]:
            raise ValueError('input cloud differs from saved comparison')
        xyz = sr.voxel_downsample(sr.read(str(path)), .1, 'cpu').xyz()
        if len(xyz) < 3 or not np.isfinite(xyz).all():
            raise ValueError('require finite geometry')
        if hashlib.sha256(path.read_bytes()).hexdigest() != before:
            raise ValueError('cloud changed while reading')
        indices = np.arange(len(xyz)) if len(xyz) <= limit else np.arange(limit) * (len(xyz)-1) // (limit-1)
        clouds[role] = xyz[indices].astype(np.float64)
        hashes[role] = before
    views = []
    for method, seed in choices:
        rows = [r for r in comparison['rows'] if (r['method'], r['seed']) == (method, seed)]
        if len(rows) != 1 or rows[0]['status'] != 'success':
            raise ValueError('chosen pose is missing, duplicated or failed')
        row = rows[0]
        if row['report']['input_file_sha256'] != hashes:
            raise ValueError('chosen report input binding differs')
        pose = rigid(row['report']['transform_source_to_target'])
        transformed = clouds['source'] @ pose[:3, :3].T + pose[:3, 3]
        if not np.isfinite(transformed).all():
            raise ValueError('display transform overflows')
        views.append(dict(method=method, seed=seed, aligned_xyz=transformed.tolist(),
                          pose=pose.tolist(), full_cloud_forward_support=row['common_support']))
    return dict(source_xyz=clouds['source'].tolist(), target_xyz=clouds['target'].tolist(), views=views,
                input_file_sha256=hashes, display_leaf_metres=.1, point_limit_per_cloud=limit,
                sampling='integer_quantiles_of_voxel_output_order')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--case-id', required=True)
    parser.add_argument('--choices', nargs='+', required=True, help='method:seed, 1..4 distinct saved poses')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory exists')
    manifest_bytes = args.cases.read_bytes()
    cases = validate_cases(json.loads(manifest_bytes))
    case = next((c for c in cases if c['case_id'] == args.case_id), None)
    if case is None:
        parser.error('case ID absent from manifest')
    choices = [(value.rsplit(':', 1)[0], int(value.rsplit(':', 1)[1])) for value in args.choices]
    comparison_bytes = args.comparison.read_bytes()
    native = Path(importlib.import_module('spatialrust.spatialrust').__file__)
    native_hash = hashlib.sha256(native.read_bytes()).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = prepare(case, json.loads(comparison_bytes), choices)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target = np.asarray(output['target_xyz'])
    aligned = [np.asarray(view['aligned_xyz']) for view in output['views']]
    all_xyz = np.concatenate([target]+aligned)
    lower, upper = all_xyz.min(axis=0), all_xyz.max(axis=0)
    span = max(float((upper-lower).max()), .1)
    center = (upper+lower)/2
    figure, axes = plt.subplots(len(aligned), 3, figsize=(11, 3.4*len(aligned)), squeeze=False, layout='constrained')
    for row, (xyz, view) in enumerate(zip(aligned, output['views'])):
        for column, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
            axis = axes[row, column]
            axis.scatter(target[:, a], target[:, b], s=3, alpha=.35, color='#64748b', label='Target')
            axis.scatter(xyz[:, a], xyz[:, b], s=3, alpha=.55,
                         color=['#2563eb', '#ea580c', '#16a34a', '#9333ea'][row], label='Transformed source')
            axis.set(xlim=(center[a]-.55*span, center[a]+.55*span), ylim=(center[b]-.55*span, center[b]+.55*span),
                     xlabel='XYZ'[a]+' (m)', ylabel='XYZ'[b]+' (m)',
                     title=f'{view["method"]}, seed {view["seed"]} — '+ 'XYZ'[a]+'XYZ'[b])
            axis.set_aspect('equal')
            if column == 0:
                axis.legend(fontsize=8, markerscale=2)
    figure.suptitle(args.case_id+'\nSame axes and display samples; projections do not certify pose correctness', fontsize=11)
    args.output_dir.mkdir()
    figure.savefig(args.output_dir/'geometry.png', dpi=160)
    figure.savefig(args.output_dir/'geometry.svg', metadata={'Date': None})
    plt.close(figure)
    if hashlib.sha256(native.read_bytes()).hexdigest() != native_hash or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != runner_hash:
        raise ValueError('display native or runner changed')
    output.update(schema='spatialrust.saved-pose-geometry.v1', case_id=args.case_id,
                  comparison_sha256=hashlib.sha256(comparison_bytes).hexdigest(),
                  cases_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  native_sha256=native_hash, numpy_version=np.__version__, runner_sha256=runner_hash,
                  matplotlib_version=matplotlib.__version__, registration_executed=False,
                  limits='Manually requested saved poses, optional post-fit diagnosis; voxel display is not full-cloud support.')
    (args.output_dir/'geometry.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
