"""Validate optimized candidate alignment on Open3D's public ICP sample pair.

Fetch the versioned DemoICPPointClouds.zip separately; no dataset bytes enter Git.
This validates SpatialRust, without executing Open3D or asserting true pose error.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'crates/spatialrust-py/examples'))
from align_point_clouds import align_files
from align_pose_candidates import evaluate_candidates
from render_alignment_report import render_report

URL = 'https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/DemoICPPointClouds.zip'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT / 'target/public-icp-data/DemoICPPointClouds.zip')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'target/public-icp-candidate-validation')
    parser.add_argument('--iterations', type=int, default=5, help='positive maximum iterations per ICP stage')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / name for name in ('cloud_bin_0.pcd', 'cloud_bin_1.pcd')]
    with zipfile.ZipFile(args.archive) as archive:
        for path in paths:
            with archive.open(path.name) as source, path.open('wb') as output:
                shutil.copyfileobj(source, output)
    # Open3D tutorial's rounded initial guess is not exactly orthogonal.
    # Project its rotation onto SO(3) to satisfy the rigid-input contract.
    rounded = np.array([[.862,.011,-.507,.5],[-.139,.967,-.215,.7],
                        [.487,.255,.835,-1.4],[0,0,0,1.]])
    u, _, vh = np.linalg.svd(rounded[:3, :3])
    prior = rounded.copy()
    prior[:3, :3] = u @ vh
    assert np.linalg.det(prior[:3, :3]) > 0
    perturbed = prior.copy()
    angle = np.deg2rad(5)
    extra = np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]])
    perturbed[:3, :3] = extra @ prior[:3, :3]
    perturbed[:3, 3] = extra @ prior[:3, 3] + [.03, -.02, .01]
    poses = [np.eye(4).tolist(), prior.tolist(), perturbed.tolist()]
    settings = dict(leaf=.05, max_distance=.1, evaluation_distance=.02, iterations=args.iterations)
    print(f'Evaluating three candidates, at most {args.iterations} iterations per stage', file=sys.stderr, flush=True)
    started = time.perf_counter()
    aligned, report = evaluate_candidates(*paths, poses, **settings)
    optimized_seconds = time.perf_counter() - started
    print(f'Candidate search completed in {optimized_seconds:.2f}s; checking independent runs', file=sys.stderr, flush=True)
    selection = report['candidate_selection']
    started = time.perf_counter()
    independent = []
    for index, pose in enumerate(poses):
        print(f'Independent candidate {index}', file=sys.stderr, flush=True)
        try:
            independent.append(align_files(*paths, initial_transform=pose, **settings)[1])
        except ValueError as error:
            independent.append(dict(error=str(error)))
    independent_seconds = time.perf_counter() - started
    ordinary = {k:v for k,v in report.items() if k != 'candidate_selection'}
    assert ordinary == independent[selection['selected_index']]
    for candidate, separate in zip(selection['candidates'], independent):
        if candidate['status'] == 'error':
            assert candidate['error'] == separate['error']
        else:
            for field in ('transform_source_to_target', 'aligned_support', 'aligned_reverse_support', 'converged'):
                assert candidate[field] == separate[field]
    sr.write(str(args.output_dir / 'aligned.pcd'), aligned)
    reloaded = sr.read(str(args.output_dir / 'aligned.pcd'))
    np.testing.assert_array_equal(reloaded.xyz(), aligned.xyz())
    assert len(aligned) == report['source_points']
    (args.output_dir / 'alignment.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (args.output_dir / 'report.html').write_text(render_report(report))
    receipt = dict(dataset_url=URL, archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        source_points=report['source_points'], target_points=report['target_points'],
        settings=settings, rounded_tutorial_prior=rounded.tolist(), initial_transforms=poses,
        selected_index=selection['selected_index'], selected_forward_support=report['aligned_support'],
        selected_reverse_support=report['aligned_reverse_support'],
        optimized_matches_independent=True, aligned_xyz_roundtrip_exact=True,
        optimized_seconds=optimized_seconds, independent_seconds=independent_seconds,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / 'crates/spatialrust-py/examples/align_point_clouds.py',
                      ROOT / 'crates/spatialrust-py/examples/align_pose_candidates.py')},
        limits='One public pair, supplied initial poses, no ground-truth transform. Sequential single-run times are illustrative, not a comparative benchmark.')
    (args.output_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
