"""Instrument native call boundaries in the candidate timing workload.

Wall times include instrumentation overhead. Components are non-overlapping;
Python validation, XYZ exports and report assembly remain in the remainder.
"""
import argparse
import hashlib
import json
import platform
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import spatialrust as sr

from benchmark_pose_candidates import ROOT, run


def profile(paths, poses, mode):
    seconds, calls = {}, {}
    def wrapper(name, function):
        index = 0
        def measured(*args, **kwargs):
            nonlocal index
            label = name
            if name == 'icp':
                label += ('_coarse', '_full')[index % 2]
            elif name == 'support':
                label += ('_before', '_initial', '_final', '_reverse')[index % 4]
            index += 1
            started = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                seconds[label] = seconds.get(label, 0.) + time.perf_counter() - started
                calls[label] = calls.get(label, 0) + 1
        return measured
    with ExitStack() as stack:
        for attribute, name in [('read', 'read'), ('voxel_downsample', 'voxel'),
                                ('apply_transform', 'transform'), ('register_icp', 'icp'),
                                ('distance_gated_support', 'support')]:
            stack.enter_context(patch.object(sr, attribute, wrapper(name, getattr(sr, attribute))))
        started = time.perf_counter()
        reports = run(paths, poses, mode)
        total = time.perf_counter() - started
    seconds['remainder'] = total - sum(seconds.values())
    assert seconds['remainder'] >= 0
    count = len(poses)
    assert calls['read'] == (count * 2 if mode == 'read_each' else 2)
    assert calls['voxel'] == (count + 1 if mode == 'shared_target_voxel' else count * 2)
    assert calls['icp_coarse'] == calls['icp_full'] == count
    assert all(calls['support_' + phase] == count for phase in ('before', 'initial', 'final', 'reverse'))
    return reports, dict(total_seconds=total, component_seconds=seconds, calls=calls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'target/candidate-component-profile')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 20000])
    parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=9)
    args = parser.parse_args()
    if args.repeats < 3 or any(n < 3 for n in args.sizes) or not 1 <= args.candidates <= 16:
        parser.error('sizes >=3, candidates 1–16 and repeats >=3 required')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ['read_each', 'read_once', 'shared_target_voxel']
    rows = []
    for size in args.sizes:
        xyz = np.random.default_rng(42).uniform(-1, 1, (size, 3)).astype(np.float32)
        paths = [args.output_dir / f'{size}-{name}.pcd' for name in ('source', 'target')]
        for path, points in zip(paths, (xyz + [.012, -.008, .005], xyz)):
            sr.write(str(path), sr.PointCloud.from_xyz(np.asarray(points, dtype=np.float32)))
        poses = []
        for index in range(args.candidates):
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = .002 * index
            poses.append(pose)
        reference = run(paths, poses, modes[0])
        for mode in modes:
            assert profile(paths, poses, mode)[0] == reference
        samples = {mode: [] for mode in modes}
        for repeat in range(args.repeats):
            for mode in modes[repeat % 3:] + modes[:repeat % 3]:
                reports, sample = profile(paths, poses, mode)
                assert reports == reference
                samples[mode].append(sample)
        for mode in modes:
            median_total = statistics.median(s['total_seconds'] for s in samples[mode])
            # Use one actual median-total sample so the stacked bar adds up.
            representative = min(samples[mode], key=lambda s: abs(s['total_seconds'] - median_total))
            rows.append(dict(points=size, candidates=args.candidates, mode=mode,
                             samples=samples[mode], representative=representative))
    result = dict(schema_version='spatialrust.candidate-component-profile.v1',
        rows=rows, python=platform.python_version(), platform=platform.platform(),
        repeats=args.repeats, seed=42, diagnostics_equal=True,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / 'scripts/benchmark_pose_candidates.py',
                      ROOT / 'crates/spatialrust-py/examples/align_point_clouds.py')})
    (args.output_dir / 'profile.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    colors = ['#475569', '#0284c7', '#7c3aed', '#059669', '#ca8a04', '#dc2626',
              '#db2777', '#0891b2', '#ea580c', '#64748b']
    parts = ['<!doctype html><meta charset="utf-8"><title>Candidate component profile</title>',
        '<style>body{font:16px system-ui;margin:2rem}.bar{display:flex;width:700px;height:28px}'
        'td,th{padding:.4rem;text-align:left}</style><h1>Candidate component profile</h1>',
        '<p>One actual median-total sample per condition. Hover over each component for time. '
        'Includes instrumentation overhead; synthetic warm-cache workload, all candidates succeed. '
        'Remainder includes XYZ copies, validation and Python assembly.</p>']
    for row in rows:
        sample = row['representative']
        parts.append(f'<h2>{row["points"]:,} points · {row["mode"]}</h2><div class="bar">')
        for index, (name, seconds) in enumerate(sample['component_seconds'].items()):
            parts.append(f'<span style="background:{colors[index]};width:{100*seconds/sample["total_seconds"]:.4f}%" '
                         f'title="{name}: {seconds*1000:.2f} ms"></span>')
        parts.append('</div><table><tr><th>Component</th><th>Time</th><th>Share</th><th>Calls</th></tr>')
        for name, seconds in sample['component_seconds'].items():
            parts.append(f'<tr><td>{name}</td><td>{seconds*1000:.2f} ms</td>'
                f'<td>{100*seconds/sample["total_seconds"]:.1f}%</td><td>{sample["calls"].get(name, "—")}</td></tr>')
        parts.append(f'</table><p>Total: {sample["total_seconds"]*1000:.2f} ms</p>')
    (args.output_dir / 'report.html').write_text(''.join(parts))
    print(json.dumps([{k:v for k,v in row.items() if k != 'samples'} for row in rows], indent=2))


if __name__ == '__main__':
    main()
