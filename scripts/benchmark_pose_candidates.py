"""Paired synthetic timings for file reads and shared target voxelization.

Run with the installed SpatialRust extension. Output includes every timing and
an offline HTML chart. This is a local workflow study, not a library ranking.
"""
import argparse
import hashlib
import html
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import spatialrust as sr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'crates/spatialrust-py/examples'))
from align_point_clouds import _align_clouds, align_files


def run(paths, poses, mode):
    """Include reads, validation, ICP and diagnostics; exclude output writes."""
    kwargs = dict(leaf=.08, max_distance=.15, evaluation_distance=.05, iterations=30)
    clouds = None if mode == 'read_each' else [sr.read(str(path)) for path in paths]
    cache = [] if mode in ('shared_target_voxel', 'shared_before_support') else None
    before_cache = [] if mode == 'shared_before_support' else None
    records = []
    for pose in poses:
        if clouds is None:
            _, report = align_files(*paths, initial_transform=pose, **kwargs)
        else:
            _, report = _align_clouds(*clouds, source_name=str(paths[0]),
                target_name=str(paths[1]), initial_transform=pose,
                target_voxel_cache=cache, before_support_cache=before_cache, **kwargs)
        records.append(report)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'target/candidate-performance')
    parser.add_argument('--sizes', type=int, nargs='+', default=[1000, 5000])
    parser.add_argument('--candidates', type=int, nargs='+', default=[1, 8])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 3 or any(n < 3 for n in args.sizes) or any(not 1 <= n <= 16 for n in args.candidates):
        parser.error('sizes must be >=3, candidates 1–16, repeats >=3')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ['read_each', 'read_once', 'shared_target_voxel', 'shared_before_support']
    rows = []
    for size in args.sizes:
        xyz = np.random.default_rng(42).uniform(-1, 1, (size, 3)).astype(np.float32)
        paths = [args.output_dir / f'{size}-{name}.pcd' for name in ('source', 'target')]
        for path, points in zip(paths, (xyz + [.012, -.008, .005], xyz)):
            sr.write(str(path), sr.PointCloud.from_xyz(np.asarray(points, dtype=np.float32)))
        for count in args.candidates:
            poses = []
            for index in range(count):
                pose = np.eye(4, dtype=np.float32)
                pose[0, 3] = .002 * index
                poses.append(pose)
            reference = run(paths, poses, modes[0])
            for mode in modes:  # Untimed warmup and complete diagnostic equality.
                assert run(paths, poses, mode) == reference
            timings = {mode: [] for mode in modes}
            for repeat in range(args.repeats):
                # Rotate execution order to reduce systematic warm-cache bias.
                for mode in modes[repeat % len(modes):] + modes[:repeat % len(modes)]:
                    start = time.perf_counter()
                    reports = run(paths, poses, mode)
                    elapsed = time.perf_counter() - start
                    assert reports == reference
                    timings[mode].append(elapsed)
            rows.append(dict(points=size, candidates=count, seconds=timings,
                             medians={k: statistics.median(v) for k, v in timings.items()},
                             diagnostics_equal=True))
    result = dict(schema_version='spatialrust.candidate-performance.v1',
        python=platform.python_version(), platform=platform.platform(),
        seed=42, repeats=args.repeats, rows=rows,
        settings=dict(leaf=.08, max_distance=.15, evaluation_distance=.05, iterations=30),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / 'crates/spatialrust-py/examples/align_point_clouds.py')},
        limits='Synthetic dense uniform clouds; warm filesystem cache; no output writes; local timing only.')
    (args.output_dir / 'timings.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    cells = []
    for row in rows:
        base = row['medians']['read_each']
        for mode in modes:
            seconds = row['medians'][mode]
            cells.append(f'<tr><td>{row["points"]}</td><td>{row["candidates"]}</td>'
                f'<td>{html.escape(mode)}</td><td>{seconds:.4f} s '
                f'({min(row["seconds"][mode]):.4f}–{max(row["seconds"][mode]):.4f})</td>'
                f'<td>{base/seconds:.2f}×</td><td><meter min="0" max="{max(row["medians"].values())}" '
                f'value="{seconds}">{seconds:.4f}</meter></td></tr>')
    (args.output_dir / 'report.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Candidate workflow timings</title><style>body{font:16px system-ui;margin:2rem}'
        'td,th{padding:.5rem;text-align:left}meter{width:220px}</style>'
        '<h1>Candidate workflow timings</h1><p>Median seconds; shorter bars are faster. '
        'All candidate diagnostics match exactly.</p><p>'+html.escape(result['limits'])+'</p>'
        '<table><tr><th>Points</th><th>Candidates</th><th>Mode</th><th>Median</th>'
        '<th>Relative to repeated reads</th><th>Elapsed time</th></tr>'+''.join(cells)+'</table>')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
