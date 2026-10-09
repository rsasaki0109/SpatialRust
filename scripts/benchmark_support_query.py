"""Local repeated support timings; use separate output paths before/after builds."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import spatialrust as sr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=15)
    parser.add_argument('--queries', type=int, default=10)
    parser.add_argument('--reuse-index', action='store_true', help='build one owned index per timed batch and reuse it')
    args = parser.parse_args()
    if args.repeats < 3 or args.queries < 1:
        parser.error('repeats >=3 and queries >=1 required')
    rows = []
    for size in (5000, 20000):
        xyz = np.random.default_rng(42).uniform(-1, 1, (size, 3)).astype(np.float32)
        target = sr.PointCloud.from_xyz(xyz)
        source = sr.PointCloud.from_xyz(xyz + np.array([.012, -.008, .005], np.float32))
        expected = sr.distance_gated_support(source, target, .05)
        samples = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            if args.reuse_index:
                index = sr.DistanceSupportIndex(target)
                results = [index.support(source, .05) for _ in range(args.queries)]
            else:
                results = [sr.distance_gated_support(source, target, .05) for _ in range(args.queries)]
            samples.append((time.perf_counter() - start) / args.queries)
            assert all(value == expected for value in results)
        rows.append(dict(points=size, result=expected, seconds_per_query=samples,
                         median_seconds=statistics.median(samples)))
    # Identify the loaded extension, not just the Python package initializer.
    native = sorted(Path(sr.__file__).parent.glob('*.so'))
    result = dict(rows=rows, repeats=args.repeats, queries=args.queries, reuse_index=args.reuse_index, seed=42,
        python=platform.python_version(), platform=platform.platform(),
        native_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in native},
        limits='Synthetic XYZ, warm execution, sequential before/after builds; includes tree construction.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps([{k:v for k,v in row.items() if k != 'seconds_per_query'} for row in rows]))


if __name__ == '__main__':
    main()
