"""Run the committed Redwood protocol, freeze poses, then evaluate separately."""
import argparse
import html
import importlib
import json
import os
from pathlib import Path
import platform

import numpy as np
import open3d as o3d
import spatialrust as sr

from compare_public_global import run
from redwood_frame_reference import checked_bytes, digest, load_plan, prepare, score


def bindings():
    root = Path(__file__).resolve().parents[1]
    native = list(Path(sr.__file__).parent.glob('*.so'))
    if len(native) != 1:
        raise ValueError('expected one Linux native extension')
    paths = native + [Path(importlib.import_module('open3d.cpu.pybind').__file__)]
    paths += [root / 'scripts' / name for name in (
        'run_redwood_reference.py', 'redwood_frame_reference.py',
        'compare_public_global.py', 'evaluate_pose_reference.py')]
    paths += [root / 'crates/spatialrust-py/examples' / name for name in (
        'align_global.py', 'align_multiscale.py', 'align_point_clouds.py')]
    return {str(p): digest(p.read_bytes()) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output directory already exists; preserve frozen results')
    if any(os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')):
        parser.error('set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1')
    plan, plan_bytes = load_plan(args.plan)
    source_bindings = bindings()
    args.output_dir.mkdir()
    prep = prepare(args.data_dir / plan['depth_archive']['name'], plan, args.output_dir / 'prepared')
    rows = []
    for pair in plan['pairs']:
        clouds, hashes = [], {}
        for role, index in zip(('source', 'target'), pair):
            path = args.output_dir / 'prepared' / f'{index:05d}.pcd'
            hashes[role] = digest(path.read_bytes())
            if hashes[role] != prep['frames'][str(index)]['pcd_sha256']:
                raise ValueError('prepared input changed')
            clouds.append(sr.read(str(path)))
        invalid = any(not 3 <= len(sr.voxel_downsample(c, .1, 'cpu')) <= 5000 for c in clouds)
        if invalid:
            pair_rows = [dict(method=m, seed=s, status='error', error='coarse cloud outside 3..5000 point bound')
                         for s in plan['seeds'] for m in ('spatialrust', 'open3d')]
        else:
            pair_rows = run(*clouds, hashes, plan['seeds'], plan['ransac_iterations'])
        for row in pair_rows:
            row['pair'] = pair
        rows.extend(pair_rows)
        # Raw checkpoints contain no reference poses or accuracy; useful if interrupted.
        (args.output_dir / f'pair-{pair[0]}-{pair[1]}.json').write_text(json.dumps(pair_rows, indent=2, allow_nan=False) + '\n')
        print(f'Finished pair {pair}; errors={sum(r["status"] == "error" for r in pair_rows)}', flush=True)
    if bindings() != source_bindings or args.plan.read_bytes() != plan_bytes:
        raise ValueError('native, helpers, or plan changed during generation')
    frozen = dict(schema='spatialrust.redwood-frozen-registrations.v1', plan_sha256=digest(plan_bytes),
                  plan=plan, bindings=source_bindings, preparation=prep,
                  versions=dict(spatialrust=sr.__version__, open3d=o3d.__version__,
                                numpy=np.__version__, python=platform.python_version()), rows=rows)
    raw_path = args.output_dir / 'registrations.json'
    raw_path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + '\n')
    frozen_bytes = raw_path.read_bytes()
    # Only now read the trajectory. No reference reaches preparation or run().
    trajectory = checked_bytes(args.data_dir / plan['trajectory']['name'], plan['trajectory']['md5'])
    result = score(json.loads(frozen_bytes), trajectory, plan)
    result.update(registrations_sha256=digest(frozen_bytes), trajectory_sha256=digest(trajectory),
                  plan_sha256=digest(plan_bytes), bindings=source_bindings)
    if raw_path.read_bytes() != frozen_bytes or bindings() != source_bindings:
        raise ValueError('frozen results or scoring helpers changed during evaluation')
    (args.output_dir / 'accuracy.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    table = []
    for row in result['rows']:
        accuracy = row.get('accuracy')
        values = (f'<td>{accuracy["translation_error"]:.6g} m</td><td>{accuracy["rotation_error_degrees"]:.6g}°</td>'
                  if accuracy else '<td colspan="2">' + html.escape(row['error']) + '</td>')
        table.append(f'<tr><td>{row["pair"]}</td><td>{row["method"]}</td><td>{row["seed"]}</td>{values}<td>{row["accurate"]}</td></tr>')
    (args.output_dir / 'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Redwood fixed frame validation</title>'
        '<h1>Redwood fixed frame validation</h1><p>' + html.escape(plan['scope']) + '</p><p>' + html.escape(plan['accuracy_definition']) +
        '</p><table><tr><th>Pair</th><th>Method</th><th>Seed</th><th>Translation error</th><th>Rotation error</th><th>Accurate</th></tr>' +
        ''.join(table) + '</table>')
    print(json.dumps(result['totals']), flush=True)


if __name__ == '__main__':
    main()
